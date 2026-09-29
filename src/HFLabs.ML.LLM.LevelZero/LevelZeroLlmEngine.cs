using System.Text;
using System.Text.Json;
using LevelZero;
using LevelZero.Kernels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
namespace HFLabs.ML.LLM.LevelZero;
/// <summary>
/// <see cref="ILlmEngine"/> that runs a HuggingFace checkpoint (Qwen2 family) on a Level Zero device:
/// BPE tokenizer, <see cref="LlmModel"/> forward pass on the GPU, and CPU sampling with <see cref="LlmSampler"/>.
/// One generation runs at a time; models are loaded on first use and cached per model id.
/// </summary>
public sealed class LevelZeroLlmEngine : ILlmEngine, IDisposable
{
    /// <summary>Default KV cache length in tokens.</summary>
    public const int DefaultMaxSeqLen = 4096;
    private readonly LlmOptions _options;
    private readonly ILogger _logger;
    private readonly int _maxSeqLen;
    private readonly LlmWeightPrecision _weightPrecision;
    private readonly LlmSampler _sampler = new();
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Dictionary<string, LoadedModel> _models = new(StringComparer.OrdinalIgnoreCase);
    private ComputeDevice? _device;
    private LlmKernelSuite? _suite;
    private bool _disposed;
    /// <summary>Creates the engine; nothing is loaded until the first generation.</summary>
    /// <param name="options">Model directory and device id.</param>
    /// <param name="logger">Optional logger.</param>
    /// <param name="maxSeqLen">KV cache length; a prompt plus its generated tokens must fit.</param>
    /// <param name="weightPrecision">Weight storage; <see cref="LlmWeightPrecision.Auto"/> uses int8 (DP4A) when the kernels support it.</param>
    public LevelZeroLlmEngine(
        LlmOptions options,
        ILogger<LevelZeroLlmEngine>? logger = null,
        int maxSeqLen = DefaultMaxSeqLen,
        LlmWeightPrecision weightPrecision = LlmWeightPrecision.Auto)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        _options = options;
        _logger = (ILogger?)logger ?? NullLogger.Instance;
        _maxSeqLen = maxSeqLen;
        _weightPrecision = weightPrecision;
    }
    /// <inheritdoc />
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        string modelId,
        LlmGenerationOptions options,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentException.ThrowIfNullOrEmpty(modelId);
        ArgumentNullException.ThrowIfNull(options);
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            (string? error, LoadedModel? loaded, int[] promptIds) =
                await Task.Run(() => Prepare(prompt, modelId, options), ct).ConfigureAwait(false);
            if (error is not null)
            {
                yield return error;
                yield break;
            }
            LlmModel model = loaded!.Model;
            HfBpeTokenizer tokenizer = loaded.Tokenizer;
            int budget = Math.Min(options.MaxNewTokens, model.MaxSeqLen - promptIds.Length);
            if (budget <= 0)
            {
                yield break;
            }
            string[] stops = options.StopSequences.Where(s => !string.IsNullOrEmpty(s)).ToArray();
            var context = new List<int>(promptIds);
            HfBpeTokenizer.StreamDecoder decoder = tokenizer.CreateStreamDecoder(skipSpecialTokens: true);
            var pending = new StringBuilder();
            float[] logits = await Task.Run(() => model.Prefill(promptIds), ct).ConfigureAwait(false);
            for (int n = 0; n < budget; n++)
            {
                ct.ThrowIfCancellationRequested();
                int next = PickToken(logits, tokenizer.VocabSize, options, context);
                if (loaded.StopTokens.Contains(next))
                {
                    break;
                }
                context.Add(next);
                pending.Append(decoder.Append(next));
                (string emit, bool stopped) = DrainPending(pending, stops, final: false);
                if (emit.Length > 0)
                {
                    yield return emit;
                }
                if (stopped)
                {
                    yield break;
                }
                if (n + 1 < budget)
                {
                    logits = await Task.Run(() => model.Decode(next), ct).ConfigureAwait(false);
                }
            }
            pending.Append(decoder.Flush());
            (string rest, _) = DrainPending(pending, stops, final: true);
            if (rest.Length > 0)
            {
                yield return rest;
            }
        }
        finally
        {
            _gate.Release();
        }
    }
    /// <summary>
    /// Beam search: returns the best <see cref="BeamSearchOptions.NumReturn"/> completions of the prompt, best first.
    /// The model's end-of-sequence tokens are added to <see cref="BeamSearchOptions.StopTokenIds"/>, and the tokenizer's
    /// vocabulary limit is applied. Text is not streamed; every beam must finish first.
    /// </summary>
    /// <returns>The completions, or a single entry whose text starts with <c>Error:</c> when the model cannot run.</returns>
    public async Task<IReadOnlyList<BeamCompletion>> GenerateBeamAsync(
        string prompt,
        string modelId,
        BeamSearchOptions options,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentException.ThrowIfNullOrEmpty(modelId);
        ArgumentNullException.ThrowIfNull(options);
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return await Task.Run(
                () =>
                {
                    (string? error, LoadedModel? loaded, int[] promptIds) = Prepare(prompt, modelId, new LlmGenerationOptions());
                    if (error is not null)
                    {
                        return new BeamCompletion[] { new(error, new BeamHypothesis([], 0f, 0f, false)) };
                    }
                    int budget = Math.Min(options.MaxNewTokens, loaded!.Model.MaxSeqLen - promptIds.Length);
                    if (budget <= 0)
                    {
                        return Array.Empty<BeamCompletion>();
                    }
                    var stops = new HashSet<int>(loaded.StopTokens);
                    foreach (int s in options.StopTokenIds ?? [])
                    {
                        stops.Add(s);
                    }
                    BeamSearchOptions effective = options with
                    {
                        MaxNewTokens = budget,
                        StopTokenIds = stops,
                        ValidVocabSize = Math.Min(options.ValidVocabSize ?? int.MaxValue, loaded.Tokenizer.VocabSize),
                    };
                    IReadOnlyList<BeamHypothesis> hyps = LlmBeamSearch.Run(loaded.Model, promptIds, effective);
                    return hyps
                        .Select(h => new BeamCompletion(loaded.Tokenizer.Decode(h.Tokens, skipSpecialTokens: true), h))
                        .ToArray();
                },
                ct).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }
    /// <summary>
    /// Multi-group beam search. <paramref name="optionsFactory"/> receives the model's tokenizer (so constraints can be
    /// built from token ids) and returns the search options; the engine then adds the model's end-of-sequence tokens,
    /// caps the length to the context, and applies the tokenizer's vocabulary limit.
    /// </summary>
    /// <returns>The completions best first, or one entry whose text starts with <c>Error:</c>.</returns>
    public async Task<IReadOnlyList<GroupBeamCompletion>> GenerateGroupBeamAsync(
        string prompt,
        string modelId,
        Func<HfBpeTokenizer, GroupBeamSearchOptions> optionsFactory,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentException.ThrowIfNullOrEmpty(modelId);
        ArgumentNullException.ThrowIfNull(optionsFactory);
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return await Task.Run(
                () =>
                {
                    (string? error, LoadedModel? loaded, int[] promptIds) = Prepare(prompt, modelId, new LlmGenerationOptions());
                    if (error is not null)
                    {
                        return new GroupBeamCompletion[]
                        {
                            new(error, new GroupBeamHypothesis("", new BeamHypothesis([], 0f, 0f, false), new Dictionary<string, float>(), 0f)),
                        };
                    }
                    GroupBeamSearchOptions requested = optionsFactory(loaded!.Tokenizer);
                    int budget = Math.Min(requested.MaxNewTokens, loaded.Model.MaxSeqLen - promptIds.Length);
                    if (budget <= 0)
                    {
                        return Array.Empty<GroupBeamCompletion>();
                    }
                    var stops = new HashSet<int>(loaded.StopTokens);
                    foreach (int s in requested.StopTokenIds ?? [])
                    {
                        stops.Add(s);
                    }
                    GroupBeamSearchOptions effective = requested with
                    {
                        MaxNewTokens = budget,
                        StopTokenIds = stops,
                        ValidVocabSize = Math.Min(requested.ValidVocabSize ?? int.MaxValue, loaded.Tokenizer.VocabSize),
                    };
                    return LlmGroupBeamSearch.Run(loaded.Model, promptIds, effective)
                        .Select(h => new GroupBeamCompletion(loaded.Tokenizer.Decode(h.Hypothesis.Tokens, skipSpecialTokens: true), h))
                        .ToArray();
                },
                ct).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }
    /// <inheritdoc />
    public void Dispose()
    {
        _gate.Wait();
        try
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            foreach (LoadedModel m in _models.Values)
            {
                m.Model.Dispose();
            }
            _models.Clear();
            _suite?.Dispose();
            _device?.Dispose();
        }
        finally
        {
            _gate.Release();
        }
    }
    /// <summary>
    /// Emits the text of <paramref name="pending"/> that can no longer be part of a stop sequence.
    /// When a stop sequence is found the text before it is emitted and <c>stopped</c> is true.
    /// </summary>
    internal static (string Emit, bool Stopped) DrainPending(StringBuilder pending, string[] stops, bool final)
    {
        string text = pending.ToString();
        int cut = -1;
        foreach (string s in stops)
        {
            int i = text.IndexOf(s, StringComparison.Ordinal);
            if (i >= 0 && (cut < 0 || i < cut))
            {
                cut = i;
            }
        }
        if (cut >= 0)
        {
            pending.Clear();
            return (text[..cut], true);
        }
        int keep = 0;
        if (!final)
        {
            foreach (string s in stops)
            {
                int max = Math.Min(s.Length - 1, text.Length);
                for (int len = max; len > keep; len--)
                {
                    if (text.EndsWith(s[..len], StringComparison.Ordinal))
                    {
                        keep = len;
                        break;
                    }
                }
            }
        }
        pending.Clear();
        pending.Append(text, text.Length - keep, keep);
        return (text[..^keep], false);
    }
    private int PickToken(float[] logits, int tokenizerVocab, LlmGenerationOptions options, List<int> context)
    {
        // The checkpoint's vocab is padded past the tokenizer's; padding ids are never valid outputs.
        int v = Math.Min(tokenizerVocab, logits.Length);
        float[] usable = logits.Length == v ? logits : logits[..v];
        if (options.Temperature <= 0f)
        {
            int best = 0;
            for (int i = 1; i < usable.Length; i++)
            {
                if (usable[i] > usable[best])
                {
                    best = i;
                }
            }
            return best;
        }
        return _sampler.Sample(usable, options.Temperature, options.TopP, options.TopK, options.RepetitionPenalty, context);
    }
    private (string? Error, LoadedModel? Loaded, int[] PromptIds) Prepare(string prompt, string modelId, LlmGenerationOptions options)
    {
        string dir = Path.Combine(_options.ModelDirectory ?? string.Empty, modelId);
        if (!Directory.Exists(dir))
        {
            _logger.LogError("Model directory not found: {Path}", dir);
            return ($"Error: Model '{modelId}' not found.", null, []);
        }
        if (!_models.TryGetValue(modelId, out LoadedModel? loaded))
        {
            try
            {
                loaded = LoadModel(dir);
            }
            catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException or NotSupportedException or InvalidDataException or JsonException)
            {
                _logger.LogError(ex, "Could not load model {ModelId} from {Path}", modelId, dir);
                return ($"Error: Model '{modelId}' cannot be loaded: {ex.Message}", null, []);
            }
            if (loaded is null)
            {
                return ("Error: No Level Zero device is available.", null, []);
            }
            _models[modelId] = loaded;
        }
        int[] ids = loaded.Tokenizer.Encode(prompt);
        if (ids.Length == 0)
        {
            return ("Error: The prompt is empty.", null, []);
        }
        if (ids.Length >= loaded.Model.MaxSeqLen)
        {
            return ($"Error: Prompt of {ids.Length} tokens does not fit the {loaded.Model.MaxSeqLen}-token context.", null, []);
        }
        return (null, loaded, ids);
    }
    private LoadedModel? LoadModel(string dir)
    {
        string tokenizerPath = File.Exists(_options.TokenizerJsonPath) ? _options.TokenizerJsonPath : Path.Combine(dir, "tokenizer.json");
        HfBpeTokenizer tokenizer = HfBpeTokenizer.Load(tokenizerPath);
        if (_device is null)
        {
            try
            {
                if (!LevelZeroRuntime.IsAvailable())
                {
                    return null;
                }
                _device = LevelZeroRuntime.GetDevice(0, (uint)Math.Max(0, _options.DeviceId));
                _suite = LlmKernelSuite.Create(_device);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Level Zero device initialisation failed");
                _suite?.Dispose();
                _device?.Dispose();
                _suite = null;
                _device = null;
                return null;
            }
        }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        LlmModel model = LlmModel.Load(_device, _suite!, dir, _maxSeqLen, _weightPrecision);
        _logger.LogInformation("Loaded {Dir} on Level Zero in {Ms} ms", dir, sw.ElapsedMilliseconds);
        var stops = new HashSet<int>(model.Config.EosTokenIds);
        foreach (int id in ReadGenerationConfigEos(dir))
        {
            stops.Add(id);
        }
        return new LoadedModel(tokenizer, model, stops);
    }
    private static IEnumerable<int> ReadGenerationConfigEos(string dir)
    {
        string path = Path.Combine(dir, "generation_config.json");
        if (!File.Exists(path))
        {
            return [];
        }
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        if (!doc.RootElement.TryGetProperty("eos_token_id", out JsonElement eos))
        {
            return [];
        }
        return eos.ValueKind switch
        {
            JsonValueKind.Number when eos.TryGetInt32(out int one) => [one],
            JsonValueKind.Array => eos.EnumerateArray().Where(e => e.ValueKind == JsonValueKind.Number).Select(e => e.GetInt32()).ToArray(),
            _ => [],
        };
    }
    private sealed record LoadedModel(HfBpeTokenizer Tokenizer, LlmModel Model, HashSet<int> StopTokens);
}