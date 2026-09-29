using LevelZero;
using LevelZero.Kernels;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// A decoder-only transformer (Qwen2 / Llama style) resident on a Level Zero GPU, with greedy
/// generation for one sequence at a time.
/// </summary>
/// <remarks>
/// All weights are held as float32 in shared memory: for Qwen2.5-0.5B that is about 2.5 GB.
/// The prompt is processed one token at a time, and logits are computed only for the last prompt
/// token. The model borrows the device and kernel suite; the caller disposes them after the model.
/// </remarks>
public sealed class LlmModel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly LlmKernelSuite _suite;
    private readonly List<LlmLayerWeights> _layerWeights = [];
    private readonly List<LlmDecoderLayer> _layers = [];
    private readonly List<LlmKvCache?> _caches = [];
    private readonly List<LlmLinearState?> _states = [];
    private SharedBuffer<float>? _embedding;
    private SharedBuffer<float>? _finalNorm;
    private LlmMatrix? _lmHead;
    private SharedBuffer<float>? _hidden;
    private SharedBuffer<float>? _normed;
    private SharedBuffer<float>? _logits;
    private SharedBuffer<int>? _tokenId;
    private SharedBuffer<int>? _sampled;
    private SharedBuffer<float>? _hiddenBatch;
    private SharedBuffer<int>? _tokenBatch;

    private LlmModel(
        ComputeDevice device,
        LlmKernelSuite suite,
        HfModelConfig config,
        int maxSeqLen,
        SafetensorsCheckpoint checkpoint,
        LlmWeightPrecision precision)
    {
        _device = device;
        _suite = suite;
        Config = config;
        MaxSeqLen = maxSeqLen;

        try
        {
            _hidden = device.AllocShared<float>(config.HiddenSize);
            _normed = device.AllocShared<float>(config.HiddenSize);
            _logits = device.AllocShared<float>(config.VocabSize);
            _tokenId = device.AllocShared<int>(1);
            _sampled = device.AllocShared<int>(1);

            _embedding = device.AllocShared(checkpoint.ReadFloat32(config.TensorPrefix + "embed_tokens.weight"));
            float[] finalNorm = checkpoint.ReadFloat32(config.TensorPrefix + "norm.weight");
            for (int i = 0; i < finalNorm.Length; i++)
            {
                finalNorm[i] += config.RmsNormOffset;
            }
            _finalNorm = device.AllocShared(finalNorm);
            string headName = config.TieWordEmbeddings ? config.TensorPrefix + "embed_tokens.weight" : "lm_head.weight";
            _lmHead = LlmMatrix.LoadTransposed(device, checkpoint, headName, config.VocabSize, config.HiddenSize, precision);

            for (int i = 0; i < config.NumLayers; i++)
            {
                var weights = LlmLayerWeights.Load(device, checkpoint, config, i, precision);
                _layerWeights.Add(weights);
                _layers.Add(new LlmDecoderLayer(device, suite, config, weights));
                if (weights.Linear is not null)
                {
                    _caches.Add(null);
                    _states.Add(new LlmLinearState(device, config));
                }
                else
                {
                    _caches.Add(new LlmKvCache(device, maxSeqLen, config.KeyValueDim));
                    _states.Add(null);
                }
            }
        }
        catch
        {
            DisposeAll();
            throw;
        }
    }
    /// <summary>The model configuration.</summary>
    public HfModelConfig Config { get; }

    /// <summary>Device bytes held by the projection matrices (all layers plus the LM head), i.e. what one token streams.</summary>
    public long ProjectionBytes => _layerWeights.Sum(w => w.ProjectionBytes) + (_lmHead?.SizeInBytes ?? 0);
    /// <summary>True when the projection matrices are stored as half precision.</summary>
    public bool UsesHalfWeights => _lmHead?.IsHalf == true;
    /// <summary>True when the projection matrices (at least the LM head) are stored as int8.</summary>
    public bool UsesInt8Weights => _lmHead?.IsInt8 == true;
    /// <summary>Longest sequence (prompt plus generated tokens) the KV caches hold.</summary>
    public int MaxSeqLen { get; }

    /// <summary>
    /// Loads a HuggingFace checkpoint directory (<c>config.json</c> plus safetensors weights).
    /// </summary>
    /// <param name="device">The compute device.</param>
    /// <param name="suite">The LLM kernel suite for <paramref name="device"/>.</param>
    /// <param name="modelDirectory">Directory holding the checkpoint.</param>
    /// <param name="maxSeqLen">KV cache length in tokens.</param>
    /// <param name="precision">Storage precision of the projection matrices. <see cref="LlmWeightPrecision.Auto"/> (default) uses int8 when the kernel binary supports it, otherwise half for 16-bit checkpoints.</param>
    /// <exception cref="InvalidDataException">The checkpoint does not match its config.</exception>
    public static LlmModel Load(
        ComputeDevice device,
        LlmKernelSuite suite,
        string modelDirectory,
        int maxSeqLen = 2048,
        LlmWeightPrecision precision = LlmWeightPrecision.Auto)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(suite);
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(maxSeqLen, LlmDecoderLayer.MaxAttendableTokens);

        HfModelConfig config = HfModelConfig.Load(modelDirectory);
        config.EnsureSupported();
        using var checkpoint = SafetensorsCheckpoint.Open(modelDirectory);
        IReadOnlyList<string> problems = config.ValidateCheckpoint(checkpoint);
        if (problems.Count > 0)
        {
            throw new InvalidDataException("Checkpoint does not match config.json: " + string.Join("; ", problems.Take(5)));
        }

        if (precision == LlmWeightPrecision.Auto && suite.GemvQ8 is not null)
        {
            precision = LlmWeightPrecision.Int8;
        }
        else if (precision == LlmWeightPrecision.Int8 && suite.GemvQ8 is null)
        {
            throw new NotSupportedException("The loaded llm_kernels binary has no int8 kernels; rebuild it or choose another precision.");
        }
        if (precision == LlmWeightPrecision.Int8)
        {
            // The scratch must be sized before any command list is recorded: the widest projection input, one prefill chunk of rows.
            int widest = Math.Max(Math.Max(config.HiddenSize, config.IntermediateSize), Math.Max(config.QueryDim, config.HasLinearLayers ? config.LinearValueDim : 0));
            suite.EnsureQ8Scratch(device, LlmGemvKernel.MaxRows, (widest + LlmGemvQ8Kernel.GroupSize - 1) / LlmGemvQ8Kernel.GroupSize * LlmGemvQ8Kernel.GroupSize);
        }
        return new LlmModel(device, suite, config, maxSeqLen, checkpoint, precision);
    }

    /// <summary>
    /// Runs the prompt and then generates greedily (argmax) until <paramref name="maxNewTokens"/> tokens
    /// exist or a stop token is produced.
    /// </summary>
    /// <param name="promptIds">Token ids of the prompt; at least one.</param>
    /// <param name="maxNewTokens">Maximum number of tokens to generate.</param>
    /// <param name="stopTokenIds">Tokens that end generation; the stop token is included in the result.</param>
    /// <returns>The generated token ids, without the prompt.</returns>
    public int[] GenerateGreedy(IReadOnlyList<int> promptIds, int maxNewTokens, IReadOnlyCollection<int>? stopTokenIds = null)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptIds));
        }

        if (promptIds.Count + maxNewTokens > MaxSeqLen)
        {
            throw new ArgumentException(
                $"Prompt of {promptIds.Count} tokens plus {maxNewTokens} new tokens exceeds the {MaxSeqLen}-token cache.");
        }

        var generated = new List<int>(maxNewTokens);
        if (maxNewTokens == 0)
        {
            return [];
        }

        int next = IngestPrompt(promptIds, StepOutput.Argmax);

        int position = promptIds.Count;
        while (true)
        {
            generated.Add(next);
            if (generated.Count >= maxNewTokens || stopTokenIds?.Contains(next) == true)
            {
                break;
            }

            next = Step(next, position, StepOutput.Argmax);
            position++;
        }

        return [.. generated];
    }

    /// <summary>
    /// Feeds one token at <paramref name="position"/> and returns the logits for the following token.
    /// Reads the shared logits buffer, so the array is a copy.
    /// </summary>
    public float[] ComputeLogits(int tokenId, int position)
    {
        _ = Step(tokenId, position, StepOutput.Logits);
        return _logits!.ToArray();
    }

    /// <summary>Feeds the tokens starting at position 0 and returns the logits after the last one.</summary>
    public float[] ComputeLogitsForPrompt(IReadOnlyList<int> promptIds)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptIds));
        }

        _ = IngestPrompt(promptIds, StepOutput.Logits);
        return _logits!.ToArray();
    }

    /// <summary>Number of tokens currently in the KV cache (the next token goes at this position).</summary>
    public int Position { get; private set; }
    /// <summary>Forgets the current sequence so a new prompt starts at position 0.</summary>
    public void ResetSequence() => Position = 0;
    /// <summary>Resets the sequence, feeds the prompt and returns the logits for the first new token.</summary>
    /// <param name="promptIds">Token ids of the prompt; at least one and at most <see cref="MaxSeqLen"/>.</param>
    public float[] Prefill(IReadOnlyList<int> promptIds)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptIds));
        }
        if (promptIds.Count > MaxSeqLen)
        {
            throw new ArgumentException($"Prompt of {promptIds.Count} tokens exceeds the {MaxSeqLen}-token cache.", nameof(promptIds));
        }
        ResetSequence();
        _ = IngestPrompt(promptIds, StepOutput.Logits);
        Position = promptIds.Count;
        return _logits!.ToArray();
    }
    /// <summary>Feeds one more token after <see cref="Prefill"/> and returns the logits for the next one.</summary>
    /// <exception cref="InvalidOperationException">The KV cache is full.</exception>
    public float[] Decode(int tokenId)
    {
        if (Position >= MaxSeqLen)
        {
            throw new InvalidOperationException($"The {MaxSeqLen}-token KV cache is full.");
        }
        _ = Step(tokenId, Position, StepOutput.Logits);
        Position++;
        return _logits!.ToArray();
    }
    /// <summary>
    /// Copies the sequence tail to host memory: KV rows <c>[basePosition, Position)</c> plus the whole linear-attention
    /// state. Pair with <see cref="RestoreState"/> to fork a sequence (beam search).
    /// </summary>
    /// <param name="basePosition">Rows below this position are assumed identical in every state that is restored (the shared prompt).</param>
    public LlmSequenceState CaptureState(int basePosition)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(basePosition);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(basePosition, Position);
        int n = _caches.Count;
        var keys = new float[]?[n];
        var values = new float[]?[n];
        var conv = new float[]?[n];
        var recurrent = new float[]?[n];
        int rows = Position - basePosition;
        for (int i = 0; i < n; i++)
        {
            if (_caches[i] is { } cache)
            {
                keys[i] = LlmSequenceState.ReadRange(cache.Keys, basePosition * cache.KvDim, rows * cache.KvDim);
                values[i] = LlmSequenceState.ReadRange(cache.Values, basePosition * cache.KvDim, rows * cache.KvDim);
            }
            if (_states[i] is { } st)
            {
                conv[i] = st.Conv.ToArray();
                recurrent[i] = st.Recurrent.ToArray();
            }
        }
        return new LlmSequenceState(basePosition, Position, keys, values, conv, recurrent);
    }

    /// <summary>Writes a captured tail back and rewinds or advances <see cref="Position"/> to the captured length.</summary>
    /// <remarks>The rows below <see cref="LlmSequenceState.BasePosition"/> must still hold the prompt the state was captured against.</remarks>
    public void RestoreState(LlmSequenceState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.Keys.Length != _caches.Count || state.Position > MaxSeqLen)
        {
            throw new ArgumentException("The state does not match this model.", nameof(state));
        }
        for (int i = 0; i < _caches.Count; i++)
        {
            if (_caches[i] is { } cache && state.Keys[i] is { } k && state.Values[i] is { } v)
            {
                LlmSequenceState.WriteRange(cache.Keys, state.BasePosition * cache.KvDim, k);
                LlmSequenceState.WriteRange(cache.Values, state.BasePosition * cache.KvDim, v);
            }
            if (_states[i] is { } st && state.Conv[i] is { } c && state.Recurrent[i] is { } r)
            {
                LlmSequenceState.WriteRange(st.Conv, 0, c);
                LlmSequenceState.WriteRange(st.Recurrent, 0, r);
            }
        }
        Position = state.Position;
    }
    /// <inheritdoc />
    public void Dispose() => DisposeAll();
    private enum StepOutput
    {
        None,
        Logits,
        Argmax,
    }

    /// <summary>
    /// Feeds a prompt from position 0 in chunks (each weight matrix is read once per chunk instead of once per token)
    /// and returns the argmax token or leaves the logits, depending on <paramref name="lastOutput"/>.
    /// </summary>
    private int IngestPrompt(IReadOnlyList<int> promptIds, StepOutput lastOutput)
    {
        int chunk = _layers[0].BatchCapacity;
        int result = -1;
        for (int start = 0; start < promptIds.Count; start += chunk)
        {
            int count = Math.Min(chunk, promptIds.Count - start);
            StepOutput output = start + count == promptIds.Count ? lastOutput : StepOutput.None;
            result = count == 1
                ? Step(promptIds[start], start, output)
                : StepBatch(promptIds, start, count, output);
        }
        return result;
    }
    /// <summary>One forward pass for <paramref name="count"/> consecutive prompt tokens starting at position <paramref name="start"/>.</summary>
    private int StepBatch(IReadOnlyList<int> promptIds, int start, int count, StepOutput output)
    {
        SharedBuffer<float> hidden = _hidden ?? throw new ObjectDisposedException(nameof(LlmModel));
        int h = Config.HiddenSize;
        int capacity = _layers[0].BatchCapacity;
        SharedBuffer<float> hiddenBatch = _hiddenBatch ??= _device.AllocShared<float>(capacity * h);
        SharedBuffer<int> tokenBatch = _tokenBatch ??= _device.AllocShared<int>(capacity);
        var ids = new int[count];
        for (int i = 0; i < count; i++)
        {
            int id = promptIds[start + i];
            if ((uint)id >= (uint)Config.VocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(promptIds), id, $"Vocabulary has {Config.VocabSize} tokens.");
            }
            ids[i] = id;
        }
        tokenBatch.Write(ids);
        bool record = !_device.IsRecording && _layers.All(l => l.Probe is null);
        if (record)
        {
            _device.BeginRecording();
        }
        try
        {
            _suite.Embedding.Execute(_device, tokenBatch, _embedding!, hiddenBatch, h, count * h);
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].ForwardBatch(hiddenBatch, count, start, _caches[i], _states[i]);
            }
            if (output != StepOutput.None)
            {
                // Only the last token of the chunk predicts the next one.
                _suite.Copy.Execute(_device, hiddenBatch, hidden, (count - 1) * h, h);
                AppendHead(output);
            }
        }
        catch
        {
            if (record)
            {
                _device.AbortRecording();
            }
            throw;
        }
        if (record)
        {
            _device.EndRecording();
        }
        return output == StepOutput.Argmax ? _sampled!.ToArray()[0] : -1;
    }
    /// <summary>Final norm and LM head on <c>_hidden</c>; with <see cref="StepOutput.Argmax"/> also picks the greedy token.</summary>
    private void AppendHead(StepOutput output)
    {
        if (output == StepOutput.None)
        {
            return;
        }
        int h = Config.HiddenSize;
        _suite.RMSNorm.Execute(_device, _hidden!, null, _finalNorm!, _normed!, Config.RmsNormEps, h, 1, addResidual: false);
        _lmHead!.Project(_device, _suite, _normed!, _logits!);
        if (output == StepOutput.Argmax)
        {
            _suite.Sampling.Execute(_device, _logits!, _sampled!, temperature: 0f, randomVal: 0f, Config.VocabSize);
        }
    }
    /// <summary>One forward pass for one token; returns the argmax token only for <see cref="StepOutput.Argmax"/>.</summary>
    private int Step(int tokenId, int position, StepOutput output)
    {
        SharedBuffer<float> hidden = _hidden ?? throw new ObjectDisposedException(nameof(LlmModel));
        SharedBuffer<int> tokenBuffer = _tokenId!;
        SharedBuffer<int> sampled = _sampled!;

        if ((uint)tokenId >= (uint)Config.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId, $"Vocabulary has {Config.VocabSize} tokens.");
        }

        int h = Config.HiddenSize;
        tokenBuffer.Write([tokenId]);
        // Record every kernel of the token into one command list (one submit, one wait) unless a layer
        // probe needs to read intermediate buffers while the pass is still running.
        bool record = !_device.IsRecording && _layers.All(l => l.Probe is null);
        if (record)
        {
            _device.BeginRecording();
        }
        try
        {
            _suite.Embedding.Execute(_device, tokenBuffer, _embedding!, hidden, h, h);
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].Forward(hidden, position, _caches[i], _states[i]);
            }
            AppendHead(output);
        }
        catch
        {
            if (record)
            {
                _device.AbortRecording();
            }
            throw;
        }
        if (record)
        {
            _device.EndRecording();
        }
        return output == StepOutput.Argmax ? sampled.ToArray()[0] : -1;
    }
    private void DisposeAll()
    {
        foreach (LlmKvCache? c in _caches)
        {
            c?.Dispose();
        }
        foreach (LlmLinearState? st in _states)
        {
            st?.Dispose();
        }

        foreach (LlmDecoderLayer l in _layers)
        {
            l.Dispose();
        }

        foreach (LlmLayerWeights w in _layerWeights)
        {
            w.Dispose();
        }

        _caches.Clear();
        _states.Clear();
        _layers.Clear();
        _layerWeights.Clear();
        _embedding?.Dispose();
        _finalNorm?.Dispose();
        _lmHead?.Dispose();
        _hidden?.Dispose();
        _normed?.Dispose();
        _logits?.Dispose();
        _tokenId?.Dispose();
        _sampled?.Dispose();
        _hiddenBatch?.Dispose();
        _tokenBatch?.Dispose();
    }
}
