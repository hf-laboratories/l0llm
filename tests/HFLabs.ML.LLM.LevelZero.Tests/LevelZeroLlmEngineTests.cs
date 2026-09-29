using System.Text;
using System.Text.Json;
using LevelZero;
using Xunit;
namespace HFLabs.ML.LLM.LevelZero.Tests;
/// <summary>Loads the engine once (model load takes seconds) for all end-to-end tests.</summary>
public sealed class Qwen25EngineFixture : IDisposable
{
    public static readonly string ModelDir = Qwen25ModelFixture.ModelDir;
    public Qwen25EngineFixture()
    {
        bool hardware;
        try
        {
            hardware = LevelZeroRuntime.IsAvailable();
        }
        catch (Exception)
        {
            hardware = false;
        }
        bool model = File.Exists(Path.Combine(ModelDir, "model.safetensors"));
        if (!hardware && Environment.GetEnvironmentVariable("IPU_L0_REQUIRE_HARDWARE") == "1")
        {
            throw new InvalidOperationException("IPU_L0_REQUIRE_HARDWARE=1 but no Level Zero device was detected.");
        }
        if (!model && Environment.GetEnvironmentVariable("HF_REQUIRE_MODEL") == "1")
        {
            throw new InvalidOperationException($"HF_REQUIRE_MODEL=1 but no checkpoint at '{ModelDir}'.");
        }
        if (!hardware || !model)
        {
            return;
        }
        string parent = Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(ModelDir))!;
        ModelId = Path.GetFileName(Path.TrimEndingDirectorySeparator(ModelDir));
        Options = new LlmOptions { ModelDirectory = parent, Backend = LlmOptions.LevelZeroNativeBackend };
        Engine = new LevelZeroLlmEngine(Options, maxSeqLen: 256, weightPrecision: LlmWeightPrecision.Float16);
    }
    public LlmOptions? Options { get; }
    public string ModelId { get; } = string.Empty;
    public LevelZeroLlmEngine? Engine { get; }
    public void Dispose() => Engine?.Dispose();
}
/// <summary>
/// M5 gate: <see cref="LevelZeroLlmEngine"/> generates, on the GPU,
/// the same text HuggingFace <c>transformers</c> produces (greedy, float32 reference goldens).
/// </summary>
public sealed class LevelZeroLlmEngineTests : IClassFixture<Qwen25EngineFixture>
{
    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-0.5b-instruct.generation.golden.json");
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    });
    private readonly Qwen25EngineFixture _fx;
    public LevelZeroLlmEngineTests(Qwen25EngineFixture fx) => _fx = fx;
    private static LlmGenerationOptions Greedy(int max, params string[] stops) =>
        new() { Temperature = 0f, MaxNewTokens = max, StopSequences = stops };
    private static async Task<string> Collect(IAsyncEnumerable<string> stream, CancellationToken ct = default)
    {
        var sb = new StringBuilder();
        await foreach (string piece in stream.WithCancellation(ct))
        {
            sb.Append(piece);
        }
        return sb.ToString();
    }
    private (string Prompt, string Expected) Case(string name)
    {
        JsonElement c = Golden.Value.GetProperty("cases").GetProperty(name);
        string text = c.GetProperty("generated_text").GetString()!;
        int end = text.IndexOf("<|im_end|>", StringComparison.Ordinal);
        return (c.GetProperty("prompt_text").GetString()!, end >= 0 ? text[..end] : text);
    }
    [Fact]
    public async Task Engine_LevelZeroNative_MatchesHuggingFaceRawPrompt()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        int n = Golden.Value.GetProperty("new_tokens").GetInt32();
        (string prompt, string expected) = Case("raw");
        string actual = await Collect(_fx.Engine.GenerateAsync(prompt, _fx.ModelId, Greedy(n)));
        Assert.Equal(expected, actual);
    }
    [Fact]
    public async Task Engine_LevelZeroNative_MatchesHuggingFaceChatPromptAndStopsAtEnd()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        int n = Golden.Value.GetProperty("new_tokens").GetInt32();
        (string prompt, string expected) = Case("chat");
        string actual = await Collect(_fx.Engine.GenerateAsync(prompt, _fx.ModelId, Greedy(n)));
        Assert.Equal(expected, actual);
        Assert.DoesNotContain("<|im_end|>", actual, StringComparison.Ordinal);
    }
    [Fact]
    public async Task Engine_StopSequence_CutsTextBeforeIt()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        (string prompt, string expected) = Case("raw");
        int at = expected.IndexOf("largest", StringComparison.Ordinal);
        Assert.True(at > 0, "test premise: golden text contains 'largest'");
        string actual = await Collect(_fx.Engine.GenerateAsync(prompt, _fx.ModelId, Greedy(32, "largest")));
        Assert.Equal(expected[..at], actual);
    }
    [Fact]
    public async Task Engine_MaxNewTokens_LimitsOutput()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        (string prompt, string expected) = Case("raw");
        string actual = await Collect(_fx.Engine.GenerateAsync(prompt, _fx.ModelId, Greedy(3)));
        Assert.StartsWith(actual, expected, StringComparison.Ordinal);
        Assert.InRange(actual.Length, 1, expected.Length - 1);
    }
    [Fact]
    public async Task Engine_Sampling_ProducesTextWithoutErrors()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        var options = new LlmGenerationOptions { Temperature = 0.7f, TopK = 20, TopP = 0.9f, MaxNewTokens = 12 };
        string actual = await Collect(_fx.Engine.GenerateAsync("The capital of France is", _fx.ModelId, options));
        Assert.NotEmpty(actual);
        Assert.False(actual.StartsWith("Error:", StringComparison.Ordinal), actual);
    }
    [Fact]
    public async Task Engine_Cancellation_StopsGeneration()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        using var cts = new CancellationTokenSource();
        int pieces = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (string _ in _fx.Engine.GenerateAsync("Count from one to a hundred:", _fx.ModelId, Greedy(64), cts.Token))
            {
                if (++pieces == 2)
                {
                    cts.Cancel();
                }
            }
        });
        Assert.InRange(pieces, 2, 63);
        // The engine must stay usable after a cancelled run.
        Assert.NotEmpty(await Collect(_fx.Engine.GenerateAsync("Hello", _fx.ModelId, Greedy(2))));
    }
    [Fact]
    public async Task Engine_PromptLongerThanContext_ReturnsError()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        string longPrompt = string.Concat(Enumerable.Repeat("word ", 400));
        string actual = await Collect(_fx.Engine.GenerateAsync(longPrompt, _fx.ModelId, Greedy(4)));
        Assert.StartsWith("Error:", actual, StringComparison.Ordinal);
        Assert.Contains("does not fit", actual, StringComparison.Ordinal);
    }
    [Fact]
    public async Task Engine_UnknownModel_ReturnsError()
    {
        if (_fx.Engine is null)
        {
            return;
        }
        string actual = await Collect(_fx.Engine.GenerateAsync("Hi", "no-such-model", Greedy(2)));
        Assert.StartsWith("Error:", actual, StringComparison.Ordinal);
        Assert.Contains("no-such-model", actual, StringComparison.Ordinal);
    }
    [Theory]
    [InlineData("hello world", new[] { "STOP" }, false, "hello world", "", false)]
    [InlineData("hello ST", new[] { "STOP" }, false, "hello ", "ST", false)]
    [InlineData("hello ST", new[] { "STOP" }, true, "hello ST", "", false)]
    [InlineData("hello STOP tail", new[] { "STOP" }, false, "hello ", "", true)]
    [InlineData("a\nUser", new[] { "\nUser:", "###" }, false, "a", "\nUser", false)]
    [InlineData("a #", new[] { "\nUser:", "###" }, false, "a ", "#", false)]
    [InlineData("plain", new string[0], false, "plain", "", false)]
    public void DrainPending_HoldsBackPossibleStopPrefixes(string text, string[] stops, bool final, string emit, string kept, bool stoppedExpected)
    {
        var pending = new StringBuilder(text);
        (string got, bool stopped) = LevelZeroLlmEngine.DrainPending(pending, stops, final);
        Assert.Equal(emit, got);
        Assert.Equal(stoppedExpected, stopped);
        Assert.Equal(kept, pending.ToString());
    }
    private sealed class StubEngine(string reply) : ILlmEngine
    {
        public string? LastPrompt { get; private set; }
        public string? LastModelId { get; private set; }
        public async IAsyncEnumerable<string> GenerateAsync(
            string prompt,
            string modelId,
            LlmGenerationOptions options,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            LastPrompt = prompt;
            LastModelId = modelId;
            await Task.Yield();
            yield return reply;
        }
    }
}