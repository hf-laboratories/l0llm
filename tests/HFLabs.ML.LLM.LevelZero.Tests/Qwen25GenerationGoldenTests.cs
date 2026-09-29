using System.Diagnostics;
using System.Text.Json;
using LevelZero;
using LevelZero.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Loads the Instruct checkpoint onto the GPU once for all generation tests.</summary>
public sealed class Qwen25ModelFixture : IDisposable
{
    public static readonly string ModelDir =
        Environment.GetEnvironmentVariable("HF_QWEN25_05B_INSTRUCT_DIR") ?? @"D:\models\Qwen2.5-0.5B-Instruct";

    private readonly ComputeDevice? _device;
    private readonly LlmKernelSuite? _suite;

    public Qwen25ModelFixture()
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

        _device = LevelZeroRuntime.GetDefaultDevice();
        try
        {
            _suite = LlmKernelSuite.Create(_device);
            var watch = Stopwatch.StartNew();
            Model = LlmModel.Load(_device, _suite, ModelDir, maxSeqLen: 256, precision: LlmWeightPrecision.Float16);
            LoadSeconds = watch.Elapsed.TotalSeconds;
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The loaded model, or null when there is no GPU or checkpoint.</summary>
    public LlmModel? Model { get; }

    public double LoadSeconds { get; }

    public void Dispose()
    {
        Model?.Dispose();
        _suite?.Dispose();
        _device?.Dispose();
    }
}

/// <summary>
/// M4 gate: greedy generation on the GPU must produce exactly the token ids that HuggingFace
/// <c>transformers</c> (float32) produces for the same prompt. References come from
/// <c>golden/gen_generation_golden.py</c>.
/// </summary>
public sealed class Qwen25GenerationGoldenTests : IClassFixture<Qwen25ModelFixture>
{
    private const int ImEnd = 151645;

    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-0.5b-instruct.generation.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });

    private readonly Qwen25ModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public Qwen25GenerationGoldenTests(Qwen25ModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Theory]
    [InlineData("chat")]
    [InlineData("raw")]
    public void GreedyGeneration_MatchesTransformersToken_ForToken(string caseName)
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        (int[] prompt, int[] expected, double[] margins) = Case(caseName);
        var watch = Stopwatch.StartNew();
        int[] actual = model.GenerateGreedy(prompt, expected.Length);
        double seconds = watch.Elapsed.TotalSeconds;
        _output.WriteLine(
            $"{caseName}: {prompt.Length} prompt + {actual.Length} new tokens in {seconds:F2}s "
            + $"({(prompt.Length + actual.Length) / seconds:F1} tokens/s); model load {_fixture.LoadSeconds:F1}s");

        int firstDiff = FirstDifference(expected, actual);
        Assert.True(
            firstDiff < 0,
            firstDiff < 0
                ? string.Empty
                : $"{caseName}: first difference at generated token {firstDiff}: expected {expected[firstDiff]}, "
                    + $"got {actual[firstDiff]} (reference top-1 margin there: {margins[firstDiff]:F4}). "
                    + $"expected [{string.Join(",", expected)}] actual [{string.Join(",", actual)}]");
    }

    [Fact]
    public void StopToken_EndsGenerationAndIsIncluded()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        (int[] prompt, int[] expected, _) = Case("chat");
        int stopAt = Array.IndexOf(expected, ImEnd);
        Assert.True(stopAt >= 0, "the chat reference is expected to contain <|im_end|>");

        int[] actual = model.GenerateGreedy(prompt, 32, [ImEnd]);

        Assert.Equal(expected.Take(stopAt + 1).ToArray(), actual);
    }

    [Fact]
    public void FirstStepLogits_AgreeWithReferenceArgmaxAndMargin()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        (int[] prompt, int[] expected, double[] margins) = Case("raw");

        float[] logits = model.ComputeLogitsForPrompt(prompt);

        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        float second = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (i != best && logits[i] > second)
            {
                second = logits[i];
            }
        }

        Assert.Equal(model.Config.VocabSize, logits.Length);
        Assert.Equal(expected[0], best);
        Assert.InRange(logits[best] - second, margins[0] - 1e-3, margins[0] + 1e-3);
    }

    [Fact]
    public void Generation_IsDeterministicAcrossRuns()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        (int[] prompt, _, _) = Case("raw");

        Assert.Equal(model.GenerateGreedy(prompt, 8), model.GenerateGreedy(prompt, 8));
    }

    [Fact]
    public void GenerateGreedy_ValidatesArguments()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        Assert.Throws<ArgumentException>(() => model.GenerateGreedy([], 4));
        Assert.Throws<ArgumentException>(() => model.GenerateGreedy([1, 2, 3], model.MaxSeqLen));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.GenerateGreedy([model.Config.VocabSize], 1));
        Assert.Empty(model.GenerateGreedy([1, 2, 3], 0));
    }

    private static (int[] Prompt, int[] Expected, double[] Margins) Case(string name)
    {
        JsonElement c = Golden.Value.GetProperty("cases").GetProperty(name);
        return (
            c.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            c.GetProperty("generated_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            c.GetProperty("margins").EnumerateArray().Select(e => e.GetDouble()).ToArray());
    }

    private static int FirstDifference(int[] expected, int[] actual)
    {
        int n = Math.Min(expected.Length, actual.Length);
        for (int i = 0; i < n; i++)
        {
            if (expected[i] != actual[i])
            {
                return i;
            }
        }

        return expected.Length == actual.Length ? -1 : n;
    }
}
