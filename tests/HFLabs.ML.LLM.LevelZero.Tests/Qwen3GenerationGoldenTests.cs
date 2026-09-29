using System.Text.Json;
using LevelZero;
using LevelZero.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Loads the Qwen3-0.6B checkpoint onto the GPU once for the QK-norm tests.</summary>
public sealed class Qwen3ModelFixture : IDisposable
{
    public static readonly string ModelDir =
        Environment.GetEnvironmentVariable("HF_QWEN3_06B_DIR") ?? @"D:\models\Qwen3-0.6B";

    private readonly ComputeDevice? _device;
    private readonly LlmKernelSuite? _suite;

    public Qwen3ModelFixture()
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
            Model = LlmModel.Load(_device, _suite, ModelDir, maxSeqLen: 256, precision: LlmWeightPrecision.Float16);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The loaded model, or null when there is no GPU or checkpoint.</summary>
    public LlmModel? Model { get; }

    public void Dispose()
    {
        Model?.Dispose();
        _suite?.Dispose();
        _device?.Dispose();
    }
}

/// <summary>
/// M7 gate for QK-norm: greedy generation of a Qwen3 checkpoint (per-head RMSNorm on Q and K) must
/// produce exactly the token ids HuggingFace <c>transformers</c> (float32) produces. References come from
/// <c>golden/gen_generation_golden.py</c> run against Qwen3-0.6B.
/// </summary>
public sealed class Qwen3GenerationGoldenTests : IClassFixture<Qwen3ModelFixture>
{
    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen3-0.6b.generation.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });

    private readonly Qwen3ModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public Qwen3GenerationGoldenTests(Qwen3ModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Fact]
    public void Config_IsQwen3WithQkNorm()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        Assert.True(model.Config.QkNorm);
        Assert.False(model.Config.AttentionBias);
    }

    [Theory]
    [InlineData("chat")]
    [InlineData("raw")]
    public void GreedyGeneration_MatchesTransformers(string caseName)
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        (int[] prompt, int[] expected, double[] margins) = Case(caseName);
        int[] actual = model.GenerateGreedy(prompt, expected.Length);
        int diff = FirstDifference(expected, actual);
        if (diff >= 0 && diff < expected.Length && diff < actual.Length)
        {
            _output.WriteLine($"first difference at step {diff}: expected {expected[diff]}, got {actual[diff]}, margin {margins[diff]:F4}");
        }

        Assert.Equal(-1, diff);
    }

    [Fact]
    public void BatchedPrefill_MatchesTokenByTokenDecode()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        (int[] prompt, _, _) = Case("chat");
        float[] batched = model.Prefill(prompt);
        float[] sequential = model.ComputeLogitsForPrompt(prompt);
        Assert.Equal(sequential.Length, batched.Length);
        double maxDiff = 0;
        for (int i = 0; i < batched.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(batched[i] - sequential[i]));
        }

        _output.WriteLine($"max abs logit diff {maxDiff:E2}");
        Assert.True(maxDiff < 2e-3, $"max abs diff {maxDiff}");
        Assert.Equal(Array.IndexOf(sequential, sequential.Max()), Array.IndexOf(batched, batched.Max()));
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
