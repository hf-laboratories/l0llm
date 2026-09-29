using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using LevelZero;
using LevelZero.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Loads the Qwen3.5-0.8B checkpoint onto the GPU once for the Gated DeltaNet tests.</summary>
public sealed class Qwen35ModelFixture : IDisposable
{
    public static readonly string ModelDir =
        Environment.GetEnvironmentVariable("HF_QWEN35_08B_DIR") ?? @"D:\models\Qwen3.5-0.8B";

    private readonly ComputeDevice? _device;
    private readonly LlmKernelSuite? _suite;

    public Qwen35ModelFixture()
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

    /// <summary>Seconds the load took.</summary>
    public double LoadSeconds { get; }

    public void Dispose()
    {
        Model?.Dispose();
        _suite?.Dispose();
        _device?.Dispose();
    }
}

/// <summary>
/// M8 gate: greedy generation of a Qwen3.5 checkpoint (Gated DeltaNet layers, gated full attention, partial rotary) must
/// produce exactly the token ids HuggingFace <c>transformers</c> (float32) produces. References come from
/// <c>golden/gen_generation_golden.py</c> run against Qwen3.5-0.8B.
/// </summary>
public sealed class Qwen35GenerationGoldenTests : IClassFixture<Qwen35ModelFixture>
{
    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen3.5-0.8b.generation.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });

    private readonly Qwen35ModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public Qwen35GenerationGoldenTests(Qwen35ModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Fact]
    public void Config_IsHybridQwen35()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }

        Assert.True(model.Config.QkNorm);
        Assert.True(model.Config.AttentionOutputGate);
        Assert.True(model.Config.HasLinearLayers);
        Assert.Equal(64, model.Config.RotaryDim);
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
/// <summary>Decode and prefill throughput of Qwen3.5-0.8B. Opt in with <c>IPU_L0_BENCH=1</c>; appends to <c>%TEMP%\l0build\bench.txt</c>.</summary>
public sealed class Qwen35BenchmarkTests : IClassFixture<Qwen35ModelFixture>
{
    private readonly Qwen35ModelFixture _fx;
    private readonly ITestOutputHelper _output;
    public Qwen35BenchmarkTests(Qwen35ModelFixture fx, ITestOutputHelper output)
    {
        _fx = fx;
        _output = output;
    }
    [Fact]
    public void Decode_Throughput()
    {
        if (Environment.GetEnvironmentVariable("IPU_L0_BENCH") != "1" || _fx.Model is null)
        {
            return;
        }
        string which = (Environment.GetEnvironmentVariable("IPU_L0_BENCH_PRECISION") ?? "both").ToLowerInvariant();
        if (which is "fp16" or "both")
        {
            Measure("fp16", _fx.Model, _fx.LoadSeconds);
        }

        if (which is "int8" or "both")
        {
            using global::LevelZero.ComputeDevice device = global::LevelZero.LevelZeroRuntime.GetDefaultDevice();
            using global::LevelZero.Kernels.LlmKernelSuite suite = global::LevelZero.Kernels.LlmKernelSuite.Create(device);
            var watch = Stopwatch.StartNew();
            using LlmModel quant = LlmModel.Load(device, suite, Qwen35ModelFixture.ModelDir, maxSeqLen: 256, precision: LlmWeightPrecision.Auto);
            Measure("int8", quant, watch.Elapsed.TotalSeconds);
        }
    }

    private void Measure(string label, LlmModel model, double loadSeconds)
    {
        int[] prompt = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198];
        _ = model.Prefill(prompt);
        _ = model.Decode(785);
        var sw = Stopwatch.StartNew();
        _ = model.Prefill(prompt);
        double prefillMs = sw.Elapsed.TotalMilliseconds;
        const int steps = 48;
        int token = 785;
        sw.Restart();
        for (int i = 0; i < steps; i++)
        {
            float[] logits = model.Decode(token);
            token = Array.IndexOf(logits, logits.Max());
        }
        double decodeMs = sw.Elapsed.TotalMilliseconds;
        string line = string.Create(
            CultureInfo.InvariantCulture,
            $"qwen3.5-0.8b [{label}] load={loadSeconds:F1}s prefill({prompt.Length} tok)={prefillMs:F0}ms ({prefillMs / prompt.Length:F1} ms/tok) decode={decodeMs / steps:F1} ms/tok = {steps * 1000.0 / decodeMs:F2} tok/s");
        _output.WriteLine(line);
        string dir = Path.Combine(Path.GetTempPath(), "l0build");
        Directory.CreateDirectory(dir);
        File.AppendAllText(Path.Combine(dir, "bench.txt"), $"{DateTime.Now:s} {Environment.GetEnvironmentVariable("IPU_L0_BENCH_TAG")} {line}{Environment.NewLine}");
    }
}