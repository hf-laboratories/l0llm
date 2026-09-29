using System.Text.Json;
using LevelZero;
using LevelZero.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Teacher-forced comparison of an int8 model against a half-precision reference model.</summary>
internal static class Int8Quality
{
    internal readonly record struct Result(int Positions, double Top1Agreement, double MeanKl, double MeanAbsLogit, double MaxAbsLogit);

    /// <summary>
    /// Feeds each golden case (prompt, then its generated tokens) to both models and compares the next-token logits at
    /// every position.
    /// </summary>
    internal static Result Measure(LlmModel reference, LlmModel quant, string goldenFile, params string[] cases)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", goldenFile);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        int total = 0;
        int agree = 0;
        double meanKl = 0;
        double maxAbs = 0;
        double meanAbs = 0;
        foreach (string name in cases)
        {
            JsonElement c = doc.RootElement.GetProperty("cases").GetProperty(name);
            int[] prompt = c.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] generated = c.GetProperty("generated_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            float[] refLogits = reference.Prefill(prompt);
            float[] quantLogits = quant.Prefill(prompt);
            for (int step = 0; step <= generated.Length; step++)
            {
                total++;
                Compare(refLogits, quantLogits, ref agree, ref meanKl, ref maxAbs, ref meanAbs);
                if (step < generated.Length)
                {
                    refLogits = reference.Decode(generated[step]);
                    quantLogits = quant.Decode(generated[step]);
                }
            }
        }

        return new Result(total, (double)agree / total, meanKl / total, meanAbs / total, maxAbs);
    }

    internal static string Describe(Result r) =>
        $"positions {r.Positions}: top-1 agreement {r.Top1Agreement:P1}, mean KL {r.MeanKl:F5}, mean |dlogit| {r.MeanAbsLogit:F3}, max |dlogit| {r.MaxAbsLogit:F3}";

    private static void Compare(float[] reference, float[] test, ref int agree, ref double meanKl, ref double maxAbs, ref double meanAbs)
    {
        int argRef = 0;
        int argTest = 0;
        double diffSum = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            if (reference[i] > reference[argRef])
            {
                argRef = i;
            }

            if (test[i] > test[argTest])
            {
                argTest = i;
            }

            double d = Math.Abs(reference[i] - test[i]);
            diffSum += d;
            maxAbs = Math.Max(maxAbs, d);
        }

        if (argRef == argTest)
        {
            agree++;
        }

        meanAbs += diffSum / reference.Length;
        meanKl += KlDivergence(reference, test);
    }

    /// <summary>KL(reference || test) of the two softmax distributions.</summary>
    private static double KlDivergence(float[] reference, float[] test)
    {
        double maxR = reference.Max();
        double maxT = test.Max();
        double sumR = 0;
        double sumT = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            sumR += Math.Exp(reference[i] - maxR);
            sumT += Math.Exp(test[i] - maxT);
        }

        double logR = Math.Log(sumR) + maxR;
        double logT = Math.Log(sumT) + maxT;
        double kl = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double lp = reference[i] - logR;
            double lq = test[i] - logT;
            kl += Math.Exp(lp) * (lp - lq);
        }

        return kl;
    }
}

/// <summary>
/// Measures how far the int8 (DP4A) model drifts from the half-precision model on teacher-forced text. Half precision is
/// the reference because it reproduces the HuggingFace float32 golden token for token.
/// </summary>
public sealed class Int8QualityTests : IClassFixture<Qwen25ModelFixture>
{
    private readonly Qwen25ModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public Int8QualityTests(Qwen25ModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Fact]
    public void Qwen25_Int8_UsesAboutHalfTheBytes_AndKeepsTop1Agreement()
    {
        if (_fixture.Model is not { } reference)
        {
            return;
        }

        Assert.True(reference.UsesHalfWeights);
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmKernelSuite suite = LlmKernelSuite.Create(device);
        using LlmModel quant = LlmModel.Load(device, suite, Qwen25ModelFixture.ModelDir, maxSeqLen: 128, precision: LlmWeightPrecision.Auto);
        Assert.True(quant.UsesInt8Weights);
        Assert.True(quant.ProjectionBytes < (reference.ProjectionBytes * 0.6), $"{quant.ProjectionBytes} vs {reference.ProjectionBytes}");

        Int8Quality.Result r = Int8Quality.Measure(reference, quant, "qwen2.5-0.5b-instruct.generation.golden.json", "chat", "raw");
        _output.WriteLine($"{Int8Quality.Describe(r)}; bytes {quant.ProjectionBytes / 1e6:F0} MB vs {reference.ProjectionBytes / 1e6:F0} MB");
        Assert.True(r.Top1Agreement >= 0.85, $"top-1 agreement {r.Top1Agreement:P1}");
        Assert.True(r.MeanKl < 0.02, $"mean KL {r.MeanKl}");
    }
}

/// <summary>The same int8-versus-half comparison for the hybrid Qwen3.5 model (Gated DeltaNet plus gated attention).</summary>
public sealed class Qwen35Int8QualityTests : IClassFixture<Qwen35ModelFixture>
{
    private readonly Qwen35ModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public Qwen35Int8QualityTests(Qwen35ModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Fact]
    public void Qwen35_Int8_KeepsTop1Agreement()
    {
        if (_fixture.Model is not { } reference)
        {
            return;
        }

        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmKernelSuite suite = LlmKernelSuite.Create(device);
        using LlmModel quant = LlmModel.Load(device, suite, Qwen35ModelFixture.ModelDir, maxSeqLen: 128, precision: LlmWeightPrecision.Auto);
        Assert.True(quant.UsesInt8Weights);
        Int8Quality.Result r = Int8Quality.Measure(reference, quant, "qwen3.5-0.8b.generation.golden.json", "chat", "raw");
        _output.WriteLine($"{Int8Quality.Describe(r)}; bytes {quant.ProjectionBytes / 1e6:F0} MB vs {reference.ProjectionBytes / 1e6:F0} MB");
        Assert.True(r.Top1Agreement >= 0.85, $"top-1 agreement {r.Top1Agreement:P1}");
        Assert.True(r.MeanKl < 0.02, $"mean KL {r.MeanKl}");
    }
}
