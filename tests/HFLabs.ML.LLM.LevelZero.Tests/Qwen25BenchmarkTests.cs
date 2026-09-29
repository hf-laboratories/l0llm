using System.Diagnostics;
using System.Globalization;
using LevelZero;
using LevelZero.Kernels;
using Xunit;
using Xunit.Abstractions;
namespace HFLabs.ML.LLM.LevelZero.Tests;
/// <summary>
/// Decode throughput on the real checkpoint. Opt in with <c>IPU_L0_BENCH=1</c>; otherwise the test returns
/// immediately so the normal suite stays fast. <c>IPU_L0_BENCH_PRECISION</c> picks <c>fp16</c>, <c>int8</c> or
/// <c>both</c> (default). Results go to the test output and to <c>%TEMP%\l0build\bench.txt</c>.
/// </summary>
public sealed class Qwen25BenchmarkTests : IClassFixture<Qwen25ModelFixture>
{
    private readonly Qwen25ModelFixture _fx;
    private readonly ITestOutputHelper _output;
    public Qwen25BenchmarkTests(Qwen25ModelFixture fx, ITestOutputHelper output)
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
            using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
            using LlmKernelSuite suite = LlmKernelSuite.Create(device);
            var watch = Stopwatch.StartNew();
            using LlmModel model = LlmModel.Load(device, suite, Qwen25ModelFixture.ModelDir, maxSeqLen: 256, precision: LlmWeightPrecision.Auto);
            Measure("int8", model, watch.Elapsed.TotalSeconds);
        }
    }
    private void Measure(string label, LlmModel model, double loadSeconds)
    {
        int[] prompt = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198];
        // Warm-up: first launches compile/patch kernels lazily on some drivers.
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
            $"[{label}] load={loadSeconds:F1}s prefill({prompt.Length} tok)={prefillMs:F0}ms ({prefillMs / prompt.Length:F1} ms/tok) decode={decodeMs / steps:F1} ms/tok = {steps * 1000.0 / decodeMs:F2} tok/s");
        _output.WriteLine(line);
        string dir = Path.Combine(Path.GetTempPath(), "l0build");
        Directory.CreateDirectory(dir);
        File.AppendAllText(Path.Combine(dir, "bench.txt"), $"{DateTime.Now:s} {Environment.GetEnvironmentVariable("IPU_L0_BENCH_TAG")} {line}{Environment.NewLine}");
    }
}