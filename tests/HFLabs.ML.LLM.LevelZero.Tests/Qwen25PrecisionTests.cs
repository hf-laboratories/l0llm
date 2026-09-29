using System.Text.Json;
using LevelZero;
using LevelZero.Kernels;
using Xunit;
namespace HFLabs.ML.LLM.LevelZero.Tests;
/// <summary>
/// The exact-match fixtures load half precision (Float16) for this BF16 checkpoint; both 16/32-bit storage modes must reproduce the
/// HuggingFace float32 token sequence.
/// </summary>
public sealed class Qwen25PrecisionTests : IClassFixture<Qwen25ModelFixture>
{
    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-0.5b-instruct.generation.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });
    private readonly Qwen25ModelFixture _fixture;
    public Qwen25PrecisionTests(Qwen25ModelFixture fixture) => _fixture = fixture;
    [Fact]
    public void Float16Precision_StoresTheBf16CheckpointAsHalf()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }
        Assert.True(model.UsesHalfWeights);
        // 24 layers of projections plus the LM head, 2 bytes each: comfortably under 1.1 GB, and above 0.9 GB.
        Assert.InRange(model.ProjectionBytes, 900_000_000L, 1_100_000_000L);
    }
    [Fact]
    public void Float32Precision_StillMatchesTransformersToken_ForToken()
    {
        if (_fixture.Model is null)
        {
            return;
        }
        JsonElement c = Golden.Value.GetProperty("cases").GetProperty("raw");
        int[] prompt = c.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[] expected = c.GetProperty("generated_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmKernelSuite suite = LlmKernelSuite.Create(device);
        using LlmModel model = LlmModel.Load(
            device, suite, Qwen25ModelFixture.ModelDir, maxSeqLen: 64, precision: LlmWeightPrecision.Float32);
        Assert.False(model.UsesHalfWeights);
        Assert.InRange(model.ProjectionBytes, 1_900_000_000L, 2_100_000_000L);
        Assert.Equal(expected, model.GenerateGreedy(prompt, expected.Length));
    }
}