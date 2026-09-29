using System.Text.Json;
using Xunit;
namespace HFLabs.ML.LLM.LevelZero.Tests;
/// <summary>
/// Batched prefill must leave the model in the same state as feeding the prompt one token at a time: the same
/// logits for the next token, and the same KV cache (checked by decoding one more token afterwards).
/// </summary>
public sealed class Qwen25PrefillTests : IClassFixture<Qwen25ModelFixture>
{
    private static readonly Lazy<int[]> ChatPrompt = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-0.5b-instruct.generation.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").GetProperty("chat").GetProperty("prompt_ids")
            .EnumerateArray().Select(e => e.GetInt32()).ToArray();
    });
    private readonly Qwen25ModelFixture _fixture;
    public Qwen25PrefillTests(Qwen25ModelFixture fixture) => _fixture = fixture;
    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }
        return best;
    }
    private static double MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }
        return max;
    }
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(35)]
    public void Prefill_MatchesTokenByTokenDecode(int length)
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }
        int[] prompt = ChatPrompt.Value.Take(length).ToArray();
        Assert.Equal(length, prompt.Length);
        // Reference: one token per pass, no batching involved.
        float[] sequential = [];
        for (int i = 0; i < prompt.Length; i++)
        {
            sequential = model.ComputeLogits(prompt[i], i);
        }
        int next = ArgMax(sequential);
        float[] sequentialAfter = model.ComputeLogits(next, prompt.Length);
        float[] batched = model.Prefill(prompt);
        Assert.Equal(prompt.Length, model.Position);
        float[] batchedAfter = model.Decode(next);
        Assert.Equal(next, ArgMax(batched));
        Assert.True(MaxAbsDiff(sequential, batched) < 2e-3, $"prefill logits differ by {MaxAbsDiff(sequential, batched)}");
        Assert.Equal(ArgMax(sequentialAfter), ArgMax(batchedAfter));
        Assert.True(
            MaxAbsDiff(sequentialAfter, batchedAfter) < 2e-3,
            $"logits after the next token differ by {MaxAbsDiff(sequentialAfter, batchedAfter)}: the KV cache differs");
    }
    [Fact]
    public void Prefill_RestartsFromPositionZeroEachTime()
    {
        if (_fixture.Model is not { } model)
        {
            return;
        }
        int[] prompt = ChatPrompt.Value.Take(12).ToArray();
        float[] first = model.Prefill(prompt);
        _ = model.Prefill(ChatPrompt.Value.Take(20).ToArray());
        float[] again = model.Prefill(prompt);
        Assert.Equal(12, model.Position);
        Assert.True(MaxAbsDiff(first, again) < 1e-4);
    }
}