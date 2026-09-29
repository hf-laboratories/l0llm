using System.Text;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Beam search through <see cref="LevelZeroLlmEngine.GenerateBeamAsync"/> on Qwen2.5-0.5B.</summary>
public sealed class LevelZeroLlmEngineBeamTests : IClassFixture<Qwen25EngineFixture>
{
    private readonly Qwen25EngineFixture _fx;

    public LevelZeroLlmEngineBeamTests(Qwen25EngineFixture fx) => _fx = fx;

    [Fact]
    public async Task WidthOne_MatchesStreamedGreedyText()
    {
        if (_fx.Engine is null)
        {
            return;
        }

        const string prompt = "The capital of France is";
        var sb = new StringBuilder();
        await foreach (string piece in _fx.Engine.GenerateAsync(
            prompt, _fx.ModelId, new LlmGenerationOptions { Temperature = 0f, MaxNewTokens = 10 }))
        {
            sb.Append(piece);
        }

        IReadOnlyList<BeamCompletion> beam = await _fx.Engine.GenerateBeamAsync(
            prompt, _fx.ModelId, new BeamSearchOptions { BeamWidth = 1, MaxNewTokens = 10, LengthPenalty = 0 });
        Assert.Single(beam);
        Assert.Equal(sb.ToString(), beam[0].Text);
    }

    [Fact]
    public async Task NBest_AreDistinctSortedAndDecoded()
    {
        if (_fx.Engine is null)
        {
            return;
        }

        IReadOnlyList<BeamCompletion> beam = await _fx.Engine.GenerateBeamAsync(
            "Write one short sentence about the sea:",
            _fx.ModelId,
            new BeamSearchOptions { BeamWidth = 4, NumReturn = 3, MaxNewTokens = 16 });
        Assert.Equal(3, beam.Count);
        Assert.All(beam, b => Assert.False(b.Text.StartsWith("Error:", StringComparison.Ordinal), b.Text));
        Assert.True(beam[0].Hypothesis.Score >= beam[1].Hypothesis.Score);
        Assert.True(beam[1].Hypothesis.Score >= beam[2].Hypothesis.Score);
        Assert.Equal(3, beam.Select(b => b.Text).Distinct().Count());
    }

    [Fact]
    public async Task Constraint_ForbidsTokens_ThroughTheEngine()
    {
        if (_fx.Engine is null)
        {
            return;
        }

        IReadOnlyList<BeamCompletion> free = await _fx.Engine.GenerateBeamAsync(
            "The capital of France is", _fx.ModelId, new BeamSearchOptions { BeamWidth = 2, MaxNewTokens = 6 });
        int banned = free[0].Hypothesis.Tokens[0];

        IReadOnlyList<BeamCompletion> constrained = await _fx.Engine.GenerateBeamAsync(
            "The capital of France is",
            _fx.ModelId,
            new BeamSearchOptions { BeamWidth = 2, MaxNewTokens = 6, Constraint = new Ban(banned) });
        Assert.DoesNotContain(banned, constrained[0].Hypothesis.Tokens);
    }

    [Fact]
    public async Task UnknownModel_ReturnsError()
    {
        if (_fx.Engine is null)
        {
            return;
        }

        IReadOnlyList<BeamCompletion> r = await _fx.Engine.GenerateBeamAsync("Hi", "no-such-model", new BeamSearchOptions());
        Assert.StartsWith("Error:", r[0].Text, StringComparison.Ordinal);
    }

    private sealed class Ban(int token) : IBeamConstraint
    {
        public void Adjust(IReadOnlyList<int> generated, Span<float> logProbs) => logProbs[token] = float.NegativeInfinity;
    }
}
