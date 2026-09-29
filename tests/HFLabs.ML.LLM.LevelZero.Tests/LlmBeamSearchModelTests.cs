using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Sequence forking and beam search on real checkpoints (skipped when the model is not present).</summary>
public sealed class Qwen25BeamSearchTests : IClassFixture<Qwen25ModelFixture>
{
    private static readonly int[] Prompt = [9707, 11, 1246, 525, 498, 30];
    private readonly Qwen25ModelFixture _fixture;

    public Qwen25BeamSearchTests(Qwen25ModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void RestoredState_ContinuesExactlyLikeTheOriginalSequence() => BeamModelChecks.RestoreIsExact(_fixture.Model);

    [Fact]
    public void WidthOne_MatchesGreedyGeneration() => BeamModelChecks.WidthOneMatchesGreedy(_fixture.Model, Prompt);

    [Fact]
    public void WiderBeam_ScoresAtLeastAsWellAsGreedy() => BeamModelChecks.WiderBeamNotWorse(_fixture.Model, Prompt);
}

public sealed class Qwen35BeamSearchTests : IClassFixture<Qwen35ModelFixture>
{
    private static readonly int[] Prompt = [9707, 11, 1246, 525, 498, 30];
    private readonly Qwen35ModelFixture _fixture;

    public Qwen35BeamSearchTests(Qwen35ModelFixture fixture) => _fixture = fixture;

    /// <summary>Qwen3.5 has recurrent (Gated DeltaNet) layers: their state must be forked too, not only the KV rows.</summary>
    [Fact]
    public void RestoredState_ContinuesExactlyLikeTheOriginalSequence() => BeamModelChecks.RestoreIsExact(_fixture.Model);

    [Fact]
    public void WidthOne_MatchesGreedyGeneration() => BeamModelChecks.WidthOneMatchesGreedy(_fixture.Model, Prompt);

    [Fact]
    public void WiderBeam_ScoresAtLeastAsWellAsGreedy() => BeamModelChecks.WiderBeamNotWorse(_fixture.Model, Prompt);
}

internal static class BeamModelChecks
{
    private static readonly int[] Prompt = [9707, 11, 1246, 525, 498, 30];

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }
        return best;
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }
        return max;
    }

    /// <summary>Two different continuations, then restore the first fork and compare with an uninterrupted run.</summary>
    public static void RestoreIsExact(LlmModel? model)
    {
        if (model is null)
        {
            return;
        }

        float[] first = model.Prefill(Prompt);
        int basePosition = model.Position;
        int a = ArgMax(first);
        float[] afterA = model.Decode(a);
        int a2 = ArgMax(afterA);
        LlmSequenceState forkA = model.CaptureState(basePosition);
        float[] reference = model.Decode(a2);

        // Wander off along another branch, overwriting the rows and the recurrent state.
        int other = (a + 1) % model.Config.VocabSize;
        _ = model.Decode(other);
        _ = model.Decode((other + 5) % model.Config.VocabSize);
        _ = model.Decode((other + 9) % model.Config.VocabSize);

        model.RestoreState(forkA);
        Assert.Equal(basePosition + 1, model.Position);
        float[] replay = model.Decode(a2);
        Assert.Equal(ArgMax(reference), ArgMax(replay));
        Assert.True(MaxAbsDiff(reference, replay) < 1e-4, $"logits after restore differ by {MaxAbsDiff(reference, replay)}");
    }

    public static void WidthOneMatchesGreedy(LlmModel? model, int[] prompt)
    {
        if (model is null)
        {
            return;
        }

        const int n = 12;
        int[] greedy = model.GenerateGreedy(prompt, n);
        IReadOnlyList<BeamHypothesis> beam = LlmBeamSearch.Run(
            model, prompt, new BeamSearchOptions { BeamWidth = 1, MaxNewTokens = n, LengthPenalty = 0 });
        Assert.Equal(greedy, beam[0].Tokens);
    }

    public static void WiderBeamNotWorse(LlmModel? model, int[] prompt)
    {
        if (model is null)
        {
            return;
        }

        var options = new BeamSearchOptions { BeamWidth = 1, MaxNewTokens = 10, LengthPenalty = 0, NumReturn = 1 };
        BeamHypothesis greedy = LlmBeamSearch.Run(model, prompt, options)[0];
        IReadOnlyList<BeamHypothesis> beams = LlmBeamSearch.Run(model, prompt, options with { BeamWidth = 4, NumReturn = 4 });
        Assert.Equal(4, beams.Count);
        Assert.True(beams[0].LogProb >= greedy.LogProb - 1e-3, $"beam {beams[0].LogProb} < greedy {greedy.LogProb}");
        for (int i = 1; i < beams.Count; i++)
        {
            Assert.True(beams[i - 1].Score >= beams[i].Score);
        }
        Assert.Equal(4, beams.Select(b => string.Join(',', b.Tokens)).Distinct().Count());

        // The reported log-probability must match a teacher-forced re-scoring of the winning tokens.
        float[] logits = model.Prefill(prompt);
        double total = 0;
        foreach (int t in beams[0].Tokens)
        {
            total += LogProbOf(logits, t);
            logits = model.Decode(t);
        }
        Assert.Equal(total, beams[0].LogProb, 2);
    }

    private static double LogProbOf(float[] logits, int token)
    {
        double max = logits.Max();
        double sum = logits.Sum(v => Math.Exp(v - max));
        return logits[token] - (max + Math.Log(sum));
    }
}
