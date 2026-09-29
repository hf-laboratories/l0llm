using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Beam search logic on a small CPU model whose next-token distribution depends on the whole history.</summary>
public sealed class LlmBeamSearchTests
{
    private const int Vocab = 5;
    private const int Stop = 4;

    /// <summary>Logits are a fixed pseudo-random function of (last token, history length); state is the history.</summary>
    private sealed class FakeModel : IBeamSequenceModel
    {
        private List<int> _history = [];
        private int _promptLength;
        public int Restores;
        public int Decodes;

        public int VocabSize => Vocab;

        public float[] Prefill(IReadOnlyList<int> promptIds)
        {
            _history = [.. promptIds];
            _promptLength = promptIds.Count;
            return Logits();
        }

        public float[] Decode(int tokenId)
        {
            Decodes++;
            _history.Add(tokenId);
            return Logits();
        }

        public object CaptureState() => _history.Skip(_promptLength).ToArray();

        public void RestoreState(object state)
        {
            Restores++;
            _history = [.. _history.Take(_promptLength), .. (int[])state];
        }

        public float[] Logits()
        {
            int h = 17;
            foreach (int t in _history)
            {
                h = unchecked((h * 31) + t + 7);
            }
            var rng = new Random(h);
            var logits = new float[Vocab];
            for (int i = 0; i < Vocab; i++)
            {
                logits[i] = (float)((rng.NextDouble() * 6) - 3);
            }
            return logits;
        }
    }

    private static float[] LogSoftmax(float[] logits)
    {
        double max = logits.Max();
        double sum = logits.Sum(v => Math.Exp(v - max));
        double log = max + Math.Log(sum);
        return logits.Select(v => (float)(v - log)).ToArray();
    }

    /// <summary>Exhaustive search over all sequences of up to <paramref name="depth"/> tokens (alpha = 0).</summary>
    private static (double Best, int[] Tokens) BruteForce(int[] prompt, int depth, Func<int[], int, bool>? allowed = null)
    {
        double best = double.NegativeInfinity;
        int[] bestTokens = [];

        void Recurse(List<int> gen, double logp)
        {
            var model = new FakeModel();
            model.Prefill(prompt);
            foreach (int t in gen)
            {
                model.Decode(t);
            }
            float[] lp = LogSoftmax(model.Logits());
            for (int t = 0; t < Vocab; t++)
            {
                if (allowed is not null && !allowed([.. gen], t))
                {
                    continue;
                }
                double total = logp + lp[t];
                if (t == Stop || gen.Count + 1 == depth)
                {
                    if (total > best)
                    {
                        best = total;
                        bestTokens = [.. gen, t];
                    }
                }
                else
                {
                    gen.Add(t);
                    Recurse(gen, total);
                    gen.RemoveAt(gen.Count - 1);
                }
            }
        }

        Recurse([], 0);
        return (best, bestTokens);
    }

    private static int[] Greedy(int[] prompt, int max)
    {
        var model = new FakeModel();
        float[] logits = model.Prefill(prompt);
        var result = new List<int>();
        for (int i = 0; i < max; i++)
        {
            int best = Array.IndexOf(logits, logits.Max());
            result.Add(best);
            if (best == Stop)
            {
                break;
            }
            logits = model.Decode(best);
        }
        return [.. result];
    }

    [Fact]
    public void WidthOne_EqualsGreedy()
    {
        for (int seed = 0; seed < 20; seed++)
        {
            int[] prompt = [seed, seed + 1];
            IReadOnlyList<BeamHypothesis> r = LlmBeamSearch.Run(
                new FakeModel(), prompt,
                new BeamSearchOptions { BeamWidth = 1, MaxNewTokens = 6, StopTokenIds = [Stop], LengthPenalty = 0 });
            Assert.Equal(Greedy(prompt, 6), r[0].Tokens);
        }
    }

    [Fact]
    public void WideBeam_FindsTheExhaustiveOptimum()
    {
        for (int seed = 0; seed < 20; seed++)
        {
            int[] prompt = [seed];
            (double best, int[] tokens) = BruteForce(prompt, 4);
            IReadOnlyList<BeamHypothesis> r = LlmBeamSearch.Run(
                new FakeModel(), prompt,
                new BeamSearchOptions { BeamWidth = 1024, MaxNewTokens = 4, StopTokenIds = [Stop], LengthPenalty = 0 });
            Assert.Equal(tokens, r[0].Tokens);
            Assert.Equal(best, r[0].LogProb, 4);
        }
    }

    [Fact]
    public void Beam_NeverScoresWorseThanGreedy_AndSometimesBetter()
    {
        int better = 0;
        for (int seed = 0; seed < 40; seed++)
        {
            int[] prompt = [seed, 3];
            var options = new BeamSearchOptions { BeamWidth = 4, MaxNewTokens = 5, StopTokenIds = [Stop], LengthPenalty = 0 };
            BeamHypothesis beam = LlmBeamSearch.Run(new FakeModel(), prompt, options)[0];
            BeamHypothesis greedy = LlmBeamSearch.Run(new FakeModel(), prompt, options with { BeamWidth = 1 })[0];
            Assert.True(beam.LogProb >= greedy.LogProb - 1e-4, $"seed {seed}: beam {beam.LogProb} < greedy {greedy.LogProb}");
            if (beam.LogProb > greedy.LogProb + 1e-4)
            {
                better++;
            }
        }
        Assert.True(better > 0, "beam search never improved on greedy; the model is too easy for the test to mean anything");
    }

    [Fact]
    public void Constraint_ForbiddenTokensNeverAppear_AndOptimumStaysExhaustive()
    {
        static bool Allowed(int[] gen, int t) => t != 2 && !(gen.Length > 0 && gen[^1] == t);
        var constraint = new DelegateConstraint(Allowed);
        for (int seed = 0; seed < 10; seed++)
        {
            int[] prompt = [seed];
            (double best, int[] tokens) = BruteForce(prompt, 4, Allowed);
            IReadOnlyList<BeamHypothesis> r = LlmBeamSearch.Run(
                new FakeModel(), prompt,
                new BeamSearchOptions { BeamWidth = 1024, MaxNewTokens = 4, StopTokenIds = [Stop], LengthPenalty = 0, Constraint = constraint });
            Assert.DoesNotContain(2, r[0].Tokens);
            Assert.Equal(tokens, r[0].Tokens);
            Assert.Equal(best, r[0].LogProb, 4);
        }
    }

    [Fact]
    public void NumReturn_IsSortedBestFirst_AndDistinct()
    {
        IReadOnlyList<BeamHypothesis> r = LlmBeamSearch.Run(
            new FakeModel(), [1, 2],
            new BeamSearchOptions { BeamWidth = 6, NumReturn = 4, MaxNewTokens = 5, StopTokenIds = [Stop] });
        Assert.Equal(4, r.Count);
        for (int i = 1; i < r.Count; i++)
        {
            Assert.True(r[i - 1].Score >= r[i].Score);
        }
        Assert.Equal(4, r.Select(h => string.Join(',', h.Tokens)).Distinct().Count());
    }

    [Fact]
    public void StopToken_EndsHypothesis_AndLengthLimitTruncates()
    {
        IReadOnlyList<BeamHypothesis> r = LlmBeamSearch.Run(
            new FakeModel(), [3],
            new BeamSearchOptions { BeamWidth = 3, NumReturn = 3, MaxNewTokens = 3, StopTokenIds = [Stop] });
        foreach (BeamHypothesis h in r)
        {
            Assert.True(h.Tokens.Length <= 3);
            Assert.Equal(h.Finished, h.Tokens[^1] == Stop);
            Assert.DoesNotContain(Stop, h.Tokens.Take(h.Tokens.Length - 1));
        }
    }

    [Fact]
    public void LengthPenalty_PrefersLongerOutputThanNoPenalty()
    {
        int longer = 0;
        for (int seed = 0; seed < 40; seed++)
        {
            var o = new BeamSearchOptions { BeamWidth = 8, MaxNewTokens = 6, StopTokenIds = [Stop] };
            int none = LlmBeamSearch.Run(new FakeModel(), [seed], o with { LengthPenalty = 0 })[0].Tokens.Length;
            int pen = LlmBeamSearch.Run(new FakeModel(), [seed], o with { LengthPenalty = 2 })[0].Tokens.Length;
            Assert.True(pen >= none, $"seed {seed}: {pen} < {none}");
            if (pen > none)
            {
                longer++;
            }
        }
        Assert.True(longer > 0);
    }

    [Fact]
    public void Restores_AreUsedForEveryDecodedBeam()
    {
        var model = new FakeModel();
        LlmBeamSearch.Run(model, [1], new BeamSearchOptions { BeamWidth = 3, MaxNewTokens = 4, StopTokenIds = [Stop] });
        Assert.True(model.Decodes > 0);
        Assert.Equal(model.Decodes, model.Restores);
    }

    [Fact]
    public void FinalAdjust_ReordersHypotheses()
    {
        var o = new BeamSearchOptions { BeamWidth = 6, NumReturn = 6, MaxNewTokens = 4, StopTokenIds = [Stop] };
        IReadOnlyList<BeamHypothesis> plain = LlmBeamSearch.Run(new FakeModel(), [2], o);
        int[] worst = plain[^1].Tokens;
        var constraint = new DelegateConstraint((_, _) => true, gen => gen.SequenceEqual(worst) ? 100f : 0f);
        IReadOnlyList<BeamHypothesis> boosted = LlmBeamSearch.Run(new FakeModel(), [2], o with { Constraint = constraint });
        Assert.Equal(worst, boosted[0].Tokens);
    }

    [Fact]
    public void ValidVocabSize_MasksPaddingTokens()
    {
        // Tokens 3 and 4 (including the stop token) are "padding": nothing may pick them, so no hypothesis finishes.
        IReadOnlyList<BeamHypothesis> r = LlmBeamSearch.Run(
            new FakeModel(), [1],
            new BeamSearchOptions { BeamWidth = 3, NumReturn = 3, MaxNewTokens = 4, StopTokenIds = [Stop], ValidVocabSize = 3 });
        Assert.All(r, h =>
        {
            Assert.All(h.Tokens, t => Assert.InRange(t, 0, 2));
            Assert.False(h.Finished);
            Assert.Equal(4, h.Tokens.Length);
        });
    }

    [Fact]
    public void RejectsBadArguments()
    {
        Assert.Throws<ArgumentException>(() => LlmBeamSearch.Run(new FakeModel(), [], new BeamSearchOptions()));
        Assert.Throws<ArgumentOutOfRangeException>(() => LlmBeamSearch.Run(new FakeModel(), [1], new BeamSearchOptions { BeamWidth = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => LlmBeamSearch.Run(new FakeModel(), [1], new BeamSearchOptions { MaxNewTokens = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => LlmBeamSearch.Run(new FakeModel(), [1], new BeamSearchOptions { LengthPenalty = -1 }));
    }

    private sealed class DelegateConstraint(Func<int[], int, bool> allowed, Func<IReadOnlyList<int>, float>? finalAdjust = null) : IBeamConstraint
    {
        public void Adjust(IReadOnlyList<int> generated, Span<float> logProbs)
        {
            int[] gen = [.. generated];
            for (int t = 0; t < logProbs.Length; t++)
            {
                if (!allowed(gen, t))
                {
                    logProbs[t] = float.NegativeInfinity;
                }
            }
        }

        public float FinalAdjust(IReadOnlyList<int> generated) => finalAdjust?.Invoke(generated) ?? 0f;
    }
}
