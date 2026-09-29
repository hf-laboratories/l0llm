using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Group (multi-beam) search logic on a CPU model whose distribution depends on the whole history.</summary>
public sealed class LlmGroupBeamSearchTests
{
    private const int Vocab = 6;
    private const int Stop = 5;

    private sealed class FakeModel : IBeamSequenceModel
    {
        private List<int> _history = [];
        private int _promptLength;

        public int VocabSize => Vocab;

        public float[] Prefill(IReadOnlyList<int> promptIds)
        {
            _history = [.. promptIds];
            _promptLength = promptIds.Count;
            return Logits();
        }

        public float[] Decode(int tokenId)
        {
            _history.Add(tokenId);
            return Logits();
        }

        public object CaptureState() => _history.Skip(_promptLength).ToArray();

        public void RestoreState(object state) => _history = [.. _history.Take(_promptLength), .. (int[])state];

        private float[] Logits()
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

    private sealed class Probe(Func<IReadOnlyList<int>, float>? support = null, int? ban = null) : IGroupBeamConstraint
    {
        public List<(int Step, int Count)> Calls { get; } = [];

        public void Adjust(int step, IReadOnlyList<int> generated, Span<float> logProbs)
        {
            Calls.Add((step, generated.Count));
            if (ban is int b)
            {
                logProbs[b] = float.NegativeInfinity;
            }
        }

        public float Support(IReadOnlyList<int> generated) => support?.Invoke(generated) ?? 0f;
    }

    [Fact]
    public void SingleGroup_EqualsPlainBeamSearch()
    {
        for (int seed = 0; seed < 15; seed++)
        {
            int[] prompt = [seed, 2];
            BeamHypothesis plain = LlmBeamSearch.Run(
                new FakeModel(), prompt,
                new BeamSearchOptions { BeamWidth = 4, MaxNewTokens = 6, StopTokenIds = [Stop], LengthPenalty = 1f })[0];
            GroupBeamHypothesis grouped = LlmGroupBeamSearch.Run(
                new FakeModel(), prompt,
                new GroupBeamSearchOptions { Groups = [new BeamGroup("g", 4)], MaxNewTokens = 6, StopTokenIds = [Stop], LengthPenalty = 1f })[0];
            Assert.Equal(plain.Tokens, grouped.Hypothesis.Tokens);
            Assert.Equal(plain.Score, grouped.Hypothesis.Score, 4);
        }
    }

    [Fact]
    public void Diversity_SpreadsGroupsOverDifferentFirstTokens()
    {
        for (int seed = 0; seed < 15; seed++)
        {
            IReadOnlyList<GroupBeamHypothesis> r = LlmGroupBeamSearch.Run(
                new FakeModel(), [seed],
                new GroupBeamSearchOptions
                {
                    Groups = [new BeamGroup("fire", 1), new BeamGroup("ice", 1), new BeamGroup("thunder", 1)],
                    MaxNewTokens = 4,
                    StopTokenIds = [Stop],
                    DiversityPenalty = 100f,
                    NumReturn = 20,
                });
            // A width-1 group commits to one first token; the penalty makes the three groups commit to three different ones.
            var firstByGroup = r.GroupBy(h => h.Group).ToDictionary(g => g.Key, g => g.Select(h => h.Hypothesis.Tokens[0]).ToHashSet());
            Assert.Equal(3, firstByGroup.Count);
            Assert.True(r.Select(h => h.Hypothesis.Tokens[0]).Distinct().Count() >= 3, $"seed {seed}");
        }
    }

    [Fact]
    public void WithoutDiversity_GroupsMayConverge_AndDuplicatesAreMerged()
    {
        IReadOnlyList<GroupBeamHypothesis> r = LlmGroupBeamSearch.Run(
            new FakeModel(), [3],
            new GroupBeamSearchOptions
            {
                Groups = [new BeamGroup("a", 2), new BeamGroup("b", 2), new BeamGroup("c", 2)],
                MaxNewTokens = 4,
                StopTokenIds = [Stop],
                DiversityPenalty = 0f,
                NumReturn = 6,
            });
        Assert.Equal(r.Count, r.Select(h => string.Join(',', h.Hypothesis.Tokens)).Distinct().Count());
    }

    [Fact]
    public void Constraint_IsToldTheStepAndTheGeneratedTokens()
    {
        var a = new Probe();
        var b = new Probe();
        LlmGroupBeamSearch.Run(
            new FakeModel(), [1],
            new GroupBeamSearchOptions
            {
                Groups = [new BeamGroup("a", 2, a), new BeamGroup("b", 2, b)],
                MaxNewTokens = 3,
                StopTokenIds = [Stop],
            });
        Assert.All(a.Calls.Concat(b.Calls), c => Assert.Equal(c.Step, c.Count));
        Assert.Contains(a.Calls, c => c.Step == 0);
        Assert.Contains(b.Calls, c => c.Step == 1);
    }

    [Fact]
    public void Constraint_OnlyAffectsItsOwnGroup()
    {
        for (int seed = 0; seed < 15; seed++)
        {
            IReadOnlyList<GroupBeamHypothesis> r = LlmGroupBeamSearch.Run(
                new FakeModel(), [seed],
                new GroupBeamSearchOptions
                {
                    Groups = [new BeamGroup("banned", 3, new Probe(ban: 0)), new BeamGroup("free", 3)],
                    MaxNewTokens = 4,
                    StopTokenIds = [Stop],
                    DiversityPenalty = 0f,
                    NumReturn = 6,
                });
            foreach (GroupBeamHypothesis h in r.Where(h => h.Group == "banned"))
            {
                Assert.DoesNotContain(0, h.Hypothesis.Tokens);
            }
        }
    }

    [Fact]
    public void Support_FromAnyGroup_ReordersResults_WhenSummed()
    {
        var options = new GroupBeamSearchOptions
        {
            Groups = [new BeamGroup("a", 4), new BeamGroup("b", 4)],
            MaxNewTokens = 3,
            StopTokenIds = [Stop],
            NumReturn = 8,
            DiversityPenalty = 0f,
        };
        IReadOnlyList<GroupBeamHypothesis> plain = LlmGroupBeamSearch.Run(new FakeModel(), [2], options);
        int[] worst = plain[^1].Hypothesis.Tokens;

        var supportive = new Probe(g => g.SequenceEqual(worst) ? 50f : 0f);
        var neutral = new Probe();
        GroupBeamSearchOptions boosted = options with
        {
            Groups = [new BeamGroup("a", 4, supportive), new BeamGroup("b", 4, neutral)],
        };
        Assert.Equal(worst, LlmGroupBeamSearch.Run(new FakeModel(), [2], boosted)[0].Hypothesis.Tokens);

        // Min: the neutral group gives no support, so the boost disappears.
        GroupBeamHypothesis minTop = LlmGroupBeamSearch.Run(new FakeModel(), [2], boosted with { Alignment = AlignmentMode.Min })[0];
        Assert.NotEqual(worst, minTop.Hypothesis.Tokens);
        Assert.Equal(0f, minTop.Alignment);
    }

    [Fact]
    public void Result_ReportsEveryJudgingGroupsSupport()
    {
        IReadOnlyList<GroupBeamHypothesis> r = LlmGroupBeamSearch.Run(
            new FakeModel(), [4],
            new GroupBeamSearchOptions
            {
                Groups = [new BeamGroup("a", 2, new Probe(_ => 1f)), new BeamGroup("b", 2, new Probe(_ => 2f)), new BeamGroup("c", 2)],
                MaxNewTokens = 3,
                StopTokenIds = [Stop],
            });
        GroupBeamHypothesis top = r[0];
        Assert.Equal(2, top.Support.Count);
        Assert.Equal(1f, top.Support["a"]);
        Assert.Equal(2f, top.Support["b"]);
        Assert.Equal(3f, top.Alignment);
    }

    [Fact]
    public void RejectsBadArguments()
    {
        static GroupBeamSearchOptions With(params BeamGroup[] groups) => new() { Groups = groups };
        Assert.Throws<ArgumentException>(() => LlmGroupBeamSearch.Run(new FakeModel(), [1], With()));
        Assert.Throws<ArgumentException>(() => LlmGroupBeamSearch.Run(new FakeModel(), [1], With(new BeamGroup("x", 1), new BeamGroup("x", 1))));
        Assert.Throws<ArgumentOutOfRangeException>(() => LlmGroupBeamSearch.Run(new FakeModel(), [1], With(new BeamGroup("x", 0))));
        Assert.Throws<ArgumentException>(() => LlmGroupBeamSearch.Run(new FakeModel(), [], With(new BeamGroup("x", 1))));
    }
}
