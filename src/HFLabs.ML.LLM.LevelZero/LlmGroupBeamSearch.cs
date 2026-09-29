namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// Steers one beam group. Unlike <see cref="IBeamConstraint"/> it is told the decoding step, so a group can change
/// its pull over time (for example a phase that rotates with the step).
/// </summary>
public interface IGroupBeamConstraint
{
    /// <summary>
    /// Edits the next-token log-probabilities of one beam of this group in place.
    /// </summary>
    /// <param name="step">Number of tokens the beam has generated so far (equal to <paramref name="generated"/>.Count).</param>
    /// <param name="generated">The beam's generated tokens.</param>
    /// <param name="logProbs">Log-softmax of the model's logits; set an entry to negative infinity to forbid it.</param>
    void Adjust(int step, IReadOnlyList<int> generated, Span<float> logProbs);

    /// <summary>
    /// How well a finished (or truncated) hypothesis is supported from this group's point of view. Every hypothesis,
    /// whichever group produced it, is judged by every group.
    /// </summary>
    float Support(IReadOnlyList<int> generated) => 0f;
}

/// <summary>One group of beams with its own steering.</summary>
/// <param name="Name">Unique group name, reported with each result.</param>
/// <param name="Width">Beams kept alive in this group.</param>
/// <param name="Constraint">Optional steering; null leaves the group unconstrained.</param>
public sealed record BeamGroup(string Name, int Width, IGroupBeamConstraint? Constraint = null);

/// <summary>How the per-group support values of a hypothesis are combined.</summary>
public enum AlignmentMode
{
    /// <summary>Sum of the groups' support: any group can lift a hypothesis.</summary>
    Sum,

    /// <summary>Smallest support: a hypothesis is only as good as its least supportive group.</summary>
    Min,
}

/// <summary>Options for <see cref="LlmGroupBeamSearch"/>.</summary>
public sealed record GroupBeamSearchOptions
{
    /// <summary>The beam groups; at least one.</summary>
    public IReadOnlyList<BeamGroup> Groups { get; init; } = [];

    /// <summary>Maximum number of generated tokens per hypothesis.</summary>
    public int MaxNewTokens { get; init; } = 64;

    /// <summary>Score is <c>(logProb + AlignmentWeight * alignment) / length^LengthPenalty</c>.</summary>
    public float LengthPenalty { get; init; } = 1.0f;

    /// <summary>
    /// Hamming diversity: a token already chosen by an earlier group in this step is ranked lower by this many
    /// nats in later groups (the cumulative log-probability is not changed). 0 lets groups converge.
    /// </summary>
    public float DiversityPenalty { get; init; } = 0.5f;

    /// <summary>Weight of the combined group support in the final score.</summary>
    public float AlignmentWeight { get; init; } = 1.0f;

    /// <summary>How the groups' support values are combined.</summary>
    public AlignmentMode Alignment { get; init; } = AlignmentMode.Sum;

    /// <summary>Tokens that end a hypothesis.</summary>
    public IReadOnlyCollection<int>? StopTokenIds { get; init; }

    /// <summary>How many distinct hypotheses to return, best first.</summary>
    public int NumReturn { get; init; } = 1;

    /// <summary>Token ids at or above this value are never chosen. Null allows all.</summary>
    public int? ValidVocabSize { get; init; }
}

/// <summary>A group-beam-search result.</summary>
/// <param name="Group">The group whose beam produced it.</param>
/// <param name="Hypothesis">Tokens with the final score (log-probability plus weighted alignment, length-normalised).</param>
/// <param name="Support">Each constrained group's support for the tokens.</param>
/// <param name="Alignment">The combined support (sum or minimum).</param>
public sealed record GroupBeamHypothesis(
    string Group,
    BeamHypothesis Hypothesis,
    IReadOnlyDictionary<string, float> Support,
    float Alignment);

/// <summary>A decoded group-beam-search result.</summary>
/// <param name="Text">The generated text (special tokens skipped).</param>
/// <param name="Result">The hypothesis with its group and support values.</param>
public sealed record GroupBeamCompletion(string Text, GroupBeamHypothesis Result);

/// <summary>
/// Diverse multi-group beam search: several groups of beams grow side by side, each steered by its own constraint,
/// nudged apart by a diversity penalty, and every finished hypothesis is finally judged by all groups together.
/// </summary>
/// <remarks>
/// Cost per step is the sum of the group widths in decodes (beams are decoded one after another).
/// A single group with the same width, length penalty and stop tokens behaves like <see cref="LlmBeamSearch"/>.
/// </remarks>
public static class LlmGroupBeamSearch
{
    /// <summary>Runs group beam search on a model resident on the GPU.</summary>
    public static IReadOnlyList<GroupBeamHypothesis> Run(LlmModel model, IReadOnlyList<int> promptIds, GroupBeamSearchOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        return Run(new LlmBeamModel(model), promptIds, options);
    }

    /// <summary>Runs group beam search on any <see cref="IBeamSequenceModel"/>.</summary>
    public static IReadOnlyList<GroupBeamHypothesis> Run(IBeamSequenceModel model, IReadOnlyList<int> promptIds, GroupBeamSearchOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxNewTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.NumReturn);
        ArgumentOutOfRangeException.ThrowIfNegative(options.LengthPenalty);
        ArgumentOutOfRangeException.ThrowIfNegative(options.DiversityPenalty);
        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptIds));
        }
        if (options.Groups.Count == 0)
        {
            throw new ArgumentException("At least one beam group is required.", nameof(options));
        }
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (BeamGroup g in options.Groups)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(g.Width);
            if (!names.Add(g.Name))
            {
                throw new ArgumentException($"Duplicate group name '{g.Name}'.", nameof(options));
            }
        }

        var stops = new HashSet<int>(options.StopTokenIds ?? []);
        int totalWidth = options.Groups.Sum(g => g.Width);

        float[] first = LlmBeamSearch.LogSoftmax(model.Prefill(promptIds));
        MaskInvalid(first, options.ValidVocabSize);
        object rootState = model.CaptureState();
        int vocab = first.Length;

        var states = new List<GroupState>();
        foreach (BeamGroup g in options.Groups)
        {
            float[] lp = (float[])first.Clone();
            g.Constraint?.Adjust(0, [], lp);
            states.Add(new GroupState(g, [new Beam([], 0f, lp, rootState)]));
        }

        var finished = new List<Raw>();
        for (int step = 0; step < options.MaxNewTokens; step++)
        {
            var chosenCount = new int[vocab];
            var plans = new List<(GroupState State, List<Candidate> Next)>();
            foreach (GroupState gs in states)
            {
                if (gs.Alive.Count == 0)
                {
                    continue;
                }
                List<Candidate> candidates = Select(gs.Alive, 2 * gs.Group.Width, chosenCount, options.DiversityPenalty);
                var next = new List<Candidate>(gs.Group.Width);
                for (int rank = 0; rank < candidates.Count; rank++)
                {
                    Candidate c = candidates[rank];
                    if (stops.Contains(c.Token))
                    {
                        if (rank < gs.Group.Width)
                        {
                            finished.Add(new Raw(gs.Group.Name, Append(gs.Alive[c.Beam].Tokens, c.Token), c.Total, true));
                            chosenCount[c.Token]++;
                        }
                    }
                    else if (next.Count < gs.Group.Width)
                    {
                        next.Add(c);
                    }
                }
                foreach (Candidate c in next)
                {
                    chosenCount[c.Token]++;
                }
                plans.Add((gs, next));
            }

            foreach (GroupState gs in states)
            {
                if (!plans.Any(p => p.State == gs))
                {
                    gs.Alive = [];
                }
            }
            if (plans.Count == 0 || plans.All(p => p.Next.Count == 0))
            {
                foreach (GroupState gs in states)
                {
                    gs.Alive = [];
                }
                break;
            }

            bool last = step == options.MaxNewTokens - 1;
            foreach ((GroupState gs, List<Candidate> next) in plans)
            {
                var grown = new List<Beam>(next.Count);
                foreach (Candidate c in next)
                {
                    Beam parent = gs.Alive[c.Beam];
                    int[] tokens = Append(parent.Tokens, c.Token);
                    if (last)
                    {
                        grown.Add(new Beam(tokens, c.Total, [], null));
                        continue;
                    }
                    model.RestoreState(parent.State!);
                    float[] lp = LlmBeamSearch.LogSoftmax(model.Decode(c.Token));
                    MaskInvalid(lp, options.ValidVocabSize);
                    gs.Group.Constraint?.Adjust(tokens.Length, tokens, lp);
                    grown.Add(new Beam(tokens, c.Total, lp, model.CaptureState()));
                }
                gs.Alive = grown;
            }

            if (!last && finished.Count >= totalWidth)
            {
                float worst = finished.Min(r => LlmBeamSearch.Normalize(r.LogProb, r.Tokens.Length, options.LengthPenalty));
                float bestAlive = states.SelectMany(s => s.Alive)
                    .Select(b => LlmBeamSearch.Normalize(b.LogProb, b.Tokens.Length, options.LengthPenalty))
                    .DefaultIfEmpty(float.NegativeInfinity)
                    .Max();
                if (worst >= bestAlive)
                {
                    break;
                }
            }
        }

        foreach (GroupState gs in states)
        {
            foreach (Beam b in gs.Alive)
            {
                if (b.Tokens.Length > 0)
                {
                    finished.Add(new Raw(gs.Group.Name, b.Tokens, b.LogProb, false));
                }
            }
        }

        BeamGroup[] judges = options.Groups.Where(g => g.Constraint is not null).ToArray();
        var scored = new List<GroupBeamHypothesis>(finished.Count);
        foreach (Raw r in finished)
        {
            var support = new Dictionary<string, float>(StringComparer.Ordinal);
            foreach (BeamGroup g in judges)
            {
                support[g.Name] = g.Constraint!.Support(r.Tokens);
            }
            float alignment = judges.Length == 0
                ? 0f
                : options.Alignment == AlignmentMode.Min ? support.Values.Min() : support.Values.Sum();
            float score = LlmBeamSearch.Normalize(r.LogProb + (options.AlignmentWeight * alignment), r.Tokens.Length, options.LengthPenalty);
            scored.Add(new GroupBeamHypothesis(r.Group, new BeamHypothesis(r.Tokens, r.LogProb, score, r.Finished), support, alignment));
        }

        return scored
            .OrderByDescending(h => h.Hypothesis.Score)
            .ThenBy(h => h.Hypothesis.Tokens.Length)
            .GroupBy(h => string.Join(',', h.Hypothesis.Tokens))
            .Select(g => g.First())
            .Take(options.NumReturn)
            .ToArray();
    }

    private static void MaskInvalid(float[] lp, int? validVocab)
    {
        if (validVocab is int valid)
        {
            for (int i = Math.Max(valid, 0); i < lp.Length; i++)
            {
                lp[i] = float.NegativeInfinity;
            }
        }
    }

    private static List<Candidate> Select(List<Beam> beams, int count, int[] chosenCount, float diversity)
    {
        var all = new List<Candidate>(beams.Count * count);
        for (int b = 0; b < beams.Count; b++)
        {
            float[] lp = beams[b].LogProbs;
            var heap = new PriorityQueue<int, float>();
            for (int t = 0; t < lp.Length; t++)
            {
                float v = lp[t];
                if (float.IsNegativeInfinity(v) || float.IsNaN(v))
                {
                    continue;
                }
                float ranked = v - (diversity * chosenCount[t]);
                if (heap.Count < count)
                {
                    heap.Enqueue(t, ranked);
                }
                else if (heap.TryPeek(out _, out float min) && ranked > min)
                {
                    heap.DequeueEnqueue(t, ranked);
                }
            }
            while (heap.TryDequeue(out int token, out float rankedValue))
            {
                all.Add(new Candidate(b, token, beams[b].LogProb + lp[token], beams[b].LogProb + rankedValue));
            }
        }

        all.Sort(static (x, y) =>
        {
            int c = y.Rank.CompareTo(x.Rank);
            if (c != 0)
            {
                return c;
            }
            c = x.Token.CompareTo(y.Token);
            return c != 0 ? c : x.Beam.CompareTo(y.Beam);
        });
        return all.Count > count ? all.GetRange(0, count) : all;
    }

    private static int[] Append(int[] tokens, int token)
    {
        var result = new int[tokens.Length + 1];
        tokens.CopyTo(result, 0);
        result[^1] = token;
        return result;
    }

    private readonly record struct Candidate(int Beam, int Token, float Total, float Rank);

    private sealed record Beam(int[] Tokens, float LogProb, float[] LogProbs, object? State);

    private sealed record Raw(string Group, int[] Tokens, float LogProb, bool Finished);

    private sealed class GroupState(BeamGroup group, List<Beam> alive)
    {
        public BeamGroup Group { get; } = group;

        public List<Beam> Alive { get; set; } = alive;
    }
}
