namespace HFLabs.ML.LLM.LevelZero;

/// <summary>A sequence model that beam search can drive: one live sequence that can be captured and restored.</summary>
public interface IBeamSequenceModel
{
    /// <summary>Number of logits returned per step.</summary>
    int VocabSize { get; }

    /// <summary>Resets the sequence, feeds the prompt and returns the logits for the first new token.</summary>
    float[] Prefill(IReadOnlyList<int> promptIds);

    /// <summary>Feeds one token and returns the logits for the next one.</summary>
    float[] Decode(int tokenId);

    /// <summary>Captures everything that changed since <see cref="Prefill"/>.</summary>
    object CaptureState();

    /// <summary>Returns the sequence to a state produced by <see cref="CaptureState"/> (same prompt).</summary>
    void RestoreState(object state);
}

/// <summary>
/// Adjusts the next-token log-probabilities of one beam. This is the hook a graph (or any other knowledge source)
/// uses to allow, forbid or favour tokens while the beams grow.
/// </summary>
public interface IBeamConstraint
{
    /// <summary>
    /// Edits <paramref name="logProbs"/> in place before candidates are chosen. Set an entry to
    /// <see cref="float.NegativeInfinity"/> to forbid the token; add a positive bonus or negative penalty to steer.
    /// The values are not renormalised.
    /// </summary>
    /// <param name="generated">The tokens this beam has generated so far (without the prompt).</param>
    /// <param name="logProbs">Log-softmax of the model's logits, one per vocabulary entry.</param>
    void Adjust(IReadOnlyList<int> generated, Span<float> logProbs);

    /// <summary>Score added to a finished (or truncated) hypothesis before length normalisation; e.g. a grounding penalty.</summary>
    float FinalAdjust(IReadOnlyList<int> generated) => 0f;
}

/// <summary>Options for <see cref="LlmBeamSearch"/>.</summary>
public sealed record BeamSearchOptions
{
    /// <summary>Number of beams kept alive.</summary>
    public int BeamWidth { get; init; } = 4;

    /// <summary>Maximum number of generated tokens per hypothesis.</summary>
    public int MaxNewTokens { get; init; } = 64;

    /// <summary>Score is <c>logProb / length^LengthPenalty</c>; 0 disables normalisation, above 0 favours longer output.</summary>
    public float LengthPenalty { get; init; } = 1.0f;

    /// <summary>Tokens that end a hypothesis; the stop token is included in the result.</summary>
    public IReadOnlyCollection<int>? StopTokenIds { get; init; }

    /// <summary>How many hypotheses to return, best first (at most <see cref="BeamWidth"/>).</summary>
    public int NumReturn { get; init; } = 1;

    /// <summary>Optional per-step log-probability adjustment.</summary>
    public IBeamConstraint? Constraint { get; init; }

    /// <summary>Token ids at or above this value are never chosen (checkpoints pad the vocabulary past the tokenizer's). Null allows all.</summary>
    public int? ValidVocabSize { get; init; }
}

/// <summary>One beam-search result.</summary>
/// <param name="Tokens">Generated token ids (without the prompt).</param>
/// <param name="LogProb">Sum of the (adjusted) log-probabilities of the tokens.</param>
/// <param name="Score">Length-normalised score used for ranking (higher is better).</param>
/// <param name="Finished">True when it ended on a stop token; false when it was cut by the length limit.</param>
public sealed record BeamHypothesis(int[] Tokens, float LogProb, float Score, bool Finished);

/// <summary>A decoded beam-search result.</summary>
/// <param name="Text">The generated text (special tokens skipped).</param>
/// <param name="Hypothesis">The token-level result with its scores.</param>
public sealed record BeamCompletion(string Text, BeamHypothesis Hypothesis);

/// <summary>Beam search over a sequence model that can fork its state.</summary>
/// <remarks>
/// Each beam keeps a host copy of its own KV rows and linear state (<see cref="LlmSequenceState"/>); a step restores
/// the parent, decodes the chosen token and captures the child. Beams are decoded one after another, so a step costs
/// <c>BeamWidth</c> decodes.
/// </remarks>
public static class LlmBeamSearch
{
    /// <summary>Runs beam search on a model resident on the GPU.</summary>
    public static IReadOnlyList<BeamHypothesis> Run(LlmModel model, IReadOnlyList<int> promptIds, BeamSearchOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        return Run(new LlmBeamModel(model), promptIds, options);
    }

    /// <summary>Runs beam search on any <see cref="IBeamSequenceModel"/>.</summary>
    public static IReadOnlyList<BeamHypothesis> Run(IBeamSequenceModel model, IReadOnlyList<int> promptIds, BeamSearchOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.BeamWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxNewTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.NumReturn);
        ArgumentOutOfRangeException.ThrowIfNegative(options.LengthPenalty);
        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt must contain at least one token.", nameof(promptIds));
        }

        int width = options.BeamWidth;
        int numReturn = Math.Min(options.NumReturn, width);
        var stops = new HashSet<int>(options.StopTokenIds ?? []);
        IBeamConstraint? constraint = options.Constraint;

        float[] first = model.Prefill(promptIds);
        var alive = new List<Beam>
        {
            new([], 0f, Prepare(first, [], options), model.CaptureState()),
        };
        var finished = new List<BeamHypothesis>();

        for (int step = 0; step < options.MaxNewTokens; step++)
        {
            List<Candidate> candidates = SelectCandidates(alive, 2 * width);
            var next = new List<Candidate>(width);
            for (int rank = 0; rank < candidates.Count; rank++)
            {
                Candidate c = candidates[rank];
                if (stops.Contains(c.Token))
                {
                    // Only a stop that ranks among the best `width` candidates counts as a finished hypothesis.
                    if (rank < width)
                    {
                        int[] tokens = Append(alive[c.Beam].Tokens, c.Token);
                        finished.Add(Hypothesis(tokens, c.Total, true, options));
                    }
                }
                else if (next.Count < width)
                {
                    next.Add(c);
                }
            }

            if (next.Count == 0)
            {
                alive.Clear();
                break;
            }

            bool last = step == options.MaxNewTokens - 1;
            var grown = new List<Beam>(next.Count);
            foreach (Candidate c in next)
            {
                Beam parent = alive[c.Beam];
                int[] tokens = Append(parent.Tokens, c.Token);
                if (last)
                {
                    grown.Add(new Beam(tokens, c.Total, [], null));
                    continue;
                }
                model.RestoreState(parent.State!);
                float[] logits = model.Decode(c.Token);
                grown.Add(new Beam(tokens, c.Total, Prepare(logits, tokens, options), model.CaptureState()));
            }
            alive = grown;

            if (finished.Count >= width && !last)
            {
                float worst = finished.Min(h => h.Score);
                float bestAlive = alive.Max(b => Normalize(b.LogProb + (constraint?.FinalAdjust(b.Tokens) ?? 0f), b.Tokens.Length, options.LengthPenalty));
                if (worst >= bestAlive)
                {
                    break;
                }
            }
        }

        // Beams cut by the length limit compete with the finished ones.
        foreach (Beam b in alive)
        {
            if (b.Tokens.Length > 0)
            {
                finished.Add(Hypothesis(b.Tokens, b.LogProb, false, options));
            }
        }

        return finished
            .OrderByDescending(h => h.Score)
            .ThenBy(h => h.Tokens.Length)
            .Take(numReturn)
            .ToArray();
    }

    private static List<Candidate> SelectCandidates(List<Beam> beams, int count)
    {
        var all = new List<Candidate>(beams.Count * count);
        for (int b = 0; b < beams.Count; b++)
        {
            float[] lp = beams[b].LogProbs;
            // A min-heap keeps the `count` best tokens of this beam.
            var heap = new PriorityQueue<int, float>();
            for (int t = 0; t < lp.Length; t++)
            {
                float v = lp[t];
                if (float.IsNegativeInfinity(v) || float.IsNaN(v))
                {
                    continue;
                }
                if (heap.Count < count)
                {
                    heap.Enqueue(t, v);
                }
                else if (heap.TryPeek(out _, out float min) && v > min)
                {
                    heap.DequeueEnqueue(t, v);
                }
            }
            while (heap.TryDequeue(out int token, out float value))
            {
                all.Add(new Candidate(b, token, beams[b].LogProb + value));
            }
        }

        all.Sort(static (x, y) =>
        {
            int c = y.Total.CompareTo(x.Total);
            if (c != 0)
            {
                return c;
            }
            c = x.Token.CompareTo(y.Token);
            return c != 0 ? c : x.Beam.CompareTo(y.Beam);
        });
        return all.Count > count ? all.GetRange(0, count) : all;
    }

    /// <summary>Log-softmax in double precision.</summary>
    internal static float[] LogSoftmax(float[] logits)
    {
        var lp = new float[logits.Length];
        double max = double.NegativeInfinity;
        foreach (float v in logits)
        {
            if (v > max)
            {
                max = v;
            }
        }
        double sum = 0;
        foreach (float v in logits)
        {
            sum += Math.Exp(v - max);
        }
        double log = max + Math.Log(sum);
        for (int i = 0; i < lp.Length; i++)
        {
            lp[i] = (float)(logits[i] - log);
        }
        return lp;
    }

    internal static float Normalize(float logProb, int length, float penalty) =>
        penalty == 0f ? logProb : logProb / MathF.Pow(Math.Max(length, 1), penalty);

    private static float[] Prepare(float[] logits, IReadOnlyList<int> generated, BeamSearchOptions options)
    {
        IBeamConstraint? constraint = options.Constraint;
        var lp = new float[logits.Length];
        double max = double.NegativeInfinity;
        foreach (float v in logits)
        {
            if (v > max)
            {
                max = v;
            }
        }
        double sum = 0;
        foreach (float v in logits)
        {
            sum += Math.Exp(v - max);
        }
        double log = max + Math.Log(sum);
        for (int i = 0; i < lp.Length; i++)
        {
            lp[i] = (float)(logits[i] - log);
        }
        if (options.ValidVocabSize is int valid)
        {
            for (int i = Math.Max(valid, 0); i < lp.Length; i++)
            {
                lp[i] = float.NegativeInfinity;
            }
        }
        constraint?.Adjust(generated, lp);
        return lp;
    }

    private static BeamHypothesis Hypothesis(int[] tokens, float logProb, bool finished, BeamSearchOptions options)
    {
        float total = logProb + (options.Constraint?.FinalAdjust(tokens) ?? 0f);
        return new BeamHypothesis(tokens, logProb, Normalize(total, tokens.Length, options.LengthPenalty), finished);
    }

    private static int[] Append(int[] tokens, int token)
    {
        var result = new int[tokens.Length + 1];
        tokens.CopyTo(result, 0);
        result[^1] = token;
        return result;
    }

    private readonly record struct Candidate(int Beam, int Token, float Total);

    private sealed record Beam(int[] Tokens, float LogProb, float[] LogProbs, object? State);
}

/// <summary>Adapts <see cref="LlmModel"/> to <see cref="IBeamSequenceModel"/>.</summary>
internal sealed class LlmBeamModel(LlmModel model) : IBeamSequenceModel
{
    private int _basePosition;

    public int VocabSize => model.Config.VocabSize;

    public float[] Prefill(IReadOnlyList<int> promptIds)
    {
        float[] logits = model.Prefill(promptIds);
        _basePosition = model.Position;
        return logits;
    }

    public float[] Decode(int tokenId) => model.Decode(tokenId);

    public object CaptureState() => model.CaptureState(_basePosition);

    public void RestoreState(object state) => model.RestoreState((LlmSequenceState)state);
}
