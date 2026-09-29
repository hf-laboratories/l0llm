using System.Globalization;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>The <c>rope_scaling</c> block of a Hugging Face <c>config.json</c>.</summary>
public sealed record HfRopeScaling
{
    /// <summary>Scaling type: <c>default</c>, <c>linear</c> or <c>yarn</c> are supported.</summary>
    public required string Type { get; init; }

    /// <summary>Context extension factor.</summary>
    public double Factor { get; init; } = 1.0;

    /// <summary>Context length the base model was trained with (YaRN); falls back to <c>max_position_embeddings</c>.</summary>
    public int? OriginalMaxPositionEmbeddings { get; init; }

    /// <summary>YaRN: rotation count above which frequencies are left alone.</summary>
    public double BetaFast { get; init; } = 32.0;

    /// <summary>YaRN: rotation count below which frequencies are fully interpolated.</summary>
    public double BetaSlow { get; init; } = 1.0;

    /// <summary>YaRN <c>mscale</c>, when configured.</summary>
    public double? Mscale { get; init; }

    /// <summary>YaRN <c>mscale_all_dim</c>, when configured.</summary>
    public double? MscaleAllDim { get; init; }

    /// <summary>YaRN attention factor override, when configured.</summary>
    public double? AttentionFactor { get; init; }

    /// <summary>YaRN: round the correction range to whole dimensions (the transformers default).</summary>
    public bool Truncate { get; init; } = true;
}

/// <summary>
/// Inverse-frequency table for rotary embeddings, including the scaled variants. Mirrors the
/// <c>ROPE_INIT_FUNCTIONS</c> of Hugging Face <c>transformers</c> (default, linear, yarn).
/// </summary>
public static class RopeFrequencies
{
    /// <summary>True when <paramref name="config"/> needs the table-driven kernel rather than the built-in one.</summary>
    public static bool IsScaled(HfModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return config.RopeScaling is { } s && !string.Equals(s.Type, "default", StringComparison.Ordinal);
    }

    /// <summary>
    /// Computes one inverse frequency per rotated pair (<c>rotaryDim / 2</c>) and the factor that multiplies
    /// cos and sin.
    /// </summary>
    /// <exception cref="NotSupportedException">The scaling type is not implemented.</exception>
    public static (float[] InverseFrequencies, float AttentionFactor) Compute(HfModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return Compute(config.RotaryDim, config.RopeTheta, config.MaxPositionEmbeddings, config.RopeScaling);
    }

    /// <summary>Computes the table for explicit parameters.</summary>
    public static (float[] InverseFrequencies, float AttentionFactor) Compute(
        int headDim, double theta, int maxPositionEmbeddings, HfRopeScaling? scaling)
    {
        if (headDim <= 0 || headDim % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "head_dim must be positive and even.");
        }

        int half = headDim / 2;
        var baseFreq = new double[half];
        for (int i = 0; i < half; i++)
        {
            baseFreq[i] = 1.0 / Math.Pow(theta, 2.0 * i / headDim);
        }

        string type = scaling?.Type ?? "default";
        switch (type)
        {
            case "default":
                return (ToFloat(baseFreq), 1f);
            case "linear":
                {
                    double factor = scaling!.Factor;
                    var scaled = new double[half];
                    for (int i = 0; i < half; i++)
                    {
                        scaled[i] = baseFreq[i] / factor;
                    }

                    return (ToFloat(scaled), 1f);
                }

            case "yarn":
                return Yarn(headDim, theta, maxPositionEmbeddings, scaling!, baseFreq);
            default:
                throw new NotSupportedException(
                    string.Create(CultureInfo.InvariantCulture, $"rope_scaling '{type}' is not supported (default, linear and yarn are)."));
        }
    }

    private static (float[], float) Yarn(int headDim, double theta, int maxPositionEmbeddings, HfRopeScaling s, double[] extrapolation)
    {
        int half = headDim / 2;
        double factor = s.Factor;
        double original = s.OriginalMaxPositionEmbeddings ?? maxPositionEmbeddings;

        double attention;
        if (s.AttentionFactor is { } given)
        {
            attention = given;
        }
        else if (s.Mscale is { } mscale && s.MscaleAllDim is { } all)
        {
            attention = Mscale(factor, mscale) / Mscale(factor, all);
        }
        else
        {
            attention = Mscale(factor, 1.0);
        }

        double low = CorrectionDim(s.BetaFast, headDim, theta, original);
        double high = CorrectionDim(s.BetaSlow, headDim, theta, original);
        if (s.Truncate)
        {
            low = Math.Floor(low);
            high = Math.Ceiling(high);
        }

        low = Math.Max(low, 0);
        high = Math.Min(high, headDim - 1);
        if (low == high)
        {
            high += 0.001;
        }

        var result = new double[half];
        for (int i = 0; i < half; i++)
        {
            double ramp = Math.Clamp((i - low) / (high - low), 0.0, 1.0);
            double extrapolationFactor = 1.0 - ramp;
            double interpolated = extrapolation[i] / factor;
            result[i] = interpolated * (1.0 - extrapolationFactor) + extrapolation[i] * extrapolationFactor;
        }

        return (ToFloat(result), (float)attention);
    }

    private static double Mscale(double scale, double mscale) => scale <= 1.0 ? 1.0 : (0.1 * mscale * Math.Log(scale)) + 1.0;

    private static double CorrectionDim(double rotations, int dim, double theta, double maxPosition) =>
        dim * Math.Log(maxPosition / (rotations * 2 * Math.PI)) / (2 * Math.Log(theta));

    private static float[] ToFloat(double[] values)
    {
        var f = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            f[i] = (float)values[i];
        }

        return f;
    }
}
