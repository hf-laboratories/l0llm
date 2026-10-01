using System.Text.Json;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>
/// Checks the CPU-side RoPE frequency tables (default, linear, YaRN) against the tables that Hugging Face
/// <c>transformers</c> produces (<c>golden/gen_rope_golden.py</c>), plus config parsing of <c>rope_scaling</c>.
/// </summary>
public sealed class RopeFrequenciesTests
{
    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "rope.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });

    [Theory]
    [InlineData("default")]
    [InlineData("linear4")]
    [InlineData("yarn4_qwen3")]
    [InlineData("yarn8_small")]
    [InlineData("yarn2_explicit")]
    public void Tables_MatchTransformers(string name)
    {
        JsonElement c = Golden.Value.GetProperty("cases").GetProperty(name);
        int headDim = c.GetProperty("head_dim").GetInt32();
        double theta = c.GetProperty("theta").GetDouble();
        int maxPos = c.GetProperty("max_position_embeddings").GetInt32();
        HfRopeScaling? scaling = c.GetProperty("scaling") is { ValueKind: JsonValueKind.Object } s ? ParseScaling(s) : null;
        double[] expected = c.GetProperty("inv_freq").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        double expectedAttention = c.GetProperty("attention_factor").GetDouble();

        (float[] actual, float attention) = RopeFrequencies.Compute(headDim, theta, maxPos, scaling);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(expected[i] - actual[i]) <= 1e-5 * Math.Abs(expected[i]) + 1e-12,
                $"{name}: inv_freq[{i}] expected {expected[i]:R}, got {actual[i]:R}");
        }

        Assert.Equal(expectedAttention, attention, 5);
    }

    [Fact]
    public void Linear_DividesEveryFrequencyByFactor()
    {
        (float[] plain, _) = RopeFrequencies.Compute(16, 10000, 2048, null);
        (float[] linear, float attention) = RopeFrequencies.Compute(
            16, 10000, 2048, new HfRopeScaling { Type = "linear", Factor = 4 });
        Assert.Equal(1f, attention);
        for (int i = 0; i < plain.Length; i++)
        {
            Assert.Equal(plain[i] / 4, linear[i], 6);
        }
    }

    [Fact]
    public void Yarn_FactorOne_IsUnscaled()
    {
        (float[] plain, _) = RopeFrequencies.Compute(16, 10000, 2048, null);
        (float[] yarn, float attention) = RopeFrequencies.Compute(
            16, 10000, 2048, new HfRopeScaling { Type = "yarn", Factor = 1 });
        Assert.Equal(1f, attention);
        Assert.Equal(plain, yarn);
    }

    [Fact]
    public void UnknownType_Throws()
    {
        Assert.Throws<NotSupportedException>(
            () => RopeFrequencies.Compute(16, 10000, 2048, new HfRopeScaling { Type = "longrope" }));
    }

    [Fact]
    public void Config_ParsesScalingAndIsScaledOnlyForRealScaling()
    {
        HfModelConfig yarn = HfModelConfig.Parse(ConfigJson(
            """{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"beta_fast":16}"""));
        Assert.Equal("yarn", yarn.RopeScaling!.Type);
        Assert.Equal(4.0, yarn.RopeScaling.Factor);
        Assert.Equal(32768, yarn.RopeScaling.OriginalMaxPositionEmbeddings);
        Assert.Equal(16.0, yarn.RopeScaling.BetaFast);
        Assert.Equal(1.0, yarn.RopeScaling.BetaSlow);
        Assert.True(RopeFrequencies.IsScaled(yarn));
        yarn.EnsureSupported();

        Assert.False(RopeFrequencies.IsScaled(HfModelConfig.Parse(ConfigJson("null"))));
        Assert.False(RopeFrequencies.IsScaled(HfModelConfig.Parse(ConfigJson("""{"rope_type":"default"}"""))));
    }

    [Fact]
    public void Config_RejectsNonPositiveFactorAndUnknownType()
    {
        Assert.Throws<InvalidDataException>(
            HfModelConfig.Parse(ConfigJson("""{"type":"linear","factor":0}""")).EnsureSupported);
        Assert.Throws<NotSupportedException>(
            HfModelConfig.Parse(ConfigJson("""{"rope_type":"dynamic","factor":8}""")).EnsureSupported);
    }

    private static HfRopeScaling ParseScaling(JsonElement s) => new()
    {
        Type = s.GetProperty("rope_type").GetString()!,
        Factor = s.GetProperty("factor").GetDouble(),
        OriginalMaxPositionEmbeddings = s.TryGetProperty("original_max_position_embeddings", out JsonElement o) ? o.GetInt32() : null,
        BetaFast = s.TryGetProperty("beta_fast", out JsonElement f) ? f.GetDouble() : 32.0,
        BetaSlow = s.TryGetProperty("beta_slow", out JsonElement w) ? w.GetDouble() : 1.0,
        AttentionFactor = s.TryGetProperty("attention_factor", out JsonElement a) ? a.GetDouble() : null,
    };

    private static string ConfigJson(string ropeScaling) =>
        $$"""
        {
          "model_type": "qwen2", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
          "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,
          "max_position_embeddings": 4096, "rope_scaling": {{ropeScaling}}
        }
        """;
}
