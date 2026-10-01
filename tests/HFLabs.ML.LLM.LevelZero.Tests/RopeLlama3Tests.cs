using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class RopeLlama3Tests
{
    // Reference values from the transformers formula for Llama-3.2-1B (head_dim 64, theta 5e5, factor 32,
    // low_freq_factor 1, high_freq_factor 4, original_max_position_embeddings 8192).
    private static readonly float[] Expected =
    [
        1.000000000e+00f, 6.636012377e-01f, 4.403666027e-01f, 2.922278226e-01f, 1.939227447e-01f, 1.286873734e-01f,
        8.539710029e-02f, 5.666962145e-02f, 3.760603093e-02f, 2.495540867e-02f, 1.656044008e-02f, 1.098952853e-02f,
        7.292664737e-03f, 4.839421346e-03f, 3.211445995e-03f, 1.290547928e-03f, 4.295567966e-04f, 9.708287803e-05f,
        1.946163818e-05f, 1.291476719e-05f, 8.570255490e-06f, 5.687232150e-06f, 3.774054294e-06f, 2.504467101e-06f,
        1.661967468e-06f, 1.102883669e-06f, 7.318749675e-07f, 4.856731343e-07f, 3.222932930e-07f, 2.138742282e-07f,
        1.419272025e-07f, 9.418306725e-08f,
    ];

    [Fact]
    public void Llama3Scaling_MatchesReference()
    {
        var scaling = new HfRopeScaling
        {
            Type = "llama3",
            Factor = 32.0,
            LowFreqFactor = 1.0,
            HighFreqFactor = 4.0,
            OriginalMaxPositionEmbeddings = 8192,
        };
        (float[] freqs, float attention) = RopeFrequencies.Compute(64, 500000.0, 131072, scaling);
        Assert.Equal(1f, attention);
        Assert.Equal(Expected.Length, freqs.Length);
        for (int i = 0; i < Expected.Length; i++)
        {
            Assert.InRange(Math.Abs(freqs[i] - Expected[i]) / Expected[i], 0.0, 1e-5);
        }
    }

    [Fact]
    public void Llama31Config_IsNowSupported()
    {
        const string json = """
            {"model_type":"llama","hidden_size":2048,"num_hidden_layers":16,"num_attention_heads":32,"num_key_value_heads":8,
             "intermediate_size":8192,"vocab_size":128256,"max_position_embeddings":131072,"rms_norm_eps":1e-05,
             "rope_theta":500000.0,"hidden_act":"silu","tie_word_embeddings":true,"head_dim":64,"eos_token_id":[128001,128008,128009],
             "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,"original_max_position_embeddings":8192,"rope_type":"llama3"}}
            """;
        HfModelConfig config = HfModelConfig.Parse(json);
        config.EnsureSupported();
        Assert.Equal("llama3", config.RopeScalingType);
        Assert.Equal("llama3.1", ChatTemplates.ForConfig(config).Name);
    }
}
