using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Config parsing, tensor inventory and weight-splitting for the hybrid Qwen3.5 architecture.</summary>
public sealed class Qwen35ConfigTests
{
    // Layout of Qwen3.5-0.8B: nested text_config, 24 layers (3 linear : 1 full), gated attention, partial rotary.
    private const string Qwen35Json = """
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","attn_output_gate":true,"head_dim":256,"hidden_size":1024,
          "intermediate_size":3584,"linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
          "linear_num_value_heads":16,"linear_value_head_dim":128,"num_attention_heads":8,"num_hidden_layers":8,
          "num_key_value_heads":2,"rms_norm_eps":1e-06,"tie_word_embeddings":true,"vocab_size":248320,
          "eos_token_id":248044,
          "layer_types":["linear_attention","linear_attention","linear_attention","full_attention",
                         "linear_attention","linear_attention","linear_attention","full_attention"],
          "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
        """;

    [Fact]
    public void Parse_ReadsNestedTextConfigAndRopeParameters()
    {
        HfModelConfig c = HfModelConfig.Parse(Qwen35Json);
        Assert.Equal("model.language_model.", c.TensorPrefix);
        Assert.Equal(1024, c.HiddenSize);
        Assert.Equal(8, c.NumLayers);
        Assert.Equal(256, c.HeadDim);
        Assert.Equal(10_000_000.0, c.RopeTheta);
        Assert.Equal(0.25, c.PartialRotaryFactor);
        Assert.Equal(64, c.RotaryDim);
        Assert.True(c.AttentionOutputGate);
        Assert.True(c.QkNorm);
        Assert.Equal(1, c.RmsNormOffset);
        Assert.True(c.HasLinearLayers);
        Assert.True(c.NeedsRopeTable);
        c.EnsureSupported();
    }

    [Fact]
    public void LinearDimensions_AreDerivedFromHeadCounts()
    {
        HfModelConfig c = HfModelConfig.Parse(Qwen35Json);
        Assert.Equal(2048, c.LinearKeyDim);
        Assert.Equal(2048, c.LinearValueDim);
        Assert.Equal((2 * 2048) + 2048, c.LinearConvDim);
        Assert.Equal(4, c.LinearConvKernel);
    }

    [Fact]
    public void IsLinearLayer_FollowsLayerTypes()
    {
        HfModelConfig c = HfModelConfig.Parse(Qwen35Json);
        bool[] expected = [true, true, true, false, true, true, true, false];
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], c.IsLinearLayer(i));
        }

        Assert.False(c.IsLinearLayer(99));
    }

    [Fact]
    public void PlainQwen25_HasNoLinearLayersAndZeroRmsOffset()
    {
        HfModelConfig c = HfModelConfig.Parse(HfModelConfigTests.Qwen25Half);
        Assert.False(c.HasLinearLayers);
        Assert.False(c.AttentionOutputGate);
        Assert.Equal(0, c.RmsNormOffset);
        Assert.Equal("model.", c.TensorPrefix);
        Assert.Equal(c.HeadDim, c.RotaryDim);
    }

    [Fact]
    public void ExpectedTensors_UseNestedPrefixAndMixerSpecificShapes()
    {
        HfModelConfig c = HfModelConfig.Parse(Qwen35Json);
        IReadOnlyList<(string Name, long[] Shape)> t = c.ExpectedTensors();
        Assert.Contains(t, x => x.Name == "model.language_model.embed_tokens.weight" && x.Shape.SequenceEqual(new long[] { 248320, 1024 }));
        Assert.Contains(t, x => x.Name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" && x.Shape.SequenceEqual(new long[] { 6144, 1024 }));
        Assert.Contains(t, x => x.Name == "model.language_model.layers.0.linear_attn.conv1d.weight" && x.Shape.SequenceEqual(new long[] { 6144, 1, 4 }));
        Assert.Contains(t, x => x.Name == "model.language_model.layers.0.linear_attn.A_log" && x.Shape.SequenceEqual(new long[] { 16 }));
        Assert.DoesNotContain(t, x => x.Name.StartsWith("model.language_model.layers.0.self_attn", StringComparison.Ordinal));

        // Gated full attention: q_proj carries query and gate rows.
        Assert.Contains(t, x => x.Name == "model.language_model.layers.3.self_attn.q_proj.weight" && x.Shape.SequenceEqual(new long[] { 2 * 8 * 256, 1024 }));
        Assert.Contains(t, x => x.Name == "model.language_model.layers.3.self_attn.o_proj.weight" && x.Shape.SequenceEqual(new long[] { 1024, 8 * 256 }));
        Assert.Contains(t, x => x.Name == "model.language_model.layers.3.self_attn.q_norm.weight");
        Assert.DoesNotContain(t, x => x.Name.StartsWith("model.language_model.layers.3.linear_attn", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("\"layer_types\":[\"linear_attention\"],", "layer_types has")]
    [InlineData("\"linear_num_value_heads\":8,", "different number")]
    [InlineData("\"linear_value_head_dim\":96,", "power of two")]
    [InlineData("\"linear_conv_kernel_dim\":9,", "2 to 8")]
    public void EnsureSupported_RejectsUnsupportedLinearLayouts(string inject, string messagePart)
    {
        // Duplicate keys: System.Text.Json keeps the last, so the override goes at the end of text_config.
        string json = Qwen35Json.Replace("\"rope_parameters\"", inject + "\"rope_parameters\"", StringComparison.Ordinal);
        HfModelConfig c = HfModelConfig.Parse(json);
        Exception ex = Assert.ThrowsAny<Exception>(c.EnsureSupported);
        Assert.Contains(messagePart, ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SplitQueryAndGate_DeinterleavesPerHeadBlocks()
    {
        const int heads = 3;
        const int headDim = 2;
        const int hidden = 4;
        const int block = headDim * hidden;

        // HF layout per head: [query rows (headDim), gate rows (headDim)], each row hidden wide.
        var hf = new float[heads * 2 * block];
        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < block; i++)
            {
                hf[(2 * h * block) + i] = 100 + (h * 10) + i; // query
                hf[(((2 * h) + 1) * block) + i] = 200 + (h * 10) + i; // gate
            }
        }

        (float[] q, float[] g) = LlmLayerWeights.SplitQueryAndGate(hf, heads, headDim, hidden);
        Assert.Equal(heads * block, q.Length);
        Assert.Equal(heads * block, g.Length);
        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < block; i++)
            {
                Assert.Equal(100 + (h * 10) + i, q[(h * block) + i]);
                Assert.Equal(200 + (h * 10) + i, g[(h * block) + i]);
            }
        }
    }

    [Fact]
    public void SplitQueryAndGate_RejectsWrongLength()
    {
        Assert.Throws<ArgumentException>(() => LlmLayerWeights.SplitQueryAndGate(new float[10], 2, 2, 2));
    }
}

