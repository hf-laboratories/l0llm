using Xunit;
using static HFLabs.ML.LLM.LevelZero.Tests.SafetensorsTestFiles;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class HfModelConfigTests
{
    internal const string Qwen25Half = """
        {
          "architectures": ["Qwen2ForCausalLM"],
          "attention_dropout": 0.0,
          "bos_token_id": 151643,
          "eos_token_id": 151643,
          "hidden_act": "silu",
          "hidden_size": 896,
          "initializer_range": 0.02,
          "intermediate_size": 4864,
          "max_position_embeddings": 32768,
          "max_window_layers": 24,
          "model_type": "qwen2",
          "num_attention_heads": 14,
          "num_hidden_layers": 24,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-06,
          "rope_theta": 1000000.0,
          "sliding_window": 32768,
          "tie_word_embeddings": true,
          "torch_dtype": "bfloat16",
          "use_sliding_window": false,
          "vocab_size": 151936
        }
        """;

    [Fact]
    public void Qwen25_ParsesAllFieldsAndDerivedValues()
    {
        HfModelConfig c = HfModelConfig.Parse(Qwen25Half);

        Assert.Equal("qwen2", c.ModelType);
        Assert.Equal(896, c.HiddenSize);
        Assert.Equal(4864, c.IntermediateSize);
        Assert.Equal(24, c.NumLayers);
        Assert.Equal(14, c.NumAttentionHeads);
        Assert.Equal(2, c.NumKeyValueHeads);
        Assert.Equal(64, c.HeadDim);
        Assert.Equal(151936, c.VocabSize);
        Assert.Equal(32768, c.MaxPositionEmbeddings);
        Assert.Equal(1e-6f, c.RmsNormEps);
        Assert.Equal(1_000_000.0, c.RopeTheta);
        Assert.True(c.TieWordEmbeddings);
        Assert.Equal("silu", c.HiddenAct);
        Assert.Equal("bfloat16", c.TorchDtype);
        Assert.True(c.AttentionBias);
        Assert.False(c.QkNorm);
        Assert.Null(c.RopeScalingType);
        Assert.Null(c.SlidingWindow); // present in the file but use_sliding_window is false
        Assert.Equal(151643, c.BosTokenId);
        Assert.Equal(new[] { 151643 }, c.EosTokenIds);
        Assert.Equal(7, c.GroupSize);
        Assert.Equal(896, c.QueryDim);
        Assert.Equal(128, c.KeyValueDim);
        c.EnsureSupported();
    }

    [Fact]
    public void Missing_KvHeads_DefaultsToAttentionHeads()
    {
        HfModelConfig c = HfModelConfig.Parse(Json(("num_key_value_heads", null)));

        Assert.Equal(c.NumAttentionHeads, c.NumKeyValueHeads);
        Assert.Equal(1, c.GroupSize);
    }

    [Fact]
    public void ExplicitHeadDim_OverridesHiddenOverHeads()
    {
        HfModelConfig c = HfModelConfig.Parse(Json(("head_dim", "128")));

        Assert.Equal(128, c.HeadDim);
        Assert.Equal(14 * 128, c.QueryDim);
    }

    [Fact]
    public void EosTokenArray_IsParsed()
    {
        HfModelConfig c = HfModelConfig.Parse(Json(("eos_token_id", "[151645, 151643]")));

        Assert.Equal(new[] { 151645, 151643 }, c.EosTokenIds);
    }

    [Fact]
    public void Qwen3_HasQkNormAndNoImplicitBias()
    {
        HfModelConfig c = HfModelConfig.Parse(Json(("model_type", "\"qwen3\""), ("head_dim", "128")));

        Assert.True(c.QkNorm);
        Assert.False(c.AttentionBias);
    }

    [Fact]
    public void Llama_AttentionBiasComesFromConfig()
    {
        Assert.False(HfModelConfig.Parse(Json(("model_type", "\"llama\""))).AttentionBias);
        Assert.True(HfModelConfig.Parse(Json(("model_type", "\"llama\""), ("attention_bias", "true"))).AttentionBias);
    }

    [Fact]
    public void RopeScaling_UsesRopeTypeThenType()
    {
        Assert.Equal(
            "yarn",
            HfModelConfig.Parse(Json(("rope_scaling", """{"rope_type":"yarn","factor":4.0}"""))).RopeScalingType);
        Assert.Equal(
            "linear",
            HfModelConfig.Parse(Json(("rope_scaling", """{"type":"linear","factor":2.0}"""))).RopeScalingType);
    }

    [Theory]
    [InlineData("model_type")]
    [InlineData("hidden_size")]
    [InlineData("num_attention_heads")]
    [InlineData("num_hidden_layers")]
    [InlineData("intermediate_size")]
    [InlineData("vocab_size")]
    public void MissingRequiredField_Throws(string field)
    {
        Assert.Throws<InvalidDataException>(() => HfModelConfig.Parse(Json((field, null))));
    }

    [Fact]
    public void InvalidJson_Throws()
    {
        Assert.Throws<InvalidDataException>(() => HfModelConfig.Parse("{ nope"));
        Assert.Throws<InvalidDataException>(() => HfModelConfig.Parse("[1,2]"));
    }

    [Theory]
    [InlineData("model_type", "\"gpt2\"")]
    [InlineData("hidden_act", "\"gelu\"")]
    [InlineData("rope_scaling", """{"rope_type":"dynamic","factor":8.0}""")]
    [InlineData("use_sliding_window", "true")]
    public void EnsureSupported_RejectsUnimplementedFeatures(string field, string value)
    {
        HfModelConfig c = HfModelConfig.Parse(Json((field, value)));

        Assert.Throws<NotSupportedException>(c.EnsureSupported);
    }

    [Fact]
    public void EnsureSupported_RejectsKvHeadsNotDividingHeads()
    {
        HfModelConfig c = HfModelConfig.Parse(Json(("num_key_value_heads", "4")));

        Assert.Throws<InvalidDataException>(c.EnsureSupported);
    }

    [Fact]
    public void EnsureSupported_RejectsOddHeadDim()
    {
        HfModelConfig c = HfModelConfig.Parse(Json(("head_dim", "63")));

        Assert.Throws<InvalidDataException>(c.EnsureSupported);
    }

    [Fact]
    public void ExpectedTensors_Qwen25_Has290TensorsWithBiasesAndNoLmHead()
    {
        HfModelConfig c = HfModelConfig.Parse(Qwen25Half);

        IReadOnlyList<(string Name, long[] Shape)> tensors = c.ExpectedTensors();

        Assert.Equal(2 + (24 * 12), tensors.Count);
        Assert.DoesNotContain(tensors, t => t.Name == "lm_head.weight");
        Assert.Contains(tensors, t => t.Name == "model.layers.23.self_attn.k_proj.bias" && t.Shape.SequenceEqual(new long[] { 128 }));
        Assert.Contains(tensors, t => t.Name == "model.layers.0.self_attn.q_proj.weight" && t.Shape.SequenceEqual(new long[] { 896, 896 }));
        Assert.Contains(tensors, t => t.Name == "model.layers.0.mlp.down_proj.weight" && t.Shape.SequenceEqual(new long[] { 896, 4864 }));
        Assert.Contains(tensors, t => t.Name == "model.embed_tokens.weight" && t.Shape.SequenceEqual(new long[] { 151936, 896 }));
    }

    [Fact]
    public void ExpectedTensors_Untied_AddsLmHead_AndQwen3AddsQkNorm()
    {
        HfModelConfig c = HfModelConfig.Parse(
            Json(("model_type", "\"qwen3\""), ("head_dim", "128"), ("tie_word_embeddings", "false")));

        IReadOnlyList<(string Name, long[] Shape)> tensors = c.ExpectedTensors();

        Assert.Contains(tensors, t => t.Name == "lm_head.weight");
        Assert.Contains(tensors, t => t.Name == "model.layers.0.self_attn.q_norm.weight" && t.Shape.SequenceEqual(new long[] { 128 }));
        Assert.DoesNotContain(tensors, t => t.Name.EndsWith(".q_proj.bias", StringComparison.Ordinal));
    }

    [Fact]
    public void ValidateCheckpoint_CompleteTinyModel_HasNoProblems()
    {
        using var dir = new TempDir();
        HfModelConfig c = TinyConfig();
        Write(dir.File("model.safetensors"), TinyEntries(c, omit: null, wrongShapeFor: null));

        using var ckpt = SafetensorsCheckpoint.Open(dir.Path);

        Assert.Empty(c.ValidateCheckpoint(ckpt));
    }

    [Fact]
    public void ValidateCheckpoint_ReportsMissingAndMisshapenTensors()
    {
        using var dir = new TempDir();
        HfModelConfig c = TinyConfig();
        Write(
            dir.File("model.safetensors"),
            TinyEntries(c, omit: "model.layers.1.mlp.up_proj.weight", wrongShapeFor: "model.norm.weight"));

        using var ckpt = SafetensorsCheckpoint.Open(dir.Path);
        IReadOnlyList<string> problems = c.ValidateCheckpoint(ckpt);

        Assert.Equal(2, problems.Count);
        Assert.Contains(problems, p => p.Contains("missing tensor 'model.layers.1.mlp.up_proj.weight'", StringComparison.Ordinal));
        Assert.Contains(problems, p => p.Contains("'model.norm.weight'", StringComparison.Ordinal) && p.Contains("shape", StringComparison.Ordinal));
    }

    [Fact]
    public void Load_ReadsConfigJsonFromDirectoryOrFile()
    {
        using var dir = new TempDir();
        File.WriteAllText(dir.File("config.json"), Qwen25Half);

        Assert.Equal(896, HfModelConfig.Load(dir.Path).HiddenSize);
        Assert.Equal(896, HfModelConfig.Load(dir.File("config.json")).HiddenSize);
    }

    private static HfModelConfig TinyConfig() => HfModelConfig.Parse(
        """
        {"model_type":"qwen2","hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,
         "num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10,"tie_word_embeddings":true}
        """);

    private static List<Entry> TinyEntries(HfModelConfig c, string? omit, string? wrongShapeFor)
    {
        var entries = new List<Entry>();
        foreach ((string name, long[] shape) in c.ExpectedTensors())
        {
            if (name == omit)
            {
                continue;
            }

            long[] actual = name == wrongShapeFor ? [shape[0] + 1] : shape;
            long count = actual.Aggregate(1L, (a, d) => a * d);
            entries.Add(F32(name, actual, new float[count]));
        }

        return entries;
    }

    /// <summary>Qwen2.5-0.5B config with fields overridden; a null value removes the field.</summary>
    private static string Json(params (string Field, string? Value)[] overrides)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(Qwen25Half);
        var fields = new Dictionary<string, string>();
        foreach (System.Text.Json.JsonProperty p in doc.RootElement.EnumerateObject())
        {
            fields[p.Name] = p.Value.GetRawText();
        }

        foreach ((string field, string? value) in overrides)
        {
            if (value is null)
            {
                fields.Remove(field);
            }
            else
            {
                fields[field] = value;
            }
        }

        return "{" + string.Join(",", fields.Select(kv => $"\"{kv.Key}\":{kv.Value}")) + "}";
    }
}
