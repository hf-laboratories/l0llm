using System.Globalization;
using System.Text.Json;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// The subset of a HuggingFace <c>config.json</c> that a decoder-only transformer needs to run,
/// with derived values (head dimension, bias and QK-norm flags) resolved.
/// </summary>
public sealed record HfModelConfig
{
    /// <summary>HuggingFace <c>model_type</c>, for example <c>qwen2</c>.</summary>
    public required string ModelType { get; init; }

    /// <summary>Residual stream width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>MLP inner width.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of decoder layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of query heads.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Number of key/value heads (equals query heads for plain multi-head attention).</summary>
    public required int NumKeyValueHeads { get; init; }

    /// <summary>Per-head width: <c>head_dim</c> if present, else <c>hidden_size / num_attention_heads</c>.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Vocabulary size (rows of the embedding matrix).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Longest context the model was trained for.</summary>
    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>Epsilon used by RMSNorm.</summary>
    public required float RmsNormEps { get; init; }

    /// <summary>Base frequency for rotary embeddings.</summary>
    public required double RopeTheta { get; init; }

    /// <summary>True when the output projection reuses the embedding matrix.</summary>
    public required bool TieWordEmbeddings { get; init; }

    /// <summary>Activation named by the config, for example <c>silu</c>.</summary>
    public required string HiddenAct { get; init; }

    /// <summary>Weight dtype the checkpoint was saved in, for example <c>bfloat16</c> (may be empty).</summary>
    public string TorchDtype { get; init; } = string.Empty;

    /// <summary>True when the Q, K and V projections carry a bias (always true for Qwen2).</summary>
    public required bool AttentionBias { get; init; }

    /// <summary>True when Q and K are RMS-normalised per head before RoPE (Qwen3).</summary>
    public required bool QkNorm { get; init; }

    /// <summary>The <c>rope_scaling</c> type when one is configured, else null.</summary>

    public string? RopeScalingType { get; init; }

    

    /// <summary>The parsed <c>rope_scaling</c> parameters, or null when the config has none.</summary>

    public HfRopeScaling? RopeScaling { get; init; }

    /// <summary>The sliding-window size when sliding-window attention is enabled, else null.</summary>
    public int? SlidingWindow { get; init; }

    /// <summary>Beginning-of-sequence token id, when configured.</summary>
    public int? BosTokenId { get; init; }

    /// <summary>End-of-sequence token ids (the config allows one or several).</summary>
    public IReadOnlyList<int> EosTokenIds { get; init; } = [];

    /// <summary>Prefix of the language-model tensors, <c>model.</c> or <c>model.language_model.</c> for multimodal checkpoints.</summary>
    public string TensorPrefix { get; init; } = "model.";
    /// <summary>Per-layer mixer type (<c>full_attention</c> or <c>linear_attention</c>); empty means every layer is full attention.</summary>
    public IReadOnlyList<string> LayerTypes { get; init; } = [];
    /// <summary>True when the query projection also produces a sigmoid output gate (Qwen3.5 full attention).</summary>
    public bool AttentionOutputGate { get; init; }
    /// <summary>Fraction of each head that rotary embeddings rotate; 1 rotates the whole head.</summary>
    public double PartialRotaryFactor { get; init; } = 1.0;
    /// <summary>Added to RMSNorm weights when they are loaded (Qwen3.5 stores them as <c>w</c> for a <c>1 + w</c> scale).</summary>
    public float RmsNormOffset { get; init; }
    /// <summary>Key heads of a Gated DeltaNet layer.</summary>
    public int LinearKeyHeads { get; init; }
    /// <summary>Value heads of a Gated DeltaNet layer.</summary>
    public int LinearValueHeads { get; init; }
    /// <summary>Width of a Gated DeltaNet key head.</summary>
    public int LinearKeyHeadDim { get; init; }
    /// <summary>Width of a Gated DeltaNet value head.</summary>
    public int LinearValueHeadDim { get; init; }
    /// <summary>Taps of a Gated DeltaNet causal convolution.</summary>
    public int LinearConvKernel { get; init; }
    /// <summary>Number of elements of each head that rotary embeddings rotate.</summary>
    public int RotaryDim => (int)(HeadDim * PartialRotaryFactor);
    /// <summary>Total width of the Gated DeltaNet keys (also of the queries).</summary>
    public int LinearKeyDim => LinearKeyHeads * LinearKeyHeadDim;
    /// <summary>Total width of the Gated DeltaNet values.</summary>
    public int LinearValueDim => LinearValueHeads * LinearValueHeadDim;
    /// <summary>Channels of the Gated DeltaNet convolution: q, k and v side by side.</summary>
    public int LinearConvDim => (2 * LinearKeyDim) + LinearValueDim;
    /// <summary>True when layer <paramref name="index"/> is a Gated DeltaNet (linear attention) layer.</summary>
    public bool IsLinearLayer(int index) =>
        index >= 0 && index < LayerTypes.Count && string.Equals(LayerTypes[index], "linear_attention", StringComparison.Ordinal);
    /// <summary>True when any layer is a Gated DeltaNet layer.</summary>
    public bool HasLinearLayers => LayerTypes.Any(t => string.Equals(t, "linear_attention", StringComparison.Ordinal));
    /// <summary>True when the rotary embeddings need the table-driven kernel (scaling or partial rotary).</summary>
    public bool NeedsRopeTable => RotaryDim != HeadDim || RopeFrequencies.IsScaled(this);    /// <summary>Number of query heads that share each key/value head.</summary>
    public int GroupSize => NumAttentionHeads / NumKeyValueHeads;

    /// <summary>Width of the concatenated query projection output.</summary>
    public int QueryDim => NumAttentionHeads * HeadDim;

    /// <summary>Width of the key (and value) projection output.</summary>
    public int KeyValueDim => NumKeyValueHeads * HeadDim;

    /// <summary>Loads <c>config.json</c> from a file path, or from a directory that contains it.</summary>
    public static HfModelConfig Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string file = Directory.Exists(path) ? Path.Combine(path, "config.json") : path;
        return Parse(File.ReadAllText(file));
    }

    /// <summary>Parses the JSON text of a HuggingFace <c>config.json</c>.</summary>
    /// <exception cref="InvalidDataException">A required field is missing or has the wrong type.</exception>
    public static HfModelConfig Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = true });
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"config.json is not valid JSON: {ex.Message}", ex);
        }

        using (doc)
        {
            JsonElement root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("config.json root is not an object.");
            }

            string modelType = RequireString(root, "model_type");
            // Multimodal checkpoints (Qwen3.5) nest the language model under text_config.
            bool nested = root.TryGetProperty("text_config", out JsonElement textConfig) && textConfig.ValueKind == JsonValueKind.Object;
            JsonElement text = nested ? textConfig : root;
            int hidden = RequireInt(text, "hidden_size");
            int heads = RequireInt(text, "num_attention_heads");
            int kvHeads = OptionalInt(text, "num_key_value_heads") ?? heads;
            int headDim = OptionalInt(text, "head_dim") ?? (heads > 0 ? hidden / heads : 0);
            bool useSliding = OptionalBool(text, "use_sliding_window") ?? false;
            int? sliding = useSliding ? OptionalInt(text, "sliding_window") : null;
            bool qwen35 = modelType.StartsWith("qwen3_5", StringComparison.Ordinal);
            JsonElement ropeParams = text.TryGetProperty("rope_parameters", out JsonElement rp) && rp.ValueKind == JsonValueKind.Object ? rp : default;
            double ropeTheta = OptionalDouble(text, "rope_theta")
                ?? (ropeParams.ValueKind == JsonValueKind.Object ? OptionalDouble(ropeParams, "rope_theta") : null)
                ?? 10000.0;
            double partial = ropeParams.ValueKind == JsonValueKind.Object ? OptionalDouble(ropeParams, "partial_rotary_factor") ?? 1.0 : 1.0;
            IReadOnlyList<string> layerTypes = ReadLayerTypes(text);
            return new HfModelConfig
            {
                ModelType = modelType,
                HiddenSize = hidden,
                IntermediateSize = RequireInt(text, "intermediate_size"),
                NumLayers = RequireInt(text, "num_hidden_layers"),
                NumAttentionHeads = heads,
                NumKeyValueHeads = kvHeads,
                HeadDim = headDim,
                VocabSize = RequireInt(text, "vocab_size"),
                MaxPositionEmbeddings = OptionalInt(text, "max_position_embeddings") ?? 2048,
                RmsNormEps = (float)(OptionalDouble(text, "rms_norm_eps") ?? 1e-6),
                RopeTheta = ropeTheta,
                TieWordEmbeddings = OptionalBool(text, "tie_word_embeddings") ?? OptionalBool(root, "tie_word_embeddings") ?? false,
                HiddenAct = OptionalString(text, "hidden_act") ?? "silu",
                TorchDtype = OptionalString(text, "torch_dtype") ?? OptionalString(text, "dtype") ?? OptionalString(root, "dtype") ?? string.Empty,
                AttentionBias = modelType == "qwen2" || (OptionalBool(text, "attention_bias") ?? false),
                QkNorm = modelType == "qwen3" || qwen35,
                RopeScalingType = ReadRopeScalingType(text),
                RopeScaling = ReadRopeScaling(text),
                SlidingWindow = sliding,
                BosTokenId = OptionalInt(text, "bos_token_id"),
                EosTokenIds = ReadEosTokenIds(text),
                TensorPrefix = nested ? "model.language_model." : "model.",
                LayerTypes = layerTypes,
                AttentionOutputGate = OptionalBool(text, "attn_output_gate") ?? false,
                PartialRotaryFactor = partial,
                RmsNormOffset = qwen35 ? 1f : 0f,
                LinearKeyHeads = OptionalInt(text, "linear_num_key_heads") ?? 0,
                LinearValueHeads = OptionalInt(text, "linear_num_value_heads") ?? 0,
                LinearKeyHeadDim = OptionalInt(text, "linear_key_head_dim") ?? 0,
                LinearValueHeadDim = OptionalInt(text, "linear_value_head_dim") ?? 0,
                LinearConvKernel = OptionalInt(text, "linear_conv_kernel_dim") ?? 0,
            };
        }
    }    /// <summary>
    /// Checks the config describes a model the native Level Zero path can run today and that its
    /// dimensions are self-consistent.
    /// </summary>
    /// <exception cref="NotSupportedException">The model needs a feature that is not implemented yet.</exception>
    /// <exception cref="InvalidDataException">The dimensions are inconsistent.</exception>
    public void EnsureSupported()
    {
        if (ModelType is not ("qwen2" or "qwen3" or "llama" or "qwen3_5" or "qwen3_5_text"))
        {
            throw new NotSupportedException($"model_type '{ModelType}' is not supported (qwen2, qwen3, qwen3_5 and llama are).");
        }

        if (!string.Equals(HiddenAct, "silu", StringComparison.Ordinal))
        {
            throw new NotSupportedException($"hidden_act '{HiddenAct}' is not supported (only silu).");
        }

        if (RopeScalingType is not null and not ("default" or "linear" or "yarn" or "llama3"))
        {
            throw new NotSupportedException($"rope_scaling '{RopeScalingType}' is not supported (default, linear, yarn and llama3 are).");
        }
        if (RopeScaling is { Factor: <= 0 })
        {
            throw new InvalidDataException("rope_scaling factor must be positive.");
        }

        if (SlidingWindow is not null)
        {
            throw new NotSupportedException("Sliding-window attention is not supported.");
        }

        if (NumAttentionHeads <= 0 || NumKeyValueHeads <= 0 || HeadDim <= 0 || NumLayers <= 0)
        {
            throw new InvalidDataException("Head counts, head_dim and layer count must be positive.");
        }

        if (NumAttentionHeads % NumKeyValueHeads != 0)
        {
            throw new InvalidDataException(
                $"num_attention_heads ({NumAttentionHeads}) is not a multiple of num_key_value_heads ({NumKeyValueHeads}).");
        }

        if (HeadDim % 2 != 0)
        {
            throw new InvalidDataException($"head_dim ({HeadDim}) must be even for rotary embeddings.");
        }
        ValidateLayerMixers();
        if (RotaryDim <= 0 || RotaryDim % 2 != 0 || RotaryDim > HeadDim)
        {
            throw new InvalidDataException($"The rotary dimension ({RotaryDim}) must be even and within head_dim ({HeadDim}).");
        }
    }

    /// <summary>
    /// Lists every weight tensor the model needs, with its expected shape, using HuggingFace names.
    /// Linear weights are <c>[out, in]</c>. The <c>lm_head.weight</c> entry is omitted when embeddings are tied.
    /// </summary>
    public IReadOnlyList<(string Name, long[] Shape)> ExpectedTensors()
    {
        var list = new List<(string, long[])>
        {
            (TensorPrefix + "embed_tokens.weight", [VocabSize, HiddenSize]),
            (TensorPrefix + "norm.weight", [HiddenSize]),
        };

        if (!TieWordEmbeddings)
        {
            list.Add(("lm_head.weight", [VocabSize, HiddenSize]));
        }

        for (int i = 0; i < NumLayers; i++)
        {
            string p = string.Create(CultureInfo.InvariantCulture, $"{TensorPrefix}layers.{i}.");
            list.Add((p + "input_layernorm.weight", [HiddenSize]));
            list.Add((p + "post_attention_layernorm.weight", [HiddenSize]));
            if (IsLinearLayer(i))
            {
                list.Add((p + "linear_attn.in_proj_qkv.weight", [LinearConvDim, HiddenSize]));
                list.Add((p + "linear_attn.in_proj_z.weight", [LinearValueDim, HiddenSize]));
                list.Add((p + "linear_attn.in_proj_b.weight", [LinearValueHeads, HiddenSize]));
                list.Add((p + "linear_attn.in_proj_a.weight", [LinearValueHeads, HiddenSize]));
                list.Add((p + "linear_attn.out_proj.weight", [HiddenSize, LinearValueDim]));
                list.Add((p + "linear_attn.conv1d.weight", [LinearConvDim, 1, LinearConvKernel]));
                list.Add((p + "linear_attn.A_log", [LinearValueHeads]));
                list.Add((p + "linear_attn.dt_bias", [LinearValueHeads]));
                list.Add((p + "linear_attn.norm.weight", [LinearValueHeadDim]));
            }
            else
            {
                list.Add((p + "self_attn.q_proj.weight", [AttentionOutputGate ? 2 * QueryDim : QueryDim, HiddenSize]));
                list.Add((p + "self_attn.k_proj.weight", [KeyValueDim, HiddenSize]));
                list.Add((p + "self_attn.v_proj.weight", [KeyValueDim, HiddenSize]));
                list.Add((p + "self_attn.o_proj.weight", [HiddenSize, QueryDim]));
                if (AttentionBias)
                {
                    list.Add((p + "self_attn.q_proj.bias", [QueryDim]));
                    list.Add((p + "self_attn.k_proj.bias", [KeyValueDim]));
                    list.Add((p + "self_attn.v_proj.bias", [KeyValueDim]));
                }
                if (QkNorm)
                {
                    list.Add((p + "self_attn.q_norm.weight", [HeadDim]));
                    list.Add((p + "self_attn.k_norm.weight", [HeadDim]));
                }
            }            list.Add((p + "mlp.gate_proj.weight", [IntermediateSize, HiddenSize]));
            list.Add((p + "mlp.up_proj.weight", [IntermediateSize, HiddenSize]));
            list.Add((p + "mlp.down_proj.weight", [HiddenSize, IntermediateSize]));
        }

        return list;
    }

    /// <summary>
    /// Compares the checkpoint against <see cref="ExpectedTensors"/> and returns one message per
    /// missing tensor or shape mismatch. An empty list means the checkpoint is complete.
    /// </summary>
    public IReadOnlyList<string> ValidateCheckpoint(SafetensorsCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var problems = new List<string>();
        foreach ((string name, long[] shape) in ExpectedTensors())
        {
            if (!checkpoint.TryGetInfo(name, out SafetensorsTensorInfo? info) || info is null)
            {
                problems.Add($"missing tensor '{name}'");
            }
            else if (!info.Shape.SequenceEqual(shape))
            {
                problems.Add(
                    $"tensor '{name}' has shape [{string.Join(", ", info.Shape)}], expected [{string.Join(", ", shape)}]");
            }
        }

        return problems;
    }

    private void ValidateLayerMixers()
    {
        if (LayerTypes.Count == 0)
        {
            return;
        }
        if (LayerTypes.Count != NumLayers)
        {
            throw new InvalidDataException($"layer_types has {LayerTypes.Count} entries but there are {NumLayers} layers.");
        }
        foreach (string type in LayerTypes)
        {
            if (type is not ("full_attention" or "linear_attention"))
            {
                throw new NotSupportedException($"layer type '{type}' is not supported (full_attention and linear_attention are).");
            }
        }
        if (!HasLinearLayers)
        {
            return;
        }
        if (LinearKeyHeads <= 0 || LinearValueHeads <= 0 || LinearKeyHeadDim <= 0 || LinearValueHeadDim <= 0)
        {
            throw new InvalidDataException("linear_num_*_heads and linear_*_head_dim must be positive for linear-attention layers.");
        }
        if (LinearKeyHeads != LinearValueHeads)
        {
            throw new NotSupportedException("Gated DeltaNet with a different number of key and value heads is not supported.");
        }
        if (LinearValueHeadDim > 1024 || (LinearValueHeadDim & (LinearValueHeadDim - 1)) != 0)
        {
            throw new NotSupportedException("linear_value_head_dim must be a power of two up to 1024.");
        }
        if (LinearConvKernel is < 2 or > 8)
        {
            throw new NotSupportedException("linear_conv_kernel_dim must be 2 to 8.");
        }
    }
    private static IReadOnlyList<string> ReadLayerTypes(JsonElement root)
    {
        if (!root.TryGetProperty("layer_types", out JsonElement types) || types.ValueKind != JsonValueKind.Array)
        {
            return [];
        }
        var list = new List<string>();
        foreach (JsonElement e in types.EnumerateArray())
        {
            list.Add(e.ValueKind == JsonValueKind.String ? e.GetString() ?? "unknown" : "unknown");
        }
        return list;
    }    private static HfRopeScaling? ReadRopeScaling(JsonElement root)
    {
        if (!TryGetScalingBlock(root, out JsonElement scaling))
        {
            return null;
        }
        return new HfRopeScaling
        {
            Type = OptionalString(scaling, "rope_type") ?? OptionalString(scaling, "type") ?? "unknown",
            Factor = OptionalDouble(scaling, "factor") ?? 1.0,
            OriginalMaxPositionEmbeddings = OptionalInt(scaling, "original_max_position_embeddings"),
            BetaFast = OptionalDouble(scaling, "beta_fast") ?? 32.0,
            BetaSlow = OptionalDouble(scaling, "beta_slow") ?? 1.0,
            Mscale = OptionalDouble(scaling, "mscale"),
            MscaleAllDim = OptionalDouble(scaling, "mscale_all_dim"),
            AttentionFactor = OptionalDouble(scaling, "attention_factor"),
            Truncate = OptionalBool(scaling, "truncate") ?? true,
            LowFreqFactor = OptionalDouble(scaling, "low_freq_factor") ?? 1.0,
            HighFreqFactor = OptionalDouble(scaling, "high_freq_factor") ?? 4.0,
        };
    }


    private static bool TryGetScalingBlock(JsonElement root, out JsonElement scaling)
    {
        if (root.TryGetProperty("rope_scaling", out scaling) && scaling.ValueKind == JsonValueKind.Object)
        {
            return true;
        }

        // Newer transformers configs keep everything under rope_parameters; only a block with a type counts.
        return root.TryGetProperty("rope_parameters", out scaling)
            && scaling.ValueKind == JsonValueKind.Object
            && (scaling.TryGetProperty("rope_type", out _) || scaling.TryGetProperty("type", out _));
    }

    private static string? ReadRopeScalingType(JsonElement root)
    {
        if (!TryGetScalingBlock(root, out JsonElement scaling))
        {
            return null;
        }

        return OptionalString(scaling, "rope_type") ?? OptionalString(scaling, "type") ?? "unknown";
    }

    private static IReadOnlyList<int> ReadEosTokenIds(JsonElement root)
    {
        if (!root.TryGetProperty("eos_token_id", out JsonElement eos))
        {
            return [];
        }

        if (eos.ValueKind == JsonValueKind.Number && eos.TryGetInt32(out int single))
        {
            return [single];
        }

        if (eos.ValueKind == JsonValueKind.Array)
        {
            var ids = new List<int>();
            foreach (JsonElement e in eos.EnumerateArray())
            {
                if (e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out int id))
                {
                    ids.Add(id);
                }
            }

            return ids;
        }

        return [];
    }

    private static string RequireString(JsonElement root, string name) =>
        OptionalString(root, name) ?? throw new InvalidDataException($"config.json is missing '{name}'.");

    private static int RequireInt(JsonElement root, string name) =>
        OptionalInt(root, name) ?? throw new InvalidDataException($"config.json is missing '{name}'.");

    private static string? OptionalString(JsonElement root, string name) =>
        root.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.String ? e.GetString() : null;

    private static int? OptionalInt(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out JsonElement e) || e.ValueKind != JsonValueKind.Number)
        {
            return null;
        }

        return e.TryGetInt32(out int v)
            ? v
            : throw new InvalidDataException($"config.json '{name}' is not a 32-bit integer.");
    }

    private static double? OptionalDouble(JsonElement root, string name) =>
        root.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.Number ? e.GetDouble() : null;

    private static bool? OptionalBool(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out JsonElement e))
        {
            return null;
        }

        return e.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null,
        };
    }
}
