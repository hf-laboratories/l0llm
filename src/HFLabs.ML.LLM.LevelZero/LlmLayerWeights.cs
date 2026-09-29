using System.Globalization;
using LevelZero;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// The weights of one decoder layer, resident in GPU-visible shared memory as float32.
/// Projection matrices are stored transposed to <c>[in, out]</c> because the GEMM kernel computes
/// <c>C = A * B</c> with <c>B[k * N + col]</c>; HuggingFace stores them as <c>[out, in]</c>.
/// </summary>
public sealed class LlmLayerWeights : IDisposable
{
    private readonly List<IDisposable> _owned;

    private LlmLayerWeights(
        List<IDisposable> owned,
        SharedBuffer<float> inputNorm,
        SharedBuffer<float> postNorm,
        LlmMatrix? qProj,
        LlmMatrix? kProj,
        LlmMatrix? vProj,
        LlmMatrix? oProj,
        SharedBuffer<float>? qBias,
        SharedBuffer<float>? kBias,
        SharedBuffer<float>? vBias,
        SharedBuffer<float>? qNorm,
        SharedBuffer<float>? kNorm,
        LlmMatrix gateProj,
        LlmMatrix upProj,
        LlmMatrix downProj,
        LlmMatrix? attentionGateProj,
        LlmLinearAttentionWeights? linear)
    {
        _owned = owned;
        InputNorm = inputNorm;
        PostNorm = postNorm;
        QProj = qProj;
        KProj = kProj;
        VProj = vProj;
        OProj = oProj;
        QBias = qBias;
        KBias = kBias;
        VBias = vBias;
        QNorm = qNorm;
        KNorm = kNorm;
        GateProj = gateProj;
        UpProj = upProj;
        DownProj = downProj;
        AttentionGateProj = attentionGateProj;
        Linear = linear;
    }

    /// <summary>RMSNorm weight applied before attention [hidden].</summary>
    public SharedBuffer<float> InputNorm { get; }

    /// <summary>RMSNorm weight applied before the MLP [hidden].</summary>
    public SharedBuffer<float> PostNorm { get; }

    /// <summary>Query projection, <c>[hidden, numHeads * headDim]</c>; null in a linear-attention layer.</summary>
    public LlmMatrix? QProj { get; }

    /// <summary>Key projection, <c>[hidden, numKvHeads * headDim]</c>.</summary>
    public LlmMatrix? KProj { get; }

    /// <summary>Value projection, <c>[hidden, numKvHeads * headDim]</c>.</summary>
    public LlmMatrix? VProj { get; }

    /// <summary>Output projection, <c>[numHeads * headDim, hidden]</c>.</summary>
    public LlmMatrix? OProj { get; }

    /// <summary>Device bytes held by the seven projection matrices.</summary>
    public long ProjectionBytes =>
        (QProj?.SizeInBytes ?? 0) + (KProj?.SizeInBytes ?? 0) + (VProj?.SizeInBytes ?? 0) + (OProj?.SizeInBytes ?? 0)
        + (AttentionGateProj?.SizeInBytes ?? 0) + (Linear?.ProjectionBytes ?? 0)
        + GateProj.SizeInBytes + UpProj.SizeInBytes + DownProj.SizeInBytes;
    /// <summary>True when the projections are stored as half precision.</summary>
    public bool UsesHalf => GateProj.IsHalf;
    /// <summary>Query bias, present when the model has attention bias.</summary>
    public SharedBuffer<float>? QBias { get; }

    /// <summary>Key bias, present when the model has attention bias.</summary>
    public SharedBuffer<float>? KBias { get; }

    /// <summary>Value bias, present when the model has attention bias.</summary>
    public SharedBuffer<float>? VBias { get; }

    /// <summary>Per-head RMSNorm weight for queries [headDim], present for Qwen3-style QK-norm.</summary>
    public SharedBuffer<float>? QNorm { get; }

    /// <summary>Per-head RMSNorm weight for keys [headDim], present for Qwen3-style QK-norm.</summary>
    public SharedBuffer<float>? KNorm { get; }

    /// <summary>Attention output-gate projection, <c>[hidden, numHeads * headDim]</c>; present for Qwen3.5 full attention.</summary>
    public LlmMatrix? AttentionGateProj { get; }

    /// <summary>Gated DeltaNet weights; non-null exactly when this layer is a linear-attention layer.</summary>
    public LlmLinearAttentionWeights? Linear { get; }

    /// <summary>MLP gate projection, <c>[hidden, intermediate]</c>.</summary>
    public LlmMatrix GateProj { get; }

    /// <summary>MLP up projection, <c>[hidden, intermediate]</c>.</summary>
    public LlmMatrix UpProj { get; }

    /// <summary>MLP down projection, <c>[intermediate, hidden]</c>.</summary>
    public LlmMatrix DownProj { get; }

    /// <summary>Reads layer <paramref name="layerIndex"/> from the checkpoint and uploads it to the device.</summary>
    /// <exception cref="InvalidDataException">A tensor is missing or has an unexpected shape.</exception>
    public static LlmLayerWeights Load(
        ComputeDevice device,
        SafetensorsCheckpoint checkpoint,
        HfModelConfig config,
        int layerIndex,
        LlmWeightPrecision precision = LlmWeightPrecision.Auto)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfNegative(layerIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layerIndex, config.NumLayers);

        string p = string.Create(CultureInfo.InvariantCulture, $"{config.TensorPrefix}layers.{layerIndex}.");
        var owned = new List<IDisposable>();
        try
        {
            SharedBuffer<float> Vector(string name, int length) => Track(owned, UploadVector(device, checkpoint, name, length, 0f));
            SharedBuffer<float> NormVector(string name, int length) =>
                Track(owned, UploadVector(device, checkpoint, name, length, config.RmsNormOffset));
            LlmMatrix Matrix(string name, int outDim, int inDim) =>
                Track(owned, UploadTransposed(device, checkpoint, name, outDim, inDim, precision));
            int h = config.HiddenSize;
            SharedBuffer<float> inputNorm = NormVector(p + "input_layernorm.weight", h);
            SharedBuffer<float> postNorm = NormVector(p + "post_attention_layernorm.weight", h);
            LlmMatrix? q = null;
            LlmMatrix? k = null;
            LlmMatrix? v = null;
            LlmMatrix? o = null;
            LlmMatrix? attentionGate = null;
            LlmLinearAttentionWeights? linear = null;
            SharedBuffer<float>? qb = null;
            SharedBuffer<float>? kb = null;
            SharedBuffer<float>? vb = null;
            SharedBuffer<float>? qn = null;
            SharedBuffer<float>? kn = null;
            if (config.IsLinearLayer(layerIndex))
            {
                string lp = p + "linear_attn.";
                int valueDim = config.LinearValueDim;
                int convDim = config.LinearConvDim;
                int heads = config.LinearValueHeads;
                RequireShape(checkpoint, lp + "conv1d.weight", convDim, 1, config.LinearConvKernel);
                linear = new LlmLinearAttentionWeights(
                    Matrix(lp + "in_proj_qkv.weight", convDim, h),
                    Matrix(lp + "in_proj_z.weight", valueDim, h),
                    Matrix(lp + "in_proj_b.weight", heads, h),
                    Matrix(lp + "in_proj_a.weight", heads, h),
                    Matrix(lp + "out_proj.weight", h, valueDim),
                    Track(owned, device.AllocShared(checkpoint.ReadFloat32(lp + "conv1d.weight"))),
                    Vector(lp + "A_log", heads),
                    Vector(lp + "dt_bias", heads),
                    Vector(lp + "norm.weight", config.LinearValueHeadDim));
            }
            else
            {
                int qDim = config.QueryDim;
                if (config.AttentionOutputGate)
                {
                    // q_proj rows are [head0 query | head0 gate | head1 query | head1 gate | ...]: split them into two projections.
                    string qName = p + "self_attn.q_proj.weight";
                    RequireShape(checkpoint, qName, 2L * qDim, h);
                    float[] both = checkpoint.ReadFloat32(qName);
                    bool sixteen = checkpoint.TryGetInfo(qName, out SafetensorsTensorInfo? qInfo)
                        && qInfo is not null
                        && qInfo.DType is (SafetensorsDType.BF16 or SafetensorsDType.F16);
                    (float[] qRows, float[] gateRows) = SplitQueryAndGate(both, config.NumAttentionHeads, config.HeadDim, h);
                    q = Track(owned, LlmMatrix.FromHuggingFace(device, qRows, sixteen, qName, qDim, h, precision));
                    attentionGate = Track(owned, LlmMatrix.FromHuggingFace(device, gateRows, sixteen, qName, qDim, h, precision));
                }
                else
                {
                    q = Matrix(p + "self_attn.q_proj.weight", qDim, h);
                }
                k = Matrix(p + "self_attn.k_proj.weight", config.KeyValueDim, h);
                v = Matrix(p + "self_attn.v_proj.weight", config.KeyValueDim, h);
                o = Matrix(p + "self_attn.o_proj.weight", h, qDim);
                if (config.AttentionBias)
                {
                    qb = Vector(p + "self_attn.q_proj.bias", qDim);
                    kb = Vector(p + "self_attn.k_proj.bias", config.KeyValueDim);
                    vb = Vector(p + "self_attn.v_proj.bias", config.KeyValueDim);
                }
                if (config.QkNorm)
                {
                    qn = NormVector(p + "self_attn.q_norm.weight", config.HeadDim);
                    kn = NormVector(p + "self_attn.k_norm.weight", config.HeadDim);
                }
            }
            LlmMatrix gate = Matrix(p + "mlp.gate_proj.weight", config.IntermediateSize, h);
            LlmMatrix up = Matrix(p + "mlp.up_proj.weight", config.IntermediateSize, h);
            LlmMatrix down = Matrix(p + "mlp.down_proj.weight", h, config.IntermediateSize);
            return new LlmLayerWeights(owned, inputNorm, postNorm, q, k, v, o, qb, kb, vb, qn, kn, gate, up, down, attentionGate, linear);
        }        catch
        {
            foreach (IDisposable d in owned)
            {
                d.Dispose();
            }

            throw;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (IDisposable d in _owned)
        {
            d.Dispose();
        }
    }

    /// <summary>
    /// Transposes a row-major <c>[rows, cols]</c> matrix to <c>[cols, rows]</c>. Exposed for tests.
    /// </summary>
    public static float[] Transpose(float[] source, int rows, int cols)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Length != (long)rows * cols)
        {
            throw new ArgumentException($"Expected {(long)rows * cols} elements, got {source.Length}.", nameof(source));
        }

        var result = new float[source.Length];
        const int block = 32;
        int rowBlocks = (rows + block - 1) / block;
        // Row blocks write disjoint columns of the result, so they can run in parallel.
        _ = Parallel.For(0, rowBlocks, rb =>
        {
            int r0 = rb * block;
            int r1 = Math.Min(r0 + block, rows);
            for (int c0 = 0; c0 < cols; c0 += block)
            {
                int c1 = Math.Min(c0 + block, cols);
                for (int r = r0; r < r1; r++)
                {
                    for (int c = c0; c < c1; c++)
                    {
                        result[(c * rows) + r] = source[(r * cols) + c];
                    }
                }
            }
        });
        return result;
    }

    private static T Track<T>(List<IDisposable> owned, T buffer)
        where T : IDisposable
    {
        owned.Add(buffer);
        return buffer;
    }

    private static SharedBuffer<float> UploadVector(
        ComputeDevice device, SafetensorsCheckpoint checkpoint, string name, int length, float offset)
    {
        RequireShape(checkpoint, name, length);
        float[] values = checkpoint.ReadFloat32(name);
        if (offset != 0f)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] += offset;
            }
        }
        return device.AllocShared(values);
    }
    /// <summary>
    /// Splits a Qwen3.5 <c>q_proj</c> (<c>[heads * 2 * headDim, hidden]</c>, each head's query rows followed by its gate rows)
    /// into a query matrix and a gate matrix, both <c>[heads * headDim, hidden]</c>. Exposed for tests.
    /// </summary>
    public static (float[] Query, float[] Gate) SplitQueryAndGate(float[] hf, int heads, int headDim, int hidden)
    {
        ArgumentNullException.ThrowIfNull(hf);
        if (hf.Length != (long)heads * 2 * headDim * hidden)
        {
            throw new ArgumentException($"Expected {(long)heads * 2 * headDim * hidden} elements, got {hf.Length}.", nameof(hf));
        }
        long blockSize = (long)headDim * hidden;
        var query = new float[heads * blockSize];
        var gate = new float[heads * blockSize];
        for (int h = 0; h < heads; h++)
        {
            Array.Copy(hf, (2L * h) * blockSize, query, h * blockSize, blockSize);
            Array.Copy(hf, ((2L * h) + 1) * blockSize, gate, h * blockSize, blockSize);
        }
        return (query, gate);
    }

    private static LlmMatrix UploadTransposed(
        ComputeDevice device, SafetensorsCheckpoint checkpoint, string name, int outDim, int inDim, LlmWeightPrecision precision)
    {
        RequireShape(checkpoint, name, outDim, inDim);
        return LlmMatrix.LoadTransposed(device, checkpoint, name, outDim, inDim, precision);
    }

    private static void RequireShape(SafetensorsCheckpoint checkpoint, string name, params long[] shape)
    {
        if (!checkpoint.TryGetInfo(name, out SafetensorsTensorInfo? info) || info is null)
        {
            throw new InvalidDataException($"Checkpoint is missing tensor '{name}'.");
        }

        if (!info.Shape.SequenceEqual(shape))
        {
            throw new InvalidDataException(
                $"Tensor '{name}' has shape [{string.Join(", ", info.Shape)}], expected [{string.Join(", ", shape)}].");
        }
    }
}
