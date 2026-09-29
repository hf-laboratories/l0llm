using LevelZero;
namespace HFLabs.ML.LLM.LevelZero;
/// <summary>
/// The weights of one Gated DeltaNet (linear attention) layer. The buffers and matrices are owned by the
/// <see cref="LlmLayerWeights"/> that created them.
/// </summary>
public sealed class LlmLinearAttentionWeights
{
    internal LlmLinearAttentionWeights(
        LlmMatrix inProjQkv,
        LlmMatrix inProjZ,
        LlmMatrix inProjB,
        LlmMatrix inProjA,
        LlmMatrix outProj,
        SharedBuffer<float> conv,
        SharedBuffer<float> aLog,
        SharedBuffer<float> dtBias,
        SharedBuffer<float> norm)
    {
        InProjQkv = inProjQkv;
        InProjZ = inProjZ;
        InProjB = inProjB;
        InProjA = inProjA;
        OutProj = outProj;
        Conv = conv;
        ALog = aLog;
        DtBias = dtBias;
        Norm = norm;
    }
    /// <summary>Fused q/k/v projection, <c>[hidden, convDim]</c>.</summary>
    public LlmMatrix InProjQkv { get; }
    /// <summary>Output-gate projection, <c>[hidden, valueDim]</c>.</summary>
    public LlmMatrix InProjZ { get; }
    /// <summary>Beta projection, <c>[hidden, valueHeads]</c>.</summary>
    public LlmMatrix InProjB { get; }
    /// <summary>Decay projection, <c>[hidden, valueHeads]</c>.</summary>
    public LlmMatrix InProjA { get; }
    /// <summary>Output projection, <c>[valueDim, hidden]</c>.</summary>
    public LlmMatrix OutProj { get; }
    /// <summary>Depthwise conv taps, <c>[convDim, kernel]</c>.</summary>
    public SharedBuffer<float> Conv { get; }
    /// <summary><c>A_log</c>, <c>[valueHeads]</c>.</summary>
    public SharedBuffer<float> ALog { get; }
    /// <summary><c>dt_bias</c>, <c>[valueHeads]</c>.</summary>
    public SharedBuffer<float> DtBias { get; }
    /// <summary>Gated RMSNorm weight, <c>[valueHeadDim]</c>, used as stored (no <c>1 + w</c>).</summary>
    public SharedBuffer<float> Norm { get; }
    /// <summary>Device bytes held by the five projection matrices.</summary>
    public long ProjectionBytes =>
        InProjQkv.SizeInBytes + InProjZ.SizeInBytes + InProjB.SizeInBytes + InProjA.SizeInBytes + OutProj.SizeInBytes;
}
/// <summary>
/// The per-sequence state of one Gated DeltaNet layer: the causal-conv history and the recurrent delta-rule
/// matrix. Unlike a KV cache it does not grow with the sequence.
/// </summary>
public sealed class LlmLinearState : IDisposable
{
    /// <summary>Allocates zeroed state for the shape described by <paramref name="config"/>.</summary>
    public LlmLinearState(ComputeDevice device, HfModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        Conv = device.AllocShared<float>(config.LinearConvDim * (config.LinearConvKernel - 1));
        try
        {
            Recurrent = device.AllocShared<float>(config.LinearValueHeads * config.LinearKeyHeadDim * config.LinearValueHeadDim);
        }
        catch
        {
            Conv.Dispose();
            throw;
        }
        Reset();
    }
    /// <summary>Conv history <c>[convDim, kernel - 1]</c>, oldest first.</summary>
    public SharedBuffer<float> Conv { get; }
    /// <summary>Recurrent state <c>[valueHeads, keyHeadDim, valueHeadDim]</c>.</summary>
    public SharedBuffer<float> Recurrent { get; }
    /// <summary>Zeroes the state, as for a new sequence.</summary>
    public void Reset()
    {
        Conv.Write(new float[Conv.Count]);
        Recurrent.Write(new float[Recurrent.Count]);
    }
    /// <inheritdoc />
    public void Dispose()
    {
        Conv.Dispose();
        Recurrent.Dispose();
    }
}