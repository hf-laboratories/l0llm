using LevelZero;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// Key and value cache for one layer, laid out <c>[maxSeqLen, numKvHeads * headDim]</c> to match the
/// fused attention kernel.
/// </summary>
public sealed class LlmKvCache : IDisposable
{
    /// <summary>Allocates zeroed caches for <paramref name="maxSeqLen"/> positions.</summary>
    public LlmKvCache(ComputeDevice device, int maxSeqLen, int kvDim)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDim);

        MaxSeqLen = maxSeqLen;
        KvDim = kvDim;
        int count = checked(maxSeqLen * kvDim);
        Keys = device.AllocShared<float>(count);
        try
        {
            Values = device.AllocShared<float>(count);
        }
        catch
        {
            Keys.Dispose();
            throw;
        }

        Keys.Write(new float[count]);
        Values.Write(new float[count]);
    }

    /// <summary>Number of positions the cache can hold.</summary>
    public int MaxSeqLen { get; }

    /// <summary>Width of one cache row (<c>numKvHeads * headDim</c>).</summary>
    public int KvDim { get; }

    /// <summary>Key cache <c>[maxSeqLen, kvDim]</c>.</summary>
    public SharedBuffer<float> Keys { get; }

    /// <summary>Value cache <c>[maxSeqLen, kvDim]</c>.</summary>
    public SharedBuffer<float> Values { get; }

    /// <inheritdoc />
    public void Dispose()
    {
        Keys.Dispose();
        Values.Dispose();
    }
}
