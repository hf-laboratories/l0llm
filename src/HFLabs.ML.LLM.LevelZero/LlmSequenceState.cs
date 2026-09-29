using System.Runtime.InteropServices;
using LevelZero;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// A host-side copy of the part of a sequence that changed after a base position: the KV rows
/// <c>[BasePosition, Position)</c> of every attention layer and the complete state of every linear-attention layer.
/// </summary>
/// <remarks>
/// Beams that share a prompt never modify the prompt's rows, so a beam only needs its own tail. Restoring a state
/// therefore costs the generated length, not the prompt length.
/// </remarks>
public sealed class LlmSequenceState
{
    internal LlmSequenceState(
        int basePosition,
        int position,
        float[]?[] keys,
        float[]?[] values,
        float[]?[] conv,
        float[]?[] recurrent)
    {
        BasePosition = basePosition;
        Position = position;
        Keys = keys;
        Values = values;
        Conv = conv;
        Recurrent = recurrent;
    }

    /// <summary>First cache position the state holds rows for.</summary>
    public int BasePosition { get; }

    /// <summary>Sequence length the state was captured at (the next token goes here).</summary>
    public int Position { get; }

    internal float[]?[] Keys { get; }
    internal float[]?[] Values { get; }
    internal float[]?[] Conv { get; }
    internal float[]?[] Recurrent { get; }

    /// <summary>Host bytes held by the copy.</summary>
    public long SizeInBytes
    {
        get
        {
            long floats = 0;
            foreach (float[]?[] group in new[] { Keys, Values, Conv, Recurrent })
            {
                foreach (float[]? a in group)
                {
                    floats += a?.Length ?? 0;
                }
            }
            return floats * sizeof(float);
        }
    }

    internal static float[] ReadRange(SharedBuffer<float> buffer, int start, int count)
    {
        var result = new float[count];
        if (count > 0)
        {
            Marshal.Copy(buffer.Pointer + start * sizeof(float), result, 0, count);
        }
        return result;
    }

    internal static void WriteRange(SharedBuffer<float> buffer, int start, float[] data)
    {
        if (data.Length > 0)
        {
            if (start < 0 || start + data.Length > buffer.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(start));
            }
            Marshal.Copy(data, 0, buffer.Pointer + start * sizeof(float), data.Length);
        }
    }
}
