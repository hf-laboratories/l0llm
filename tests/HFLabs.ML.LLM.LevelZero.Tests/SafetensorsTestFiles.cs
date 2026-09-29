using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Writes small safetensors files for tests, independent of the reader under test.</summary>
internal static class SafetensorsTestFiles
{
    public sealed record Entry(string Name, string DType, long[] Shape, byte[] Data);

    public static Entry F32(string name, long[] shape, params float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        }

        return new Entry(name, "F32", shape, bytes);
    }

    public static Entry Raw16(string name, string dtype, long[] shape, params ushort[] bits)
    {
        var bytes = new byte[bits.Length * 2];
        for (int i = 0; i < bits.Length; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), bits[i]);
        }

        return new Entry(name, dtype, shape, bytes);
    }

    public static Entry F64(string name, long[] shape, params double[] values)
    {
        var bytes = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteDoubleLittleEndian(bytes.AsSpan(i * 8), values[i]);
        }

        return new Entry(name, "F64", shape, bytes);
    }

    /// <summary>Writes a valid file with tensors packed back to back, padded header, optional metadata.</summary>
    public static void Write(string path, IReadOnlyList<Entry> entries, Dictionary<string, string>? metadata = null)
    {
        using var headerStream = new MemoryStream();
        using (var w = new Utf8JsonWriter(headerStream))
        {
            w.WriteStartObject();
            if (metadata is not null)
            {
                w.WriteStartObject("__metadata__");
                foreach ((string k, string v) in metadata)
                {
                    w.WriteString(k, v);
                }

                w.WriteEndObject();
            }

            long offset = 0;
            foreach (Entry e in entries)
            {
                w.WriteStartObject(e.Name);
                w.WriteString("dtype", e.DType);
                w.WriteStartArray("shape");
                foreach (long d in e.Shape)
                {
                    w.WriteNumberValue(d);
                }

                w.WriteEndArray();
                w.WriteStartArray("data_offsets");
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + e.Data.Length);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += e.Data.Length;
            }

            w.WriteEndObject();
        }

        var data = new MemoryStream();
        foreach (Entry e in entries)
        {
            data.Write(e.Data);
        }

        WriteRaw(path, Encoding.UTF8.GetString(headerStream.ToArray()), data.ToArray());
    }

    /// <summary>Writes a file with an arbitrary header string (padded to 8 bytes) and data section.</summary>
    public static void WriteRaw(string path, string headerJson, byte[] data)
    {
        int pad = (8 - (Encoding.UTF8.GetByteCount(headerJson) % 8)) % 8;
        byte[] header = Encoding.UTF8.GetBytes(headerJson + new string(' ', pad));
        using var fs = File.Create(path);
        Span<byte> len = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)header.Length);
        fs.Write(len);
        fs.Write(header);
        fs.Write(data);
    }
}

/// <summary>A temporary directory deleted on dispose.</summary>
internal sealed class TempDir : IDisposable
{
    public TempDir()
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mllmlz-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public string File(string name) => System.IO.Path.Combine(Path, name);

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, recursive: true);
        }
        catch (IOException)
        {
            // best effort
        }
    }
}
