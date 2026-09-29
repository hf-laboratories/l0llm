using System.Buffers;
using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>Element types defined by the safetensors format that this reader understands.</summary>
public enum SafetensorsDType
{
    Bool,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

/// <summary>Location and shape of one tensor inside a safetensors file.</summary>
/// <param name="Name">Tensor name as stored in the header.</param>
/// <param name="DType">Element type.</param>
/// <param name="Shape">Dimensions, outermost first. Empty for a scalar.</param>
/// <param name="DataOffset">Byte offset from the start of the data section.</param>
/// <param name="ByteLength">Size of the tensor data in bytes.</param>
public sealed record SafetensorsTensorInfo(
    string Name,
    SafetensorsDType DType,
    IReadOnlyList<long> Shape,
    long DataOffset,
    long ByteLength)
{
    /// <summary>Number of elements (product of the shape; 1 for a scalar).</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (long d in Shape)
            {
                count = checked(count * d);
            }

            return count;
        }
    }
}

/// <summary>
/// Reads tensors from a single <c>.safetensors</c> file through a read-only memory map.
/// The header is fully validated on open: offsets must lie inside the file, must not overlap, and
/// each tensor's byte length must equal <c>element count * element size</c>.
/// </summary>
public sealed class SafetensorsReader : IDisposable
{
    private const long MaxHeaderBytes = 100_000_000;
    private const int ChunkBytes = 1 << 20;

    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly long _dataStart;
    private readonly Dictionary<string, SafetensorsTensorInfo> _tensors;
    private bool _disposed;

    private SafetensorsReader(
        string path,
        MemoryMappedFile file,
        MemoryMappedViewAccessor view,
        long dataStart,
        Dictionary<string, SafetensorsTensorInfo> tensors,
        IReadOnlyDictionary<string, string> metadata)
    {
        Path = path;
        _file = file;
        _view = view;
        _dataStart = dataStart;
        _tensors = tensors;
        Metadata = metadata;
    }

    /// <summary>Path of the opened file.</summary>
    public string Path { get; }

    /// <summary>All tensors in the file, keyed by name.</summary>
    public IReadOnlyDictionary<string, SafetensorsTensorInfo> Tensors => _tensors;

    /// <summary>Free-form string metadata from the <c>__metadata__</c> header entry (may be empty).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Size in bytes of one element of <paramref name="dtype"/>.</summary>
    public static int ElementSize(SafetensorsDType dtype) => dtype switch
    {
        SafetensorsDType.Bool or SafetensorsDType.U8 or SafetensorsDType.I8 => 1,
        SafetensorsDType.I16 or SafetensorsDType.U16 or SafetensorsDType.F16 or SafetensorsDType.BF16 => 2,
        SafetensorsDType.I32 or SafetensorsDType.U32 or SafetensorsDType.F32 => 4,
        SafetensorsDType.F64 or SafetensorsDType.I64 or SafetensorsDType.U64 => 8,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Unknown safetensors dtype."),
    };

    /// <summary>Opens and validates a safetensors file.</summary>
    /// <exception cref="InvalidDataException">The file is not a well-formed safetensors file.</exception>
    /// <exception cref="NotSupportedException">The file uses a dtype this reader does not support.</exception>
    public static SafetensorsReader Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("Safetensors reading assumes a little-endian host.");
        }

        byte[] header;
        long fileLength;
        long headerLength;
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096))
        {
            fileLength = stream.Length;
            if (fileLength < 8)
            {
                throw new InvalidDataException($"'{path}' is too small to be a safetensors file ({fileLength} bytes).");
            }

            Span<byte> lengthBytes = stackalloc byte[8];
            stream.ReadExactly(lengthBytes);
            ulong declared = BitConverter.ToUInt64(lengthBytes);
            if (declared > (ulong)MaxHeaderBytes || (long)declared > fileLength - 8)
            {
                throw new InvalidDataException(
                    $"'{path}' declares a header of {declared} bytes, which does not fit the {fileLength}-byte file.");
            }

            headerLength = (long)declared;
            header = new byte[headerLength];
            stream.ReadExactly(header);
        }

        long dataStart = 8 + headerLength;
        long dataLength = fileLength - dataStart;
        var (tensors, metadata) = ParseHeader(path, header, dataLength);

        MemoryMappedFile file = MemoryMappedFile.CreateFromFile(
            path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor view;
        try
        {
            view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }
        catch
        {
            file.Dispose();
            throw;
        }

        return new SafetensorsReader(path, file, view, dataStart, tensors, metadata);
    }

    /// <summary>Returns the info for <paramref name="name"/>, or throws <see cref="KeyNotFoundException"/>.</summary>
    public SafetensorsTensorInfo GetInfo(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _tensors.TryGetValue(name, out SafetensorsTensorInfo? info)
            ? info
            : throw new KeyNotFoundException($"Tensor '{name}' is not in '{Path}'.");
    }

    /// <summary>Reads a tensor's raw bytes exactly as stored.</summary>
    public byte[] ReadRawBytes(string name)
    {
        SafetensorsTensorInfo info = GetInfo(name);
        ThrowIfDisposed();
        if (info.ByteLength > Array.MaxLength)
        {
            throw new NotSupportedException($"Tensor '{name}' is {info.ByteLength} bytes, too large for one array.");
        }

        var bytes = new byte[info.ByteLength];
        _view.ReadArray(_dataStart + info.DataOffset, bytes, 0, bytes.Length);
        return bytes;
    }

    /// <summary>Reads a tensor as float32, converting from F16, BF16 or F64 when needed.</summary>
    public float[] ReadFloat32(string name)
    {
        SafetensorsTensorInfo info = GetInfo(name);
        long count = info.ElementCount;
        if (count > Array.MaxLength)
        {
            throw new NotSupportedException($"Tensor '{name}' has {count} elements, too many for one array.");
        }

        var result = new float[count];
        ReadFloat32(name, result);
        return result;
    }

    /// <summary>
    /// Reads a tensor as float32 into <paramref name="destination"/>, which must have exactly the
    /// tensor's element count. Supports F32, F16, BF16 and F64 sources.
    /// </summary>
    public void ReadFloat32(string name, Span<float> destination)
    {
        SafetensorsTensorInfo info = GetInfo(name);
        ThrowIfDisposed();
        long count = info.ElementCount;
        if (destination.Length != count)
        {
            throw new ArgumentException(
                $"Tensor '{name}' has {count} elements but the destination holds {destination.Length}.",
                nameof(destination));
        }

        if (info.DType is not (SafetensorsDType.F32 or SafetensorsDType.F16 or SafetensorsDType.BF16 or SafetensorsDType.F64))
        {
            throw new NotSupportedException(
                $"Tensor '{name}' is {info.DType}; ReadFloat32 supports F32, F16, BF16 and F64.");
        }

        int elementSize = ElementSize(info.DType);
        int perChunk = ChunkBytes / elementSize;
        byte[] buffer = ArrayPool<byte>.Shared.Rent(ChunkBytes);
        try
        {
            long done = 0;
            while (done < count)
            {
                int n = (int)Math.Min(count - done, perChunk);
                int bytes = n * elementSize;
                _view.ReadArray(_dataStart + info.DataOffset + (done * elementSize), buffer, 0, bytes);
                ConvertToFloat32(info.DType, buffer.AsSpan(0, bytes), destination.Slice((int)done, n));
                done += n;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _view.Dispose();
        _file.Dispose();
    }

    private static void ConvertToFloat32(SafetensorsDType dtype, ReadOnlySpan<byte> source, Span<float> destination)
    {
        switch (dtype)
        {
            case SafetensorsDType.F32:
                MemoryMarshal.Cast<byte, float>(source).CopyTo(destination);
                break;
            case SafetensorsDType.F16:
                ReadOnlySpan<Half> halves = MemoryMarshal.Cast<byte, Half>(source);
                for (int i = 0; i < halves.Length; i++)
                {
                    destination[i] = (float)halves[i];
                }

                break;
            case SafetensorsDType.BF16:
                ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<byte, ushort>(source);
                for (int i = 0; i < bits.Length; i++)
                {
                    destination[i] = BitConverter.Int32BitsToSingle(bits[i] << 16);
                }

                break;
            case SafetensorsDType.F64:
                ReadOnlySpan<double> doubles = MemoryMarshal.Cast<byte, double>(source);
                for (int i = 0; i < doubles.Length; i++)
                {
                    destination[i] = (float)doubles[i];
                }

                break;
            default:
                throw new NotSupportedException($"Cannot convert {dtype} to float32.");
        }
    }

    private static (Dictionary<string, SafetensorsTensorInfo> Tensors, IReadOnlyDictionary<string, string> Metadata)
        ParseHeader(string path, byte[] header, long dataLength)
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(header);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"'{path}' has an invalid safetensors header: {ex.Message}", ex);
        }

        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException($"'{path}' header is not a JSON object.");
            }

            var tensors = new Dictionary<string, SafetensorsTensorInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

            foreach (JsonProperty entry in doc.RootElement.EnumerateObject())
            {
                if (entry.Name == "__metadata__")
                {
                    if (entry.Value.ValueKind == JsonValueKind.Object)
                    {
                        foreach (JsonProperty m in entry.Value.EnumerateObject())
                        {
                            metadata[m.Name] = m.Value.ValueKind == JsonValueKind.String
                                ? m.Value.GetString() ?? string.Empty
                                : m.Value.ToString();
                        }
                    }

                    continue;
                }

                SafetensorsTensorInfo info = ParseTensor(path, entry.Name, entry.Value, dataLength);
                if (!tensors.TryAdd(info.Name, info))
                {
                    throw new InvalidDataException($"'{path}' lists tensor '{info.Name}' more than once.");
                }
            }

            EnsureNoOverlap(path, tensors.Values);
            return (tensors, metadata);
        }
    }

    private static SafetensorsTensorInfo ParseTensor(string path, string name, JsonElement element, long dataLength)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"'{path}': tensor '{name}' entry is not an object.");
        }

        if (!element.TryGetProperty("dtype", out JsonElement dtypeElement) || dtypeElement.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException($"'{path}': tensor '{name}' has no dtype.");
        }

        SafetensorsDType dtype = ParseDType(name, dtypeElement.GetString()!);

        if (!element.TryGetProperty("shape", out JsonElement shapeElement) || shapeElement.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException($"'{path}': tensor '{name}' has no shape.");
        }

        var shape = new List<long>();
        long elementCount = 1;
        foreach (JsonElement dim in shapeElement.EnumerateArray())
        {
            if (dim.ValueKind != JsonValueKind.Number || !dim.TryGetInt64(out long d) || d < 0)
            {
                throw new InvalidDataException($"'{path}': tensor '{name}' has an invalid dimension.");
            }

            shape.Add(d);
            try
            {
                elementCount = checked(elementCount * d);
            }
            catch (OverflowException)
            {
                throw new InvalidDataException($"'{path}': tensor '{name}' shape overflows.");
            }
        }

        if (!element.TryGetProperty("data_offsets", out JsonElement offsets)
            || offsets.ValueKind != JsonValueKind.Array
            || offsets.GetArrayLength() != 2
            || !offsets[0].TryGetInt64(out long begin)
            || !offsets[1].TryGetInt64(out long end))
        {
            throw new InvalidDataException($"'{path}': tensor '{name}' has invalid data_offsets.");
        }

        if (begin < 0 || end < begin || end > dataLength)
        {
            throw new InvalidDataException(
                $"'{path}': tensor '{name}' data_offsets [{begin}, {end}] fall outside the {dataLength}-byte data section.");
        }

        long expected;
        try
        {
            expected = checked(elementCount * ElementSize(dtype));
        }
        catch (OverflowException)
        {
            throw new InvalidDataException($"'{path}': tensor '{name}' size overflows.");
        }

        if (end - begin != expected)
        {
            throw new InvalidDataException(
                $"'{path}': tensor '{name}' spans {end - begin} bytes but shape [{string.Join(", ", shape)}] "
                + $"of {dtype} needs {expected}.");
        }

        return new SafetensorsTensorInfo(name, dtype, shape.ToArray(), begin, end - begin);
    }

    private static SafetensorsDType ParseDType(string tensorName, string text) => text switch
    {
        "BOOL" => SafetensorsDType.Bool,
        "U8" => SafetensorsDType.U8,
        "I8" => SafetensorsDType.I8,
        "I16" => SafetensorsDType.I16,
        "U16" => SafetensorsDType.U16,
        "F16" => SafetensorsDType.F16,
        "BF16" => SafetensorsDType.BF16,
        "I32" => SafetensorsDType.I32,
        "U32" => SafetensorsDType.U32,
        "F32" => SafetensorsDType.F32,
        "F64" => SafetensorsDType.F64,
        "I64" => SafetensorsDType.I64,
        "U64" => SafetensorsDType.U64,
        _ => throw new NotSupportedException(
            string.Create(CultureInfo.InvariantCulture, $"Tensor '{tensorName}' uses unsupported dtype '{text}'.")),
    };

    private static void EnsureNoOverlap(string path, IEnumerable<SafetensorsTensorInfo> tensors)
    {
        long previousEnd = 0;
        string previousName = string.Empty;
        foreach (SafetensorsTensorInfo t in tensors.Where(t => t.ByteLength > 0).OrderBy(t => t.DataOffset))
        {
            if (t.DataOffset < previousEnd)
            {
                throw new InvalidDataException($"'{path}': tensors '{previousName}' and '{t.Name}' overlap.");
            }

            previousEnd = t.DataOffset + t.ByteLength;
            previousName = t.Name;
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
