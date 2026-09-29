using System.Text.Json;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// A HuggingFace checkpoint made of one <c>model.safetensors</c> or several shards described by
/// <c>model.safetensors.index.json</c>. Presents all shards as one name-to-tensor map.
/// </summary>
public sealed class SafetensorsCheckpoint : IDisposable
{
    /// <summary>File name of the single-file checkpoint.</summary>
    public const string SingleFileName = "model.safetensors";

    /// <summary>File name of the shard index.</summary>
    public const string IndexFileName = "model.safetensors.index.json";

    private readonly List<SafetensorsReader> _readers;
    private readonly Dictionary<string, SafetensorsReader> _owner;

    private SafetensorsCheckpoint(List<SafetensorsReader> readers, Dictionary<string, SafetensorsReader> owner)
    {
        _readers = readers;
        _owner = owner;
    }

    /// <summary>Every tensor name in the checkpoint.</summary>
    public IEnumerable<string> Names => _owner.Keys;

    /// <summary>Number of tensors across all shards.</summary>
    public int Count => _owner.Count;

    /// <summary>Number of shard files opened.</summary>
    public int ShardCount => _readers.Count;

    /// <summary>
    /// Opens a checkpoint from a directory (index file first, then <c>model.safetensors</c>) or from a
    /// single <c>.safetensors</c> file path.
    /// </summary>
    /// <exception cref="FileNotFoundException">No checkpoint was found at <paramref name="path"/>.</exception>
    /// <exception cref="InvalidDataException">The index and shards disagree, or a tensor appears in two shards.</exception>
    public static SafetensorsCheckpoint Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        if (File.Exists(path))
        {
            return OpenFiles([path], expected: null);
        }

        if (!Directory.Exists(path))
        {
            throw new FileNotFoundException($"No checkpoint file or directory at '{path}'.", path);
        }

        string index = Path.Combine(path, IndexFileName);
        if (File.Exists(index))
        {
            Dictionary<string, string> weightMap = ReadIndex(index);
            string[] shards = weightMap.Values
                .Distinct(StringComparer.Ordinal)
                .Select(f => Path.Combine(path, f))
                .ToArray();
            return OpenFiles(shards, weightMap);
        }

        string single = Path.Combine(path, SingleFileName);
        if (File.Exists(single))
        {
            return OpenFiles([single], expected: null);
        }

        throw new FileNotFoundException(
            $"'{path}' has neither {IndexFileName} nor {SingleFileName}.", single);
    }

    /// <summary>True when the checkpoint contains <paramref name="name"/>.</summary>
    public bool Contains(string name) => _owner.ContainsKey(name);

    /// <summary>Looks up a tensor's info.</summary>
    public bool TryGetInfo(string name, out SafetensorsTensorInfo? info)
    {
        if (_owner.TryGetValue(name, out SafetensorsReader? reader))
        {
            info = reader.GetInfo(name);
            return true;
        }

        info = null;
        return false;
    }

    /// <summary>Returns a tensor's info or throws <see cref="KeyNotFoundException"/>.</summary>
    public SafetensorsTensorInfo GetInfo(string name) => Owner(name).GetInfo(name);

    /// <summary>Reads a tensor as float32.</summary>
    public float[] ReadFloat32(string name) => Owner(name).ReadFloat32(name);

    /// <summary>Reads a tensor's raw stored bytes.</summary>
    public byte[] ReadRawBytes(string name) => Owner(name).ReadRawBytes(name);

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (SafetensorsReader reader in _readers)
        {
            reader.Dispose();
        }
    }

    private static Dictionary<string, string> ReadIndex(string indexPath)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(indexPath));
        if (doc.RootElement.ValueKind != JsonValueKind.Object
            || !doc.RootElement.TryGetProperty("weight_map", out JsonElement map)
            || map.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"'{indexPath}' has no weight_map object.");
        }

        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (JsonProperty entry in map.EnumerateObject())
        {
            string? file = entry.Value.ValueKind == JsonValueKind.String ? entry.Value.GetString() : null;
            if (string.IsNullOrEmpty(file)
                || file.Contains("..", StringComparison.Ordinal)
                || Path.IsPathRooted(file))
            {
                throw new InvalidDataException($"'{indexPath}' maps '{entry.Name}' to an invalid file name.");
            }

            result[entry.Name] = file;
        }

        return result;
    }

    private static SafetensorsCheckpoint OpenFiles(string[] files, Dictionary<string, string>? expected)
    {
        var readers = new List<SafetensorsReader>();
        var owner = new Dictionary<string, SafetensorsReader>(StringComparer.Ordinal);
        try
        {
            foreach (string file in files)
            {
                var reader = SafetensorsReader.Open(file);
                readers.Add(reader);
                foreach (string name in reader.Tensors.Keys)
                {
                    if (!owner.TryAdd(name, reader))
                    {
                        throw new InvalidDataException(
                            $"Tensor '{name}' appears in both '{owner[name].Path}' and '{file}'.");
                    }
                }
            }

            if (expected is not null)
            {
                foreach ((string name, string file) in expected)
                {
                    if (!owner.TryGetValue(name, out SafetensorsReader? reader)
                        || !string.Equals(Path.GetFileName(reader.Path), Path.GetFileName(file), StringComparison.Ordinal))
                    {
                        throw new InvalidDataException(
                            $"Index says '{name}' is in '{file}', but that shard does not contain it.");
                    }
                }
            }

            return new SafetensorsCheckpoint(readers, owner);
        }
        catch
        {
            foreach (SafetensorsReader r in readers)
            {
                r.Dispose();
            }

            throw;
        }
    }

    private SafetensorsReader Owner(string name) =>
        _owner.TryGetValue(name, out SafetensorsReader? reader)
            ? reader
            : throw new KeyNotFoundException($"Tensor '{name}' is not in the checkpoint.");
}
