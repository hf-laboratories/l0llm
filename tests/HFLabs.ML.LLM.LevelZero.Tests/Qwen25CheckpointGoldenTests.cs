using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>
/// Compares the reader against values produced by Python's <c>safetensors</c> + torch
/// (see <c>golden/gen_safetensors_golden.py</c>) on the real Qwen2.5-0.5B checkpoint.
/// The checkpoint directory comes from <c>HF_QWEN25_05B_DIR</c> (default <c>D:\models\Qwen2.5-0.5B</c>).
/// Without the checkpoint the tests pass trivially, unless <c>HF_REQUIRE_MODEL=1</c> is set, in which
/// case they fail so a green run proves the comparison happened.
/// </summary>
public sealed class Qwen25CheckpointGoldenTests
{
    private static readonly string ModelDir =
        Environment.GetEnvironmentVariable("HF_QWEN25_05B_DIR") ?? @"D:\models\Qwen2.5-0.5B";

    private static readonly Lazy<JsonElement> Golden = new(LoadGolden);

    private static bool ModelAvailable()
    {
        bool present = File.Exists(Path.Combine(ModelDir, "model.safetensors"))
            && File.Exists(Path.Combine(ModelDir, "config.json"));
        if (!present && Environment.GetEnvironmentVariable("HF_REQUIRE_MODEL") == "1")
        {
            throw new InvalidOperationException(
                $"HF_REQUIRE_MODEL=1 but no Qwen2.5-0.5B checkpoint at '{ModelDir}'.");
        }

        return present;
    }

    private static JsonElement LoadGolden()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-0.5b.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("tensors").Clone();
    }

    [Fact]
    public void Config_MatchesTheCheckpointAndIsSupported()
    {
        if (!ModelAvailable())
        {
            return;
        }

        HfModelConfig config = HfModelConfig.Load(ModelDir);
        config.EnsureSupported();
        using var ckpt = SafetensorsCheckpoint.Open(ModelDir);

        Assert.Empty(config.ValidateCheckpoint(ckpt));
        Assert.Equal(290, ckpt.Count);
        Assert.Equal(config.ExpectedTensors().Count, ckpt.Count);
    }

    [Fact]
    public void EveryTensor_HasSameDTypeAndShapeAsPython()
    {
        if (!ModelAvailable())
        {
            return;
        }

        using var ckpt = SafetensorsCheckpoint.Open(ModelDir);
        JsonElement golden = Golden.Value;

        Assert.Equal(golden.EnumerateObject().Count(), ckpt.Count);
        foreach (JsonProperty entry in golden.EnumerateObject())
        {
            SafetensorsTensorInfo info = ckpt.GetInfo(entry.Name);
            Assert.Equal(entry.Value.GetProperty("dtype").GetString(), DTypeName(info.DType));
            Assert.Equal(
                entry.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                info.Shape.ToArray());
        }
    }

    [Fact]
    public void EveryTensor_RawBytesHaveSameSha256AsPython()
    {
        if (!ModelAvailable())
        {
            return;
        }

        using var ckpt = SafetensorsCheckpoint.Open(ModelDir);
        foreach (JsonProperty entry in Golden.Value.EnumerateObject())
        {
            string actual = Convert.ToHexString(SHA256.HashData(ckpt.ReadRawBytes(entry.Name))).ToLowerInvariant();
            Assert.True(
                entry.Value.GetProperty("sha256").GetString() == actual,
                $"sha256 mismatch for '{entry.Name}'");
        }
    }

    [Fact]
    public void EveryTensor_Float32ConversionMatchesPython()
    {
        if (!ModelAvailable())
        {
            return;
        }

        using var ckpt = SafetensorsCheckpoint.Open(ModelDir);
        foreach (JsonProperty entry in Golden.Value.EnumerateObject())
        {
            float[] values = ckpt.ReadFloat32(entry.Name);

            double sum = 0;
            double absSum = 0;
            foreach (float v in values)
            {
                sum += v;
                absSum += Math.Abs((double)v);
            }

            double expectedAbs = entry.Value.GetProperty("abs_sum").GetDouble();
            double expectedSum = entry.Value.GetProperty("sum").GetDouble();
            double tolerance = (1e-9 * expectedAbs) + 1e-6;
            Assert.True(Math.Abs(absSum - expectedAbs) <= tolerance, $"abs_sum mismatch for '{entry.Name}'");
            Assert.True(Math.Abs(sum - expectedSum) <= tolerance, $"sum mismatch for '{entry.Name}'");

            AssertEdge(entry.Name, entry.Value.GetProperty("first"), values, fromEnd: false);
            AssertEdge(entry.Name, entry.Value.GetProperty("last"), values, fromEnd: true);
        }
    }

    private static void AssertEdge(string name, JsonElement expected, float[] values, bool fromEnd)
    {
        float[] want = expected.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        int start = fromEnd ? values.Length - want.Length : 0;
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(
                BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(values[start + i]),
                $"'{name}' element {start + i}: expected {want[i]}, got {values[start + i]}");
        }
    }

    private static string DTypeName(SafetensorsDType d) => d switch
    {
        SafetensorsDType.BF16 => "BF16",
        SafetensorsDType.F16 => "F16",
        SafetensorsDType.F32 => "F32",
        SafetensorsDType.F64 => "F64",
        _ => d.ToString(),
    };
}
