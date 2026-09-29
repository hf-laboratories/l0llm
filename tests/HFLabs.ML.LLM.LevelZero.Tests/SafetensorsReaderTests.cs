using Xunit;
using static HFLabs.ML.LLM.LevelZero.Tests.SafetensorsTestFiles;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class SafetensorsReaderTests
{
    [Fact]
    public void F32_RoundTripsShapeAndValues()
    {
        using var dir = new TempDir();
        string path = dir.File("a.safetensors");
        Write(path, [F32("w", [2, 3], 1f, -2.5f, 3f, 4f, 5.25f, -6f)]);

        using var reader = SafetensorsReader.Open(path);

        SafetensorsTensorInfo info = reader.GetInfo("w");
        Assert.Equal(SafetensorsDType.F32, info.DType);
        Assert.Equal(new long[] { 2, 3 }, info.Shape);
        Assert.Equal(6, info.ElementCount);
        Assert.Equal(new[] { 1f, -2.5f, 3f, 4f, 5.25f, -6f }, reader.ReadFloat32("w"));
    }

    [Fact]
    public void Bf16_ConvertsByShiftingIntoFloatHighBits()
    {
        using var dir = new TempDir();
        string path = dir.File("bf.safetensors");
        // 0x3F80 = 1.0, 0xC000 = -2.0, 0x3FC0 = 1.5, 0x0000 = 0, 0x7F80 = +inf
        Write(path, [Raw16("t", "BF16", [5], 0x3F80, 0xC000, 0x3FC0, 0x0000, 0x7F80)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Equal(new[] { 1f, -2f, 1.5f, 0f, float.PositiveInfinity }, reader.ReadFloat32("t"));
    }

    [Fact]
    public void F16_ConvertsToFloat()
    {
        using var dir = new TempDir();
        string path = dir.File("h.safetensors");
        // 0x3C00 = 1.0, 0xC000 = -2.0, 0x3800 = 0.5, 0x7BFF = 65504 (max half)
        Write(path, [Raw16("t", "F16", [4], 0x3C00, 0xC000, 0x3800, 0x7BFF)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Equal(new[] { 1f, -2f, 0.5f, 65504f }, reader.ReadFloat32("t"));
    }

    [Fact]
    public void F64_NarrowsToFloat()
    {
        using var dir = new TempDir();
        string path = dir.File("d.safetensors");
        Write(path, [F64("t", [2], 0.25, -1e10)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Equal(new[] { 0.25f, -1e10f }, reader.ReadFloat32("t"));
    }

    [Fact]
    public void ScalarTensor_HasOneElementAndEmptyShape()
    {
        using var dir = new TempDir();
        string path = dir.File("s.safetensors");
        Write(path, [F32("s", [], 7f)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Empty(reader.GetInfo("s").Shape);
        Assert.Equal(new[] { 7f }, reader.ReadFloat32("s"));
    }

    [Fact]
    public void MultipleTensors_AreReadIndependently()
    {
        using var dir = new TempDir();
        string path = dir.File("m.safetensors");
        Write(path, [F32("a", [2], 1f, 2f), Raw16("b", "BF16", [2], 0x3F80, 0x4000), F32("c", [1], 9f)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Equal(3, reader.Tensors.Count);
        Assert.Equal(new[] { 9f }, reader.ReadFloat32("c"));
        Assert.Equal(new[] { 1f, 2f }, reader.ReadFloat32("b"));
        Assert.Equal(new[] { 1f, 2f }, reader.ReadFloat32("a"));
    }

    [Fact]
    public void LargeBf16Tensor_ConvertsCorrectlyAcrossChunkBoundaries()
    {
        // 1 MiB chunk = 524288 BF16 elements; 700001 elements crosses one boundary with a ragged tail.
        const int count = 700_001;
        var bits = new ushort[count];
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            bits[i] = (ushort)(0x3F80 + (i % 0x0400)); // 1.0 upward, finite positive values
            expected[i] = BitConverter.Int32BitsToSingle(bits[i] << 16);
        }

        using var dir = new TempDir();
        string path = dir.File("big.safetensors");
        Write(path, [Raw16("t", "BF16", [count], bits)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Equal(expected, reader.ReadFloat32("t"));
    }

    [Fact]
    public void ReadRawBytes_ReturnsStoredBytesUnchanged()
    {
        using var dir = new TempDir();
        string path = dir.File("r.safetensors");
        Entry e = Raw16("t", "BF16", [3], 0x1234, 0xABCD, 0xFFFF);
        Write(path, [e]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Equal(e.Data, reader.ReadRawBytes("t"));
    }

    [Fact]
    public void Metadata_IsExposed()
    {
        using var dir = new TempDir();
        string path = dir.File("meta.safetensors");
        Write(path, [F32("x", [1], 1f)], new Dictionary<string, string> { ["format"] = "pt" });

        using var reader = SafetensorsReader.Open(path);

        Assert.Equal("pt", reader.Metadata["format"]);
        Assert.Single(reader.Tensors);
    }

    [Fact]
    public void IntegerTensor_CannotBeReadAsFloat_ButRawBytesWork()
    {
        using var dir = new TempDir();
        string path = dir.File("i.safetensors");
        Write(path, [new Entry("ids", "I64", [2], new byte[16])]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Throws<NotSupportedException>(() => reader.ReadFloat32("ids"));
        Assert.Equal(16, reader.ReadRawBytes("ids").Length);
    }

    [Fact]
    public void UnknownDType_IsRejectedOnOpen()
    {
        using var dir = new TempDir();
        string path = dir.File("f8.safetensors");
        Write(path, [new Entry("t", "F8_E4M3", [4], new byte[4])]);

        Assert.Throws<NotSupportedException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void MissingTensor_ThrowsKeyNotFound()
    {
        using var dir = new TempDir();
        string path = dir.File("k.safetensors");
        Write(path, [F32("a", [1], 1f)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Throws<KeyNotFoundException>(() => reader.ReadFloat32("nope"));
    }

    [Fact]
    public void WrongSizedDestination_Throws()
    {
        using var dir = new TempDir();
        string path = dir.File("dst.safetensors");
        Write(path, [F32("a", [4], 1f, 2f, 3f, 4f)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Throws<ArgumentException>(() => reader.ReadFloat32("a", new float[3]));
    }

    [Fact]
    public void ReadAfterDispose_Throws()
    {
        using var dir = new TempDir();
        string path = dir.File("disp.safetensors");
        Write(path, [F32("a", [1], 1f)]);
        var reader = SafetensorsReader.Open(path);
        reader.Dispose();

        Assert.Throws<ObjectDisposedException>(() => reader.ReadFloat32("a"));
    }

    [Fact]
    public void DataOffsetsBeyondFile_AreRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("trunc.safetensors");
        WriteRaw(path, """{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}""", new byte[8]);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void ByteLengthNotMatchingShape_IsRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("len.safetensors");
        WriteRaw(path, """{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}""", new byte[8]);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void OverlappingTensors_AreRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("overlap.safetensors");
        WriteRaw(
            path,
            """{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}""",
            new byte[12]);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void DuplicateTensorName_IsRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("dup.safetensors");
        WriteRaw(
            path,
            """{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}""",
            new byte[8]);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void HeaderLengthLargerThanFile_IsRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("hdr.safetensors");
        File.WriteAllBytes(path, [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 1, 2, 3]);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void FileShorterThanEightBytes_IsRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("tiny.safetensors");
        File.WriteAllBytes(path, [1, 2, 3]);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void HeaderThatIsNotJson_IsRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("junk.safetensors");
        WriteRaw(path, "not json at all", []);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void NegativeDimension_IsRejected()
    {
        using var dir = new TempDir();
        string path = dir.File("neg.safetensors");
        WriteRaw(path, """{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}""", []);

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.Open(path));
    }

    [Fact]
    public void EmptyTensor_IsAllowed()
    {
        using var dir = new TempDir();
        string path = dir.File("empty.safetensors");
        Write(path, [F32("e", [0]), F32("a", [1], 3f)]);

        using var reader = SafetensorsReader.Open(path);

        Assert.Empty(reader.ReadFloat32("e"));
        Assert.Equal(new[] { 3f }, reader.ReadFloat32("a"));
    }

    [Fact]
    public void ElementSize_MatchesDTypeWidths()
    {
        Assert.Equal(1, SafetensorsReader.ElementSize(SafetensorsDType.U8));
        Assert.Equal(2, SafetensorsReader.ElementSize(SafetensorsDType.BF16));
        Assert.Equal(2, SafetensorsReader.ElementSize(SafetensorsDType.F16));
        Assert.Equal(4, SafetensorsReader.ElementSize(SafetensorsDType.F32));
        Assert.Equal(8, SafetensorsReader.ElementSize(SafetensorsDType.F64));
    }
}
