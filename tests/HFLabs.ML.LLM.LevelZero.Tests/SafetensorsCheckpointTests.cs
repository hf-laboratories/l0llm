using Xunit;
using static HFLabs.ML.LLM.LevelZero.Tests.SafetensorsTestFiles;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class SafetensorsCheckpointTests
{
    [Fact]
    public void Directory_WithSingleFile_OpensIt()
    {
        using var dir = new TempDir();
        Write(dir.File("model.safetensors"), [F32("a", [2], 1f, 2f), F32("b", [1], 3f)]);

        using var ckpt = SafetensorsCheckpoint.Open(dir.Path);

        Assert.Equal(1, ckpt.ShardCount);
        Assert.Equal(2, ckpt.Count);
        Assert.Equal(new[] { 1f, 2f }, ckpt.ReadFloat32("a"));
    }

    [Fact]
    public void FilePath_OpensThatFile()
    {
        using var dir = new TempDir();
        string path = dir.File("custom.safetensors");
        Write(path, [F32("a", [1], 5f)]);

        using var ckpt = SafetensorsCheckpoint.Open(path);

        Assert.Equal(new[] { 5f }, ckpt.ReadFloat32("a"));
    }

    [Fact]
    public void ShardedCheckpoint_MergesShardsUsingIndex()
    {
        using var dir = new TempDir();
        Write(dir.File("model-00001-of-00002.safetensors"), [F32("a", [1], 1f)]);
        Write(dir.File("model-00002-of-00002.safetensors"), [F32("b", [1], 2f), F32("c", [1], 3f)]);
        File.WriteAllText(
            dir.File("model.safetensors.index.json"),
            """
            {"metadata":{"total_size":12},"weight_map":{
              "a":"model-00001-of-00002.safetensors",
              "b":"model-00002-of-00002.safetensors",
              "c":"model-00002-of-00002.safetensors"}}
            """);

        using var ckpt = SafetensorsCheckpoint.Open(dir.Path);

        Assert.Equal(2, ckpt.ShardCount);
        Assert.Equal(3, ckpt.Count);
        Assert.Equal(new[] { 1f }, ckpt.ReadFloat32("a"));
        Assert.Equal(new[] { 3f }, ckpt.ReadFloat32("c"));
        Assert.True(ckpt.Contains("b"));
        Assert.False(ckpt.Contains("z"));
    }

    [Fact]
    public void Index_PointingToShardWithoutTheTensor_IsRejected()
    {
        using var dir = new TempDir();
        Write(dir.File("s1.safetensors"), [F32("a", [1], 1f)]);
        Write(dir.File("s2.safetensors"), [F32("b", [1], 2f)]);
        File.WriteAllText(
            dir.File("model.safetensors.index.json"),
            """{"weight_map":{"a":"s2.safetensors","b":"s2.safetensors"}}""");

        Assert.Throws<InvalidDataException>(() => SafetensorsCheckpoint.Open(dir.Path));
    }

    [Fact]
    public void Index_WithPathTraversal_IsRejected()
    {
        using var dir = new TempDir();
        File.WriteAllText(
            dir.File("model.safetensors.index.json"),
            """{"weight_map":{"a":"../evil.safetensors"}}""");

        Assert.Throws<InvalidDataException>(() => SafetensorsCheckpoint.Open(dir.Path));
    }

    [Fact]
    public void TensorInTwoShards_IsRejected()
    {
        using var dir = new TempDir();
        Write(dir.File("s1.safetensors"), [F32("a", [1], 1f)]);
        Write(dir.File("s2.safetensors"), [F32("a", [1], 2f)]);
        File.WriteAllText(
            dir.File("model.safetensors.index.json"),
            """{"weight_map":{"a":"s1.safetensors","b":"s2.safetensors"}}""");

        Assert.Throws<InvalidDataException>(() => SafetensorsCheckpoint.Open(dir.Path));
    }

    [Fact]
    public void EmptyDirectory_ThrowsFileNotFound()
    {
        using var dir = new TempDir();

        Assert.Throws<FileNotFoundException>(() => SafetensorsCheckpoint.Open(dir.Path));
    }

    [Fact]
    public void MissingPath_ThrowsFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(
            () => SafetensorsCheckpoint.Open(Path.Combine(Path.GetTempPath(), "definitely-not-here-" + Guid.NewGuid())));
    }

    [Fact]
    public void UnknownTensor_ThrowsKeyNotFound_AndTryGetInfoReturnsFalse()
    {
        using var dir = new TempDir();
        Write(dir.File("model.safetensors"), [F32("a", [1], 1f)]);

        using var ckpt = SafetensorsCheckpoint.Open(dir.Path);

        Assert.Throws<KeyNotFoundException>(() => ckpt.ReadFloat32("nope"));
        Assert.False(ckpt.TryGetInfo("nope", out _));
        Assert.True(ckpt.TryGetInfo("a", out SafetensorsTensorInfo? info));
        Assert.Equal(SafetensorsDType.F32, info!.DType);
    }
}
