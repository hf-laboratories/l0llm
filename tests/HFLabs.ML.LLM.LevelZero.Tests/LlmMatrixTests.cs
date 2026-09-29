using LevelZero;
using LevelZero.Kernels;
using Xunit;
using static HFLabs.ML.LLM.LevelZero.Tests.SafetensorsTestFiles;
namespace HFLabs.ML.LLM.LevelZero.Tests;
public sealed class LlmMatrixTests
{
    private static bool HardwareAvailable
    {
        get
        {
            bool available;
            try
            {
                available = LevelZeroRuntime.IsAvailable();
            }
            catch (Exception)
            {
                available = false;
            }
            if (!available && Environment.GetEnvironmentVariable("IPU_L0_REQUIRE_HARDWARE") == "1")
            {
                throw new InvalidOperationException("IPU_L0_REQUIRE_HARDWARE=1 but no Level Zero device was detected.");
            }
            return available;
        }
    }
    private static ushort Bf16(float v) => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
    private static float[] Values(int count, float scale, int seed)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++)
        {
            a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        }
        return a;
    }
    [Fact]
    public void FitsInHalf_RejectsOverflowAndNonFinite()
    {
        Assert.True(LlmMatrix.FitsInHalf([0f, -1f, 64000f, 1e-9f]));
        Assert.False(LlmMatrix.FitsInHalf([1f, 70000f]));
        Assert.False(LlmMatrix.FitsInHalf([1f, float.NaN]));
        Assert.False(LlmMatrix.FitsInHalf([float.NegativeInfinity]));
    }
    [Theory]
    [InlineData(64, 48, true)]
    [InlineData(64, 48, false)]
    [InlineData(30, 17, true)]
    [InlineData(30, 17, false)]
    public void Project_MatchesCpuReference(int outDim, int inDim, bool half)
    {
        if (!HardwareAvailable)
        {
            return;
        }
        float[] w = Values(inDim * outDim, 0.1f, outDim + inDim);
        float[] x = Values(inDim, 1f, 3);
        ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using (device)
        using (var kernels = LlmKernelSuite.Create(device))
        using (LlmMatrix m = LlmMatrix.Upload(device, w, inDim, outDim, half))
        using (SharedBuffer<float> xBuf = device.AllocShared(x))
        using (SharedBuffer<float> yBuf = device.AllocShared<float>(outDim))
        {
            // Width 30 is not a multiple of 4: the matrix must silently stay float32.
            Assert.Equal(half && outDim % 4 == 0, m.IsHalf);
            m.Project(device, kernels, xBuf, yBuf);
            float[] y = yBuf.ToArray();
            for (int col = 0; col < outDim; col++)
            {
                double expected = 0;
                for (int i = 0; i < inDim; i++)
                {
                    float weight = m.IsHalf ? (float)(Half)w[(i * outDim) + col] : w[(i * outDim) + col];
                    expected += (double)x[i] * weight;
                }
                Assert.True(Math.Abs(expected - y[col]) < 1e-4, $"col {col}: expected {expected}, got {y[col]}");
            }
        }
    }
    [Fact]
    public void Half_HalvesTheDeviceBytes()
    {
        if (!HardwareAvailable)
        {
            return;
        }
        float[] w = Values(16 * 8, 1f, 1);
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmMatrix h = LlmMatrix.Upload(device, w, 16, 8, half: true);
        using LlmMatrix f = LlmMatrix.Upload(device, w, 16, 8, half: false);
        Assert.Equal(16 * 8 * 2, h.SizeInBytes);
        Assert.Equal(16 * 8 * 4, f.SizeInBytes);
        Assert.Null(h.Float32Buffer);
        Assert.NotNull(f.Float32Buffer);
    }
    [Theory]
    [InlineData("BF16", LlmWeightPrecision.Auto, true)]
    [InlineData("BF16", LlmWeightPrecision.Float32, false)]
    [InlineData("BF16", LlmWeightPrecision.Float16, true)]
    [InlineData("F32", LlmWeightPrecision.Auto, false)]
    [InlineData("F32", LlmWeightPrecision.Float16, true)]
    public void LoadTransposed_ChoosesStorageByPrecisionAndSourceDtype(string dtype, LlmWeightPrecision precision, bool expectHalf)
    {
        if (!HardwareAvailable)
        {
            return;
        }
        // [out=4, in=2] tensor; the loaded matrix must be its transpose [in=2, out=4].
        float[] hf = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        using var dir = new TempDir();
        Entry entry = dtype == "F32"
            ? F32("w", [4, 2], hf)
            : Raw16("w", "BF16", [4, 2], hf.Select(Bf16).ToArray());
        Write(dir.File("model.safetensors"), [entry]);
        using var checkpoint = SafetensorsCheckpoint.Open(dir.Path);
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmMatrix m = LlmMatrix.LoadTransposed(device, checkpoint, "w", 4, 2, precision);
        Assert.Equal(expectHalf, m.IsHalf);
        Assert.Equal(2, m.InDim);
        Assert.Equal(4, m.OutDim);
        using var kernels = LlmKernelSuite.Create(device);
        using SharedBuffer<float> x = device.AllocShared(new float[] { 1f, 10f });
        using SharedBuffer<float> y = device.AllocShared<float>(4);
        m.Project(device, kernels, x, y);
        // y[o] = 1 * W[o, 0] + 10 * W[o, 1]
        Assert.Equal(new float[] { 21f, 43f, 65f, 87f }, y.ToArray());
    }
    [Fact]
    public void LoadTransposed_AutoKeepsFloat32WhenValuesDoNotFitHalf()
    {
        if (!HardwareAvailable)
        {
            return;
        }
        using var dir = new TempDir();
        Write(dir.File("model.safetensors"), [Raw16("w", "BF16", [4, 2], new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 1e30f }.Select(Bf16).ToArray())]);
        using var checkpoint = SafetensorsCheckpoint.Open(dir.Path);
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmMatrix auto = LlmMatrix.LoadTransposed(device, checkpoint, "w", 4, 2, LlmWeightPrecision.Auto);
        Assert.False(auto.IsHalf);
        _ = Assert.Throws<InvalidDataException>(
            () => LlmMatrix.LoadTransposed(device, checkpoint, "w", 4, 2, LlmWeightPrecision.Float16));
    }}