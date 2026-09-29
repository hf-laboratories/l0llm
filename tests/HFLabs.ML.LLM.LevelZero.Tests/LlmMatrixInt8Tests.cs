using LevelZero;
using LevelZero.Kernels;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>Int8 (DP4A) storage and projection paths of <see cref="LlmMatrix"/>.</summary>
public sealed class LlmMatrixInt8Tests
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

    private static float[] Values(int count, float scale, int seed)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++)
        {
            a[i] = (float)(((rng.NextDouble() * 2) - 1) * scale);
        }

        return a;
    }

    [Fact]
    public void Int8_StoresQuartersOfFloat32Bytes_PlusScales()
    {
        if (!HardwareAvailable)
        {
            return;
        }

        const int outDim = 64;
        const int inDim = 96;
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmMatrix m = LlmMatrix.FromHuggingFace(device, Values(outDim * inDim, 0.3f, 1), true, "w", outDim, inDim, LlmWeightPrecision.Int8);
        Assert.True(m.IsInt8);
        Assert.False(m.IsHalf);
        Assert.Equal((outDim * inDim) + (inDim / 32 * outDim * 2), m.SizeInBytes);
    }

    [Theory]
    [InlineData(30, 64)] // output width not a multiple of 4
    [InlineData(64, 50)] // input not a multiple of the group size
    public void Int8_FallsBackToHalfForUnsupportedShapes(int outDim, int inDim)
    {
        if (!HardwareAvailable)
        {
            return;
        }

        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmMatrix m = LlmMatrix.FromHuggingFace(device, Values(outDim * inDim, 0.3f, 2), true, "w", outDim, inDim, LlmWeightPrecision.Int8);
        Assert.False(m.IsInt8);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public void Project_MatchesFloatReferenceWithinQuantizationError(int rows)
    {
        if (!HardwareAvailable)
        {
            return;
        }

        const int outDim = 96;
        const int inDim = 160;
        float[] hf = Values(outDim * inDim, 0.2f, 3);
        float[] x = Values(rows * inDim, 1.5f, 4);
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmKernelSuite suite = LlmKernelSuite.Create(device);
        suite.EnsureQ8Scratch(device, 8, inDim);
        using LlmMatrix m = LlmMatrix.FromHuggingFace(device, hf, true, "w", outDim, inDim, LlmWeightPrecision.Int8);
        using SharedBuffer<float> xb = device.AllocShared(x);
        using SharedBuffer<float> yb = device.AllocShared<float>(rows * outDim);
        m.ProjectAny(device, suite, rows, xb, yb);
        float[] y = yb.ToArray();
        double err = 0;
        double norm = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < outDim; j++)
            {
                double truth = 0;
                for (int i = 0; i < inDim; i++)
                {
                    truth += (double)hf[(j * inDim) + i] * x[(r * inDim) + i];
                }

                double d = y[(r * outDim) + j] - truth;
                err += d * d;
                norm += truth * truth;
            }
        }

        double rel = Math.Sqrt(err / norm);
        Assert.True(rel < 0.02, $"relative RMSE {rel:F4}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(6)]
    public void ProjectShared_EqualsSeparateProjections(int rows)
    {
        if (!HardwareAvailable)
        {
            return;
        }

        const int inDim = 128;
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmKernelSuite suite = LlmKernelSuite.Create(device);
        suite.EnsureQ8Scratch(device, 8, inDim);
        int[] outs = [64, 32, 16, 8];
        var matrices = outs.Select((o, i) => LlmMatrix.FromHuggingFace(device, Values(o * inDim, 0.25f, 10 + i), true, "w", o, inDim, LlmWeightPrecision.Int8)).ToArray();
        try
        {
            using SharedBuffer<float> xb = device.AllocShared(Values(rows * inDim, 1f, 99));
            SharedBuffer<float>[] shared = outs.Select(o => device.AllocShared<float>(rows * o)).ToArray();
            SharedBuffer<float>[] separate = outs.Select(o => device.AllocShared<float>(rows * o)).ToArray();
            try
            {
                LlmMatrix.ProjectShared(device, suite, rows, xb, matrices[0], shared[0], matrices[1], shared[1], matrices[2], shared[2], matrices[3], shared[3]);
                for (int i = 0; i < matrices.Length; i++)
                {
                    matrices[i].ProjectAny(device, suite, rows, xb, separate[i]);
                    Assert.Equal(separate[i].ToArray(), shared[i].ToArray());
                }
            }
            finally
            {
                foreach (SharedBuffer<float> b in shared.Concat(separate))
                {
                    b.Dispose();
                }
            }
        }
        finally
        {
            foreach (LlmMatrix m in matrices)
            {
                m.Dispose();
            }
        }
    }
}
