using LevelZero;
using LevelZero.Kernels;
namespace HFLabs.ML.LLM.LevelZero;
/// <summary>
/// A projection matrix on the device, stored <c>[in, out]</c> row-major (so <c>y = x * W</c>), as float32 or as half.
/// </summary>
public sealed class LlmMatrix : IDisposable
{
    private readonly SharedBuffer<float>? _f32;
    private readonly SharedBuffer<Half>? _f16;
    private readonly SharedBuffer<int>? _q8Words;
    private readonly SharedBuffer<Half>? _q8Scales;
    private LlmMatrix(int inDim, int outDim, SharedBuffer<float>? f32, SharedBuffer<Half>? f16, SharedBuffer<int>? q8Words = null, SharedBuffer<Half>? q8Scales = null)
    {
        InDim = inDim;
        OutDim = outDim;
        _f32 = f32;
        _f16 = f16;
        _q8Words = q8Words;
        _q8Scales = q8Scales;
    }
    /// <summary>Number of inputs (rows).</summary>
    public int InDim { get; }
    /// <summary>Number of outputs (columns).</summary>
    public int OutDim { get; }
    /// <summary>True when the weights are stored as half.</summary>
    public bool IsHalf => _f16 is not null;
    /// <summary>Device bytes held by the weights.</summary>
    public long SizeInBytes => IsInt8
        ? ((long)InDim * OutDim) + ((long)InDim / LlmGemvQ8Kernel.GroupSize * OutDim * 2)
        : (long)InDim * OutDim * (IsHalf ? 2 : 4);
    /// <summary>True when the weights are stored as int8 with per-group half scales.</summary>
    public bool IsInt8 => _q8Words is not null;
    /// <summary>The float32 storage, or null when the matrix is stored as half. For tests and inspection.</summary>
    public SharedBuffer<float>? Float32Buffer => _f32;
    /// <summary>
    /// Uploads <paramref name="data"/> (already <c>[in, out]</c> row-major).
    /// </summary>
    /// <param name="device">The compute device.</param>
    /// <param name="data">The <c>[inDim * outDim]</c> weights.</param>
    /// <param name="inDim">Number of inputs.</param>
    /// <param name="outDim">Number of outputs.</param>
    /// <param name="half">Store as half when the shape allows it; ignored (float32 is kept) otherwise.</param>
    public static LlmMatrix Upload(ComputeDevice device, float[] data, int inDim, int outDim, bool half)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length != (long)inDim * outDim)
        {
            throw new ArgumentException($"Expected {(long)inDim * outDim} elements, got {data.Length}.", nameof(data));
        }
        if (half && LlmGemvKernel.Supports(outDim))
        {
            return new LlmMatrix(inDim, outDim, null, device.AllocShared(ToHalf(data)));
        }
        return new LlmMatrix(inDim, outDim, device.AllocShared(data), null);
    }
    /// <summary>
    /// Reads a HuggingFace <c>[out, in]</c> tensor, transposes it to <c>[in, out]</c> and uploads it.
    /// </summary>
    /// <exception cref="InvalidDataException"><paramref name="precision"/> is Float16 but a value does not fit a half.</exception>
    internal static LlmMatrix LoadTransposed(
        ComputeDevice device,
        SafetensorsCheckpoint checkpoint,
        string name,
        int outDim,
        int inDim,
        LlmWeightPrecision precision)
    {
        float[] hf = checkpoint.ReadFloat32(name);
        bool sixteenBitSource = checkpoint.TryGetInfo(name, out SafetensorsTensorInfo? info)
            && info is not null
            && info.DType is (SafetensorsDType.BF16 or SafetensorsDType.F16);
        return FromHuggingFace(device, hf, sixteenBitSource, name, outDim, inDim, precision);
    }
    /// <summary>
    /// Transposes HuggingFace <c>[out, in]</c> values that are already in memory and uploads them, applying the
    /// same half-precision rules as <see cref="LoadTransposed"/>.
    /// </summary>
    internal static LlmMatrix FromHuggingFace(
        ComputeDevice device,
        float[] hf,
        bool sixteenBitSource,
        string name,
        int outDim,
        int inDim,
        LlmWeightPrecision precision)
    {
        if (precision == LlmWeightPrecision.Int8 && LlmGemvQ8Kernel.Supports(outDim, inDim) && FitsInHalf(hf))
        {
            // Quantize straight from the HuggingFace [out, in] layout: no transpose needed.
            LlmQ8Quantizer.QuantizeOutIn(hf, outDim, inDim, out int[] words, out Half[] scales);
            return new LlmMatrix(inDim, outDim, null, null, device.AllocShared(words), device.AllocShared(scales));
        }
        bool half = false;
        if (precision != LlmWeightPrecision.Float32)
        {
            bool fits = FitsInHalf(hf);
            if (precision == LlmWeightPrecision.Float16 && !fits)
            {
                throw new InvalidDataException($"Tensor '{name}' has values outside the half-precision range.");
            }
            half = precision == LlmWeightPrecision.Float16 || (sixteenBitSource && fits);
        }
        return Upload(device, LlmLayerWeights.Transpose(hf, outDim, inDim), inDim, outDim, half);
    }    /// <summary>
    /// True when every value converts to a finite half; values below the half range flush towards zero,
    /// which is negligible next to weight magnitudes.
    /// </summary>
    public static bool FitsInHalf(ReadOnlySpan<float> values)
    {
        foreach (float v in values)
        {
            if (!float.IsFinite(v) || MathF.Abs(v) > 65000f)
            {
                return false;
            }
        }
        return true;
    }
    /// <summary>Computes <c>y = x * W</c> for one row.</summary>
    public void Project(ComputeDevice device, LlmKernelSuite suite, SharedBuffer<float> x, SharedBuffer<float> y)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(suite);
        if (_q8Words is not null)
        {
            LlmQ8Activations scratch = RequireScratch(suite, 1);
            suite.GemvQ8!.Quantize(device, 1, InDim, x, scratch);
            suite.GemvQ8.Execute(device, OutDim, InDim, scratch, _q8Words, _q8Scales!, y);
        }
        else if (_f16 is not null)
        {
            suite.Gemv.Execute(device, OutDim, InDim, x, _f16, y);
        }
        else if (LlmGemvKernel.Supports(OutDim))
        {
            suite.Gemv.Execute(device, OutDim, InDim, x, _f32!, y);
        }
        else
        {
            suite.Gemm.Execute(device, 1, OutDim, InDim, x, _f32!, y);
        }
    }
    /// <summary>
    /// Computes <c>Y = X * W</c> for <paramref name="rows"/> rows at once (prefill). Each weight is read once per
    /// <see cref="LlmGemvKernel.MaxRows"/> rows.
    /// </summary>
    public void ProjectRows(ComputeDevice device, LlmKernelSuite suite, int rows, SharedBuffer<float> x, SharedBuffer<float> y)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(suite);
        if (_q8Words is not null)
        {
            LlmQ8Activations scratch = RequireScratch(suite, rows);
            suite.GemvQ8!.Quantize(device, rows, InDim, x, scratch);
            suite.GemvQ8.ExecuteRows(device, rows, OutDim, InDim, scratch, _q8Words, _q8Scales!, y);
        }
        else if (_f16 is not null)
        {
            suite.Gemv.ExecuteRows(device, rows, OutDim, InDim, x, _f16, y);
        }
        else if (LlmGemvKernel.Supports(OutDim))
        {
            suite.Gemv.ExecuteRows(device, rows, OutDim, InDim, x, _f32!, y);
        }
        else
        {
            suite.Gemm.Execute(device, rows, OutDim, InDim, x, _f32!, y);
        }
    }
    /// <summary>
    /// Computes several projections of the same input <paramref name="x"/>. When every matrix is int8 the activations are
    /// quantized once and shared, saving a kernel launch per extra projection; otherwise each matrix projects on its own.
    /// One row uses the GEMV kernels, more rows the multi-row kernels.
    /// </summary>
    public static void ProjectShared(
        ComputeDevice device,
        LlmKernelSuite suite,
        int rows,
        SharedBuffer<float> x,
        LlmMatrix m0,
        SharedBuffer<float> y0,
        LlmMatrix m1,
        SharedBuffer<float> y1,
        LlmMatrix? m2 = null,
        SharedBuffer<float>? y2 = null,
        LlmMatrix? m3 = null,
        SharedBuffer<float>? y3 = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(suite);
        ArgumentNullException.ThrowIfNull(m0);
        ArgumentNullException.ThrowIfNull(m1);
        bool shared = m0.IsInt8 && m1.IsInt8 && m0.InDim == m1.InDim
            && (m2 is null || (m2.IsInt8 && m2.InDim == m0.InDim))
            && (m3 is null || (m3.IsInt8 && m3.InDim == m0.InDim));
        if (shared)
        {
            LlmQ8Activations scratch = RequireScratch(suite, rows);
            suite.GemvQ8!.Quantize(device, rows, m0.InDim, x, scratch);
            m0.ProjectQuantized(device, suite, rows, scratch, y0);
            m1.ProjectQuantized(device, suite, rows, scratch, y1);
            m2?.ProjectQuantized(device, suite, rows, scratch, y2!);
            m3?.ProjectQuantized(device, suite, rows, scratch, y3!);
            return;
        }
        m0.ProjectAny(device, suite, rows, x, y0);
        m1.ProjectAny(device, suite, rows, x, y1);
        if (m2 is not null)
        {
            m2.ProjectAny(device, suite, rows, x, y2!);
        }
        if (m3 is not null)
        {
            m3.ProjectAny(device, suite, rows, x, y3!);
        }
    }
    /// <summary>One row via <see cref="Project"/>, several via <see cref="ProjectRows"/>.</summary>
    public void ProjectAny(ComputeDevice device, LlmKernelSuite suite, int rows, SharedBuffer<float> x, SharedBuffer<float> y)
    {
        if (rows == 1)
        {
            Project(device, suite, x, y);
        }
        else
        {
            ProjectRows(device, suite, rows, x, y);
        }
    }
    private void ProjectQuantized(ComputeDevice device, LlmKernelSuite suite, int rows, LlmQ8Activations scratch, SharedBuffer<float> y)
    {
        if (rows == 1)
        {
            suite.GemvQ8!.Execute(device, OutDim, InDim, scratch, _q8Words!, _q8Scales!, y);
        }
        else
        {
            suite.GemvQ8!.ExecuteRows(device, rows, OutDim, InDim, scratch, _q8Words!, _q8Scales!, y);
        }
    }
    private static LlmQ8Activations RequireScratch(LlmKernelSuite suite, int rows)
    {
        if (suite.GemvQ8 is null || suite.Q8Scratch is not { } scratch)
        {
            throw new InvalidOperationException("Int8 weights need the int8 kernels and activation scratch (see LlmKernelSuite.EnsureQ8Scratch).");
        }
        if (rows > scratch.RowCapacity)
        {
            throw new InvalidOperationException($"Activation scratch holds {scratch.RowCapacity} rows but {rows} were requested.");
        }
        return scratch;
    }    /// <inheritdoc />
    public void Dispose()
    {
        _f32?.Dispose();
        _f16?.Dispose();
        _q8Words?.Dispose();
        _q8Scales?.Dispose();
    }
    private static Half[] ToHalf(float[] source)
    {
        var result = new Half[source.Length];
        const int chunk = 1 << 20;
        int chunks = (source.Length + chunk - 1) / chunk;
        _ = Parallel.For(0, chunks, c =>
        {
            int start = c * chunk;
            int end = Math.Min(start + chunk, source.Length);
            for (int i = start; i < end; i++)
            {
                result[i] = (Half)source[i];
            }
        });
        return result;
    }
}