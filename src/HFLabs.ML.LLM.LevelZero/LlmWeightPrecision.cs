namespace HFLabs.ML.LLM.LevelZero;
/// <summary>How projection matrices are stored on the device.</summary>
public enum LlmWeightPrecision
{
    /// <summary>
    /// Half precision when the checkpoint itself is 16-bit (BF16 or F16) and every value fits, otherwise float32.
    /// A BF16 value has fewer mantissa bits than a half, so this conversion changes nothing that matters.
    /// </summary>
    Auto,
    /// <summary>Always float32 (twice the memory traffic; the reference path).</summary>
    Float32,
    /// <summary>Half precision wherever the shape allows it (output width a multiple of 4).</summary>
    Float16,
    /// <summary>
    /// Int8 weights with per-group scales, multiplied with DP4A: about half the weight traffic of half precision.
    /// Matrices whose shape does not fit (output width a multiple of 4, input a multiple of 32) fall back to the
    /// <see cref="Auto"/> rule. <see cref="LlmModel"/> selects this for <see cref="Auto"/> when the kernels support it.
    /// </summary>
    Int8,
}