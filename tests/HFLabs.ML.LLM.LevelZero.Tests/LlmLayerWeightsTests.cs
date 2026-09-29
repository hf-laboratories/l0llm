using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class LlmLayerWeightsTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    [InlineData(33, 65)]
    [InlineData(64, 64)]
    [InlineData(70, 5)]
    public void Transpose_MovesElementAtRowColToColRow(int rows, int cols)
    {
        var source = new float[rows * cols];
        for (int i = 0; i < source.Length; i++)
        {
            source[i] = i;
        }

        float[] result = LlmLayerWeights.Transpose(source, rows, cols);

        Assert.Equal(source.Length, result.Length);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                Assert.Equal(source[(r * cols) + c], result[(c * rows) + r]);
            }
        }
    }

    [Fact]
    public void Transpose_Twice_RestoresTheOriginal()
    {
        var rng = new Random(3);
        var source = new float[37 * 91];
        for (int i = 0; i < source.Length; i++)
        {
            source[i] = (float)rng.NextDouble();
        }

        float[] back = LlmLayerWeights.Transpose(LlmLayerWeights.Transpose(source, 37, 91), 91, 37);

        Assert.Equal(source, back);
    }

    [Fact]
    public void Transpose_RejectsWrongLength()
    {
        Assert.Throws<ArgumentException>(() => LlmLayerWeights.Transpose(new float[5], 2, 3));
    }
}
