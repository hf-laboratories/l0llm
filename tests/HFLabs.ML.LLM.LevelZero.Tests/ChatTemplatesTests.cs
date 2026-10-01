using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class ChatTemplatesTests
{
    private static readonly ChatTurn[] UserOnly = [new("user", "Hi")];

    [Theory]
    [InlineData("qwen2", "qwen2.5")]
    [InlineData("qwen3", "qwen3")]
    [InlineData("qwen3_5", "qwen3")]
    [InlineData("qwen3_5_text", "qwen3")]
    [InlineData("llama", "llama3")]
    public void ForModelType_PicksTemplate(string modelType, string expected) =>
        Assert.Equal(expected, ChatTemplates.ForModelType(modelType).Name);

    [Fact]
    public void ForModelType_UnknownThrows() =>
        Assert.Throws<NotSupportedException>(() => ChatTemplates.ForModelType("mistral"));

    [Fact]
    public void Qwen25_InjectsDefaultSystemPrompt() =>
        Assert.Equal(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
            Qwen25ChatTemplate.Instance.Format(UserOnly));

    [Fact]
    public void Qwen3_HasNoDefaultSystemPrompt() =>
        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", Qwen3ChatTemplate.Instance.Format(UserOnly));

    [Fact]
    public void Qwen3_StripsThinkFromEarlierAssistantTurns()
    {
        ChatTurn[] turns = [new("user", "Hi"), new("assistant", "<think>\nhmm\n</think>\n\nHello"), new("user", "Again")];
        string text = Qwen3ChatTemplate.Instance.Format(turns);
        Assert.DoesNotContain("<think>", text, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>assistant\nHello<|im_end|>\n", text, StringComparison.Ordinal);
    }

    [Fact]
    public void Llama3_UsesHeaderFormat() =>
        Assert.Equal(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            Llama3ChatTemplate.Instance.Format([new("system", "Be brief."), new("user", "Hi")]));

    [Fact]
    public void WithoutGenerationPrompt_EndsAfterLastTurn()
    {
        Assert.EndsWith("<|eot_id|>", Llama3ChatTemplate.Instance.Format(UserOnly, addGenerationPrompt: false), StringComparison.Ordinal);
        Assert.EndsWith("<|im_end|>\n", Qwen3ChatTemplate.Instance.Format(UserOnly, addGenerationPrompt: false), StringComparison.Ordinal);
    }

    [Fact]
    public void UnsupportedRoleThrows()
    {
        ChatTurn[] turns = [new("tool", "x")];
        Assert.Throws<NotSupportedException>(() => Qwen3ChatTemplate.Instance.Format(turns));
        Assert.Throws<NotSupportedException>(() => Llama3ChatTemplate.Instance.Format(turns));
    }
}
