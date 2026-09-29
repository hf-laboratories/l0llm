using System.Text.Json;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class QwenChatTemplateTests
{
    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-tokenizer.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });

    [Fact]
    public void Format_MatchesTheHuggingFaceTemplate_ForEveryRecordedConversation()
    {
        int checkedCount = 0;
        foreach (JsonElement chat in Golden.Value.GetProperty("chats").EnumerateArray())
        {
            var turns = chat.GetProperty("messages").EnumerateArray()
                .Select(m => new ChatTurn(m.GetProperty("role").GetString()!, m.GetProperty("content").GetString()!))
                .ToList();
            bool generationPrompt = !chat.TryGetProperty("no_generation_prompt", out _);

            Assert.Equal(chat.GetProperty("rendered").GetString(), QwenChatTemplate.Format(turns, generationPrompt));
            checkedCount++;
        }

        Assert.Equal(6, checkedCount);
    }

    [Fact]
    public void Format_UsesTheProvidedSystemMessageInsteadOfTheDefault()
    {
        string text = QwenChatTemplate.Format([new ChatTurn("system", "Be brief."), new ChatTurn("user", "Hi")]);

        Assert.Equal("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", text);
        Assert.DoesNotContain("Alibaba", text, StringComparison.Ordinal);
    }

    [Fact]
    public void Format_RejectsToolRoles()
    {
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Format([new ChatTurn("tool", "{}")]));
    }

    [Fact]
    public void Format_OfNoMessages_IsJustTheDefaultSystemTurnAndGenerationPrompt()
    {
        Assert.Equal(
            $"<|im_start|>system\n{QwenChatTemplate.DefaultSystemPrompt}<|im_end|>\n<|im_start|>assistant\n",
            QwenChatTemplate.Format([]));
    }
}
