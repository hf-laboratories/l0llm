using System.Text.Json;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>
/// Checks every template against output rendered by the real Hugging Face chat templates (jinja2), including tool
/// calling, thinking mode and multi-turn history. <c>ChatTemplateGoldens.json</c> holds the cases.
/// </summary>
public sealed class ChatTemplatesTests
{
    public static IEnumerable<object[]> Goldens()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "ChatTemplateGoldens.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        foreach (JsonElement c in doc.RootElement.EnumerateArray())
        {
            yield return [c.GetProperty("name").GetString()!, c.GetRawText()];
        }
    }

    private static IChatTemplate Template(string name) => name switch
    {
        "qwen2.5" => Qwen25ChatTemplate.Instance,
        "qwen3" => Qwen3ChatTemplate.Instance,
        "qwen3.5" => Qwen35ChatTemplate.Instance,
        "llama3.1" => Llama31ChatTemplate.Instance,
        "llama3" => Llama3ChatTemplate.Instance,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [Theory]
    [MemberData(nameof(Goldens))]
    public void Matches_HuggingFace_Template(string name, string caseJson)
    {
        _ = name;
        using JsonDocument doc = JsonDocument.Parse(caseJson);
        JsonElement c = doc.RootElement;
        var turns = new List<ChatTurn>();
        foreach (JsonElement m in c.GetProperty("messages").EnumerateArray())
        {
            List<ChatToolCall>? calls = null;
            if (m.TryGetProperty("tool_calls", out JsonElement tcs))
            {
                calls = [];
                foreach (JsonElement tc in tcs.EnumerateArray())
                {
                    JsonElement fn = tc.GetProperty("function");
                    calls.Add(new ChatToolCall(fn.GetProperty("name").GetString()!, fn.GetProperty("arguments").GetRawText()));
                }
            }

            turns.Add(new ChatTurn(m.GetProperty("role").GetString()!, m.GetProperty("content").GetString() ?? string.Empty, calls));
        }

        JsonElement think = c.GetProperty("enableThinking");
        JsonElement date = c.GetProperty("date");
        var options = new ChatTemplateOptions
        {
            AddGenerationPrompt = c.GetProperty("addGenerationPrompt").GetBoolean(),
            EnableThinking = think.ValueKind == JsonValueKind.Null ? null : think.GetBoolean(),
            DateString = date.ValueKind == JsonValueKind.Null ? "30 Sep 2026" : date.GetString(),
            ToolsJson = c.GetProperty("tools").EnumerateArray().Select(t => t.GetRawText()).ToList(),
        };

        string actual = Template(c.GetProperty("template").GetString()!).Format(turns, options);
        Assert.Equal(c.GetProperty("expected").GetString(), actual);
    }

    [Theory]
    [InlineData("qwen2", null, "qwen2.5")]
    [InlineData("qwen3", null, "qwen3")]
    [InlineData("qwen3_5", null, "qwen3.5")]
    [InlineData("qwen3_5_text", null, "qwen3.5")]
    [InlineData("llama", null, "llama3")]
    [InlineData("llama", "default", "llama3")]
    [InlineData("llama", "llama3", "llama3.1")]
    public void ForModelType_PicksTemplate(string modelType, string? rope, string expected) =>
        Assert.Equal(expected, ChatTemplates.ForModelType(modelType, rope).Name);

    [Fact]
    public void ForModelType_UnknownThrows() =>
        Assert.Throws<NotSupportedException>(() => ChatTemplates.ForModelType("mistral"));

    [Fact]
    public void UnsupportedRoleThrows()
    {
        ChatTurn[] turns = [new("critic", "x")];
        Assert.Throws<NotSupportedException>(() => Qwen3ChatTemplate.Instance.Format(turns));
        Assert.Throws<NotSupportedException>(() => Llama3ChatTemplate.Instance.Format(turns));
    }

    [Fact]
    public void Qwen35_RequiresUserQuery() =>
        Assert.Throws<InvalidOperationException>(() => Qwen35ChatTemplate.Instance.Format([new ChatTurn("system", "s")]));

    [Fact]
    public void Llama31_RejectsParallelToolCalls()
    {
        ChatToolCall[] calls = [new("a", "{}"), new("b", "{}")];
        ChatTurn[] turns = [new("user", "hi"), new("assistant", string.Empty, calls)];
        Assert.Throws<InvalidOperationException>(() => Llama31ChatTemplate.Instance.Format(turns));
    }

    [Fact]
    public void StaticQwenTemplate_StillWorks() =>
        Assert.StartsWith("<|im_start|>system\nYou are Qwen", QwenChatTemplate.Format([new ChatTurn("user", "Hi")]), StringComparison.Ordinal);

    [Fact]
    public void PythonJson_MatchesJsonDumps()
    {
        Assert.Equal("{\"a\": [1, 2.5, true, null, \"é\\n\"], \"b\": {}}", PythonJson.Reformat("{\"a\":[1,2.5,true,null,\"é\\n\"],\"b\":{}}"));
        Assert.Equal("{\n  \"a\": [\n    1\n  ]\n}", PythonJson.Reformat("{\"a\":[1]}", indent: 2));
    }
}
