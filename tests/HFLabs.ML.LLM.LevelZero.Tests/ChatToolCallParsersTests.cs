using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

public sealed class ChatToolCallParsersTests
{
    [Fact]
    public void Qwen_ParsesJsonToolCall()
    {
        ChatReply r = Qwen3ChatTemplate.Instance.ParseReply(
            "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n</tool_call>");
        Assert.Equal("Let me check.", r.Content);
        ChatToolCall call = Assert.Single(r.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\": \"Paris\", \"days\": 2}", call.ArgumentsJson);
    }

    [Fact]
    public void Qwen_ParsesParallelCalls()
    {
        ChatReply r = Qwen25ChatTemplate.Instance.ParseReply(
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>");
        Assert.Equal(["a", "b"], r.ToolCalls.Select(c => c.Name));
        Assert.Equal(string.Empty, r.Content);
    }

    [Fact]
    public void Qwen_MalformedCallStaysText()
    {
        const string text = "<tool_call>\n{\"name\": \n</tool_call>";
        ChatReply r = Qwen3ChatTemplate.Instance.ParseReply(text);
        Assert.Empty(r.ToolCalls);
        Assert.Equal(text, r.Content);
    }

    [Fact]
    public void Qwen35_ParsesXmlCallWithTypedValues()
    {
        ChatReply r = Qwen35ChatTemplate.Instance.ParseReply(
            "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n</function>\n</tool_call>");
        Assert.Equal("Checking.", r.Content);
        ChatToolCall call = Assert.Single(r.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\":\"Paris\",\"days\":2,\"flag\":true}", call.ArgumentsJson);
    }

    [Fact]
    public void Llama31_ParsesWholeReplyJson()
    {
        ChatReply r = Llama31ChatTemplate.Instance.ParseReply("{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}");
        ChatToolCall call = Assert.Single(r.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\": \"Rome\"}", call.ArgumentsJson);
    }

    [Fact]
    public void Llama31_PlainTextIsNotACall()
    {
        ChatReply r = Llama31ChatTemplate.Instance.ParseReply("It is sunny in Rome.");
        Assert.Empty(r.ToolCalls);
        Assert.Equal("It is sunny in Rome.", r.Content);
    }

    [Fact]
    public void Roundtrip_RenderedCallParsesBack()
    {
        var call = new ChatToolCall("get_weather", "{\"city\": \"Paris\"}");
        string prompt = Qwen3ChatTemplate.Instance.Format([new("user", "w?"), new("assistant", string.Empty, [call])], new ChatTemplateOptions { AddGenerationPrompt = false });
        int start = prompt.IndexOf("<tool_call>", StringComparison.Ordinal);
        ChatReply r = Qwen3ChatTemplate.Instance.ParseReply(prompt[start..]);
        Assert.Equal(call, Assert.Single(r.ToolCalls));
    }
}
