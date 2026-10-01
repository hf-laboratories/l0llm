using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>An assistant reply split into plain text and the function calls it asked for.</summary>
/// <param name="Content">The reply text with any tool-call markup removed.</param>
/// <param name="ToolCalls">The function calls, in order; empty when the model answered in text.</param>
public sealed record ChatReply(string Content, IReadOnlyList<ChatToolCall> ToolCalls)
{
    /// <summary>A reply with no tool calls.</summary>
    public static ChatReply Text(string content) => new(content, []);
}

/// <summary>Reads the tool calls each model family writes, the inverse of what the chat templates render.</summary>
internal static partial class ChatToolCallParsers
{
    /// <summary>Qwen2.5 and Qwen3: <c>&lt;tool_call&gt;{"name": ..., "arguments": {...}}&lt;/tool_call&gt;</c>.</summary>
    public static ChatReply ParseQwenJson(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var calls = new List<ChatToolCall>();
        foreach (Match m in JsonCallPattern().Matches(text))
        {
            try
            {
                using JsonDocument doc = JsonDocument.Parse(m.Groups[1].Value);
                if (doc.RootElement.TryGetProperty("name", out JsonElement name) && name.ValueKind == JsonValueKind.String)
                {
                    calls.Add(new ChatToolCall(name.GetString()!, ArgumentsOf(doc.RootElement, "arguments")));
                }
            }
            catch (JsonException)
            {
                // Malformed call: leave it in the text rather than invent arguments.
            }
        }

        return calls.Count == 0 ? ChatReply.Text(text) : new ChatReply(JsonCallPattern().Replace(text, string.Empty).Trim(), calls);
    }

    /// <summary>Qwen3.5: <c>&lt;tool_call&gt;&lt;function=name&gt;&lt;parameter=p&gt;value&lt;/parameter&gt;...</c>.</summary>
    public static ChatReply ParseQwen35Xml(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var calls = new List<ChatToolCall>();
        foreach (Match m in XmlCallPattern().Matches(text))
        {
            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream))
            {
                writer.WriteStartObject();
                foreach (Match p in ParameterPattern().Matches(m.Groups[2].Value))
                {
                    writer.WritePropertyName(p.Groups[1].Value);
                    WriteValue(writer, p.Groups[2].Value);
                }

                writer.WriteEndObject();
            }

            calls.Add(new ChatToolCall(m.Groups[1].Value, Encoding.UTF8.GetString(stream.ToArray())));
        }

        return calls.Count == 0 ? ChatReply.Text(text) : new ChatReply(XmlCallPattern().Replace(text, string.Empty).Trim(), calls);
    }

    /// <summary>Llama 3.1/3.2: the whole reply is <c>{"name": ..., "parameters": {...}}</c>.</summary>
    public static ChatReply ParseLlama31(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string body = text.Trim();
        if (body.StartsWith("<|python_tag|>", StringComparison.Ordinal))
        {
            body = body["<|python_tag|>".Length..].Trim();
        }

        if (body.EndsWith("<|eom_id|>", StringComparison.Ordinal))
        {
            body = body[..^"<|eom_id|>".Length].Trim();
        }

        if (!body.StartsWith('{'))
        {
            return ChatReply.Text(text);
        }

        try
        {
            using JsonDocument doc = JsonDocument.Parse(body);
            JsonElement root = doc.RootElement;
            if (root.ValueKind == JsonValueKind.Object
                && root.TryGetProperty("name", out JsonElement name) && name.ValueKind == JsonValueKind.String
                && (root.TryGetProperty("parameters", out _) || root.TryGetProperty("arguments", out _)))
            {
                string args = ArgumentsOf(root, root.TryGetProperty("parameters", out _) ? "parameters" : "arguments");
                return new ChatReply(string.Empty, [new ChatToolCall(name.GetString()!, args)]);
            }
        }
        catch (JsonException)
        {
            // Not a call; treat it as text.
        }

        return ChatReply.Text(text);
    }

    private static string ArgumentsOf(JsonElement call, string property)
    {
        if (!call.TryGetProperty(property, out JsonElement args))
        {
            return "{}";
        }

        return args.ValueKind == JsonValueKind.String ? args.GetString() ?? "{}" : args.GetRawText();
    }

    private static void WriteValue(Utf8JsonWriter writer, string raw)
    {
        string value = raw.Trim('\n');
        try
        {
            using JsonDocument doc = JsonDocument.Parse(value.Trim());
            if (doc.RootElement.ValueKind is JsonValueKind.Number or JsonValueKind.True or JsonValueKind.False
                or JsonValueKind.Null or JsonValueKind.Object or JsonValueKind.Array)
            {
                doc.RootElement.WriteTo(writer);
                return;
            }
        }
        catch (JsonException)
        {
            // Plain text: fall through to a string.
        }

        writer.WriteStringValue(value);
    }

    [GeneratedRegex(@"<tool_call>\s*(\{.*?\})\s*</tool_call>", RegexOptions.Singleline)]
    private static partial Regex JsonCallPattern();

    [GeneratedRegex(@"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", RegexOptions.Singleline)]
    private static partial Regex XmlCallPattern();

    [GeneratedRegex(@"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", RegexOptions.Singleline)]
    private static partial Regex ParameterPattern();
}
