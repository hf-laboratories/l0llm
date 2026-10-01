using System.Globalization;
using System.Text;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>A function call the assistant made.</summary>
/// <param name="Name">The function name.</param>
/// <param name="ArgumentsJson">The arguments as a JSON object.</param>
public sealed record ChatToolCall(string Name, string ArgumentsJson);

/// <summary>Per-request switches for <see cref="IChatTemplate.Format"/>.</summary>
public sealed record ChatTemplateOptions
{
    /// <summary>Append the assistant header so the model writes the next reply.</summary>
    public bool AddGenerationPrompt { get; init; } = true;

    /// <summary>
    /// Thinking mode for Qwen3 and Qwen3.5. Null keeps each model's own default (on for Qwen3, off for Qwen3.5).
    /// </summary>
    public bool? EnableThinking { get; init; }

    /// <summary>Tool definitions, each a JSON object such as <c>{"type":"function","function":{...}}</c>.</summary>
    public IReadOnlyList<string>? ToolsJson { get; init; }

    /// <summary>Date shown in the Llama 3.1/3.2 system header (<c>30 Sep 2026</c>); null means today.</summary>
    public string? DateString { get; init; }
}

/// <summary>Renders a conversation as the prompt text a particular model family was trained on.</summary>
public interface IChatTemplate
{
    /// <summary>Short name of the template, for example <c>qwen2.5</c>.</summary>
    string Name { get; }

    /// <summary>Formats <paramref name="messages"/> as a prompt.</summary>
    /// <param name="messages">The conversation, oldest first.</param>
    /// <param name="options">Generation prompt, thinking and tool switches; null uses the defaults.</param>
    /// <exception cref="NotSupportedException">A message has a role the template does not support.</exception>
    /// <exception cref="InvalidOperationException">The conversation breaks a rule of the template.</exception>
    string Format(IReadOnlyList<ChatTurn> messages, ChatTemplateOptions? options = null);

    /// <summary>Splits model output into text and the function calls it contains (none unless tools were offered).</summary>
    ChatReply ParseReply(string output);
}

/// <summary>Chooses the <see cref="IChatTemplate"/> that matches a checkpoint.</summary>
public static class ChatTemplates
{
    /// <summary>Template for a Hugging Face <c>model_type</c> (and, for Llama, its <c>rope_scaling</c> type).</summary>
    /// <param name="modelType">The <c>model_type</c> from <c>config.json</c>.</param>
    /// <param name="ropeScalingType">
    /// The <c>rope_scaling</c> type; <c>llama3</c> marks Llama 3.1 and later, which add a dated system header.
    /// </param>
    /// <exception cref="NotSupportedException">No template exists for the model type.</exception>
    public static IChatTemplate ForModelType(string modelType, string? ropeScalingType = null)
    {
        ArgumentNullException.ThrowIfNull(modelType);
        return modelType switch
        {
            "qwen2" => Qwen25ChatTemplate.Instance,
            "qwen3" => Qwen3ChatTemplate.Instance,
            "qwen3_5" or "qwen3_5_text" => Qwen35ChatTemplate.Instance,
            "llama" when string.Equals(ropeScalingType, "llama3", StringComparison.Ordinal) => Llama31ChatTemplate.Instance,
            "llama" => Llama3ChatTemplate.Instance,
            _ => throw new NotSupportedException($"No chat template for model_type '{modelType}' (qwen2, qwen3, qwen3_5 and llama have one)."),
        };
    }

    /// <summary>Template for a parsed <c>config.json</c>.</summary>
    public static IChatTemplate ForConfig(HfModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return ForModelType(config.ModelType, config.RopeScalingType);
    }

    /// <summary>Template for the checkpoint in <paramref name="modelDirectory"/>, chosen from its <c>config.json</c>.</summary>
    public static IChatTemplate ForModelDirectory(string modelDirectory)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        return ForConfig(HfModelConfig.Load(modelDirectory));
    }

    internal static void RequireRole(ChatTurn message, bool allowTool)
    {
        ArgumentNullException.ThrowIfNull(message);
        if (message.Role is not ("system" or "user" or "assistant") && !(allowTool && message.Role == "tool"))
        {
            throw new NotSupportedException($"Chat role '{message.Role}' is not supported.");
        }
    }

    internal static IReadOnlyList<string> Tools(ChatTemplateOptions? options) => options?.ToolsJson ?? [];

    internal static bool IsToolResponse(string content) =>
        content.StartsWith("<tool_response>", StringComparison.Ordinal) && content.EndsWith("</tool_response>", StringComparison.Ordinal);

    /// <summary>Index of the last real user query (not a tool response), else the last message.</summary>
    internal static int LastQueryIndex(IReadOnlyList<ChatTurn> messages, bool trim)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == "user" && !IsToolResponse(trim ? messages[i].Content.Trim() : messages[i].Content))
            {
                return i;
            }
        }

        return messages.Count - 1;
    }

    /// <summary>Splits an assistant message into its <c>&lt;think&gt;</c> text and the answer after <c>&lt;/think&gt;</c>.</summary>
    internal static (string Reasoning, string Content) SplitThinking(string content)
    {
        int first = content.IndexOf("</think>", StringComparison.Ordinal);
        if (first < 0)
        {
            return (string.Empty, content);
        }

        string before = content[..first].TrimEnd('\n');
        int open = before.LastIndexOf("<think>", StringComparison.Ordinal);
        string reasoning = (open >= 0 ? before[(open + "<think>".Length)..] : before).TrimStart('\n');
        int last = content.LastIndexOf("</think>", StringComparison.Ordinal);
        return (reasoning, content[(last + "</think>".Length)..].TrimStart('\n'));
    }

    internal static string Today() => DateTime.Now.ToString("dd MMM yyyy", CultureInfo.InvariantCulture);
}

/// <summary>ChatML as used by Qwen2 and Qwen2.5, with the Qwen2.5 default system prompt and tool calling.</summary>
public sealed class Qwen25ChatTemplate : IChatTemplate
{
    /// <summary>The system prompt Qwen2.5 inserts when the conversation does not start with one.</summary>
    public const string DefaultSystemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    /// <summary>The shared instance.</summary>
    public static Qwen25ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "qwen2.5";

    /// <inheritdoc />
    public ChatReply ParseReply(string output) => ChatToolCallParsers.ParseQwenJson(output);

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, ChatTemplateOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        options ??= new ChatTemplateOptions();
        IReadOnlyList<string> tools = ChatTemplates.Tools(options);
        var sb = new StringBuilder();
        bool firstIsSystem = messages.Count > 0 && messages[0].Role == "system";
        string system = firstIsSystem ? messages[0].Content : DefaultSystemPrompt;
        if (tools.Count > 0)
        {
            _ = sb.Append("<|im_start|>system\n").Append(system)
                .Append("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
            foreach (string tool in tools)
            {
                _ = sb.Append('\n').Append(PythonJson.Reformat(tool));
            }

            _ = sb.Append("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
        }
        else
        {
            _ = sb.Append("<|im_start|>system\n").Append(system).Append("<|im_end|>\n");
        }

        for (int i = 0; i < messages.Count; i++)
        {
            ChatTurn m = messages[i];
            ChatTemplates.RequireRole(m, allowTool: true);
            bool hasCalls = m.ToolCalls is { Count: > 0 };
            if (m.Role == "user" || (m.Role == "system" && i > 0) || (m.Role == "assistant" && !hasCalls))
            {
                _ = sb.Append("<|im_start|>").Append(m.Role).Append('\n').Append(m.Content).Append("<|im_end|>\n");
            }
            else if (m.Role == "assistant")
            {
                _ = sb.Append("<|im_start|>assistant");
                if (m.Content.Length > 0)
                {
                    _ = sb.Append('\n').Append(m.Content);
                }

                foreach (ChatToolCall call in m.ToolCalls!)
                {
                    _ = sb.Append("\n<tool_call>\n{\"name\": \"").Append(call.Name).Append("\", \"arguments\": ")
                        .Append(PythonJson.Reformat(call.ArgumentsJson)).Append("}\n</tool_call>");
                }

                _ = sb.Append("<|im_end|>\n");
            }
            else if (m.Role == "tool")
            {
                QwenTools.AppendToolResponse(sb, messages, i, m.Content);
            }
        }

        if (options.AddGenerationPrompt)
        {
            _ = sb.Append("<|im_start|>assistant\n");
        }

        return sb.ToString();
    }
}

/// <summary>
/// ChatML as used by Qwen3: no default system prompt, earlier assistant turns lose their <c>&lt;think&gt;</c> block,
/// JSON tool calls, and an optional non-thinking prefill.
/// </summary>
public sealed class Qwen3ChatTemplate : IChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Qwen3ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "qwen3";

    /// <inheritdoc />
    public ChatReply ParseReply(string output) => ChatToolCallParsers.ParseQwenJson(output);

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, ChatTemplateOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        options ??= new ChatTemplateOptions();
        IReadOnlyList<string> tools = ChatTemplates.Tools(options);
        var sb = new StringBuilder();
        bool firstIsSystem = messages.Count > 0 && messages[0].Role == "system";
        if (tools.Count > 0)
        {
            _ = sb.Append("<|im_start|>system\n");
            if (firstIsSystem)
            {
                _ = sb.Append(messages[0].Content).Append("\n\n");
            }

            _ = sb.Append("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
            foreach (string tool in tools)
            {
                _ = sb.Append('\n').Append(PythonJson.Reformat(tool));
            }

            _ = sb.Append("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
        }
        else if (firstIsSystem)
        {
            _ = sb.Append("<|im_start|>system\n").Append(messages[0].Content).Append("<|im_end|>\n");
        }

        int lastQuery = ChatTemplates.LastQueryIndex(messages, trim: false);
        for (int i = 0; i < messages.Count; i++)
        {
            ChatTurn m = messages[i];
            ChatTemplates.RequireRole(m, allowTool: true);
            if (m.Role == "user" || (m.Role == "system" && i > 0))
            {
                _ = sb.Append("<|im_start|>").Append(m.Role).Append('\n').Append(m.Content).Append("<|im_end|>\n");
            }
            else if (m.Role == "assistant")
            {
                (string reasoning, string content) = ChatTemplates.SplitThinking(m.Content);
                bool last = i == messages.Count - 1;
                if (i > lastQuery && (last || reasoning.Length > 0))
                {
                    _ = sb.Append("<|im_start|>assistant\n<think>\n").Append(reasoning.Trim('\n')).Append("\n</think>\n\n").Append(content.TrimStart('\n'));
                }
                else
                {
                    _ = sb.Append("<|im_start|>assistant\n").Append(content);
                }

                if (m.ToolCalls is { Count: > 0 } calls)
                {
                    for (int c = 0; c < calls.Count; c++)
                    {
                        if ((c == 0 && content.Length > 0) || c > 0)
                        {
                            _ = sb.Append('\n');
                        }

                        _ = sb.Append("<tool_call>\n{\"name\": \"").Append(calls[c].Name).Append("\", \"arguments\": ")
                            .Append(PythonJson.Reformat(calls[c].ArgumentsJson)).Append("}\n</tool_call>");
                    }
                }

                _ = sb.Append("<|im_end|>\n");
            }
            else if (m.Role == "tool")
            {
                QwenTools.AppendToolResponse(sb, messages, i, m.Content);
            }
        }

        if (options.AddGenerationPrompt)
        {
            _ = sb.Append("<|im_start|>assistant\n");
            if (options.EnableThinking == false)
            {
                _ = sb.Append("<think>\n\n</think>\n\n");
            }
        }

        return sb.ToString();
    }
}

/// <summary>
/// ChatML as used by Qwen3.5: trimmed message text, an XML tool-call format
/// (<c>&lt;function=...&gt;&lt;parameter=...&gt;</c>), and thinking switched off unless requested.
/// </summary>
public sealed class Qwen35ChatTemplate : IChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Qwen35ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "qwen3.5";

    /// <inheritdoc />
    public ChatReply ParseReply(string output) => ChatToolCallParsers.ParseQwen35Xml(output);

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, ChatTemplateOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
        {
            throw new InvalidOperationException("No messages provided.");
        }

        options ??= new ChatTemplateOptions();
        IReadOnlyList<string> tools = ChatTemplates.Tools(options);
        var sb = new StringBuilder();
        bool firstIsSystem = messages[0].Role == "system";
        if (tools.Count > 0)
        {
            _ = sb.Append("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
            foreach (string tool in tools)
            {
                _ = sb.Append('\n').Append(PythonJson.Reformat(tool));
            }

            _ = sb.Append("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
            if (firstIsSystem && messages[0].Content.Trim().Length > 0)
            {
                _ = sb.Append("\n\n").Append(messages[0].Content.Trim());
            }

            _ = sb.Append("<|im_end|>\n");
        }
        else if (firstIsSystem)
        {
            _ = sb.Append("<|im_start|>system\n").Append(messages[0].Content.Trim()).Append("<|im_end|>\n");
        }

        int lastQuery = -1;
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == "user" && !ChatTemplates.IsToolResponse(messages[i].Content.Trim()))
            {
                lastQuery = i;
                break;
            }
        }

        if (lastQuery < 0)
        {
            throw new InvalidOperationException("No user query found in messages.");
        }

        for (int i = 0; i < messages.Count; i++)
        {
            ChatTurn m = messages[i];
            ChatTemplates.RequireRole(m, allowTool: true);
            string content = m.Content.Trim();
            if (m.Role == "system")
            {
                if (i > 0)
                {
                    throw new InvalidOperationException("System message must be at the beginning.");
                }
            }
            else if (m.Role == "user")
            {
                _ = sb.Append("<|im_start|>user\n").Append(content).Append("<|im_end|>\n");
            }
            else if (m.Role == "assistant")
            {
                (string reasoningRaw, string answer) = ChatTemplates.SplitThinking(content);
                string reasoning = reasoningRaw.Trim();
                if (i > lastQuery)
                {
                    _ = sb.Append("<|im_start|>assistant\n<think>\n").Append(reasoning).Append("\n</think>\n\n").Append(answer);
                }
                else
                {
                    _ = sb.Append("<|im_start|>assistant\n").Append(answer);
                }

                if (m.ToolCalls is { Count: > 0 } calls)
                {
                    for (int c = 0; c < calls.Count; c++)
                    {
                        if (c == 0 && answer.Trim().Length > 0)
                        {
                            _ = sb.Append("\n\n");
                        }
                        else if (c > 0)
                        {
                            _ = sb.Append('\n');
                        }

                        _ = sb.Append("<tool_call>\n<function=").Append(calls[c].Name).Append(">\n");
                        AppendParameters(sb, calls[c].ArgumentsJson);
                        _ = sb.Append("</function>\n</tool_call>");
                    }
                }

                _ = sb.Append("<|im_end|>\n");
            }
            else if (m.Role == "tool")
            {
                if (i > 0 && messages[i - 1].Role != "tool")
                {
                    _ = sb.Append("<|im_start|>user");
                }

                _ = sb.Append("\n<tool_response>\n").Append(content).Append("\n</tool_response>");
                if (i == messages.Count - 1 || messages[i + 1].Role != "tool")
                {
                    _ = sb.Append("<|im_end|>\n");
                }
            }
        }

        if (options.AddGenerationPrompt)
        {
            _ = sb.Append("<|im_start|>assistant\n");
            _ = sb.Append(options.EnableThinking == true ? "<think>\n" : "<think>\n\n</think>\n\n");
        }

        return sb.ToString();
    }

    private static void AppendParameters(StringBuilder sb, string argumentsJson)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(argumentsJson);
        if (doc.RootElement.ValueKind != System.Text.Json.JsonValueKind.Object)
        {
            return;
        }

        foreach (System.Text.Json.JsonProperty p in doc.RootElement.EnumerateObject())
        {
            string value = p.Value.ValueKind switch
            {
                System.Text.Json.JsonValueKind.Object or System.Text.Json.JsonValueKind.Array => PythonJson.Reformat(p.Value.GetRawText()),
                System.Text.Json.JsonValueKind.String => p.Value.GetString() ?? string.Empty,
                System.Text.Json.JsonValueKind.True => "True",
                System.Text.Json.JsonValueKind.False => "False",
                System.Text.Json.JsonValueKind.Null => "None",
                _ => p.Value.GetRawText(),
            };
            _ = sb.Append("<parameter=").Append(p.Name).Append(">\n").Append(value).Append("\n</parameter>\n");
        }
    }
}

internal static class QwenTools
{
    /// <summary>Qwen2.5 / Qwen3: consecutive tool results share one <c>user</c> turn.</summary>
    public static void AppendToolResponse(StringBuilder sb, IReadOnlyList<ChatTurn> messages, int i, string content)
    {
        if (i == 0 || messages[i - 1].Role != "tool")
        {
            _ = sb.Append("<|im_start|>user");
        }

        _ = sb.Append("\n<tool_response>\n").Append(content).Append("\n</tool_response>");
        if (i == messages.Count - 1 || messages[i + 1].Role != "tool")
        {
            _ = sb.Append("<|im_end|>\n");
        }
    }
}

/// <summary>The Llama 3.0 header format (<c>&lt;|start_header_id|&gt;</c> / <c>&lt;|eot_id|&gt;</c>), no tools.</summary>
public sealed class Llama3ChatTemplate : IChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Llama3ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "llama3";

    /// <inheritdoc />
    public ChatReply ParseReply(string output) => ChatReply.Text(output);

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, ChatTemplateOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        options ??= new ChatTemplateOptions();
        var sb = new StringBuilder("<|begin_of_text|>");
        foreach (ChatTurn message in messages)
        {
            ChatTemplates.RequireRole(message, allowTool: false);
            _ = sb.Append("<|start_header_id|>").Append(message.Role).Append("<|end_header_id|>\n\n")
                .Append(message.Content.Trim()).Append("<|eot_id|>");
        }

        if (options.AddGenerationPrompt)
        {
            _ = sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        return sb.ToString();
    }
}

/// <summary>
/// The Llama 3.1/3.2 template: a dated system header, tool definitions placed in the first user message, one JSON
/// tool call per assistant turn, and tool results under the <c>ipython</c> role.
/// </summary>
public sealed class Llama31ChatTemplate : IChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Llama31ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "llama3.1";

    /// <inheritdoc />
    public ChatReply ParseReply(string output) => ChatToolCallParsers.ParseLlama31(output);

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, ChatTemplateOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        options ??= new ChatTemplateOptions();
        IReadOnlyList<string> tools = ChatTemplates.Tools(options);
        var sb = new StringBuilder("<|begin_of_text|>");
        int start = 0;
        string system = string.Empty;
        if (messages.Count > 0 && messages[0].Role == "system")
        {
            system = messages[0].Content.Trim();
            start = 1;
        }

        _ = sb.Append("<|start_header_id|>system<|end_header_id|>\n\n");
        if (tools.Count > 0)
        {
            _ = sb.Append("Environment: ipython\n");
        }

        _ = sb.Append("Cutting Knowledge Date: December 2023\nToday Date: ").Append(options.DateString ?? ChatTemplates.Today()).Append("\n\n")
            .Append(system).Append("<|eot_id|>");

        if (tools.Count > 0)
        {
            if (start >= messages.Count)
            {
                throw new InvalidOperationException("Cannot put tools in the first user message when there's no first user message!");
            }

            string firstUser = messages[start].Content.Trim();
            start++;
            _ = sb.Append("<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n");
            foreach (string tool in tools)
            {
                _ = sb.Append(PythonJson.Reformat(tool, indent: 4)).Append("\n\n");
            }

            _ = sb.Append(firstUser).Append("<|eot_id|>");
        }

        for (int i = start; i < messages.Count; i++)
        {
            ChatTurn m = messages[i];
            ChatTemplates.RequireRole(m, allowTool: true);
            if (m.ToolCalls is { Count: > 0 } calls)
            {
                if (calls.Count != 1)
                {
                    throw new InvalidOperationException("This model only supports single tool-calls at once!");
                }

                _ = sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"").Append(calls[0].Name).Append("\", \"parameters\": ")
                    .Append(PythonJson.Reformat(calls[0].ArgumentsJson)).Append("}<|eot_id|>");
            }
            else if (m.Role == "tool")
            {
                // Jinja treats a string as iterable, so the template JSON-encodes the result text.
                _ = sb.Append("<|start_header_id|>ipython<|end_header_id|>\n\n").Append(PythonJson.Quote(m.Content)).Append("<|eot_id|>");
            }
            else
            {
                _ = sb.Append("<|start_header_id|>").Append(m.Role).Append("<|end_header_id|>\n\n").Append(m.Content.Trim()).Append("<|eot_id|>");
            }
        }

        if (options.AddGenerationPrompt)
        {
            _ = sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        return sb.ToString();
    }
}
