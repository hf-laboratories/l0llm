using System.Text;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>One message of a conversation.</summary>
/// <param name="Role"><c>system</c>, <c>user</c> or <c>assistant</c>.</param>
/// <param name="Content">The message text.</param>
public sealed record ChatTurn(string Role, string Content);

/// <summary>
/// Renders conversations in the ChatML format that Qwen2 and Qwen2.5 chat models use, matching the
/// <c>chat_template</c> shipped with Qwen2.5-Instruct (without tool calling).
/// </summary>
public static class QwenChatTemplate
{
    /// <summary>The system prompt Qwen2.5 inserts when the conversation does not start with one.</summary>
    public const string DefaultSystemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    /// <summary>Formats <paramref name="messages"/> as a prompt.</summary>
    /// <param name="messages">The conversation, oldest first.</param>
    /// <param name="addGenerationPrompt">Append <c>&lt;|im_start|&gt;assistant</c> so the model writes the next reply.</param>
    /// <exception cref="NotSupportedException">A message has a role other than system, user or assistant.</exception>
    public static string Format(IReadOnlyList<ChatTurn> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var sb = new StringBuilder();
        bool startsWithSystem = messages.Count > 0 && messages[0].Role == "system";
        if (!startsWithSystem)
        {
            AppendTurn(sb, "system", DefaultSystemPrompt);
        }

        foreach (ChatTurn message in messages)
        {
            ArgumentNullException.ThrowIfNull(message);
            if (message.Role is not ("system" or "user" or "assistant"))
            {
                throw new NotSupportedException($"Chat role '{message.Role}' is not supported.");
            }

            AppendTurn(sb, message.Role, message.Content);
        }

        if (addGenerationPrompt)
        {
            _ = sb.Append("<|im_start|>assistant\n");
        }

        return sb.ToString();
    }

    private static void AppendTurn(StringBuilder sb, string role, string content) =>
        sb.Append("<|im_start|>").Append(role).Append('\n').Append(content).Append("<|im_end|>\n");
}
