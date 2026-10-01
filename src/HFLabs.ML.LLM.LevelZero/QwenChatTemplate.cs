namespace HFLabs.ML.LLM.LevelZero;

/// <summary>One message of a conversation.</summary>
/// <param name="Role"><c>system</c>, <c>user</c>, <c>assistant</c> or <c>tool</c>.</param>
/// <param name="Content">The message text.</param>
/// <param name="ToolCalls">Functions the assistant called in this message, if any.</param>
public sealed record ChatTurn(string Role, string Content, IReadOnlyList<ChatToolCall>? ToolCalls = null);

/// <summary>
/// Renders conversations in the ChatML format that Qwen2 and Qwen2.5 chat models use. Kept for compatibility;
/// <see cref="ChatTemplates"/> picks the right template for any supported model.
/// </summary>
public static class QwenChatTemplate
{
    /// <summary>The system prompt Qwen2.5 inserts when the conversation does not start with one.</summary>
    public const string DefaultSystemPrompt = Qwen25ChatTemplate.DefaultSystemPrompt;

    /// <summary>Formats <paramref name="messages"/> as a prompt.</summary>
    /// <param name="messages">The conversation, oldest first.</param>
    /// <param name="addGenerationPrompt">Append <c>&lt;|im_start|&gt;assistant</c> so the model writes the next reply.</param>
    /// <exception cref="NotSupportedException">A message has a role other than system, user, assistant or tool.</exception>
    public static string Format(IReadOnlyList<ChatTurn> messages, bool addGenerationPrompt = true) =>
        Qwen25ChatTemplate.Instance.Format(messages, new ChatTemplateOptions { AddGenerationPrompt = addGenerationPrompt });
}
