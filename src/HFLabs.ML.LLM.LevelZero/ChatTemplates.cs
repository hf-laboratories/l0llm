using System.Text;
using System.Text.RegularExpressions;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>Renders a conversation as the prompt text a particular model family was trained on.</summary>
public interface IChatTemplate
{
    /// <summary>Short name of the template, for example <c>qwen2.5</c>.</summary>
    string Name { get; }

    /// <summary>Formats <paramref name="messages"/> as a prompt.</summary>
    /// <param name="messages">The conversation, oldest first.</param>
    /// <param name="addGenerationPrompt">Append the assistant header so the model writes the next reply.</param>
    /// <exception cref="NotSupportedException">A message has a role the template does not support.</exception>
    string Format(IReadOnlyList<ChatTurn> messages, bool addGenerationPrompt = true);
}

/// <summary>Chooses the <see cref="IChatTemplate"/> that matches a checkpoint.</summary>
public static class ChatTemplates
{
    /// <summary>Template for a Hugging Face <c>model_type</c>.</summary>
    /// <exception cref="NotSupportedException">No template exists for the model type.</exception>
    public static IChatTemplate ForModelType(string modelType)
    {
        ArgumentNullException.ThrowIfNull(modelType);
        return modelType switch
        {
            "qwen2" => Qwen25ChatTemplate.Instance,
            "qwen3" or "qwen3_5" or "qwen3_5_text" => Qwen3ChatTemplate.Instance,
            "llama" => Llama3ChatTemplate.Instance,
            _ => throw new NotSupportedException($"No chat template for model_type '{modelType}' (qwen2, qwen3, qwen3_5 and llama have one)."),
        };
    }

    /// <summary>Template for the checkpoint in <paramref name="modelDirectory"/>, chosen from its <c>config.json</c>.</summary>
    public static IChatTemplate ForModelDirectory(string modelDirectory)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        return ForModelType(HfModelConfig.Load(modelDirectory).ModelType);
    }

    internal static void RequireRole(ChatTurn message)
    {
        ArgumentNullException.ThrowIfNull(message);
        if (message.Role is not ("system" or "user" or "assistant"))
        {
            throw new NotSupportedException($"Chat role '{message.Role}' is not supported.");
        }
    }
}

/// <summary>ChatML as used by Qwen2 and Qwen2.5, with the Qwen2.5 default system prompt.</summary>
public sealed class Qwen25ChatTemplate : IChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Qwen25ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "qwen2.5";

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, bool addGenerationPrompt = true) =>
        QwenChatTemplate.Format(messages, addGenerationPrompt);
}

/// <summary>
/// ChatML as used by Qwen3 and Qwen3.5 (no default system prompt; earlier assistant turns lose their
/// <c>&lt;think&gt;</c> block). Thinking stays enabled, which is the models' default; tool calling is not rendered.
/// </summary>
public sealed partial class Qwen3ChatTemplate : IChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Qwen3ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "qwen3";

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var sb = new StringBuilder();
        foreach (ChatTurn message in messages)
        {
            ChatTemplates.RequireRole(message);
            string content = message.Role == "assistant" ? ThinkBlock().Replace(message.Content, string.Empty).TrimStart('\n') : message.Content;
            _ = sb.Append("<|im_start|>").Append(message.Role).Append('\n').Append(content).Append("<|im_end|>\n");
        }

        if (addGenerationPrompt)
        {
            _ = sb.Append("<|im_start|>assistant\n");
        }

        return sb.ToString();
    }

    [GeneratedRegex("<think>.*?</think>", RegexOptions.Singleline)]
    private static partial Regex ThinkBlock();
}

/// <summary>
/// The Llama 3 header format (<c>&lt;|start_header_id|&gt;</c> / <c>&lt;|eot_id|&gt;</c>). Llama 3.1 and 3.2 also
/// inject a dated system header, which this template does not.
/// </summary>
public sealed class Llama3ChatTemplate : IChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Llama3ChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Name => "llama3";

    /// <inheritdoc />
    public string Format(IReadOnlyList<ChatTurn> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var sb = new StringBuilder("<|begin_of_text|>");
        foreach (ChatTurn message in messages)
        {
            ChatTemplates.RequireRole(message);
            _ = sb.Append("<|start_header_id|>").Append(message.Role).Append("<|end_header_id|>\n\n")
                .Append(message.Content.Trim()).Append("<|eot_id|>");
        }

        if (addGenerationPrompt)
        {
            _ = sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        return sb.ToString();
    }
}
