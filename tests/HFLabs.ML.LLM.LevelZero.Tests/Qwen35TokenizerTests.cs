using System.Text.Json;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>
/// The byte-level BPE tokenizer must also match HuggingFace for the Qwen3.5 vocabulary (248k tokens).
/// References come from <c>golden/gen_tokenizer_golden.py</c> run against Qwen3.5-0.8B. Chat templating is
/// not covered: <see cref="QwenChatTemplate"/> renders the Qwen2.5 format.
/// </summary>
public sealed class Qwen35TokenizerTests
{
    private static readonly string ModelDir =
        Environment.GetEnvironmentVariable("HF_QWEN35_08B_DIR") ?? @"D:\models\Qwen3.5-0.8B";

    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen3.5-tokenizer.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });

    private static readonly Lazy<HfBpeTokenizer?> Tokenizer = new(() =>
    {
        if (!File.Exists(Path.Combine(ModelDir, "tokenizer.json")))
        {
            if (Environment.GetEnvironmentVariable("HF_REQUIRE_MODEL") == "1")
            {
                throw new InvalidOperationException($"HF_REQUIRE_MODEL=1 but no tokenizer.json in '{ModelDir}'.");
            }

            return null;
        }

        return HfBpeTokenizer.Load(ModelDir);
    });

    [Fact]
    public void VocabSize_MatchesHuggingFace()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        Assert.Equal(Golden.Value.GetProperty("vocab_size").GetInt32(), tok.VocabSize);
    }

    [Fact]
    public void Encode_MatchesHuggingFaceIds_ForEveryCase()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        var failures = new List<string>();
        foreach (JsonElement c in Golden.Value.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            int[] expected = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] actual = tok.Encode(text);
            if (!expected.SequenceEqual(actual))
            {
                failures.Add($"'{text.ReplaceLineEndings("\\n")}': expected [{string.Join(",", expected)}...] got [{string.Join(",", actual)}...]");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void Decode_MatchesHuggingFaceText_WithAndWithoutSpecialTokens()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        var failures = new List<string>();
        foreach (JsonElement c in Golden.Value.GetProperty("cases").EnumerateArray())
        {
            int[] ids = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            if (tok.Decode(ids) != c.GetProperty("decoded").GetString()!)
            {
                failures.Add($"decode: {c.GetProperty("text").GetString()!.ReplaceLineEndings("\\n")}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}

