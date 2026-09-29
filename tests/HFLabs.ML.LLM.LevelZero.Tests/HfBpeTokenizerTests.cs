using System.Text;
using System.Text.Json;
using Xunit;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>
/// The byte-level BPE tokenizer must produce exactly the ids and text HuggingFace produces.
/// References come from <c>golden/gen_tokenizer_golden.py</c>. Needs <c>tokenizer.json</c> from the
/// Qwen2.5 checkpoint; without it the tests pass trivially unless <c>HF_REQUIRE_MODEL=1</c>.
/// </summary>
public sealed class HfBpeTokenizerTests
{
    private static readonly string ModelDir =
        Environment.GetEnvironmentVariable("HF_QWEN25_05B_INSTRUCT_DIR") ?? @"D:\models\Qwen2.5-0.5B-Instruct";

    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-tokenizer.golden.json");
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
                failures.Add($"{Preview(text)}: expected [{string.Join(",", expected.Take(20))}...] got [{string.Join(",", actual.Take(20))}...]");
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
            string all = c.GetProperty("decoded").GetString()!;
            string plain = c.GetProperty("decoded_skip_special").GetString()!;
            if (tok.Decode(ids) != all)
            {
                failures.Add($"decode: {Preview(c.GetProperty("text").GetString()!)}");
            }

            if (tok.Decode(ids, skipSpecialTokens: true) != plain)
            {
                failures.Add($"decode(skip special): {Preview(c.GetProperty("text").GetString()!)}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void Decode_OfPartialMultiByteSequences_MatchesHuggingFace()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        foreach (JsonElement p in Golden.Value.GetProperty("prefix_decodes").EnumerateArray())
        {
            int[] ids = p.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(p.GetProperty("decoded").GetString(), tok.Decode(ids));
        }
    }

    [Fact]
    public void StreamDecoder_ConcatenatesToTheFullDecode_AndNeverSplitsACharacter()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        foreach (JsonElement c in Golden.Value.GetProperty("cases").EnumerateArray())
        {
            int[] ids = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            HfBpeTokenizer.StreamDecoder decoder = tok.CreateStreamDecoder();
            var sb = new StringBuilder();
            foreach (int id in ids)
            {
                string piece = decoder.Append(id);
                Assert.DoesNotContain('\uFFFD', piece);
                _ = sb.Append(piece);
            }

            _ = sb.Append(decoder.Flush());
            Assert.Equal(c.GetProperty("decoded").GetString(), sb.ToString());
        }
    }

    [Fact]
    public void StreamDecoder_HoldsBackIncompleteCharactersUntilTheyFinish()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        const string text = "\U0001F642\U0001F468\u200D\U0001F469";
        int[] ids = tok.Encode(text);
        HfBpeTokenizer.StreamDecoder decoder = tok.CreateStreamDecoder();

        var pieces = ids.Select(decoder.Append).ToList();

        Assert.Contains(string.Empty, pieces); // at least one token ended mid-character
        Assert.All(pieces, p => Assert.DoesNotContain('\uFFFD', p));
        Assert.Equal(text, string.Concat(pieces) + decoder.Flush());
    }
    [Fact]
    public void SpecialTokens_AreRecognisedLiterallyAndLookUpById()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        Assert.True(tok.TryGetAddedTokenId("<|im_end|>", out int imEnd));
        Assert.Equal(151645, imEnd);
        Assert.True(tok.TryGetAddedTokenId("<|endoftext|>", out int eot));
        Assert.Equal(151643, eot);
        Assert.False(tok.TryGetAddedTokenId("<|nope|>", out _));
        Assert.Equal(new[] { 151644 }, tok.Encode("<|im_start|>"));
        Assert.True(tok.IsSpecial(151645));
        Assert.False(tok.IsSpecial(0));
    }

    [Fact]
    public void Decode_IgnoresIdsWithoutAToken()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        int[] ids = tok.Encode("hi");

        Assert.Equal("hi", tok.Decode([.. ids, 151900, -1, 99_999_999]));
    }

    [Fact]
    public void RoundTrip_OfArbitraryUnicode_IsLossless()
    {
        if (Tokenizer.Value is not { } tok)
        {
            return;
        }

        string text = "Grüße 你好 \U0001F642 e\u0301 \t\r\n <|im_end|> end";

        Assert.Equal(text.Normalize(NormalizationForm.FormC), tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void Load_RejectsUnsupportedTokenizerShapes()
    {
        string dir = Path.Combine(Path.GetTempPath(), "mllmlz-tok-" + Guid.NewGuid().ToString("N"));
        _ = Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(
                Path.Combine(dir, "tokenizer.json"),
                """{"normalizer":{"type":"Lowercase"},"pre_tokenizer":null,"model":{"type":"BPE","vocab":{},"merges":[]}}""");
            Assert.Throws<NotSupportedException>(() => HfBpeTokenizer.Load(dir));

            File.WriteAllText(
                Path.Combine(dir, "tokenizer.json"),
                """{"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false},"model":{"type":"WordPiece","vocab":{}}}""");
            Assert.Throws<NotSupportedException>(() => HfBpeTokenizer.Load(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static string Preview(string text)
    {
        string flat = text.Replace("\n", "\\n", StringComparison.Ordinal).Replace("\r", "\\r", StringComparison.Ordinal);
        return flat.Length > 40 ? flat[..40] + "..." : flat;
    }
}
