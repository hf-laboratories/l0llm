using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// A byte-level BPE tokenizer (GPT-2 / Qwen2 family) driven by a HuggingFace <c>tokenizer.json</c>.
/// </summary>
/// <remarks>
/// Supports the structure Qwen2, Qwen2.5 and Qwen3 ship: an optional NFC normalizer, a pre-tokenizer
/// that is a regex <c>Split</c> (behavior <c>Isolated</c>) followed by <c>ByteLevel</c>, a BPE model with
/// string merges, and added tokens that are matched literally in the text. Anything else throws
/// <see cref="NotSupportedException"/> at load time instead of tokenizing incorrectly.
/// </remarks>
public sealed class HfBpeTokenizer
{
    private static readonly char[] ByteToChar = BuildByteToChar();
    private static readonly Dictionary<char, byte> CharToByte = BuildCharToByte();

    private readonly Dictionary<string, int> _vocab;
    private readonly string?[] _idToToken;
    private readonly Dictionary<(string Left, string Right), int> _mergeRanks;
    private readonly Regex _splitRegex;
    private readonly bool _nfc;
    private readonly AddedToken[] _addedByLength;
    private readonly Dictionary<int, AddedToken> _addedById;
    private readonly ConcurrentDictionary<string, int[]> _wordCache = new(StringComparer.Ordinal);

    private HfBpeTokenizer(
        Dictionary<string, int> vocab,
        string?[] idToToken,
        Dictionary<(string, string), int> mergeRanks,
        Regex splitRegex,
        bool nfc,
        List<AddedToken> added)
    {
        _vocab = vocab;
        _idToToken = idToToken;
        _mergeRanks = mergeRanks;
        _splitRegex = splitRegex;
        _nfc = nfc;
        _addedByLength = [.. added.OrderByDescending(a => a.Content.Length).ThenBy(a => a.Id)];
        _addedById = added.ToDictionary(a => a.Id);
    }

    /// <summary>Number of token ids (one more than the largest id).</summary>
    public int VocabSize => _idToToken.Length;

    /// <summary>Loads <c>tokenizer.json</c> from a file path or a directory that contains it.</summary>
    /// <exception cref="NotSupportedException">The tokenizer uses features this class does not implement.</exception>
    /// <exception cref="InvalidDataException">The file is malformed.</exception>
    public static HfBpeTokenizer Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string file = Directory.Exists(path) ? Path.Combine(path, "tokenizer.json") : path;
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(file));
        JsonElement root = doc.RootElement;

        bool nfc = ParseNormalizer(root);
        Regex regex = ParsePreTokenizer(root);

        JsonElement model = Require(root, "model");
        if (Str(model, "type") != "BPE")
        {
            throw new NotSupportedException($"Tokenizer model type '{Str(model, "type")}' is not supported (only BPE).");
        }

        if (model.TryGetProperty("byte_fallback", out JsonElement bf) && bf.ValueKind == JsonValueKind.True)
        {
            throw new NotSupportedException("byte_fallback BPE is not supported.");
        }

        if (model.TryGetProperty("continuing_subword_prefix", out JsonElement csp)
            && csp.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(csp.GetString()))
        {
            throw new NotSupportedException("continuing_subword_prefix is not supported.");
        }

        if (model.TryGetProperty("end_of_word_suffix", out JsonElement eow)
            && eow.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(eow.GetString()))
        {
            throw new NotSupportedException("end_of_word_suffix is not supported.");
        }

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        int maxId = -1;
        foreach (JsonProperty entry in Require(model, "vocab").EnumerateObject())
        {
            int id = entry.Value.GetInt32();
            vocab[entry.Name] = id;
            maxId = Math.Max(maxId, id);
        }

        var added = new List<AddedToken>();
        if (root.TryGetProperty("added_tokens", out JsonElement addedArray) && addedArray.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement a in addedArray.EnumerateArray())
            {
                int id = Require(a, "id").GetInt32();
                string content = Str(a, "content");
                bool special = a.TryGetProperty("special", out JsonElement sp) && sp.ValueKind == JsonValueKind.True;
                if (content.Length == 0)
                {
                    throw new InvalidDataException("An added token has empty content.");
                }

                added.Add(new AddedToken(id, content, special));
                maxId = Math.Max(maxId, id);
            }
        }

        var idToToken = new string?[maxId + 1];
        foreach ((string token, int id) in vocab)
        {
            idToToken[id] = token;
        }

        var ranks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (JsonElement merge in Require(model, "merges").EnumerateArray())
        {
            (string left, string right) = ParseMerge(merge);
            _ = ranks.TryAdd((left, right), rank);
            rank++;
        }

        return new HfBpeTokenizer(vocab, idToToken, ranks, regex, nfc, added);
    }

    /// <summary>Looks up an added (special) token by its literal content, such as <c>&lt;|im_end|&gt;</c>.</summary>
    public bool TryGetAddedTokenId(string content, out int id)
    {
        ArgumentNullException.ThrowIfNull(content);
        foreach (AddedToken a in _addedByLength)
        {
            if (string.Equals(a.Content, content, StringComparison.Ordinal))
            {
                id = a.Id;
                return true;
            }
        }

        id = -1;
        return false;
    }

    /// <summary>Encodes text to token ids without adding any special tokens on its own.</summary>
    /// <param name="text">The text. Literal added-token strings (for example <c>&lt;|im_start|&gt;</c>) become their ids.</param>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = new List<int>();
        int segmentStart = 0;
        int i = 0;
        while (i < text.Length)
        {
            AddedToken? match = MatchAddedToken(text, i);
            if (match is null)
            {
                i++;
                continue;
            }

            EncodeSegment(text.AsSpan(segmentStart, i - segmentStart), ids);
            ids.Add(match.Id);
            i += match.Content.Length;
            segmentStart = i;
        }

        EncodeSegment(text.AsSpan(segmentStart), ids);
        return [.. ids];
    }

    /// <summary>Decodes token ids to text. Ids with no token are skipped.</summary>
    /// <param name="ids">Token ids.</param>
    /// <param name="skipSpecialTokens">Leave out added tokens marked special (such as <c>&lt;|im_end|&gt;</c>).</param>
    public string Decode(IReadOnlyList<int> ids, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var bytes = new List<byte>(ids.Count * 3);
        foreach (int id in ids)
        {
            AppendTokenBytes(id, skipSpecialTokens, bytes);
        }

        return Encoding.UTF8.GetString([.. bytes]);
    }

    /// <summary>Creates an incremental decoder that never emits half of a multi-byte character.</summary>
    public StreamDecoder CreateStreamDecoder(bool skipSpecialTokens = false) => new(this, skipSpecialTokens);

    /// <summary>True when <paramref name="id"/> is an added token flagged special.</summary>
    public bool IsSpecial(int id) => _addedById.TryGetValue(id, out AddedToken? a) && a.Special;

    internal void AppendTokenBytes(int id, bool skipSpecialTokens, List<byte> destination)
    {
        if (_addedById.TryGetValue(id, out AddedToken? added))
        {
            if (!(skipSpecialTokens && added.Special))
            {
                destination.AddRange(Encoding.UTF8.GetBytes(added.Content));
            }

            return;
        }

        if ((uint)id >= (uint)_idToToken.Length || _idToToken[id] is not { } token)
        {
            return;
        }

        foreach (char c in token)
        {
            if (CharToByte.TryGetValue(c, out byte b))
            {
                destination.Add(b);
            }
            else
            {
                destination.AddRange(Encoding.UTF8.GetBytes(c.ToString()));
            }
        }
    }

    private AddedToken? MatchAddedToken(string text, int index)
    {
        foreach (AddedToken a in _addedByLength)
        {
            if (a.Content[0] == text[index]
                && string.CompareOrdinal(text, index, a.Content, 0, a.Content.Length) == 0)
            {
                return a;
            }
        }

        return null;
    }

    private void EncodeSegment(ReadOnlySpan<char> segment, List<int> ids)
    {
        if (segment.IsEmpty)
        {
            return;
        }

        string text = segment.ToString();
        if (_nfc)
        {
            text = text.Normalize(NormalizationForm.FormC);
        }

        foreach (string piece in Split(text))
        {
            ids.AddRange(EncodeWord(piece));
        }
    }

    // Split with behavior "Isolated": every regex match is its own piece, and any text between matches too.
    private IEnumerable<string> Split(string text)
    {
        int last = 0;
        foreach (Match m in _splitRegex.Matches(text))
        {
            if (m.Index > last)
            {
                yield return text[last..m.Index];
            }

            if (m.Length > 0)
            {
                yield return m.Value;
            }

            last = m.Index + m.Length;
        }

        if (last < text.Length)
        {
            yield return text[last..];
        }
    }

    private int[] EncodeWord(string piece)
    {
        if (_wordCache.TryGetValue(piece, out int[]? cached))
        {
            return cached;
        }

        byte[] utf8 = Encoding.UTF8.GetBytes(piece);
        var symbols = new List<string>(utf8.Length);
        foreach (byte b in utf8)
        {
            symbols.Add(ByteToChar[b].ToString());
        }

        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_mergeRanks.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                break;
            }

            // Merge every non-overlapping occurrence of the best pair, left to right.
            string left = symbols[bestIndex];
            string right = symbols[bestIndex + 1];
            string merged = left + right;
            var next = new List<string>(symbols.Count);
            for (int i = 0; i < symbols.Count; i++)
            {
                if (i < symbols.Count - 1 && symbols[i] == left && symbols[i + 1] == right)
                {
                    next.Add(merged);
                    i++;
                }
                else
                {
                    next.Add(symbols[i]);
                }
            }

            symbols = next;
        }

        var result = new int[symbols.Count];
        for (int i = 0; i < symbols.Count; i++)
        {
            result[i] = _vocab.TryGetValue(symbols[i], out int id)
                ? id
                : throw new InvalidDataException($"Token '{symbols[i]}' is not in the vocabulary.");
        }

        if (_wordCache.Count < 200_000)
        {
            _wordCache[piece] = result;
        }

        return result;
    }

    private static bool ParseNormalizer(JsonElement root)
    {
        if (!root.TryGetProperty("normalizer", out JsonElement n) || n.ValueKind == JsonValueKind.Null)
        {
            return false;
        }

        string type = Str(n, "type");
        return type switch
        {
            "NFC" => true,
            _ => throw new NotSupportedException($"Normalizer '{type}' is not supported (only NFC or none)."),
        };
    }

    private static Regex ParsePreTokenizer(JsonElement root)
    {
        if (!root.TryGetProperty("pre_tokenizer", out JsonElement pre) || pre.ValueKind == JsonValueKind.Null)
        {
            throw new NotSupportedException("A tokenizer without a pre_tokenizer is not supported.");
        }

        var parts = new List<JsonElement>();
        if (Str(pre, "type") == "Sequence")
        {
            parts.AddRange(Require(pre, "pretokenizers").EnumerateArray());
        }
        else
        {
            parts.Add(pre);
        }

        Regex? regex = null;
        bool sawByteLevel = false;
        foreach (JsonElement part in parts)
        {
            switch (Str(part, "type"))
            {
                case "Split":
                    if (regex is not null || sawByteLevel)
                    {
                        throw new NotSupportedException("Only one Split before ByteLevel is supported.");
                    }

                    if (Str(part, "behavior") != "Isolated"
                        || (part.TryGetProperty("invert", out JsonElement inv) && inv.ValueKind == JsonValueKind.True))
                    {
                        throw new NotSupportedException("Only Split with behavior Isolated and invert=false is supported.");
                    }

                    JsonElement pattern = Require(part, "pattern");
                    if (!pattern.TryGetProperty("Regex", out JsonElement rx))
                    {
                        throw new NotSupportedException("Only regex Split patterns are supported.");
                    }

                    regex = new Regex(rx.GetString()!, RegexOptions.CultureInvariant | RegexOptions.Compiled);
                    break;
                case "ByteLevel":
                    if (part.TryGetProperty("add_prefix_space", out JsonElement aps) && aps.ValueKind == JsonValueKind.True)
                    {
                        throw new NotSupportedException("ByteLevel add_prefix_space=true is not supported.");
                    }

                    sawByteLevel = true;
                    break;
                default:
                    throw new NotSupportedException($"Pre-tokenizer '{Str(part, "type")}' is not supported.");
            }
        }

        if (!sawByteLevel)
        {
            throw new NotSupportedException("A byte-level BPE tokenizer must use the ByteLevel pre-tokenizer.");
        }

        // A lone ByteLevel with use_regex=true applies the GPT-2 pattern.
        return regex ?? new Regex(
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.CultureInvariant | RegexOptions.Compiled);
    }

    private static (string Left, string Right) ParseMerge(JsonElement merge)
    {
        if (merge.ValueKind == JsonValueKind.String)
        {
            string s = merge.GetString()!;
            int space = s.IndexOf(' ', StringComparison.Ordinal);
            if (space <= 0 || space == s.Length - 1)
            {
                throw new InvalidDataException($"Malformed merge '{s}'.");
            }

            return (s[..space], s[(space + 1)..]);
        }

        if (merge.ValueKind == JsonValueKind.Array && merge.GetArrayLength() == 2)
        {
            return (merge[0].GetString()!, merge[1].GetString()!);
        }

        throw new InvalidDataException("Malformed merge entry.");
    }

    private static JsonElement Require(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement e)
            ? e
            : throw new InvalidDataException($"tokenizer.json is missing '{name}'.");

    private static string Str(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.String
            ? e.GetString()!
            : string.Empty;

    // GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to U+0100 and up.
    private static char[] BuildByteToChar()
    {
        var table = new char[256];
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            table[b] = printable ? (char)b : (char)(256 + n++);
        }

        return table;
    }

    private static Dictionary<char, byte> BuildCharToByte()
    {
        var map = new Dictionary<char, byte>(256);
        for (int b = 0; b < 256; b++)
        {
            map[ByteToChar[b]] = (byte)b;
        }

        return map;
    }

    private sealed record AddedToken(int Id, string Content, bool Special);

    /// <summary>
    /// Decodes a token stream piece by piece. Bytes of an unfinished multi-byte character are held back
    /// until the rest arrives, so the output never contains a split character.
    /// </summary>
    public sealed class StreamDecoder
    {
        private readonly HfBpeTokenizer _tokenizer;
        private readonly bool _skipSpecial;
        private readonly List<byte> _pending = [];

        internal StreamDecoder(HfBpeTokenizer tokenizer, bool skipSpecial)
        {
            _tokenizer = tokenizer;
            _skipSpecial = skipSpecial;
        }

        /// <summary>Adds one token and returns the text that became complete (possibly empty).</summary>
        public string Append(int id)
        {
            _tokenizer.AppendTokenBytes(id, _skipSpecial, _pending);
            int complete = CompletePrefixLength(_pending);
            if (complete == 0)
            {
                return string.Empty;
            }

            string text = Encoding.UTF8.GetString([.. _pending.Take(complete)]);
            _pending.RemoveRange(0, complete);
            return text;
        }

        /// <summary>Returns whatever is still held back (an unfinished character becomes U+FFFD).</summary>
        public string Flush()
        {
            string text = Encoding.UTF8.GetString([.. _pending]);
            _pending.Clear();
            return text;
        }

        // Length of the prefix that does not end in the middle of a valid multi-byte sequence.
        private static int CompletePrefixLength(List<byte> bytes)
        {
            int n = bytes.Count;
            for (int back = 1; back <= Math.Min(3, n); back++)
            {
                byte b = bytes[n - back];
                if ((b & 0xC0) == 0x80)
                {
                    continue; // continuation byte, keep looking for its lead
                }

                int needed = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
                return needed > back ? n - back : n;
            }

            return n;
        }
    }
}
