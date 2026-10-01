using System.Text;
using System.Text.Json;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// Re-serialises JSON the way Python's <c>json.dumps(..., ensure_ascii=False)</c> does (the <c>tojson</c> filter of
/// Hugging Face chat templates): <c>", "</c> and <c>": "</c> separators, or one item per line when indented, and
/// non-ASCII text left as is. Keeps the byte-for-byte prompt format the models were trained on.
/// </summary>
internal static class PythonJson
{
    /// <summary>Formats <paramref name="json"/> like <c>tojson</c>; <paramref name="indent"/> is Jinja's <c>indent</c>.</summary>
    public static string Reformat(string json, int? indent = null)
    {
        ArgumentNullException.ThrowIfNull(json);
        using JsonDocument doc = JsonDocument.Parse(json);
        var sb = new StringBuilder();
        Write(doc.RootElement, sb, indent, 0);
        return sb.ToString();
    }

    /// <summary>Formats a string as a JSON string literal the way Python does.</summary>
    public static string Quote(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        var sb = new StringBuilder(value.Length + 2);
        WriteString(sb, value);
        return sb.ToString();
    }

    internal static void Write(JsonElement element, StringBuilder sb, int? indent, int level)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                WriteContainer(sb, '{', '}', element.EnumerateObject().Count(), indent, level, (i, item) =>
                {
                    JsonProperty p = element.EnumerateObject().ElementAt(i);
                    WriteString(sb, p.Name);
                    _ = sb.Append(": ");
                    Write(p.Value, sb, indent, level + 1);
                });
                break;
            case JsonValueKind.Array:
                WriteContainer(sb, '[', ']', element.GetArrayLength(), indent, level, (i, item) => Write(element[i], sb, indent, level + 1));
                break;
            case JsonValueKind.String:
                WriteString(sb, element.GetString() ?? string.Empty);
                break;
            case JsonValueKind.True:
                _ = sb.Append("true");
                break;
            case JsonValueKind.False:
                _ = sb.Append("false");
                break;
            case JsonValueKind.Null:
                _ = sb.Append("null");
                break;
            default:
                _ = sb.Append(element.GetRawText());
                break;
        }
    }

    private static void WriteContainer(StringBuilder sb, char open, char close, int count, int? indent, int level, Action<int, int> writeItem)
    {
        _ = sb.Append(open);
        if (count == 0)
        {
            _ = sb.Append(close);
            return;
        }

        for (int i = 0; i < count; i++)
        {
            if (i > 0)
            {
                _ = sb.Append(indent is null ? ", " : ",");
            }

            NewLine(sb, indent, level + 1);
            writeItem(i, i);
        }

        NewLine(sb, indent, level);
        _ = sb.Append(close);
    }

    private static void NewLine(StringBuilder sb, int? indent, int level)
    {
        if (indent is { } width)
        {
            _ = sb.Append('\n').Append(' ', width * level);
        }
    }

    private static void WriteString(StringBuilder sb, string value)
    {
        _ = sb.Append('"');
        foreach (char c in value)
        {
            switch (c)
            {
                case '"': _ = sb.Append("\\\""); break;
                case '\\': _ = sb.Append("\\\\"); break;
                case '\n': _ = sb.Append("\\n"); break;
                case '\r': _ = sb.Append("\\r"); break;
                case '\t': _ = sb.Append("\\t"); break;
                case '\b': _ = sb.Append("\\b"); break;
                case '\f': _ = sb.Append("\\f"); break;
                default:
                    if (c < 0x20)
                    {
                        _ = sb.Append("\\u").Append(((int)c).ToString("x4", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        _ = sb.Append(c);
                    }

                    break;
            }
        }

        _ = sb.Append('"');
    }
}
