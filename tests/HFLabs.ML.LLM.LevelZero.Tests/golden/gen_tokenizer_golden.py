"""Record HuggingFace tokenizer behaviour for the C# byte-level BPE tokenizer tests.

Usage:  python gen_tokenizer_golden.py <checkpoint_dir> <out.json>

For each text: the token ids (no special tokens added), decode(ids), and decode(ids, skip_special_tokens=True).
Also records prefix decodes of a multi-byte string (partial UTF-8 handling) and the rendered chat template
for several conversations.
"""
import json, os
import random
import sys

from transformers import AutoTokenizer

TEXTS = [
    "",
    " ",
    "\n",
    "\n\n",
    "Hello, world!",
    "  two leading spaces",
    "trailing space ",
    "tab\there and\r\nCRLF and lone\rCR",
    "don't I'll THEY'RE we've She'S you'd it'm",
    "1234567890 3.14159 1,000,000 x2 0xFF",
    "def f(x):\n    return x**2  # square\n\n\nprint(f(3))",
    "你好，世界！今天天气很好。",
    "こんにちは世界 안녕하세요 세계",
    "Привет, мир! Ελληνικά العربية हिन्दी ภาษาไทย",
    "emoji \U0001F642\U0001F44D\U0001F3FD family \U0001F468‍\U0001F469‍\U0001F467 flag \U0001F1EB\U0001F1F7",
    "café vs café vs Å (angstrom) vs Å",
    "nbsp here ideographic　space em space",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<|endoftext|>",
    "text<|endoftext|>more<|im_end|>",
    "<|im_start| not quite special <|im_end",
    "<|file_sep|>path<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>",
    "{\"key\": [1, 2, 3], \"nested\": {\"a\": null}}",
    "https://example.com/path?query=1&x=%20y#frag",
    "a" * 300,
    "  ".join(["word"] * 80),
    "line1\n\n\n  indented\n\t\ttabbed\n",
    "!!!??? ... --- ~~~ ((())) [[[ ]]] {{{ }}}",
    "The quick brown fox jumps over the lazy dog. " * 12,
]

# A reproducible mix of scripts and punctuation.
rng = random.Random(1234)
ALPHABET = list("abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789\n\t.,;:!?'\"()-_/\\") + list("éüñçøßЖдя日本語한글🙂✓")
TEXTS.append("".join(rng.choice(ALPHABET) for _ in range(2000)))

CHATS = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ],
]


def main(directory: str, out: str) -> None:
    tok = AutoTokenizer.from_pretrained(directory)
    # Raw tokenizer.json is authoritative for ids: transformers may substitute its own pre-tokenizer regex
    # (Qwen3.5 adds \p{M} to the letter class; transformers 5.x splits combining marks off, e.g. Devanagari).
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(os.path.join(directory, "tokenizer.json")) if os.environ.get("RAW_TOKENIZER") == "1" else None
    cases = []
    for text in TEXTS:
        ids = raw.encode(text, add_special_tokens=False).ids if raw else tok(text, add_special_tokens=False)["input_ids"]
        cases.append(
            {
                "text": text,
                "ids": ids,
                "decoded": tok.decode(ids),
                "decoded_skip_special": tok.decode(ids, skip_special_tokens=True),
            }
        )

    emoji_ids = tok("\U0001F642\U0001F468‍\U0001F469", add_special_tokens=False)["input_ids"]
    prefixes = [{"ids": emoji_ids[:k], "decoded": tok.decode(emoji_ids[:k])} for k in range(1, len(emoji_ids) + 1)]

    chats = []
    for messages in CHATS:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chats.append({"messages": messages, "rendered": rendered})
        rendered_no_gen = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        chats.append({"messages": messages, "rendered": rendered_no_gen, "no_generation_prompt": True})

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {"source": directory, "vocab_size": raw.get_vocab_size() if raw else len(tok), "cases": cases, "prefix_decodes": prefixes, "chats": chats},
            fh,
            ensure_ascii=True,
        )
    print(f"{len(cases)} cases, {len(prefixes)} prefixes, {len(chats)} chats")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

