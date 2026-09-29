"""Record greedy generations from HuggingFace transformers (float32) for the M4 comparison.

Usage:  python gen_generation_golden.py <checkpoint_dir> <out.json>

Two prompts: a chat-templated question and a raw completion prompt. Generation is pure greedy
(no sampling, no repetition penalty, no early stop) for 32 new tokens. For each step the margin between
the best and second-best logit is stored, so a mismatch on the GPU can be judged against how close
the tie was.
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEW_TOKENS = 32


def main(directory: str, out: str) -> None:
    tok = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    model.generation_config.eos_token_id = None
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.0
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    chat = tok.apply_chat_template(
        [{"role": "user", "content": "Give me a one-sentence fact about the moon."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts = {"chat": chat, "raw": "The capital of France is"}

    doc = {"source": directory, "torch": torch.__version__, "new_tokens": NEW_TOKENS, "cases": {}}
    for name, text in prompts.items():
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            result = model.generate(
                ids,
                max_new_tokens=NEW_TOKENS,
                min_new_tokens=NEW_TOKENS,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tok.pad_token_id,
            )
        generated = result.sequences[0, ids.shape[1]:].tolist()
        margins = []
        for step in result.scores:
            top2 = torch.topk(step[0], 2).values
            margins.append(float(top2[0] - top2[1]))
        doc["cases"][name] = {
            "prompt_text": text,
            "prompt_ids": ids[0].tolist(),
            "generated_ids": generated,
            "generated_text": tok.decode(generated),
            "margins": margins,
        }
        print(name, repr(tok.decode(generated)), "min margin", min(margins))

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
