"""Dump decoder-layer inputs, outputs and intermediates from HuggingFace transformers (float32).

Usage:  python gen_layer_golden.py <checkpoint_dir> <out.json> [layer ...]

Runs the full model on a fixed prompt with eager attention and records, for each chosen layer:
the layer input (hidden_states[i]) and output (hidden_states[i + 1]), plus the outputs of the input
norm, q/k/v projections (with bias, before RoPE), the attention block (after o_proj), the
post-attention norm and the MLP. All tensors are float32 arrays [tokens, width], stored
little-endian and base64 encoded. Layers must be below the last layer, because the model's final
hidden state has the output norm applied.
"""
import base64
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The quick brown fox jumps over the lazy dog. The capital of France is"


def enc(t: torch.Tensor) -> dict:
    a = t.detach().to(torch.float32).reshape(t.shape[-2], t.shape[-1]).contiguous().numpy()
    return {"shape": list(a.shape), "data": base64.b64encode(a.astype("<f4").tobytes()).decode("ascii")}


def main(directory: str, out: str, layers: list[int]) -> None:
    tok = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]

    captured: dict[int, dict[str, torch.Tensor]] = {i: {} for i in layers}
    hooks = []
    for i in layers:
        layer = model.model.layers[i]

        def make(idx: int, name: str):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                captured[idx][name] = value.detach().clone()
            return hook

        hooks.append(layer.input_layernorm.register_forward_hook(make(i, "input_norm")))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make(i, "q")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make(i, "k")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make(i, "v")))
        hooks.append(layer.self_attn.register_forward_hook(make(i, "attn_out")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make(i, "post_norm")))
        hooks.append(layer.mlp.register_forward_hook(make(i, "mlp_out")))

    with torch.no_grad():
        result = model(ids, output_hidden_states=True, use_cache=False)
    for h in hooks:
        h.remove()

    hs = result.hidden_states
    n_layers = model.config.num_hidden_layers
    doc = {
        "source": directory,
        "prompt": PROMPT,
        "token_ids": ids[0].tolist(),
        "torch": torch.__version__,
        "layers": {},
        "logits_last": enc(result.logits[:, -1:, :].reshape(1, -1)),
    }
    for i in layers:
        if i >= n_layers - 1:
            raise SystemExit(f"layer {i} is the last layer; its output has the final norm applied")
        entry = {"input": enc(hs[i][0]), "output": enc(hs[i + 1][0])}
        for name, value in captured[i].items():
            entry[name] = enc(value[0])
        doc["layers"][str(i)] = entry

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    print(f"wrote layers {layers}, {len(doc['token_ids'])} tokens to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]] or [0, 11, 22])
