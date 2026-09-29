"""Record RoPE inverse-frequency tables from HuggingFace transformers for the M7 scaling tests.
Usage:  python gen_rope_golden.py <out.json>
For each case the reference (inv_freq, attention_factor) comes straight from ROPE_INIT_FUNCTIONS.
"""
import json
import sys
import torch
from transformers import Qwen3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

CASES = {
    "default": dict(head_dim=128, theta=1000000.0, max_pos=40960, scaling=None),
    "linear4": dict(head_dim=64, theta=10000.0, max_pos=2048, scaling={"rope_type": "linear", "factor": 4.0}),
    "yarn4_qwen3": dict(
        head_dim=128, theta=1000000.0, max_pos=131072,
        scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}),
    "yarn8_small": dict(
        head_dim=64, theta=10000.0, max_pos=2048,
        scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 2048}),
    "yarn2_explicit": dict(
        head_dim=64, theta=10000.0, max_pos=4096,
        scaling={"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 2048,
                 "beta_fast": 16.0, "beta_slow": 2.0, "attention_factor": 1.25}),
}


def main(out: str) -> None:
    doc = {"torch": torch.__version__, "cases": {}}
    for name, c in CASES.items():
        params = {"rope_theta": c["theta"]}
        if c["scaling"] is not None:
            params.update(c["scaling"])
        else:
            params["rope_type"] = "default"
        cfg = Qwen3Config(
            hidden_size=c["head_dim"] * 8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=c["head_dim"],
            max_position_embeddings=c["max_pos"],
            rope_parameters=params,
        )
        rope_type = params["rope_type"]
        if rope_type == "default":
            inv_freq, attention = Qwen3RotaryEmbedding.compute_default_rope_parameters(cfg, "cpu")
        else:
            inv_freq, attention = ROPE_INIT_FUNCTIONS[rope_type](cfg, "cpu")
        doc["cases"][name] = {
            "head_dim": c["head_dim"],
            "theta": c["theta"],
            "max_position_embeddings": c["max_pos"],
            "scaling": c["scaling"],
            "inv_freq": [float(x) for x in inv_freq.tolist()],
            "attention_factor": float(attention),
        }
        print(name, len(inv_freq), float(attention), float(inv_freq[0]), float(inv_freq[-1]))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
