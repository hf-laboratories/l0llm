"""Generate golden values for SafetensorsReader tests from Python's reference safetensors + torch.

Usage:  python gen_safetensors_golden.py <checkpoint_dir> <out.json>

For every tensor records dtype, shape, sha256 of the raw stored bytes, the float32 sum and
sum-of-absolutes (double precision) after conversion to float32, and the first and last 8 float32
values. The C# tests recompute these from the same checkpoint.
"""
import hashlib
import json
import sys

import torch
from safetensors import safe_open

DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64"}


def main(directory: str, out: str) -> None:
    path = f"{directory}/model.safetensors"
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            t = f.get_tensor(name).contiguous()
            raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
            flat = t.float().flatten()
            tensors[name] = {
                "dtype": DTYPE_NAMES[t.dtype],
                "shape": list(t.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "sum": float(flat.double().sum()),
                "abs_sum": float(flat.double().abs().sum()),
                "first": flat[:8].tolist(),
                "last": flat[-8:].tolist(),
            }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"source": "Qwen/Qwen2.5-0.5B", "torch": torch.__version__, "tensors": tensors}, fh, indent=1)
    print(f"wrote {len(tensors)} tensors to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
