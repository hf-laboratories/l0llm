# Golden Test Fixtures & Reference Generators

This directory contains **Golden Ground Truth** test data and offline reference generation scripts for the `l0llm` test suite.

---

## 📌 Purpose & Architecture

1. **Zero-Python CI & Runtime**:
   - `l0llm` and its xUnit test suite (`dotnet test`) are **100% pure C# / .NET 10**.
   - Running tests or building the project **does not execute Python** and requires **no Python, Conda, or PyTorch environment**.
   - The test runner reads the static `.golden.json` files directly from disk to verify mathematical parity between our native C# / Level Zero GPU operators and upstream reference baselines.

2. **Offline Python Generator Scripts (`gen_*.py`)**:
   - The `.py` scripts in this directory are optional, offline developer utilities used during development to generate or update the static `.golden.json` baseline files.
   - They run the official Python Hugging Face `transformers` and PyTorch libraries in FP32 on official model checkpoints to extract reference layer tensors, RoPE rotations, BPE tokenizations, and greedy generation tokens.

---

## 📁 File Manifest

| File | Purpose |
| :--- | :--- |
| `gen_generation_golden.py` | Generates greedy token generation sequences and logit margin baselines (`*.generation.golden.json`). |
| `gen_layer_golden.py` | Dumps intermediate tensor outputs across individual Transformer decoder layers (`*.layers.golden.json`). |
| `gen_rope_golden.py` | Computes reference Rotary Position Embedding (RoPE) frequencies and cos/sin caches (`rope.golden.json`). |
| `gen_safetensors_golden.py` | Extracts reference tensor shapes, data types, and byte offsets from `model.safetensors` files (`*.golden.json`). |
| `gen_tokenizer_golden.py` | Validates Byte-Pair Encoding (BPE) tokenization, regex splitting, and decode outputs (`*-tokenizer.golden.json`). |
| `*.golden.json` | The static ground-truth test vectors consumed by C# xUnit tests. |

---

## 🔄 How to Regenerate Golden Fixtures (Optional / Developers Only)

If you are adding support for a new model architecture or updating a checkpoint baseline, you can run the generators in a Python environment with PyTorch and Hugging Face `transformers` installed:

```bash
# Example: Generate generation test vectors for a model checkpoint
python gen_generation_golden.py /path/to/checkpoint qwen2.5-0.5b-instruct.generation.golden.json

# Example: Generate layer-by-layer intermediate verification vectors
python gen_layer_golden.py /path/to/checkpoint qwen2.5-0.5b-instruct.layers.golden.json

# Example: Generate tokenizer verification vectors
python gen_tokenizer_golden.py /path/to/checkpoint qwen2.5-tokenizer.golden.json
```
