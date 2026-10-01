l0llm package - local Qwen models on the Intel Iris Xe GPU (Level Zero)
=======================================================================

What this is
  A self-contained build of the native Level Zero LLM backend from IPUServices (milestones M0-M9) plus a small
  command-line host, l0llm.exe. It runs Qwen2.5-0.5B-Instruct, Qwen3-0.6B and Qwen3.5-0.8B on the integrated GPU
  with int8 (DP4A) weights by default. No .NET install is needed.

Target machine
  Windows x64 with an Intel Iris Xe (Tiger Lake, tgllp) GPU and the Intel graphics driver. Built and tested on
  driver 32.0.101.6737 with an i5-1135G7; use the same driver build or newer. About 6 GB of free RAM is needed for the
  largest model (Qwen3.5-0.8B).

Quick start
  1. Copy this whole folder anywhere (a path without special characters is best).
  2. Double-click check.cmd (or run it in a terminal). It prints the GPU, runs a short generation and a benchmark.
     Expected on the reference machine: "Level Zero: available", "OK", roughly 16-20 ms/token decode (50-60 tokens/s)
     for Qwen2.5-0.5B-Instruct. Timings move by 10-20% with background load.
  3. Chat:  chat.cmd                       (Qwen2.5-0.5B-Instruct)
            chat.cmd Qwen3-0.6B            (any folder name under models\)
            chat.cmd Qwen3.5-0.8B

Commands (l0llm.cmd sets the models folder for you)
  l0llm info                               GPU and the models found under models\
  l0llm check  [--model DIR]               smoke test
  l0llm chat   [--model DIR] [--system "text"] [--temp 0.7] [--max-new 512]
  l0llm run    --model DIR --prompt "text" [--raw] [--temp 0]
  l0llm bench  [--model DIR] [--precision int8|fp16|fp32]
  Options for all: --max-seq N (KV cache, default 4096, hard limit 15000), --top-p, --top-k, --repeat.
  In chat, /reset clears the conversation and /exit quits.

Notes
  - Weights are int8 by default (about half the memory traffic of fp16, roughly 1.7-1.9x faster decode). Use
    --precision fp16 for the exact half-precision path that matches HuggingFace transformers token for token.
  - The first run extracts the GPU shim to %TEMP%\LevelZero.NET (a few MB) and loading a model takes 15-40 s.
  - The chat template is chosen from the model's model_type: Qwen2.5 ChatML, Qwen3/Qwen3.5 ChatML, or Llama 3
    headers. Tool calling is not rendered, and Llama 3.1/3.2 do not get their dated default system prompt.
  - Text only; Qwen3.5 vision and multi-token-prediction tensors are ignored.

If something fails
  "Level Zero: NOT available"     install or update the Intel graphics driver (it provides the Level Zero loader).
                                  To check your GPU and driver on their own, use l0check:
                                  https://github.com/hf-laboratories/l0check
  "zeMemAllocShared failed"       not enough shared GPU memory: close other GPU-heavy apps, use a smaller model,
                                  or pass --max-seq 1024.
  Anything else                   run  l0llm info  and keep its output together with the error text.

Contents
  app\      published l0llm.exe and its dependencies (win-x64, self-contained)
  models\   Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Qwen3.5-0.8B (HuggingFace safetensors, unmodified)
  SHA256SUMS.txt   checksums of every file; verify a file after copying with:  certutil -hashfile <file> SHA256

Source
  IPUServices repo, branch main: backend M0-M9 at a59585e, l0llm host at 1283549.
  Design doc: "Level Zero backend for local HuggingFace models: design plan" (Claude Docs).
