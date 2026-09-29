# l0llm — Pure C# & Intel Level Zero LLM Runner and Server

`l0llm` is an ultra-fast, zero-allocation, native LLM runner and OpenAI-compatible REST server designed specifically for Intel® GPUs (Iris Xe, Arc A-Series, Lunar Lake, Arrow Lake, Battlemage, Meteor Lake, and Data Center GPU Flex / Max).

Built in 100% managed C# on top of [LevelZero.NET](https://github.com/hf-laboratories/LevelZero.NET), it requires **zero python runtime, zero conda environments, and zero heavyweight native runtimes**.

---

## ⚡ Performance Benchmarks

Measured on **Intel® Iris® Xe Graphics (96 EU, 15W TDP mobile)** running Qwen 2.5:

| Metric | Result |
| :--- | :--- |
| **Decode Throughput** | **41.3 tokens / sec** |
| **Prefill Latency** | **18.7 ms** |
| **Memory Allocations (GC)** | **0 B per token (Zero Allocations)** |
| **GPU Subsystem** | Direct Intel oneAPI Level Zero command lists |

---

## 🌟 Key Features

- **Multi-Model Support**: Direct support for `LLaMA` (3/3.1/3.2), `Qwen` (2.5/3/3.5), `DeepSeek`, `Mistral`, and `Phi` architectures.
- **Checkpoint Formats**: Reads standard Hugging Face `model.safetensors` as well as sharded index formats (`model.safetensors.index.json`).
- **OpenAI-Compatible REST API**:
  - `POST /v1/chat/completions` (Server-Sent Events streaming & non-streaming)
  - `GET /v1/models`
  - `GET /health`
  - Built-in CORS for plug-and-play compatibility with [Open-WebUI](https://github.com/open-webui/open-webui), Chatbox, and LangChain.
- **Precision Modes**: `Float16`, `BFloat16`, `Int8`, and quantized weight formats.
- **Standalone Binary**: Single executable distribution requiring only the standard Intel graphics driver.

---

## 🚀 Quick Start

### 1. Interactive Console Chat
```cmd
l0llm.exe chat -m path\to\model.safetensors -t path\to\tokenizer.json
```

### 2. OpenAI-Compatible REST Server
```cmd
l0llm.exe serve -m path\to\model.safetensors -t path\to\tokenizer.json --port 8080
```

#### Test with cURL:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Explain Level Zero in 2 sentences."}],
    "stream": true
  }'
```

### 3. One-shot Text Generation
```cmd
l0llm.exe generate -m path\to\model.safetensors -t path\to\tokenizer.json -p "Once upon a time"
```

### 4. Hardware Benchmark Mode
```cmd
l0llm.exe bench -m path\to\model.safetensors -t path\to\tokenizer.json -n 128
```

---

## 🛠️ Building from Source

Requirements:
- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- Intel Graphics Driver with Level Zero runtime (`ze_loader.dll`)

```shell
git clone https://github.com/hf-laboratories/l0llm.git
cd l0llm
dotnet build
dotnet test
```

### Publish Self-Contained Binary
```shell
dotnet publish src/l0llm/l0llm.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -o dist/l0llm-win-x64
```

---

## 📄 License

Apache-2.0. See [LICENSE](LICENSE) for details.
