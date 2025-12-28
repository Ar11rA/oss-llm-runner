# Model Inference

This document covers model inference: formats, quantization, MLX for Apple Silicon, and building inference servers.

---

## Table of Contents

1. [Model Formats](#model-formats)
2. [Quantization](#quantization)
3. [KV Cache Deep Dive](#kv-cache-deep-dive)
4. [MLX on Apple Silicon](#mlx-on-apple-silicon)
5. [Adapter Formats & Conversion](#adapter-formats--conversion)
6. [Building Inference Servers](#building-inference-servers)
7. [Code Walkthrough](#code-walkthrough)

---

## Model Formats

### Common Formats

| Format | Description | Used By |
|--------|-------------|---------|
| **PyTorch (.bin)** | Original PyTorch format | Legacy HuggingFace |
| **SafeTensors (.safetensors)** | Safe, fast loading | Modern HuggingFace |
| **GGUF** | Quantized for llama.cpp | Local inference |
| **MLX** | Apple Silicon optimized | MLX-LM |
| **ONNX** | Cross-platform | Production |
| **TensorRT** | NVIDIA optimized | High-performance |

### SafeTensors vs PyTorch

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SafeTensors Advantages                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PyTorch .bin:                      SafeTensors:                           │
│   ┌─────────────────┐                ┌─────────────────┐                   │
│   │ Uses pickle     │                │ No pickle       │                   │
│   │ (security risk) │                │ (safe by design)│                   │
│   │                 │                │                 │                   │
│   │ Full load       │                │ Lazy loading    │                   │
│   │ into memory     │                │ (memory-mapped) │                   │
│   │                 │                │                 │                   │
│   │ Slow loading    │                │ Fast loading    │                   │
│   └─────────────────┘                └─────────────────┘                   │
│                                                                             │
│   Rule: Always prefer SafeTensors (.safetensors) when available            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quantization

### What is Quantization?

Quantization reduces the precision of model weights to save memory and speed up inference.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Quantization Levels                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FP32:    32 bits per weight   [████████████████████████████████]         │
│            Full precision                                                   │
│                                                                             │
│   FP16:    16 bits per weight   [████████████████]                         │
│            Half precision                                                   │
│                                                                             │
│   BF16:    16 bits per weight   [████████████████]                         │
│            Brain floating point (better for training)                       │
│                                                                             │
│   INT8:    8 bits per weight    [████████]                                 │
│            1/4 the memory of FP32                                          │
│                                                                             │
│   INT4:    4 bits per weight    [████]                                     │
│            1/8 the memory of FP32                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Savings

| Precision | Bytes/Param | 7B Model | 70B Model |
|-----------|-------------|----------|-----------|
| FP32 | 4 | 28 GB | 280 GB |
| FP16/BF16 | 2 | 14 GB | 140 GB |
| INT8 | 1 | 7 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB |

### Quality vs Compression Trade-offs

| Precision | Memory | Speed | Quality |
|-----------|--------|-------|---------|
| FP32 | 1× | 1× | Best |
| FP16 | 0.5× | 1.5× | Excellent |
| INT8 | 0.25× | 2× | Very Good |
| INT4 | 0.125× | 2.5× | Good |
| INT2 | 0.0625× | 3× | Degraded |

### When to Use Each

| Use Case | Precision |
|----------|-----------|
| Training | FP32 or BF16 |
| Fine-tuning | FP16 or BF16 |
| Production inference | FP16 or INT8 |
| Consumer hardware | INT4 (GGUF) |
| Extreme memory constraints | INT4 with GPTQ/AWQ |

---

## KV Cache Deep Dive

### What is KV Cache?

During autoregressive generation, the model computes Key and Value vectors for each token. The **KV Cache** stores these to avoid recomputation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Without KV Cache                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Token 1: Compute K,V for "The"                                           │
│   Token 2: Compute K,V for "The", "cat"         ← Redundant!               │
│   Token 3: Compute K,V for "The", "cat", "sat"  ← Very redundant!          │
│                                                                             │
│   Complexity: O(n²) per generation step                                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           With KV Cache                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Token 1: Compute K,V for "The"        → Cache: [(K₁,V₁)]                 │
│   Token 2: Compute K,V for "cat" only   → Cache: [(K₁,V₁), (K₂,V₂)]        │
│   Token 3: Compute K,V for "sat" only   → Cache: [(K₁,V₁), (K₂,V₂), ...]   │
│                                                                             │
│   Complexity: O(n) per generation step                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### KV Cache Memory Formula

```
KV Cache Size = 2 × num_layers × 2 × hidden_size × num_kv_heads/num_heads × max_seq_len × batch_size × bytes_per_param
```

Simplified:
```
KV Cache ≈ 4 × num_layers × hidden_size × kv_heads × max_seq_len × bytes
```

### KV Cache by Model Size

| Model | Layers | Hidden | KV Heads | 4K Context | 32K Context |
|-------|--------|--------|----------|------------|-------------|
| 0.5B | 28 | 1024 | 2 | 0.5 GB | 4 GB |
| 7B | 32 | 4096 | 8 | 2 GB | 16 GB |
| 13B | 40 | 5120 | 8 | 3 GB | 24 GB |
| 70B | 80 | 8192 | 8 | 10 GB | 80 GB |

### Why This Matters

**The model weights are often smaller than the KV cache!**

```
Example: Qwen3-0.6B with 40K context
├── Model weights (FP16): 1.2 GB
├── KV Cache: 9.2 GB
└── Total VRAM: ~11 GB

Solution: Limit max_model_len!
```

```python
# Memory-efficient vLLM config
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=2048,           # Limit context (vs 40,960 default)
    gpu_memory_utilization=0.7,   # Reserve 30% headroom
)
```

---

## MLX on Apple Silicon

### What is MLX?

MLX is Apple's machine learning framework, optimized for Apple Silicon (M1/M2/M3/M4).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLX Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                    Unified Memory                           │          │
│   │                                                             │          │
│   │   ┌─────────────┐         ┌─────────────┐                  │          │
│   │   │   CPU       │ ←─────→ │   GPU       │                  │          │
│   │   │   Cores     │  Shared │   Cores     │                  │          │
│   │   └─────────────┘  Memory └─────────────┘                  │          │
│   │         ↑                       ↑                          │          │
│   │         └───────────────────────┘                          │          │
│   │              No data copying!                              │          │
│   │                                                             │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│   Benefits:                                                                 │
│   • No CPU↔GPU memory transfers                                            │
│   • Larger effective VRAM (uses system RAM)                                │
│   • Lower latency                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MLX-LM Quick Start

```python
from mlx_lm import load, generate

# Load model
model, tokenizer = load("Qwen/Qwen3-0.6B")

# Generate
response = generate(
    model, 
    tokenizer, 
    prompt="What is machine learning?",
    max_tokens=200
)
print(response)
```

### MLX vs PyTorch on Apple Silicon

| Aspect | PyTorch (MPS) | MLX |
|--------|---------------|-----|
| Memory efficiency | Good | Excellent |
| Native optimization | Partial | Full |
| API style | PyTorch | NumPy-like |
| Training support | Yes | Yes |
| Inference speed | Good | Better |
| Compatibility | Cross-platform | Apple only |

### When to Use MLX

| Scenario | Recommendation |
|----------|----------------|
| Mac inference | MLX |
| Mac fine-tuning | MLX or PyTorch |
| Cross-platform code | PyTorch |
| Production on Linux | PyTorch/vLLM |

---

## Adapter Formats & Conversion

### PEFT vs MLX Adapter Formats

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Adapter Formats                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PEFT/HuggingFace Format:           MLX Format:                            │
│   ├── adapter_config.json            ├── adapter_config.json                │
│   ├── adapter_model.safetensors      ├── adapters.safetensors  ← Different!│
│   └── tokenizer files                └── tokenizer files                    │
│                                                                             │
│   Trained with: PEFT library         Trained with: mlx_lm.lora             │
│   Works with: HuggingFace, vLLM      Works with: MLX-LM                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Converting PEFT to MLX: Merge First

Since PEFT adapters can't be directly loaded by MLX-LM, you need to merge them first:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Merge weights
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.save_pretrained("./merged-model")
```

Then load with MLX:

```python
from mlx_lm import load, generate

model, tokenizer = load("./merged-model")
```

### Options for Using Adapters

| Method | Format | Hot-swappable | Memory |
|--------|--------|---------------|--------|
| Merge + Load | Merged model | No | Full model |
| PEFT runtime | PEFT adapter | Yes | Base + adapter |
| MLX native adapter | MLX adapter | Yes | Base + adapter |
| vLLM LoRA | PEFT adapter | Yes | Base + adapter |

---

## Building Inference Servers

### Option 1: Native MLX-LM Server

```bash
# Start server
uv run python -m mlx_lm.server --model ./model --port 8000

# Test (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Option 2: Custom FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, generate
import uvicorn

# Load model at startup
model, tokenizer = load("./model")

app = FastAPI(title="LLM Server")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

class GenerateResponse(BaseModel):
    prompt: str
    response: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    response = generate(
        model, tokenizer,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
    )
    return GenerateResponse(prompt=request.prompt, response=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Option 3: HuggingFace with FastAPI

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = FastAPI()

@app.post("/generate")
def generate(prompt: str, max_tokens: int = 100):
    result = pipe(prompt, max_new_tokens=max_tokens)
    return {"response": result[0]["generated_text"]}
```

### Server Comparison

| Feature | MLX-LM Server | Custom FastAPI | vLLM Server |
|---------|---------------|----------------|-------------|
| OpenAI API | ✅ | Manual | ✅ |
| Streaming | ✅ | Manual | ✅ |
| Batching | ❌ | Manual | ✅ (continuous) |
| LoRA hot-swap | ❌ | Manual | ✅ |
| Platform | Mac only | Any | CUDA only |

---

## Code Walkthrough

### File: `01_peft_to_mlx.py`

Converting PEFT adapter to merged model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load and merge
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "./adapter")
merged_model = model.merge_and_unload()

# Save
merged_model.save_pretrained("./merged")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.save_pretrained("./merged")
```

### File: `02_mlx_lora.py`

Comparing base vs fine-tuned model:

```python
from mlx_lm import load, generate

# Load both models
base_model, base_tokenizer = load("Qwen/Qwen3-0.6B")
merged_model, merged_tokenizer = load("./merged-model")

# Compare outputs
prompts = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
]

for prompt in prompts:
    print(f"\n=== {prompt} ===")
    
    print("\n🔵 BASE:")
    print(generate(base_model, base_tokenizer, prompt=prompt, max_tokens=200))
    
    print("\n🟣 FINE-TUNED:")
    print(generate(merged_model, merged_tokenizer, prompt=prompt, max_tokens=200))
```

### File: `03_server.py`

FastAPI inference server:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, generate
import uvicorn

MODEL_PATH = "./merged-model"
model, tokenizer = load(MODEL_PATH)

app = FastAPI(title="MLX-LM Inference Server")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    response = generate(model, tokenizer, 
                       prompt=request.prompt, 
                       max_tokens=request.max_tokens)
    return {"prompt": request.prompt, "response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run:**
```bash
uv run python src/inferencing_and_advanced/03_server.py
```

**Test:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is diabetes?", "max_tokens": 100}'
```

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **SafeTensors** | Preferred format, safe & fast |
| **Quantization** | INT4/INT8 for memory savings |
| **KV Cache** | Often larger than model weights |
| **MLX** | Best for Apple Silicon inference |
| **Adapter conversion** | Merge PEFT → load with MLX |
| **Servers** | FastAPI for custom, mlx_lm.server for quick |

---

## Next Steps

- [04_tool_calling.md](04_tool_calling.md) - Add tool/function calling
- [05_deployment.md](05_deployment.md) - Production deployment with vLLM

