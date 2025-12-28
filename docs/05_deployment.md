# Production Deployment

This document covers deploying LLMs to production: vLLM for high-performance serving, GPU memory optimization, LoRA adapter serving, and cloud deployment options.

---

## Table of Contents

1. [vLLM: High-Performance Inference](#vllm-high-performance-inference)
2. [GPU Memory Management](#gpu-memory-management)
3. [LoRA Adapter Serving](#lora-adapter-serving)
4. [Notebook Environments (Colab/Kaggle)](#notebook-environments-colabkaggle)
5. [Cloud Deployment Options](#cloud-deployment-options)
6. [Production Architecture](#production-architecture)
7. [Optimization Techniques](#optimization-techniques)
8. [Reasoning Models (DeepSeek R1)](#reasoning-models-deepseek-r1)
9. [Large Models with Quantization](#large-models-with-quantization)
10. [Image Generation (Diffusers)](#image-generation-diffusers)
11. [Code Walkthrough](#code-walkthrough)

---

## vLLM: High-Performance Inference

### What is vLLM?

vLLM is a fast, memory-efficient LLM inference engine that implements:
- **PagedAttention**: Efficient KV cache management
- **Continuous Batching**: Dynamic request batching
- **OpenAI-Compatible API**: Drop-in replacement

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           vLLM Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Requests ───────────────────────────────────────────────────────┐        │
│   [User 1]  [User 2]  [User 3]  [User 4]  [User 5]                │        │
│       ↓         ↓         ↓         ↓         ↓                   │        │
│   ┌───────────────────────────────────────────────────────────┐   │        │
│   │                  Continuous Batching                      │   │        │
│   │                                                           │   │        │
│   │   • Requests join/leave batch dynamically                 │   │        │
│   │   • No waiting for slowest request                        │   │        │
│   │   • Maximizes GPU utilization                             │   │        │
│   └───────────────────────────────────────────────────────────┘   │        │
│       ↓                                                           │        │
│   ┌───────────────────────────────────────────────────────────┐   │        │
│   │                  PagedAttention                           │   │        │
│   │                                                           │   │        │
│   │   • KV cache stored in blocks (like virtual memory)       │   │        │
│   │   • No memory fragmentation                               │   │        │
│   │   • Efficient memory sharing for beam search              │   │        │
│   └───────────────────────────────────────────────────────────┘   │        │
│       ↓                                                           │        │
│   GPU ←───────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Basic vLLM Usage

**Python API:**
```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(model="Qwen/Qwen3-0.6B")

# Configure sampling
params = SamplingParams(
    temperature=0.7,
    max_tokens=200,
    top_p=0.9,
)

# Generate
outputs = llm.generate(["What is machine learning?"], params)
print(outputs[0].outputs[0].text)
```

**Server Mode (OpenAI-compatible):**
```bash
# Start server
vllm serve Qwen/Qwen3-0.6B --port 8000

# Call API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### vLLM Configuration Options

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `gpu_memory_utilization` | Fraction of GPU memory to use | 0.9 |
| `max_model_len` | Maximum context length | Model default |
| `enforce_eager` | Disable CUDA graphs | False |
| `tensor_parallel_size` | Multi-GPU parallelism | 1 |
| `enable_lora` | Enable LoRA adapter support | False |
| `max_lora_rank` | Maximum LoRA rank | 16 |

---

## GPU Memory Management

### VRAM Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GPU Memory Usage                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Total VRAM: 16 GB (Tesla T4 example)                                      │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────┐       │
│   │ Model Weights (FP16)                                    1.2 GB │       │
│   ├────────────────────────────────────────────────────────────────┤       │
│   │                                                                │       │
│   │                                                                │       │
│   │            KV Cache                                   10-12 GB │       │
│   │            (depends on context length)                         │       │
│   │                                                                │       │
│   │                                                                │       │
│   ├────────────────────────────────────────────────────────────────┤       │
│   │ CUDA Kernels + Overhead                                 1-2 GB │       │
│   └────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   Note: KV cache often > model weights for large context lengths!          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory by Model Size and Context

| Model | Weights (FP16) | KV Cache (4K) | KV Cache (32K) | Total (4K) | Total (32K) |
|-------|---------------|---------------|----------------|------------|-------------|
| 0.5B | 1 GB | 0.5 GB | 4 GB | ~2 GB | ~6 GB |
| 1.5B | 3 GB | 0.8 GB | 6 GB | ~4 GB | ~10 GB |
| 7B | 14 GB | 2 GB | 16 GB | ~17 GB | ~32 GB |
| 13B | 26 GB | 3 GB | 24 GB | ~30 GB | ~52 GB |
| 70B | 140 GB | 10 GB | 80 GB | ~155 GB | ~230 GB |

### Memory Optimization Strategies

```python
# Strategy 1: Limit context length
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=2048,  # Instead of 40,960
)

# Strategy 2: Reduce memory utilization
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.7,  # Leave 30% headroom
)

# Strategy 3: Use quantization
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    quantization="awq",  # or "gptq"
)

# Strategy 4: Multi-GPU (tensor parallelism)
llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,  # Split across 4 GPUs
)
```

---

## LoRA Adapter Serving

### Why Runtime Adapters?

| Approach | Pros | Cons |
|----------|------|------|
| **Merged model** | Simple, single model | One model per task |
| **Runtime adapters** | Multiple tasks, single base | Slightly more complex |

### vLLM with LoRA

```bash
# Start server with LoRA enabled
vllm serve Qwen/Qwen3-0.6B \
  --enable-lora \
  --lora-modules medical=./adapters/medical legal=./adapters/legal \
  --max-lora-rank 16
```

```python
# Python API
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_lora=True,
    max_lora_rank=16,
)

# Create LoRA request
lora_request = LoRARequest(
    "medical",                    # Adapter name
    1,                            # Unique ID
    "./adapters/medical-lora"     # Path to PEFT adapter
)

params = SamplingParams(temperature=0.7, max_tokens=200)

# Generate with adapter
outputs = llm.generate(
    ["What are symptoms of diabetes?"],
    params,
    lora_request=lora_request
)
```

### Multi-Tenant Serving

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Tenant LoRA Serving                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Base Model: Qwen/Qwen3-0.6B (loaded once)                                │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────┐             │
│   │                     vLLM Server                          │             │
│   │                                                          │             │
│   │   Adapters:                                              │             │
│   │   ├── medical-lora (12 MB)                              │             │
│   │   ├── legal-lora (12 MB)                                │             │
│   │   └── finance-lora (12 MB)                              │             │
│   │                                                          │             │
│   └──────────────────────────────────────────────────────────┘             │
│         ↑              ↑              ↑                                    │
│         │              │              │                                    │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                               │
│   │ Medical │    │  Legal  │    │ Finance │                               │
│   │   App   │    │   App   │    │   App   │                               │
│   └─────────┘    └─────────┘    └─────────┘                               │
│                                                                             │
│   API Call: model="medical" → uses medical-lora                            │
│   API Call: model="legal" → uses legal-lora                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Notebook Environments (Colab/Kaggle)

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| V1 engine crash | Multiprocessing conflict | `VLLM_USE_V1=0` |
| "Engine initialization failed" | CUDA already initialized | Restart runtime |
| Out of memory | Context too long | Reduce `max_model_len` |
| Slow first run | Model downloading | Normal, cached after |

### Colab/Kaggle Setup

```python
# FIRST CELL - Run before ANY imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_USE_V1"] = "0"  # Disable V1 engine
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# SECOND CELL - Now import and use
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enforce_eager=True,           # Required for notebooks
    gpu_memory_utilization=0.7,   # Leave headroom
    max_model_len=2048,           # Limit context
)

params = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0].outputs[0].text)
```

### Cleaning Up Between Runs

```python
import gc
import torch

# Delete model
del llm
gc.collect()
torch.cuda.empty_cache()

# Recreate
llm = LLM(...)
```

---

## Cloud Deployment Options

### Comparison Table

| Platform | Service | GPU Options | Pricing Model |
|----------|---------|-------------|---------------|
| **AWS** | SageMaker | ml.g4dn, ml.g5, ml.p4d | Per hour |
| **AWS** | ECS + EC2 | g4dn, g5, p4d instances | Per hour |
| **Azure** | Azure ML | NC-series, ND-series | Per hour |
| **GCP** | Vertex AI | T4, A100, H100 | Per hour |
| **Modal** | Serverless | T4, A10G, A100 | Per second |
| **Replicate** | API | Various | Per prediction |

### AWS SageMaker Deployment

```python
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
)

response = predictor.predict({
    "inputs": "What is machine learning?",
    "parameters": {"max_new_tokens": 100}
})
```

### Docker Deployment

```dockerfile
FROM vllm/vllm-openai:latest

# Copy adapter if using LoRA
COPY ./adapters /app/adapters

ENV HF_TOKEN=your_token

CMD ["--model", "Qwen/Qwen3-0.6B", \
     "--enable-lora", \
     "--lora-modules", "medical=/app/adapters/medical", \
     "--port", "8000"]
```

### ECS Task Definition

```json
{
  "family": "vllm-inference",
  "requiresCompatibilities": ["EC2"],
  "containerDefinitions": [{
    "name": "vllm",
    "image": "your-ecr/vllm:latest",
    "memory": 16384,
    "cpu": 4096,
    "portMappings": [{"containerPort": 8000}],
    "resourceRequirements": [{
      "type": "GPU",
      "value": "1"
    }],
    "environment": [
      {"name": "HF_TOKEN", "value": "..."}
    ]
  }]
}
```

---

## Production Architecture

### Simple Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Simple Production Setup                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Users ─────→ Load Balancer ─────→ vLLM Server ─────→ GPU                 │
│                                          │                                  │
│                                          ↓                                  │
│                                     Model Store                             │
│                                     (S3/GCS)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Scalable Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Scalable Production Setup                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Users                                                                      │
│     │                                                                       │
│     ↓                                                                       │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │                  API Gateway                            │              │
│   │              (Rate limiting, Auth)                      │              │
│   └─────────────────────────────────────────────────────────┘              │
│     │                                                                       │
│     ↓                                                                       │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │              Load Balancer (ALB/NLB)                    │              │
│   └─────────────────────────────────────────────────────────┘              │
│     │                │                │                                     │
│     ↓                ↓                ↓                                     │
│   ┌──────┐       ┌──────┐       ┌──────┐                                   │
│   │vLLM 1│       │vLLM 2│       │vLLM 3│    (Auto-scaling group)          │
│   │ GPU  │       │ GPU  │       │ GPU  │                                   │
│   └──────┘       └──────┘       └──────┘                                   │
│     ↑                ↑                ↑                                     │
│     └────────────────┴────────────────┘                                     │
│                      │                                                      │
│                      ↓                                                      │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │              Model Store (S3/EFS)                       │              │
│   │         Adapters, Checkpoints, Configs                  │              │
│   └─────────────────────────────────────────────────────────┘              │
│                                                                             │
│   Monitoring: Prometheus + Grafana                                          │
│   Logging: CloudWatch / Datadog                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Optimization Techniques

### Quantization

| Method | Bits | Quality | Speed | Memory |
|--------|------|---------|-------|--------|
| FP16 | 16 | Best | 1× | 1× |
| INT8 | 8 | Excellent | 1.5× | 0.5× |
| AWQ | 4 | Very Good | 2× | 0.25× |
| GPTQ | 4 | Very Good | 2× | 0.25× |

```python
# AWQ quantization
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
)
```

### Speculative Decoding

Use a small model to draft tokens, verified by the large model:

```python
llm = LLM(
    model="meta-llama/Llama-3-70B",
    speculative_model="meta-llama/Llama-3-8B",
    num_speculative_tokens=5,
)
```

### Flash Attention

Optimized attention implementation (enabled by default in vLLM):
- 2-4× faster attention
- 5-20× less memory for long sequences

### Continuous Batching

vLLM automatically batches requests without waiting:

```
Traditional:    [Req1, Req2, Req3] → Wait for all → Process batch
Continuous:     Req1 joins → Req2 joins mid-generation → Req1 leaves
```

---

## Reasoning Models (DeepSeek R1)

### What are Reasoning Models?

DeepSeek R1 and similar models use **chain-of-thought reasoning** with explicit `<think>...</think>` tokens:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DeepSeek R1 Reasoning Flow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User: "Code to check if number is prime"                                  │
│            │                                                                │
│            ▼                                                                │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │ <think>                                                            │   │
│   │ Okay, I need to write a function to check if a number is prime.   │   │
│   │ A prime is only divisible by 1 and itself. I should:              │   │
│   │ 1. Handle edge cases (n < 2 → False)                              │   │
│   │ 2. Check divisibility up to sqrt(n) for efficiency               │   │
│   │ 3. Special case for 2 (only even prime)                          │   │
│   │ ...                                                                │   │
│   │ </think>                                     [500-2000 tokens!]   │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│   def is_prime(n):          ← Actual answer after thinking                 │
│       if n < 2: return False                                                │
│       ...                                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Token Budget for Reasoning

| Task Complexity | Thinking Tokens | Answer Tokens | Total Needed |
|-----------------|-----------------|---------------|--------------|
| Simple code | 500-1000 | 200 | ~1500 |
| Medium problem | 1000-2000 | 500 | ~3000 |
| Complex reasoning | 2000-4000 | 1000 | ~5000 |
| Math proofs | 3000-6000 | 500 | ~7000 |

### DeepSeek R1 Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

# Format with chat template
messages = [{"role": "user", "content": "What is 25 * 47?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Generate with enough tokens for thinking + answer
outputs = model.generate(**inputs, max_new_tokens=2000)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

# Extract just the answer (after </think>)
if "</think>" in response:
    answer = response.split("</think>")[-1].strip()
else:
    answer = response
print(answer)
```

### vLLM with R1 Models

```python
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_model_len=8192,  # Reduced from 131k default
)

params = SamplingParams(
    temperature=0.7,
    max_tokens=3000,  # Enough for thinking + answer
)

outputs = llm.generate(["Explain quantum entanglement"], params)
full_response = outputs[0].outputs[0].text

# Parse out the thinking vs answer
if "</think>" in full_response:
    thinking = full_response.split("</think>")[0].replace("<think>", "")
    answer = full_response.split("</think>")[1]
    print(f"Thinking: {len(thinking.split())} words")
    print(f"Answer: {answer}")
```

---

## Large Models with Quantization

### When to Use FP8/INT8/INT4

| Model Size | GPU VRAM | FP16 Fits? | Solution |
|------------|----------|------------|----------|
| 7B | 16 GB | ✅ Yes (14 GB) | Direct |
| 14B | 16 GB | ❌ No (28 GB) | INT8/INT4 |
| 24B | 40 GB | ❌ No (48 GB) | FP8 |
| 33B | 40 GB | ❌ No (66 GB) | INT4 |
| 70B | 80 GB | ❌ No (140 GB) | INT4 or 2× GPU |

### FP8 Quantization (Best Quality)

FP8 preserves more precision than INT8 while using the same memory:

```python
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams

# 24B model on 40GB GPU with FP8
llm = LLM(
    model="mistralai/Devstral-Small-2-24B-Instruct-2512",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="float16",       # Load weights in FP16
    quantization="fp8",    # Quantize to FP8 for inference
)

# Code generation with system prompt
messages = [
    {
        "role": "system",
        "content": "You are a Python programmer. Output ONLY code, no explanations."
    },
    {
        "role": "user",
        "content": "Code to check if number is prime"
    }
]

params = SamplingParams(temperature=0.2, max_tokens=500)
outputs = llm.chat([messages], params)
print(outputs[0].outputs[0].text)
```

### Memory Comparison

```
24B Model on 40GB GPU:

FP16:    [========================X] 48 GB ← DOESN'T FIT
          ▲
          └── Needs 48 GB, GPU only has 40 GB

FP8:     [===================     ] 24 GB ← FITS!
          ▲
          └── 50% memory reduction

INT4:    [=========                ] 12 GB ← Easily fits
          ▲
          └── 75% memory reduction (some quality loss)
```

### Model Architecture from config.json

You can find layer/hidden size info for any model:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

print(f"Layers: {config.num_hidden_layers}")            # 40
print(f"Hidden size: {config.hidden_size}")             # 5120
print(f"KV heads: {config.num_key_value_heads}")        # 8 (GQA)
print(f"Max context: {config.max_position_embeddings}") # 131072
```

This helps calculate KV cache memory:

```
KV Cache = 2 × layers × (hidden / heads × kv_heads) × seq_len × 2 bytes

For 14B model, 8k context:
= 2 × 40 × (5120/40 × 8) × 8192 × 2
= 2 × 40 × 1024 × 8192 × 2
≈ 1.3 GB
```

---

## Image Generation (Diffusers)

### What is Diffusion?

Diffusion models generate images by learning to reverse a noise-adding process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Diffusion Process                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FORWARD (Training): Add noise progressively                               │
│   Image ──→ Noisy ──→ Noisier ──→ ... ──→ Pure Noise                       │
│   🖼️         ░░░       ▒▒▒▒▒         ████████████                           │
│                                                                             │
│   REVERSE (Inference): Learn to denoise                                     │
│   Pure Noise ──→ Less Noisy ──→ ... ──→ Image                              │
│   ████████████    ▒▒▒▒▒▒▒▒▒▒       🖼️                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_inference_steps` | Denoising iterations | 4 (turbo) to 50 (standard) |
| `guidance_scale` | How closely to follow prompt (CFG) | 0.0 (turbo) to 7.5 (standard) |
| `negative_prompt` | What to avoid in the image | "blurry, low quality" |
| `width`, `height` | Output image dimensions | 1024×1024, 1664×928 |

### Turbo vs Standard Models

| Model Type | Steps | Guidance | Speed | Quality |
|------------|-------|----------|-------|---------|
| **Standard** (SDXL) | 20-50 | 7.5 | Slow | Highest |
| **Turbo** (SD3.5 Turbo) | 4 | 0.0 | Fast | Excellent |

**Why Turbo uses `guidance_scale=0.0`:** The model was *distilled* with guidance baked in. It already knows to follow prompts closely.

### Stable Diffusion 3.5 Example

```python
import torch
from diffusers import StableDiffusion3Pipeline

# Load model (requires ~18GB VRAM for float16)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo", 
    torch_dtype=torch.bfloat16,  # Use float16 for T4/older GPUs
)
pipe = pipe.to("cuda")

# Generate image
image = pipe(
    "A cyberpunk city at sunset, neon lights",
    num_inference_steps=4,    # Turbo = few steps
    guidance_scale=0.0,       # Turbo = no CFG needed
).images[0]

image.save("cyberpunk.png")
```

### Memory Optimization

```python
# For smaller GPUs (12-16GB)
pipe.enable_model_cpu_offload()

# For very small GPUs (8-10GB)
pipe.enable_sequential_cpu_offload()
```

### GPU Compatibility

| GPU | bfloat16 Support | Recommendation |
|-----|------------------|----------------|
| T4, V100 | ❌ No | Use `torch.float16` |
| A100, H100 | ✅ Yes | Use `torch.bfloat16` |
| RTX 3090/4090 | ✅ Yes | Use `torch.bfloat16` |

### Aspect Ratios

```python
aspect_ratios = {
    "1:1": (1328, 1328),    # Square
    "16:9": (1664, 928),    # Landscape/video
    "9:16": (928, 1664),    # Portrait/mobile
    "4:3": (1472, 1140),    # Classic photo
    "3:2": (1584, 1056),    # DSLR ratio
}

width, height = aspect_ratios["16:9"]
image = pipe(prompt, width=width, height=height, ...).images[0]
```

---

## Code Walkthrough

### File: `07_vllm_cuda_plus_tools.py`

Complete Colab/Kaggle setup:

```python
# Environment setup (FIRST CELL)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# vLLM inference
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enforce_eager=True,
    gpu_memory_utilization=0.7,
)

params = SamplingParams(temperature=0.7, max_tokens=2000)
outputs = llm.generate(["What are the symptoms of diabetes?"], params)
print(outputs[0].outputs[0].text)

# LoRA training in same notebook
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Train
trainer = SFTTrainer(...)
trainer.train()

# Inference with LoRA (using vLLM)
from vllm.lora.request import LoRARequest

llm_with_lora = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_lora=True,
    max_lora_rank=16,
    enforce_eager=True,
)

lora_request = LoRARequest(
    "medical",
    1,
    "./output_models/qwen3-0.6b-medical-lora/final"
)

outputs = llm_with_lora.generate(
    ["What is diabetes?"],
    params,
    lora_request=lora_request
)
```

### File: `08_deepseek_example.py`

DeepSeek R1 reasoning model example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load DeepSeek R1 (33B - needs large GPU or quantization)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-33B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-33B")

# Use chat template for proper formatting
messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# vLLM version (faster for production)
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-33B",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, max_tokens=2000)
outputs = llm.generate(["What are the symptoms of diabetes?"], params)
print(outputs[0].outputs[0].text)
```

### File: `09_mistral_example.py`

Large model (24B) with FP8 quantization:

```python
import os
os.environ["VLLM_USE_V1"] = "0"  # Required for notebooks
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
import gc, torch

gc.collect()
torch.cuda.empty_cache()

# 24B model with FP8 quantization (fits in 40GB GPU)
llm = LLM(
    model="mistralai/Devstral-Small-2-24B-Instruct-2512",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="float16",
    quantization="fp8",
)

# Code generation with system prompt
messages = [
    {
        "role": "system",
        "content": "You are a Python code generator. Output ONLY code."
    },
    {"role": "user", "content": "Code to check if number is prime"}
]

params = SamplingParams(temperature=0.2, max_tokens=2500)
outputs = llm.chat([messages], params)
print(outputs[0].outputs[0].text)
```

### File: `10_diffuser_image.py`

Image generation with Stable Diffusion 3.5 Turbo:

```python
import torch
from diffusers import StableDiffusion3Pipeline

# Load SD3.5 Turbo (requires bfloat16-capable GPU like A100/H100)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

# Generate with turbo settings
image = pipe(
    "Korra from avatar series in real life",
    num_inference_steps=4,   # Turbo needs only 4 steps
    guidance_scale=0.0,      # CFG baked into turbo models
).images[0]

image.save("avatar.png")

# For more control, use aspect ratios and seeds
aspect_ratios = {"16:9": (1664, 928), "9:16": (928, 1664)}
width, height = aspect_ratios["16:9"]

image = pipe(
    prompt="A coffee shop with neon signs",
    width=width,
    height=height,
    num_inference_steps=50,  # More steps for non-turbo
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]
```

---

## GPU Recommendations

### By Use Case

| Use Case | GPU | VRAM | Cost/hr |
|----------|-----|------|---------|
| **Development** | T4 | 16 GB | $0.35 |
| **Small models (≤7B)** | A10G | 24 GB | $1.00 |
| **Medium models (≤13B)** | A100-40GB | 40 GB | $3.00 |
| **Large models (≤70B)** | A100-80GB | 80 GB | $5.00 |
| **Largest models** | H100 | 80 GB | $10.00 |

### Multi-GPU Scaling

| Model Size | GPUs Needed (FP16) | With INT4 |
|------------|-------------------|-----------|
| 7B | 1× A10G | 1× T4 |
| 13B | 1× A100-40GB | 1× A10G |
| 70B | 2× A100-80GB | 1× A100-80GB |
| 405B | 8× H100 | 4× H100 |

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **vLLM** | High-performance inference with PagedAttention |
| **KV Cache** | Often larger than model weights |
| **LoRA Serving** | One base model, multiple adapters |
| **Notebooks** | Use `VLLM_USE_V1=0` and `enforce_eager=True` |
| **Cloud** | SageMaker/ECS for AWS, Azure ML for Azure |
| **Optimization** | Quantization, speculative decoding, flash attention |
| **Reasoning Models** | DeepSeek R1 uses `<think>` tokens, needs 3000+ max_tokens |
| **Large Models** | Use FP8/INT4 quantization to fit in smaller GPUs |
| **Image Generation** | Diffusers library, turbo models use 4 steps + no CFG |

---

## Quick Reference

### vLLM Commands

```bash
# Start server
vllm serve MODEL --port 8000

# With LoRA
vllm serve MODEL --enable-lora --lora-modules name=path

# With quantization
vllm serve MODEL --quantization awq
```

### Memory Estimation

```python
# Model weights
weights_gb = params_billions * 2  # FP16

# KV Cache
kv_cache_gb = 4 * num_layers * hidden_dim * kv_heads * max_seq_len * 2 / 1e9

# Total
total_gb = weights_gb + kv_cache_gb + 2  # +2 GB overhead
```

### Troubleshooting

| Error | Solution |
|-------|----------|
| OOM | Reduce `max_model_len` or `gpu_memory_utilization` |
| V1 engine crash | Set `VLLM_USE_V1=0` |
| Multiprocessing error | Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| Slow first request | Model loading, normal |

