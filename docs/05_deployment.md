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
8. [Code Walkthrough](#code-walkthrough)

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

