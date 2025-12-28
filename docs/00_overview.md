# LLM Deeper: Project Overview

This document provides an overview of the LLM Deeper project, introduces core LLM concepts, and guides you through the learning path.

---

## Table of Contents

1. [What is a Large Language Model?](#what-is-a-large-language-model)
2. [The Transformer Architecture](#the-transformer-architecture)
3. [Pre-training vs Fine-tuning vs Inference](#pre-training-vs-fine-tuning-vs-inference)
4. [Parameter Counts and Model Sizes](#parameter-counts-and-model-sizes)
5. [Learning Path](#learning-path)
6. [File Index](#file-index)
7. [Hardware Requirements](#hardware-requirements)

---

## What is a Large Language Model?

A **Large Language Model (LLM)** is a neural network trained to predict the next token (word piece) given previous tokens. Despite this simple objective, LLMs develop remarkable capabilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM: Next Token Prediction                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input:  "The capital of France is"                                        │
│                     ↓                                                       │
│           ┌─────────────────┐                                               │
│           │   LLM Model     │                                               │
│           │                 │                                               │
│           │  P("Paris") = 0.92                                              │
│           │  P("Lyon")  = 0.03                                              │
│           │  P("the")   = 0.01                                              │
│           │  ...                                                            │
│           └─────────────────┘                                               │
│                     ↓                                                       │
│   Output: "Paris"                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Properties

| Property | Description |
|----------|-------------|
| **Autoregressive** | Generates one token at a time, feeding output back as input |
| **Context Window** | Can "see" a fixed number of previous tokens (e.g., 4K, 32K, 128K) |
| **Emergent Abilities** | Larger models exhibit capabilities not present in smaller ones |
| **In-Context Learning** | Can learn from examples in the prompt without weight updates |

### Why "Large"?

| Model | Parameters | Training Data | Training Compute |
|-------|------------|---------------|------------------|
| GPT-2 (2019) | 1.5B | 40GB text | ~$50K |
| GPT-3 (2020) | 175B | 570GB text | ~$5M |
| GPT-4 (2023) | ~1.8T* | Unknown | ~$100M+ |
| Llama 3.1 (2024) | 405B | 15T tokens | Unknown |

*Estimated, not officially confirmed

---

## The Transformer Architecture

The **Transformer** (Vaswani et al., 2017) is the architecture behind all modern LLMs. It consists of stacked blocks, each containing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Transformer Block                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: (batch, seq_len, d_model)                                          │
│            │                                                                │
│            ▼                                                                │
│   ┌────────────────────────────────────────────┐                           │
│   │        Multi-Head Self-Attention           │                           │
│   │                                            │                           │
│   │  • Split into multiple "heads"             │                           │
│   │  • Each head: Attention(Q, K, V)           │                           │
│   │  • Concatenate and project                 │                           │
│   └────────────────────────────────────────────┘                           │
│            │                                                                │
│            ▼  + Residual Connection                                         │
│   ┌────────────────────────────────────────────┐                           │
│   │           Layer Normalization              │                           │
│   └────────────────────────────────────────────┘                           │
│            │                                                                │
│            ▼                                                                │
│   ┌────────────────────────────────────────────┐                           │
│   │        Feed-Forward Network (FFN)          │                           │
│   │                                            │                           │
│   │  • Dense(d_model → 4×d_model, activation)  │                           │
│   │  • Dense(4×d_model → d_model)              │                           │
│   └────────────────────────────────────────────┘                           │
│            │                                                                │
│            ▼  + Residual Connection                                         │
│   ┌────────────────────────────────────────────┐                           │
│   │           Layer Normalization              │                           │
│   └────────────────────────────────────────────┘                           │
│            │                                                                │
│            ▼                                                                │
│   Output: (batch, seq_len, d_model)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Intuition:**
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide?"
- **QK^T**: Compute similarity between all token pairs
- **softmax**: Convert to probabilities (attention weights)
- **√d_k**: Scaling to prevent gradient saturation

### Causal Masking

For language models, we mask future tokens so position `i` can only attend to positions `0...i`:

```
         Position 0  1  2  3
Position 0:  [1.0, -∞, -∞, -∞]   ← Can only see itself
Position 1:  [0.5, 0.5, -∞, -∞]  ← Can see 0 and itself
Position 2:  [0.3, 0.3, 0.4, -∞] ← Can see 0, 1, and itself
Position 3:  [0.2, 0.3, 0.2, 0.3] ← Can see everything
```

---

## Pre-training vs Fine-tuning vs Inference

### The Three Phases of LLM Usage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LLM Development Lifecycle                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  PRE-TRAINING   │───▶│  FINE-TUNING    │───▶│   INFERENCE     │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  • Internet-scale       • Task-specific         • Serve predictions        │
│    text data              dataset               • Real-time                │
│  • Next-token           • Instruction           • Optimized for            │
│    prediction             following               latency                   │
│  • Months of GPU        • Hours to days         • Milliseconds per         │
│    time                 • Consumer GPU            token                     │
│  • $1M - $100M+         • $10 - $1000           • $0.001 per request       │
│                                                                             │
│  YOU: Never do this     YOU: Do this!           YOU: Deploy this!          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Phase | Data | Compute | Cost | Result |
|-------|------|---------|------|--------|
| **Pre-training** | Trillions of tokens | 1000s of GPUs for months | $1M-$100M | Foundation model |
| **Fine-tuning** | 1K-1M examples | 1 GPU for hours | $10-$1000 | Specialized model |
| **Inference** | User queries | 1 GPU per request | $0.001/request | Predictions |

### Types of Fine-tuning

| Method | Trainable Params | Memory | Speed | Quality |
|--------|-----------------|--------|-------|---------|
| **Full Fine-tuning** | 100% | High | Slow | Best |
| **LoRA** | 0.1-1% | Low | Fast | Very Good |
| **QLoRA** | 0.1-1% | Very Low | Fast | Good |
| **Prompt Tuning** | <0.01% | Minimal | Fastest | Moderate |

---

## Parameter Counts and Model Sizes

### Understanding Model Sizes

A parameter is a single number (weight) in the model. More parameters = more capacity = better performance (usually).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Parameter Count Formula                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For a Transformer with:                                                   │
│   • L layers                                                                │
│   • d_model embedding dimension                                             │
│   • V vocabulary size                                                       │
│   • h attention heads                                                       │
│                                                                             │
│   Approximate parameters:                                                   │
│                                                                             │
│   Embeddings:     V × d_model                                               │
│   Per Layer:      12 × d_model²  (attention + FFN)                         │
│   LM Head:        d_model × V                                               │
│                                                                             │
│   Total ≈ 2×V×d_model + L×12×d_model²                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Size to VRAM

| Parameters | FP32 | FP16/BF16 | INT8 | INT4 |
|------------|------|-----------|------|------|
| 0.5B | 2 GB | 1 GB | 0.5 GB | 0.25 GB |
| 1.5B | 6 GB | 3 GB | 1.5 GB | 0.75 GB |
| 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 70B | 280 GB | 140 GB | 70 GB | 35 GB |

**Formula:** `VRAM = Parameters × Bytes per Parameter`
- FP32: 4 bytes/param
- FP16/BF16: 2 bytes/param
- INT8: 1 byte/param
- INT4: 0.5 bytes/param

---

## Learning Path

### Suggested Order

```
Week 1: Foundations (src/basics/)
├── 00_pandas.py          - Data manipulation refresher
├── 01_numpy.py           - Tensor operations
├── 02_math.py            - Mathematical foundations
├── 03_attention_math.py  - Attention step-by-step ⭐
├── 04_attention_impl.py  - Full attention implementation
├── 05_attention_positional_encoding.py - Position info
├── 06_mha.py             - Multi-head attention ⭐
└── 07_conclusion.py      - Putting it together

Week 2: Training (src/hfcore/)
├── 00_tinygpt.py         - Build GPT from scratch ⭐⭐⭐
├── 01_transformers.py    - HuggingFace basics
├── 02_peft_lora.py       - LoRA concepts
├── 03_dataset_plus_finetune_full.py - Full fine-tuning
├── 04_dataset_plus_finetune_lora.py - LoRA fine-tuning ⭐⭐
├── 05_finetuned_model_inference.py
└── 06_evals.py           - Evaluation metrics

Week 3+: Production (src/inferencing_and_advanced/)
├── 00_base.py
├── 01_peft_to_mlx.py     - Convert adapters
├── 02_mlx_lora.py        - MLX inference
├── 03_server.py          - FastAPI server
├── 04_tool_calling_basic.py - Function calling ⭐
├── 05_tool_calling_advanced.py - Advanced patterns
└── 07_vllm_cuda_plus_tools.py - Production vLLM ⭐⭐
```

⭐ = Highly recommended

---

## File Index

### src/basics/

| File | Description | Key Concepts |
|------|-------------|--------------|
| `00_pandas.py` | Data manipulation basics | DataFrames, filtering |
| `01_numpy.py` | Tensor operations | Shapes, broadcasting, matmul |
| `02_math.py` | Mathematical foundations | Linear algebra basics |
| `03_attention_math.py` | Attention step-by-step | Q, K, V, scores, softmax |
| `04_attention_impl.py` | Full attention | Complete implementation |
| `05_attention_positional_encoding.py` | Position information | Sinusoidal, learned |
| `06_mha.py` | Multi-head attention | Head splitting, concatenation |
| `07_conclusion.py` | Summary | Putting it all together |

### src/hfcore/

| File | Description | Key Concepts |
|------|-------------|--------------|
| `00_tinygpt.py` | Minimal GPT from scratch | Tokenization, training loop |
| `01_transformers.py` | HuggingFace basics | AutoModel, AutoTokenizer |
| `02_peft_lora.py` | LoRA concepts | Low-rank adaptation |
| `03_dataset_plus_finetune_full.py` | Full fine-tuning | All parameters updated |
| `04_dataset_plus_finetune_lora.py` | LoRA fine-tuning | Efficient adaptation |
| `05_finetuned_model_inference.py` | Using fine-tuned models | Loading, generation |
| `06_evals.py` | Evaluation metrics | ROUGE, BERTScore, perplexity |

### src/inferencing_and_advanced/

| File | Description | Key Concepts |
|------|-------------|--------------|
| `00_base.py` | Base concepts | Setup |
| `01_peft_to_mlx.py` | Convert PEFT → MLX | Adapter formats |
| `02_mlx_lora.py` | MLX inference | Apple Silicon |
| `03_server.py` | FastAPI server | REST API |
| `04_tool_calling_basic.py` | Function calling | Tool schemas |
| `05_tool_calling_advanced.py` | Advanced patterns | Multi-turn, registry |
| `07_vllm_cuda_plus_tools.py` | Production vLLM | CUDA, KV cache |

---

## Hardware Requirements

### Minimum Requirements by Task

| Task | CPU | Apple Silicon | NVIDIA GPU |
|------|-----|---------------|------------|
| **Basics (attention math)** | ✅ Any | ✅ Any | Not needed |
| **TinyGPT training** | ✅ Any | ✅ Any | Not needed |
| **LoRA fine-tuning (0.5B)** | ⚠️ Slow | ✅ M1+ 8GB | ✅ 8GB+ |
| **LoRA fine-tuning (7B)** | ❌ | ✅ M1+ 16GB | ✅ 16GB+ |
| **vLLM inference** | ❌ | ❌ | ✅ Required |
| **MLX inference** | ❌ | ✅ Required | ❌ |

### Recommended Setups

| Budget | Setup | What You Can Do |
|--------|-------|-----------------|
| **$0** | Google Colab (free) | Basic training, limited time |
| **$1K** | Mac Mini M2 16GB | Full LoRA fine-tuning, MLX inference |
| **$2K** | Mac Studio M2 32GB | 7B+ models, fast inference |
| **$1K/mo** | Cloud GPU (A100) | Production vLLM, 70B models |

---

## Next Steps

1. **Start with fundamentals**: [01_basics.md](01_basics.md)
2. **Learn training**: [02_hfcore.md](02_hfcore.md)
3. **Deploy models**: [03_inferencing.md](03_inferencing.md)
4. **Add capabilities**: [04_tool_calling.md](04_tool_calling.md)
5. **Go to production**: [05_deployment.md](05_deployment.md)

---

## Quick Reference

### Key Formulas

| Concept | Formula |
|---------|---------|
| Attention | `softmax(QK^T / √d_k) × V` |
| LoRA | `W' = W + BA` where `B: d×r`, `A: r×d` |
| Perplexity | `exp(loss)` |
| VRAM (weights) | `params × bytes_per_param` |
| KV Cache | `2 × layers × hidden × seq_len × kv_heads × 2` |

### Common Commands

```bash
# Run any script
uv run python src/path/to/file.py

# Start MLX server
uv run python -m mlx_lm.server --model MODEL --port 8000

# Start vLLM server
vllm serve MODEL --enable-lora --lora-modules name=path
```
