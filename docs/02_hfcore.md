# HuggingFace Training Pipeline

This document covers the complete LLM training pipeline using HuggingFace libraries: from building TinyGPT from scratch to fine-tuning with LoRA and evaluation.

---

## Table of Contents

1. [Tokenization](#tokenization)
2. [The HuggingFace Ecosystem](#the-huggingface-ecosystem)
3. [TinyGPT: Building from Scratch](#tinygpt-building-from-scratch)
4. [Using Transformers for Inference](#using-transformers-for-inference)
5. [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
6. [Full Fine-Tuning vs LoRA](#full-fine-tuning-vs-lora)
7. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Code Walkthrough](#code-walkthrough)

---

## Tokenization

### What is Tokenization?

Tokenization converts text into numerical IDs that the model can process.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Tokenization Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   "Hello, how are you?"                                                     │
│            ↓                                                                │
│   ┌────────────────────────────────────┐                                   │
│   │         Tokenizer                  │                                   │
│   │                                    │                                   │
│   │   Split into subwords:             │                                   │
│   │   ["Hello", ",", " how", " are",   │                                   │
│   │    " you", "?"]                    │                                   │
│   │                                    │                                   │
│   │   Map to IDs:                      │                                   │
│   │   [15496, 11, 703, 527, 499, 30]   │                                   │
│   │                                    │                                   │
│   └────────────────────────────────────┘                                   │
│            ↓                                                                │
│   [15496, 11, 703, 527, 499, 30]                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tokenization Algorithms

| Algorithm | Description | Used By |
|-----------|-------------|---------|
| **BPE** | Byte Pair Encoding - merges frequent byte pairs | GPT-2, GPT-3, Llama |
| **WordPiece** | Similar to BPE, splits into subwords | BERT |
| **SentencePiece** | Language-agnostic, works on raw text | Llama, T5 |
| **Tiktoken** | Optimized BPE for GPT | GPT-4, OpenAI |

### Special Tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `[PAD]` | Padding for batching | `tokenizer.pad_token` |
| `[BOS]` | Beginning of sequence | `<s>` |
| `[EOS]` | End of sequence | `</s>` |
| `[UNK]` | Unknown token | `<unk>` |
| `[SEP]` | Separator | `[SEP]` |

### Vocabulary Size Trade-offs

| Vocab Size | Pros | Cons |
|------------|------|------|
| **Small (1K)** | Fast, simple | Many tokens per word |
| **Medium (32K)** | Good balance | Standard choice |
| **Large (100K+)** | Better multi-lingual | More memory, slower |

---

## The HuggingFace Ecosystem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HuggingFace Libraries                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                   │
│   │ transformers │   │   datasets   │   │    peft      │                   │
│   ├──────────────┤   ├──────────────┤   ├──────────────┤                   │
│   │ AutoModel    │   │ load_dataset │   │ LoraConfig   │                   │
│   │ AutoTokenizer│   │ Dataset.map  │   │ get_peft_    │                   │
│   │ Trainer      │   │ streaming    │   │   model      │                   │
│   └──────────────┘   └──────────────┘   └──────────────┘                   │
│         ↑                   ↑                   ↑                          │
│         └───────────────────┴───────────────────┘                          │
│                             │                                               │
│                    ┌────────┴────────┐                                      │
│                    │       trl       │                                      │
│                    ├─────────────────┤                                      │
│                    │ SFTTrainer      │                                      │
│                    │ DPOTrainer      │                                      │
│                    │ PPOTrainer      │                                      │
│                    └─────────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Libraries

| Library | Purpose | Key Classes |
|---------|---------|-------------|
| **transformers** | Models & training | `AutoModel`, `Trainer` |
| **datasets** | Data loading | `load_dataset`, `Dataset` |
| **peft** | Efficient fine-tuning | `LoraConfig`, `get_peft_model` |
| **trl** | RLHF & SFT | `SFTTrainer`, `DPOTrainer` |
| **accelerate** | Multi-GPU/TPU | `Accelerator` |

---

## TinyGPT: Building from Scratch

TinyGPT is a minimal GPT implementation that demonstrates all core concepts in ~100 lines.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TinyGPT (2,453 params)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   "hell" (input)                                                            │
│       ↓                                                                     │
│   [2, 1, 3, 3] (token IDs)                                                 │
│       ↓                                                                     │
│   ┌─────────────────────────────────────┐                                  │
│   │ Token Embedding (5 × 16) = 80 params │                                  │
│   │ + Position Embedding (4 × 16) = 64   │                                  │
│   └─────────────────────────────────────┘                                  │
│       ↓ (4, 16)                                                            │
│   ┌─────────────────────────────────────┐                                  │
│   │ Transformer Block                    │                                  │
│   │ • MultiHeadAttention (2 heads)       │                                  │
│   │ • FFN (16 → 32 → 16)                │                                  │
│   │ • LayerNorm × 2                      │                                  │
│   └─────────────────────────────────────┘                                  │
│       ↓ (4, 16)                                                            │
│   ┌─────────────────────────────────────┐                                  │
│   │ LM Head (16 → 5) = 85 params         │                                  │
│   └─────────────────────────────────────┘                                  │
│       ↓                                                                     │
│   (4, 5) logits → predict next token at each position                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parameter Count Breakdown

| Component | Shape | Parameters |
|-----------|-------|------------|
| Token Embedding | (5, 16) | 80 |
| Position Embedding | (4, 16) | 64 |
| MHA: Q, K, V, O projections | 4 × (16×16 + 16) | 1,088 |
| FFN: Dense layers | (16×32+32) + (32×16+16) | 1,072 |
| LayerNorm × 2 | 2 × (16 + 16) | 64 |
| LM Head | (16×5 + 5) | 85 |
| **Total** | | **2,453** |

### Training on "hello hello"

```python
text = "hello hello"
# Vocabulary: {' ': 0, 'e': 1, 'h': 2, 'l': 3, 'o': 4}
# Encoded: [2, 1, 3, 3, 4, 0, 2, 1, 3, 3, 4]

# Sliding window creates X, y pairs:
# X: [2,1,3,3]  y: [1,3,3,4]  ("hell" → "ello")
# X: [1,3,3,4]  y: [3,3,4,0]  ("ello" → "llo ")
# ... 7 training samples total
```

---

## Using Transformers for Inference

### Basic Inference Pattern

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

# Generate
output = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)

# Decode
response = tokenizer.decode(output[0], skip_special_tokens=True)
```

### Generation Parameters

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `max_new_tokens` | Maximum tokens to generate | 100-2000 |
| `temperature` | Randomness (higher = more random) | 0.7-1.0 |
| `top_p` | Nucleus sampling threshold | 0.9 |
| `top_k` | Top-k sampling | 50 |
| `do_sample` | Enable sampling (vs greedy) | True |
| `repetition_penalty` | Penalize repeated tokens | 1.1-1.2 |
| `no_repeat_ngram_size` | Block repeated n-grams | 3 |

---

## LoRA: Low-Rank Adaptation

### The Problem with Full Fine-Tuning

```
Full Fine-Tuning:
• Update ALL 600M parameters
• Need 600M × 4 bytes × 3 = 7.2GB for weights + gradients + optimizer
• Risk of catastrophic forgetting
• One model per task
```

### The LoRA Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LoRA Decomposition                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Original weight: W ∈ ℝ^(d × d)                                           │
│                                                                             │
│   LoRA adds:       ΔW = B × A                                              │
│                                                                             │
│   Where:           B ∈ ℝ^(d × r)   ← Trainable                             │
│                    A ∈ ℝ^(r × d)   ← Trainable                             │
│                    r << d          ← Low rank (4, 8, 16)                   │
│                                                                             │
│   Forward pass:    h = (W + αBA/r) × x                                     │
│                                                                             │
│   Parameters:      Full W: d × d = d²                                      │
│                    LoRA:   d×r + r×d = 2dr                                 │
│                                                                             │
│   Example:         d=4096, r=16                                            │
│                    Full: 16,777,216 params                                 │
│                    LoRA: 131,072 params (0.78%)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LoRA Implementation

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=4, alpha=1.0):
        super().__init__()
        
        # Original weight (FROZEN)
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim) * 0.02,
            requires_grad=False  # ← KEY: frozen!
        )
        
        # LoRA matrices (TRAINABLE)
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_dim, r))  # ← Init to zero!
        
        self.scaling = alpha / r
    
    def forward(self, x):
        # Original forward
        base = x @ self.weight.T
        
        # LoRA addition
        lora = (x @ self.A.T) @ self.B.T
        
        return base + self.scaling * lora
```

**Why initialize B to zero?**
At the start of training, `ΔW = B×A = 0`, so the model behaves exactly like the original. Training gradually learns the adaptation.

### LoRA Hyperparameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `r` | Rank of decomposition | 8, 16, 32, 64 |
| `lora_alpha` | Scaling factor | Usually = r or 2×r |
| `target_modules` | Layers to apply LoRA | `["q_proj", "v_proj"]` |
| `lora_dropout` | Dropout rate | 0.05 |
| `bias` | Train biases? | "none" |

### Which Layers to Target?

| Configuration | Trainable % | Quality |
|---------------|-------------|---------|
| Q, V only | ~0.1% | Good |
| Q, K, V, O | ~0.3% | Better |
| + gate, up, down (FFN) | ~0.5% | Best |

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # FFN
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

---

## Full Fine-Tuning vs LoRA

### Comparison Table

| Aspect | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| **Trainable Params** | 100% | 0.1-1% |
| **Memory (0.5B)** | ~8GB | ~2GB |
| **Memory (7B)** | ~56GB | ~8GB |
| **Training Speed** | Slower | 2-3× faster |
| **Storage per Task** | Full model | Adapter only (10-50MB) |
| **Catastrophic Forgetting** | Higher risk | Lower risk |
| **Quality** | Best | Very close |
| **Multi-Task** | 1 model/task | 1 base + N adapters |

### When to Use Each

| Use Case | Recommendation |
|----------|----------------|
| Limited GPU memory | LoRA |
| Multiple tasks/domains | LoRA (share base) |
| Maximum quality needed | Full fine-tuning |
| Quick experimentation | LoRA |
| Deployment flexibility | LoRA |

---

## Supervised Fine-Tuning (SFT)

### What is SFT?

SFT trains a pre-trained model on instruction-response pairs to follow instructions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SFT Training                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Dataset Format:                                                           │
│   ┌────────────────────────────────────────────────────────────┐           │
│   │ {"messages": [                                             │           │
│   │     {"role": "system", "content": "You are a doctor..."},  │           │
│   │     {"role": "user", "content": "What are symptoms..."},   │           │
│   │     {"role": "assistant", "content": "Common symptoms..."} │           │
│   │ ]}                                                         │           │
│   └────────────────────────────────────────────────────────────┘           │
│                        ↓                                                    │
│                  Format & Tokenize                                          │
│                        ↓                                                    │
│   "<|system|>You are...<|user|>What are...<|assistant|>Common..."          │
│                        ↓                                                    │
│              Next-Token Prediction Loss                                     │
│              (only on assistant response)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Using SFTTrainer

```python
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

# Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)

# Configure training
training_args = SFTConfig(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # Effective batch = 32
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    max_length=512,
)

# Format function
def formatting_func(example):
    return example["text"]

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

trainer.train()
```

### Key SFT Concepts

| Concept | Description |
|---------|-------------|
| **Chat Template** | Formats messages with special tokens |
| **Response Masking** | Only compute loss on assistant tokens |
| **Packing** | Combine short examples for efficiency |
| **Gradient Checkpointing** | Trade compute for memory |

---

## Evaluation Metrics

### Perplexity

Measures how "surprised" the model is by the text:

$$\text{Perplexity} = e^{\text{loss}} = e^{-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i|x_{<i})}$$

| Perplexity | Interpretation |
|------------|----------------|
| 1 | Perfect prediction |
| 10 | Good model |
| 100 | Mediocre |
| 1000+ | Poor |

### ROUGE

Measures lexical overlap with reference text:

| Metric | Measures |
|--------|----------|
| **ROUGE-1** | Unigram overlap |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence |

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(reference, generated)
# {'rouge1': Score(precision=0.8, recall=0.7, fmeasure=0.75), ...}
```

### BERTScore

Measures semantic similarity using BERT embeddings:

```python
from bert_score import score

P, R, F1 = score(
    cands=[generated],
    refs=[reference],
    lang="en",
)
# F1 score between 0 and 1
```

| Metric | What it Measures |
|--------|------------------|
| Perplexity | Model confidence |
| ROUGE | Surface-level similarity |
| BERTScore | Semantic similarity |

---

## Code Walkthrough

### File: `00_tinygpt.py`

Builds a complete GPT from scratch:

```python
import tensorflow as tf

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, vocab_size, d_model):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_emb = tf.keras.layers.Embedding(seq_len, d_model)
    
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1])
        return self.token_emb(x) + self.pos_emb(positions)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        attn_out = self.attn(x, x, use_causal_mask=True)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class TinyGPT(tf.keras.Model):
    def __init__(self, seq_len, vocab_size, d_model, num_heads, d_ff):
        super().__init__()
        self.embed = TokenAndPositionEmbedding(seq_len, vocab_size, d_model)
        self.block = TransformerBlock(d_model, num_heads, d_ff)
        self.lm_head = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x):
        x = self.embed(x)
        x = self.block(x)
        return self.lm_head(x)
```

### File: `01_transformers.py`

HuggingFace inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

### File: `02_peft_lora.py`

LoRA from scratch:

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=4, alpha=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.scaling = alpha / r
    
    def forward(self, x):
        return x @ self.weight.T + self.scaling * (x @ self.A.T) @ self.B.T
```

### File: `04_dataset_plus_finetune_lora.py`

Complete LoRA fine-tuning:

```python
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

# Load dataset
dataset = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B")

# Configure LoRA
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
# trainable params: 3,407,872 || all params: 630,000,000 || trainable%: 0.54%

# Train
trainer = SFTTrainer(model=model, args=training_args, ...)
trainer.train()

# Save adapter
model.save_pretrained("./adapter")
```

### File: `06_evals.py`

Evaluation pipeline:

```python
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# Compute perplexity
def compute_perplexity(model, dataset):
    total_loss = 0
    for batch in dataset:
        with torch.no_grad():
            outputs = model(**batch)
            total_loss += outputs.loss
    return torch.exp(total_loss / len(dataset))

# Compute ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(reference, generated)

# Compute BERTScore
P, R, F1 = bert_score_fn(cands=[generated], refs=[reference], lang="en")
```

---

## Memory Requirements

### Training Memory by Model Size

| Model | Full FT (FP16) | LoRA (FP16) | QLoRA (4-bit) |
|-------|---------------|-------------|---------------|
| 0.5B | 4 GB | 2 GB | 1 GB |
| 1.5B | 12 GB | 4 GB | 2 GB |
| 7B | 56 GB | 16 GB | 6 GB |
| 13B | 104 GB | 26 GB | 10 GB |
| 70B | 560 GB | 140 GB | 48 GB |

### Memory Breakdown

```
Training Memory = Model + Gradients + Optimizer + Activations

For FP16:
- Model: 2 bytes/param
- Gradients: 2 bytes/param
- Adam optimizer: 8 bytes/param (2 states × 4 bytes)
- Activations: Variable (gradient checkpointing helps)

Total ≈ 12 bytes/param for full fine-tuning
```

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Tokenization** | Text → IDs using BPE/WordPiece |
| **HuggingFace** | transformers + datasets + peft + trl |
| **TinyGPT** | ~2,500 params demonstrates all concepts |
| **LoRA** | W' = W + BA, train only A,B (0.1-1%) |
| **SFT** | Train on instruction-response pairs |
| **Evaluation** | Perplexity, ROUGE, BERTScore |

---

## Next Steps

- [03_inferencing.md](03_inferencing.md) - Deploy your fine-tuned model

