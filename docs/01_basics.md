# Foundations: From NumPy to Attention

This document covers the mathematical foundations you need to understand transformers, from basic tensor operations to multi-head attention.

---

## Table of Contents

1. [Tensors and NumPy](#tensors-and-numpy)
2. [Matrix Multiplication in ML](#matrix-multiplication-in-ml)
3. [Embeddings](#embeddings)
4. [The Attention Mechanism](#the-attention-mechanism)
5. [Positional Encoding](#positional-encoding)
6. [Multi-Head Attention](#multi-head-attention)
7. [Layer Normalization & Residual Connections](#layer-normalization--residual-connections)
8. [Code Walkthrough](#code-walkthrough)

---

## Tensors and NumPy

### What is a Tensor?

A **tensor** is a multi-dimensional array. It's the fundamental data structure in deep learning.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Tensor Dimensions                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Scalar (0D):     3                           shape: ()                     │
│                                                                             │
│  Vector (1D):     [1, 2, 3]                   shape: (3,)                   │
│                                                                             │
│  Matrix (2D):     [[1, 2, 3],                 shape: (2, 3)                 │
│                    [4, 5, 6]]                                                │
│                                                                             │
│  3D Tensor:       [[[1,2], [3,4]],            shape: (2, 2, 2)              │
│                    [[5,6], [7,8]]]                                          │
│                                                                             │
│  4D Tensor:       Batch of images             shape: (batch, height,        │
│                                                        width, channels)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Common Shapes in LLMs

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Token IDs | `(batch, seq_len)` | Batch of sequences |
| Embeddings | `(batch, seq_len, d_model)` | Each token → vector |
| Attention Weights | `(batch, heads, seq_len, seq_len)` | Token-to-token attention |
| Logits | `(batch, seq_len, vocab_size)` | Next-token probabilities |

### Broadcasting

Broadcasting allows operations between tensors of different shapes:

```python
import numpy as np

X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # shape (2, 3)

bias = np.array([10, 20, 30])  # shape (3,)

Y = X + bias  # Broadcasting: (2, 3) + (3,) → (2, 3)
# [[11, 22, 33],
#  [14, 25, 36]]
```

**How it works:**
```
    X:      (2, 3)
    bias:      (3,)  ← Broadcast to (1, 3) then (2, 3)
    Result: (2, 3)
```

---

## Matrix Multiplication in ML

Matrix multiplication (`@` or `matmul`) is the core operation in neural networks. Almost everything is a matmul!

### The Shapes Rule

```
(a, b) @ (b, c) → (a, c)
        ↑
    Must match!
```

### Why Matmul is Central

Every linear layer is a matmul:

```python
# Dense layer: output = input @ weights + bias
input_dim = 4
output_dim = 8
batch_size = 32
seq_len = 10

X = np.random.randn(batch_size, seq_len, input_dim)  # (32, 10, 4)
W = np.random.randn(input_dim, output_dim)            # (4, 8)

Y = X @ W  # (32, 10, 4) @ (4, 8) → (32, 10, 8)
```

### Computational Complexity

For matrices A(m×n) and B(n×p):
- **Operations**: O(m × n × p)
- **Memory**: O(m × p) for result

For attention with sequence length `n`:
- **QK^T**: O(n² × d) — this is why long contexts are expensive!

---

## Embeddings

### The Problem

Neural networks need numbers, but text is... text.

```
"hello" → ??? → Neural Network
```

### The Solution: Embedding Tables

An **embedding** is a learned lookup table that converts discrete tokens to continuous vectors.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Embedding Lookup                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Vocabulary: {"cat": 0, "dog": 1, "eats": 2, "fish": 3}                   │
│                                                                             │
│   Embedding Table (vocab_size × d_model):                                   │
│                                                                             │
│       ID 0 (cat):  [0.12, -0.34, 0.56, 0.78]                               │
│       ID 1 (dog):  [0.45, 0.23, -0.12, 0.89]                               │
│       ID 2 (eats): [0.78, -0.91, 0.34, 0.12]                               │
│       ID 3 (fish): [-0.23, 0.67, 0.45, -0.56]                              │
│                                                                             │
│   Input: [0, 2, 3]  ("cat eats fish")                                      │
│                                                                             │
│   Output: [[0.12, -0.34, 0.56, 0.78],    ← Look up ID 0                    │
│            [0.78, -0.91, 0.34, 0.12],    ← Look up ID 2                    │
│            [-0.23, 0.67, 0.45, -0.56]]   ← Look up ID 3                    │
│                                                                             │
│   Shape: (3,) → (3, 4)                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Embeddings Work

1. **Learned representations**: Similar words get similar vectors
2. **Semantic relationships**: `king - man + woman ≈ queen`
3. **Dense vs sparse**: 4 numbers per word vs 50,000-dimensional one-hot

---

## The Attention Mechanism

Attention is the breakthrough that makes transformers work. It allows each token to "look at" all other tokens and decide what to focus on.

### The Intuition

Imagine reading a sentence:

```
"The cat sat on the mat because it was tired"
                                 ↑
                          What does "it" refer to?
```

To understand "it", you need to **attend** to "cat". Attention learns these relationships automatically.

### Query, Key, Value

The attention mechanism uses three projections:

| Component | Intuition | Role |
|-----------|-----------|------|
| **Query (Q)** | "What am I looking for?" | The current token asking a question |
| **Key (K)** | "What do I contain?" | Other tokens advertising their content |
| **Value (V)** | "What info do I provide?" | The actual information to retrieve |

### Step-by-Step Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Attention Computation                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input X: (seq_len, d_model) = (3, 4)                                      │
│                                                                             │
│   Step 1: Create Q, K, V                                                    │
│   ─────────────────────────                                                 │
│       Q = X @ W_q    (3, 4) @ (4, 4) → (3, 4)                              │
│       K = X @ W_k    (3, 4) @ (4, 4) → (3, 4)                              │
│       V = X @ W_v    (3, 4) @ (4, 4) → (3, 4)                              │
│                                                                             │
│   Step 2: Compute attention scores                                          │
│   ────────────────────────────────                                          │
│       scores = Q @ K^T                                                      │
│       (3, 4) @ (4, 3) → (3, 3)                                             │
│                                                                             │
│       scores[i][j] = how much token i attends to token j                   │
│                                                                             │
│   Step 3: Scale                                                             │
│   ─────────                                                                 │
│       scores = scores / √d_k                                               │
│                                                                             │
│       Why? Prevents softmax saturation when d is large                     │
│                                                                             │
│   Step 4: Apply causal mask (for language models)                          │
│   ─────────────────────────────────────────────                            │
│       mask = [[0, -∞, -∞],                                                  │
│               [0,  0, -∞],                                                  │
│               [0,  0,  0]]                                                  │
│                                                                             │
│       scores = scores + mask                                                │
│                                                                             │
│   Step 5: Softmax                                                           │
│   ──────────────                                                            │
│       weights = softmax(scores, axis=-1)                                   │
│                                                                             │
│       Each row sums to 1.0                                                 │
│                                                                             │
│   Step 6: Weighted sum of values                                           │
│   ──────────────────────────────                                            │
│       output = weights @ V                                                 │
│       (3, 3) @ (3, 4) → (3, 4)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Code Implementation

```python
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    
    # Step 1: Compute scores
    scores = Q @ K.T  # (seq, d) @ (d, seq) → (seq, seq)
    
    # Step 2: Scale
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask (optional)
    if mask is not None:
        scores = scores + mask
    
    # Step 4: Softmax
    def softmax(x):
        x = x - np.max(x, axis=-1, keepdims=True)  # Numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    weights = softmax(scores)
    
    # Step 5: Weighted sum
    output = weights @ V  # (seq, seq) @ (seq, d) → (seq, d)
    
    return output, weights
```

### Why Scale by √d_k?

Without scaling, when `d_k` is large:
- Dot products grow large
- Softmax becomes nearly one-hot (saturated)
- Gradients vanish

Scaling by `√d_k` keeps the variance of scores at ~1.

---

## Positional Encoding

### The Problem

Transformers process all tokens in parallel. They have no inherent notion of order!

```
"The cat sat on the mat"
"mat the on sat cat The"
          ↑
Same embeddings, different meaning!
```

### The Solution: Add Position Information

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Positional Encoding                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Token Embeddings:                                                         │
│       "cat"  → [0.12, -0.34, 0.56, 0.78]                                   │
│       "eats" → [0.78, -0.91, 0.34, 0.12]                                   │
│       "fish" → [-0.23, 0.67, 0.45, -0.56]                                  │
│                                                                             │
│   Position Embeddings:                                                      │
│       pos 0  → [0.11, 0.22, 0.33, 0.44]                                    │
│       pos 1  → [0.55, 0.66, 0.77, 0.88]                                    │
│       pos 2  → [0.99, 0.11, 0.22, 0.33]                                    │
│                                                                             │
│   Final Embeddings = Token + Position:                                      │
│       "cat" at pos 0  → [0.23, -0.12, 0.89, 1.22]                          │
│       "eats" at pos 1 → [1.33, -0.25, 1.11, 1.00]                          │
│       "fish" at pos 2 → [0.76, 0.78, 0.67, -0.23]                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Types of Position Encoding

| Type | Description | Used By |
|------|-------------|---------|
| **Sinusoidal** | Fixed sine/cosine patterns | Original Transformer |
| **Learned** | Trained embedding table | GPT, BERT |
| **Rotary (RoPE)** | Rotation-based, relative | Llama, Qwen |
| **ALiBi** | Attention bias, no embedding | BLOOM |

### Learned Positional Embedding

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
```

---

## Multi-Head Attention

### The Problem with Single-Head Attention

One attention head can only learn one type of relationship. But language has many!

- Syntactic: subject-verb agreement
- Semantic: pronoun resolution
- Positional: nearby words
- Long-range: topic coherence

### The Solution: Multiple Heads

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Head Attention                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input X: (batch, seq_len, d_model)                                        │
│         │                                                                   │
│         ├─────────┬─────────┬─────────┬─────────┐                          │
│         ↓         ↓         ↓         ↓         │                          │
│      Head 1    Head 2    Head 3    Head 4       │  (h=4 heads)             │
│      (d_k)     (d_k)     (d_k)     (d_k)        │  d_k = d_model/h        │
│         │         │         │         │         │                          │
│         ↓         ↓         ↓         ↓         │                          │
│    Attention  Attention  Attention  Attention   │                          │
│         │         │         │         │         │                          │
│         └─────────┴─────────┴─────────┘         │                          │
│                     │                           │                          │
│                     ↓ Concatenate               │                          │
│              (batch, seq_len, d_model)          │                          │
│                     │                           │                          │
│                     ↓ Output Projection (W_o)   │                          │
│              (batch, seq_len, d_model)          │                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Shape Analysis

For `d_model=512` and `num_heads=8`:

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input | (batch, seq, 512) | |
| Q, K, V per head | (batch, seq, 64) | 512/8 = 64 |
| Attention per head | (batch, seq, 64) | |
| Concatenated | (batch, seq, 512) | 8 × 64 = 512 |
| After W_o | (batch, seq, 512) | |

### What Different Heads Learn

Research has shown different heads specialize:
- **Head 1**: Attends to previous token
- **Head 2**: Attends to sentence start
- **Head 3**: Tracks syntactic relationships
- **Head 4**: Long-range dependencies

---

## Layer Normalization & Residual Connections

### Residual Connections

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Residual Connection                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Without residual:              With residual:                             │
│                                                                             │
│   x ─── Layer ──→ y              x ─┬─ Layer ──┬──→ x + y                  │
│                                     │          │                            │
│                                     └──────────┘                            │
│                                        (skip)                               │
│                                                                             │
│   Problem: Gradients vanish        Solution: Gradient can flow             │
│   through deep networks            directly through skip connection         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why it works:**
- Gradient flows directly: `∂(x+y)/∂x = 1 + ∂y/∂x`
- Network learns "what to add" not "what to output"
- Enables training of very deep networks (100+ layers)

### Layer Normalization

Normalizes activations across the feature dimension:

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**Why normalize?**
- Stabilizes training
- Reduces internal covariate shift
- Enables higher learning rates

### Pre-Norm vs Post-Norm

```
Post-Norm (Original):           Pre-Norm (Modern):
x ─→ Attn ─→ Add ─→ Norm        x ─→ Norm ─→ Attn ─→ Add
                                      │              │
                                      └──────────────┘
```

Most modern LLMs use **Pre-Norm** for training stability.

---

## Code Walkthrough

### File: `00_pandas.py`

Data manipulation basics. Not directly LLM-related but useful for data preparation.

### File: `01_numpy.py`

Tensor operations fundamentals:

```python
import numpy as np

# Scalar, Vector, Matrix
a = np.array(3)           # shape: ()
v = np.array([1, 2, 3])   # shape: (3,)
M = np.array([[1,2,3],
              [4,5,6]])   # shape: (2, 3)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Matrix product

# Dot product
dot = np.dot(x, y)  # Sum of element-wise products

# Broadcasting
X = np.array([[1,2,3], [4,5,6]])  # (2, 3)
bias = np.array([10, 20, 30])     # (3,)
Y = X + bias                       # (2, 3) + (3,) → (2, 3)
```

### File: `03_attention_math.py`

Step-by-step attention implementation:

```python
import numpy as np

# Input: 3 tokens, each with 4 features
tokens = np.array([
    [1.0, 0.0, 1.0, 0.0],   # token 1
    [0.0, 1.0, 1.0, 0.0],   # token 2
    [0.0, 1.0, 0.0, 1.0]    # token 3
])

# Random projection matrices
W_q = np.random.randn(4, 4)
W_k = np.random.randn(4, 4)
W_v = np.random.randn(4, 4)

# Create Q, K, V
Q = tokens @ W_q  # (3, 4)
K = tokens @ W_k  # (3, 4)
V = tokens @ W_v  # (3, 4)

# Compute attention scores
scores = Q @ K.T  # (3, 3)

# Softmax
def stable_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = stable_softmax(scores)  # (3, 3)

# Weighted sum of values
output = attention_weights @ V  # (3, 4)
```

### File: `04_attention_impl.py`

Full attention with embeddings and training:

```python
import tensorflow as tf

# Vocabulary and embeddings
vocab = {"cat": 0, "dog": 1, "eats": 2, "fish": 3, "meat": 4}
embedding = tf.keras.layers.Embedding(len(vocab), 4)

# Projection layers
Wq = tf.keras.layers.Dense(4, use_bias=False)
Wk = tf.keras.layers.Dense(4, use_bias=False)
Wv = tf.keras.layers.Dense(4, use_bias=False)

def attention_block(X):
    Q = Wq(X)
    K = Wk(X)
    V = Wv(X)
    
    scores = tf.matmul(Q, K, transpose_b=True)
    scores /= tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
    
    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, V)
    
    return output, weights
```

### File: `05_attention_positional_encoding.py`

Adding position information:

```python
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = tf.keras.layers.Embedding(seq_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len)
        return x + self.pos_emb(positions)
```

### File: `06_mha.py`

Multi-head attention using Keras:

```python
import tensorflow as tf

d_model = 8
num_heads = 2
seq_len = 3

# Multi-head attention layer
mha = tf.keras.layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=d_model // num_heads  # d_k per head
)

# Forward pass
X = tf.random.normal((1, seq_len, d_model))
output, attn_scores = mha(X, X, return_attention_scores=True)

# output shape: (1, 3, 8)
# attn_scores shape: (1, 2, 3, 3) = (batch, heads, seq, seq)
```

---

## Summary

| Concept | Key Idea | Formula/Shape |
|---------|----------|---------------|
| **Tensor** | Multi-dimensional array | `(batch, seq, d_model)` |
| **Embedding** | Token ID → Vector | `(vocab, d_model)` lookup |
| **Attention** | Token-to-token focus | `softmax(QK^T/√d) × V` |
| **Position** | Add order information | `token_emb + pos_emb` |
| **Multi-Head** | Parallel attention | `Concat(head_1, ..., head_h) × W_o` |
| **Residual** | Skip connections | `x + Layer(x)` |
| **LayerNorm** | Normalize features | `(x - μ) / σ × γ + β` |

---

## Next Steps

Now that you understand the foundations, proceed to:
- [02_hfcore.md](02_hfcore.md) - Build and train a complete GPT model
