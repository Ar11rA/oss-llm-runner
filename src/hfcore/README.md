# HFCore: LLM Learning from Scratch

This directory contains implementations for understanding Large Language Models (LLMs) from first principles.

---

## Files

| File | Description |
|------|-------------|
| `00_tinygpt.py` | Minimal GPT implementation from scratch using TensorFlow/Keras |
| `01_transformers.py` | Using HuggingFace Transformers library for inference |

---

# TinyGPT: Complete Detailed Explanation

## 🎯 Big Picture

TinyGPT is a **character-level language model** that learns to predict the next character given previous characters. It's trained on "hello hello" and learns to continue patterns like "hell" → "hello hello...".

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TinyGPT Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│  "hello hello"                                                          │
│       ↓ tokenize                                                        │
│  [2,1,3,3,4,0,2,1,3,3,4]                                               │
│       ↓ sliding window                                                  │
│  X,y training pairs                                                     │
│       ↓ train 200 epochs                                                │
│  TinyGPT Model (2,453 params)                                          │
│       ↓ generate                                                        │
│  "hello hello hello..."                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Part 1: Data Preparation

### 1.1 Tokenization

```python
text = "hello hello"
chars = sorted(list(set(text)))  # [' ', 'e', 'h', 'l', 'o']
vocab_size = len(chars)          # 5
```

**What's happening:**
- Extract unique characters
- Sort them for consistency
- This is our **vocabulary** of 5 tokens

**Mappings:**
```python
stoi = {ch: i for i, ch in enumerate(chars)}
# {' ': 0, 'e': 1, 'h': 2, 'l': 3, 'o': 4}

itos = {i: ch for ch, i in stoi.items()}
# {0: ' ', 1: 'e', 2: 'h', 3: 'l', 4: 'o'}
```

**Encoding the text:**
```python
encoded = np.array([stoi[c] for c in text])
# "hello hello" → [2, 1, 3, 3, 4, 0, 2, 1, 3, 3, 4]
#                  h  e  l  l  o     h  e  l  l  o
```

### 1.2 Creating Training Data

```python
seq_len = 4  # Context window size
```

**Sliding window to create X,y pairs:**
```python
for i in range(len(encoded) - seq_len):
    X.append(encoded[i:i+seq_len])      # Input: 4 tokens
    y.append(encoded[i+1:i+seq_len+1])  # Target: shifted by 1
```

**Visual of sliding window:**
```
encoded: [2, 1, 3, 3, 4, 0, 2, 1, 3, 3, 4]
          h  e  l  l  o     h  e  l  l  o

Window 1: X=[2,1,3,3]  y=[1,3,3,4]   "hell" → "ello"
Window 2: X=[1,3,3,4]  y=[3,3,4,0]   "ello" → "llo "
Window 3: X=[3,3,4,0]  y=[3,4,0,2]   "llo " → "lo h"
Window 4: X=[3,4,0,2]  y=[4,0,2,1]   "lo h" → "o he"
Window 5: X=[4,0,2,1]  y=[0,2,1,3]   "o he" → " hel"
Window 6: X=[0,2,1,3]  y=[2,1,3,3]   " hel" → "hell"
Window 7: X=[2,1,3,3]  y=[1,3,3,4]   "hell" → "ello"

X.shape = (7, 4)   # 7 samples, 4 tokens each
y.shape = (7, 4)   # 7 targets, 4 tokens each
```

**Why y is shifted by 1:**
```
X: [h, e, l, l]
y: [e, l, l, o]
    ↑  ↑  ↑  ↑
Position 0: given 'h', predict 'e'
Position 1: given 'h','e', predict 'l'
Position 2: given 'h','e','l', predict 'l'
Position 3: given 'h','e','l','l', predict 'o'
```

---

## 🏗️ Part 2: Model Architecture

### 2.1 Hyperparameters

```python
d_model = 16      # Embedding dimension
num_heads = 2     # Attention heads
d_ff = 32         # FFN hidden size (2× d_model)
num_layers = 1    # Number of transformer blocks
```

### 2.2 TokenAndPositionEmbedding

```python
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, vocab_size, d_model):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_emb = tf.keras.layers.Embedding(seq_len, d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1])
        return self.token_emb(x) + self.pos_emb(positions)
```

**Two embedding tables:**

| Layer | Shape | Parameters | Purpose |
|-------|-------|------------|---------|
| `token_emb` | (5, 16) | 80 | Convert token ID → 16-dim vector |
| `pos_emb` | (4, 16) | 64 | Convert position → 16-dim vector |

**What happens:**
```
Input: [2, 1, 3, 3] (token IDs for "hell")

Token embeddings:
  2 → [0.12, -0.34, ..., 0.56]  (16 values)
  1 → [0.45, 0.23, ..., -0.12]
  3 → [0.78, -0.91, ..., 0.34]
  3 → [0.78, -0.91, ..., 0.34]

Position embeddings:
  0 → [0.11, 0.22, ..., 0.33]
  1 → [0.44, -0.55, ..., 0.66]
  2 → [-0.77, 0.88, ..., 0.99]
  3 → [0.10, 0.20, ..., 0.30]

Output = token_emb + pos_emb  (element-wise)
Shape: (batch, 4, 16)
```

**Why add positions?**
Transformers process all tokens in parallel — they have no inherent sense of order. Position embeddings tell the model WHERE each token is.

---

### 2.3 TransformerBlock

```python
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,           # 2 heads
            key_dim=d_model // num_heads   # 8 dims per head
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="relu"),  # 16 → 32
            tf.keras.layers.Dense(d_model),                   # 32 → 16
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
```

**Components:**

| Component | Parameters | Purpose |
|-----------|------------|---------|
| Multi-Head Attention | 1,088 | Let tokens communicate |
| Feed-Forward Network | 1,072 | Process each token independently |
| LayerNorm ×2 | 64 | Stabilize training |

**The call method:**
```python
def call(self, x):
    # Sub-layer 1: Multi-Head Self-Attention
    attn_out = self.attn(x, x, use_causal_mask=True)
    x = self.norm1(x + attn_out)  # Residual + LayerNorm

    # Sub-layer 2: Feed-Forward Network
    ffn_out = self.ffn(x)
    return self.norm2(x + ffn_out)  # Residual + LayerNorm
```

**Multi-Head Attention Detail:**
```
Input x: (batch, 4, 16)
         │
    ┌────┴────┬────────────┐
    ↓         ↓            ↓
   W_Q       W_K          W_V
    ↓         ↓            ↓
Q:(4,16)   K:(4,16)    V:(4,16)
    │         │
    ↓ reshape for 2 heads
Q:(2,4,8)  K:(2,4,8)    V:(2,4,8)
    │         │
    └────┬────┘
         ↓
    Q @ K^T / √8   →   (2, 4, 4) attention scores
         │
         ↓ apply causal mask
    ┌─────────────────────┐
    │ 1.0  -∞   -∞   -∞   │  Position 0 sees only itself
    │ 0.5  0.5  -∞   -∞   │  Position 1 sees 0,1
    │ 0.3  0.3  0.4  -∞   │  Position 2 sees 0,1,2
    │ 0.2  0.2  0.3  0.3  │  Position 3 sees all
    └─────────────────────┘
         │
         ↓ softmax (row-wise)
    attention weights (2, 4, 4)
         │
         ↓ @ V
    output (2, 4, 8)
         │
         ↓ concat heads + project
    output (4, 16)
```

**Causal Mask Purpose:**
Prevents position N from "seeing" positions N+1, N+2, ... (the future). Essential for autoregressive generation.

**Feed-Forward Network Detail:**
```
Input: (4, 16)
         │
         ↓ Dense(32, relu)
       (4, 32)  ← expand
         │
         ↓ Dense(16)
       (4, 16)  ← contract back
         │
       Output
```

Each position processed independently. The expansion allows learning more complex patterns.

**Residual Connections:**
```python
x = self.norm1(x + attn_out)  # x + attention output
```
- Adds input directly to output
- Helps gradient flow during training
- Layer learns "what to add" not "what to output"

---

### 2.4 TinyGPT Model

```python
class TinyGPT(tf.keras.Model):
    def __init__(self, seq_len, vocab_size, d_model, num_heads, d_ff):
        super().__init__()
        self.embed = TokenAndPositionEmbedding(seq_len, vocab_size, d_model)
        self.block = TransformerBlock(d_model, num_heads, d_ff)
        self.lm_head = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embed(x)      # (7, 4) → (7, 4, 16)
        x = self.block(x)      # (7, 4, 16) → (7, 4, 16)
        return self.lm_head(x) # (7, 4, 16) → (7, 4, 5)
```

**Complete forward pass:**
```
Input: (7, 4) token IDs
         │
         ↓ embed
       (7, 4, 16)  ← each token is now 16-dim vector with position info
         │
         ↓ block (attention + FFN)
       (7, 4, 16)  ← contextualized representations
         │
         ↓ lm_head (Dense 16→5)
       (7, 4, 5)   ← logits: 5 scores per position
         │
       For each of 4 positions, 5 scores predicting next token
```

**LM Head:**
```python
self.lm_head = tf.keras.layers.Dense(vocab_size)  # 16 → 5
```
- Simple linear layer: `output = hidden @ W + b`
- W shape: (16, 5) = 80 parameters
- b shape: (5,) = 5 parameters
- Converts hidden state → vocabulary logits

---

## 📊 Parameter Count

```
┌────────────────────────────────────────────────────────┐
│ Component                    │ Shape        │ Params   │
├──────────────────────────────┼──────────────┼──────────┤
│ token_emb                    │ (5, 16)      │ 80       │
│ pos_emb                      │ (4, 16)      │ 64       │
├──────────────────────────────┼──────────────┼──────────┤
│ MHA: W_Q                     │ (16,16)+16   │ 272      │
│ MHA: W_K                     │ (16,16)+16   │ 272      │
│ MHA: W_V                     │ (16,16)+16   │ 272      │
│ MHA: W_O                     │ (16,16)+16   │ 272      │
├──────────────────────────────┼──────────────┼──────────┤
│ FFN: Dense 16→32             │ (16,32)+32   │ 544      │
│ FFN: Dense 32→16             │ (32,16)+16   │ 528      │
├──────────────────────────────┼──────────────┼──────────┤
│ LayerNorm1 (γ,β)             │ (16,)+(16,)  │ 32       │
│ LayerNorm2 (γ,β)             │ (16,)+(16,)  │ 32       │
├──────────────────────────────┼──────────────┼──────────┤
│ lm_head                      │ (16,5)+5     │ 85       │
├──────────────────────────────┼──────────────┼──────────┤
│ TOTAL                        │              │ 2,453    │
└────────────────────────────────────────────────────────┘
```

---

## 🎓 Part 3: Training

```python
model = TinyGPT(seq_len, vocab_size, d_model, num_heads, d_ff)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
```

**Loss Function:**
- **Sparse**: Labels are integers (0,1,2,3,4), not one-hot
- **Categorical**: Multi-class (5 possible tokens)
- **CrossEntropy**: Penalizes wrong predictions
- **from_logits=True**: Model outputs raw scores, loss applies softmax

**Training Loop:**
```python
for epoch in range(200):
    with tf.GradientTape() as tape:
        logits = model(X)        # Forward pass: (7,4) → (7,4,5)
        loss = loss_fn(y, logits) # Compare predictions to targets

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

**What happens each epoch:**
```
1. FORWARD PASS
   X (7,4) → model → logits (7,4,5)
   
2. LOSS COMPUTATION
   For each position, compare predicted distribution to target:
   Loss = -log(probability of correct token)
   
   Example: Position predicts [0.1, 0.1, 0.1, 0.2, 0.5]
            Target is 'o' (index 4)
            Loss = -log(0.5) = 0.69
   
3. BACKWARD PASS
   Compute ∂Loss/∂weight for every weight
   
4. UPDATE
   weight = weight - learning_rate × gradient
```

**Learning from ALL positions simultaneously:**
```
Input:  h  e  l  l
        ↓  ↓  ↓  ↓
Logits: L0 L1 L2 L3
        ↓  ↓  ↓  ↓
Target: e  l  l  o

Loss = CrossEntropy(L0, e) + CrossEntropy(L1, l) + 
       CrossEntropy(L2, l) + CrossEntropy(L3, o)
       
All 4 predictions contribute! (4× learning signal per sample)
```

---

## 🚀 Part 4: Generation

```python
def generate(start, length=20):
    context = [stoi[c] for c in start]  # "hell" → [2,1,3,3]

    for _ in range(length):
        x = np.array(context[-seq_len:])[None, :]  # Last 4 tokens, add batch dim
        logits = model(x)                           # (1,4,5)
        next_id = tf.argmax(logits[0, -1]).numpy()  # Get last position, argmax
        context.append(next_id)

    return "".join(itos[i] for i in context)
```

**Step-by-step generation:**
```
Start: context = [2,1,3,3] → "hell"

STEP 1:
  Input:  [2,1,3,3]
  Model → logits[0,-1] = [0.1, 0.1, 0.1, 0.2, 0.5]
                                              ↑
                         argmax = 4 → 'o'
  context = [2,1,3,3,4] → "hello"

STEP 2:
  Input:  [1,3,3,4] (last 4)
  Model → argmax = 0 → ' '
  context = [2,1,3,3,4,0] → "hello "

STEP 3:
  Input:  [3,3,4,0]
  Model → argmax = 2 → 'h'
  context = [2,1,3,3,4,0,2] → "hello h"

... repeat 20 times ...

Output: "hello hello hello hel..."
```

**Why `logits[0, -1]`:**
```
logits shape: (1, 4, 5)
               │  │  └── 5 token scores
               │  └──── 4 positions
               └─────── 1 batch

logits[0]     → first batch → shape (4, 5)
logits[0, -1] → last position → shape (5,)

We want the prediction AFTER all input tokens = last position
```

---

## 🔄 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  "hello hello"                                                               │
│        ↓                                                                     │
│  [2,1,3,3,4,0,2,1,3,3,4]  (tokenize)                                        │
│        ↓                                                                     │
│  X: (7, 4)     y: (7, 4)  (sliding window)                                  │
│        ↓              ↓                                                      │
│  ┌──────────────────────────────────────────┐                               │
│  │ TokenAndPositionEmbedding                │                               │
│  │ - token lookup (5×16 table)              │                               │
│  │ - position lookup (4×16 table)           │                               │
│  │ - add together                           │                               │
│  └──────────────────────────────────────────┘                               │
│        ↓ (7, 4, 16)                                                         │
│  ┌──────────────────────────────────────────┐                               │
│  │ TransformerBlock                         │                               │
│  │ ┌────────────────────────────────────┐   │                               │
│  │ │ MultiHeadAttention (2 heads)       │   │                               │
│  │ │ - Q,K,V projections                │   │                               │
│  │ │ - scaled dot-product attention     │   │                               │
│  │ │ - causal mask (no future peeking)  │   │                               │
│  │ │ - concat heads + output project    │   │                               │
│  │ └────────────────────────────────────┘   │                               │
│  │        ↓ + residual + LayerNorm          │                               │
│  │ ┌────────────────────────────────────┐   │                               │
│  │ │ FFN: Dense(32,relu) → Dense(16)    │   │                               │
│  │ └────────────────────────────────────┘   │                               │
│  │        ↓ + residual + LayerNorm          │                               │
│  └──────────────────────────────────────────┘                               │
│        ↓ (7, 4, 16)                                                         │
│  ┌──────────────────────────────────────────┐                               │
│  │ LM Head: Dense(5)                        │                               │
│  │ - linear projection 16 → 5               │                               │
│  │ - outputs raw logits                     │                               │
│  └──────────────────────────────────────────┘                               │
│        ↓ (7, 4, 5)                         ↓ y: (7, 4)                       │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │ SparseCategoricalCrossEntropy                        │                   │
│  │ - softmax(logits) → probabilities                    │                   │
│  │ - -log(prob of correct token)                        │                   │
│  │ - average over all positions                         │                   │
│  └──────────────────────────────────────────────────────┘                   │
│        ↓                                                                     │
│     Loss (scalar)                                                            │
│        ↓                                                                     │
│  ┌──────────────────────────────────────────┐                               │
│  │ Backpropagation                          │                               │
│  │ - compute gradients for all 2,453 params │                               │
│  └──────────────────────────────────────────┘                               │
│        ↓                                                                     │
│  ┌──────────────────────────────────────────┐                               │
│  │ Adam Optimizer                           │                               │
│  │ - param = param - lr × gradient          │                               │
│  └──────────────────────────────────────────┘                               │
│                                                                              │
│  Repeat 200 epochs until loss ≈ 0                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              GENERATION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  "hell"                                                                      │
│     ↓                                                                        │
│  [2,1,3,3] ─────────────────────────────────────────────────────────┐       │
│     │                                                                │       │
│     ↓                                                                │       │
│  Model(x) → logits (1,4,5)                                          │       │
│     │                                                                │       │
│     ↓                                                                │       │
│  logits[0,-1] → [0.1, 0.1, 0.1, 0.2, 0.5]                           │       │
│     │                                                                │       │
│     ↓ argmax                                                         │       │
│     4 → 'o'                                                          │       │
│     │                                                                │       │
│     ↓ append                                                         │       │
│  [2,1,3,3,4] → "hello" ─────────────────────────────────────────────┤       │
│     │                                                                │       │
│     │  (take last 4: [1,3,3,4])                                      │       │
│     │                                                                │       │
│     └────────────────────────────── LOOP 20 times ──────────────────┘       │
│                                                                              │
│  Final: "hello hello hello hello h..."                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Concepts Summary

| Concept | What It Does | Where in Code |
|---------|--------------|---------------|
| **Tokenization** | Text → integer IDs | Lines 5-14 |
| **Vocabulary** | Mapping between chars and IDs | `stoi`, `itos` |
| **Sliding Window** | Create input-target pairs | Lines 21-23 |
| **Token Embedding** | ID → learned vector | Line 40 |
| **Position Embedding** | Position → learned vector | Line 41 |
| **Self-Attention** | Tokens communicate | Line 62 |
| **Causal Mask** | No peeking at future | `use_causal_mask=True` |
| **Multi-Head** | Multiple attention patterns | `num_heads=2` |
| **FFN** | Per-position processing | Lines 54-57 |
| **Residual Connection** | `x + output` | Lines 63, 66 |
| **LayerNorm** | Normalize activations | Lines 58-59, 63, 66 |
| **LM Head** | Hidden → vocab logits | Line 73 |
| **Cross-Entropy Loss** | Penalize wrong predictions | Line 82 |
| **Backpropagation** | Compute gradients | Line 90 |
| **Autoregressive Gen** | Predict one token at a time | Lines 102-106 |
| **Greedy Decoding** | Pick highest probability | Line 105 |

---

## 📈 What the Model Learns

After 200 epochs on "hello hello":
- Learns that 'h' is usually followed by 'e'
- Learns that 'hel' is followed by 'l'
- Learns that 'hell' is followed by 'o'
- Learns that 'hello' is followed by ' '
- Learns that 'hello ' is followed by 'h'

The attention mechanism learns **which previous characters matter** for predicting the next one!

---

## 🔢 The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right) V$$

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I offer?"
- **V** (Value): "What information do I carry?"
- **mask**: Prevents looking at future tokens
- **√d_k**: Scaling factor to prevent gradient saturation

---

## 📚 Comparison with Real Models

| Model | Parameters | Layers | d_model | Heads |
|-------|------------|--------|---------|-------|
| **TinyGPT** | 2,453 | 1 | 16 | 2 |
| GPT-1 | 117M | 12 | 768 | 12 |
| GPT-2 | 1.5B | 48 | 1600 | 25 |
| GPT-3 | 175B | 96 | 12288 | 96 |
| GPT-4 | ~1.8T | ? | ? | ? |

TinyGPT is **~48,000× smaller** than GPT-1, but uses the **exact same architecture concepts**!

