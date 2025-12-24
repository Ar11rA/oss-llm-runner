import numpy as np
import tensorflow as tf


text = "hello hello"
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encoded = np.array([stoi[c] for c in text])
print(encoded)
print(encoded.shape)

seq_len = 4

X = []
y = []

for i in range(len(encoded) - seq_len):
    X.append(encoded[i:i+seq_len])
    y.append(encoded[i+1:i+seq_len+1])

X = np.array(X)
y = np.array(y)

print(X)
print(y)
print(X.shape, y.shape)

d_model = 16
num_heads = 2
d_ff = 32
num_layers = 1

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
            num_heads=num_heads,
            key_dim=d_model // num_heads
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

model = TinyGPT(seq_len, vocab_size, d_model, num_heads, d_ff)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for epoch in range(200):
    with tf.GradientTape() as tape:
        logits = model(X)
        loss = loss_fn(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 20 == 0:
      print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

print("weights:", model.weights)


def generate(start, length=20):
    context = [stoi[c] for c in start]

    for _ in range(length):
        x = np.array(context[-seq_len:])[None, :]
        logits = model(x)
        next_id = tf.argmax(logits[0, -1]).numpy()
        context.append(next_id)

    return "".join(itos[i] for i in context)

print(generate("hell"))
