import tensorflow as tf

d_model = 8
num_heads = 2
seq_len = 3
batch_size = 1

# Dummy input
X = tf.random.normal((batch_size, seq_len, d_model))

print("Input shape:", X.shape)
print("Input:", X)

mha = tf.keras.layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=d_model // num_heads
)

output, attn_scores = mha(
    X, X, return_attention_scores=True
)

print("Output shape:", output.shape)
print("Output:", output)
print("Attention scores shape:", attn_scores.shape)
print("Attention scores:", attn_scores)
