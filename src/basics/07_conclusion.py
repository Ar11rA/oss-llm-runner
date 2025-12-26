import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf

batch = 1
seq_len = 3
d_model = 4

# Dummy token embeddings
X = tf.constant(
    [[[1., 0., 1., 0.],
      [0., 1., 1., 0.],
      [0., 1., 0., 1.]]]
)

print("Input shape:", X.shape)  # (1, 3, 4)
print("Input:", X)

Wq = tf.keras.layers.Dense(d_model, use_bias=False)
Wk = tf.keras.layers.Dense(d_model, use_bias=False)
Wv = tf.keras.layers.Dense(d_model, use_bias=False)

Q = Wq(X)
K = Wk(X)
V = Wv(X)

scores = tf.matmul(Q, K, transpose_b=True)
scores /= tf.math.sqrt(tf.cast(d_model, tf.float32))

weights = tf.nn.softmax(scores, axis=-1)
output_single = tf.matmul(weights, V)

print("Single-head output shape:", output_single.shape)
print("Single-head output:\n", output_single.numpy())
print("Single-head attention matrix shape:\n", weights.shape)
print("Single-head attention matrix:\n", weights.numpy())

num_heads = 2
head_dim = d_model // num_heads  # 2

mha = tf.keras.layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=head_dim
)

output_multi, attn_scores = mha(
    X, X, return_attention_scores=True
)

print("Multi-head output shape:", output_multi.shape)
print("Multi-head output:\n", output_multi.numpy())
print("Multi-head attention shape:", attn_scores.shape)
print("Multi-head attention:\n", attn_scores.numpy())

# masking for causal attention
output_masked, attn_scores_masked = mha(
    X, X, use_causal_mask=True, return_attention_scores=True
)

print("Masked output shape:", output_masked.shape)
print("Masked output:\n", output_masked.numpy())
print("Masked attention shape:", attn_scores_masked.shape)
print("Masked attention:\n", attn_scores_masked.numpy())