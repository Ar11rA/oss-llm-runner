# Speed up TensorFlow import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf

# Disable eager execution overhead messages
tf.get_logger().setLevel('ERROR')

vocab = {
    "cat": 0,
    "dog": 1,
    "eats": 2,
    "fish": 3,
    "meat": 4
}

# Reverse vocab for converting IDs back to words
id_to_word = {v: k for k, v in vocab.items()}

sentences = [
    ["cat", "eats", "fish"],
    ["dog", "eats", "meat"]
]

token_ids = [[vocab[w] for w in s] for s in sentences]
token_ids = tf.constant(token_ids)

print(token_ids)
print(token_ids.shape)  # (2 sentences, 3 tokens)

embedding_dim = 4

embedding = tf.keras.layers.Embedding(
    input_dim=len(vocab),
    output_dim=embedding_dim
)

X = embedding(token_ids)

print("Embeddings shape:", X.shape)

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

attn_output, attn_weights = attention_block(X)

sentence_repr = tf.reduce_mean(attn_output, axis=1)
print("Sentence representation shape:", sentence_repr.shape)
print("Sentence representation:", sentence_repr)


def loss_fn(repr):
    return tf.reduce_mean(tf.square(repr[0] - repr[1]))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

for step in range(50):
    with tf.GradientTape() as tape:
        X = embedding(token_ids)
        attn_output, _ = attention_block(X)
        sentence_repr = tf.reduce_mean(attn_output, axis=1)
        loss = loss_fn(sentence_repr)

    vars = (
        embedding.trainable_variables +
        Wq.trainable_variables +
        Wk.trainable_variables +
        Wv.trainable_variables
    )

    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.numpy():.4f}")

test_sentence = ["cat", "eats", "meat"]
test_token_ids = tf.constant([[vocab[w] for w in test_sentence]])

print(test_token_ids)        # shape (1, 3)

def predict_sentence_embedding(token_ids, sentence_words):
    X = embedding(token_ids)                 # embeddings
    attn_output, attn_weights = attention_block(X)
    sentence_repr = tf.reduce_mean(attn_output, axis=1)
    return sentence_repr, attn_weights

pred_repr, pred_attn = predict_sentence_embedding(test_token_ids, test_sentence)

# Print sentence in string form
print(f"\nTest sentence: \"{' '.join(test_sentence)}\"")

print("\nPredicted sentence embedding:")
print(pred_repr.numpy())

print("\nAttention weights:")
print(pred_attn.numpy())

# Show attention per word
print("\nAttention breakdown per token:")
attn_matrix = pred_attn.numpy()[0]  # shape (3, 3)
for i, query_word in enumerate(test_sentence):
    print(f"  '{query_word}' attends to:")
    for j, key_word in enumerate(test_sentence):
        print(f"    '{key_word}': {attn_matrix[i][j]:.4f}")
