import tensorflow as tf

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=seq_len,
            output_dim=d_model
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)
        return x + pos_embeddings

embedding_dim = 4
seq_len = 10

embedding = PositionalEmbedding(seq_len, embedding_dim)

x = tf.random.normal((1, seq_len, embedding_dim))

print(x.shape)
print(x)

y = embedding(x)
print(y.shape)
print(y)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ])

    def call(self, x):
        return self.net(x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, d_ff):
        super().__init__()
        self.pos_emb = PositionalEmbedding(seq_len, d_model)

        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=d_model
        )

        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        # Add position info
        x = self.pos_emb(x)

        # Attention + residual
        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)

        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

model = tf.keras.Sequential([
    embedding,
    TransformerBlock(seq_len, embedding_dim, 128)
])

x = tf.random.normal((1, seq_len, embedding_dim))
y = model(x)
print(y.shape)
print(y)  