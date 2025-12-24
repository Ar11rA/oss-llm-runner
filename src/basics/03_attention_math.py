import numpy as np

tokens = np.array([
    [1.0, 0.0, 1.0, 0.0],   # token 1
    [0.0, 1.0, 1.0, 0.0],   # token 2
    [0.0, 1.0, 0.0, 1.0]    # token 3
])

print(tokens.shape)  # (3, 4)

W_q = np.random.randn(4, 4)
W_k = np.random.randn(4, 4)
W_v = np.random.randn(4, 4)

Q = tokens @ W_q
K = tokens @ W_k
V = tokens @ W_v

print(Q.shape, K.shape, V.shape)

print("Q:")
print(Q)
print("K:")
print(K)
print("V:")
print(V)

scores = Q @ K.T
print(scores.shape) 
print("Scores:")
print(scores)

def stable_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = stable_softmax(scores)
print("Attention Weights:")
print(attention_weights)

output = attention_weights @ V
print(output.shape)  # (3, 4)
print("Output:")
print(output)