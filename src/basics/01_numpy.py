import numpy as np

# Scalar
a = np.array(3)

# Vector (1D)
v = np.array([1, 2, 3])

# Matrix (2D)
M = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(a.shape)  # ()
print(v.shape)  # (3,)
print(M.shape) 

A = np.array([
    [1, 2],
    [3, 4]
])  # shape (2, 2)

B = np.array([
    [5, 6],
    [7, 8]
])  # shape (2, 2)

C = A @ B
print(C)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

dot = np.dot(x, y)
print(dot)

X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # shape (2, 3)

bias = np.array([10, 20, 30])  # shape (3,)

Y = X + bias
print(Y)