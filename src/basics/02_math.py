import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print(probs)
print(np.sum(probs))  # must be 1

true_label = 0  # correct class index
predicted_probs = np.array([0.7, 0.2, 0.1])

loss = -np.log(predicted_probs[true_label])
print(loss)

# softmax + loss + gradient descent example below
def softmax_loss_gradient(scores, true_label):
    probs = softmax(scores)
    loss = -np.log(probs[true_label])
    gradient = probs.copy()
    gradient[true_label] -= 1
    return loss, gradient

# iteratively update the scores
scores = np.array([2.0, 1.0, 0.1])
for i in range(10):
    loss, gradient = softmax_loss_gradient(scores, 0)
    scores -= 0.1 * gradient
    print(scores)