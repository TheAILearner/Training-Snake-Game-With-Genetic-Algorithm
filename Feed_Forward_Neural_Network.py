import numpy as np

n_x = 7
n_h = 9
n_h2 = 15
n_y = 3
W1_shape = (9,7)
W2_shape = (15,9)
W3_shape = (3,15)

def get_weights_from_encoded(individual):
    W1 = individual[0:W1_shape[0] * W1_shape[1]]
    W2 = individual[W1_shape[0] * W1_shape[1]:W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]]
    W3 = individual[W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]:]

    return (
    W1.reshape(W1_shape[0], W1_shape[1]), W2.reshape(W2_shape[0], W2_shape[1]), W3.reshape(W3_shape[0], W3_shape[1]))


def softmax(z):
    s = np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1, 1)

    return s


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s


def forward_propagation(X, individual):
    W1, W2, W3 = get_weights_from_encoded(individual)

    Z1 = np.matmul(W1, X.T)
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1)
    A2 = np.tanh(Z2)
    Z3 = np.matmul(W3, A2)
    A3 = softmax(Z3)
    return A3