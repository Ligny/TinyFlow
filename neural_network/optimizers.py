import numpy as np # type: ignore

class AdamOptimizer:
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, layer, d_weights: np.ndarray, d_biases: np.ndarray):
        if layer not in self.m:
            self.m[layer] = {"weights": np.zeros_like(d_weights), "biases": np.zeros_like(d_biases)}
            self.v[layer] = {"weights": np.zeros_like(d_weights), "biases": np.zeros_like(d_biases)}
        self.t += 1
        self.m[layer]["weights"] = self.beta1 * self.m[layer]["weights"] + (1 - self.beta1) * d_weights
        self.m[layer]["biases"] = self.beta1 * self.m[layer]["biases"] + (1 - self.beta1) * d_biases

        self.v[layer]["weights"] = self.beta2 * self.v[layer]["weights"] + (1 - self.beta2) * (d_weights ** 2)
        self.v[layer]["biases"] = self.beta2 * self.v[layer]["biases"] + (1 - self.beta2) * (d_biases ** 2)

        m_hat_weights = self.m[layer]["weights"] / (1 - self.beta1 ** self.t)
        m_hat_biases = self.m[layer]["biases"] / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v[layer]["weights"] / (1 - self.beta2 ** self.t)
        v_hat_biases = self.v[layer]["biases"] / (1 - self.beta2 ** self.t)

        layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)