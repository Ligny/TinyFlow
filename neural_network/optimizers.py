import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=5.0, decay=0.0):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
        self.clip_value = clip_value
        self.decay = decay  # Ajout du decay

    def update(self, layer, d_weights, d_biases):
        if layer not in self.m:
            self.m[layer] = {"weights": np.zeros_like(d_weights), "biases": np.zeros_like(d_biases)}
            self.v[layer] = {"weights": np.zeros_like(d_weights), "biases": np.zeros_like(d_biases)}

        self.t += 1

        # **Mise à jour du learning rate avec decay**
        self.learning_rate = self.initial_learning_rate / (1 + self.decay * self.t)

        # Appliquer le clipping des gradients
        d_weights = np.clip(d_weights, -self.clip_value, self.clip_value)
        d_biases = np.clip(d_biases, -self.clip_value, self.clip_value)
        if np.isnan(d_weights).any() or np.isnan(d_biases).any():
            raise ValueError("NaN detected in gradients during optimization!")

        # Mise à jour des moments
        self.m[layer]["weights"] = self.beta1 * self.m[layer]["weights"] + (1 - self.beta1) * d_weights
        self.m[layer]["biases"] = self.beta1 * self.m[layer]["biases"] + (1 - self.beta1) * d_biases

        self.v[layer]["weights"] = self.beta2 * self.v[layer]["weights"] + (1 - self.beta2) * (d_weights ** 2)
        self.v[layer]["biases"] = self.beta2 * self.v[layer]["biases"] + (1 - self.beta2) * (d_biases ** 2)

        # Correction de biais
        m_hat_weights = self.m[layer]["weights"] / (1 - self.beta1 ** self.t)
        m_hat_biases = self.m[layer]["biases"] / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v[layer]["weights"] / (1 - self.beta2 ** self.t)
        v_hat_biases = self.v[layer]["biases"] / (1 - self.beta2 ** self.t)

        # Mise à jour des poids avec decay L2
        layer.weights -= self.learning_rate * (m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon) + 0.00005 * layer.weights)
        layer.biases -= self.learning_rate * (m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon))

        
        print(f"Mean Weight Update: {np.mean(layer.weights)}, Mean Bias Update: {np.mean(layer.biases)}")
