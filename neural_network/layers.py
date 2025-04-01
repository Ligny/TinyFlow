import numpy as np
from typing import Tuple, Optional
from neural_network.activations import ActivationReLU, ActivationSigmoid

class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int, activation=None, dropout_rate: float = 0.05, layer_norm: bool = False):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.epsilon = 1e-5 if layer_norm else None
        self.gamma = np.ones((1, n_neurons)) if layer_norm else None
        self.beta = np.zeros((1, n_neurons)) if layer_norm else None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.inputs = np.atleast_2d(inputs)
        self.z = np.dot(self.inputs, self.weights) + self.biases

        if self.layer_norm:
            mean = np.mean(self.z, axis=0, keepdims=True)
            std = np.std(self.z, axis=0, keepdims=True) + self.epsilon
            self.z = self.gamma * (self.z - mean) / std + self.beta

        self.output = self.activation.forward(self.z) if self.activation else self.z

        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.rand(*self.output.shape) > self.dropout_rate
            self.output *= self.dropout_mask

        return self.output

    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.dropout_rate > 0:
            d_output *= self.dropout_mask

        if self.layer_norm:
            mean = np.mean(d_output, axis=0, keepdims=True)
            std = np.std(d_output, axis=0, keepdims=True) + self.epsilon
            d_output = (d_output - mean) / std

        d_activation = self.activation.backward(d_output) if self.activation else d_output

        self.d_weights = np.dot(self.inputs.T, d_activation)
        self.d_biases = np.sum(d_activation, axis=0, keepdims=True)

        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

        return np.dot(d_activation, self.weights.T)

class LayerSoftmaxCrossEntropy:
    def __init__(self, n_inputs: int, n_neurons: int):
        limit = np.sqrt(6 / (n_inputs + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray, y_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[float]]:
        self.inputs = np.atleast_2d(inputs)
        self.z = np.dot(self.inputs, self.weights) + self.biases
        exp_values = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        if y_true is not None:
            clipped_output = np.clip(self.output, 1e-7, 1 - 1e-7)
            correct_confidences = np.sum(clipped_output * y_true, axis=1)
            self.loss = -np.mean(np.log(correct_confidences))
            return self.output, self.loss
        return self.output, None

    def backward(self, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        samples = y_true.shape[0]
        self.dinputs = (self.output - y_true) / samples
        d_weights = np.dot(self.inputs.T, self.dinputs)
        d_biases = np.sum(self.dinputs, axis=0, keepdims=True)
        return d_weights, d_biases

class BatchNormalization:
    def __init__(self, n_neurons: int, momentum: float = 0.9, epsilon: float = 1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones((1, n_neurons))
        self.beta = np.zeros((1, n_neurons))
        self.running_mean = np.zeros((1, n_neurons))
        self.running_var = np.ones((1, n_neurons))

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.inputs = inputs

        if training:
            self.mean = np.mean(inputs, axis=0, keepdims=True)
            self.var = np.var(inputs, axis=0, keepdims=True)
            self.normalized = (inputs - self.mean) / np.sqrt(self.var + self.epsilon)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            self.normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        self.output = self.gamma * self.normalized + self.beta
        return self.output

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        batch_size = d_values.shape[0]
        self.dgamma = np.sum(d_values * self.normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(d_values, axis=0, keepdims=True)

        dnormalized = d_values * self.gamma
        safe_var = np.maximum(self.var, 1e-7)
        dvar = np.sum(dnormalized * (self.inputs - self.mean) * -0.5 * np.power(safe_var + self.epsilon, -1.5), axis=0, keepdims=True)
        dmean = np.sum(dnormalized * -1 / np.sqrt(self.var + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (self.inputs - self.mean), axis=0, keepdims=True) / batch_size

        return dnormalized / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.inputs - self.mean) / batch_size + dmean / batch_size