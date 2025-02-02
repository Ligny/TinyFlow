from neural_network.activations import ActivationReLU, ActivationSigmoid
import numpy as np # type: ignore
from typing import Optional

class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int, activation=None, dropout_rate: float = 0.05, layer_norm: bool = False):
        limit = np.sqrt(1 / n_inputs) if isinstance(activation, ActivationSigmoid) else np.sqrt(2 / n_inputs)
        self.weights = np.random.randn(n_inputs, n_neurons) * limit
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        
        if layer_norm:
            self.gamma = np.ones((1, n_neurons))
            self.beta = np.zeros((1, n_neurons))
            self.epsilon = 1e-5 

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.inputs = np.atleast_2d(inputs)
        self.z = np.dot(self.inputs, self.weights) + self.biases

        if self.layer_norm:
            mean = np.mean(self.z, axis=1, keepdims=True)
            std = np.std(self.z, axis=1, keepdims=True)
            self.z = (self.z - mean) / (std + self.epsilon)
            self.z = self.gamma * self.z + self.beta

        self.output = self.activation.forward(self.z) if self.activation else self.z

        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.rand(*self.output.shape) > self.dropout_rate
            self.output *= self.dropout_mask

        return self.output

    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.dropout_rate > 0:
            d_output *= self.dropout_mask

        if self.layer_norm:
            mean = np.mean(d_output, axis=1, keepdims=True)
            std = np.std(d_output, axis=1, keepdims=True)
            d_output = (d_output - mean) / (std + self.epsilon)

        d_activation = self.activation.backward(d_output) if self.activation else d_output
        d_weights = np.dot(self.inputs.T, d_activation)
        d_biases = np.sum(d_activation, axis=0, keepdims=True)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        if self.layer_norm:
            self.gamma -= learning_rate * np.sum(d_output * self.z, axis=0, keepdims=True)
            self.beta -= learning_rate * np.sum(d_output, axis=0, keepdims=True)

        return np.dot(d_activation, self.weights.T)
