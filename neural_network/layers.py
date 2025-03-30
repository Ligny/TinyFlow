import numpy as np
from typing import List, Tuple
from neural_network.activations import ActivationReLU, ActivationSigmoid

class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int, activation=None, dropout_rate: float = 0.05, layer_norm: bool = False):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm

        if layer_norm:
            self.gamma = np.ones((1, n_neurons))
            self.beta = np.zeros((1, n_neurons))
            self.epsilon = 1e-5  # Petite valeur pour éviter la division par zéro

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.inputs = np.atleast_2d(inputs)
        self.z = np.dot(self.inputs, self.weights) + self.biases

        if self.layer_norm:
            mean = np.mean(self.z, axis=0, keepdims=True)
            std = np.std(self.z, axis=0, keepdims=True) + self.epsilon  # Évite la division par 0
            self.z = (self.z - mean) / std
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
        print(f"Initial weight range: {self.weights.min()} to {self.weights.max()}")
        self.biases = np.zeros((1, n_neurons))
        self.loss = None

    def forward(self, inputs: np.ndarray, y_true: np.ndarray = None) -> Tuple[np.ndarray, float]:
        self.inputs = np.atleast_2d(inputs)
        self.z = np.dot(self.inputs, self.weights) + self.biases

        exp_values = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
        self.output = exp_values / (np.sum(exp_values, axis=1, keepdims=True) + 1e-7)


        if y_true is not None:
            clipped_output = np.clip(self.output, 1e-7, 1 - 1e-7)
            correct_confidences = np.take_along_axis(clipped_output, np.argmax(y_true, axis=1, keepdims=True), axis=1).flatten()
            self.loss = -np.mean(np.log(correct_confidences))
            return self.output, self.loss
        return self.output

    def backward(self, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        samples = y_true.shape[0]
        self.dinputs = (self.output - y_true) / samples
        if np.isnan(self.dinputs).any():
            raise ValueError("NaN detected in dinputs during backward pass!")
        if np.isnan(self.dinputs).any():
            raise ValueError("NaN detected in gradients!")

        d_weights = np.dot(self.inputs.T, self.dinputs)
        d_biases = np.sum(self.dinputs, axis=0, keepdims=True)
        print(f"Mean Gradient Weights: {np.mean(d_weights)}, Mean Gradient Biases: {np.mean(d_biases)}")


        return d_weights, d_biases

class BatchNormalization:
    def __init__(self, n_neurons, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon

        self.weights = self.gamma = np.ones((1, n_neurons))
        self.biases = self.beta = np.zeros((1, n_neurons))

        self.running_mean = np.zeros((1, n_neurons))
        self.running_var = np.ones((1, n_neurons))

    def forward(self, inputs, training=True):
        self.inputs = inputs

        if training:
            self.mean = np.mean(inputs, axis=0, keepdims=True)
            self.var = np.var(inputs, axis=0, keepdims=True)

            self.normalized = (inputs - self.mean) / np.sqrt(self.var + self.epsilon)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            
            if np.isnan(self.mean).any() or np.isnan(self.var).any():
                raise ValueError("NaN detected in batch normalization forward pass!")
        else:
            self.normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        self.output = self.weights * self.normalized + self.biases
        print(f"Gamma: {self.gamma.mean()}, Beta: {self.beta.mean()}")
        return self.output

    def backward(self, dvalues, learning_rate=None):  
        batch_size = dvalues.shape[0]
        self.d_weights = self.dgamma = np.sum(dvalues * self.normalized, axis=0, keepdims=True)
        self.d_biases = self.dbeta = np.sum(dvalues, axis=0, keepdims=True)

        dnormalized = dvalues * self.weights
        safe_var = np.maximum(self.var, 1e-7)
        dvar = np.sum(dnormalized * (self.inputs - self.mean) * -0.5 * np.power(safe_var + self.epsilon, -1.5), axis=0, keepdims=True)
        dmean = np.sum(dnormalized * -1 / np.sqrt(self.var + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (self.inputs - self.mean), axis=0, keepdims=True) / batch_size

        self.dinputs = dnormalized / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.inputs - self.mean) / batch_size + dmean / batch_size
        return self.dinputs