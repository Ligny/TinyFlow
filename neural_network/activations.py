import math
import numpy as np  # type: ignore
from typing import List, Union

class ActivationReLU:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, d_values: np.ndarray, *args, **kwargs) -> np.ndarray:
        d_inputs = d_values.copy()
        d_inputs[self.output <= 0] = 0
        return d_inputs

class ActivationSigmoid:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, d_values: np.ndarray, *args, **kwargs) -> np.ndarray:
        return d_values * self.output * (1 - self.output)
      
class ActivationSoftmax:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, d_values: np.ndarray, *args, **kwargs) -> np.ndarray:
        self.dinputs = d_values
        return self.dinputs