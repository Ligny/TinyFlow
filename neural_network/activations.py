import math
import numpy as np  # type: ignore
from typing import List, Union

class ActivationReLU:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        d_inputs = d_values.copy()
        d_inputs[self.output <= 0] = 0
        return d_inputs

class ActivationSigmoid:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        return d_values * self.output * (1 - self.output)
      
class ActivationLeakyReLU:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.where(inputs > 0, inputs, 0.01 * inputs)
        return self.output

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        d_inputs = np.ones_like(d_values)
        d_inputs[d_values < 0] = 0.01
        return d_inputs