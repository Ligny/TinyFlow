import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from neural_network.layers import LayerDense
from neural_network.activations import ActivationReLU

def test_layer_forward():
    layer = LayerDense(n_inputs=2, n_neurons=3, activation=ActivationReLU())
    inputs = np.array([[1, 2], [3, 4]])
    output = layer.forward(inputs)
    assert output.shape == (2, 3)

def test_layer_backward():
    layer = LayerDense(n_inputs=2, n_neurons=3, activation=ActivationReLU())
    inputs = np.array([[1, 2]])
    layer.forward(inputs)
    d_output = np.array([[0.5, 0.1, -0.3]])
    learning_rate = 0.01
    layer.backward(d_output, learning_rate)
    assert layer.weights.shape == (2, 3)
    assert layer.biases.shape == (1, 3)