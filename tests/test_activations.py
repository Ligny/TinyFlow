import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from neural_network.activations import ActivationReLU, ActivationSigmoid, ActivationLeakyReLU

def test_relu_forward():
    relu = ActivationReLU()
    inputs = np.array([[-1, 2], [3, -4]])
    output = relu.forward(inputs)
    expected = np.array([[0, 2], [3, 0]])
    assert np.array_equal(output, expected)

def test_relu_backward():
    relu = ActivationReLU()
    relu.forward(np.array([[-1, 2], [3, -4]]))
    d_values = np.array([[1, 1], [1, 1]])
    d_inputs = relu.backward(d_values)
    expected = np.array([[0, 1], [1, 0]])
    assert np.array_equal(d_inputs, expected)

def test_sigmoid_forward():
    sigmoid = ActivationSigmoid()
    inputs = np.array([[0], [1], [-1]])
    output = sigmoid.forward(inputs)
    expected = np.array([[0.5], [1 / (1 + np.exp(-1))], [1 / (1 + np.exp(1))]])
    assert np.allclose(output, expected, atol=1e-6)

def test_leaky_relu_forward():
    leaky_relu = ActivationLeakyReLU()
    inputs = np.array([[-1, 2], [3, -4]])
    output = leaky_relu.forward(inputs)
    expected = np.array([[-0.01, 2], [3, -0.04]])
    assert np.allclose(output, expected, atol=1e-6)