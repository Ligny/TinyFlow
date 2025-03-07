import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from neural_network.layers import LayerDense
from neural_network.activations import ActivationReLU
from neural_network.optimizers import AdamOptimizer

def test_adam_update():
    layer = LayerDense(2, 3, ActivationReLU())
    optimizer = AdamOptimizer(learning_rate=0.001)
    d_weights = np.ones_like(layer.weights)
    d_biases = np.ones_like(layer.biases)
    optimizer.update(layer, d_weights, d_biases)
    assert layer.weights.shape == (2, 3)
    assert layer.biases.shape == (1, 3)