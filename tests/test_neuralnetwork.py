import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from neural_network.layers import LayerDense
from neural_network.activations import ActivationSigmoid
from neural_network.loss import LossMSE
from neural_network.network import NeuralNetwork
from neural_network.optimizers import AdamOptimizer

def test_neuralnetwork_forward():
    layers = [
        LayerDense(2, 3, ActivationSigmoid()),
        LayerDense(3, 1, ActivationSigmoid())
    ]
    nn = NeuralNetwork(layers, LossMSE())
    inputs = np.array([[1, 2]])
    output = nn.forward(inputs)
    assert output.shape == (1, 1)

def test_neuralnetwork_train():
    layers = [
        LayerDense(2, 3, ActivationSigmoid()),
        LayerDense(3, 1, ActivationSigmoid())
    ]
    nn = NeuralNetwork(layers, LossMSE())
    optimizer = AdamOptimizer()
    dataset = [
        (np.array([1, 2]), np.array([1])),
        (np.array([3, 4]), np.array([0]))
    ]
    nn.train(dataset, epochs=5, batch_size=1, optimizer=optimizer)
    assert True