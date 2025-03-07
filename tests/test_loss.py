import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from neural_network.loss import LossMSE, LossCrossEntropy

def test_mse_forward():
    loss = LossMSE()
    predicted = [0.9, 0.2, 0.3]
    actual = [1, 0, 0]
    assert np.isclose(loss.forward(predicted, actual), 0.046666, atol=1e-5)

def test_crossentropy_forward():
    loss = LossCrossEntropy()
    predicted = np.array([[0.9], [0.1], [0.3]])
    actual = np.array([[1], [0], [0]])
    result = loss.forward(predicted, actual)
    assert result > 0 and result < 1