import numpy as np
from typing import List

class Loss:
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        raise NotImplementedError()

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class LossMSE(Loss):
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.mean((predicted - actual) ** 2)

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        samples = len(actual)
        return 2 * (predicted - actual) / samples

class LossCrossEntropy(Loss):
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        epsilon = 1e-9 
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -(actual / predicted) + (1 - actual) / (1 - predicted)

class LossCategoricalCrossentropy(Loss):
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(predicted * actual, axis=1)
        return -np.mean(np.log(correct_confidences))

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        samples = predicted.shape[0]
        predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
        return -actual / predicted / samples
