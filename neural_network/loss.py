import math
from typing import List
import numpy as np # type: ignore

class LossMSE:
    def forward(self, predicted: List[float], actual: List[float]) -> float:
        return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(actual)

    def backward(self, predicted: List[float], actual: List[float]) -> List[float]:
        return [2 * (p - a) / len(actual) for p, a in zip(predicted, actual)]

class LossCrossEntropy:
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        epsilon = 1e-9 
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -(actual / predicted) + (1 - actual) / (1 - predicted)