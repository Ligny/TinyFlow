import numpy as np

class Loss:
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        raise NotImplementedError()

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class LossMSE(Loss):
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.mean((predicted - actual) ** 2)

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 2 * (predicted - actual) / len(actual)

class LossCrossEntropy(Loss):
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        predicted = np.clip(predicted, 1e-9, 1 - 1e-9)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, 1e-9, 1 - 1e-9)
        return -(actual / predicted) + (1 - actual) / (1 - predicted)

class LossCategoricalCrossentropy(Loss):
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(predicted * actual, axis=1)
        return -np.mean(np.log(correct_confidences))

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
        return -actual / predicted / predicted.shape[0]