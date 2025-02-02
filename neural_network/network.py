from typing import List, Tuple
import numpy as np # type: ignore
from neural_network.optimizers import AdamOptimizer

class NeuralNetwork:
    def __init__(self, layers: List, loss_function):
        self.layers = layers
        self.loss_function = loss_function

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, predicted: np.ndarray, actual: np.ndarray, optimizer: AdamOptimizer) -> None:
        loss_gradients = self.loss_function.backward(predicted, actual)

        for layer in reversed(self.layers):
            d_output = layer.activation.backward(loss_gradients) if layer.activation else loss_gradients
            d_weights = np.dot(layer.inputs.T, d_output)
            d_biases = np.sum(d_output, axis=0, keepdims=True)
            optimizer.update(layer, d_weights, d_biases)
            loss_gradients = np.dot(d_output, layer.weights.T)

    def train(self, dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 1000, batch_size: int = 32, optimizer: AdamOptimizer = None) -> None:
        X_train, y_train = zip(*dataset)
        X_train, y_train = np.array(X_train), np.array(y_train)

        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                predicted = self.forward(X_batch)
                loss = self.loss_function.forward(predicted, y_batch)
                self.backward(predicted, y_batch, optimizer)

                total_loss += loss

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {float(total_loss) / len(X_train):.6f}")
                      
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        correct = 0
        for inputs, expected in zip(X_test, y_test):
            predicted = self.forward(inputs)
            correct += int((predicted > 0.5) == expected)

        accuracy = correct / len(y_test)
        print(f"✅ Accuracy: {accuracy * 100:.2f}%")
        return accuracy