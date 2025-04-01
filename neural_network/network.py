import numpy as np
from typing import List, Optional, Tuple
from neural_network.optimizers import AdamOptimizer
from neural_network.layers import LayerDense, LayerSoftmaxCrossEntropy
from neural_network.loss import Loss

class NeuralNetwork:
    def __init__(self, layers: List, loss_function: Optional[Loss] = None):
        self.layers = layers
        self.loss_function = loss_function
        self.last_output = None
        self.last_loss = None

    def forward(self, inputs: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, LayerSoftmaxCrossEntropy):
                self.last_output, self.last_loss = layer.forward(inputs, y_true) if y_true is not None else (layer.forward(inputs), None)
                return self.last_output
            inputs = layer.forward(inputs)
        self.last_output = inputs
        return self.last_output

    def backward(self, y_true: np.ndarray, optimizer: AdamOptimizer) -> None:
        if isinstance(self.layers[-1], LayerSoftmaxCrossEntropy):
            d_weights, d_biases = self.layers[-1].backward(y_true)
            optimizer.update(self.layers[-1], d_weights, d_biases)
            loss_gradients = np.dot(self.layers[-1].dinputs, self.layers[-1].weights.T)
        elif self.loss_function:
            loss_gradients = self.loss_function.backward(self.last_output, y_true)
        else:
            raise ValueError("Missing loss function for binary classification.")

        for layer in reversed(self.layers[:-1] if isinstance(self.layers[-1], LayerSoftmaxCrossEntropy) else self.layers):
            loss_gradients = layer.backward(loss_gradients, optimizer.learning_rate)
            if hasattr(layer, "weights") and hasattr(layer, "d_weights"):
                optimizer.update(layer, layer.d_weights, layer.d_biases)

    def train(self, dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 1000, batch_size: int = 32, optimizer: Optional[AdamOptimizer] = None) -> None:
        X_train, y_train = map(np.array, zip(*dataset))

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total_samples = 0

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                predicted = self.forward(X_batch, y_batch)
                loss = self.last_loss if isinstance(self.layers[-1], LayerSoftmaxCrossEntropy) else self.loss_function.forward(predicted, y_batch)
                self.backward(y_batch, optimizer)

                total_loss += loss
                total_samples += len(y_batch)

                predicted_classes = (predicted > 0.5).astype(int) if predicted.shape[-1] == 1 else np.argmax(predicted, axis=1)
                y_batch_classes = y_batch if predicted.shape[-1] == 1 else np.argmax(y_batch, axis=1)
                correct += np.sum(predicted_classes == y_batch_classes)

            if epoch % 100 == 0:
                accuracy = correct / total_samples * 100
                avg_loss = total_loss / total_samples
                print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        correct = 0
        for inputs, expected in zip(X_test, y_test):
            predicted = self.forward(inputs)
            predicted_class = (predicted > 0.5).astype(int)[0] if predicted.shape[-1] == 1 else np.argmax(predicted)
            expected_class = expected[0] if predicted.shape[-1] == 1 else np.argmax(expected)
            correct += int(predicted_class == expected_class)

        accuracy = correct / len(y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy