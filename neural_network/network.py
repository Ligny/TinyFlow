from typing import List, Optional, Tuple
import numpy as np
from neural_network.optimizers import AdamOptimizer
from neural_network.layers import LayerDense, LayerSoftmaxCrossEntropy
from neural_network.loss import Loss

class NeuralNetwork:
    def __init__(self, layers: List, loss_function: Optional[Loss] = None):
        self.layers = layers
        self.loss_function = loss_function  # Optionnel pour le cas binaire

        # Ces deux attributs sont nécessaires pour le cas binaire
        self.last_output = None
        self.last_loss = None

    def forward(self, inputs, y_true=None):
        for layer in self.layers:
            if isinstance(layer, LayerSoftmaxCrossEntropy):
                if y_true is not None:
                    self.last_output, self.last_loss = layer.forward(inputs, y_true)
                else:
                    self.last_output = layer.forward(inputs)  # Pas de loss en mode prédiction
            else:
                inputs = layer.forward(inputs)

        self.last_output = inputs
        return self.last_output


    def backward(self, y_true: np.ndarray, optimizer: AdamOptimizer) -> None:
        if isinstance(self.layers[-1], LayerSoftmaxCrossEntropy):
            d_weights, d_biases = self.layers[-1].backward(y_true)
            optimizer.update(self.layers[-1], d_weights, d_biases)  # Mise à jour avec Adam
            loss_gradients = np.dot(self.layers[-1].dinputs, self.layers[-1].weights.T)
            if np.isnan(loss_gradients).any():
                raise ValueError("NaN detected in loss_gradients!")
        elif self.loss_function:
            loss_gradients = self.loss_function.backward(self.last_output, y_true)
        else:
            raise ValueError("Missing loss function for binary classification.")

        for layer in reversed(self.layers[:-1] if isinstance(self.layers[-1], LayerSoftmaxCrossEntropy) else self.layers):
            loss_gradients = layer.backward(loss_gradients, optimizer.learning_rate)
            optimizer.update(layer, layer.d_weights, layer.d_biases)

    def train(self, dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 1000, batch_size: int = 32, optimizer: AdamOptimizer = None) -> None:
        X_train, y_train = zip(*dataset)
        X_train, y_train = np.array(X_train), np.array(y_train)

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total_samples = 0
            avg_grad_weights, avg_grad_biases = 0, 0
            avg_weight_norm = 0

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                predicted = self.forward(X_batch, y_true=y_batch)

                # Calcul de la perte
                if isinstance(self.layers[-1], LayerSoftmaxCrossEntropy):
                    loss = self.layers[-1].loss
                elif self.loss_function:
                    loss = self.loss_function.forward(predicted, y_batch)
                else:
                    raise ValueError("Missing loss function for binary classification.")

                self.backward(y_batch, optimizer)

                total_loss += loss
                total_samples += len(y_batch)

                # Calcul de l'accuracy sur le batch
                if predicted.shape[-1] == 1:  # Binaire
                    predicted_classes = (predicted > 0.5).astype(int)
                else:  # Multi-class
                    predicted_classes = np.argmax(predicted, axis=1)
                    y_batch = np.argmax(y_batch, axis=1)

                correct += np.sum(predicted_classes == y_batch)

                # Suivi des gradients et poids
                batch_grad_weights, batch_grad_biases = 0, 0
                batch_weight_norm = 0
                for layer in self.layers:
                    if hasattr(layer, "d_weights"):
                        batch_grad_weights += np.mean(np.abs(layer.d_weights))
                        batch_grad_biases += np.mean(np.abs(layer.d_biases))
                    if hasattr(layer, "weights"):
                        batch_weight_norm += np.mean(np.linalg.norm(layer.weights, axis=1))

                avg_grad_weights += batch_grad_weights / len(self.layers)
                avg_grad_biases += batch_grad_biases / len(self.layers)
                avg_weight_norm += batch_weight_norm / len(self.layers)

            if epoch % 100 == 0:
                accuracy = correct / total_samples * 100
                avg_loss = total_loss / total_samples
                print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}% | "
                    f"Avg Grad Weights: {avg_grad_weights:.6f} | Avg Grad Biases: {avg_grad_biases:.6f} | "
                    f"Avg Weight Norm: {avg_weight_norm:.6f}")


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        correct = 0
        for inputs, expected in zip(X_test, y_test):
            predicted = self.forward(inputs)

            if predicted.shape[-1] == 1:  # Binaire
                predicted = (predicted > 0.5).astype(int)
                correct += int(predicted[0] == expected[0])
            else:  # Multi-class
                correct += int(np.argmax(predicted) == np.argmax(expected))

        accuracy = correct / len(y_test)
        print(f"✅ Accuracy: {accuracy * 100:.2f}%")
        return accuracy
