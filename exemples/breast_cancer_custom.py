import os
import sys
import time
import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network.layers import LayerDense
from neural_network.activations import ActivationReLU, ActivationSigmoid
from neural_network.loss import LossCrossEntropy
from neural_network.optimizers import AdamOptimizer
from neural_network.network import NeuralNetwork

def get_custom_model(input_dim: int) -> NeuralNetwork:
    return NeuralNetwork([
        LayerDense(input_dim, 64, activation=ActivationReLU(), dropout_rate=0.02),
        LayerDense(64, 32, activation=ActivationReLU(), dropout_rate=0.02),
        LayerDense(32, 1, activation=ActivationSigmoid())
    ], loss_function=LossCrossEntropy())

def evaluate_custom(network: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> float:
    correct = 0
    for inputs, expected in zip(X_test, y_test):
        predicted = network.forward(inputs)
        predicted = (predicted > 0.5).astype(int).flatten()
        correct += int(predicted[0] == expected[0])
    return correct / len(y_test)

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_network = get_custom_model(X_train.shape[1])
    dataset = list(zip(X_train, y_train))

    optimizer = AdamOptimizer(learning_rate=0.0005, clip_value=5.0, decay=1e-4)

    start_time = time.time()
    custom_network.train(dataset, epochs=5000, optimizer=optimizer, batch_size=64)
    custom_train_time = time.time() - start_time

    custom_accuracy = evaluate_custom(custom_network, X_test, y_test)

    print(f"Results for Breast Cancer Dataset:")
    print(f"Custom Neural Network Accuracy: {custom_accuracy * 100:.2f}%, Training Time: {custom_train_time:.2f}s")