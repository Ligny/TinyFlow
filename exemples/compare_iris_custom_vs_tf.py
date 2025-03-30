import os
import sys
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_network.layers import LayerDense, LayerSoftmaxCrossEntropy, BatchNormalization
from neural_network.activations import ActivationReLU
from neural_network.optimizers import AdamOptimizer
from neural_network.network import NeuralNetwork

from tensorflow import keras
from tensorflow.keras import layers

def get_custom_model(input_dim: int, output_dim: int) -> NeuralNetwork:
    return NeuralNetwork([
        LayerDense(input_dim, 16),
        BatchNormalization(16),
        ActivationReLU(),
        LayerDense(16, 8),
        BatchNormalization(8),
        ActivationReLU(),
        LayerDense(8, output_dim),
        LayerSoftmaxCrossEntropy(output_dim, output_dim)
    ])

def evaluate_custom(network, X_test, y_test) -> float:
    correct = 0
    for inputs, expected in zip(X_test, y_test):
        predicted = network.forward(inputs)
        predicted_label = np.argmax(predicted)
        expected_label = np.argmax(expected)
        correct += int(predicted_label == expected_label)
    return correct / len(y_test)
  
def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target

    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_network = get_custom_model(X_train.shape[1], y_train.shape[1])
    dataset = list(zip(X_train, y_train))

    optimizer = AdamOptimizer(learning_rate=0.001, clip_value=1.0, decay=1e-4)

    start_time = time.time()
    custom_network.train(dataset, epochs=10000, optimizer=optimizer, batch_size=32)
    custom_train_time = time.time() - start_time

    custom_accuracy = evaluate_custom(custom_network, X_test, y_test)

    print("\n✅ Results for Iris Dataset (3 Classes - Multi-class Classification):")
    print(f"Custom Neural Network Accuracy: {custom_accuracy * 100:.2f}%, Training Time: {custom_train_time:.2f}s")
    
    sample = X_test[:1]
    true_label = np.argmax(y_test[:1])

    predicted = custom_network.forward(sample)
    softmax_probs = softmax(predicted)

    predicted_label = np.argmax(softmax_probs)

    print(f"Raw output before softmax: {predicted}")
    print(f"Softmax probabilities: {softmax_probs}")
    print(f"True label: {true_label}, Predicted: {predicted_label}")