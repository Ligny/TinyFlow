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

def get_custom_model(input_dim: int, output_dim: int) -> NeuralNetwork:
    return NeuralNetwork([
        LayerDense(input_dim, 32, activation=ActivationReLU(), dropout_rate=0.1),
        LayerDense(32, 16, activation=ActivationReLU(), dropout_rate=0.1),
        LayerSoftmaxCrossEntropy(16, output_dim)
    ])

def evaluate_custom(network, X_test, y_test) -> float:
    correct = 0
    for inputs, expected in zip(X_test, y_test):
        predicted = network.forward(inputs)
        predicted_label = np.argmax(predicted)
        expected_label = np.argmax(expected)
        correct += int(predicted_label == expected_label)
    return correct / len(y_test)

if __name__ == "__main__":
    # Charger et préparer les données
    data = load_iris()
    X, y = data.data, data.target

    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle
    custom_network = get_custom_model(X_train.shape[1], y_train.shape[1])
    dataset = list(zip(X_train, y_train))

    optimizer = AdamOptimizer(learning_rate=0.005, clip_value=2.0, decay=1e-5)

    custom_network.train(dataset, epochs=1000, optimizer=optimizer, batch_size=16)  # Réduit à 1000 epochs

    # Évaluer
    custom_accuracy = evaluate_custom(custom_network, X_test, y_test)

    print("\n✅ Results for Iris Dataset (3 Classes - Multi-class Classification):")
    print(f"Custom Neural Network Accuracy: {custom_accuracy * 100:.2f}%")

    # Exemple de prédiction
    sample = X_test[:1]
    true_label = np.argmax(y_test[:1])
    predicted = custom_network.forward(sample)
    predicted_label = np.argmax(predicted)

    print(f"Softmax probabilities: {predicted}")
    print(f"True label: {true_label}, Predicted: {predicted_label}")