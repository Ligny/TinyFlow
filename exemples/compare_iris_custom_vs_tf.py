import os
import sys
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# Import depuis le package neural_network
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_network.layers import LayerDense, LayerSoftmaxCrossEntropy
from neural_network.activations import ActivationReLU
from neural_network.optimizers import AdamOptimizer
from neural_network.network import NeuralNetwork

from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

def get_custom_model(input_dim: int, output_dim: int) -> NeuralNetwork:
    return NeuralNetwork([
        LayerDense(input_dim, 16, activation=ActivationReLU()),
        LayerDense(16, 8, activation=ActivationReLU()),
        LayerSoftmaxCrossEntropy(8, output_dim)
    ])

def get_tf_model(input_dim: int, output_dim: int) -> keras.Sequential:
    model = keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(input_dim,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def evaluate_custom(network, X_test, y_test) -> float:
    correct = 0
    for inputs, expected in zip(X_test, y_test):
        predicted = network.forward(inputs)
        correct += int(np.argmax(predicted) == np.argmax(expected))
    return correct / len(y_test)

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

    optimizer = AdamOptimizer(learning_rate=0.0005, clip_value=5.0, decay=1e-4)

    start_time = time.time()
    custom_network.train(dataset, epochs=2000, optimizer=optimizer, batch_size=32)
    custom_train_time = time.time() - start_time

    custom_accuracy = evaluate_custom(custom_network, X_test, y_test)

    #tf_model = get_tf_model(X_train.shape[1], y_train.shape[1])

    #start_time = time.time()
    #tf_model.fit(X_train, y_train, epochs=5000, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    #tf_train_time = time.time() - start_time

    #_, tf_accuracy = tf_model.evaluate(X_test, y_test, verbose=0)

    print("\n✅ Results for Iris Dataset (3 Classes - Multi-class Classification):")
    print(f"Custom Neural Network Accuracy: {custom_accuracy * 100:.2f}%, Training Time: {custom_train_time:.2f}s")
    #print(f"TensorFlow Neural Network Accuracy: {tf_accuracy * 100:.2f}%, Training Time: {tf_train_time:.2f}s")
