
# Mini TensorFlow

![CI](https://img.shields.io/badge/build-passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-100%25-success)

## Description
Mini TensorFlow is a lightweight, custom-built neural network library implemented in Python using NumPy. It provides a simple yet flexible framework for creating, training, and evaluating neural networks for both binary and multi-class classification tasks. This library is designed for educational purposes and small-scale machine learning experiments, offering core functionalities inspired by TensorFlow but with a minimalistic approach.

## Features
- **Modular Layers**: Dense layers with ReLU/Sigmoid/Softmax activations, dropout, and optional layer normalization.
- **Loss Functions**: Mean Squared Error (MSE), Binary Cross-Entropy, and Categorical Cross-Entropy.
- **Optimizer**: Adam optimizer with gradient clipping, learning rate decay, and L2 regularization.
- **Flexible Architecture**: Supports custom network configurations for binary and multi-class problems.
- **Examples**: Includes scripts to train and evaluate models on Breast Cancer (binary) and Iris (multi-class) datasets.

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/mini-tensorflow.git
   cd mini-tensorflow
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy sklearn
   ```

3. **Optional: Install TensorFlow for comparison scripts**:
   ```bash
   pip install tensorflow
   ```

4. **Ensure the project structure is set up correctly**:
   ```
   mini-tensorflow/
   ├── neural_network/
   │   ├── activations.py
   │   ├── layers.py
   │   ├── loss.py
   │   ├── network.py
   │   └── optimizers.py
   ├── examples/
   │   ├── compare_breast_cancer.py
   │   └── compare_iris.py
   └── README.md
   ```

## Usage

Mini TensorFlow provides two example scripts to demonstrate its capabilities:

### Running Examples

**Breast Cancer (Binary Classification)**:
```bash
python examples/compare_breast_cancer.py
```
This trains a neural network on the Breast Cancer dataset and outputs the accuracy and training time.

**Iris (Multi-Class Classification)**:
```bash
python examples/compare_iris.py
```
This trains a neural network on the Iris dataset and displays accuracy along with an example prediction.

### Building Your Own Model

**Define the Network**:
```python
from neural_network.network import NeuralNetwork
from neural_network.layers import LayerDense, LayerSoftmaxCrossEntropy
from neural_network.activations import ActivationReLU, ActivationSigmoid
from neural_network.loss import LossCrossEntropy
from neural_network.optimizers import AdamOptimizer

# Example: Binary classification network
model = NeuralNetwork([
    LayerDense(input_dim=30, n_neurons=64, activation=ActivationReLU(), dropout_rate=0.02),
    LayerDense(64, 32, activation=ActivationReLU(), dropout_rate=0.02),
    LayerDense(32, 1, activation=ActivationSigmoid())
], loss_function=LossCrossEntropy())
```

**Prepare Data**:
```python
import numpy as np
X_train = np.array([...])  # Your input data (n_samples, n_features)
y_train = np.array([...])  # Your labels (n_samples, 1) for binary or (n_samples, n_classes) for multi-class
dataset = list(zip(X_train, y_train))
```

**Train the Model**:
```python
optimizer = AdamOptimizer(learning_rate=0.001, clip_value=5.0, decay=1e-4)
model.train(dataset, epochs=1000, batch_size=32, optimizer=optimizer)
```

**Evaluate**:
```python
X_test = np.array([...])
y_test = np.array([...])
accuracy = model.evaluate(X_test, y_test)
```

## Key Parameters

### NeuralNetwork
- `layers`: List of layer objects (e.g., LayerDense, LayerSoftmaxCrossEntropy).
- `loss_function`: Optional loss object (e.g., LossCrossEntropy) for binary classification; not needed if using LayerSoftmaxCrossEntropy.

### LayerDense
- `n_inputs`: Number of input features.
- `n_neurons`: Number of neurons in the layer.
- `activation`: Activation function (e.g., ActivationReLU(), ActivationSigmoid()).
- `dropout_rate`: Fraction of units to drop (default: 0.05).
- `layer_norm`: Enable layer normalization (default: False).

### LayerSoftmaxCrossEntropy
- `n_inputs`: Number of input features.
- `n_neurons`: Number of output classes (for multi-class tasks).

### AdamOptimizer
- `learning_rate`: Initial learning rate (default: 0.001).
- `beta1`: Exponential decay rate for the first moment (default: 0.9).
- `beta2`: Exponential decay rate for the second moment (default: 0.999).
- `epsilon`: Small value to prevent division by zero (default: 1e-8).
- `clip_value`: Maximum gradient magnitude (default: 5.0).
- `decay`: Learning rate decay factor (default: 0.0).

### Train Method
- `dataset`: List of (input, label) tuples.
- `epochs`: Number of training iterations (default: 1000).
- `batch_size`: Number of samples per batch (default: 32).
- `optimizer`: Optimizer instance (e.g., AdamOptimizer).

## Example Output

**For compare_iris.py**:
```
Results for Iris Dataset (3 Classes - Multi-class Classification):
Custom Neural Network Accuracy: 93.33%
Softmax probabilities: [[0.02 0.95 0.03]]
True label: 1, Predicted: 1
```

## Contributing

Feel free to submit issues or pull requests to enhance this library. Suggestions for additional features (e.g., new layers, optimizers) are welcome!

## License

This project is licensed under the MIT License.
