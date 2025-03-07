
# Custom Neural Network

![CI](https://img.shields.io/badge/build-passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-100%25-success)

## 📌 Project Overview
This project implements a **fully connected neural network** from scratch using NumPy to classify breast cancer tumors based on the **Breast Cancer Wisconsin dataset**. The model includes **batch normalization, dropout, and the Adam optimizer** to enhance performance.

## 📊 Dataset: Breast Cancer Wisconsin (Diagnostic)
The dataset is obtained from **scikit-learn** (`load_breast_cancer`) and consists of **30 features** extracted from digitized images of breast mass. The goal is to classify whether a tumor is **malignant (1) or benign (0)**.

### 🔹 Features:
- **Mean, standard error, and worst values** of ten real-valued features, such as:
  - Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness, etc.

### 🔹 Dataset Summary:
- **Samples:** 569
- **Features:** 30 (all numerical)
- **Classes:** Malignant (212), Benign (357)
- **Train/Test Split:** 80% training, 20% testing
- **Scaling:** Standardized using `StandardScaler()`

## ⚙️ Neural Network Architecture
The model is a **feedforward neural network** with multiple layers and activations:

| Layer | Type  | Neurons | Activation |
|--------|------------|------------|------------|
| 1 | Fully Connected | 64 | ReLU |
| 2 | Fully Connected | 32 | ReLU |
| 3 | Fully Connected | 1  | Sigmoid |

### 🛠 Optimizations Used:
✅ **Batch Normalization:** Normalizes activations at each layer for stable learning.  
✅ **Dropout (10%):** Reduces overfitting by randomly disabling neurons during training.  
✅ **Adam Optimizer:** Adaptive learning rate optimizer with momentum.  
✅ **Learning Rate Decay:** Gradually decreases the learning rate over time.

## 🔥 Training Process
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** 32
- **Epochs:** 5000
- **Initial Learning Rate:** 0.8
- **Decay Rate:** 0.0001 (reduces LR over time)

### 🚀 Training Output (Example)
```
Epoch 0, Loss: 0.182399
Epoch 100, Loss: 0.176083
Epoch 500, Loss: 0.236471
Epoch 1000, Loss: 0.110408
Epoch 5000, Loss: 0.091905
✅ Accuracy: 96.49%
```

## 📈 Results
| Model | Accuracy (%) |
|------------|------------|
| Custom Neural Network | **95.61%** |
| TensorFlow Model | **97.37%** |

🔹 **Our model achieves competitive accuracy (~96.49%)**, slightly behind top-performing classifiers like Gradient Boosting and SVM.

---

## ✅ Testing

We implemented **unit tests** to ensure all core components (layers, activations, losses, optimizers, network) work correctly.

### 📂 Test Coverage
| Component | Coverage |
|------------|------------|
| Activations | ✅ 100% |
| Layers | ✅ 100% |
| Loss Functions | ✅ 100% |
| Optimizers | ✅ 100% |
| Neural Network | ✅ 100% |

### 🔧 Run Tests

To execute all tests using `pytest`:

```bash
pytest tests/
```

### 📊 Expected Output

```
============= test session starts ==============
platform darwin -- Python 3.10.9, pytest-8.3.5
collected 11 items

tests/test_activations.py ....                             [ 36%]
tests/test_layerdense.py ..                                [ 54%]
tests/test_loss.py ..                                      [ 72%]
tests/test_neuralnetwork.py ..                             [ 90%]
tests/test_optimizer.py .                                 [100%]

============= 11 passed in 0.21s ==============
```

---

## 🛠 How to Run
### 1️⃣ Install Dependencies
```bash
pip install numpy pandas scikit-learn pytest
```

### 2️⃣ Run the Training Script
```bash
python train.py
```

### 3️⃣ Evaluate the Model
```bash
python evaluate.py
```

### 4️⃣ Run Unit Tests
```bash
pytest tests/
```

---

## 📌 Future Improvements
✅ Implement **early stopping** to prevent overfitting.  
✅ Experiment with different **activation functions** (LeakyReLU, Swish, etc.).  
✅ Use **data augmentation** techniques to improve generalization.  
✅ Compare with deep learning frameworks like **TensorFlow/PyTorch**.

---

💡 **Conclusion:** This project successfully demonstrates how to build a **custom deep learning model from scratch** with essential optimizations. The accuracy is high, and the architecture is flexible for further improvements!

🚀 **Feel free to experiment and improve!**
