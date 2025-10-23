# 🧠 Handwritten Digit Recognition using Neural Networks

A deep learning project that classifies handwritten digits from the MNIST dataset using a simple Artificial Neural Network (ANN) built with TensorFlow and Keras

---

## 📊 Project Overview

This project implements a neural network to recognize handwritten digits (0-9) from the MNIST dataset. The model achieves over **96% accuracy** on the test set using a relatively simple architecture with just a few dense layers.

---

## 🧠 Model Architecture

- **Input:** Flattened 28×28 grayscale images (784 neurons)  
- **Hidden Layer 1:** 128 neurons with ReLU activation  
- **Hidden Layer 2:** 32 neurons with ReLU activation  
- **Output Layer:** 10 neurons with Softmax activation (one for each digit)  

**Total Parameters:** 104,938


# Model Architecture
```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Model Summary
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
 dense (Dense)               (None, 128)               100480    
 dense_1 (Dense)             (None, 32)                4128      
 dense_2 (Dense)             (None, 10)                330       
=================================================================
Total params: 104,938
Trainable params: 104,938
Non-trainable params: 0
"""
