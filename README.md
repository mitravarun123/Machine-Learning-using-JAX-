# 🚀 Machine Learning From Scratch using JAX

A research-style implementation of Machine Learning algorithms **from scratch** using **JAX**, following the functional programming philosophy used by **Google DeepMind** and modern ML research labs.

This repository focuses on **understanding core machine learning algorithms mathematically and computationally**, without relying on high-level frameworks like Scikit-Learn training APIs or PyTorch trainers.

---

## 🎯 Project Goal

Most ML libraries hide important learning concepts behind `.fit()` functions.

This project aims to:

✅ Build ML algorithms from scratch
✅ Understand gradients and optimization deeply
✅ Use **JAX autodiff + JIT compilation**
✅ Follow real research training loop design
✅ Bridge theory → real ML systems

---

## 🧠 Why JAX?

JAX is used by:

* Google DeepMind
* Google Research
* OpenAI research prototypes
* Modern scientific ML projects

Key advantages:

* ⚡ Just-In-Time (JIT) compilation
* 🔁 Automatic differentiation
* 🧩 Functional programming design
* 🚀 GPU/TPU scalability
* 🔬 Research-friendly experimentation

---

## 📂 Current Implementation

### ✅ Linear Regression (From Scratch)

Implemented using:

* Pure functional model definition
* Mean Squared Error loss
* Gradient computation via `value_and_grad`
* Custom SGD optimizer
* JIT-compiled training step
* Real-world dataset (California Housing)

#### Training Pipeline

```
Parameters
    ↓
Model Forward Pass
    ↓
Loss Function
    ↓
Automatic Gradients
    ↓
Optimizer Update
    ↓
Updated Parameters
```

---

## 🏗️ Project Structure

```
.
├── linear_regression_jax.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
pip install jax jaxlib scikit-learn matplotlib
```

---

## ▶️ Run Training

```bash
python linear_regression_jax.py
```

Example output:

```
Epoch 000 | Loss 5.23 | Train RMSE 2.10 | Test RMSE 2.08
Epoch 020 | Loss 0.89 | Train RMSE 0.94 | Test RMSE 0.96
...
Final Test RMSE: 0.70
```

---

## 🧩 Core Concepts Demonstrated

* Linear Regression mathematics
* Gradient descent optimization
* Automatic differentiation
* Functional ML design
* JAX JIT compilation
* Research-style training loops

---

## 🔬 Design Philosophy (DeepMind Style)

Instead of:

```python
model.fit(X, y)
```

we explicitly implement:

```
params → forward → loss → gradients → update → new params
```

This mirrors how modern ML research systems train large models.

---

## 🚧 Upcoming Implementations

More algorithms are being implemented **from scratch** using the same JAX research template:

### 🌲 Classical Machine Learning

* Random Forest (from scratch)
* Decision Trees
* Gradient Boosting
* K-Nearest Neighbors
* Naive Bayes

### 📈 Linear Models

* Logistic Regression
* Ridge & Lasso Regression
* Polynomial Regression

### 🧠 Neural Networks

* Multilayer Perceptron (MLP)
* Backpropagation from scratch
* Custom activation functions
* Mini-batch training

### 🔥 Deep Learning (From Scratch)

* Fully Connected Neural Networks
* CNN fundamentals
* Optimizers (Adam, RMSProp)
* Regularization techniques
* Training loops inspired by DeepMind

### ⚡ Advanced JAX Topics (Coming Soon)

* `vmap` batching
* `pmap` multi-device training
* Functional optimizers
* Experiment tracking setup

---

## 🎓 Learning Objectives

By following this repository, you will learn:

* How ML algorithms actually work internally
* How gradients are computed
* How large models train efficiently
* How research codebases are structured
* How JAX differs from PyTorch/TensorFlow

---

## 🤝 Contributions

This is an educational and research-focused project.

Suggestions, improvements, and discussions are welcome!

---

## 📚 References

* JAX Documentation — https://jax.readthedocs.io
* DeepMind Engineering Blog
* Pattern Recognition and Machine Learning — Bishop
* Hands-On Machine Learning — Aurélien Géron

---

## ⭐ Future Vision

This repository will evolve into a **complete Machine Learning From Scratch using JAX** collection — covering classical ML → modern deep learning systems.

More exciting implementations coming soon 🚀

---

### 👨‍💻 Author

Built as part of a journey toward mastering:

**Machine Learning • Deep Learning • LLMs • Research Engineering**
