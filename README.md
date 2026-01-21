# 🧠 Girlfriend Prediction Using Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **learning project** focused on understanding core machine learning concepts by implementing Logistic Regression both from scratch and using scikit-learn.

> 🎓 **Purpose**: This project is designed to learn and demonstrate understanding of:
> - **Gradient descent optimization** from first principles
> - **Binary cross-entropy loss** function
> - **Sklearn Pipelines** for clean ML workflows
>
> *Uses synthetic data to focus on algorithm implementation rather than real-world prediction.*

> ⚠️ **Disclaimer**: This is purely an educational project. Relationship outcomes depend on countless factors beyond what any model can capture!

---

## 📋 Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Learning Outcomes](#learning-outcomes)

---

## 🎯 Overview

This project contains **two implementations** of Logistic Regression:

| Notebook | Approach | Purpose |
|----------|----------|----------|
| `gf_pred_manual.ipynb` | From scratch with NumPy | Understand gradient descent |
| `gf_pred_sklearn.ipynb` | Using sklearn Pipeline | Learn production-style workflow |

### Key Highlights
- ✨ Manual implementation with no ML libraries
- 📊 Custom gradient descent with 100,000 iterations
- 📈 Visualization of cost function convergence
- 🔧 Sklearn Pipeline for comparison

---

## 📊 Dataset

The dataset (`indian_boys_gf_prediction_balanced.csv`) contains **300 balanced samples** with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| `age` | Age in years | 18-30 |
| `height_cm` | Height in centimeters | 155-190 |
| `income_lpa` | Annual income (Lakhs/year) | 1.5-20 |
| `fitness_level` | Self-rated fitness score | 1-10 |
| `confidence` | Self-rated confidence score | 1-10 |
| `social_media_hours` | Daily social media usage (hours) | 0.5-6.0 |
| **`has_gf`** | Target variable | 0 (No) / 1 (Yes) |

### Dataset Distribution
- **Class 0 (No GF)**: ~50%
- **Class 1 (Has GF)**: ~50%

---

## 🔬 Technical Approach

### 1. Feature Scaling (Standardization)
```
z = (x - μ) / σ
```
Ensures all features are on the same scale for faster gradient descent convergence.

### 2. Sigmoid Activation
```
σ(z) = 1 / (1 + e^(-z))
```
Maps linear output to probability between 0 and 1.

### 3. Binary Cross-Entropy Loss
```
J(W, B) = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### 4. Gradient Descent Update
```
W = W - α · (∂J/∂W)
B = B - α · (∂J/∂B)
```
Where α = 0.1 (learning rate)

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~65% |
| Test Accuracy | ~54% |
| Iterations | 100,000 |
| Learning Rate | 0.1 |

### Observations
- Model converges successfully (cost function decreases)
- Moderate accuracy expected due to:
  - Inherently noisy/random nature of the target
  - Limited feature set
  - Small dataset size

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/logistic_regression_gf_pred_proj.git
cd logistic_regression_gf_pred_proj

# Install dependencies
pip install pandas numpy matplotlib scikit-learn jupyter
```

---

## 💻 Usage

### Run the Notebooks
```bash
# Manual implementation (from scratch)
jupyter notebook gf_pred_manual.ipynb

# Sklearn implementation
jupyter notebook gf_pred_sklearn.ipynb
```

### Make a Prediction
```python
# Example: Predict for a new user
user_input = pd.DataFrame([{
    "age": 25,
    "height_cm": 180,
    "income_lpa": 12,
    "fitness_level": 10,
    "confidence": 7,
    "social_media_hours": 5
}])

# Scale using training statistics
scaled_input = (user_input.values - X_train.mean(axis=0)) / X_train.std(axis=0)

# Get probability
probability = sigmoid(np.dot(scaled_input, W) + B)
print(f"Probability of having GF: {probability[0][0]:.2%}")
```

---

## 📁 Project Structure

```
logistic_regression_gf_pred_proj/
├── gf_pred_manual.ipynb                    # From-scratch implementation
├── gf_pred_sklearn.ipynb                   # Sklearn Pipeline implementation
├── indian_boys_gf_prediction_balanced.csv  # Dataset
├── add_docs.py                             # Docs script (manual notebook)
├── add_docs_sklearn.py                     # Docs script (sklearn notebook)
└── README.md                               # This file
```

---

## 🎓 Learning Outcomes

This project teaches:
- ✅ Logistic Regression theory and math
- ✅ Gradient Descent from first principles
- ✅ Binary Cross-Entropy loss function
- ✅ Feature scaling importance
- ✅ Train/Test split to prevent overfitting
- ✅ Sklearn Pipelines for clean workflows

---

## 🔮 Next Steps

For future projects with real-world data:
1. **EDA**: Exploratory Data Analysis before modeling
2. **Feature Engineering**: Create meaningful features
3. **Cross-Validation**: k-fold CV for robust evaluation
4. **Hyperparameter Tuning**: GridSearchCV / RandomizedSearchCV
5. **Model Comparison**: Try multiple algorithms

---

## 📚 Learning Outcomes

This project teaches:
- ✅ Logistic Regression theory and implementation
- ✅ Gradient Descent optimization
- ✅ Binary Cross-Entropy loss function
- ✅ Feature scaling importance
- ✅ Train/Test split to prevent overfitting
- ✅ Model evaluation metrics

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

<p align="center">
  Made with ❤️ for learning Machine Learning
</p>
