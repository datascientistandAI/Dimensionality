# Dimensionality Reduction Techniques

## Project Overview
This project explores dimensionality reduction techniques in machine learning, focusing on Principal Component Analysis (PCA) and its implementation using Python. The notebook provides theoretical insights, practical implementations, and visualizations to enhance understanding of how dimensionality reduction can simplify models and improve performance.

## What I've Learned

- **Dimensionality Reduction Basics**: Gained insights into why dimensionality reduction is important, including its role in reducing computational costs and avoiding the curse of dimensionality.
- **Principal Component Analysis (PCA)**: Implemented PCA from scratch using Singular Value Decomposition (SVD) to identify the principal components of a dataset.
- **Data Visualization**: Visualized the results of dimensionality reduction to understand how data points are distributed in a lower-dimensional space.

## Getting Started

### Prerequisites

To run this notebook, ensure you have the following:

- Python version ≥ 3.5
- Scikit-Learn version ≥ 0.20
- Required libraries:
  - NumPy
  - Matplotlib

You can install the necessary libraries using pip:

```bash
pip install numpy matplotlib scikit-learn
```

### Running the Notebook

To execute the notebook, follow these steps:

1. Clone or download the repository containing the notebook.
2. Open the notebook in a Jupyter environment.
3. Run the cells sequentially to load data, perform dimensionality reduction, and visualize results.

## Key Concepts Covered

### 1. Setup

The notebook starts by importing necessary libraries and setting up the environment for reproducibility. Key imports include:

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Data Generation

Synthetic data is generated to illustrate dimensionality reduction. The following code creates a 3D dataset:

```python
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
```

### 3. Centering the Data

Before applying PCA, the data must be centered. This is done by subtracting the mean from each feature:

```python
X_centered = X - X.mean(axis=0)
```

### 4. Singular Value Decomposition (SVD)

The notebook uses NumPy's `svd()` function to perform SVD on the centered data, extracting the principal components:

```python
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]  # First principal component
c2 = Vt.T[:, 1]  # Second principal component
```

### 5. Reconstruction and Verification

The reconstruction of the centered data is verified to ensure that the SVD was performed correctly:

```python
np.allclose(X_centered, U.dot(S).dot(Vt))
```

### 6. Visualization of Results

The results of the dimensionality reduction are visualized, allowing for a better understanding of how data points are distributed in lower dimensions.

## Conclusion

This project serves as an educational resource for anyone interested in machine learning and data science, focusing specifically on dimensionality reduction techniques. By exploring PCA and its implementation using SVD, users gain a deeper understanding of how these methods can simplify complex datasets and enhance model performance.
