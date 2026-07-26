"""Logistic Regression revision examples.

This file contains two small examples:
1. A practical scikit-learn workflow using the breast-cancer dataset.
2. A compact from-scratch binary Logistic Regression implementation.

The code is intentionally written for revision and readability rather than
maximum performance.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sklearn_example() -> None:
    """Train and evaluate Logistic Regression with scikit-learn."""
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Fit preprocessing on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = model.predict(X_test_scaled)

    print("=== scikit-learn Logistic Regression ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nFirst five Class-1 probabilities:")
    print(np.round(probabilities[:5], 4))
    print("\nClassification report:")
    print(classification_report(y_test, predictions))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Convert raw scores into values between 0 and 1."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Return average Binary Cross-Entropy loss."""
    p = np.clip(probabilities, eps, 1.0 - eps)
    return float(
        -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    )


def fit_logistic_regression_from_scratch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.1,
    epochs: int = 2000,
) -> tuple[np.ndarray, float]:
    """Fit binary Logistic Regression using Gradient Descent.

    Core training loop:
        z = X @ w + b
        p = sigmoid(z)
        error = p - y
        dw = X.T @ error / m
        db = mean(error)
        w -= learning_rate * dw
        b -= learning_rate * db
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    for epoch in range(epochs):
        # FORWARD: weighted sum -> sigmoid -> probability.
        z = X @ weights + bias
        probabilities = sigmoid(z)

        # BACKWARD: gradients for weights and bias.
        error = probabilities - y
        d_weights = (X.T @ error) / n_samples
        d_bias = float(np.mean(error))

        # GRADIENT DESCENT: update trainable parameters.
        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias

        if epoch % 500 == 0 or epoch == epochs - 1:
            loss = binary_cross_entropy(y, probabilities)
            print(f"Epoch {epoch:4d} | BCE loss = {loss:.6f}")

    return weights, bias


def predict_probability_from_scratch(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    """Return Class-1 probability estimates."""
    return sigmoid(X @ weights + bias)


def predict_class_from_scratch(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert probability estimates into Class 0 / Class 1 labels."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    probabilities = predict_probability_from_scratch(X, weights, bias)
    return (probabilities >= threshold).astype(int)


def from_scratch_example() -> None:
    """Train a tiny binary classifier from scratch."""
    # Simple two-feature dataset.
    X = np.array(
        [
            [0.2, 1.0],
            [0.5, 1.4],
            [0.8, 1.1],
            [1.1, 0.9],
            [2.0, 2.1],
            [2.2, 2.5],
            [2.6, 2.3],
            [3.0, 2.8],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

    # Standardize using statistics from this training set.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std

    print("\n=== From-scratch Logistic Regression ===")
    weights, bias = fit_logistic_regression_from_scratch(
        X_scaled,
        y,
        learning_rate=0.1,
        epochs=2000,
    )

    probabilities = predict_probability_from_scratch(X_scaled, weights, bias)
    predictions = predict_class_from_scratch(X_scaled, weights, bias)

    print("\nLearned weights:", np.round(weights, 4))
    print("Learned bias:", round(bias, 4))
    print("Probabilities:", np.round(probabilities, 4))
    print("Predictions:", predictions)
    print("Actual labels:", y.astype(int))


if __name__ == "__main__":
    sklearn_example()
    from_scratch_example()
