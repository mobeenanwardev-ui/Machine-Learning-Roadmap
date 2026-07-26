"""Beginner-friendly Linear Regression example.

This script is intentionally written so that a reader can understand the
whole workflow from top to bottom:

    data -> train/test split -> scaling -> model -> prediction -> evaluation

It also shows Lasso regression and GridSearchCV because those are useful
extensions of ordinary Linear Regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# 1. Create a small, self-contained building-energy dataset
# -----------------------------------------------------------------------------
# The values are synthetic so the example can run without downloading files.
# The relationship is designed to be intuitive:
# - larger buildings usually need more heating,
# - better insulation usually reduces heating demand,
# - more outside-wall area usually increases heating demand.

rng = np.random.default_rng(42)
n_samples = 250

floor_area_m2 = rng.uniform(50, 250, n_samples)
insulation_score = rng.uniform(1, 10, n_samples)
outside_wall_area_m2 = rng.uniform(30, 180, n_samples)

noise = rng.normal(0, 4, n_samples)
heating_load_kw = (
    0.12 * floor_area_m2
    - 2.4 * insulation_score
    + 0.08 * outside_wall_area_m2
    + 20
    + noise
)

df = pd.DataFrame(
    {
        "floor_area_m2": floor_area_m2,
        "insulation_score": insulation_score,
        "outside_wall_area_m2": outside_wall_area_m2,
        "heating_load_kw": heating_load_kw,
    }
)

# X = information given to the model.
# y = numerical value we want the model to predict.
X = df[["floor_area_m2", "insulation_score", "outside_wall_area_m2"]]
y = df["heating_load_kw"]


# -----------------------------------------------------------------------------
# 2. Split the data
# -----------------------------------------------------------------------------
# Training data teaches the model.
# Test data checks whether the model works on unseen examples.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# -----------------------------------------------------------------------------
# 3. Scale the features
# -----------------------------------------------------------------------------
# Fit ONLY on the training data to avoid leaking information from the test set.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------------------------------------------------
# 4. Train ordinary Linear Regression
# -----------------------------------------------------------------------------

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict heating demand for unseen buildings.
y_pred = model.predict(X_test_scaled)


# -----------------------------------------------------------------------------
# 5. Evaluate the predictions
# -----------------------------------------------------------------------------
# R²: closer to 1 is better.
# MSE: smaller is better.

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Linear Regression")
print(f"R²:  {r2:.3f}")
print(f"MSE: {mse:.3f}\n")


# -----------------------------------------------------------------------------
# 6. Inspect the learned coefficients
# -----------------------------------------------------------------------------
# A positive coefficient pushes the prediction upward.
# A negative coefficient pushes the prediction downward.

coefficients = pd.Series(model.coef_, index=X.columns)
print("Learned coefficients after standardisation:")
print(coefficients.sort_values(ascending=False))
print()


# -----------------------------------------------------------------------------
# 7. Visualise actual values versus predictions
# -----------------------------------------------------------------------------

plt.figure()
plt.scatter(y_test, y_pred)
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())
plt.plot([minimum, maximum], [minimum, maximum])
plt.xlabel("Actual heating load (kW)")
plt.ylabel("Predicted heating load (kW)")
plt.title("Linear Regression: actual vs predicted")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 8. Lasso Regression: Linear Regression + L1 regularisation
# -----------------------------------------------------------------------------
# Lasso can shrink some coefficients toward zero. Alpha controls how strongly
# we regularise the model. GridSearchCV tries several alpha values and selects
# the one that performs best during cross-validation.

alphas = np.logspace(-4, 0, 50)

lasso_search = GridSearchCV(
    estimator=Lasso(max_iter=10_000),
    param_grid={"alpha": alphas},
    cv=5,
    scoring="neg_mean_squared_error",
)

lasso_search.fit(X_train_scaled, y_train)

best_lasso = lasso_search.best_estimator_
lasso_pred = best_lasso.predict(X_test_scaled)

print("Lasso Regression")
print(f"Best alpha: {lasso_search.best_params_['alpha']:.6f}")
print(f"R²:         {r2_score(y_test, lasso_pred):.3f}")
print(f"MSE:        {mean_squared_error(y_test, lasso_pred):.3f}")

lasso_coefficients = pd.Series(best_lasso.coef_, index=X.columns)
print("\nLasso coefficients:")
print(lasso_coefficients.sort_values(ascending=False))


# -----------------------------------------------------------------------------
# Human memory map
# -----------------------------------------------------------------------------
# Linear Regression:
#   inputs -> weighted sum + bias -> continuous prediction
#
# fit()      = learn the weights
# predict()  = use those learned weights on new data
# R²         = how much variation the model explains
# MSE        = average squared prediction error
# Lasso      = Linear Regression + L1 penalty
# alpha      = regularisation strength
# GridSearch = automatically try several hyperparameter values
