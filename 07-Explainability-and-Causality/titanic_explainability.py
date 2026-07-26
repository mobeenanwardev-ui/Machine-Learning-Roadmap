"""
Titanic Explainability Exercise

Covers:
1. Random Forest built-in impurity (MDI/Gini) importance
2. Permutation Feature Importance
3. Tree SHAP
4. Global SHAP bar plot
5. SHAP beeswarm plot

This script mirrors the course exercise while keeping the workflow easy to read.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# 1. Load and prepare the Titanic data
# ---------------------------------------------------------------------------

df = sns.load_dataset("titanic")

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

# Keep only the variables used in the exercise and remove incomplete rows.
df = df[features + ["survived"]].dropna()

X = df[features].copy()
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)


# ---------------------------------------------------------------------------
# 2. Encode categorical features
# ---------------------------------------------------------------------------

sex_encoder = LabelEncoder()
X_train["sex"] = sex_encoder.fit_transform(X_train["sex"])
X_test["sex"] = sex_encoder.transform(X_test["sex"])

embarked_encoder = LabelEncoder()
X_train["embarked"] = embarked_encoder.fit_transform(X_train["embarked"])
X_test["embarked"] = embarked_encoder.transform(X_test["embarked"])


# ---------------------------------------------------------------------------
# 3. Train the Random Forest
# ---------------------------------------------------------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# 4. Built-in Random Forest importance: MDI / Gini importance
# ---------------------------------------------------------------------------

mdi_df = pd.DataFrame(
    {
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_,
    }
).sort_values("Importance", ascending=False)

print("\n=== Random Forest MDI / Gini importance ===")
print(mdi_df.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.barh(
    mdi_df["Feature"][::-1],
    mdi_df["Importance"][::-1],
)
plt.title("Random Forest MDI / Gini Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 5. Permutation Feature Importance
# ---------------------------------------------------------------------------

# The model is NOT retrained here.
# Each feature is shuffled and we measure how much the F1 score decreases.
permutation_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
)

permutation_df = pd.DataFrame(
    {
        "Feature": X_test.columns,
        "Importance": permutation_result.importances_mean,
        "Std": permutation_result.importances_std,
    }
).sort_values("Importance", ascending=False)

print("\n=== Permutation importance (mean decrease in F1) ===")
print(permutation_df.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.barh(
    permutation_df["Feature"][::-1],
    permutation_df["Importance"][::-1],
)
plt.title("Permutation Feature Importance (F1 Score)")
plt.xlabel("Mean decrease in F1 score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 6. Tree SHAP
# ---------------------------------------------------------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# The course exercise uses [:, :, 1] for class 1 = survived.
# Depending on the SHAP version, the returned shape can differ.
if isinstance(shap_values, list):
    survival_shap_values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    survival_shap_values = shap_values[:, :, 1]
else:
    survival_shap_values = shap_values


# ---------------------------------------------------------------------------
# 7. Global SHAP importance
# ---------------------------------------------------------------------------

print("\nDisplaying global SHAP feature importance...")
shap.summary_plot(
    survival_shap_values,
    X_test,
    plot_type="bar",
)


# ---------------------------------------------------------------------------
# 8. SHAP beeswarm plot
# ---------------------------------------------------------------------------

print("\nDisplaying SHAP beeswarm plot...")
shap.summary_plot(
    survival_shap_values,
    X_test,
)


# ---------------------------------------------------------------------------
# Interpretation reminder
# ---------------------------------------------------------------------------

print(
    "\nRemember: feature importance and SHAP explain what the model uses. "
    "They do not prove that changing a feature will cause the real-world outcome to change."
)
