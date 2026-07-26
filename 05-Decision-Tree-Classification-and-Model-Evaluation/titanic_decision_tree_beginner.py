"""
Beginner Titanic Decision Tree Project

This file is written from top to bottom with simple steps.
It covers preprocessing, Decision Tree training, predictions,
Confusion Matrix, Accuracy, Precision, Recall, F1 Score,
Classification Report, ROC Curve, AUC, and thresholds.

Positive class: 1 = Survived
Negative class: 0 = Did not survive
"""

# ============================================================
# 1. Import libraries
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
)


# ============================================================
# 2. Load the Titanic dataset
# ============================================================

url = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
data = pd.read_csv(url)

print("First five rows:")
print(data.head())


# ============================================================
# 3. Basic cleaning
# ============================================================

# Some missing values are written as the text "?".
data = data.replace("?", np.nan)

# We use only a few easy features in this beginner example.
data = data[
    ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
]

# Rename columns so their meaning is clearer.
data = data.rename(
    columns={
        "pclass": "passenger_class",
        "sex": "passenger_sex",
        "age": "age_years",
        "sibsp": "siblings_spouses_aboard",
        "parch": "parents_children_aboard",
        "fare": "ticket_fare",
        "embarked": "embarkation_port",
    }
)

# Convert text numbers into real numerical values.
data["age_years"] = pd.to_numeric(data["age_years"], errors="coerce")
data["ticket_fare"] = pd.to_numeric(data["ticket_fare"], errors="coerce")
data["survived"] = pd.to_numeric(data["survived"], errors="coerce")

# Remove rows only when the answer itself is missing.
data = data.dropna(subset=["survived"])
data["survived"] = data["survived"].astype(int)

print("\nMissing values before filling:")
print(data.isnull().sum())


# ============================================================
# 4. Separate features and target
# ============================================================

# X contains the input features.
# y contains the answer we want to predict.
X = data.drop(columns="survived")
y = data["survived"]


# ============================================================
# 5. Create training and test sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y,
)

X_train = X_train.copy()
X_test = X_test.copy()

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 6. Fill missing values
# ============================================================

# Learn replacement values only from the training data.
age_median = X_train["age_years"].median()
fare_median = X_train["ticket_fare"].median()
sex_mode = X_train["passenger_sex"].mode()[0]
port_mode = X_train["embarkation_port"].mode()[0]

# Use the same training values for both training and test data.
X_train["age_years"] = X_train["age_years"].fillna(age_median)
X_test["age_years"] = X_test["age_years"].fillna(age_median)

X_train["ticket_fare"] = X_train["ticket_fare"].fillna(fare_median)
X_test["ticket_fare"] = X_test["ticket_fare"].fillna(fare_median)

X_train["passenger_sex"] = X_train["passenger_sex"].fillna(sex_mode)
X_test["passenger_sex"] = X_test["passenger_sex"].fillna(sex_mode)

X_train["embarkation_port"] = X_train["embarkation_port"].fillna(port_mode)
X_test["embarkation_port"] = X_test["embarkation_port"].fillna(port_mode)


# ============================================================
# 7. Convert categories into numbers
# ============================================================

# Convert male/female into 0/1.
sex_mapping = {"male": 0, "female": 1}
X_train["passenger_sex"] = X_train["passenger_sex"].map(sex_mapping)
X_test["passenger_sex"] = X_test["passenger_sex"].map(sex_mapping)

# Convert embarkation ports into separate 0/1 columns.
X_train = pd.get_dummies(
    X_train,
    columns=["embarkation_port"],
    drop_first=True,
    dtype=int,
)

X_test = pd.get_dummies(
    X_test,
    columns=["embarkation_port"],
    drop_first=True,
    dtype=int,
)

# Make sure train and test have exactly the same columns.
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nFinal feature columns:")
print(X_train.columns.tolist())


# ============================================================
# 8. Create and train the Decision Tree
# ============================================================

model = DecisionTreeClassifier(max_depth=4, random_state=0)
model.fit(X_train, y_train)


# ============================================================
# 9. Make predictions
# ============================================================

# predict() gives the final class: 0 or 1.
y_pred = model.predict(X_test)

print("\nFirst ten predictions:")
print(y_pred[:10])

print("\nFirst ten real answers:")
print(y_test.iloc[:10].to_numpy())


# ============================================================
# 10. Confusion Matrix
# ============================================================

matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(matrix)

# The order is: TN, FP, FN, TP.
tn, fp, fn, tp = matrix.ravel()

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# 11. Calculate metrics manually
# ============================================================

manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
manual_precision = tp / (tp + fp)
manual_recall = tp / (tp + fn)
manual_f1 = 2 * (manual_precision * manual_recall) / (
    manual_precision + manual_recall
)

print("\nManual calculations:")
print("Accuracy:", round(manual_accuracy, 4))
print("Precision:", round(manual_precision, 4))
print("Recall:", round(manual_recall, 4))
print("F1 Score:", round(manual_f1, 4))


# ============================================================
# 12. Calculate metrics with Scikit-Learn
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nScikit-Learn calculations:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nInterpretation:")
print(f"Accuracy: {accuracy:.1%} of all predictions were correct.")
print(f"Precision: {precision:.1%} of predicted survivors really survived.")
print(f"Recall: {recall:.1%} of actual survivors were detected.")
print(f"F1 Score: the Precision/Recall balance is {f1:.1%}.")


# ============================================================
# 13. Classification Report
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Did not survive", "Survived"],
    )
)


# ============================================================
# 14. Draw the Confusion Matrix
# ============================================================

ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Did not survive", "Survived"],
).plot()

plt.title("Titanic Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 15. Prediction probabilities
# ============================================================

# predict_proba() gives probabilities for class 0 and class 1.
# [:, 1] selects the probability of survival.
y_probability = model.predict_proba(X_test)[:, 1]

print("\nFirst ten survival probabilities:")
print(y_probability[:10])


# ============================================================
# 16. ROC Curve and AUC
# ============================================================

# roc_curve() automatically tests many thresholds.
fpr, tpr, thresholds = roc_curve(y_test, y_probability)
auc = roc_auc_score(y_test, y_probability)

print("\nROC-AUC:", round(auc, 4))

plt.figure()
plt.plot(fpr, tpr, label=f"Decision Tree, AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.title("Titanic Decision Tree - ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 17. Try different decision thresholds
# ============================================================

# Lower threshold: more positive predictions and often higher Recall.
# Higher threshold: fewer positive predictions and often higher Precision.

for threshold in [0.30, 0.50, 0.70]:
    custom_predictions = (y_probability >= threshold).astype(int)

    threshold_precision = precision_score(
        y_test, custom_predictions, zero_division=0
    )
    threshold_recall = recall_score(
        y_test, custom_predictions, zero_division=0
    )
    threshold_f1 = f1_score(
        y_test, custom_predictions, zero_division=0
    )

    print("\nThreshold:", threshold)
    print("Precision:", round(threshold_precision, 4))
    print("Recall:", round(threshold_recall, 4))
    print("F1 Score:", round(threshold_f1, 4))


print("\nProject completed successfully.")
