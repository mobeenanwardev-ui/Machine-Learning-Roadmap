"""
Titanic Decision Tree Classification and Model Evaluation
=========================================================

This script follows the complete workflow used in the exercise:

1. Load the Titanic dataset.
2. Replace hidden missing-value symbols.
3. Rename unclear columns.
4. Convert numerical text columns into numbers.
5. Engineer passenger title and cabin deck features.
6. Remove high-cardinality and leakage columns.
7. Split features and target into training and test sets.
8. Fit missing-value handling on training data only.
9. Fit OneHotEncoder on training data only.
10. Train a Decision Tree Classifier.
11. Make class and probability predictions.
12. Calculate the confusion matrix and evaluation metrics.
13. Draw the confusion matrix and ROC Curve.
14. Compare different decision thresholds.

Positive class in this project:
    survived = 1

Negative class:
    survived = 0
"""

# =============================================================================
# 1. Import libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# Display every DataFrame column when printing tables.
pd.set_option("display.max_columns", None)

# Using a fixed random seed makes the train/test split and tree reproducible.
RANDOM_SEED = 0

# Remote CSV used in the university exercise.
DATA_URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"


# =============================================================================
# 2. Helper functions for feature engineering
# =============================================================================


def get_first_cabin(cabin_value: object) -> object:
    """Keep only the first cabin when several cabins are written in one cell.

    Example:
        "C22 C26" -> "C22"
        "B57 B59 B63 B66" -> "B57"
        NaN -> NaN
    """

    if pd.isna(cabin_value):
        return np.nan

    return str(cabin_value).split()[0]



def extract_title(passenger_name: object) -> str:
    """Extract a small, useful title category from the full passenger name.

    Example:
        "Braund, Mr. Owen Harris" -> "Mr"
        "Cumings, Mrs. John Bradley" -> "Mrs"

    Rare titles are grouped into "Other" to avoid creating too many categories.
    """

    if pd.isna(passenger_name):
        return "Other"

    name_text = str(passenger_name)

    try:
        # The Titanic names usually follow: Surname, Title. First names
        title = name_text.split(",", maxsplit=1)[1].split(".", maxsplit=1)[0].strip()
    except (IndexError, AttributeError):
        return "Other"

    common_titles = {"Mr", "Mrs", "Miss", "Master"}

    if title in common_titles:
        return title

    return "Other"



def get_cabin_deck(cabin_value: object) -> object:
    """Extract the first letter of the cleaned cabin number.

    Example:
        "C22" -> "C"
        "B57" -> "B"
        NaN -> NaN
    """

    if pd.isna(cabin_value):
        return np.nan

    cabin_text = str(cabin_value)
    return cabin_text[0] if cabin_text else np.nan


# =============================================================================
# 3. Load and clean the raw dataset
# =============================================================================


def load_and_prepare_raw_data() -> pd.DataFrame:
    """Load the Titanic data and perform cleaning that does not learn statistics."""

    # Read the CSV into a pandas DataFrame.
    data = pd.read_csv(DATA_URL)

    print("\nFirst five raw rows:")
    print(data.head())

    print("\nRaw dataset shape:")
    print(data.shape)

    # In this dataset, some missing values are stored as the text "?".
    # Replacing them with np.nan allows pandas and Scikit-Learn to recognise them.
    data = data.replace("?", np.nan)

    # Rename unclear abbreviations so the remaining code is easier to understand.
    column_names = {
        "pclass": "passenger_class",
        "survived": "survived",
        "name": "passenger_name",
        "sex": "passenger_sex",
        "age": "age_years",
        "sibsp": "siblings_spouses_aboard",
        "parch": "parents_children_aboard",
        "ticket": "ticket_number",
        "fare": "ticket_fare",
        "cabin": "cabin_number",
        "embarked": "embarkation_port",
        "boat": "lifeboat_number",
        "body": "body_number",
        "home.dest": "home_destination",
    }

    data = data.rename(columns=column_names)

    # These columns should be numerical but were loaded as object/text columns.
    # errors="coerce" converts invalid values into NaN instead of raising an error.
    numerical_text_columns = ["age_years", "ticket_fare", "body_number"]

    for column_name in numerical_text_columns:
        data[column_name] = pd.to_numeric(data[column_name], errors="coerce")

    # Feature engineering: simplify high-cardinality text columns.
    data["cabin_number"] = data["cabin_number"].apply(get_first_cabin)
    data["cabin_deck"] = data["cabin_number"].apply(get_cabin_deck)
    data["passenger_title"] = data["passenger_name"].apply(extract_title)

    # Remove columns that are identifiers, too detailed, or leak the target.
    columns_to_drop = [
        "passenger_name",      # The useful title has already been extracted.
        "ticket_number",       # High-cardinality identifier.
        "cabin_number",        # The simpler cabin_deck feature is retained.
        "lifeboat_number",     # Strong target leakage: known after the event.
        "body_number",         # Strong target leakage: known after the event.
        "home_destination",    # Many categories and many missing values.
    ]

    data = data.drop(columns=columns_to_drop)

    print("\nPrepared columns before train/test splitting:")
    print(data.columns.tolist())

    print("\nMissing-value percentage before imputation:")
    print((data.isnull().mean() * 100).sort_values(ascending=False).round(2))

    return data


# =============================================================================
# 4. Separate features and target, then create train/test sets
# =============================================================================


def create_train_test_sets(data: pd.DataFrame):
    """Separate X and y, then split them together.

    The split happens before fitting imputers and encoders. This prevents the
    test set from influencing preprocessing statistics or learned categories.
    """

    # X contains model inputs. y contains the answer to predict.
    X = data.drop(columns="survived")
    y = data["survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("\nTrain/test sizes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 5. Handle missing values and encode categorical features
# =============================================================================


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """Fit preprocessing on training data and apply it to both datasets."""

    # Work on copies so the original split objects are not modified unexpectedly.
    X_train = X_train.copy()
    X_test = X_test.copy()

    numerical_columns = [
        "passenger_class",
        "age_years",
        "siblings_spouses_aboard",
        "parents_children_aboard",
        "ticket_fare",
    ]

    categorical_columns = [
        "passenger_sex",
        "embarkation_port",
        "passenger_title",
        "cabin_deck",
    ]

    # -------------------------------------------------------------------------
    # 5.1 Numerical missing values
    # -------------------------------------------------------------------------

    # The imputer learns one median for each numerical training column.
    numerical_imputer = SimpleImputer(strategy="median")

    X_train[numerical_columns] = numerical_imputer.fit_transform(
        X_train[numerical_columns]
    )

    # The test set uses exactly the medians learned from training data.
    X_test[numerical_columns] = numerical_imputer.transform(
        X_test[numerical_columns]
    )

    print("\nNumerical medians learned from training data:")
    print(
        pd.Series(
            numerical_imputer.statistics_,
            index=numerical_columns,
            name="training_median",
        )
    )

    # -------------------------------------------------------------------------
    # 5.2 Categorical missing values
    # -------------------------------------------------------------------------

    categorical_imputer = SimpleImputer(strategy="most_frequent")

    X_train[categorical_columns] = categorical_imputer.fit_transform(
        X_train[categorical_columns]
    )

    X_test[categorical_columns] = categorical_imputer.transform(
        X_test[categorical_columns]
    )

    print("\nMost frequent categories learned from training data:")
    print(
        pd.Series(
            categorical_imputer.statistics_,
            index=categorical_columns,
            name="most_frequent_category",
        )
    )

    # -------------------------------------------------------------------------
    # 5.3 One-hot encoding
    # -------------------------------------------------------------------------

    # drop="first" creates k-1 binary variables for every categorical feature.
    # handle_unknown="ignore" prevents errors if the test set contains a new
    # category that did not appear in the training data.
    # sparse_output=False returns a normal NumPy array that is easy to inspect.
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    X_train_encoded_array = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded_array = encoder.transform(X_test[categorical_columns])

    encoded_column_names = encoder.get_feature_names_out(categorical_columns)

    X_train_encoded = pd.DataFrame(
        X_train_encoded_array,
        columns=encoded_column_names,
        index=X_train.index,
    )

    X_test_encoded = pd.DataFrame(
        X_test_encoded_array,
        columns=encoded_column_names,
        index=X_test.index,
    )

    # Remove the original string columns and add the new numerical columns.
    X_train_numeric = X_train.drop(columns=categorical_columns)
    X_test_numeric = X_test.drop(columns=categorical_columns)

    X_train_final = pd.concat(
        [X_train_numeric, X_train_encoded],
        axis=1,
    )

    X_test_final = pd.concat(
        [X_test_numeric, X_test_encoded],
        axis=1,
    )

    # Make sure train and test contain the same columns in the same order.
    X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

    print("\nFinal model features:")
    print(X_train_final.columns.tolist())

    print("\nRemaining missing values:")
    print(f"Training: {int(X_train_final.isnull().sum().sum())}")
    print(f"Testing:  {int(X_test_final.isnull().sum().sum())}")

    # Decision Trees do not require feature scaling because they use threshold
    # splits rather than distance or gradient calculations.

    return X_train_final, X_test_final, encoder


# =============================================================================
# 6. Train the Decision Tree Classifier
# =============================================================================


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """Create and fit the Decision Tree Classifier."""

    model = DecisionTreeClassifier(random_state=RANDOM_SEED)

    # fit() learns the decision rules from the training features and answers.
    model.fit(X_train, y_train)

    return model


# =============================================================================
# 7. Evaluate the model
# =============================================================================


def evaluate_model(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Calculate metrics, print interpretations, and draw evaluation plots."""

    # Final class predictions: 0 or 1.
    y_pred = model.predict(X_test)

    # Probability for the positive class, survived = 1.
    y_probability = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # 7.1 Confusion Matrix
    # -------------------------------------------------------------------------

    matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matrix.ravel()

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nConfusion Matrix values:")
    print(f"True Negative  (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive  (TP): {tp}")

    # -------------------------------------------------------------------------
    # 7.2 Accuracy, Precision, Recall, and F1
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nMain evaluation metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nInterpretation:")
    print(
        f"- Accuracy: {accuracy:.1%} of all test predictions were correct."
    )
    print(
        f"- Precision: when the model predicted survival, it was correct "
        f"{precision:.1%} of the time."
    )
    print(
        f"- Recall: the model detected {recall:.1%} of all passengers who "
        f"actually survived."
    )
    print(
        f"- F1 Score: the balance between Precision and Recall was {f1:.1%}."
    )

    # -------------------------------------------------------------------------
    # 7.3 Classification Report
    # -------------------------------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Did not survive", "Survived"],
            zero_division=0,
        )
    )

    # -------------------------------------------------------------------------
    # 7.4 ROC Curve and AUC
    # -------------------------------------------------------------------------

    fpr, tpr, thresholds = roc_curve(y_test, y_probability)
    auc = roc_auc_score(y_test, y_probability)

    print(f"ROC-AUC: {auc:.4f}")

    # Display a few threshold points so the relationship is visible.
    threshold_table = pd.DataFrame(
        {
            "threshold": thresholds,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
        }
    )

    print("\nFirst ROC threshold points:")
    print(threshold_table.head(10))

    # -------------------------------------------------------------------------
    # 7.5 Confusion Matrix plot
    # -------------------------------------------------------------------------

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Did not survive", "Survived"],
    ).plot(values_format="d")

    plt.title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 7.6 ROC Curve plot
    # -------------------------------------------------------------------------

    plt.figure()
    plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {auc:.3f})")
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("Decision Tree ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 7.7 Compare custom decision thresholds
    # -------------------------------------------------------------------------

    print("\nPerformance at different decision thresholds:")

    threshold_results = []

    for threshold in [0.30, 0.50, 0.70]:
        custom_predictions = (y_probability >= threshold).astype(int)

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_test,
                    custom_predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    custom_predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_test,
                    custom_predictions,
                    zero_division=0,
                ),
            }
        )

    threshold_results_frame = pd.DataFrame(threshold_results)
    print(threshold_results_frame.round(4))

    print(
        "\nA lower threshold usually predicts more positive cases, which may "
        "increase Recall and false alarms. A higher threshold is stricter and "
        "may increase Precision while missing more positive cases."
    )


# =============================================================================
# 8. Run the complete project
# =============================================================================


def main() -> None:
    """Run every stage of the Titanic Decision Tree project."""

    data = load_and_prepare_raw_data()

    X_train, X_test, y_train, y_test = create_train_test_sets(data)

    X_train_final, X_test_final, _encoder = preprocess_features(
        X_train,
        X_test,
    )

    model = train_decision_tree(X_train_final, y_train)

    evaluate_model(model, X_test_final, y_test)


if __name__ == "__main__":
    main()
