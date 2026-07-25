# Explainability and Causality

This section documents the ideas behind model explainability and then applies them to a Random Forest trained on the Titanic dataset.

The goal is not only to generate feature-importance plots, but to understand **what each method is actually telling us** and what it is **not** telling us.

## Files in this section

- [`README.md`](./README.md) — concepts, exam-focused interpretation, code snippets, and recorded outputs from the course exercise.
- [`titanic_explainability.py`](./titanic_explainability.py) — executable Python version of the complete workflow.

---

# 1. Prediction versus explanation

A predictive model answers:

> **What will the model predict?**

Explainability answers:

> **Why or how did the model arrive at that prediction?**

For example, a model may predict that a Titanic passenger is likely to survive. Explainability helps us inspect which features, such as sex, passenger class, age, or fare, influenced that prediction.

Explainability is useful for:

- justifying model decisions,
- debugging and improving models,
- building trust,
- auditing behaviour,
- and discovering what information the model relies on.

In high-stakes areas such as healthcare, finance, security, and law, explanations are especially important because a wrong or unexplained decision can have serious consequences.

---

# 2. Global versus local explainability

## Local explanation

A local explanation focuses on **one prediction**.

Example:

> Why did the model predict that this particular passenger would survive?

SHAP is especially useful for this because it can assign a contribution to every feature for one individual prediction.

## Global explanation

A global explanation focuses on the **overall behaviour of the model**.

Example:

> Across all passengers, which features does the model rely on most?

Built-in Random Forest importance, permutation importance, and aggregated SHAP values can all provide global information.

A useful memory rule is:

```text
Local  -> Why this prediction?
Global -> What generally drives the model?
```

---

# 3. Inherent versus post-hoc explainability

Some models are easier to understand directly from their structure.

Examples:

- **Decision Tree** — follow the decision path from root to leaf.
- **Logistic Regression** — inspect the learned coefficients.

These are often called **glass-box** or inherently interpretable models.

A Random Forest contains many Decision Trees. Each individual tree can be inspected, but the complete forest is difficult to understand manually. We therefore often use a separate explainability method after training.

```text
Glass box:
Model itself -> Explanation

More complex model:
Model -> Explainability method -> Explanation
```

This distinction is about **where the explanation comes from**, not simply whether the explanation is produced before or after a prediction.

---

# 4. Random Forest built-in feature importance

Scikit-Learn exposes:

```python
model.feature_importances_
```

For a Random Forest, this is an impurity-based importance, often called:

- Mean Decrease in Impurity (MDI)
- Gini importance

The idea is:

```text
Feature used in useful tree splits
          ↓
Impurity decreases
          ↓
Feature receives importance
```

A larger value means the feature contributed more to useful splits across the forest.

This method is fast and convenient, but it describes how the trained forest used its features. It should not be interpreted as causality.

### Output from the exercise

Approximate values read from the saved course-exercise plot:

```text
Feature     MDI / Gini importance
age         0.273
sex         0.252
fare        0.251
pclass      0.122
sibsp       0.046
parch       0.032
embarked    0.022
```

So `age`, `sex`, and `fare` receive the largest impurity-based importance values in this run.

These numbers are recorded from the saved exercise output. Small differences are possible if the dataset or library versions change.

---

# 5. Permutation Feature Importance

Permutation importance asks a different question:

> **How much does model performance suffer when the information in one feature is destroyed?**

The algorithm is:

```text
1. Evaluate the trained model normally
2. Choose one feature
3. Shuffle that feature across rows
4. Do NOT retrain the model
5. Evaluate the model again
6. Importance = baseline score - shuffled score
7. Repeat the shuffle several times and average
```

In this exercise the performance metric is the **F1 score**.

```python
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
)
```

The average importance is available through:

```python
result.importances_mean
```

## How to interpret it

```text
Large positive value
-> performance drops strongly after shuffling
-> model relies on the feature

Near zero
-> shuffling has little effect

Negative value
-> the model performs slightly better after shuffling
```

Negative importance does **not** mean "very important in the opposite direction." It can indicate noise, overfitting, redundancy, or an unstable estimate.

## Important limitation: correlated features

Suppose `age` and `year_of_birth` contain almost the same information.

If only `age` is shuffled, `year_of_birth` may still provide the same signal. The measured importance of `age` can therefore look smaller than expected.

Permutation importance measures the importance of a feature **given that all the other features are still available**.

### Output from the exercise

Approximate values read from the saved course-exercise plot:

```text
Feature     Mean decrease in F1
sex          0.202
pclass       0.116
age          0.079
fare         0.026
sibsp        0.014
parch       -0.005
embarked    -0.005
```

`sex` produces the largest decrease in F1 score when shuffled, followed by `pclass` and `age`.

The slightly negative values for `parch` and `embarked` mean that this particular model/run performed a little better after their information was shuffled. They should not be interpreted as "negative causes" of survival.

---

# 6. SHAP and Shapley values

SHAP is based on Shapley values from cooperative game theory.

The mapping to machine learning is:

```text
Players -> Features
Payout  -> Model prediction
```

The question becomes:

> How much credit or blame should each feature receive for the prediction?

## Marginal contribution

For feature `i` and an already available set of features `S`:

```text
Contribution of i = f(S ∪ {i}) - f(S)
```

The contribution of a feature can change depending on which other features are already present. SHAP therefore considers different possible feature contexts/orderings and combines them fairly.

This is a conceptual calculation. We are **not physically rearranging the columns of the dataset**.

---

# 7. Tree SHAP

Because the model in this exercise is a Random Forest, the SHAP package provides `TreeExplainer`.

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

`shap_values` contains feature contributions for the test observations.

For a binary classifier in the version used by this exercise:

```python
shap_values[:, :, 1]
```

selects the SHAP values for class `1`, where:

```text
1 = survived
0 = did not survive
```

---

# 8. Global SHAP importance

A global SHAP bar plot aggregates the magnitude of local SHAP values across many observations.

```python
shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
    plot_type="bar",
)
```

The plot uses the mean absolute SHAP value:

```text
mean(|SHAP value|)
```

The absolute value is important because both of these can represent a strong influence:

```text
+0.30 -> strong push toward class 1
-0.30 -> strong push away from class 1
```

For global importance, both have magnitude `0.30`.

### Output from the exercise

Approximate mean absolute SHAP values read from the saved course-exercise plot:

```text
Feature     mean(|SHAP value|)
sex          0.212
pclass       0.114
fare         0.057
age          0.056
sibsp        0.021
parch        0.017
embarked     0.015
```

So the ranking is approximately:

```text
sex > pclass > fare ≈ age > sibsp > parch > embarked
```

This means `sex` has the largest average influence on the Random Forest's output in this run.

---

# 9. SHAP beeswarm plot

The beeswarm plot contains more information than the bar plot.

```python
shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
)
```

How to read it:

| Plot element | Meaning |
|---|---|
| One dot | One sample/passenger |
| Y-axis | Features, ranked by overall importance |
| X-axis | SHAP value / impact on the model output |
| Positive SHAP value | Pushes the prediction toward class 1 |
| Negative SHAP value | Pushes the prediction away from class 1 |
| Colour | Original feature value for that passenger |

For this exercise:

```text
Right side  -> pushes toward survival
Left side   -> pushes toward non-survival
```

The beeswarm shows both **global importance** and the **direction of individual feature effects on model predictions**.

---

# 10. Feature importance is not causality

This is one of the most important lessons in this section.

Suppose SHAP says that `fare` is important.

That means:

> The model uses fare strongly when making predictions.

It does **not** automatically mean:

> Increasing a passenger's fare would have caused that passenger to survive.

Fare may carry information about other factors such as passenger class, cabin location, access to lifeboats, or other conditions.

```text
Predictive importance:
"The model uses X."

Causal effect:
"Changing X changes the real-world outcome."
```

SHAP explains the **model**. It does not prove real-world cause and effect.

---

# 11. Causal analysis

Causal analysis asks:

> **What happens to Y if we actively change X?**

This is stronger than finding a correlation or a predictive association.

Three important ideas are:

## Causal inference

Determine whether a change in `X` actually causes a change in `Y`.

## Counterfactual

Ask:

> What would have happened to the same or a comparable case under a different action?

For example:

```text
Observed:
Patient received Treatment A -> recovered

Counterfactual question:
What would have happened without Treatment A?
```

## Intervention

Actively change or set something and study the effect on the outcome.

```text
Apply treatment
      ↓
Does recovery improve?
```

Causal reasoning is therefore especially important for **decision-making**, because decisions require us to understand the effect of actions, not only what a predictive model has learned.

---

# 12. Confounders

A confounder is a variable that influences both the apparent cause and the outcome, creating a misleading relationship.

Classic structure:

```text
        Confounder
        /        \
       ↓          ↓
      X    ---->  Y
```

An observed association between `X` and `Y` may partly or completely come from the confounder.

This is why predictive importance alone cannot answer causal questions.

---

# 13. Complete Titanic workflow

The practical part of this section follows this pipeline:

```text
Load Titanic data
       ↓
Select features
       ↓
Remove missing rows
       ↓
Train/test split
       ↓
Encode categorical features
       ↓
Train Random Forest
       ↓
Built-in MDI/Gini importance
       ↓
Permutation importance
       ↓
Tree SHAP
       ↓
Global SHAP bar plot
       ↓
SHAP beeswarm plot
       ↓
Interpret model behaviour
       ↓
Remember: explanation ≠ causality
```

---

# 14. Compact code cheat sheet

```python
# Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Built-in MDI / Gini importance
feature_names = model.feature_names_in_
mdi_importance = model.feature_importances_

# Permutation importance
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
)
permutation_scores = result.importances_mean

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global SHAP
shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
    plot_type="bar",
)

# SHAP beeswarm
shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
)
```

## Three-method memory rule

```text
MDI / Gini
-> How useful was the feature for tree splits?

Permutation importance
-> What happens to performance when I shuffle the feature?

SHAP
-> How much did the feature contribute to predictions?
```

And always remember:

```text
Explainability / feature importance ≠ causality
```

---

# Running the code

Install the required packages:

```bash
pip install pandas matplotlib seaborn scikit-learn shap
```

Then run:

```bash
python titanic_explainability.py
```

The Python script prints the importance tables and displays the three main explainability visualisations.

The output tables above are recorded from the saved course exercise, while the script lets the same workflow be rerun and the plots regenerated.
