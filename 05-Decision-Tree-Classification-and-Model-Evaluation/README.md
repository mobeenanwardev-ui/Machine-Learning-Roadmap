# Decision Tree Classification and Model Evaluation

## What this section covers

This section connects data preprocessing with the first complete classification model.

The project uses the Titanic dataset to predict whether a passenger survived. A Decision Tree Classifier learns patterns from the training data, makes predictions for unseen test data, and is then evaluated using several classification metrics.

The complete workflow is:

```text
Load the Titanic dataset
        ↓
Inspect and clean the data
        ↓
Create useful features
        ↓
Split features and target
        ↓
Create training and test sets
        ↓
Fit preprocessing on training data
        ↓
Transform training and test data
        ↓
Train a Decision Tree Classifier
        ↓
Make predictions
        ↓
Evaluate the model
```

The evaluation topics covered here are:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- ROC Curve
- AUC

---

# 1. The classification problem

The target column is `survived`.

| Target value | Meaning |
|---:|---|
| `0` | Passenger did not survive |
| `1` | Passenger survived |

This is a **binary classification problem** because the target has two possible classes.

The model receives features such as:

- passenger class,
- sex,
- age,
- number of family members aboard,
- ticket fare,
- embarkation port,
- passenger title,
- and cabin deck.

It then predicts either `0` or `1` for each passenger.

## Features and target

In Scikit-Learn, the input features are normally called `X`, while the target is called `y`.

```python
X = data.drop(columns="survived")
y = data["survived"]
```

`X` contains the information given to the model.

`y` contains the correct answer the model should learn to predict.

The target must not remain inside `X`. Otherwise, the model would receive the answer as an input, which would create data leakage.

---

# 2. Training and test data

The dataset is divided into two parts:

| Dataset | Purpose |
|---|---|
| Training set | Used by the model to learn patterns |
| Test set | Used to evaluate the trained model on unseen records |

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y,
)
```

If the dataset contains 1,000 rows and `test_size=0.20`:

- approximately 800 rows go to training,
- approximately 200 rows go to testing.

The features and targets are split together so that every feature row stays connected to its correct answer.

`stratify=y` helps preserve a similar class ratio in the training and test sets.

---

# 3. Decision Tree Classifier

A Decision Tree Classifier learns a sequence of decision rules.

A simplified tree may behave like this:

```text
Is passenger_sex = female?
        │
   ┌────┴────┐
  Yes        No
   │          │
Is class   Is age
1 or 2?    below 10?
   │          │
Prediction Prediction
```

The real tree is learned automatically from the training data. We do not manually write the rules.

## Creating the model

```python
model = DecisionTreeClassifier(random_state=0)
```

At this point, the model exists but has not learned anything.

## Training the model

```python
model.fit(X_train, y_train)
```

`fit()` means **learn from the training data**.

For a Decision Tree, fitting means learning:

- which feature should be checked,
- which threshold should be used,
- how the data should be divided,
- and which class should be predicted at each final leaf.

## Making predictions

```python
y_pred = model.predict(X_test)
```

`predict()` returns the final predicted class for every test passenger.

Example:

```text
[1, 0, 1, 0, 0, 1]
```

For this project:

- `1` means predicted survival,
- `0` means predicted non-survival.

## Prediction probabilities

```python
y_probability = model.predict_proba(X_test)[:, 1]
```

`predict_proba()` returns probabilities for both classes.

For a binary classifier, an output might look like:

| Probability of class 0 | Probability of class 1 |
|---:|---:|
| 0.20 | 0.80 |
| 0.75 | 0.25 |

`[:, 1]` selects the probability of the positive class, which is class `1` in this project.

This is different from Precision.

- `predict_proba()` gives a probability for each individual record.
- Precision evaluates how reliable the model's positive predictions were across many records.

---

# 4. Defining the positive class

Before interpreting a confusion matrix, Precision, or Recall, I must define which class is positive.

In this Titanic project:

```text
Positive class = Survived = 1
Negative class = Did not survive = 0
```

Positive does not mean good, and negative does not mean bad.

Positive simply means the class chosen as the event of interest.

Examples:

| Problem | Positive class |
|---|---|
| Titanic survival | Survived |
| Machine maintenance | Machine failure |
| Cancer screening | Has cancer |
| Fraud detection | Fraudulent transaction |
| Spam filtering | Spam |

Once the positive class is defined, TP, TN, FP, and FN have a fixed meaning.

---

# 5. Confusion Matrix

A confusion matrix compares the model's predictions with the actual test answers.

For binary classification, it is a **2 × 2 matrix containing four outcomes**.

|  | Actual negative | Actual positive |
|---|---:|---:|
| Predicted negative | True Negative | False Negative |
| Predicted positive | False Positive | True Positive |

Using Titanic survival as the positive class:

| Outcome | Meaning |
|---|---|
| True Positive (TP) | Predicted survived, actually survived |
| True Negative (TN) | Predicted did not survive, actually did not survive |
| False Positive (FP) | Predicted survived, actually did not survive |
| False Negative (FN) | Predicted did not survive, actually survived |

The easiest way to name an outcome is:

1. Look at what the model predicted: positive or negative.
2. Check whether the prediction was correct: true or false.

Example:

```text
Prediction = Positive
Prediction is wrong
Result = False Positive
```

The confusion matrix is the basic scoreboard. Accuracy, Precision, Recall, and F1 are calculated from these four values.

## Code

```python
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
```

---

# 6. Accuracy

Accuracy answers:

> Out of all predictions, how many were correct?

The correct predictions are TP and TN.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Example:

If a model makes 100 predictions and 84 are correct:

```text
Accuracy = 84 / 100 = 84%
```

## Why Accuracy can be misleading

Suppose a machine dataset contains:

- 990 healthy machines,
- 10 failed machines.

A useless model predicts every machine as healthy.

It gets 990 predictions correct:

```text
Accuracy = 990 / 1000 = 99%
```

However, it detects zero failures.

Therefore, Accuracy is useful when classes are reasonably balanced and mistakes have similar costs. It should not be trusted alone for strongly imbalanced problems.

## Code

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
```

---

# 7. Precision

Precision starts from the model's positive predictions.

It answers:

> When the model predicts the positive class, how often is it correct?

For a machine-failure system:

> Out of all machines predicted to fail, how many actually failed?

```text
Precision = TP / (TP + FP)
```

Example:

The model raises 20 failure alarms:

- 15 are real failures,
- 5 are healthy machines.

```text
Precision = 15 / 20 = 75%
```

Interpretation:

> When this model predicts failure, it is correct 75% of the time.

Precision does **not** mean that an individual machine has a 75% probability of failure. Individual probability comes from `predict_proba()`.

A high-Precision model produces fewer false alarms.

Precision is especially important when a false positive is expensive.

Examples:

- an important email incorrectly moved to spam,
- a healthy machine unnecessarily stopped for inspection,
- an innocent transaction incorrectly blocked as fraud.

## Code

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
```

---

# 8. Recall

Recall starts from reality: all actual positive cases.

It answers:

> Out of all actual positive cases, how many did the model successfully detect?

For a machine-failure system:

> Out of all machines that actually failed, how many failures did the model find?

```text
Recall = TP / (TP + FN)
```

Example:

Reality contains 100 failed machines:

- the model detects 80,
- the model misses 20.

```text
Recall = 80 / 100 = 80%
```

Interpretation:

> The model detected 80% of all real failures and missed 20%.

High Recall means fewer false negatives.

Recall is especially important when missing a positive case is dangerous.

Examples:

- missing a machine that is about to fail,
- missing a patient with cancer,
- missing a fraudulent transaction,
- missing a safety-critical defect.

## Precision versus Recall

| Metric | Starting point | Main question |
|---|---|---|
| Precision | Model's positive predictions | Can I trust the alarms? |
| Recall | Actual positive cases | Did I find the real cases? |

A useful memory rule is:

```text
Precision starts from the model.
Recall starts from reality.
```

## Code

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
```

---

# 9. F1 Score

Precision and Recall can move in opposite directions.

A model may have:

- high Precision but low Recall,
- or high Recall but low Precision.

F1 Score combines them into one number.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 uses the harmonic mean, which strongly punishes an extreme imbalance between Precision and Recall.

Examples:

| Precision | Recall | General result |
|---:|---:|---|
| 100% | 0% | F1 = 0% |
| 80% | 80% | F1 = 80% |
| 95% | 30% | F1 becomes much lower than 95% |

F1 answers:

> How well is the model balancing trustworthy positive predictions and detection of actual positives?

F1 is useful when:

- the target is imbalanced,
- both false positives and false negatives matter,
- and one combined score is needed.

## Code

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
```

---

# 10. Classification Report

A classification report prints several evaluation results together.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Typical output:

```text
              precision    recall  f1-score   support

           0       0.82      0.88      0.85       160
           1       0.76      0.67      0.71       102

    accuracy                       0.80       262
   macro avg       0.79      0.77      0.78       262
weighted avg       0.80      0.80      0.80       262
```

## Support

Support is the number of actual examples belonging to a class.

If class `1` has support 102, the test set contains 102 actual survivors.

Support is a count, not a performance metric.

## Macro average

Macro average calculates a simple average across classes.

Each class receives equal importance, regardless of how many examples it contains.

This is useful when minority-class performance matters.

## Weighted average

Weighted average considers the support of each class.

A class with more examples contributes more to the final average.

This can be useful, but in imbalanced data it may hide weak performance on a small class.

---

# 11. Worked example using one confusion matrix

Suppose the confusion matrix contains:

```text
TN = 150
FP = 20
FN = 25
TP = 67
```

Total records:

```text
150 + 20 + 25 + 67 = 262
```

## Accuracy

```text
(150 + 67) / 262 = 82.8%
```

## Precision

```text
67 / (67 + 20) = 77.0%
```

Interpretation:

> Of all predicted survivors, 77% actually survived.

## Recall

```text
67 / (67 + 25) = 72.8%
```

Interpretation:

> The model found about 72.8% of all actual survivors.

## F1 Score

```text
2 × 67 / (2 × 67 + 20 + 25) = 74.9%
```

This one confusion matrix provides the values used by all four metrics.

---

# 12. Decision threshold

A classifier often calculates a probability before producing a final class.

Example:

| Passenger | Probability of survival |
|---|---:|
| A | 0.91 |
| B | 0.63 |
| C | 0.48 |
| D | 0.12 |

With a threshold of `0.50`:

```text
Probability >= 0.50 → Predict survived
Probability < 0.50  → Predict did not survive
```

Changing the threshold changes the predictions.

## High threshold

A high threshold such as `0.80` is strict.

- fewer records are predicted positive,
- false positives may decrease,
- true positives may also decrease,
- Recall often decreases.

## Low threshold

A low threshold such as `0.30` is relaxed.

- more records are predicted positive,
- more actual positives may be detected,
- false positives may increase,
- Recall often increases.

The default threshold is not automatically the best threshold for every business problem.

---

# 13. ROC Curve

ROC stands for **Receiver Operating Characteristic**.

The ROC Curve evaluates the classifier across many decision thresholds.

For each threshold, it calculates:

- True Positive Rate,
- False Positive Rate.

## True Positive Rate

True Positive Rate is the same as Recall:

```text
TPR = TP / (TP + FN)
```

It measures how many actual positives were detected.

## False Positive Rate

```text
FPR = FP / (FP + TN)
```

It measures how many actual negatives were incorrectly predicted as positive.

## How the curve is created

```text
Choose a threshold
        ↓
Convert probabilities into class predictions
        ↓
Build a confusion matrix
        ↓
Calculate TPR and FPR
        ↓
Plot one point
        ↓
Repeat with another threshold
```

Scikit-Learn automatically tests many thresholds based on the model's probability scores.

## Code

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability,
)
```

Then plot:

```python
plt.figure()
plt.plot(fpr, tpr, label="Decision Tree")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

The diagonal line represents random guessing.

A better curve bends toward the upper-left corner, where Recall is high and False Positive Rate is low.

---

# 14. AUC

AUC means **Area Under the ROC Curve**.

It summarizes the ROC Curve with one number.

| AUC | General interpretation |
|---:|---|
| 0.50 | Similar to random ranking |
| 0.60–0.70 | Weak to moderate separation |
| 0.70–0.80 | Reasonable separation |
| 0.80–0.90 | Strong separation |
| 0.90–1.00 | Very strong separation |
| 1.00 | Perfect separation on the evaluated data |

A high AUC means the model generally ranks positive examples above negative examples.

AUC does not automatically select the final operating threshold. The threshold still depends on the cost of false positives and false negatives.

## Code

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_probability)
```

---

# 15. Choosing the correct metric

There is no universally best metric. The correct choice depends on the cost of mistakes.

| Real-world concern | Metric to focus on |
|---|---|
| Overall correctness with balanced classes | Accuracy |
| False alarms are expensive | Precision |
| Missing a positive case is dangerous | Recall |
| Both Precision and Recall matter | F1 Score |
| Need to inspect all error types | Confusion Matrix |
| Need performance across many thresholds | ROC Curve and AUC |

Examples:

## Predictive maintenance

Missing a dangerous failure can stop production or cause a safety problem. Recall may be more important than Precision.

## Spam filtering

Moving an important email into spam can be costly. Precision for the spam class may be important.

## Medical screening

Missing a sick patient can be dangerous. High Recall is often a priority during screening.

## Expensive manual inspection

If every positive prediction triggers a costly inspection, Precision becomes important.

The final threshold and evaluation metric should match the real decision the model supports.

---

# 16. Main lessons

1. A Decision Tree learns decision rules from training examples.
2. `predict()` gives a final class for each record.
3. `predict_proba()` gives probabilities for individual records.
4. The confusion matrix summarizes TP, TN, FP, and FN.
5. Accuracy measures overall correctness but can hide failure on a rare class.
6. Precision measures the reliability of positive predictions.
7. Recall measures how many actual positives were detected.
8. F1 Score balances Precision and Recall.
9. The classification report prints several metrics together.
10. ROC evaluates many possible decision thresholds.
11. AUC summarizes the model's ranking ability across thresholds.
12. The best metric depends on the practical cost of mistakes.

> Model evaluation is not about finding the largest number. It is about measuring the type of performance that matters for the real problem.

---

# Files in this folder

- `README.md` — theory and interpretation of the Decision Tree and evaluation metrics.
- `titanic_decision_tree_pipeline.py` — complete executable Titanic classification pipeline with section-by-section comments.
