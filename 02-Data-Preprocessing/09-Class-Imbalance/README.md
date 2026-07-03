# Class Imbalance

## What class imbalance means

Class imbalance appears when one target class has many more examples than another.

Suppose a machine-failure dataset contains:

| Class | Number of records | Percentage |
|---|---:|---:|
| Normal | 9,661 | 96.61% |
| Failure | 339 | 3.39% |

A model could predict `Normal` for every record and still achieve 96.61% accuracy.

| Prediction strategy | Accuracy | Failures detected |
|---|---:|---:|
| Always predict Normal | 96.61% | 0 |

The accuracy looks excellent, but the model is useless because it misses every failure.

## Why accuracy can be misleading

For imbalanced data, I need to check more than one metric.

| Metric | Question it answers |
|---|---|
| Accuracy | How many total predictions were correct? |
| Recall | How many real positive cases were found? |
| Precision | How many predicted positive cases were actually positive? |
| F1-score | How well are precision and recall balanced? |
| Confusion matrix | What types of correct and incorrect predictions occurred? |

## Confusion-matrix example

Suppose the test set contains 1,000 machines:

| Actual class | Predicted normal | Predicted failure |
|---|---:|---:|
| Normal | 930 | 20 |
| Failure | 20 | 30 |

From this table:

- 30 real failures were detected.
- 20 real failures were missed.
- 20 healthy machines received a false warning.

The failure recall is:

`30 / (30 + 20) = 60%`

This tells me that the model finds 60% of the real failures.

## The important metric depends on the problem

| Problem | More costly mistake | Metric that may matter more |
|---|---|---|
| Machine failure prediction | Missing a real failure | Recall |
| Spam detection | Marking an important email as spam | Precision for spam predictions |
| Medical screening | Missing a sick patient | Recall |
| Expensive manual inspection | Too many false alarms | Precision |

There is no single best metric for every problem. The real cost of each mistake decides what matters.

## Possible approaches

### Collect more minority examples

The best improvement may be collecting more real failure cases, although this is not always possible.

### Class weights

The model can be told that mistakes on the rare class should have a larger cost.

### Oversampling

The minority class is repeated or synthetically increased in the training data.

### Undersampling

Some majority-class records are removed.

### Change the decision threshold

A classifier may normally predict failure when probability is above 0.5. Lowering the threshold can detect more failures, but it may also create more false alarms.

| Threshold | Failure recall | Precision |
|---:|---:|---:|
| 0.70 | 40% | 90% |
| 0.50 | 65% | 75% |
| 0.30 | 85% | 52% |

This shows the trade-off: detecting more failures can reduce precision.

## Important rule for resampling

Oversampling or undersampling should be applied only to the training set.

| Dataset part | Keep realistic distribution? | Apply resampling? |
|---|---|---|
| Training | Not necessarily | Yes, when useful |
| Validation | Yes | No |
| Test | Yes | No |

Validation and test data should represent the real environment.

## Main lesson

> A high accuracy score does not prove that an imbalanced classifier is useful. I must check whether it can recognize the rare class that actually matters.
