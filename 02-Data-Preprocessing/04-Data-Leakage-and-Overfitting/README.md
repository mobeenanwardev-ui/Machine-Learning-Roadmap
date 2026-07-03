# Data Leakage and Overfitting

## Two different reasons a model can fail

A model can look successful during development and still fail in real use. Two common reasons are **data leakage** and **overfitting**.

| Problem | What happens |
|---|---|
| Data leakage | The model receives information it should not have |
| Overfitting | The model learns the training data too specifically |

These problems can both create misleading results, but they are not the same.

## Data leakage

Data leakage happens when information from outside the proper training process influences the model.

### Example: predicting subscription cancellation

Suppose the goal is to predict whether a customer will cancel next month.

| Feature | Available before cancellation? | Safe to use? |
|---|---|---|
| Number of logins | Yes | Yes |
| Support complaints | Yes | Yes |
| Cancellation date | No | No |
| Final refund amount | No | No |

The cancellation date may make prediction easy, but it only exists after cancellation. The model would be reading the answer instead of predicting it.

### Leakage during preprocessing

This order is wrong:

1. Fill missing values using the complete dataset.
2. Scale all records.
3. Apply PCA.
4. Split into training and test data.

The test set has already influenced the training process.

The safer order is:

1. Split the data.
2. Fit imputation, scaling, and PCA only on training data.
3. Apply the learned transformations to validation and test data.

| Preprocessing step | Learn from | Apply to |
|---|---|---|
| Median imputation | Training set | Train, validation, test |
| Standardization | Training set | Train, validation, test |
| PCA components | Training set | Train, validation, test |

## Overfitting

Overfitting happens when the model learns noise and special details from the training set.

| Model | Training accuracy | Validation accuracy | Interpretation |
|---|---:|---:|---|
| Model A | 99% | 72% | Probably overfitting |
| Model B | 91% | 88% | Better generalization |

Model A looks stronger if I only check training accuracy, but Model B is more useful on new data.

## Why overfitting happens

Common reasons include:

- a model that is too complex,
- too many features compared with the number of observations,
- noisy data,
- too little training data,
- or training for too long.

Possible responses include:

- collecting more representative data,
- simplifying the model,
- removing irrelevant features,
- using regularization,
- and checking performance with cross-validation.

## Main difference

| Question | Data leakage | Overfitting |
|---|---|---|
| Did forbidden information enter training? | Yes | Not necessarily |
| Did the model memorize training details? | Not necessarily | Yes |
| Can the score look unrealistically high? | Yes | Yes |
| Main prevention | Correct data pipeline | Better model control and validation |

## Main lesson

> A strong score is not automatically trustworthy. I must check both the data pipeline and the difference between training and validation performance.
