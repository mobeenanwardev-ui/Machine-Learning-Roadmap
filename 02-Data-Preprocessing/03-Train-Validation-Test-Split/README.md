# Train, Validation, and Test Split

## Why the dataset is divided

A model should not be trained and judged on exactly the same records. Otherwise, I only measure how well it remembers data it has already seen.

| Dataset part | Main purpose |
|---|---|
| Training set | Teach the model |
| Validation set | Compare models and settings |
| Test set | Final check on unseen data |

A split such as 70% training, 15% validation, and 15% testing is common, but the correct percentages depend on the amount and type of data.

## A simple learning example

The training set is similar to practice questions. The validation set is like a mock test used to identify which preparation method works best. The test set is the final set of new questions.

If the final check contains exactly the same examples used during learning, a strong result may only show memorization.

## Choosing between models

Suppose I test three values of `k` for KNN:

| KNN setting | Training accuracy | Validation accuracy |
|---|---:|---:|
| k = 3 | 98% | 84% |
| k = 5 | 94% | 88% |
| k = 9 | 90% | 86% |

The model with `k = 3` fits the training data best, but `k = 5` works better on validation data. Therefore, I would select `k = 5` and then evaluate it once on the test set.

Validation data influences decisions such as:

- which algorithm to use,
- which settings to choose,
- how many PCA components to keep,
- which features to include,
- and which missing-value strategy works best.

That is why the test set should remain untouched until the final stage.

## Random splitting is not always correct

### Time-based data

For electricity-demand prediction, the correct order may be:

| Time period | Use |
|---|---|
| January to September | Training |
| October | Validation |
| November to December | Testing |

Randomly mixing future and past dates could make the result unrealistic.

### Grouped data

If one machine produces many similar sensor records, those records should normally stay in the same split.

| Machine | Split |
|---|---|
| Machine A | Training |
| Machine B | Training |
| Machine C | Validation |
| Machine D | Testing |

Otherwise, the model may see almost identical records from the same machine during training and testing.

### Imbalanced data

For rare failure cases, stratified splitting can keep a similar class ratio in every part:

| Split | Normal | Failure |
|---|---:|---:|
| Full dataset | 95% | 5% |
| Training | 95% | 5% |
| Validation | 95% | 5% |
| Testing | 95% | 5% |

## Main lesson

> Training data teaches the model, validation data helps me make choices, and test data provides the final estimate on unseen data.
