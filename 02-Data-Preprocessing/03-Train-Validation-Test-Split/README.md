# Train, Validation, and Test Split

## What I understood

A model should not be trained and judged on exactly the same data. If I do that, I only learn how well it remembers the examples it has already seen.

That is why the dataset is divided into separate parts.

- **Training set:** used to learn the model parameters.
- **Validation set:** used to compare models, tune settings, or choose preprocessing decisions.
- **Test set:** used once at the end to estimate how the final system performs on unseen data.

## Why the split is necessary

Imagine I give a student the exact exam questions before the exam. A high score would not prove that the student understands the subject. It may only prove that the student memorized those questions.

The same problem happens in machine learning. Training accuracy tells me how well the model fits known data. Test performance tells me whether it can generalize.

## Validation versus test data

The validation set helps during development. For example, I may compare:

- two different models,
- different values of `k` in KNN,
- different numbers of PCA components,
- or different missing-value strategies.

Because I repeatedly use validation results to make decisions, the validation set indirectly influences the model. The test set must therefore remain untouched until the final evaluation.

## Splitting must match the real problem

A random split is not always correct.

For time-series data, I should usually train on older data and test on newer data. Randomly mixing future and past records could give the model information that would not exist at prediction time.

For patient or machine data, records from the same patient or machine may need to stay in one split. Otherwise, the model may see almost identical examples in training and testing.

## Main lesson

> The test set represents the future. I should protect it from every decision made during model development.

## How I would explain it in an interview

I train the model on the training set, use validation data for model and preprocessing choices, and use the test set only for the final unbiased estimate. I also choose the splitting method according to the data structure, especially for time series, grouped observations, or imbalanced classes.