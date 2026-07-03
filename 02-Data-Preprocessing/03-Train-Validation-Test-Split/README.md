# Train, Validation, and Test Split

## What I understood

A model should not be trained and evaluated on exactly the same data. Otherwise, I only measure how well it remembers examples it has already seen.

The dataset is divided into three parts:

- **Training set:** used to learn the model parameters.
- **Validation set:** used to compare models and tune settings.
- **Test set:** kept separate until the final evaluation on unseen data.

## Why the split is necessary

Training performance shows how well the model fits known data. It does not prove that the same model will work on new records. A separate test set gives a more realistic estimate of generalization.

## Validation versus test data

The validation set helps me compare choices such as different models, different values of `k` in KNN, different numbers of PCA components, or different missing-value strategies.

Because validation results influence these choices, the test set should remain untouched until the end.

## Splitting must match the real problem

A random split is not always correct. For time-based data, I should normally train on older observations and test on newer ones. For grouped data, records from the same machine should stay in one split so that nearly identical observations do not appear in both training and testing.

## Main lesson

> The test set represents unseen data and should not influence decisions made during model development.
