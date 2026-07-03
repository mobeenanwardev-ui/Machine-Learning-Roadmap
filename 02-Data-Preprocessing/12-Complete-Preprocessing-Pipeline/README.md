# Complete Data Preprocessing Pipeline

## What I understood

Data preprocessing is not a collection of random cleaning steps. It is an ordered process that protects the model from bad data and prevents information from leaking from the future into training.

A general workflow looks like this:

1. define the real problem,
2. understand the meaning and source of the data,
3. separate features and target,
4. create training, validation, and test sets,
5. inspect missing, incorrect, duplicated, and unusual values,
6. fit preprocessing steps on the training set,
7. transform validation and test data using the same fitted steps,
8. train the model,
9. evaluate it on unseen data,
10. compare the result with a simple baseline.

## Why the order matters

The split should happen before preprocessing methods learn anything from the data.

For example, if I standardize the full dataset before splitting, the mean and standard deviation from the future test records influence the training transformation. The same problem occurs with imputation, feature selection, oversampling, and PCA.

The correct idea is:

- **fit on training data,**
- **transform training, validation, and test data.**

## Not every step is always required

A useful pipeline depends on the dataset and model.

- Categorical data may require encoding.
- Missing values may require imputation.
- KNN and PCA usually require scaling.
- Decision trees usually do not need scaling.
- PCA is optional and should only be used when it gives a measurable benefit.
- Class balancing methods are relevant only when the target is imbalanced.

This is why I should understand the algorithm instead of blindly applying every preprocessing technique.

## Example workflow

For a machine-failure classification problem, I might:

1. define failure within the next 24 hours as the target,
2. remove identifiers that contain no predictive meaning,
3. split data by time so future records stay in the test set,
4. calculate missing-value replacements from training data,
5. encode machine type,
6. standardize sensor readings,
7. compare models with and without PCA,
8. train the classifier,
9. evaluate recall, precision, F1-score, and the confusion matrix.

## Using a software pipeline

In practical Python work, tools such as a scikit-learn `Pipeline` help keep preprocessing and modeling together. This reduces mistakes and ensures the same transformations are applied during training and prediction.

## Main lesson

> A reliable model depends not only on the algorithm, but also on a correct and repeatable preprocessing process.

## How I would explain it in an interview

I split the data early, fit all preprocessing only on the training set, and apply the learned transformations unchanged to validation and test data. I select preprocessing steps according to the feature types and model, then validate the complete pipeline on unseen data.