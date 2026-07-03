# Data Leakage and Overfitting

## What I understood

Data leakage happens when information that should not be available during training accidentally enters the model-building process. It can make the results look excellent even though the model will fail in real use.

Overfitting is different. It happens when the model learns the training data too closely, including noise and special cases, instead of learning a pattern that generalizes.

## A leakage example

Suppose I want to predict whether a customer will cancel a subscription. My dataset contains a column called `cancellation_date`.

That column gives away the answer. It may produce very high accuracy, but it would not be available before the customer cancels. The model is not predicting the future; it is indirectly reading the result.

Leakage can also happen during preprocessing. For example, this is wrong:

1. calculate the mean using the full dataset,
2. fill missing values,
3. scale all records,
4. apply PCA,
5. and only then split into train and test data.

The test data has already influenced the mean, scale, and PCA directions.

The correct order is:

1. split the data,
2. fit preprocessing steps only on the training set,
3. apply the learned transformations to validation and test data.

## Overfitting example

A very flexible model may memorize unusual details from the training data. It can achieve 99% training accuracy but much lower test accuracy.

Common signs include:

- a large gap between training and validation performance,
- validation performance becoming worse while training improves,
- or a model that changes strongly when the training sample changes.

Possible responses include collecting more representative data, simplifying the model, applying regularization, reducing unnecessary features, or using cross-validation.

## Main difference

- **Leakage:** the model receives information it should not have.
- **Overfitting:** the model learns the available training data too specifically.

Both can create misleading performance, but the causes are different.

## Main lesson

> A strong score is not automatically trustworthy. I must check how the data was created, split, and transformed before believing the result.

## How I would explain it in an interview

I prevent leakage by separating the test data early and fitting all preprocessing only on training data, preferably inside a pipeline. I detect overfitting by comparing training and validation performance and checking whether the model generalizes across different splits.