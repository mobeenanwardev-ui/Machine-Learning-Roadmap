# Feature Scaling

## What I understood

Feature scaling changes numerical features so that their sizes are comparable.

This matters because some algorithms use distance or gradient calculations. If one feature has values in thousands and another has values between 0 and 1, the large-scale feature can dominate even when it is not more important.

For example:

- age: 18 to 70,
- annual income: 20,000 to 120,000.

In KNN, the income difference may dominate the distance calculation simply because the numbers are larger.

## Standardization

Standardization usually transforms a feature so that it has a mean close to 0 and a standard deviation close to 1.

It is useful when features have different units and for methods such as KNN, SVM, logistic regression, neural networks, and PCA.

## Normalization

Min-max normalization usually moves values into a fixed range such as 0 to 1.

It can be useful when a bounded range is preferred, but it is sensitive to extreme values because the minimum and maximum determine the transformation.

## When scaling is less important

Tree-based models such as decision trees and random forests usually do not need scaling because they split features using thresholds. The order of values matters more than the numerical distance between them.

This taught me that preprocessing should depend on the model. Scaling everything without understanding the algorithm is not a good habit.

## Leakage rule

The scaler must be fitted on the training data only. The same learned mean, standard deviation, minimum, or maximum is then used to transform validation and test data.

## Main lesson

> Scaling does not make a feature more important. It prevents the unit or numerical range from deciding its influence in algorithms that are sensitive to scale.

## How I would explain it in an interview

I scale features when the algorithm depends on distances, gradients, or variance. I fit the scaler only on training data and apply the same transformation to unseen data. For tree-based models, scaling is usually unnecessary.