# Feature Scaling

## Why scaling is needed

Feature scaling changes numerical features so that their sizes are comparable.

This matters for algorithms that calculate distances, gradients, or variance. A feature with large numbers can dominate another feature only because of its unit, not because it is more important.

## KNN example

Suppose KNN compares two customers using age and annual income.

| Customer | Age | Annual income |
|---|---:|---:|
| A | 25 | 30,000 |
| B | 35 | 31,000 |

The differences are:

| Feature | Difference |
|---|---:|
| Age | 10 |
| Income | 1,000 |

Income dominates the distance calculation because its numerical range is much larger. The algorithm may almost ignore age.

After scaling, the values may look like this:

| Customer | Scaled age | Scaled income |
|---|---:|---:|
| A | -0.8 | -0.4 |
| B | 0.2 | -0.2 |

Now both features can contribute more fairly.

## Standardization

Standardization transforms a feature using its mean and standard deviation:

`standardized value = (value - mean) / standard deviation`

Suppose exam scores are:

| Student | Score |
|---|---:|
| A | 40 |
| B | 50 |
| C | 60 |

If the mean is 50 and the standard deviation is 10:

| Score | Standardized value |
|---:|---:|
| 40 | -1 |
| 50 | 0 |
| 60 | 1 |

A value of `0` means the original value is at the mean. A positive value is above the mean, and a negative value is below it.

## Min-max normalization

Min-max normalization commonly moves values into the range 0 to 1:

`normalized value = (value - minimum) / (maximum - minimum)`

For values 20, 40, and 60:

| Original value | Normalized value |
|---:|---:|
| 20 | 0.0 |
| 40 | 0.5 |
| 60 | 1.0 |

This is useful when a fixed range is preferred. However, an extreme maximum or minimum can strongly affect all values.

## Standardization versus normalization

| Method | Typical result | Main consideration |
|---|---|---|
| Standardization | Mean near 0, standard deviation near 1 | Common for many ML algorithms |
| Min-max normalization | Usually values between 0 and 1 | Sensitive to extreme values |

The best choice depends on the data and model.

## Which models usually need scaling?

| Model | Is scaling usually important? | Reason |
|---|---|---|
| KNN | Yes | Uses distance |
| K-Means | Yes | Uses distance to centroids |
| SVM | Yes | Sensitive to feature magnitudes |
| Logistic regression | Usually yes | Helps optimization and regularization |
| Neural network | Usually yes | Helps stable training |
| PCA | Usually yes | Large-scale features can dominate variance |
| Decision tree | Usually no | Uses threshold splits |
| Random forest | Usually no | Built from decision trees |

This shows why preprocessing should depend on the algorithm instead of being applied blindly.

## Scaling does not change importance

Scaling does not mean that every feature becomes equally useful. It only removes the unfair effect of different units.

For example, converting income from euros into thousands of euros changes the number, but it does not change the information contained in the feature.

## Fit the scaler only on training data

Suppose the training mean is 50 and the test mean is 70. The scaler must still use the training mean for the test data.

| Dataset | Mean used for transformation |
|---|---:|
| Training | 50 |
| Validation | 50 |
| Test | 50 |

This prevents test information from entering the training process.

## Main lesson

> Scaling prevents the unit or numerical range of a feature from unfairly controlling algorithms that are sensitive to distance, gradients, or variance.
