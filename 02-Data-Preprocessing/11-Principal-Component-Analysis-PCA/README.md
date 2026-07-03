# Principal Component Analysis (PCA)

## What I understood

PCA is a dimensionality-reduction technique. It does not simply choose a few original columns. It creates new features called principal components by combining the original numerical features.

The computer does not understand that a column represents height, weight, pressure, or shoe size. It only sees numbers. PCA looks at how those numerical columns vary together.

If two or more features often increase and decrease together, they may contain overlapping information. PCA tries to represent that shared variation with fewer new components.

## How PCA decides what to keep

PCA searches for directions in the data that contain the greatest variance.

- **PC1** is the direction containing the largest amount of variation.
- **PC2** captures the largest remaining variation while being independent of PC1.
- Further components continue in the same way.

Each principal component is a weighted combination of the original features.

For example, a component could look conceptually like:

`PC1 = 0.6 × height + 0.5 × weight + 0.4 × shoe_size`

The exact weights are calculated from the data.

## What PCA does not know

PCA is unsupervised. It does not look at the target variable while finding components.

This creates an important limitation: the direction with the most variance is not always the direction that is most useful for prediction.

A low-variance feature might be highly important for detecting a rare failure. PCA could reduce or remove part of that information because it only focuses on variance, not business importance or target relevance.

## Why scaling usually comes first

If one feature is measured in very large units, it can dominate the variance. For this reason, numerical features are commonly standardized before PCA when their scales are different.

The scaler and PCA must both be fitted on training data only. Validation and test data are transformed using the same fitted objects.

## When PCA may help

PCA may be useful when:

- there are many numerical features,
- several features are strongly related,
- training is slow,
- noise or redundancy is present,
- or the data needs to be reduced to two or three dimensions for visualization.

## Limitations

PCA can make interpretation harder because the new components are mixtures of original features. It can also reduce model performance if important predictive information is lost.

Therefore, I should not use PCA automatically. I should compare model performance with and without PCA and inspect the explained variance.

## Main lesson

> PCA preserves directions of high variation, not necessarily the features that are most important for the target.

## How I would explain it in an interview

PCA transforms correlated numerical features into a smaller set of uncorrelated principal components. The components are ordered by explained variance. I usually standardize features first, fit PCA only on training data, and validate whether the reduced representation actually improves the final model.