# Curse of Dimensionality

## What I understood

The curse of dimensionality describes the problems that appear when a dataset contains a very large number of features.

More features may sound better, but they do not always add more useful information. In a high-dimensional space, data points become sparse and it becomes harder to find reliable patterns.

This is especially important for distance-based methods such as KNN. When there are many dimensions, the difference between near and far points can become less meaningful.

## Why it matters

Too many features can lead to:

- more computation,
- higher memory use,
- greater risk of overfitting,
- noisy or irrelevant inputs,
- and lower interpretability.

For example, 500 sensors may not provide 500 independent pieces of information. Several sensors may measure almost the same physical behaviour.

## Possible responses

Possible solutions include:

- removing constant or duplicate features,
- selecting variables using domain knowledge,
- checking redundancy and correlation,
- using feature-selection methods,
- collecting more observations,
- or applying dimensionality reduction such as PCA.

## Main lesson

> More features do not automatically mean more useful information. The important question is how much independent and relevant information they contain.
