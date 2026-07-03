# Principal Component Analysis (PCA)

## Why PCA exists

PCA is a dimensionality-reduction technique. It is useful when a dataset has many numerical features and several of them contain overlapping information.

PCA does not simply choose a few original columns. It creates new features called **principal components** by combining the original features.

## A simple example

Suppose we record height, weight, and shoe size:

| Person | Height (cm) | Weight (kg) | Shoe size |
|---|---:|---:|---:|
| A | 165 | 60 | 39 |
| B | 172 | 70 | 42 |
| C | 180 | 82 | 44 |
| D | 188 | 91 | 46 |

These features often increase together. They are not identical, but they contain related information about body size.

PCA may combine them into a new component:

`PC1 = 0.58 × height + 0.56 × weight + 0.59 × shoe size`

The exact weights are learned from the data. PC1 could be interpreted loosely as an overall body-size direction, although PCA itself does not understand that meaning.

## What the computer actually sees

The computer does not understand height, weight, or shoe size. It only sees numerical columns:

| X1 | X2 | X3 |
|---:|---:|---:|
| 165 | 60 | 39 |
| 172 | 70 | 42 |
| 180 | 82 | 44 |

PCA asks:

- Which columns change together?
- In which direction is the data spread the most?
- Can that variation be represented with fewer new columns?

It does not need the column names to perform the calculation.

## Principal components

The components are ordered by the amount of variance they explain.

| Component | Meaning |
|---|---|
| PC1 | Direction containing the most variation |
| PC2 | Largest remaining variation, independent of PC1 |
| PC3 | Next remaining variation |

Suppose the explained variance is:

| Component | Explained variance | Cumulative variance |
|---|---:|---:|
| PC1 | 72% | 72% |
| PC2 | 20% | 92% |
| PC3 | 8% | 100% |

Keeping PC1 and PC2 would reduce three features to two while preserving 92% of the variation.

## PCA creates new features

Before PCA:

| Temperature | Pressure | Vibration | Current |
|---:|---:|---:|---:|
| Original feature | Original feature | Original feature | Original feature |

After PCA:

| PC1 | PC2 |
|---:|---:|
| Combination of all original features | Another combination of all original features |

This is different from feature selection.

| Method | What it does |
|---|---|
| Feature selection | Keeps some original columns |
| PCA | Creates new combined columns |

## Why scaling usually comes first

Suppose one feature is measured in euros and another between 0 and 1:

| Feature | Typical range |
|---|---:|
| Annual revenue | 100,000 to 5,000,000 |
| Defect rate | 0.01 to 0.20 |

Without scaling, revenue may dominate the variance only because its numbers are much larger. Standardization gives features a comparable scale before PCA.

## Important limitation: variance is not the same as usefulness

PCA is unsupervised. It does not look at the target variable.

Suppose a rare sensor changes only slightly, but that small change is strongly connected to failure. Because the feature has low variance, PCA may reduce part of its information.

| Feature | Variance | Importance for failure prediction |
|---|---:|---|
| Sensor A | High | Low |
| Sensor B | Low | High |

PCA may favour Sensor A because it has higher variance, even though Sensor B is more useful for the target.

That is why PCA must be tested rather than trusted automatically.

## Example with 500 sensors

If 500 sensors contain many related signals, PCA may reduce them to a smaller number of components.

| Before PCA | After PCA |
|---|---|
| 500 sensor columns | 20 principal components |
| High memory and computation | Faster training |
| Many correlated features | Components are uncorrelated |
| Original features are easy to name | Components are harder to interpret |

The model should be compared with and without PCA to check whether accuracy, speed, or stability improves.

## Correct data flow

1. Split the data.
2. Fit the scaler on training features.
3. Transform training, validation, and test features with that scaler.
4. Fit PCA on the scaled training features.
5. Transform validation and test features with the same PCA object.

PCA should not be fitted on the complete dataset before splitting because that would allow test information to influence the components.

## When PCA may help

- many numerical features,
- strongly related features,
- slow model training,
- noisy or redundant data,
- or two-dimensional visualization.

## When PCA may not help

- only a few features,
- original features must remain easy to explain,
- low-variance features contain important target information,
- or the model already handles the original dimensions well.

## Main lesson

> PCA keeps directions with high variation, not necessarily the features that are most important for the prediction target.

It is a useful tool, but the final decision should come from validation results and domain understanding.
