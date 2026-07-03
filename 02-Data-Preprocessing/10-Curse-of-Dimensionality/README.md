# Curse of Dimensionality

## What it means

The curse of dimensionality describes the problems that appear when a dataset contains a very large number of features.

More features may sound better, but the data space grows very quickly.

| Dimensions | Regions when each dimension is divided into 10 parts |
|---:|---:|
| 1 | 10 |
| 2 | 100 |
| 3 | 1,000 |
| 10 | 10,000,000,000 |

The same number of observations becomes spread across a much larger space. This can make patterns and nearby observations harder to identify.

## Why this affects distance-based models

Suppose customer similarity is based on two useful features:

| Customer | Age | Monthly spending |
|---|---:|---:|
| A | 25 | 300 |
| B | 27 | 320 |
| C | 55 | 900 |

A and B are clearly similar. If hundreds of random or irrelevant columns are added, those extra differences may hide the useful similarity.

This is especially important for KNN and K-Means because they depend on distances.

## Example with 500 sensors

A factory may collect 500 sensor columns, but that does not always mean 500 independent pieces of information.

| Sensor situation | Effect |
|---|---|
| Twenty sensors measure almost the same heat pattern | Redundant information |
| A sensor always produces one value | No useful variation |
| A damaged sensor produces random readings | Noise |
| Several sensors are strongly related | Repeated information |

Keeping every column may add complexity without adding much knowledge.

## Common problems

| Problem | Why it happens |
|---|---|
| More computation | More values must be processed |
| Overfitting | The model can fit noise in many columns |
| Sparse data | Observations are spread across a large space |
| Weak distance measurement | Near and far points become less different |
| Low interpretability | Too many features are difficult to explain |

## Possible solutions

- remove constant and duplicate columns,
- remove irrelevant identifiers,
- use domain knowledge to keep useful variables,
- apply feature-selection methods,
- collect more observations,
- or use dimensionality reduction such as PCA.

## Feature selection versus PCA

| Method | Result |
|---|---|
| Feature selection | Keeps some original columns |
| PCA | Creates new combined components |

Feature selection is usually easier to explain. PCA may reduce redundancy more strongly, but its components are mixtures of the original features.

## Main lesson

> More features do not automatically mean more useful information. The important question is how much independent and relevant information those features contain.
