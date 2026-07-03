# Missing Values

## What a missing value really means

A missing value is not only an empty cell. It may tell us something about how the data was collected.

A value can be missing because:

- a sensor stopped working,
- a customer skipped a question,
- a measurement was not required,
- the information did not exist at that time,
- or a data-transfer problem occurred.

Before filling the value, I should first understand why it is missing.

## Simple example

| Customer | Age | Monthly income | Purchased |
|---|---:|---:|---|
| A | 24 | 2,200 | Yes |
| B | 31 | missing | No |
| C | missing | 3,100 | Yes |
| D | 45 | 4,600 | No |

The missing age and missing income may have different causes, so they do not necessarily need the same treatment.

## Common strategies

### 1. Remove rows

This may be reasonable when very few rows are affected.

| Situation | Decision |
|---|---|
| 3 missing rows out of 100,000 | Removing them may be acceptable |
| 4,000 missing rows out of 10,000 | Removing them may destroy too much information |

Removing rows can also create bias. For example, if low-income customers are more likely to leave the income field empty, deleting those rows may remove an important group.

### 2. Remove a column

This may make sense when most values are missing and the feature is not important.

| Feature | Missing percentage | Possible action |
|---|---:|---|
| Age | 2% | Keep and handle missing values |
| Optional comment | 92% | Possibly remove |

### 3. Mean imputation

The mean uses the average value.

For values `20, 22, 24, 26`, the mean is `23`.

The mean is simple, but it is sensitive to extreme values.

### 4. Median imputation

For values `20, 22, 24, 100`, the mean is `41.5`, while the median is `23`.

| Method | Replacement value |
|---|---:|
| Mean | 41.5 |
| Median | 23 |

The median is often more suitable for skewed data or data containing outliers.

### 5. Most frequent category

For a categorical feature such as city, the most common city can be used. This is simple, but it may make the largest category even more dominant.

### 6. Add a missing indicator

Sometimes the fact that a value is missing is useful.

| Income | income_was_missing |
|---:|---:|
| 2,200 | 0 |
| filled with median | 1 |

This allows the model to use both the replacement value and the fact that the original information was absent.

## Important rule: fit on training data only

Suppose the training-set median income is 3,000 and the test-set median is 4,500. I should use the training median for both sets.

| Dataset | Median used for imputation |
|---|---:|
| Training | 3,000 |
| Validation | 3,000 |
| Test | 3,000 |

Using the test-set median would allow test information to influence preprocessing.

## Questions to ask before choosing a method

1. How much data is missing?
2. Is the missingness random or connected to a group?
3. Is the feature important?
4. Does the feature contain outliers?
5. Will removing data change the population?
6. Could a missing indicator contain useful information?

## Main lesson

> There is no single best strategy for missing values. The correct choice depends on the cause, amount, distribution, and meaning of the missing data.
