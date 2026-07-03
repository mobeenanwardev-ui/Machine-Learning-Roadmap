# Missing Values

## What I understood

Missing values are not just empty cells. They can tell us something about how the data was collected.

A value may be missing because a sensor failed, a customer skipped a question, a measurement was not required, or the information did not exist at that time. Before filling anything, I should understand why it is missing.

## Common strategies

### Remove rows or columns

Deleting can be reasonable when only a very small number of rows are affected and the missingness is not important. Removing an entire column may make sense when most values are missing and the feature is not essential.

The danger is losing useful information or introducing bias. For example, if a certain customer group is more likely to leave a field empty, deleting those rows can make that group disappear from the dataset.

### Mean imputation

The mean is simple, but it is sensitive to extreme values. It can also create an artificial concentration around the average.

### Median imputation

The median is often safer for skewed numerical data or data containing outliers.

### Most frequent value

This can be used for categorical features, but it may over-represent the most common category.

### Missing indicator

Sometimes the fact that a value is missing is itself useful. In that case, I can add a separate yes/no column such as `income_was_missing`.

## Important leakage rule

The imputation value must be learned from the training set only.

For example, I calculate the training-set median and then use that same median for validation and test data. I should not calculate a new median from the test set.

## Main lesson

> There is no universally correct missing-value strategy. The decision depends on why the value is missing, how much data is affected, and how the chosen method changes the distribution.
