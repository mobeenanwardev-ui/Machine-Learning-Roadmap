# Complete Data Preprocessing Pipeline

## Why a pipeline is needed

Data preprocessing is not a collection of random cleaning steps. It is an ordered process that turns raw data into a reliable input for a model.

The order matters because validation and test data must not influence what the model learns.

## Example dataset

Suppose the goal is to predict whether a machine will fail within the next 24 hours.

| Temperature | Pressure | Machine type | Vibration | Failure in 24h |
|---:|---:|---|---:|---|
| 72 | 118 | A | 2.4 | No |
| missing | 121 | B | 5.8 | Yes |
| 81 | missing | A | 6.2 | Yes |
| 65 | 116 | C | 2.0 | No |

This small table already contains several preprocessing questions:

- What exactly is the prediction target?
- How should missing values be handled?
- How should machine type be converted into numbers?
- Do numerical features need scaling?
- Is the failure class rare?
- Would PCA help if there were hundreds of sensors?

## Complete workflow

| Step | Main question | Example action |
|---:|---|---|
| 1 | What problem am I solving? | Predict failure within 24 hours |
| 2 | What does the data mean? | Understand sensors and machine types |
| 3 | Which column is the target? | Separate `Failure in 24h` from inputs |
| 4 | How should data be split? | Train on older records and test on newer records |
| 5 | Are values missing or incorrect? | Fill training medians and remove impossible readings |
| 6 | Are there text categories? | One-hot encode machine type |
| 7 | Does the model need scaling? | Standardize sensor values for KNN or PCA |
| 8 | Are there too many features? | Compare performance with and without PCA |
| 9 | Is the target imbalanced? | Use suitable metrics and possibly class weights |
| 10 | Does the model generalize? | Evaluate on untouched test data |

## The most important rule

Every preprocessing method that learns values from data must be fitted on the training set only.

| Preprocessing step | What it learns |
|---|---|
| Median imputation | Median of each training feature |
| Standardization | Training mean and standard deviation |
| One-hot encoding | Categories seen during training |
| PCA | Principal directions from training data |
| Feature selection | Features chosen using training information |

The learned transformation is then applied unchanged to validation and test data.

## Wrong and correct order

### Wrong order

1. Fill missing values using all records.
2. Scale the complete dataset.
3. Apply PCA to all records.
4. Split the data.
5. Train and test the model.

The test set has already influenced preprocessing, so the final result may look better than reality.

### Correct order

1. Separate features and target.
2. Split into training, validation, and test data.
3. Fit preprocessing on training data.
4. Transform all three sets using the fitted preprocessing.
5. Train the model on the transformed training data.
6. Compare choices using validation data.
7. Evaluate the final version on test data.

## A worked example

Assume the training-set median temperature is 72 and the median pressure is 118.

### Step 1: fill missing values

| Temperature | Pressure | Machine type | Vibration |
|---:|---:|---|---:|
| 72 | 118 | A | 2.4 |
| 72 | 121 | B | 5.8 |
| 81 | 118 | A | 6.2 |
| 65 | 116 | C | 2.0 |

### Step 2: encode machine type

| type_A | type_B | type_C |
|---:|---:|---:|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 0 | 0 | 1 |

### Step 3: scale numerical features when needed

The original values may be converted into standardized values so that temperature, pressure, and vibration use comparable numerical ranges.

### Step 4: optional dimensionality reduction

If the dataset contains hundreds of sensor columns, PCA can be tested. It should be kept only when it provides a useful improvement in speed, stability, visualization, or model performance.

## Not every step is always required

| Situation | Likely preprocessing need |
|---|---|
| Decision tree with clean numerical data | Scaling may not be needed |
| KNN with features in different units | Scaling is important |
| Dataset with only five useful features | PCA may be unnecessary |
| Thousands of text categories | One-hot encoding may be too large |
| Balanced target | Resampling may not be needed |

A good pipeline is based on the problem and algorithm, not on blindly applying every available technique.

## Why a software pipeline helps

In practical Python work, a scikit-learn `Pipeline` can keep preprocessing and the model together.

Conceptually:

```text
Raw data
   ↓
Missing-value handling
   ↓
Categorical encoding
   ↓
Feature scaling
   ↓
Optional PCA
   ↓
Machine-learning model
```

This makes the process repeatable and reduces the chance of applying different transformations during training and prediction.

## Final checklist

Before trusting a model, I should check:

- Is the problem clearly defined?
- Does the sample represent the real environment?
- Was the split performed before learning preprocessing values?
- Were missing values handled for a logical reason?
- Were categories encoded without creating false order?
- Was scaling used only when needed?
- Was PCA validated instead of applied automatically?
- Were class imbalance and suitable metrics considered?
- Was the final model tested on untouched data?

## Main lesson

> A reliable model depends not only on the algorithm. It also depends on a correct, repeatable, and leakage-free preprocessing pipeline.
