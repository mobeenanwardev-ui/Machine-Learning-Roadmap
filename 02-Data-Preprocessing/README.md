# Data Preprocessing

## What preprocessing means

Raw data is rarely ready for a machine-learning model. It may contain missing values, text categories, incorrect records, different numerical scales, duplicated information, or a highly imbalanced target.

Data preprocessing is the process of converting that raw data into a reliable representation for learning.

A simple raw dataset may look like this:

| Temperature | Pressure | Machine type | Vibration | Failure |
|---:|---:|---|---:|---|
| 72 | 118 | A | 2.4 | No |
| missing | 121 | B | 5.8 | Yes |
| 81 | missing | A | 6.2 | Yes |

Before training a model, I need to understand the problem, split the data correctly, handle the missing values, encode the machine type, decide whether scaling is needed, and choose suitable evaluation methods.

## Why preprocessing is not one fixed checklist

The correct steps depend on the dataset and algorithm.

| Situation | Preprocessing decision |
|---|---|
| KNN with age and income | Scaling is important because KNN uses distance |
| Decision tree with the same features | Scaling is usually unnecessary |
| Five clear numerical features | PCA may not be useful |
| Five hundred related sensor features | PCA may be worth testing |
| City names | Categorical encoding is needed |
| Rare machine failures | Accuracy alone is not enough |

This means I should understand why a step is needed instead of applying every technique automatically.

## Topics covered

| Topic | Main idea |
|---|---|
| [Problem Understanding](./01-Problem-Understanding/) | Define the real question before choosing a model |
| [Sampling and Generalization](./02-Sampling-and-Generalization/) | Make sure the available data represents the real environment |
| [Train, Validation, and Test Split](./03-Train-Validation-Test-Split/) | Separate learning, model selection, and final evaluation |
| [Data Leakage and Overfitting](./04-Data-Leakage-and-Overfitting/) | Understand why a model can look strong but fail in real use |
| [Feature Engineering](./05-Feature-Engineering/) | Create a representation that makes useful patterns clearer |
| [Missing Values](./06-Missing-Values/) | Choose a strategy based on why data is missing |
| [Feature Scaling](./07-Feature-Scaling/) | Prevent numerical units from unfairly controlling some models |
| [Categorical Encoding](./08-Categorical-Encoding/) | Convert categories into numbers without creating false order |
| [Class Imbalance](./09-Class-Imbalance/) | Evaluate rare classes with suitable metrics |
| [Curse of Dimensionality](./10-Curse-of-Dimensionality/) | Understand the problems caused by too many features |
| [Principal Component Analysis](./11-Principal-Component-Analysis-PCA/) | Reduce dimensions by creating new combined components |
| [Complete Preprocessing Pipeline](./12-Complete-Preprocessing-Pipeline/) | Connect all steps in the correct order |

## The general workflow

```text
Understand the real problem
          ↓
Understand the data and its source
          ↓
Separate features and target
          ↓
Create training, validation, and test sets
          ↓
Fit preprocessing only on training data
          ↓
Transform validation and test data using the same rules
          ↓
Train and compare models
          ↓
Evaluate the final model on untouched test data
```

## The central rule

Any preprocessing step that learns something from the data must learn it from the training set only.

| Step | What is learned from training data |
|---|---|
| Missing-value imputation | Mean, median, or most frequent value |
| Scaling | Mean, standard deviation, minimum, or maximum |
| Encoding | Known categories |
| PCA | Principal directions |
| Feature selection | Which variables should be kept |

The same learned rules are then applied to validation, test, and future production data.

## Main lesson

> Preprocessing is not only about cleaning a table. It is about building a fair and repeatable path from raw data to a model that can work on unseen data.
