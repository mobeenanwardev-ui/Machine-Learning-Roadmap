# Class Imbalance

## What I understood

Class imbalance appears when one target class has many more examples than another.

For example, suppose a dataset contains:

- 9,661 normal cases,
- 339 failure cases.

A model that predicts "normal" for every record would be correct most of the time. Its accuracy would look high, but it would detect no failures.

This is why accuracy alone can be misleading.

## What should be checked

For an imbalanced classification problem, I should also look at:

- **Recall:** how many real positive cases were found,
- **Precision:** how many predicted positive cases were actually positive,
- **F1-score:** balance between precision and recall,
- **Confusion matrix:** counts of correct and incorrect predictions,
- **ROC-AUC or PR-AUC:** performance across different thresholds.

The right metric depends on the real cost of mistakes.

In predictive maintenance, missing a real machine failure may be more expensive than sending an unnecessary inspection warning. In that case, recall for the failure class may be especially important.

## Possible approaches

Possible responses include:

- collecting more examples of the minority class,
- using class weights,
- oversampling the minority class,
- undersampling the majority class,
- adjusting the prediction threshold,
- and evaluating with suitable metrics.

These methods should be applied only to the training data. The validation and test sets should keep a realistic class distribution.

## Main lesson

> A high accuracy score does not prove that an imbalanced classifier is useful. I must check whether it can recognize the rare class that actually matters.

## How I would explain it in an interview

I first inspect the class distribution and define which mistakes are most costly. Then I use metrics such as recall, precision, F1-score, and the confusion matrix rather than relying only on accuracy. Any resampling is performed only on the training set.