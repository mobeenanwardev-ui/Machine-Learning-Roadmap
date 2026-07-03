# Data Preprocessing

Data preprocessing is the work required to turn raw data into a form that a model can use reliably.

I learned that preprocessing is not one fixed checklist. The correct steps depend on the problem, the dataset, and the algorithm. For example, KNN is sensitive to feature scales, while decision trees usually are not. PCA may help with hundreds of related features, but it may also remove information that matters for prediction.

The general workflow covered in this section is:

1. understand the problem and the meaning of the data,
2. check whether the sample represents the real population,
3. split the data into training, validation, and test sets,
4. prevent data leakage,
5. handle missing and incorrect values,
6. engineer or encode useful features,
7. scale features when the algorithm needs it,
8. reduce dimensions only when it gives a measurable benefit,
9. train and evaluate the model on unseen data.

The main principle is simple:

> Preprocessing decisions must be learned from the training data and then applied unchanged to validation and test data.

This section contains separate explanations for each part of that workflow.