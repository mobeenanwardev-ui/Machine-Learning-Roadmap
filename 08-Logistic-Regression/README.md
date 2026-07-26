# Logistic Regression

This section is a **concept-first, exam-revision guide** to Logistic Regression.

The goal is not to memorize formulas blindly. The goal is to understand what the model is doing, why the sigmoid exists, why Binary Cross-Entropy is used, how Gradient Descent trains the model, how multiclass extensions work, and where Logistic Regression reaches its limits.

The recommended revision order is:

```text
NORMAL IDEA
    ↓
WHY IT EXISTS
    ↓
MACHINE-LEARNING MEANING
    ↓
FORMULA
    ↓
INTERPRETATION
    ↓
EXAM MEMORY RULE
```

---

# 1. Why do we need Logistic Regression?

Linear Regression predicts a **continuous numerical value**.

Examples:

```text
House price → 320000
Temperature → 24.7
Salary      → 45000
```

But many machine-learning problems ask for a **class** instead:

```text
Email       → Spam / Not Spam
Patient     → Disease / No Disease
Transaction → Fraud / Not Fraud
Customer    → Churn / No Churn
```

For binary classification, the target is often encoded as:

```text
Class 0
Class 1
```

Logistic Regression is designed to estimate the probability of one of those classes.

A useful memory rule:

```text
Linear Regression   → number
Logistic Regression → probability → class
```

---

# 2. Logistic Regression starts like Linear Regression

The first step is still a weighted sum of the input features:

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

or compactly:

```text
z = w^T x + b
```

where:

```text
x = input features
w = weights
b = bias / intercept
z = raw linear score
```

Example:

```text
Features:
Age
Blood pressure
Cholesterol

        ↓

z = w1*Age + w2*BloodPressure + w3*Cholesterol + b
```

At this point the model has only produced a **raw score**.

That score can be any real number:

```text
-15
-2.3
0
4.8
100
```

The problem is that those values are not directly probabilities.

For classification, we would prefer something between:

```text
0 and 1
```

That is why Logistic Regression needs the **sigmoid function**.

---

# 3. Sigmoid Function

The sigmoid function is:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

Do not begin by memorizing the formula.

Its job is much easier to remember:

> Take any real-valued score and squeeze it into a value between 0 and 1.

Examples:

```text
z = -10 → sigmoid(z) ≈ 0
z = -2  → sigmoid(z) ≈ 0.12
z = 0   → sigmoid(z) = 0.50
z = 2   → sigmoid(z) ≈ 0.88
z = 10  → sigmoid(z) ≈ 1
```

Conceptually:

```text
raw score z
     ↓
sigmoid
     ↓
probability-like value between 0 and 1
```

So Logistic Regression becomes:

```text
features
   ↓
weighted sum + bias
   ↓
z = w^T x + b
   ↓
sigmoid
   ↓
probability p
```

---

# 4. What does the sigmoid output mean?

Suppose:

```text
Class 1 = Spam
Class 0 = Not Spam
```

and the model produces:

```text
p = 0.87
```

We interpret that as:

> The model estimates a high probability for Class 1.

So:

```text
p close to 1 → model strongly favors Class 1
p close to 0 → model strongly favors Class 0
```

Important:

The probability is a **model estimate**. It is not a guarantee that the event will happen.

---

# 5. From probability to class: the threshold

The sigmoid gives us a probability, but many applications finally need a class label.

A common threshold is:

```text
0.5
```

Then:

```text
p >= 0.5 → Class 1
p <  0.5 → Class 0
```

Example:

```text
0.92 → Class 1
0.73 → Class 1
0.51 → Class 1
0.49 → Class 0
0.20 → Class 0
0.03 → Class 0
```

The threshold is **not always fixed at 0.5**.

For example, in a high-risk medical screening system, we may choose a lower threshold because missing a true positive could be very costly.

Memory rule:

```text
Sigmoid  → probability
Threshold → class decision
```

---

# 6. Linear Regression vs Logistic Regression

## Linear Regression

```text
features
   ↓
w^T x + b
   ↓
continuous output
```

Example:

```text
Predicted house price = 300000
```

## Logistic Regression

```text
features
   ↓
w^T x + b
   ↓
sigmoid
   ↓
probability
   ↓
threshold
   ↓
class 0 / class 1
```

The weighted-sum foundation is the same.

The important difference is the sigmoid and the classification interpretation.

---

# 7. Connection to an artificial neuron

An artificial neuron performs:

```text
inputs
   ↓
weights + bias
   ↓
weighted sum z
   ↓
activation function
   ↓
output
```

Logistic Regression performs:

```text
inputs
   ↓
weights + bias
   ↓
z
   ↓
sigmoid activation
   ↓
probability
```

This is why Logistic Regression is an important bridge toward Neural Networks.

A useful conceptual connection is:

```text
Linear Regression
→ weighted sum + identity output

Logistic Regression
→ weighted sum + sigmoid

Neural Network
→ many weighted-sum + activation blocks connected together
```

---

# 8. How do we measure Logistic Regression error?

The model predicts a **probability**, so we need a loss function that evaluates probability predictions.

For binary classification, the standard choice is:

# Binary Cross-Entropy (BCE)

Also called **log loss**.

The formula for one example is:

```text
L = -[y*log(p) + (1-y)*log(1-p)]
```

where:

```text
y = actual class, either 0 or 1
p = predicted probability for Class 1
```

Again, understand the behavior before memorizing the formula.

---

# 9. BCE when the true class is 1

Suppose:

```text
Actual y = 1
```

Then BCE effectively becomes:

```text
L = -log(p)
```

So we want:

```text
p → 1
```

Examples:

```text
Actual = 1

p = 0.99 → tiny loss
p = 0.80 → small loss
p = 0.50 → larger loss
p = 0.10 → very large loss
p = 0.01 → extremely large loss
```

The more confidently wrong the model is, the more heavily BCE punishes it.

---

# 10. BCE when the true class is 0

Suppose:

```text
Actual y = 0
```

Then BCE effectively becomes:

```text
L = -log(1-p)
```

Now we want:

```text
p → 0
```

Examples:

```text
Actual = 0

p = 0.01 → tiny loss
p = 0.20 → small loss
p = 0.50 → larger loss
p = 0.90 → very large loss
p = 0.99 → extremely large loss
```

---

# 11. Why does confidence matter?

Suppose the actual class is:

```text
1
```

Prediction A:

```text
p = 0.51
```

Prediction B:

```text
p = 0.99
```

With a 0.5 threshold, both produce Class 1.

But they are not equally good probability predictions.

```text
0.51 → barely confident
0.99 → strongly confident
```

Binary Cross-Entropy captures this difference.

That is why BCE is more informative than simply asking whether the final class label was correct.

---

# 12. Why not simply use MSE?

Mean Squared Error is naturally associated with continuous regression problems.

Logistic Regression predicts probabilities for binary outcomes.

Binary Cross-Entropy is designed for that probability setting and gives particularly strong penalties for confidently wrong predictions.

Memory rule:

```text
Linear Regression   → MSE
Logistic Regression → Binary Cross-Entropy
```

This is a useful exam simplification.

---

# 13. Loss vs Cost

For one training example:

```text
Loss = error for that example
```

For many examples:

```text
Cost = average loss over the training set or batch
```

Conceptually:

```text
example 1 → loss
example 2 → loss
example 3 → loss
...

average them
     ↓
Cost J(w,b)
```

Training tries to find weights and bias that minimize this cost.

---

# 14. How Logistic Regression learns

The training process is:

```text
Current weights + bias
        ↓
Calculate z = w^T x + b
        ↓
Apply sigmoid
        ↓
Get probability p
        ↓
Calculate BCE
        ↓
Calculate gradients
        ↓
Gradient Descent / optimizer
        ↓
Update weights and bias
        ↓
Repeat
```

This is almost the same optimization idea used in Linear Regression.

The model and loss changed, but Gradient Descent still updates the parameters.

---

# 15. Gradient Descent in Logistic Regression

The generic update rule remains:

```text
new parameter = old parameter - learning_rate * gradient
```

For a weight:

```text
w = w - alpha * dJ/dw
```

For the bias:

```text
b = b - alpha * dJ/db
```

Important:

> Gradient Descent does NOT directly change the predicted probability.

It changes the **weights and bias**.

That then changes the whole chain:

```text
weights / bias change
        ↓
z changes
        ↓
sigmoid output changes
        ↓
probability changes
        ↓
BCE changes
```

---

# 16. Human memory rule for Gradient Descent

The easiest way to remember Gradient Descent is the hill/valley analogy.

```text
Cost
↑
|       \        /
|        \      /
|         \____/
|           ↑
|        minimum
+----------------→ parameter
```

Think of standing on a hill in the dark.

You cannot see the whole landscape. You can only feel the local slope.

```text
Cost
→ tells me how bad I am right now.

Gradient
→ tells me the local slope / uphill direction.

Negative gradient
→ points toward a downhill direction.

Learning rate
→ tells me how large a step to take.
```

Then:

```text
Gradient Descent
→ repeatedly take downhill steps
→ reduce cost
→ find better parameters
```

This same idea applies to Logistic Regression.

The only difference is that the cost now comes from probability predictions and Binary Cross-Entropy.

---

# 17. Why does the Chain Rule appear?

A weight does not directly touch the final loss.

The dependency chain is:

```text
weight
  ↓
z = w^T x + b
  ↓
sigmoid
  ↓
probability p
  ↓
BCE loss
```

So the question is:

> If I change the weight slightly, how does that change z, then the sigmoid output, then the probability, and finally the loss?

The **Chain Rule** connects those effects.

Conceptually:

```text
weight
→ z
→ sigmoid
→ probability
→ loss
```

During differentiation, we trace that dependency backward.

This is also the foundation of **backpropagation** in Neural Networks.

---

# 18. Important connection to Backpropagation

For Logistic Regression:

```text
weight
  ↓
z
  ↓
sigmoid
  ↓
probability
  ↓
loss
```

For a Neural Network:

```text
weight
  ↓
neuron
  ↓
hidden layer
  ↓
next hidden layer
  ↓
output
  ↓
loss
```

The Neural Network chain is longer, but the idea is the same:

> Work backward through the chain and calculate gradients.

So Logistic Regression is a very useful preparation for understanding Neural Networks.

---

# 19. A very small training example

Suppose:

```text
Actual class = 1
```

At the beginning:

```text
p = 0.20
```

The BCE is high.

After Gradient Descent updates the weights:

```text
p = 0.45
```

Later:

```text
p = 0.72
```

Later:

```text
p = 0.91
```

Conceptually the model is learning parameter values that reduce the BCE cost.

Do not interpret this as the optimizer directly editing `p`.

The optimizer changes the parameters, and the new parameters create new probabilities.

---

# 20. Binary Logistic Regression summary

The complete binary pipeline is:

```text
Features x
   ↓
Linear score
z = w^T x + b
   ↓
Sigmoid
   ↓
Probability p
   ↓
Threshold
   ↓
Class 0 / 1
```

During training:

```text
Probability p
   ↓
Compare with actual y
   ↓
Binary Cross-Entropy
   ↓
Gradients
   ↓
Gradient Descent / optimizer
   ↓
Update w and b
```

---

# 21. What if there are more than two classes?

Binary Logistic Regression naturally handles:

```text
Class 0
vs
Class 1
```

But many problems contain several classes:

```text
Cat
Dog
Horse
```

Two common ideas are:

```text
1. One-vs-Rest (OvR)
2. Softmax / Multinomial Logistic Regression
```

---

# 22. One-vs-Rest (OvR)

Suppose the classes are:

```text
Cat
Dog
Horse
```

OvR builds a separate binary classifier for each class.

```text
Classifier 1:
Cat vs Not Cat

Classifier 2:
Dog vs Not Dog

Classifier 3:
Horse vs Not Horse
```

For a new example, each classifier produces a score/probability-like output.

Example:

```text
Cat   → 0.20
Dog   → 0.75
Horse → 0.10
```

The highest score wins:

```text
Prediction = Dog
```

Memory rule:

```text
One class
vs
all remaining classes
```

That is why it is called **One-vs-Rest**.

---

# 23. Softmax

Softmax is commonly used for mutually exclusive multiclass classification.

Suppose the model produces raw class scores:

```text
Cat   → z1
Dog   → z2
Horse → z3
```

Softmax converts those scores into probabilities that sum to 1.

Example:

```text
Cat   = 0.10
Dog   = 0.70
Horse = 0.20
----------------
Total = 1.00
```

The class with the highest probability becomes the prediction.

Memory rule:

```text
raw class scores
      ↓
Softmax
      ↓
probability distribution across classes
```

---

# 24. Sigmoid vs Softmax

## Sigmoid

Typically used for binary output:

```text
one raw score
    ↓
sigmoid
    ↓
probability of Class 1
```

Example:

```text
Fraud probability = 0.85
```

## Softmax

Used for mutually exclusive multiclass output:

```text
score Cat
score Dog
score Horse
     ↓
Softmax
     ↓
probability for each class
```

Example:

```text
Cat   = 0.10
Dog   = 0.70
Horse = 0.20
```

---

# 25. OvR vs Softmax

## One-vs-Rest

```text
multiple independent binary classifiers
```

Example:

```text
Cat vs Rest
Dog vs Rest
Horse vs Rest
```

The outputs are not inherently forced to sum to exactly 1.

## Softmax

```text
one multiclass probability distribution
```

The outputs are normalized so that:

```text
all class probabilities sum to 1
```

Simple memory:

```text
OvR     → many binary problems
Softmax → one multiclass probability distribution
```

---

# 26. Main limitation: linear decision boundary

The most important limitation of plain Logistic Regression is that it fundamentally creates a **linear decision boundary** in the original feature space.

In two dimensions, the boundary is a line.

```text
Class A:  ● ● ●
          ● ● ●

-------------------

Class B:  x x x
          x x x
```

This is easy for Logistic Regression.

But imagine a nonlinear pattern:

```text
●       x
    x
●       ●
```

A single straight line may not separate the classes properly.

This is why Logistic Regression can struggle when the data is not linearly separable.

---

# 27. Limited automatic feature interactions

Suppose we have:

```text
x1 = temperature
x2 = humidity
```

Maybe neither feature is sufficient alone, but the combination is important:

```text
high temperature AND high humidity
```

Plain Logistic Regression does not automatically create all possible complex feature interactions.

We may manually engineer terms such as:

```text
x3 = x1 * x2
```

Neural Networks can learn many complex combinations internally through hidden layers.

---

# 28. Limited capacity for highly complex patterns

Logistic Regression is often excellent for relatively simple and interpretable classification tasks.

But consider image recognition.

The model may need to learn a hierarchy like:

```text
pixels
  ↓
edges
  ↓
shapes
  ↓
object parts
  ↓
full object
```

A single Logistic Regression layer does not naturally learn that hierarchy.

This motivates Neural Networks and deep learning.

---

# 29. Sensitivity to outliers and noise

Extreme observations can influence the learned parameters and therefore the decision boundary.

Example:

```text
Most Class A points: ● ● ● ● ●

One strange Class A point very far away:                 ●
```

That unusual example can affect the fitted coefficients.

This does not mean Logistic Regression always fails in the presence of outliers, but data quality and preprocessing still matter.

---

# 30. Multiclass is an extension

The basic Logistic Regression idea is naturally binary.

For multiple classes we use extensions such as:

```text
One-vs-Rest
Multinomial Logistic Regression / Softmax
```

So multiclass classification is possible, but it extends the original binary formulation.

---

# 31. Why Neural Networks come next

Logistic Regression can be viewed as one relatively simple transformation:

```text
Inputs
   ↓
weighted sum + bias
   ↓
sigmoid
   ↓
output
```

A Neural Network connects many such computational blocks:

```text
Inputs
   ↓
Hidden Layer
   ↓
Hidden Layer
   ↓
Output Layer
```

The hidden layers allow the network to learn nonlinear feature transformations and more complex representations.

This is the conceptual bridge:

```text
Logistic Regression
→ one sigmoid-style classification block

Neural Network
→ many connected neurons + hidden layers
```

---

# 32. Common misconceptions to avoid

## Misconception 1

```text
Sigmoid directly chooses the final class.
```

Not exactly.

Correct:

```text
Sigmoid → probability
Threshold → class
```

## Misconception 2

```text
Gradient Descent directly changes the probability.
```

Wrong.

Correct:

```text
Gradient Descent changes weights and bias
→ z changes
→ sigmoid output changes
→ probability changes
```

## Misconception 3

```text
A probability above 0.5 must always be Class 1 in every application.
```

Wrong.

The threshold can be changed according to the application and metric priorities.

## Misconception 4

```text
Logistic Regression is a regression model because of its name.
```

For normal ML usage, Logistic Regression is primarily a **classification algorithm**.

## Misconception 5

```text
A large coefficient automatically means a feature is globally the most important.
```

Not necessarily.

Feature scale, encoding, correlation, regularization, and model context all matter.

---

# 33. Practical scikit-learn workflow

A common implementation looks like:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
X, y = load_breast_cancer(return_X_y=True)

# 2. Split before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Fit scaling on TRAINING data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# 5. Probabilities
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# 6. Final class predictions
predictions = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
```

The conceptual translation is:

```text
prepare data
    ↓
scale features
    ↓
learn weights + bias
    ↓
calculate probabilities
    ↓
apply decision rule
    ↓
evaluate predictions
```

---

# 34. predict_proba vs predict

In scikit-learn:

```python
model.predict_proba(X)
```

returns class probabilities.

For binary classification it usually returns two columns:

```text
P(Class 0), P(Class 1)
```

while:

```python
model.predict(X)
```

returns final class labels.

Memory rule:

```text
predict_proba → probabilities
predict       → classes
```

---

# 35. A from-scratch view

The essential mathematics can be written in a few steps.

```python
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_probability(X, w, b):
    z = X @ w + b
    return sigmoid(z)


def binary_cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```

Then the training loop conceptually becomes:

```text
z = Xw + b
p = sigmoid(z)
loss = BCE(y, p)
calculate gradients
update w and b
repeat
```

The companion Python file in this folder includes both a scikit-learn example and a compact from-scratch implementation.

---

# 36. Exam-style questions

## Q1. Why is Logistic Regression used?

It is mainly used for classification, especially binary classification, where the model estimates the probability of a class.

## Q2. What is the role of the sigmoid function?

It converts the raw linear score `z = w^T x + b` into a value between 0 and 1 that can be interpreted as a class probability estimate.

## Q3. What is the role of the threshold?

It converts the predicted probability into a final class label.

## Q4. What loss function is commonly used for binary Logistic Regression?

Binary Cross-Entropy / log loss.

## Q5. Why is BCE useful?

It evaluates probability predictions and strongly penalizes confidently wrong predictions.

## Q6. How are the weights learned?

The model calculates gradients of the cost with respect to the weights and bias and uses Gradient Descent or another optimizer to update them.

## Q7. What is the Chain Rule used for?

It connects how a change in a parameter affects the raw score, sigmoid output, probability, and finally the loss.

## Q8. What is One-vs-Rest?

A multiclass strategy that trains one binary classifier per class: one class versus all remaining classes.

## Q9. What does Softmax do?

It converts multiple class scores into a probability distribution whose values sum to 1.

## Q10. What is the main limitation of Logistic Regression?

Plain Logistic Regression learns a linear decision boundary in the original feature space and therefore struggles with strongly nonlinear patterns unless features are transformed or engineered.

---

# 37. ⭐ CHEAT SHEET

```text
LOGISTIC REGRESSION

Purpose:
→ classification
→ especially binary classification

----------------------------------------

CORE MODEL

z = w^T x + b

z
→ raw linear score

----------------------------------------

SIGMOID

p = sigmoid(z)

Purpose:
→ squeeze any real number into 0–1
→ obtain probability estimate for Class 1

----------------------------------------

THRESHOLD

p >= threshold
→ Class 1

p < threshold
→ Class 0

Common threshold:
→ 0.5

But threshold can be changed.

----------------------------------------

TRAINING LOSS

Binary Cross-Entropy:

L = -[y log(p) + (1-y) log(1-p)]

Correct + confident
→ low loss

Wrong + confident
→ very high loss

----------------------------------------

GRADIENT DESCENT

Cost
→ how wrong am I?

Gradient
→ which direction increases cost / local slope

Negative gradient
→ downhill direction

Learning rate
→ step size

Gradient Descent
→ updates weights + bias to reduce cost

----------------------------------------

IMPORTANT CHAIN

weights change
→ z changes
→ sigmoid changes
→ probability changes
→ BCE changes

----------------------------------------

CHAIN RULE

weight
→ z
→ sigmoid
→ probability
→ loss

Used to calculate gradients.

This is the bridge to backpropagation.

----------------------------------------

MULTICLASS

OvR
→ one binary classifier per class
→ one class vs rest

Softmax
→ converts class scores to probabilities
→ probabilities sum to 1

----------------------------------------

SIGMOID VS SOFTMAX

Sigmoid
→ binary output probability

Softmax
→ mutually exclusive multiclass probabilities

----------------------------------------

MAIN LIMITATION

Plain Logistic Regression
→ linear decision boundary
→ limited automatic nonlinear feature interaction
→ limited capacity for highly complex patterns

----------------------------------------

BRIDGE TO NEURAL NETWORKS

Logistic Regression:
inputs
→ weighted sum + bias
→ sigmoid
→ output

Neural Network:
many such computational blocks
→ hidden layers
→ nonlinear transformations
→ complex prediction

----------------------------------------

SCIKIT-LEARN

model.fit(...)
→ learn parameters

model.predict_proba(...)
→ probabilities

model.predict(...)
→ class labels
```

---

# Final mental model

Do not think of Logistic Regression as a collection of disconnected formulas.

Think of one complete pipeline:

```text
FEATURES
   ↓
WEIGHTS + BIAS
   ↓
LINEAR SCORE z
   ↓
SIGMOID
   ↓
PROBABILITY
   ↓
THRESHOLD
   ↓
CLASS
```

During training:

```text
PROBABILITY
   ↓
BINARY CROSS-ENTROPY
   ↓
GRADIENTS
   ↓
GRADIENT DESCENT / OPTIMIZER
   ↓
BETTER WEIGHTS + BIAS
   ↓
REPEAT
```

That is Logistic Regression.
