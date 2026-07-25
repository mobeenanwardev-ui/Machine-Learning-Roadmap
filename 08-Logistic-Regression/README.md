# Logistic Regression

This section is a **concept-first, exam-revision guide** to Logistic Regression.

The goal is not to memorize formulas blindly. The goal is to understand why Logistic Regression exists, how it turns a linear score into a probability, how Binary Cross-Entropy measures classification error, how Gradient Descent learns the parameters, how multiclass extensions work, and why Logistic Regression eventually leads us toward Neural Networks.

The recommended revision order is:

```text
WHY
 ↓
CORE IDEA
 ↓
ONE EXAMPLE
 ↓
FORMULA
 ↓
TRAINING FLOW
 ↓
LIMITATIONS
 ↓
EXAM MEMORY RULE
```

---

# 1. Why do we need Logistic Regression?

Linear Regression predicts a **continuous value**.

Examples:

```text
House price → 320000
Temperature → 24.7
Salary      → 45000
```

But many machine-learning problems ask for a **class**.

Examples:

```text
Email       → Spam / Not Spam
Patient     → Disease / No Disease
Transaction → Fraud / Not Fraud
Customer    → Churn / Stay
```

So the main problem is:

> How can we use a regression-like weighted model but produce a probability and then a class?

That is exactly what Logistic Regression does.

---

# 2. The first part is still linear

Logistic Regression starts with the same weighted-sum structure we already know from Linear Regression:

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

or compactly:

```text
z = w^T x + b
```

where:

```text
x1, x2, ... = input features
w1, w2, ... = weights
b           = bias / intercept
z           = raw linear score
```

Example:

```text
x1 = age
x2 = blood pressure
x3 = cholesterol
```

The model calculates:

```text
features
   ↓
weights
   ↓
weighted sum + bias
   ↓
z
```

Important:

> `z` is not yet a probability.

It may be any real number:

```text
-10
-2
0
3.5
20
```

So we need another function to convert this raw score into something between `0` and `1`.

---

# 3. Enter the Sigmoid function

The Sigmoid function is:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

Its job is simple:

> Take any real-valued score and squeeze it into the interval between `0` and `1`.

Examples:

```text
z = -10 → sigmoid(z) ≈ 0
z = -2  → sigmoid(z) ≈ 0.12
z = 0   → sigmoid(z) = 0.50
z = 2   → sigmoid(z) ≈ 0.88
z = 10  → sigmoid(z) ≈ 1
```

The shape is S-like:

```text
Probability
1.0 |                 ________
    |              __/
    |            _/
0.5 |-----------●------------
    |         _/
    |      __/
0.0 |_____/
    +------------------------→ z
              0
```

---

# 4. What does the Sigmoid output mean?

Suppose:

```text
sigmoid(z) = 0.87
```

For a binary problem where:

```text
Class 1 = Spam
Class 0 = Not Spam
```

we interpret `0.87` as:

> The model estimates a high probability for Class 1.

So Logistic Regression produces a probability-like output:

```text
0.00 -------------------------- 1.00
Class 0 side                    Class 1 side
```

Very important:

> Logistic Regression does not immediately output the final class. It first produces a probability, then a threshold is applied.

---

# 5. Classification threshold

A common threshold is:

```text
0.5
```

Then:

```text
Probability >= 0.5 → Class 1
Probability <  0.5 → Class 0
```

Examples:

```text
0.92 → Class 1
0.73 → Class 1
0.51 → Class 1

0.49 → Class 0
0.20 → Class 0
0.03 → Class 0
```

The threshold is not always required to be `0.5`.

For some problems we may choose another threshold depending on the cost of false positives and false negatives.

For exam memory:

```text
Sigmoid gives probability.
Threshold converts probability into a class.
```

---

# 6. Logistic Regression complete prediction flow

```text
Features X
   ↓
Weighted sum
   ↓
z = w^T x + b
   ↓
Sigmoid
   ↓
Probability p
   ↓
Threshold
   ↓
Class 0 or Class 1
```

That is the basic forward computation of Logistic Regression.

---

# 7. Linear Regression vs Logistic Regression

## Linear Regression

```text
Features
   ↓
w^T x + b
   ↓
Continuous number
```

Example:

```text
Predicted house price = 300000
```

## Logistic Regression

```text
Features
   ↓
w^T x + b
   ↓
Sigmoid
   ↓
Probability
   ↓
Threshold
   ↓
Class
```

Example:

```text
Fraud probability = 0.91
→ Class 1 = Fraud
```

Main memory rule:

```text
Linear Regression
→ continuous prediction

Logistic Regression
→ probability → class
```

---

# 8. Connection to an artificial neuron

A basic artificial neuron performs:

```text
inputs
   ↓
weights + bias
   ↓
activation function
   ↓
output
```

Logistic Regression does:

```text
inputs
   ↓
weights + bias
   ↓
Sigmoid activation
   ↓
probability
```

This is an extremely important bridge toward Neural Networks.

A useful comparison is:

```text
Linear Regression
→ weighted sum + identity activation

Logistic Regression
→ weighted sum + sigmoid activation
```

So one Logistic Regression unit already looks very much like a simple neuron.

---

# 9. What is the decision boundary?

The probability threshold `0.5` corresponds to:

```text
sigmoid(z) = 0.5
```

Sigmoid gives `0.5` when:

```text
z = 0
```

So the basic decision boundary is:

```text
w^T x + b = 0
```

In two dimensions this is a line.

In three dimensions it is a plane.

In higher dimensions it is a hyperplane.

That explains why standard Logistic Regression mainly learns a **linear decision boundary** in the original feature space.

---

# 10. Why do we need a loss function?

Suppose the true class is:

```text
Actual y = 1
```

Now compare three predictions:

```text
Model A → p = 0.99
Model B → p = 0.60
Model C → p = 0.05
```

All three probabilities are different in quality.

We need a function that tells us:

> How bad was this probability compared with the true class?

For binary Logistic Regression, a standard choice is **Binary Cross-Entropy**.

---

# 11. Binary Cross-Entropy (BCE)

The formula is:

```text
L = -[y log(p) + (1-y) log(1-p)]
```

where:

```text
y = actual class, either 0 or 1
p = predicted probability for Class 1
```

Do not memorize the full formula without splitting it into the two possible cases.

---

# 12. BCE when the actual class is 1

If:

```text
y = 1
```

then:

```text
L = -log(p)
```

Meaning:

> When the true class is 1, the predicted probability should be close to 1.

Examples:

```text
Actual y = 1

p = 0.99 → tiny loss
p = 0.80 → small loss
p = 0.50 → bigger loss
p = 0.10 → very large loss
```

So BCE punishes a confident wrong prediction strongly.

---

# 13. BCE when the actual class is 0

If:

```text
y = 0
```

then:

```text
L = -log(1-p)
```

Meaning:

> When the true class is 0, the predicted probability should be close to 0.

Examples:

```text
Actual y = 0

p = 0.01 → tiny loss
p = 0.20 → small loss
p = 0.50 → bigger loss
p = 0.90 → very large loss
```

Again, being confidently wrong produces a large penalty.

---

# 14. Why BCE is conceptually useful

Think of BCE this way:

```text
Correct + confident
→ very small loss

Correct but uncertain
→ moderate loss

Wrong + confident
→ very large loss
```

Example:

```text
Actual = 1

Prediction 0.99
→ excellent

Prediction 0.51
→ technically correct at threshold 0.5, but uncertain

Prediction 0.01
→ confidently wrong
```

BCE distinguishes these cases.

---

# 15. Why not simply use MSE?

Linear Regression naturally uses squared error for continuous targets.

Logistic Regression predicts probabilities for binary outcomes.

Binary Cross-Entropy is better aligned with that probabilistic classification setting.

Conceptually:

```text
Linear Regression
→ continuous output
→ MSE

Logistic Regression
→ binary probability
→ BCE
```

BCE also has very useful mathematical properties when combined with the Sigmoid function, making the gradient expressions clean for Logistic Regression training.

For exam understanding, remember:

> MSE measures numeric prediction distance. BCE evaluates how well predicted probabilities match binary targets.

---

# 16. Logistic Regression training

At the beginning we do not know the best:

```text
w1, w2, ... wn
b
```

Training means finding weights and bias that reduce the Binary Cross-Entropy objective.

The full process is:

```text
Current weights + bias
        ↓
Calculate z = w^T x + b
        ↓
Apply Sigmoid
        ↓
Get probability p
        ↓
Compare p with actual y
        ↓
Calculate BCE
        ↓
Calculate gradients
        ↓
Gradient Descent
        ↓
Update weights + bias
        ↓
Repeat
```

---

# 17. Gradient Descent in Logistic Regression

The optimization idea is the same as in Linear Regression.

For a parameter `theta`:

```text
theta_new = theta_old - learning_rate * gradient
```

For weights and bias:

```text
w ← w - α * ∂J/∂w
b ← b - α * ∂J/∂b
```

where:

```text
α = learning rate
J = objective / cost
```

The big difference is not Gradient Descent itself.

The difference is what comes before it:

```text
Linear Regression
→ prediction
→ MSE
→ gradients
→ Gradient Descent

Logistic Regression
→ sigmoid probability
→ BCE
→ gradients
→ Gradient Descent
```

---

# 18. Very important: Gradient Descent does not directly change probability

A common misunderstanding would be:

```text
Gradient Descent changes probability directly ❌
```

What actually happens is:

```text
Gradient Descent changes weights + bias
        ↓
z changes
        ↓
Sigmoid(z) changes
        ↓
Probability changes
        ↓
Loss changes
```

So the optimizer acts on the **parameters**, not directly on the predicted probability.

---

# 19. Chain Rule in Logistic Regression

The weight affects the final loss through several intermediate steps:

```text
weight
  ↓
z
  ↓
Sigmoid
  ↓
probability p
  ↓
BCE loss
```

So when we ask:

> How does changing a weight affect the final loss?

we need to follow this dependency chain backward.

That is what the **chain rule** does mathematically.

Conceptually:

```text
change weight
    ↓
changes z
    ↓
changes sigmoid output
    ↓
changes probability
    ↓
changes BCE
```

This same idea becomes the core of **backpropagation** in Neural Networks.

---

# 20. Forward computation vs backward gradient calculation

This distinction becomes increasingly important later.

## Forward computation

```text
input
 ↓
weighted sum
 ↓
sigmoid
 ↓
probability
 ↓
loss
```

## Backward gradient calculation

```text
loss
 ↑
probability
 ↑
sigmoid
 ↑
z
 ↑
weights / bias
```

In Logistic Regression this chain is short.

In Neural Networks the same idea extends through many layers.

---

# 21. One simple training story

Suppose:

```text
Actual class = 1
```

Current model predicts:

```text
p = 0.20
```

That produces a relatively high BCE loss.

Gradients are calculated.

Gradient Descent updates the weights and bias.

Next iteration:

```text
p = 0.45
```

Then perhaps:

```text
p = 0.72
```

Then:

```text
p = 0.91
```

The exact numbers are only illustrative, but the idea is:

```text
weights change
→ probability improves
→ loss becomes smaller
```

---

# 22. Binary classification complete workflow

```text
1. Start with features X
2. Compute z = w^T x + b
3. Apply sigmoid
4. Obtain probability p
5. Compare p with actual target y
6. Compute BCE loss
7. Compute gradients
8. Apply Gradient Descent
9. Update weights and bias
10. Repeat until convergence
11. Use threshold for final class prediction
```

---

# 23. Multiclass classification

Standard Logistic Regression is naturally introduced for two classes:

```text
Class 0
vs
Class 1
```

But real problems can have more than two classes.

Example:

```text
Cat
Dog
Horse
```

Two important extensions are:

```text
1. One-vs-Rest (OvR)
2. Softmax / multinomial classification
```

---

# 24. One-vs-Rest (OvR)

Suppose we have:

```text
Cat
Dog
Horse
```

Build one binary classifier for each class.

```text
Classifier 1:
Cat vs Rest

Classifier 2:
Dog vs Rest

Classifier 3:
Horse vs Rest
```

Each classifier produces its own score or probability-like estimate.

Example:

```text
Cat   → 0.20
Dog   → 0.75
Horse → 0.10
```

Choose the highest:

```text
Dog → prediction
```

Memory rule:

```text
ONE class
vs
ALL remaining classes
```

---

# 25. Softmax

Softmax takes several raw class scores and converts them into a probability distribution.

Suppose raw scores are:

```text
Cat   → z1
Dog   → z2
Horse → z3
```

Softmax converts them into:

```text
P(Cat)
P(Dog)
P(Horse)
```

with:

```text
P(Cat) + P(Dog) + P(Horse) = 1
```

Example:

```text
Cat   = 0.10
Dog   = 0.70
Horse = 0.20
----------------
Total = 1.00
```

The class with the highest probability becomes the prediction.

---

# 26. Sigmoid vs Softmax

## Sigmoid

Typical binary setting:

```text
one score
   ↓
sigmoid
   ↓
probability of Class 1
```

## Softmax

Multiclass setting:

```text
score for Class A
score for Class B
score for Class C
       ↓
     Softmax
       ↓
probability distribution across classes
```

Memory rule:

```text
Sigmoid → binary probability
Softmax → multiclass probability distribution
```

---

# 27. OvR vs Softmax

```text
One-vs-Rest
→ several independent binary classifiers

Softmax
→ one joint multiclass probability distribution
```

With OvR, separate outputs are not inherently required to sum to 1.

Example:

```text
Cat   = 0.70
Dog   = 0.60
Horse = 0.20
```

Then choose the highest score.

With Softmax:

```text
Cat   = 0.60
Dog   = 0.30
Horse = 0.10
----------------
Total = 1.00
```

---

# 28. Main limitation: linear decision boundary

The most important limitation is:

> Logistic Regression mainly produces a linear decision boundary in the original feature space.

Easy case:

```text
● ● ● ●
● ● ● ●

---------------------

× × × ×
× × × ×
```

A straight line can separate the classes.

Logistic Regression can work very well.

Harder case:

```text
●       ×
    ×
●       ●
```

There may be no single straight line that separates the classes properly.

Then the model may struggle unless features are transformed or engineered.

---

# 29. Limited feature interactions

Suppose:

```text
x1 = temperature
x2 = humidity
```

Maybe neither feature alone is enough.

But the combination:

```text
high temperature AND high humidity
```

may matter strongly.

Plain Logistic Regression does not automatically create arbitrary complex feature interactions.

We may need to manually engineer features such as:

```text
x3 = x1 * x2
```

Neural Networks can learn much richer interactions automatically through hidden layers.

---

# 30. Limited capacity for highly complex data

Logistic Regression is often excellent for relatively simple structured classification problems.

But consider image recognition.

An image may require a hierarchy such as:

```text
pixels
 ↓
edges
 ↓
shapes
 ↓
parts
 ↓
object
```

One Logistic Regression layer does not naturally learn such a deep hierarchy of representations.

This motivates multi-layer Neural Networks.

---

# 31. Sensitivity to outliers and noisy observations

Extreme observations can influence fitted coefficients and the decision boundary.

This does not mean Logistic Regression is useless in noisy data.

It means preprocessing, diagnostics, robust feature design, and careful evaluation can matter.

Exam memory:

```text
Outliers/noise
→ may distort the learned boundary
```

---

# 32. Multiclass is an extension

Standard Logistic Regression is naturally binary.

For more classes we need techniques such as:

```text
One-vs-Rest
Softmax / multinomial Logistic Regression
```

So multiclass handling is not the simplest original form of the model.

---

# 33. Important nuance about scale and high-dimensional data

A course may list large and high-dimensional data as a limitation.

Do not translate that into:

```text
Logistic Regression cannot work on large datasets ❌
```

Logistic Regression can actually be computationally efficient for many large structured problems.

The deeper conceptual limitation for this lecture is:

> Its representational capacity is limited compared with models that learn nonlinear hierarchical features.

That is the bridge toward Neural Networks.

---

# 34. Why Neural Networks after Logistic Regression?

Logistic Regression looks like:

```text
Inputs
   ↓
weighted sum
   ↓
Sigmoid
   ↓
Output
```

A Neural Network extends this idea:

```text
Inputs
   ↓
Hidden neurons
   ↓
Hidden neurons
   ↓
Output neuron(s)
```

Instead of one simple transformation, the network stacks many transformations.

This allows it to learn:

```text
nonlinear patterns
complex feature interactions
hierarchical representations
```

That is why Logistic Regression is an excellent stepping stone toward Neural Networks.

---

# 35. Minimal scikit-learn example

The following example uses the built-in Breast Cancer Wisconsin dataset from scikit-learn.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Scaling + Logistic Regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)

# Train
model.fit(X_train, y_train)

# Class prediction
predictions = model.predict(X_test)

# Probability prediction
probabilities = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print("First probability pair:", probabilities[0])
```

Conceptually:

```text
fit(...)
→ learn weights and bias

predict_proba(...)
→ obtain class probabilities

predict(...)
→ apply model decision rule and return classes
```

---

# 36. Manual Sigmoid example

```python
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for z in [-5, -2, 0, 2, 5]:
    print(z, sigmoid(z))
```

Expected pattern:

```text
large negative z → probability close to 0
z = 0            → probability 0.5
large positive z → probability close to 1
```

---

# 37. Binary Cross-Entropy from scratch

```python
import numpy as np


def binary_cross_entropy(y_true, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


print(binary_cross_entropy(1, 0.99))
print(binary_cross_entropy(1, 0.10))
```

Interpretation:

```text
actual = 1, predicted = 0.99
→ very small loss

actual = 1, predicted = 0.10
→ much larger loss
```

---

# 38. Logistic Regression from scratch with Gradient Descent

This is a small educational implementation so the training logic is visible.

```python
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Small example dataset
X = np.array([
    [0.0],
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
])

y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

# Parameters
w = np.zeros(X.shape[1])
b = 0.0
learning_rate = 0.1
n = len(y)

for _ in range(3000):
    # Forward pass
    z = X @ w + b
    p = sigmoid(z)

    # Gradients for sigmoid + BCE Logistic Regression
    error = p - y
    dw = (X.T @ error) / n
    db = np.mean(error)

    # Gradient Descent
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Final probabilities and classes
probabilities = sigmoid(X @ w + b)
predictions = (probabilities >= 0.5).astype(int)

print("Weight:", w)
print("Bias:", b)
print("Probabilities:", probabilities)
print("Predictions:", predictions)
```

---

# 39. Read the from-scratch code like a story

```python
z = X @ w + b
```

means:

> Calculate the raw linear score.

```python
p = sigmoid(z)
```

means:

> Convert the raw score into probabilities.

```python
error = p - y
```

captures the probability mismatch that appears in the gradient for Logistic Regression with BCE.

```python
dw = (X.T @ error) / n
db = np.mean(error)
```

means:

> Calculate how the objective changes with respect to weights and bias.

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

means:

> Use Gradient Descent to improve the parameters.

```python
predictions = (probabilities >= 0.5).astype(int)
```

means:

> Convert probabilities into final binary classes.

---

# 40. Common misunderstandings

## "Logistic Regression predicts a continuous target"

```text
Wrong ❌
```

It is mainly used for classification and models class probabilities.

## "z is already the probability"

```text
Wrong ❌
```

`z = w^T x + b` is a raw score.

The Sigmoid transforms it into a value between 0 and 1.

## "Sigmoid directly chooses the class"

```text
Wrong ❌
```

Sigmoid gives a probability.

The threshold gives the class.

## "Gradient Descent changes the probability directly"

```text
Wrong ❌
```

Gradient Descent changes weights and bias.

Those parameter changes indirectly change the probability.

## "BCE only checks whether the final class is correct"

```text
Wrong ❌
```

BCE uses the predicted probability and therefore distinguishes confidence levels.

## "Softmax and Sigmoid are the same"

```text
Wrong ❌
```

Sigmoid is commonly used for binary probability.

Softmax creates a probability distribution across mutually exclusive classes.

## "Logistic Regression can automatically learn any complex nonlinear boundary"

```text
Wrong ❌
```

Standard Logistic Regression learns a linear boundary in the original feature space unless nonlinear features are engineered.

---

# 41. Exam-style questions and answers

## Q1. What is Logistic Regression?

Logistic Regression is a classification model that computes a linear score from input features and transforms that score through a Logistic/Sigmoid function to estimate a probability.

## Q2. What is the linear score?

```text
z = w^T x + b
```

## Q3. Why do we need Sigmoid?

Because the raw linear score can be any real number. Sigmoid maps it into the interval `(0,1)` so it can be interpreted as a probability-like output.

## Q4. What does a threshold do?

It converts the predicted probability into a class label.

Example:

```text
p >= 0.5 → Class 1
p <  0.5 → Class 0
```

## Q5. What loss is commonly used for binary Logistic Regression?

Binary Cross-Entropy.

```text
L = -[y log(p) + (1-y) log(1-p)]
```

## Q6. What happens when y = 1?

The loss becomes:

```text
-log(p)
```

so the model is rewarded for pushing `p` toward 1.

## Q7. What happens when y = 0?

The loss becomes:

```text
-log(1-p)
```

so the model is rewarded for pushing `p` toward 0.

## Q8. Why is BCE useful?

It evaluates the quality of predicted probabilities and strongly penalizes confident wrong predictions.

## Q9. How is Logistic Regression trained?

```text
forward calculation
→ sigmoid probability
→ BCE
→ gradients
→ Gradient Descent
→ parameter updates
```

## Q10. What does Gradient Descent update?

The weights and bias.

It does not directly change the probability.

## Q11. Why is chain rule needed?

Because the loss depends on the weights indirectly through:

```text
weight → z → sigmoid → probability → loss
```

## Q12. What is One-vs-Rest?

One binary classifier is trained for each class against all remaining classes.

## Q13. What is Softmax?

Softmax converts multiple class scores into probabilities that sum to 1.

## Q14. Sigmoid vs Softmax?

```text
Sigmoid → mainly binary setting
Softmax → multiclass probability distribution
```

## Q15. What is the main limitation of Logistic Regression?

Its standard decision boundary is linear in the original feature space, so it can struggle with highly nonlinear and complex relationships.

## Q16. Why do Neural Networks help?

They combine many nonlinear computational units and hidden layers, allowing much richer feature interactions and hierarchical representations.

---

# 42. ⭐ CHEAT SHEET

```text
LOGISTIC REGRESSION
→ classification model
→ mainly introduced for binary classification

LINEAR SCORE
→ z = w^T x + b

z
→ raw score
→ NOT yet a probability

SIGMOID
→ 1 / (1 + e^(-z))
→ maps any real number into 0...1

PROBABILITY
→ p = sigmoid(z)

THRESHOLD
→ converts probability to class
→ common example: 0.5

BINARY FLOW
X
↓
z = w^T x + b
↓
Sigmoid
↓
p
↓
threshold
↓
Class 0 / Class 1

BCE
→ Binary Cross-Entropy
→ compares actual binary target with predicted probability

BCE FORMULA
L = -[y log(p) + (1-y) log(1-p)]

IF y = 1
→ loss = -log(p)
→ want p close to 1

IF y = 0
→ loss = -log(1-p)
→ want p close to 0

BCE INTUITION
correct + confident → tiny loss
wrong + confident   → huge loss

TRAINING
weights + bias
↓
z
↓
Sigmoid
↓
probability
↓
BCE
↓
gradients
↓
Gradient Descent
↓
update weights + bias

GRADIENT DESCENT
→ same optimization idea as Linear Regression
→ update parameter opposite to gradient

IMPORTANT
Gradient Descent changes parameters,
NOT probability directly.

CHAIN RULE
weight
↓
z
↓
Sigmoid
↓
probability
↓
BCE

MULTICLASS
→ One-vs-Rest
→ Softmax

ONE-vs-REST
→ one classifier for each class against all remaining classes

SOFTMAX
→ turns multiple class scores into probabilities
→ probabilities sum to 1

SIGMOID
→ binary probability

SOFTMAX
→ multiclass probability distribution

DECISION BOUNDARY
→ w^T x + b = 0
→ linear in original feature space

LIMITATIONS
→ linear decision boundary
→ limited automatic feature interactions
→ limited capacity for highly complex data
→ sensitive to noisy/extreme observations
→ multiclass requires extension

WHY NEURAL NETWORKS?
→ hidden layers
→ nonlinear transformations
→ richer feature interactions
→ hierarchical feature learning
```

---

# Final mental model

Do not remember Logistic Regression only as:

```text
sigmoid + classification
```

Remember the complete system:

```text
Features
   ↓
Weighted combination + bias
   ↓
Raw score z
   ↓
Sigmoid
   ↓
Probability
   ↓
Compare with actual class
   ↓
Binary Cross-Entropy
   ↓
Gradients
   ↓
Gradient Descent
   ↓
Update weights + bias
   ↓
Repeat
```

And the most important transition toward Neural Networks is:

> **Logistic Regression already looks like one simple neuron. Neural Networks become powerful by connecting many such computational units through hidden layers and nonlinear activations.**
