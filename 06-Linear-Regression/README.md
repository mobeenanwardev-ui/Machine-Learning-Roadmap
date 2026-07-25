# Linear Regression

This section is a **concept-first, exam-revision guide** to Linear Regression.

The goal is not to memorize formulas blindly. The goal is to understand what the model is doing, why each mathematical component exists, how Gradient Descent actually learns the parameters, and how all of the pieces connect.

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

# 1. What problem does Linear Regression solve?

Linear Regression is used when the target we want to predict is a **continuous numerical value**.

Examples:

- house price,
- salary,
- temperature,
- sales amount,
- crop yield,
- insurance claim amount.

The basic idea is:

> Use information we already know about an observation to estimate a numerical output.

For a house-price problem:

```text
Inputs / features
-----------------
area
number of rooms
location
age of house

        ↓

Linear Regression model

        ↓

Predicted house price
```

The input variables are usually called **features**, **independent variables**, or **predictors**.

The value we are trying to predict is the **target**, **dependent variable**, or **output**.

---

# 2. The Linear Regression equation

For one feature:

```text
y_hat = w*x + b
```

For many features:

```text
y_hat = w1*x1 + w2*x2 + ... + wn*xn + b
```

In compact notation:

```text
y_hat = x^T w + b
```

where:

```text
x1, x2, ... = input features
w1, w2, ... = weights / coefficients
b           = bias / intercept
y_hat       = predicted value
```

The hat in `y_hat` means that the value is **estimated or predicted**, not the true observed value.

---

# 3. What do the weights mean?

A weight tells us how the prediction changes when a feature changes, while the other features are held fixed.

Example:

```text
Predicted price = 2000 * area + 10000 * rooms + b
```

Here:

```text
2000  = weight for area
10000 = weight for number of rooms
```

Conceptually:

```text
Feature
   ↓
multiplied by its weight
   ↓
contribution to prediction
```

A larger positive weight means increasing that feature tends to increase the prediction more strongly.

A negative weight means increasing that feature tends to decrease the prediction.

Important:

> A large raw coefficient does not automatically mean a feature is globally "more important" than another feature, because feature scales, units, and correlation between predictors matter.

---

# 4. What is the bias?

This was an important concept to correct during study.

The bias is **not**:

```text
an outlier remover       ❌
a prediction error       ❌
a mechanism that deletes mistakes ❌
```

The bias is the **intercept / baseline** of the model.

Consider:

```text
y = 2x + 10
```

Here:

```text
w = 2
b = 10
```

If `x = 0`, the prediction is still `10`.

So the bias allows the regression line to shift vertically instead of being forced through `(0, 0)`.

A useful geometric memory rule:

```text
Weight → changes the slope / tilt
Bias   → shifts the line up or down
```

During training, **both weights and bias are adjusted**.

After training, the learned parameters are normally kept fixed and used to predict new observations.

---

# 5. Why is it called "Linear" Regression?

With one feature, the fitted relationship looks like a line:

```text
y
↑
|        /
|      /
|    /
|  /
+--------------→ x
```

With multiple features, it becomes a plane or a higher-dimensional hyperplane.

More importantly, the model is called linear because it is **linear in its parameters**:

```text
y_hat = w1*x1 + w2*x2 + ... + wn*xn + b
```

The weights are added in a linear weighted combination.

---

# 6. Connection to an artificial neuron

A simple artificial neuron performs:

```text
inputs
   ↓
weights
   ↓
weighted sum + bias
   ↓
activation
   ↓
output
```

Linear Regression performs:

```text
inputs
   ↓
weights
   ↓
weighted sum + bias
   ↓
identity activation
   ↓
continuous output
```

An identity activation simply returns its input:

```text
g(z) = z
```

So a Linear Regression model can be viewed as a very simple neuron whose activation does not transform the weighted sum.

This connection becomes useful later when moving from Linear Regression to Logistic Regression and Neural Networks.

---

# 7. What does "fitting" or "training" mean?

At the beginning we do not know the best values of:

```text
w1, w2, ..., wn, b
```

Training means:

> Find parameter values that make the model's predictions as close as possible to the observed target values according to a chosen objective function.

Conceptually:

```text
Historical data
      ↓
Choose / initialize parameters
      ↓
Make predictions
      ↓
Measure error
      ↓
Improve parameters
      ↓
Repeat
      ↓
Trained model
```

The parameters are often collectively represented by:

```text
theta (θ)
```

So we can think of:

```text
θ = {weights, bias}
```

---

# 8. Assumptions of Linear Regression

For the classical Linear Regression model, the lecture highlights five important assumptions.

## 8.1 Linearity

The expected relationship between predictors and the target should be adequately represented by a linear combination of the predictors.

Example:

```text
house size increases
        ↓
house price tends to change systematically
```

If the true relationship is strongly curved, ordinary Linear Regression may fit badly unless the features are transformed.

## 8.2 Independence

Observations should be reasonably independent of one another.

Example:

Randomly sampled houses from different locations are preferable to many near-duplicate observations that are strongly dependent on each other.

## 8.3 Homoscedasticity

The variance of the residuals should be approximately constant across the prediction range.

In normal words:

> The model should not be extremely accurate for cheap houses but wildly inaccurate for expensive houses in a systematic way.

## 8.4 Normality of errors

Residuals are commonly assumed to be approximately normally distributed, especially when doing statistical inference with the fitted model.

A useful intuition:

```text
small errors → common
very large errors → rarer
```

## 8.5 No severe multicollinearity

Predictors should not be excessively correlated with each other.

Example:

```text
number of bedrooms
number of rooms
```

may contain highly overlapping information.

Strong multicollinearity can make coefficient estimates unstable and difficult to interpret.

### Exam memory rule

```text
Linear
Independent
Constant variance
Normal residuals
No severe multicollinearity
```

---

# 9. Prediction error: Loss versus Cost

Once the model predicts something, we need to answer:

> How wrong was the prediction?

Suppose:

```text
Actual house price    = 300
Predicted house price = 280
```

The prediction error is related to:

```text
actual - predicted
```

## Loss

A **loss function** measures how wrong the model is for **one observation**.

```text
One house
    ↓
One prediction
    ↓
One loss
```

## Cost / Objective function

A **cost function** or **objective function** summarizes the error across the training dataset.

```text
Many observations
      ↓
Many prediction errors
      ↓
One overall objective / cost
```

A practical memory rule:

```text
Loss → one example
Cost → whole dataset / overall objective
```

Terminology can vary between textbooks and software libraries, so always check how a source defines the terms. For this course, this distinction is useful.

---

# 10. Mean Squared Error (MSE)

A common objective for Linear Regression is Mean Squared Error.

Course-style form:

```text
J(θ) = (1 / 2M) * Σ (y_i - y_hat_i)^2
```

where:

```text
M       = number of training examples
y_i     = actual target for example i
y_hat_i = predicted target for example i
θ       = model parameters
```

The factor `1/2` is often included because it makes the derivative cleaner. It does not change where the minimum occurs.

---

# 11. Why square the errors?

This deserves intuition rather than memorization.

Suppose three prediction errors are:

```text
+10
-20
+10
```

If we simply add them:

```text
10 - 20 + 10 = 0
```

That would incorrectly make the model look perfect.

Squaring solves the sign-cancellation problem:

```text
10²     = 100
(-20)²  = 400
10²     = 100
```

It also penalizes large mistakes more strongly:

```text
error = 2   → squared error = 4
error = 10  → squared error = 100
```

So two useful reasons are:

```text
1. Positive and negative errors cannot cancel.
2. Large errors receive a larger penalty.
```

Another mathematical advantage is that squared error is smooth and differentiable, making gradient-based optimization convenient.

---

# 12. Why does the MSE cost curve look like a parabola?

This was one of the lecture quizzes.

For a simplified case with one parameter, MSE is a **quadratic function** of that parameter.

A quadratic function has the familiar U-shaped form:

```text
Cost
 ↑
 |      \       /
 |       \     /
 |        \   /
 |         \_/
 +----------------→ parameter θ
```

The important interpretation is:

```text
x-axis → parameter value θ
y-axis → cost J(θ)
```

The bottom of the curve is the parameter value that produces the smallest cost.

Very important:

> The x-axis is not the actual target and it is not the prediction. It represents a candidate model parameter.

The minimum means:

> This parameter value gives the smallest overall objective value among the values considered by the optimization process.

---

# 13. Why not simply use `np.min()`?

`np.min()` can find the smallest value in a set of values that has **already been calculated**.

Example:

```python
values = [10, 7, 3, 8]
print(min(values))
# 3
```

But in model training we do not already know every possible combination of:

```text
w1
w2
w3
...
b
```

These parameters are continuous and can form a huge high-dimensional search space.

Trying every possible combination would be a brute-force search and quickly becomes impractical.

We therefore use optimization algorithms that exploit the mathematical structure of the objective function.

One of the most important is **Gradient Descent**.

---

# 14. Gradient Descent — normal-human intuition first

This is the most important intuition in this section.

Forget machine learning for a moment.

Imagine you are standing somewhere on a mountain at night.

Your goal is:

> Reach the lowest point of the valley.

You cannot see the entire landscape, but you can feel the slope of the ground where you are standing.

So you do this:

```text
Feel the slope
     ↓
Find which way is downhill
     ↓
Take a step
     ↓
Feel the slope again
     ↓
Take another downhill step
     ↓
Repeat
     ↓
Reach the valley
```

That is the basic idea of **Gradient Descent**.

---

# 15. What exactly is the gradient?

The gradient tells us two important things about a function at our current position:

```text
Direction
→ which direction increases the function fastest

Magnitude
→ how steeply the function changes
```

Think of standing on a hill.

Very steep ground:

```text
large gradient
```

Nearly flat ground:

```text
small gradient
```

At a smooth minimum:

```text
gradient ≈ 0
```

The gradient itself points toward the **steepest increase**.

But our cost should go down.

Therefore Gradient Descent moves in the **opposite direction**.

```text
Gradient         → uphill
Negative gradient → downhill
```

---

# 16. Cost versus Gradient — do not confuse them

This distinction is fundamental.

```text
COST
→ How bad is my model right now?

GRADIENT
→ Which direction should the parameters move to change the cost most strongly?

GRADIENT DESCENT
→ Actually take a step in the direction that reduces the cost.
```

A normal-life analogy:

```text
Cost:
"You are 10 km away from home."

Gradient:
"The direction of improvement is to your left."

Gradient Descent:
"Take a step left, check again, and repeat."
```

---

# 17. Convert the hill example into Machine Learning

Now return to Linear Regression.

Suppose the model has one simplified parameter `θ`.

Different values of `θ` produce different costs:

```text
θ = 1 → high cost
θ = 2 → lower cost
θ = 3 → even lower
θ = 4 → minimum cost
θ = 5 → higher again
```

The cost landscape might look like:

```text
Cost
 ↑
 |  ●                 ●
 |    \             /
 |      ●         ●
 |        \       /
 |          \___/
 |            ●
 +----------------------→ θ
              best θ
```

If we start on the right side, Gradient Descent tells us to move left.

If we start on the left side, it tells us to move right.

The aim is to move toward a parameter value with smaller cost.

---

# 18. Gradient Descent update rule

The standard update is:

```text
θ_j ← θ_j - α * ∂J/∂θ_j
```

Read this in plain English:

```text
new parameter
=
old parameter
-
learning rate
×
gradient
```

Where:

```text
θ_j       = one model parameter
α         = learning rate
J         = cost function
∂J/∂θ_j   = how the cost changes with respect to that parameter
```

Do not memorize the symbols without the meaning.

The equation simply says:

> Look at the slope of the cost function and move the parameter a controlled step in the opposite direction.

---

# 19. Why is there a minus sign?

The gradient points in the direction of steepest **increase**.

But we want to minimize the cost.

Therefore:

```text
θ - gradient
```

moves us in the opposite direction.

Memory rule:

```text
Gradient → uphill
Minus gradient → downhill
```

---

# 20. Derivative versus Gradient

For a function with one parameter, we often talk about a **derivative**.

```text
one parameter
→ one slope
→ derivative
```

For a model with many parameters, we have many partial derivatives together.

```text
w1
w2
w3
...
b
```

The collection of all of these derivatives is the **gradient**.

Conceptually:

```text
Derivative → slope with respect to one variable
Gradient   → collection/vector of slopes for many variables
```

---

# 21. Learning Rate

The learning rate is usually written as:

```text
α
```

It controls the **step size** of Gradient Descent.

A useful memory rule:

```text
Gradient     → WHERE should I move?
Learning rate → HOW FAR should I move?
```

## Learning rate too large

```text
large steps
→ may jump over the minimum repeatedly
→ training may oscillate or diverge
```

Think:

```text
       minimum
          ↓
       \      /
   ● ---------> ●
      <---------
```

## Learning rate too small

```text
very small steps
→ stable but slow
→ many iterations required
```

```text
● → ● → ● → ● → ● → ● → minimum
```

## Reasonable learning rate

```text
controlled steps
→ moves efficiently toward lower cost
```

There is no single learning rate that is automatically best for every problem.

---

# 22. Why can the step become smaller near the minimum?

Even if `α` remains constant, the actual update contains:

```text
α × gradient
```

Far from the minimum, the slope may be steep:

```text
large gradient
→ larger update
```

Near a smooth minimum, the surface becomes flatter:

```text
small gradient
→ smaller update
```

At convergence:

```text
gradient ≈ 0
```

so updates become very small.

---

# 23. Multiple parameters

A real regression model may contain:

```text
w1
w2
w3
...
wn
b
```

Gradient Descent needs to determine how each parameter should change.

For every parameter, we ask:

```text
How does the cost change if this parameter changes slightly?
```

For example:

```text
∂J/∂w1
∂J/∂w2
∂J/∂w3
...
∂J/∂b
```

Each parameter is then updated using its own gradient component.

```text
w ← w - α * ∂J/∂w
b ← b - α * ∂J/∂b
```

With many features, the same idea is applied to every weight.

---

# 24. What is a partial derivative?

A partial derivative asks:

> How does the cost change with respect to one particular parameter while considering the others as temporarily fixed?

For example:

```text
∂J/∂w1
```

means:

> How sensitive is the cost to a change in `w1`?

Similarly:

```text
∂J/∂b
```

means:

> How sensitive is the cost to a change in the bias?

This allows the optimizer to know how each trainable parameter contributes to movement in the objective.

---

# 25. Chain Rule — intuition

The weight does not directly "touch" the cost.

There is a chain:

```text
weight
  ↓
prediction
  ↓
error
  ↓
cost
```

So to understand:

> How does changing the weight affect the final cost?

we follow the dependency chain.

That is what the **chain rule** allows mathematically.

Conceptually:

```text
change weight
   ↓
changes prediction
   ↓
changes error
   ↓
changes cost
```

This same idea later becomes extremely important in Neural Networks, where the chain may contain many layers. Backpropagation uses the chain rule to calculate gradients through those layers.

---

# 26. Convex versus Non-Convex functions

Gradient Descent is easier to understand when we look at the shape of the objective function.

## Convex function

Think of one clean bowl:

```text
Cost
 ↑
 |       \       /
 |        \     /
 |         \   /
 |          \_/
 +----------------→ parameters
```

There is one global basin.

For ordinary Linear Regression with MSE, the optimization problem is convex.

With a suitable optimization setup and learning rate, Gradient Descent can move toward the **global minimum**.

## Non-convex function

A non-convex landscape can contain several valleys and hills:

```text
Cost
 ↑
 |     /\          /\
 |    /  \   __   /  \
 |___/    \_/  \_/    \___
 +--------------------------→ parameters
```

It may contain:

```text
local minima
global minimum
saddle regions
peaks
```

## Local minimum

A point that is lower than nearby points.

## Global minimum

The lowest point of the complete objective landscape.

Memory rule:

```text
Linear Regression + MSE
→ convex
→ optimization is comparatively easier

Deep Neural Network
→ generally non-convex
→ optimization is more complicated
```

---

# 27. What does "minimum cost" really mean?

The minimum does **not** necessarily mean:

```text
prediction = actual value for every training observation
```

Real data contains noise and the model may not perfectly represent the true process.

The optimization goal is:

> Find parameters that minimize the chosen overall objective function.

So a trained model can still have individual prediction errors even when its cost has been minimized.

---

# 28. Polynomial Regression

Ordinary Linear Regression may be too rigid when the relationship is curved.

Standard model:

```text
y = b + w1*x
```

Polynomial Regression adds transformed features such as:

```text
x²
x³
...
```

Example:

```text
y = w0 + w1*x + w2*x²
```

This allows curved relationships to be represented.

Important exam point:

> Polynomial Regression can model a nonlinear relationship in `x`, while still being linear in its parameters `w0`, `w1`, `w2`, ...

So it still belongs to the family of linear models in the parameter-estimation sense.

---

# 29. Why regularization is needed

A very flexible model can start fitting random noise in the training data.

That is **overfitting**.

```text
Training performance → excellent
New-data performance → poor
```

Regularization adds a penalty for model complexity, especially large coefficients.

General intuition:

```text
Fit the data
+
penalize unnecessarily extreme coefficients
=
more controlled model
```

---

# 30. Ridge Regression — L2 regularization

Ridge adds a squared-coefficient penalty.

Conceptually:

```text
Cost
=
prediction error
+
λ * sum of squared weights
```

A common form is:

```text
J_ridge = MSE + λ Σ w_j²
```

where `λ` controls the strength of regularization.

Effect:

```text
large weights
   ↓
large penalty
   ↓
optimizer prefers smaller coefficients
```

Ridge usually **shrinks coefficients toward zero** but does not normally force them exactly to zero.

Ridge is useful when predictors are correlated and when we want to reduce overfitting without removing features entirely.

---

# 31. Lasso Regression — L1 regularization

Lasso uses an absolute-value penalty:

```text
J_lasso = MSE + λ Σ |w_j|
```

The important practical effect is:

> Lasso can force some coefficients exactly to zero.

Example:

```text
age       → weight = 4.2
salary    → weight = 7.1
shoe_size → weight = 0
```

A zero coefficient means that feature no longer contributes to the prediction in the fitted model.

Therefore Lasso can perform a form of **feature selection**.

---

# 32. Elastic Net

Elastic Net combines the ideas of Ridge and Lasso.

```text
Elastic Net
=
L1 penalty
+
L2 penalty
```

It can:

- shrink coefficients,
- set some coefficients to zero,
- behave more stably than pure Lasso when predictors are highly correlated.

---

# 33. What does lambda `λ` do?

`λ` controls how strongly the model is regularized.

```text
small λ
→ weak penalty
→ model has more freedom

large λ
→ strong penalty
→ coefficients are pushed smaller
```

Too little regularization may leave overfitting uncontrolled.

Too much regularization may make the model too simple and cause **underfitting**.

So regularization also involves a bias-complexity trade-off.

---

# 34. Linear vs Polynomial vs Ridge vs Lasso vs Elastic Net

```text
LINEAR REGRESSION
→ weighted linear relationship
→ continuous target

POLYNOMIAL REGRESSION
→ adds x², x³, ...
→ captures curved relationships

RIDGE
→ L2 penalty
→ shrinks coefficients
→ usually keeps all features

LASSO
→ L1 penalty
→ can make coefficients exactly zero
→ can perform feature selection

ELASTIC NET
→ combines L1 + L2
→ useful when predictors are numerous or correlated
```

A strong conceptual sentence:

> Polynomial Regression increases flexibility; Ridge, Lasso, and Elastic Net control model complexity.

---

# 35. Applications of Linear Regression

Typical applications include:

## Real estate

Predict house prices from:

```text
size
location
number of bedrooms
```

## Salary estimation

Predict salary from:

```text
experience
education
job-related attributes
```

## Sales forecasting

Predict sales from:

```text
advertising spend
seasonality-related features
market information
```

## Risk and insurance

Predict numerical claim amounts from customer or vehicle characteristics.

## Agriculture

Predict crop yield from:

```text
rainfall
fertilizer use
temperature
```

## Education analytics

Predict a numerical performance measure from attendance, study hours, and related variables.

---

# 36. Complete Linear Regression learning workflow

The entire process can now be connected:

```text
1. Collect historical data
        ↓
2. Choose features X and continuous target y
        ↓
3. Define model
   y_hat = Xw + b
        ↓
4. Initialize / estimate weights and bias
        ↓
5. Forward prediction
        ↓
6. Compare prediction with actual target
        ↓
7. Calculate loss / MSE cost
        ↓
8. Calculate gradients
        ↓
9. Gradient Descent updates weights and bias
        ↓
10. Repeat until convergence
        ↓
11. Use learned parameters on new data
```

---

# 37. Gradient Descent in one complete example

Imagine a one-feature model:

```text
y_hat = w*x + b
```

Suppose the current parameters produce poor predictions.

```text
Current model
w = 10
b = 50

Cost = very high
```

Training performs:

```text
Make predictions
      ↓
Calculate MSE
      ↓
Calculate ∂J/∂w and ∂J/∂b
      ↓
Use learning rate α
      ↓
Update w and b
      ↓
Make new predictions
      ↓
Calculate new MSE
      ↓
Repeat
```

A possible conceptual progression might be:

```text
Iteration 1 → high cost
Iteration 2 → lower cost
Iteration 3 → lower cost
...
Convergence → gradient approximately zero
```

The exact numerical path depends on the dataset, initialization, scaling, and learning rate.

---

# 38. Minimal scikit-learn example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# One feature: house area
X = np.array([[50], [70], [90], [110], [130]])

# Continuous target: house price in arbitrary units
y = np.array([150, 190, 230, 270, 310])

model = LinearRegression()
model.fit(X, y)

print("Weight:", model.coef_[0])
print("Bias:", model.intercept_)

new_house = np.array([[100]])
prediction = model.predict(new_house)
print("Prediction for area=100:", prediction[0])
```

What happens conceptually:

```text
model.fit(...)
→ estimates the parameters

model.coef_
→ learned weight(s)

model.intercept_
→ learned bias

model.predict(...)
→ uses the learned model on new data
```

---

# 39. Gradient Descent from scratch — learning code

This version is intentionally simple so the optimization idea is visible.

```python
import numpy as np

X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 5.0, 7.0, 9.0])

w = 0.0
b = 0.0
learning_rate = 0.01
n = len(X)

for _ in range(2000):
    # Forward prediction
    y_hat = w * X + b

    # Error
    error = y_hat - y

    # Gradients of MSE
    dw = (2 / n) * np.sum(error * X)
    db = (2 / n) * np.sum(error)

    # Gradient Descent update
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Learned weight:", w)
print("Learned bias:", b)
```

This dataset follows approximately:

```text
y = 2x + 1
```

so after enough stable updates we expect the learned values to approach:

```text
w ≈ 2
b ≈ 1
```

The purpose of this code is not to replace optimized libraries. It is to make the training logic visible.

---

# 40. Read the from-scratch code like a story

```python
y_hat = w * X + b
```

means:

> Make predictions using the current parameters.

```python
error = y_hat - y
```

means:

> Compare predictions with the actual targets.

```python
dw = ...
db = ...
```

means:

> Calculate how the cost changes with respect to the weight and bias.

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

means:

> Move the parameters in the direction that reduces the cost.

That is Gradient Descent.

---

# 41. Common misunderstandings

## "Bias removes the error"

```text
Wrong ❌
```

Bias is the intercept / baseline of the prediction.

## "Gradient is the error"

```text
Wrong ❌
```

Cost/loss measures how wrong the model is.

Gradient tells us how the cost changes when parameters change.

## "The bottom of the parabola is the actual prediction"

```text
Wrong ❌
```

The horizontal axis of the simplified cost graph represents parameter values.

The bottom represents the parameter value giving minimum cost.

## "Gradient Descent randomly tries every parameter"

```text
Wrong ❌
```

Gradient Descent uses local slope information to choose the update direction.

## "A minimum means zero error on every sample"

```text
Wrong ❌
```

It means the chosen objective has been minimized according to the optimization process.

## "A large coefficient always means the feature is most important"

```text
Not necessarily ❌
```

Coefficient magnitude depends on scaling, units, correlation, and model specification.

---

# 42. Exam-style questions and answers

## Q1. What is Linear Regression?

Linear Regression models the relationship between one or more input features and a continuous target using a weighted linear combination of the inputs plus an intercept.

## Q2. What are the parameters?

The trainable parameters are the **weights / coefficients** and the **bias / intercept**.

```text
θ = {w, b}
```

## Q3. What is fitting?

Fitting means finding parameter values that minimize an objective function while producing a useful model.

## Q4. Difference between loss and cost?

```text
Loss → error for one observation
Cost/objective → summarized error over the dataset
```

## Q5. Why does MSE create a parabola/paraboloid?

Because MSE is quadratic in the parameters for Linear Regression.

## Q6. Why square the errors?

Because squaring prevents positive and negative errors from cancelling and penalizes large errors more strongly.

## Q7. Why not brute-force with `np.min()`?

Because model parameters live in a continuous, potentially high-dimensional space. `np.min()` only chooses the smallest value from already computed candidates; it does not efficiently discover optimal parameter combinations.

## Q8. What is the gradient?

The gradient is the collection of derivatives of the objective with respect to the parameters. It points in the direction of steepest increase and its magnitude represents steepness.

## Q9. What is Gradient Descent?

An iterative optimization algorithm that updates parameters in the opposite direction of the gradient to reduce the objective function.

## Q10. What does the learning rate do?

It controls the size of each parameter update.

```text
Too large → overshoot / instability
Too small → very slow convergence
```

## Q11. What is a partial derivative?

It measures how the objective changes with respect to one particular parameter.

## Q12. Why use the chain rule?

Because a parameter affects the final cost through intermediate calculations such as prediction and error. The chain rule connects those dependencies.

## Q13. Why is convexity useful?

For a convex Linear Regression objective such as MSE, there is no worse local minimum hiding above a better one; with a suitable optimization setup, reaching the minimum means reaching the global optimum of that objective.

## Q14. Difference between Ridge and Lasso?

```text
Ridge → L2 penalty → shrinks coefficients
Lasso → L1 penalty → can set coefficients exactly to zero
```

## Q15. What is Elastic Net?

A regularized regression method combining L1 and L2 penalties.

---

# 43. ⭐ CHEAT SHEET

```text
LINEAR REGRESSION
→ predicts continuous values

MODEL
→ y_hat = Xw + b

FEATURES
→ input variables X

TARGET
→ continuous output y

WEIGHTS
→ determine how features contribute to prediction

BIAS
→ intercept / baseline
→ shifts prediction surface
→ does NOT remove outliers or errors

TRAINING / FITTING
→ find weights + bias that minimize the objective

LOSS
→ error for one prediction

COST / OBJECTIVE
→ overall error over the training data

MSE
→ mean of squared prediction errors
→ prevents sign cancellation
→ strongly penalizes large errors

PARABOLA / PARABOLOID
→ caused by quadratic MSE structure
→ x-axis = parameter(s)
→ y-axis = cost
→ bottom = minimum-cost parameter value

GRADIENT
→ direction of steepest increase
→ magnitude = steepness

GRADIENT DESCENT
→ move opposite to gradient
→ repeatedly reduce cost

UPDATE
→ new parameter = old parameter - learning rate × gradient

LEARNING RATE α
→ step size
→ too large = overshoot / unstable
→ too small = slow

DERIVATIVE
→ slope for one variable

GRADIENT
→ collection/vector of partial derivatives

PARTIAL DERIVATIVE
→ how cost changes with respect to one parameter

CHAIN RULE
→ follows parameter → prediction → error → cost
→ later becomes central to backpropagation

CONVEX
→ one global basin
→ Linear Regression + MSE is convex

NON-CONVEX
→ multiple valleys / complex landscape
→ common in deep neural networks

POLYNOMIAL REGRESSION
→ adds x², x³, ...
→ captures curved relationships
→ still linear in parameters

RIDGE
→ L2 regularization
→ shrinks coefficients

LASSO
→ L1 regularization
→ may set some coefficients to zero

ELASTIC NET
→ L1 + L2

FULL TRAINING FLOW
X
↓
y_hat = Xw + b
↓
compare with y
↓
MSE
↓
gradients
↓
Gradient Descent
↓
update w and b
↓
repeat until convergence
```

---

# Final mental model

Do not remember Linear Regression as only:

```text
y = mx + b
```

Remember it as a complete learning system:

```text
Features
   ↓
Weighted combination + bias
   ↓
Prediction
   ↓
Compare with reality
   ↓
Measure cost
   ↓
Gradient tells us how parameters affect that cost
   ↓
Gradient Descent updates the parameters
   ↓
Repeat until a good minimum is reached
```

And the most important Gradient Descent memory sentence is:

> **Cost tells me how wrong I am. The gradient tells me which way the cost changes. Gradient Descent uses that information to move the parameters toward lower cost.**
