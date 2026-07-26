# Neural Networks

This section explains neural networks .

The goal is not to begin with intimidating matrix notation. The goal is to understand what the network is trying to do, why hidden layers exist, how learning happens, and how the code connects to the theory.

---

# 1. Start with a normal-life analogy

Imagine you are deciding whether a photo contains a cat.

You might subconsciously look at several clues:

- ear shape,
- eyes,
- fur texture,
- whiskers,
- face shape.

You do not treat every clue equally. Some clues matter more than others.

A neural network follows a similar idea:

```text
Input information
      ↓
Give different importance to different signals
      ↓
Combine them
      ↓
Transform the result
      ↓
Pass it to the next layer
      ↓
Final prediction
```

The "importance" values are the **weights** learned during training.

---

# 2. The artificial neuron

A simple neuron receives inputs:

```text
x1, x2, x3, ...
```

Each input is multiplied by a weight:

```text
w1*x1 + w2*x2 + w3*x3 + ...
```

Then a bias is added:

```text
z = w1*x1 + w2*x2 + ... + b
```

Finally an activation function transforms `z`:

```text
output = activation(z)
```

Human interpretation:

```text
input      = evidence
weight     = importance of that evidence
bias       = baseline adjustment
activation = transformation / decision behaviour
```

---

# 3. Why does this look like regression?

Linear Regression also calculates:

```text
y_hat = w1*x1 + w2*x2 + ... + b
```

So a neuron begins with the same weighted-sum idea.

The big difference is that neural networks usually add **non-linear activation functions** and connect many neurons together.

That allows the model to learn much more complicated patterns than one straight line or one linear decision boundary.

---

# 4. Why do we need hidden layers?

Suppose we want to recognise a face.

A network does not need to jump directly from raw pixels to "this is a face".

Different layers can gradually build more useful representations:

```text
Pixels
  ↓
Edges
  ↓
Simple shapes
  ↓
Eyes / nose / mouth-like patterns
  ↓
Face-level representation
  ↓
Prediction
```

This is called **hierarchical feature learning**.

A hidden layer is simply an intermediate processing layer between input and output.

```text
Input layer
    ↓
Hidden layer(s)
    ↓
Output layer
```

---

# 5. Why activation functions matter

Without non-linear activations, stacking many linear layers still behaves like one large linear transformation.

So depth alone would not give the network much extra expressive power.

Non-linear activations let the model learn bends, interactions, thresholds, and complex patterns.

Common choices:

```text
Hidden layers
→ ReLU

Binary classification output
→ Sigmoid

Multiclass classification output
→ Softmax

Regression output
→ Linear / no special activation
```

---

# 6. ReLU

ReLU means Rectified Linear Unit.

```text
ReLU(z) = max(0, z)
```

In plain language:

```text
negative value → 0
positive value → keep it
```

It is widely used in hidden layers because it is simple and works well in many deep-learning problems.

---

# 7. Sigmoid

Sigmoid turns a number into a value between 0 and 1.

That makes it useful for binary classification probabilities.

```text
large negative z → probability near 0
z around 0       → probability around 0.5
large positive z → probability near 1
```

Example:

```text
0.91 → likely class 1
0.08 → likely class 0
```

---

# 8. Softmax

Softmax is useful when there are several possible classes.

For example, image classification might have:

```text
cat   = 0.70
dog   = 0.20
horse = 0.07
bird  = 0.03
```

The probabilities add to approximately 1.

The class with the highest probability is usually selected:

```python
predicted_class = np.argmax(probabilities)
```

---

# 9. Forward propagation

Forward propagation means:

> Send the input through the network to produce a prediction.

For a small network:

```text
Input
  ↓
Weighted sum + bias
  ↓
Activation
  ↓
Hidden representation
  ↓
Weighted sum + bias
  ↓
Output activation
  ↓
Prediction
```

A two-layer notation might look like:

```text
z[1] = W[1]x + b[1]
a[1] = g(z[1])

z[2] = W[2]a[1] + b[2]
a[2] = g(z[2])
```

Important:

> Forward propagation calculates the prediction. It does not itself update the weights.

---

# 10. Loss: how wrong was the prediction?

After the network predicts, we need to measure the error.

The loss function provides that signal.

Typical examples:

```text
Regression
→ Mean Squared Error

Binary classification
→ Binary Cross-Entropy

Multiclass classification
→ Categorical Cross-Entropy
```

The loss answers:

> How bad was this prediction according to the objective we care about?

---

# 11. Backpropagation

Backpropagation is one of the most important ideas in neural networks.

Normal-human explanation:

Imagine a company produces a bad final product.

Management asks:

> Which earlier decisions contributed to the final mistake, and by how much?

Backpropagation performs a similar responsibility calculation through the network.

```text
Prediction
   ↓
Loss
   ↓
Work backward through the network
   ↓
Calculate how each weight influenced the loss
   ↓
Produce gradients
```

Backpropagation uses the **chain rule** to move the error signal backward through the layers.

Important distinction:

```text
Forward propagation
→ prediction

Backpropagation
→ gradients / responsibility

Optimizer
→ actually updates weights
```

---

# 12. Gradient Descent

A gradient tells us how the loss changes when a parameter changes.

Gradient Descent uses that information to update parameters in the direction that should reduce the loss.

```text
new parameter
=
old parameter - learning_rate * gradient
```

Human analogy:

Imagine walking down a hill in fog.

You cannot see the entire landscape, but you can feel the local slope.

You repeatedly take a step downhill.

That is the intuition behind Gradient Descent.

---

# 13. Learning rate

The learning rate controls the size of the update step.

```text
Too small
→ learning can be painfully slow

Too large
→ updates can overshoot and training may become unstable

Reasonable value
→ steady progress toward a useful solution
```

Learning-rate schedules can reduce the step size later in training so the model can make finer adjustments.

---

# 14. Batch, stochastic and mini-batch training

## Batch Gradient Descent

Uses the whole training dataset for an update.

```text
stable but potentially expensive
```

## Stochastic Gradient Descent

Uses one training example at a time.

```text
fast updates but noisy
```

## Mini-Batch Gradient Descent

Uses a small group of examples at a time.

```text
practical compromise
```

Modern neural-network training usually relies on mini-batches.

---

# 15. Adam optimizer

Adam is a widely used optimizer.

At a high level, Adam combines ideas similar to:

- momentum from previous gradients,
- adaptive step sizes for different parameters.

Important:

> Adam does not replace backpropagation.

The relationship is:

```text
Backpropagation
→ calculates gradients

Adam
→ uses those gradients to update the weights
```

---

# 16. Epoch and batch size

## Epoch

One epoch means the training process has gone through the entire training dataset once.

```text
50 epochs
→ the model gets 50 full passes through the training set
```

## Batch size

Batch size is how many training examples are processed before one parameter update.

Example:

```text
batch_size = 64
```

means the model processes 64 examples, calculates gradients, updates parameters, then continues with the next batch.

---

# 17. Why normalise the inputs?

Suppose one feature is:

```text
0 to 1
```

and another is:

```text
0 to 100,000
```

Optimisation can become harder because the scales are dramatically different.

Normalisation / standardisation can make training more stable and efficient.

A common standardisation formula is:

```text
z = (x - mean) / standard_deviation
```

For image pixels stored from 0 to 255, a common simple scaling step is:

```python
X = X.astype("float32") / 255.0
```

which places values roughly between 0 and 1.

---

# 18. Overfitting

Overfitting means:

```text
Training performance   → excellent
Validation/test result → poor
```

The model has learned the training data too specifically and does not generalise well.

Common tools to reduce overfitting include:

- dropout,
- L1/L2 regularisation,
- early stopping,
- more representative training data,
- simpler architecture.

---

# 19. Dropout

During training, dropout temporarily disables a random fraction of neuron outputs.

The idea is to stop the network from depending too heavily on a small set of neurons.

Conceptually:

```text
Normal training
A B C D E

One dropout step
A - C - E
```

Different neurons are dropped on different steps.

---

# 20. Early stopping

During training we monitor validation performance.

If training loss keeps improving but validation performance stops improving or gets worse, we may stop training.

```text
training improves
validation improves
        ↓
continue

training improves
validation worsens
        ↓
possible overfitting
        ↓
stop
```

---

# 21. Vanishing and exploding gradients

Deep networks multiply many derivative terms during backpropagation.

If these values repeatedly become very small:

```text
Gradient
→ tiny
→ earlier layers learn extremely slowly
```

This is the **vanishing-gradient problem**.

If they repeatedly become very large:

```text
Gradient
→ huge
→ unstable updates
```

This is the **exploding-gradient problem**.

Activation choice, initialisation, normalisation and modern architecture design help manage these problems.

---

# 22. Saddle points and difficult optimisation

A saddle point is a location where the surface may look flat in some directions but curve upward or downward in others.

So a near-zero gradient does not always mean we found the best possible solution.

Deep-network optimisation is difficult because the loss surface can contain:

- flat regions,
- saddle points,
- many interacting parameters,
- noisy gradients.

Modern optimizers help navigate these surfaces more effectively.

---

# 23. How a neural network is built in Keras

A simple image classifier can look like this:

```python
model = Sequential([
    keras.Input(shape=(32, 32, 3)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
```

Human interpretation:

```text
32×32 colour image
      ↓
Flatten pixels into numbers
      ↓
128 hidden neurons learn useful combinations
      ↓
10 output probabilities
      ↓
choose most likely class
```

---

# 24. Compile, fit and predict

## compile()

Defines how the network should learn.

```python
model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

## fit()

Runs the training process.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64
)
```

## predict()

Uses the trained network on new data.

```python
probabilities = model.predict(X_test)
```

---

# 25. Complete neural-network workflow

```text
Collect data
    ↓
Train / validation / test split
    ↓
Scale / normalise inputs
    ↓
Choose architecture
    ↓
Initialise weights
    ↓
Forward propagation
    ↓
Calculate loss
    ↓
Backpropagation
    ↓
Optimizer updates weights
    ↓
Repeat over mini-batches and epochs
    ↓
Monitor validation performance
    ↓
Regularise / tune hyperparameters
    ↓
Evaluate on unseen test data
```

---

# 26. Parameters vs hyperparameters

## Parameters

Learned automatically from data:

```text
weights
biases
```

## Hyperparameters

Chosen or tuned by us:

```text
learning rate
number of hidden layers
neurons per layer
batch size
epochs
dropout rate
optimizer
```

Memory rule:

```text
Parameters      → model learns them
Hyperparameters → we configure/tune them
```

---

# 27. Practical CIFAR-10 example

The accompanying script [`neural_network_cifar10.py`](./neural_network_cifar10.py) demonstrates:

- loading CIFAR-10 images,
- flattening labels,
- pixel scaling,
- one-hot encoding,
- a hidden ReLU layer,
- Softmax output,
- SGD optimisation,
- categorical cross-entropy,
- train/validation loss curves,
- classification report,
- confusion matrix,
- learning-rate comparison.

This mirrors the main practical ideas used in the course exercise while keeping the script readable from top to bottom.

---

# 28. Quick memory sheet

```text
Neuron
→ weighted sum + bias + activation

Hidden layer
→ learns intermediate representations

ReLU
→ common hidden activation

Sigmoid
→ binary probability

Softmax
→ multiclass probabilities

Forward propagation
→ calculate prediction

Loss
→ measure how wrong prediction is

Backpropagation
→ calculate gradients

Optimizer
→ update weights using gradients

Learning rate
→ update step size

Epoch
→ one full pass through training data

Batch size
→ examples per update

Dropout / early stopping
→ reduce overfitting

Normalisation
→ make optimisation easier
```

---

# Final Explanation

A neural network is not magic.

At its core it repeatedly does this:

```text
Take numbers
→ multiply by learned importance values
→ combine them
→ transform them
→ compare prediction with reality
→ calculate who contributed to the error
→ adjust the importance values
→ repeat many times
```

The power comes from connecting many of these simple operations into layers so that the model can gradually learn useful representations from data.
