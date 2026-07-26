"""Beginner-friendly Neural Network example on CIFAR-10.

The script follows the learning flow used in the course:

    load images -> scale -> encode labels -> build model -> train -> evaluate

CIFAR-10 is downloaded automatically by TensorFlow/Keras the first time the
script is run.
"""

import numpy as np
import matplotlib.pyplot as plt
import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# -----------------------------------------------------------------------------
# 1. Load CIFAR-10
# -----------------------------------------------------------------------------
# Each image is 32 x 32 pixels with 3 colour channels.
# There are 10 possible classes.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# CIFAR labels arrive as shape (n, 1). Flatten makes them simple class numbers.
y_train = y_train.flatten()
y_test = y_test.flatten()


# -----------------------------------------------------------------------------
# 2. Scale pixel values
# -----------------------------------------------------------------------------
# Raw pixels are 0-255. Dividing by 255 places them in the 0-1 range.

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# -----------------------------------------------------------------------------
# 3. One-hot encode the labels
# -----------------------------------------------------------------------------
# Example: class 2 becomes [0, 0, 1, 0, ..., 0].
# We do this because the model below uses categorical_crossentropy.

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)


# -----------------------------------------------------------------------------
# 4. Build the Neural Network
# -----------------------------------------------------------------------------
# Flatten: convert a 32x32x3 image into one long vector of numbers.
# Dense(128, relu): hidden layer learns useful combinations of pixel signals.
# Dense(10, softmax): output one probability for each class.

model = Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)


# -----------------------------------------------------------------------------
# 5. Tell the model how to learn
# -----------------------------------------------------------------------------
# SGD = optimizer that updates weights using gradients from backpropagation.
# learning_rate = size of each update step.
# categorical_crossentropy = loss for one-hot multiclass classification.

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# -----------------------------------------------------------------------------
# 6. Train the model
# -----------------------------------------------------------------------------
# In a real project, use a separate validation set and leave the test set
# untouched until final evaluation. validation_split keeps this example simple.

history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.20,
    epochs=20,
    batch_size=64,
    verbose=1,
)


# -----------------------------------------------------------------------------
# 7. Plot training and validation loss
# -----------------------------------------------------------------------------
# If training loss keeps improving while validation loss gets worse, that is a
# warning sign for overfitting.

plt.figure()
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Neural Network Training")
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 8. Evaluate on unseen test data
# -----------------------------------------------------------------------------

probabilities = model.predict(X_test, verbose=0)

# Softmax returns probabilities. argmax picks the class with highest probability.
predicted_classes = np.argmax(probabilities, axis=1)

print("Classification report:")
print(classification_report(y_test, predicted_classes))

ConfusionMatrixDisplay.from_predictions(y_test, predicted_classes)
plt.title("Neural Network Confusion Matrix")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 9. Compare learning rates
# -----------------------------------------------------------------------------
# Rebuild the model for every experiment so every learning rate starts with
# fresh weights. Otherwise the comparison would be unfair.


def build_model() -> Sequential:
    """Return a fresh copy of the same network architecture."""
    return Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


print("\nLearning-rate comparison:")

# Keep these experiments short so the script remains practical to run.
for learning_rate in [0.001, 0.01, 0.1]:
    experiment = build_model()

    experiment.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    result = experiment.fit(
        X_train,
        y_train_cat,
        validation_split=0.20,
        epochs=5,
        batch_size=64,
        verbose=0,
    )

    print(
        f"LR={learning_rate:<5} "
        f"train_accuracy={result.history['accuracy'][-1]:.3f} "
        f"validation_accuracy={result.history['val_accuracy'][-1]:.3f}"
    )


# -----------------------------------------------------------------------------
# Human memory map
# -----------------------------------------------------------------------------
# Flatten               = image -> vector
# Dense                  = fully connected layer
# ReLU                   = hidden-layer non-linearity
# Softmax                = multiclass probabilities
# categorical_crossentropy = multiclass loss
# forward propagation    = calculate prediction
# backpropagation        = calculate gradients
# SGD                    = use gradients to update weights
# learning rate          = update step size
# epoch                  = one full pass through training data
# batch size             = examples processed before an update
# argmax                 = probability vector -> final class number
