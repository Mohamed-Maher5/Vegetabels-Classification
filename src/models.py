"""Model definition.

`build_vegetable_cnn()` reproduces the exact architecture of the trained model
saved in `models/vegetable_cnn.h5` (VGG-style stack, 150x150 RGB input):

    Conv(32, 5x5)  -> Conv(32, 5x5)  -> BatchNorm -> MaxPool 2x2 -> Dropout(0.25)
    Conv(64, 3x3)  -> Conv(64, 3x3)  -> BatchNorm -> MaxPool 2x2 -> Dropout(0.25)
    Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(num_classes, softmax)

Keeping the definition here (instead of only inside notebooks) means the
architecture notebook, the Kaggle training notebook and any retraining script
all share one source of truth.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from .config import IMG_SIZE, NUM_CLASSES


def build_vegetable_cnn(
    input_shape: "tuple[int, int, int]" = IMG_SIZE + (3,),
    num_classes: int = NUM_CLASSES,
) -> models.Model:
    """Build the vegetable classification CNN (VGG-style from-scratch)."""
    model = models.Sequential(
        [
            layers.Conv2D(32, (5, 5), activation="relu", input_shape=input_shape),
            layers.Conv2D(32, (5, 5), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="vegetable_cnn",
    )
    return model


def compile_vegetable_cnn(model: models.Model) -> models.Model:
    """Compile the model with the training recipe used on Kaggle."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model