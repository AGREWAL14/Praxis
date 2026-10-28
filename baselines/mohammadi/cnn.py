"""The architecture from Mohammadi et al. (2024), arXiv:2410.23306.

Reproduced as published. Not tuned, not corrected, not improved.

Two properties of this model matter more than the layer sizes and are kept
deliberately.

The input is reshaped to `(samples, features, 1)`. The convolution therefore
slides along the feature axis of a single record, not along time, and the model
never sees two records at once. Whatever it learns about the ordering of the
columns is an artefact of the order the columns happen to be in.

Training runs for a fixed ten epochs at batch 32 with no class weighting, no
validation split and no early stopping. On a corpus with a 2,157:1 imbalance the
loss is dominated by the largest classes throughout, and nothing stops training
when the rare ones stop improving.
"""

from __future__ import annotations

ARCHITECTURE = [
    "Reshape input to (samples, features, 1)",
    "Conv1D, 32 filters, kernel 3, ReLU",
    "MaxPooling1D, pool 2",
    "Conv1D, 64 filters, kernel 3, ReLU",
    "MaxPooling1D, pool 2",
    "Flatten",
    "Dense 128, ReLU",
    "Dense n_classes, softmax",
]

COMPILE = {
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
}

FIT = {
    "batch_size": 32,
    "epochs": 10,
    "shuffle": True,
    "class_weight": None,
    "validation_split": 0.0,
    "early_stopping": False,
}


def reshape(X):
    """`(samples, features)` to `(samples, features, 1)`.

    A reshape and not a copy, so an array of several million rows is not
    duplicated to be fed to the model.
    """
    return X.reshape((X.shape[0], X.shape[1], 1))


def build_model(n_features: int, n_classes: int):
    """The network, compiled and ready to fit.

    The caller seeds before calling this. Nothing is seeded here, because a
    function that reseeds on its own hides where the randomness entered.
    """
    import keras
    from keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(int(n_features), 1)),
            layers.Conv1D(32, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(int(n_classes), activation="softmax"),
        ]
    )
    model.compile(**COMPILE)
    return model


def fit(model, X, y_codes, callbacks=(), *, n_classes: int, verbose: int = 2):
    """Ten epochs at batch 32, as published.

    `y_codes` are integer class positions and are turned into the one-hot targets
    the categorical cross-entropy expects. The one-hot matrix is built once here
    rather than by the caller, so no run can accidentally train against a
    different encoding of the same labels.

    `callbacks` exists so the notebook can attach a per-epoch checkpoint. A
    callback observes training and does not change it, and nothing that would
    stop or reweight training is ever passed.
    """
    import keras
    import numpy as np

    targets = keras.utils.to_categorical(
        np.asarray(y_codes).astype("int64"), num_classes=int(n_classes)
    )
    return model.fit(
        X,
        targets,
        epochs=FIT["epochs"],
        batch_size=FIT["batch_size"],
        shuffle=FIT["shuffle"],
        class_weight=FIT["class_weight"],
        callbacks=list(callbacks),
        verbose=verbose,
    )


def describe(n_features: int, n_classes: int) -> dict:
    """The model as plain data, for the run's config file."""
    return {
        "source": "Mohammadi et al. (2024), arXiv:2410.23306",
        "architecture": list(ARCHITECTURE),
        "input_shape": [int(n_features), 1],
        "n_classes": int(n_classes),
        "compile": dict(COMPILE),
        "fit": dict(FIT),
    }
