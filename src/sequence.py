"""The sequence model: the reproduced record encoder, read across a window.

The convolutional network in `baselines/mohammadi/` takes one record at a time.
Its input is `(features, 1)`, its convolutions slide along the feature axis, and
it has no way of seeing what came before the record it is looking at. Everything
in this module is built around that network rather than in place of it.

`record_encoder` takes the published model as the baseline builds it and removes
only its final softmax layer, keeping the two convolutions, the two pooling
layers, the flatten and the dense 128 exactly as published. What is left turns
one record into 128 numbers. `build_model` applies that same encoder to each of
the fifty records in a window, runs one LSTM across the fifty results, and puts a
softmax over the classes on the end.

Nothing under `baselines/` is imported for anything but this, and nothing under
`baselines/` is changed. The published head is dropped rather than retrained,
because a classification head that read a single record is not the head this
model needs, and the encoder underneath it is the part the comparison is about.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from baselines import mohammadi as mo
from src import runs as rn

# The width of the encoder's output, so the layer that reads the window is the
# same width as the representation it reads. Recorded in every run's config.
LSTM_UNITS = 128


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def record_encoder(n_features: int, n_classes: int):
    """The published network with its classification layer taken off.

    The model is built by the baseline's own `build_model` and its layers are
    then reused, so the encoder cannot drift from the published architecture:
    changing the baseline would change this, and the baseline is pinned.

    `n_classes` is still required because the baseline builds the head before it
    can be dropped. It has no effect on what comes back.
    """
    import keras

    published = mo.build_model(int(n_features), int(n_classes))
    kept = list(published.layers)[:-1]
    encoder = keras.Sequential(
        [keras.Input(shape=(int(n_features), 1)), *kept], name="mohammadi_record_encoder"
    )
    return encoder


def build_model(
    n_features: int, n_classes: int, *, window, lstm_units: int = LSTM_UNITS, loss=None
):
    """The encoder over each record of the window, one LSTM across it, then softmax.

    The caller seeds before calling this. Nothing is seeded here, for the same
    reason the baseline seeds nothing: a function that reseeds on its own hides
    where the randomness entered.

    Compiled with the baseline's optimizer and loss, unchanged, so the only thing
    that differs between this and a single-record run is the shape of what goes in
    and the layer that reads across it.

    `loss` replaces the baseline's cross-entropy for a run whose one change is the
    loss. Left at None the compile settings are the baseline's exactly, so a run
    that is not about the loss cannot differ from its parent in it by accident.

    `window` at None builds the model over an unfixed number of records instead of
    a fixed fifty. The encoder reads one record at a time and the LSTM reads across
    however many it is given, so neither holds a weight that depends on the length,
    and a model built this way has exactly the parameter count a model built at a
    fixed length has. What it allows is one model asked for a prediction after five
    records and again after fifty. Every batch still has to hold windows of one
    length, because an array has one shape.
    """
    import keras
    from keras import layers

    encoder = record_encoder(int(n_features), int(n_classes))
    model = keras.Sequential(
        [
            keras.Input(
                shape=(None if window is None else int(window), int(n_features), 1)
            ),
            layers.TimeDistributed(encoder, name="per_record"),
            layers.LSTM(int(lstm_units), name="across_the_window"),
            layers.Dense(int(n_classes), activation="softmax", name="classifier"),
        ],
        name="mohammadi_cnn_lstm",
    )
    settings = dict(mo.COMPILE)
    if loss is not None:
        settings["loss"] = loss
    model.compile(**settings)
    return model


def fit_model(model, X, y_codes, callbacks, *, n_classes: int, class_weight=None, verbose: int = 2):
    """The baseline's ten epochs at batch 32, with an optional class weighting.

    With no weighting this is the baseline's own `fit` and nothing else, so the
    published training procedure runs the published way. With a weighting the
    baseline's function cannot be used, because it fixes `class_weight` at None as
    published and nothing under `baselines/` is edited. The settings are then read
    off the baseline's `FIT` rather than written out again, so the two paths
    cannot drift apart in the number of epochs or the size of a batch.
    """
    if class_weight is None:
        return mo.fit(model, X, y_codes, callbacks, n_classes=int(n_classes), verbose=verbose)

    import keras

    targets = keras.utils.to_categorical(
        np.asarray(y_codes).astype("int64"), num_classes=int(n_classes)
    )
    return model.fit(
        X,
        targets,
        epochs=mo.FIT["epochs"],
        batch_size=mo.FIT["batch_size"],
        shuffle=mo.FIT["shuffle"],
        class_weight={int(code): float(weight) for code, weight in class_weight.items()},
        callbacks=list(callbacks),
        verbose=verbose,
    )


def fit_group_dro(
    model,
    X,
    y_codes,
    groups,
    callbacks,
    *,
    n_classes: int,
    weights,
    group_names,
    seed: int = 42,
    checkpoint_path=None,
    verbose: int = 2,
):
    """The baseline's ten epochs at batch 32, with the batches weighted by group.

    Nothing in the current programme calls this. It was written for NB07b, which is
    withdrawn under `DECISIONS.md` 2026-08-11, and it is kept because the intervention it
    implements is specified rather than speculative: `PREREGISTRATION.md` Amendments 14,
    16 and 17 fix the objective, the group variable, the counts and the chance rates, and
    those amendments stand. `tools/dry_run/profiles/nb07b.py` is what still exercises it.

    A Keras loss sees the targets and the predictions and nothing else, so a run
    whose objective depends on which capture a window came from cannot be
    expressed by replacing the loss. Neither can it be expressed by
    `class_weight`, because eight classes hold thirty-four captures between them
    and a per-class weight cannot tell those apart. So the loop is written out.

    What it is not is a second training procedure. The number of epochs, the size
    of a batch and whether the data is shuffled are read from the baseline's own
    `FIT` rather than written again here, so they cannot drift from the published
    ones, and the model is the same compiled model every other run trains, with
    the same optimizer. What differs is one line: the loss the gradient is taken
    of is a weighted sum over the batch rather than a mean over it.

    The model stays an ordinary `Sequential`, so the run saves and loads back like
    every other run in the project. Nothing here is a subclass and nothing has to
    be registered by name before `model.keras` can be read.

    `weights` is an `interventions.GroupWeights`. It supplies the per-window
    weights for a batch and is updated from the group losses that batch produced.
    With `eta` at zero it never moves, which is how the loop is checked against
    `model.fit`.

    Shuffling is seeded per epoch rather than left to Keras, because the loop has
    to know which windows are in a batch in order to know which groups are.
    """
    import keras

    if keras.backend.backend() != "tensorflow":
        raise RuntimeError(
            f"the group-weighted loop needs the tensorflow backend and keras reports "
            f"{keras.backend.backend()!r}. It takes gradients itself rather than through "
            "model.fit, and this project has only run it on tensorflow."
        )
    import tensorflow as tf

    y_codes = np.asarray(y_codes).astype("int64")
    groups = np.asarray(groups).astype("int64")
    if len(groups) != len(y_codes):
        raise ValueError(f"{len(groups)} group codes for {len(y_codes)} windows")

    targets = keras.utils.to_categorical(y_codes, num_classes=int(n_classes))
    epochs = int(mo.FIT["epochs"])
    batch_size = int(mo.FIT["batch_size"])
    shuffle = bool(mo.FIT["shuffle"])
    n = len(y_codes)

    optimizer = model.optimizer
    # Keras creates an optimizer's slot variables the first time it is used. Inside a
    # compiled function that would be variable creation on a traced call, which raises,
    # so they are built here, before anything is traced.
    optimizer.build(model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def step(xb, yb, wb):
        """One batch, compiled: the weighted loss, the update, the per-window losses back.

        The tape is built inside this function rather than around the call to it, so it
        is traced once and the graph is reused for every batch afterwards. Run without
        this wrapper the loop dispatches every operation from Python one at a time and
        blocks on the result of each batch, which leaves the device idle and the run
        takes tens of minutes an epoch. The arithmetic is the same either way; what
        changes is how it is scheduled.

        `reduce_retracing` is on because the last batch of an epoch is short, so the
        function sees two shapes rather than one.
        """
        with tf.GradientTape() as tape:
            probabilities = model(xb, training=True)
            per_window = keras.losses.categorical_crossentropy(yb, probabilities)
            loss = tf.reduce_sum(wb * tf.cast(per_window, tf.float32))
        optimizer.apply_gradients(
            zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
        )
        return per_window

    history = {"loss": [], "worst_group_loss": [], "mean_group_loss": []}

    for epoch in range(epochs):
        rng = np.random.default_rng(int(seed) + epoch)
        order = rng.permutation(n) if shuffle else np.arange(n)

        epoch_loss, epoch_windows = 0.0, 0
        # Per-group totals for the epoch, accumulated with bincount rather than by
        # looping over the windows of every batch in Python. The figures are the same
        # and the loop was two and a half million dictionary operations an epoch.
        totals = np.zeros(weights.n_groups, dtype="float64")
        counts = np.zeros(weights.n_groups, dtype="int64")
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            codes = groups[index]
            xb = tf.convert_to_tensor(X[index])
            yb = tf.convert_to_tensor(targets[index])
            wb = tf.convert_to_tensor(weights.batch_weights(codes), dtype=tf.float32)

            # One read back per batch, and it cannot be avoided: the weights this batch
            # produces are what the next batch is weighted by, so the dependency is
            # serial. Inside a compiled step it is a hundred and twenty-eight bytes after
            # work the device was going to do anyway.
            observed = np.asarray(step(xb, yb, wb))

            weights.update(codes, observed)
            epoch_loss += float(observed.sum())
            epoch_windows += len(index)
            totals += np.bincount(codes, weights=observed, minlength=weights.n_groups)
            counts += np.bincount(codes, minlength=weights.n_groups)

        present = counts > 0
        group_means = totals[present] / counts[present]
        history["loss"].append(epoch_loss / max(epoch_windows, 1))
        history["worst_group_loss"].append(float(group_means.max()))
        history["mean_group_loss"].append(float(group_means.mean()))
        weights.snapshot(epoch=epoch + 1, group_names=group_names)

        if checkpoint_path is not None:
            model.save(checkpoint_path)
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  mean loss {history['loss'][-1]:.4f}  "
                  f"worst group {history['worst_group_loss'][-1]:.4f}  "
                  f"heaviest weight {weights.q.max():.4f}")

    for callback in callbacks or ():
        if hasattr(callback, "on_train_end"):
            callback.on_train_end(None)

    return type("GroupDROHistory", (), {"history": history})()


class PrefixBatches:
    """Batches of windows, each batch cut to one length drawn at random.

    A model that is asked to answer after any number of records has to have been
    trained after any number of records. Every batch is cut to a single length
    because an array has one shape, and the length is drawn per batch rather than
    per window for the same reason.

    The draw is uniform over every length the window allows, one to `window`. That
    gives a five-record prefix the same share of the training as a fifty-record one
    although it carries a fraction of the information. That is a property of the
    design and it is stated rather than corrected: weighting the lengths would be a
    second choice inside a run whose one change is the training input.

    The permutation and the lengths are drawn from a generator seeded per epoch, so
    a run repeats.
    """

    def __init__(self, X, y_codes, *, n_classes: int, batch_size: int, seed: int = 42):
        self.X = X
        self.y = np.asarray(y_codes).astype("int64")
        self.n_classes = int(n_classes)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.window = int(X.shape[1])
        self.epoch = 0
        self.lengths_drawn = []
        self._shuffle()

    def _shuffle(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.order = rng.permutation(len(self.y))
        self.lengths = rng.integers(1, self.window + 1, size=len(self))
        self.lengths_drawn.append([int(v) for v in self.lengths])

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, i):
        import keras

        index = self.order[i * self.batch_size : (i + 1) * self.batch_size]
        k = int(self.lengths[i])
        X = np.ascontiguousarray(self.X[index][:, :k])
        y = keras.utils.to_categorical(self.y[index], num_classes=self.n_classes)
        return X, y

    def on_epoch_end(self):
        self.epoch += 1
        self._shuffle()


def fit_mixed_length(model, X, y_codes, callbacks, *, n_classes: int, seed: int = 42, verbose: int = 2):
    """The baseline's ten epochs at batch 32, each batch cut to its own length.

    The epochs, the batch size and the shuffling are the baseline's `FIT` read
    rather than written again, so they cannot drift from the published ones. What
    differs from `fit_model` is the input: instead of one array of fifty-record
    windows, the model is handed a batch at a time and each batch has been cut to a
    length drawn uniformly from one to fifty.

    Keras shuffles nothing here, because `PrefixBatches` shuffles itself and has to,
    in order to draw a length for each batch it hands over.

    `PrefixBatches` holds the state and knows nothing about Keras, so it can be read
    and tested on its own. Keras 3 wants a `PyDataset` rather than any object with a
    length and an index, so the wrapper that makes it one is built here, where keras
    is already imported, rather than at the top of this module.
    """
    import keras

    source = PrefixBatches(
        X, y_codes, n_classes=int(n_classes), batch_size=int(mo.FIT["batch_size"]), seed=int(seed)
    )

    class _Batches(keras.utils.PyDataset):
        def __len__(self):
            return len(source)

        def __getitem__(self, i):
            return source[i]

        def on_epoch_end(self):
            source.on_epoch_end()

    history = model.fit(
        _Batches(),
        epochs=mo.FIT["epochs"],
        shuffle=False,
        callbacks=list(callbacks),
        verbose=verbose,
    )
    # Keras calls on_epoch_end once more than there are epochs, and its prefetching
    # touches batches one epoch past the last it trains on, so the generator draws two
    # lengths more than it uses. Counted rather than assumed: ten epochs of five batches
    # execute fifty training steps against fifty-two __getitem__ calls spread over eleven
    # epoch counters and twelve draws. The draws are positionally aligned from zero, the
    # record at index n being the epoch that trained n+1, so the surplus is at the tail
    # and the record is truncated there rather than shifted.
    history.prefix_lengths = source.lengths_drawn[: int(mo.FIT["epochs"])]
    return history


def reshape(X):
    """`(samples, window, features)` to `(samples, window, features, 1)`.

    A reshape and not a copy, so an array of two gigabytes is not duplicated to
    be fed to the model. Each timestep then arrives at the encoder as
    `(features, 1)`, which is the shape the published model was written for.
    """
    X = np.asarray(X)
    return X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))


def prefix(X, k: int):
    """The first `k` records of every window, as a contiguous array.

    An observation budget is how much of a window the model is allowed to see. Taking
    it as a prefix of windows that already exist keeps the number of windows and which
    records fall in which window exactly as they were built, so two budgets are two
    models scored on the same items. Cutting the windows again at a shorter length
    would change both, and the runs would no longer be comparable.

    The copy is made here rather than left as a view. A slice along the second axis is
    not contiguous, and the reshape that follows would make the copy anyway, at a point
    where the memory it needs is harder to see.
    """
    X = np.asarray(X)
    k = int(k)
    if not 1 <= k <= X.shape[1]:
        raise ValueError(f"budget {k} is not between 1 and the {X.shape[1]} records in a window")
    if k == X.shape[1]:
        return X
    return np.ascontiguousarray(X[:, :k])


def describe(n_features: int, n_classes: int, *, window: int, lstm_units: int = LSTM_UNITS) -> dict:
    """The model as plain data, for the run's config file."""
    return {
        "encoder": {
            "source": "Mohammadi et al. (2024), arXiv:2410.23306, reproduced in baselines/mohammadi",
            "architecture": list(mo.ARCHITECTURE),
            "used_up_to": "Dense 128, ReLU",
            "head_dropped": "Dense n_classes, softmax",
            "applied": "to each record of the window, the same weights at every timestep",
            "input_shape_per_record": [int(n_features), 1],
        },
        "added": [
            f"TimeDistributed over {'any number of' if window is None else int(window)} records",
            f"LSTM, {int(lstm_units)} units",
            "Dense n_classes, softmax",
        ],
        "input_shape": [None if window is None else int(window), int(n_features), 1],
        "window": None if window is None else int(window),
        "lstm_units": int(lstm_units),
        "n_classes": int(n_classes),
        "compile": dict(mo.COMPILE),
        "fit": dict(mo.FIT),
    }


def encoder_matches_baseline(model, n_features: int, n_classes: int) -> dict:
    """Check the encoder inside `model` against a freshly built published model.

    Layer types, output shapes and parameter counts, compared position by
    position. What it cannot check is that the weights are the published ones,
    because they are not: the encoder is trained here. What it checks is that the
    thing being trained is the published architecture with its head removed.
    """
    inner = model.get_layer("per_record").layer
    published = mo.build_model(int(n_features), int(n_classes))
    expected = list(published.layers)[:-1]
    found = list(inner.layers)

    rows = []
    for position, (want, got) in enumerate(zip(expected, found)):
        rows.append(
            {
                "position": position,
                "baseline": want.__class__.__name__,
                "encoder": got.__class__.__name__,
                "baseline_output": tuple(want.output.shape),
                "encoder_output": tuple(got.output.shape),
                "baseline_params": int(want.count_params()),
                "encoder_params": int(got.count_params()),
            }
        )
    agrees = len(expected) == len(found) and all(
        row["baseline"] == row["encoder"]
        and row["baseline_output"] == row["encoder_output"]
        and row["baseline_params"] == row["encoder_params"]
        for row in rows
    )
    return {
        "agrees": bool(agrees),
        "n_layers": len(found),
        "rows": rows,
        "encoder_params": int(inner.count_params()),
        "baseline_params_without_head": int(sum(layer.count_params() for layer in expected)),
    }


# --------------------------------------------------------------------------
# the fields metrics.json carries beyond the usual ones
# --------------------------------------------------------------------------


def thin_class_caveats(*, sequences: dict, flagged: dict, thin_below: int) -> dict:
    """Classes whose per-class F1 rests on too few sequences to read.

    `sequences` is the per-partition count for every class, taken from the
    manifest rather than retyped. `flagged` names the classes carrying a caveat
    and says why for each. Anything below `thin_below` in a partition that is not
    in `flagged` comes back under `unflagged_and_thin`, so a class that is thin
    and was not thought about shows up here instead of passing unnoticed.
    """
    partitions = list(sequences)
    counts = {
        label: {partition: int(sequences[partition].get(label, 0)) for partition in partitions}
        for label in sorted({label for part in sequences.values() for label in part})
    }

    scored = [p for p in partitions if p != "train"]
    thin = {
        label: values
        for label, values in counts.items()
        if min(values[p] for p in scored) < int(thin_below)
    }
    return {
        "threshold_sequences": int(thin_below),
        "counted_on": scored,
        "flagged": [
            {"label": label, "sequences": counts[label], "reading": reason}
            for label, reason in flagged.items()
        ],
        "unflagged_and_thin": sorted(set(thin) - set(flagged)),
        "reading": (
            "F1 on a handful of sequences takes only a handful of values, so the per-class "
            "figures for these classes are reported and not interpreted as measurements of "
            "how well the class is detected."
        ),
    }


def comparison_against(
    metrics: dict, published: dict, *, run_id: str, classes, threshold: float = 0.50
) -> dict:
    """This run's scores set against a single-record run's, class by class.

    `classes` is the set of classes the comparison is about. Both sides are read
    out of the two metrics documents rather than typed in, and every difference
    is this run minus the other one, so a positive number is an improvement.

    The two runs are scored on different test partitions, which is stated in what
    comes back rather than left for a reader to notice.
    """
    per_class = {}
    for label in classes:
        theirs = float(published["per_class_f1"][label])
        ours = float(metrics["per_class_f1"][label])
        per_class[label] = {
            "published": theirs,
            "this_run": ours,
            "difference": ours - theirs,
            "detected_published": bool(theirs >= threshold),
            "detected_this_run": bool(ours >= threshold),
        }

    return {
        "against": run_id,
        "detected_at": threshold,
        "macro_f1": {
            "published": float(published["macro_f1"]),
            "this_run": float(metrics["macro_f1"]),
            "difference": float(metrics["macro_f1"]) - float(published["macro_f1"]),
        },
        "weighted_f1": {
            "published": float(published["weighted_f1"]),
            "this_run": float(metrics["weighted_f1"]),
            "difference": float(metrics["weighted_f1"]) - float(published["weighted_f1"]),
        },
        "per_class_f1": per_class,
        "n_classes_compared": len(per_class),
        "n_detected_published": sum(v["detected_published"] for v in per_class.values()),
        "n_detected_this_run": sum(v["detected_this_run"] for v in per_class.values()),
        "newly_detected": sorted(
            label
            for label, v in per_class.items()
            if v["detected_this_run"] and not v["detected_published"]
        ),
        "test_partitions_differ": (
            f"{published['n_test']:,} test rows there, {metrics['n_test']:,} test sequences here, "
            "on different splits. The two scores are the two runs' own scores and not a paired "
            "comparison on the same items."
        ),
    }


# --------------------------------------------------------------------------
# fitting and saving in one statement
# --------------------------------------------------------------------------


def fit_and_save(
    out_dir,
    name,
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    classes,
    config,
    parent=None,
    window,
    n_features,
    lstm_units=LSTM_UNITS,
    seed=None,
    extra_metrics=None,
    checkpoint=True,
    predict_batch_size=512,
    verbose=2,
    loss=None,
    class_weight=None,
    mixed_length=False,
    groups=None,
    group_weights=None,
    group_names=None,
    decision_rule=None,
    code_dtype="int8",
):
    """Seed, build, fit, predict, score and write the run, in one statement.

    `assert_single_change` runs before anything is built, so a run that would
    confound two changes never starts. The seed is set immediately before the
    model is constructed, so no code can run between the two.

    `y_train` and `y_test` are integer positions into `classes`, and that is what
    `y_true.npy` and `y_pred.npy` hold: integer codes at `code_dtype`, with the
    list they index in `metrics.json`.

    `extra_metrics` is a function of the metrics document that returns more
    fields to put in it. It runs here rather than in the caller so that whatever
    it produces is written by the same statement that fitted the model.

    Three arguments exist for runs whose one change is an intervention, and all
    three are None for a run without one, leaving this function doing exactly what
    it did before they were added. `loss` replaces the loss at compile time.
    `class_weight` weights the classes in the fit. `decision_rule` is a function
    of the fitted model returning a rule for turning probabilities into codes and
    a record of how the rule was arrived at; the record goes into the metrics
    under `decision_rule`. A rule is built after the fit and before the test
    partition is touched, so a rule that needs a fitted model to choose its
    settings can have one without any of them being chosen on test.

    `groups`, `group_weights` and `group_names` are for a run whose one change is
    the training objective. With them the fit goes through `fit_group_dro` rather
    than `fit_model`, because an objective that depends on which capture a window
    came from cannot be expressed as a loss or as a class weighting. They are None
    for every other run, which leaves the fit exactly where it was.
    """
    import keras

    rn.assert_single_change(config, parent)

    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    classes = np.asarray(classes, dtype=str)

    if seed is not None:
        keras.utils.set_random_seed(int(seed))
    model = build_model(
        int(n_features),
        len(classes),
        window=None if window is None else int(window),
        lstm_units=int(lstm_units),
        loss=loss,
    )

    callbacks = []
    if checkpoint and group_weights is None:
        # A full pass is a long time to lose. The checkpoint is written to Drive
        # after every epoch, so a dropped session costs the epoch it was in.
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "checkpoint.keras"), save_freq="epoch", verbose=0
            )
        )

    started = time.perf_counter()
    if mixed_length:
        history = fit_mixed_length(
            model,
            X_train,
            np.asarray(y_train),
            callbacks,
            n_classes=len(classes),
            seed=int(seed) if seed is not None else 42,
            verbose=verbose,
        )
    elif group_weights is None:
        history = fit_model(
            model,
            X_train,
            np.asarray(y_train),
            callbacks,
            n_classes=len(classes),
            class_weight=class_weight,
            verbose=verbose,
        )
    else:
        history = fit_group_dro(
            model,
            X_train,
            np.asarray(y_train),
            groups,
            callbacks,
            n_classes=len(classes),
            weights=group_weights,
            group_names=group_names,
            seed=int(seed) if seed is not None else 42,
            checkpoint_path=(run_dir / "checkpoint.keras") if checkpoint else None,
            verbose=verbose,
        )
    train_seconds = time.perf_counter() - started

    decide, rule_record = (None, None)
    if decision_rule is not None:
        decide, rule_record = decision_rule(model)

    started = time.perf_counter()
    probabilities = model.predict(X_test, batch_size=int(predict_batch_size), verbose=0)
    inference_seconds = time.perf_counter() - started

    if decide is None:
        y_pred_codes = np.asarray(probabilities).argmax(axis=1).astype(code_dtype)
    else:
        y_pred_codes = np.asarray(decide(probabilities)).astype(code_dtype)
    y_true_codes = np.asarray(y_test).astype(code_dtype)

    metrics = rn.classification_metrics(
        classes[y_true_codes.astype("int64")],
        classes[y_pred_codes.astype("int64")],
        labels=classes.tolist(),
    )
    metrics["train_seconds"] = round(train_seconds, 3)
    metrics["inference_seconds"] = round(inference_seconds, 3)
    metrics["inference_rows_per_second"] = round(len(y_pred_codes) / max(inference_seconds, 1e-9), 1)
    metrics["n_train"] = int(len(y_train))
    metrics["n_parameters"] = int(model.count_params())
    metrics["label_encoding"] = (
        f"y_true.npy and y_pred.npy hold {code_dtype} positions into the labels list above"
    )
    if rule_record is not None:
        metrics["decision_rule"] = rule_record
    if group_weights is not None:
        metrics["group_dro"] = group_weights.record(group_names)
    if mixed_length and getattr(history, "prefix_lengths", None) is not None:
        drawn = [k for epoch in history.prefix_lengths for k in epoch]
        metrics["prefix_lengths"] = {
            "rule": "one length per batch, drawn uniformly from 1 to the window",
            "n_batches": len(drawn),
            "min": int(min(drawn)),
            "max": int(max(drawn)),
            "mean": round(float(np.mean(drawn)), 3),
            "per_epoch": history.prefix_lengths,
        }
    if history is not None and getattr(history, "history", None):
        metrics["history"] = {
            key: [float(v) for v in values] for key, values in history.history.items()
        }
    if extra_metrics is not None:
        metrics.update(extra_metrics(metrics))

    return rn.save_run_codes(
        out_dir,
        name,
        config=config,
        metrics=metrics,
        y_true=y_true_codes,
        y_pred=y_pred_codes,
        labels=classes.tolist(),
        model=model,
        code_dtype=code_dtype,
    )


def load_run(out_dir, name) -> dict | None:
    """A run already written to `out_dir/name`, or None if it is not there."""
    run = rn.load_run(out_dir, name)
    if run is not None:
        run["labels"] = list(run["metrics"]["labels"])
    return run
