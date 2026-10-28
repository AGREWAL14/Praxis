"""A two-head model: the sequence classifier, and an adversary that names the session.

The classifier is the one this project already trains, unchanged: the published record
encoder applied to each record of a window, one LSTM across the fifty results, a softmax
over the classes. What is added is a second head reading the same representation through
a gradient reversal, whose job is to name which recording session the window came from.
Because the gradient is reversed on the way back, the trunk is pushed to make that job
harder while the classifier's job stays the same, and the strength of that push is one
number, `lambda`, which is a variable rather than a constant so a sweep does not rebuild
the model.

Two things about the adversary's task decide whether any of this measures what it is
meant to.

It is conditional on the class. Each capture belongs to exactly one class, so class is a
function of capture identity, and an adversary asked to name the capture with no further
information could satisfy itself by naming the class. A representation made independent of
that is a representation made independent of the label, which is not invariance, it is
destruction. So the true class is an input, and the adversary's output is masked to the
captures of that class: the question it answers is which of *this class's* sessions the
window came from, which is the question NB03 asked of the raw features.

It sees only the windows a session question can be put to. Eleven of the nineteen classes
were recorded once, where the capture and the class are the same set of windows and the
question is empty, and some captures are held back to be probed rather than trained on. A
per-window flag gates the adversary: the reversed representation is multiplied by it, so a
window outside the set contributes no gradient through the reversal at all. The flag is
belt and braces with the sample weight the loss should also carry, and it is the half that
is visible in the graph.

Nothing here trains. `build` returns a compiled-ready model; what to compile it with, how
to weight the two losses, and how to sweep lambda are the caller's.
"""

from __future__ import annotations

import time

import numpy as np

# The offset added to a masked logit. Large enough that the softmax gives it no mass,
# small enough not to produce a NaN when it meets a float32 exponential.
MASK_OFFSET = 1e9

CAPTURE_HIDDEN = 128


def _keras():
    import keras

    return keras


def class_capture_membership(classes, captures, class_of_capture) -> np.ndarray:
    """A `(n_classes, n_captures)` matrix, 1 where that capture belongs to that class.

    This is what turns the adversary's question from "which of all the sessions" into
    "which of this class's sessions". Every capture belongs to exactly one class, so
    every column holds exactly one 1, and a class's row sums to how many sessions it has
    for the adversary to tell apart.
    """
    classes, captures = list(classes), list(captures)
    matrix = np.zeros((len(classes), len(captures)), dtype="float32")
    for column, capture in enumerate(captures):
        label = class_of_capture[capture]
        if label not in classes:
            raise ValueError(f"capture {capture} belongs to {label}, which is not in classes")
        matrix[classes.index(label), column] = 1.0
    per_column = matrix.sum(axis=0)
    if not (per_column == 1).all():
        stray = [captures[i] for i in np.flatnonzero(per_column != 1)]
        raise ValueError(f"these captures do not belong to exactly one class: {stray}")
    empty = [classes[i] for i in np.flatnonzero(matrix.sum(axis=1) == 0)]
    if len(empty) == len(classes):
        raise ValueError("no class holds any of these captures")
    return matrix


def chance_rate(membership) -> float:
    """The rate a guesser reaches, averaged over the classes that have captures here.

    A class with one capture in the set contributes 1.0, because there is nothing to tell
    apart; a class with none contributes nothing to the average.

    Counted in double precision. The membership matrix is float32 because it ends up as a
    layer's weights, and a reciprocal taken in float32 puts 1/7 out by about 1e-8, which
    is enough to make this disagree with the same rate counted from the captures
    themselves. The counts are small integers, so widening them first makes the two
    routes to this number agree exactly rather than nearly.
    """
    counts = np.asarray(membership, dtype="float64").sum(axis=1)
    counts = counts[counts > 0]
    return float(np.mean(1.0 / counts))


def _register(cls):
    keras = _keras()
    return keras.saving.register_keras_serializable(package="ag_praxis")(cls)


def gradient_reversal_layer():
    """The reversal, as a layer class.

    Built inside a function because importing keras at module import time would make
    every module that touches this one pay for it.

    Forward it is the identity. Backward it multiplies the gradient by `-lambda`, so the
    trunk moves away from whatever helps the head above it. `lambda` is a non-trainable
    weight rather than a Python float, so a sweep assigns it and the same built model is
    reused, and so its value is saved with the model rather than remembered.
    """
    keras = _keras()
    from keras import layers

    @_register
    class GradientReversal(layers.Layer):
        def __init__(self, lam: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.initial_lam = float(lam)

        def build(self, input_shape):
            self.lam = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(self.initial_lam),
                trainable=False,
                dtype="float32",
                name="lam",
            )
            super().build(input_shape)

        def call(self, inputs):
            import tensorflow as tf

            lam = self.lam

            @tf.custom_gradient
            def reverse(x):
                def backward(dy):
                    return -lam * dy

                return tf.identity(x), backward

            return reverse(inputs)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            return {**super().get_config(), "lam": self.initial_lam}

    return GradientReversal


def within_class_softmax_layer():
    """A softmax over only the captures the given class owns.

    Takes the capture logits and the row of the membership matrix the class selects, and
    puts every capture outside that class beyond the reach of the softmax. Without this
    the adversary can answer by naming the class, and a trunk trained against it loses the
    label rather than the session.
    """
    from keras import layers, ops

    @_register
    class WithinClassSoftmax(layers.Layer):
        def call(self, inputs):
            logits, allowed = inputs
            return ops.softmax(logits - (1.0 - allowed) * MASK_OFFSET, axis=-1)

        def compute_output_shape(self, input_shape):
            return input_shape[0]

    return WithinClassSoftmax


def build(
    *,
    encoder,
    n_features: int,
    n_classes: int,
    n_captures: int,
    window: int,
    membership,
    lstm_units: int = 128,
    lam: float = 0.0,
    capture_hidden: int = CAPTURE_HIDDEN,
):
    """The two-head model, ready to compile.

    `encoder` is the published record encoder with its head removed, passed in rather than
    built here so this module cannot drift from the one the rest of the project trains.

    Three inputs. The windows. The true class as a one-hot, which conditions the adversary
    and is not seen by the classifier. And a per-window flag, 1 where the adversary is
    allowed to look, which gates the reversed representation so a window outside the
    adversary's set sends no gradient back through it.

    Two outputs, `attack` over the classes and `capture` over the captures. The trunk
    layers keep the names the rest of the project reads them by, so anything that taps
    `across_the_window` for a representation keeps working.
    """
    keras = _keras()
    from keras import layers

    membership = np.asarray(membership, dtype="float32")
    if membership.shape != (n_classes, n_captures):
        raise ValueError(
            f"membership is {membership.shape}, expected ({n_classes}, {n_captures})"
        )

    GradientReversal = gradient_reversal_layer()
    WithinClassSoftmax = within_class_softmax_layer()

    windows = keras.Input(shape=(int(window), int(n_features), 1), name="windows")
    class_onehot = keras.Input(shape=(int(n_classes),), name="class_onehot")
    in_domain = keras.Input(shape=(1,), name="in_domain")

    # The trunk, layer for layer what the single-head model holds, under the same names.
    per_record = layers.TimeDistributed(encoder, name="per_record")(windows)
    representation = layers.LSTM(int(lstm_units), name="across_the_window")(per_record)

    attack = layers.Dense(int(n_classes), activation="softmax", name="attack")(representation)

    reversed_representation = GradientReversal(lam=float(lam), name="gradient_reversal")(
        representation
    )
    gated = layers.Multiply(name="only_domain_windows")([reversed_representation, in_domain])
    conditioned = layers.Concatenate(name="with_the_class")([gated, class_onehot])
    hidden = layers.Dense(int(capture_hidden), activation="relu", name="capture_hidden")(
        conditioned
    )
    logits = layers.Dense(int(n_captures), name="capture_logits")(hidden)

    # The class picks its own row of the membership matrix: a fixed, untrained projection.
    allowed = layers.Dense(
        int(n_captures), use_bias=False, trainable=False, name="class_to_captures"
    )(class_onehot)
    capture = WithinClassSoftmax(name="capture")([logits, allowed])

    model = keras.Model(
        inputs={"windows": windows, "class_onehot": class_onehot, "in_domain": in_domain},
        outputs={"attack": attack, "capture": capture},
        name="cnn_lstm_two_head",
    )
    model.get_layer("class_to_captures").set_weights([membership])
    return model


def set_lambda(model, value: float) -> float:
    """Assign the reversal strength on a built model, and read it back."""
    layer = model.get_layer("gradient_reversal")
    layer.lam.assign(float(value))
    return float(layer.lam.numpy())


def parameter_split(model) -> dict:
    """How many parameters belong to the trunk and the attack head, and how many the
    adversary adds.

    The first number is what the single-head model has, so a run can say the classifier it
    trains is the same size as the one it is compared against, and the second is the cost
    of the head that is thrown away after training.
    """
    adversary = {"capture_hidden", "capture_logits", "class_to_captures", "gradient_reversal"}
    shared, extra = 0, 0
    for layer in model.layers:
        count = int(sum(int(np.prod(w.shape)) for w in layer.weights))
        if layer.name in adversary:
            extra += count
        else:
            shared += count
    return {
        "trunk_and_attack_head": shared,
        "adversary": extra,
        "total": shared + extra,
    }


def fit_adversarial(
    model,
    *,
    X,
    attack_targets,
    class_onehot,
    in_domain,
    capture_targets,
    epochs: int,
    batch_size: int,
    seed: int,
    capture_weight: float = 1.0,
    rows=None,
    checkpoint_path=None,
    verbose: int = 2,
):
    """The two-head fit, written out because no `model.fit` call expresses it.

    The structure is the one `sequence.fit_group_dro` already uses, for the same reason:
    the objective depends on which capture a window came from, and a Keras loss sees the
    targets and the predictions and nothing else. The batch step is compiled and traced
    once, the optimizer's slot variables are built before that trace so no variable is
    created inside it, and shuffling is seeded per epoch here rather than left to Keras.

    The loss is the attack cross-entropy plus the capture cross-entropy. The second is
    summed over the windows the adversary is allowed to see and divided by how many there
    were, so a batch that happens to hold few of them does not contribute a smaller loss
    per window than one that holds many. Rows outside that set are zeroed by the same flag
    that gates the reversed representation, so they are excluded twice over: no gradient
    reaches the trunk through the adversary, and no loss is counted for them.

    `lambda` is not here. It lives on the reversal layer and is read from the model, so a
    sweep assigns it and calls this again without rebuilding anything.

    `capture_weight` at 0 takes the capture loss out of the objective altogether, which is
    not the same statement as lambda at 0 even though the two should coincide. Lambda at 0
    leaves the adversary in the objective and multiplies by zero the gradient it sends back
    through the reversal; the capture head still trains, on a loss that is still reported.
    Weight at 0 removes the term, so the head does not train either and what is optimised
    is the attack loss alone. That is the single-head model, and it is what a sweep over
    lambda has to be read against. The branch is resolved when the step is traced, so the
    weightless run does not carry the arithmetic it is not using.
    """
    import keras
    import tensorflow as tf

    held = len(attack_targets)
    for name, array in (("class_onehot", class_onehot), ("in_domain", in_domain),
                        ("capture_targets", capture_targets), ("X", X)):
        if len(array) != held:
            raise ValueError(f"{name} holds {len(array)} rows against {held} targets")

    # `rows` selects which windows are trained on, as an index rather than a slice of the
    # arrays. Slicing would copy a couple of gigabytes of windows to leave some out, and
    # whether that fits would depend on the runtime rather than on the experiment.
    rows = np.arange(held) if rows is None else np.asarray(rows).astype("int64")
    n = len(rows)
    if n == 0:
        raise ValueError("no rows to train on")

    optimizer = model.optimizer
    if optimizer is None:
        raise ValueError("the model has no optimizer; compile it before fitting")
    optimizer.build(model.trainable_variables)
    weight = float(capture_weight)

    @tf.function(reduce_retracing=True)
    def step(windows, classes, domain, attack_true, capture_true, n_domain):
        with tf.GradientTape() as tape:
            out = model(
                {"windows": windows, "class_onehot": classes, "in_domain": domain},
                training=True,
            )
            attack_each = keras.losses.categorical_crossentropy(attack_true, out["attack"])
            capture_each = keras.losses.categorical_crossentropy(capture_true, out["capture"])
            attack_loss = tf.reduce_mean(attack_each)
            # The flag is the sample weight. A batch with no window the adversary may see
            # contributes nothing rather than dividing by zero.
            weighted = tf.reduce_sum(capture_each * tf.squeeze(domain, axis=-1))
            capture_loss = weighted / n_domain
            # Resolved at trace time, so a run with no adversary traces a graph without it.
            if weight == 0.0:
                loss = attack_loss
            else:
                loss = attack_loss + weight * capture_loss
        # With the capture loss out of the objective the adversary's own weights have no
        # gradient at all, and handing the optimizer a None for them warns on every run.
        # Filtering is resolved when the step is traced, so nothing is decided per batch.
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
        )
        correct = tf.reduce_sum(
            tf.cast(
                tf.equal(tf.argmax(out["attack"], axis=1), tf.argmax(attack_true, axis=1)),
                tf.float32,
            )
        )
        return attack_loss, capture_loss, correct

    history = {"attack_loss": [], "capture_loss": [], "attack_accuracy": [],
               "domain_windows": [], "seconds": []}
    lam = float(model.get_layer("gradient_reversal").lam.numpy())

    for epoch in range(int(epochs)):
        started = time.perf_counter()
        rng = np.random.default_rng(int(seed) + epoch)
        order = rows[rng.permutation(n)]

        attack_total, capture_total, correct_total = 0.0, 0.0, 0.0
        seen, domain_seen, capture_batches = 0, 0, 0
        for start in range(0, n, int(batch_size)):
            index = order[start : start + int(batch_size)]
            domain_batch = in_domain[index]
            n_domain = float(domain_batch.sum())
            attack_loss, capture_loss, correct = step(
                tf.convert_to_tensor(X[index]),
                tf.convert_to_tensor(class_onehot[index]),
                tf.convert_to_tensor(domain_batch),
                tf.convert_to_tensor(attack_targets[index]),
                tf.convert_to_tensor(capture_targets[index]),
                tf.constant(max(n_domain, 1.0), dtype=tf.float32),
            )
            attack_total += float(attack_loss) * len(index)
            correct_total += float(correct)
            seen += len(index)
            domain_seen += int(n_domain)
            if n_domain > 0:
                capture_total += float(capture_loss)
                capture_batches += 1

        history["attack_loss"].append(attack_total / max(seen, 1))
        history["capture_loss"].append(capture_total / max(capture_batches, 1))
        history["attack_accuracy"].append(correct_total / max(seen, 1))
        history["domain_windows"].append(domain_seen)
        history["seconds"].append(round(time.perf_counter() - started, 2))

        if checkpoint_path is not None:
            model.save(checkpoint_path)
        if verbose:
            print(
                f"  epoch {epoch + 1}/{epochs}  attack loss {history['attack_loss'][-1]:.4f}"
                f"   capture loss {history['capture_loss'][-1]:.4f}"
                + ("  (not in the objective)" if weight == 0.0 else "")
                + f"   attack accuracy {history['attack_accuracy'][-1]:.4f}"
                f"   [{history['seconds'][-1]:,.0f}s, lambda {lam:g}]"
            )

    history["lambda"] = lam
    history["capture_weight"] = weight
    history["objective"] = (
        "attack cross-entropy alone" if weight == 0.0
        else f"attack cross-entropy plus {weight:g} times the capture cross-entropy"
    )
    history["epochs"] = int(epochs)
    history["batch_size"] = int(batch_size)
    history["seed"] = int(seed)
    return history


# --------------------------------------------------------------------------
# reading the session back off the representation
# --------------------------------------------------------------------------


def representation(model, X, index, *, layer: str = "across_the_window", chunk: int = 8192,
                   batch: int = 512):
    """The trunk's output for each of the given windows, before either head.

    Read a chunk of the index at a time rather than by slicing the whole index out of X
    first. That slice would be a second copy of a gigabyte of windows, and whether it fits
    would depend on which runtime the session got rather than on anything about the
    experiment.

    The sub-model is built from the windows input alone, so nothing has to be invented for
    the two inputs only the adversary reads.
    """
    import keras

    index = np.asarray(index)
    reader = keras.Model(model.get_layer("windows").output, model.get_layer(layer).output)
    pieces = []
    for start in range(0, len(index), int(chunk)):
        taken = index[start : start + int(chunk)]
        pieces.append(np.asarray(reader.predict(X[taken], batch_size=int(batch), verbose=0)))
    return np.concatenate(pieces) if pieces else np.empty((0, 0))


def equal_draw(index, group_codes, *, seed: int, per_group: int | None = None):
    """The same number of windows from every capture present in `index`.

    Drawing equally is what stops a capture being identifiable by being larger than the
    others, which is the rule the capture-identification measurement in this project has
    used from the start. The count is the smallest capture's unless one is given.
    """
    index = np.asarray(index)
    codes = np.asarray(group_codes)[index]
    present = np.unique(codes)
    sizes = {int(code): int((codes == code).sum()) for code in present}
    take = int(per_group) if per_group is not None else min(sizes.values())
    rng = np.random.default_rng(int(seed))
    keep = []
    for code in present:
        where = index[codes == code]
        keep.append(rng.choice(where, size=take, replace=False) if take < len(where) else where)
    return np.sort(np.concatenate(keep)), take


def capture_probe(H, capture_labels, class_labels, *, n_estimators: int, min_samples_leaf: int,
                  test_fraction: float, seed: int):
    """Can the capture be named from the representation? Pooled, and with the class fixed.

    The forest and the split are the ones the feature-level measurement used, so the two
    numbers are read the same way. Pooled over every capture the answer is not the
    interesting one, because a capture belongs to one class and naming it well could be
    naming the class; the row that matters is the class held fixed, where the captures
    being told apart all carry the same label.

    Returns one row per scope: how many captures were in play, what guessing would score,
    and what the forest scored on the part of the draw it was not fitted on.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    H = np.asarray(H)
    capture_labels = np.asarray(capture_labels).astype(str)
    class_labels = np.asarray(class_labels).astype(str)
    rows = []

    def one(mask, scope):
        targets = capture_labels[mask]
        distinct = np.unique(targets)
        if len(distinct) < 2:
            rows.append({"scope": scope, "captures": int(len(distinct)), "windows": int(mask.sum()),
                         "chance": float("nan"), "accuracy": float("nan"),
                         "note": "one capture here, so there is nothing to tell apart"})
            return
        train_H, test_H, train_y, test_y = train_test_split(
            H[mask], targets, test_size=float(test_fraction),
            random_state=int(seed), stratify=targets,
        )
        forest = RandomForestClassifier(
            n_estimators=int(n_estimators), min_samples_leaf=int(min_samples_leaf),
            n_jobs=-1, random_state=int(seed),
        )
        forest.fit(train_H, train_y)
        forest.n_jobs = 1
        rows.append({
            "scope": scope,
            "captures": int(len(distinct)),
            "windows": int(mask.sum()),
            "chance": 1.0 / len(distinct),
            "accuracy": float((forest.predict(test_H) == test_y).mean()),
            "note": "",
        })

    one(np.ones(len(capture_labels), dtype=bool), "pooled")
    for label in sorted(set(class_labels.tolist())):
        one(class_labels == label, f"within {label}")
    return rows


def probe_summary(rows) -> dict:
    """The two figures the probe exists to produce, out of the per-scope rows."""
    pooled = next((r for r in rows if r["scope"] == "pooled"), None)
    within = [r for r in rows if r["scope"] != "pooled" and np.isfinite(r["accuracy"])]
    return {
        "pooled_accuracy": float(pooled["accuracy"]) if pooled else float("nan"),
        "pooled_chance": float(pooled["chance"]) if pooled else float("nan"),
        "class_held_fixed_accuracy": float(np.mean([r["accuracy"] for r in within])) if within else float("nan"),
        "class_held_fixed_chance": float(np.mean([r["chance"] for r in within])) if within else float("nan"),
        "n_classes": len(within),
    }
