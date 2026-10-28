"""Writing a run to disk, and the metrics every run reports.

A run is a directory holding five files: the configuration it was produced under,
its metrics, the true labels, the predicted labels, and the fitted model. Always
the same five, so a result can be re-read later without the notebook that made it.

The `fit_and_save` functions exist so that fitting and saving are a single
statement. A cell that fits in one statement and saves in the next loses the fit
if the session drops in between, and a Colab session drops. Nothing here prints;
the notebook keeps the narrative and the printing.

`assert_single_change` is the other half of the discipline. Every run is a config
that inherits from a parent and overrides one key, and the check runs before the
fit rather than after it, so a run that would confound two changes never starts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Keys that describe a run rather than define it, so a difference in one of them
# is not a change to the experiment. `observed` is where a caller puts shapes,
# counts and other things read off the data after the fact: they differ between
# runs as a consequence of the one change, not as a second change.
IGNORED_KEYS = ("parent", "run_id", "notes", "observed")


# --------------------------------------------------------------------------
# one change per run
# --------------------------------------------------------------------------


def assert_single_change(cfg: dict, parent: dict | None, *, ignore=IGNORED_KEYS) -> set:
    """Raise unless `cfg` differs from `parent` in at most one key.

    Keys are compared both ways rather than only over `cfg`, so a key the child
    drops counts as a change too; a run that silently stops setting something is
    as much a second change as one that sets it differently.

    A run with no parent is a root. There is nothing to compare it against, so
    the check passes and returns an empty set. Roots are declared by writing
    `parent=None`, which makes "this is where a chain starts" a statement in the
    notebook rather than an omission.
    """
    if parent is None:
        return set()
    keys = set(cfg) | set(parent)
    diff = {k for k in keys if cfg.get(k) != parent.get(k)} - set(ignore)
    assert len(diff) <= 1, (
        f"Two changes in one run: {sorted(diff)}. "
        f"{cfg.get('run_id', 'this run')} would differ from "
        f"{parent.get('run_id', 'its parent')} in more than one key, so a difference in "
        "the result could not be attributed to either."
    )
    return diff


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def classification_metrics(y_true, y_pred, *, labels=None) -> dict:
    """Accuracy, macro and weighted precision, recall and F1, and the per-class table.

    Macro and weighted are always both here because they answer different
    questions on an imbalanced problem: weighted is what the common classes did,
    macro is what the average class did, and a gap between them is the rare
    classes failing.

    The two baselines are carried alongside because accuracy on its own is
    unreadable when the number of classes changes between runs. Chance is one
    over the number of classes; the majority rate is what always answering the
    largest class would score.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [str(v) for v in labels]

    precision, recall, per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    support = {label: int(n) for label, n in zip(labels, support)}

    return {
        "n_test": int(len(y_true)),
        "n_classes": len(labels),
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class_precision": {l: float(v) for l, v in zip(labels, precision)},
        "per_class_recall": {l: float(v) for l, v in zip(labels, recall)},
        "per_class_f1": {l: float(v) for l, v in zip(labels, per_class)},
        "support": support,
        "chance_rate": 1.0 / max(len(labels), 1),
        "majority_class_rate": max(support.values()) / max(len(y_true), 1) if support else 0.0,
    }


def per_class_frame(metrics: dict):
    """The per-class precision, recall, F1 and support as a table that can be printed."""
    import pandas as pd

    labels = list(metrics["per_class_f1"])
    frame = pd.DataFrame(
        {
            "label": labels,
            "precision": [metrics["per_class_precision"][l] for l in labels],
            "recall": [metrics["per_class_recall"][l] for l in labels],
            "f1": [metrics["per_class_f1"][l] for l in labels],
            "support": [metrics["support"][l] for l in labels],
        }
    ).sort_values("f1")
    for column in ("precision", "recall", "f1", "support"):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{column} came out as {frame[column].dtype}, not numeric")
    return frame.reset_index(drop=True)


def feature_importance(model, features, *, top_k: int = 15) -> dict | None:
    """Every importance the model exposes, and the `top_k` largest by name."""
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return None
    pairs = sorted(zip(list(features), [float(v) for v in values]), key=lambda p: -p[1])
    return {
        "all": {name: value for name, value in pairs},
        "top": [{"feature": name, "importance": value} for name, value in pairs[:top_k]],
    }


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------


def save_run(out_dir, name=None, *, config, metrics, y_true, y_pred, model=None) -> dict:
    """Write one run into `out_dir/name` and return what was written.

    Five files: `config.json`, `metrics.json`, `y_true.npy`, `y_pred.npy` and the
    model. The model file is `model.keras` for anything with a `save` method and
    `model.joblib` otherwise, so a scikit-learn run and a Keras run leave the
    same five files behind under different extensions.

    `name` defaults to `config["run_id"]`, so the directory a run lands in is the
    identifier its config already carries and the two cannot drift apart.

    Timings are required rather than optional. Training and inference seconds are
    a reported feasibility observation for this project, and a run that did not
    record them cannot be added to that table later.
    """
    name = str(config["run_id"]) if name is None else str(name)
    missing = [k for k in ("train_seconds", "inference_seconds") if k not in metrics]
    if missing:
        raise KeyError(
            f"{name}: metrics is missing {missing}. Every run records how long it took to "
            "train and how long it took to predict."
        )

    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    np.save(run_dir / "y_true.npy", np.asarray(y_true).astype(str))
    np.save(run_dir / "y_pred.npy", np.asarray(y_pred).astype(str))

    model_file = None
    if model is not None:
        if hasattr(model, "save"):
            model_file = run_dir / "model.keras"
            model.save(model_file)
        else:
            import joblib

            model_file = run_dir / "model.joblib"
            joblib.dump(model, model_file)

    return {
        "name": name,
        "run_dir": str(run_dir),
        "model_file": str(model_file) if model_file else None,
        "config": config,
        "metrics": metrics,
    }


def save_run_codes(
    out_dir,
    name=None,
    *,
    config,
    metrics,
    y_true,
    y_pred,
    labels,
    model=None,
    code_dtype="int8",
) -> dict:
    """The same five files as `save_run`, with the labels written as codes.

    `y_true.npy` and `y_pred.npy` hold integer positions into `labels` rather than
    the class names themselves, and `labels` is the list those positions index,
    carried in `metrics.json`. Fifty thousand predictions are fifty thousand
    single bytes instead of fifty thousand strings, and the arrays this project
    stores its rows in already hold their labels the same way.

    An array of class names is not merely larger, it is larger by a factor that
    depends on the longest name: numpy stores fixed-width unicode, so every element
    of an array holding one twenty-three character label occupies ninety-two bytes
    whatever it says. A million-row run saved that way runs to over a hundred
    megabytes and will not go into a git repository.

    `code_dtype` is the width those codes are written at. The default is one byte,
    which holds this project's nineteen classes with room to spare; a caller that
    wants headroom for a wider class set can ask for `int16` and pay two bytes.
    The width is checked against the number of labels rather than assumed.

    `labels` is required rather than inferred. A file of integers with no list to
    read it against says nothing at all, so the list has to be written by the same
    call that writes the integers and cannot be reconstructed afterwards from
    whichever ordering a later reader happens to guess.
    """
    labels = [str(v) for v in labels]
    if not labels:
        raise ValueError("labels is empty, so the codes in y_true and y_pred index nothing")
    limit = int(np.iinfo(np.dtype(code_dtype)).max)
    if len(labels) > limit:
        raise ValueError(
            f"{len(labels)} classes do not fit in {code_dtype}, which reaches {limit}. "
            "Ask for a wider code_dtype."
        )
    if metrics.get("labels") != labels:
        raise ValueError(
            "metrics['labels'] is not the label list the codes index, so metrics.json and "
            "y_pred.npy would disagree about which class each code names."
        )

    coded = {}
    for which, values in (("y_true", y_true), ("y_pred", y_pred)):
        codes = np.asarray(values)
        if codes.dtype.kind not in "iu":
            raise TypeError(
                f"{which} arrived as {codes.dtype}, not an integer type. These files hold "
                "positions into the label list, and casting names to integers does not "
                "produce them."
            )
        if codes.size and (codes.min() < 0 or codes.max() >= len(labels)):
            raise ValueError(
                f"{which} holds a code outside 0..{len(labels) - 1}, so it does not index labels"
            )
        coded[which] = codes.astype(code_dtype)

    name = str(config["run_id"]) if name is None else str(name)
    missing = [k for k in ("train_seconds", "inference_seconds") if k not in metrics]
    if missing:
        raise KeyError(
            f"{name}: metrics is missing {missing}. Every run records how long it took to "
            "train and how long it took to predict."
        )

    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    np.save(run_dir / "y_true.npy", coded["y_true"])
    np.save(run_dir / "y_pred.npy", coded["y_pred"])

    model_file = None
    if model is not None:
        model_file = run_dir / "model.keras"
        model.save(model_file)

    return {
        "name": name,
        "run_dir": str(run_dir),
        "model_file": str(model_file) if model_file else None,
        "config": config,
        "metrics": metrics,
        "labels": labels,
    }


def write_label_map(path, labels, *, code_dtype="int8", written_by=None) -> dict:
    """Write the code-to-name mapping the saved arrays index, once, beside the runs.

    Every run already carries its label list inside `metrics.json`, so this file
    adds nothing a run cannot answer on its own. What it adds is one place to look
    when several runs sit under one directory, and a check that they all mean the
    same thing by the same integer: the map is written once and every run's list is
    compared against it rather than each being read on its own terms.
    """
    labels = [str(v) for v in labels]
    document = {
        "written_by": written_by,
        "code_dtype": str(code_dtype),
        "n_classes": len(labels),
        "labels": labels,
        "code_to_label": {str(i): label for i, label in enumerate(labels)},
        "reading": (
            "y_true.npy and y_pred.npy under each run directory hold positions into labels, "
            "in the order listed here. The same order is in every run's metrics.json."
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return document


# --------------------------------------------------------------------------
# fitting and saving in one statement
# --------------------------------------------------------------------------


def fit_and_save(
    out_dir,
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    features,
    target,
    notes=None,
    extra_config=None,
    top_k: int = 15,
) -> dict:
    """Fit a scikit-learn estimator, predict, score, and write the run, in one statement.

    Training and inference time are measured here rather than by the caller, so
    every run in the project records them the same way.
    """
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    # Read the parameters the fit actually ran under, before the line below
    # changes one of them.
    params = model.get_params()

    # Predict on one thread. A forest fitted with n_jobs=-1 comes out identical
    # every time, but its prediction does not: the per-tree probabilities are
    # summed into a shared array from several threads, and floating-point
    # addition is not associative, so the totals differ in the last bit and any
    # class the trees split exactly evenly flips. Fitting stays parallel;
    # only the accumulation is serialised.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    started = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_seconds = time.perf_counter() - started

    metrics = classification_metrics(y_test, y_pred)
    metrics["train_seconds"] = round(train_seconds, 3)
    metrics["inference_seconds"] = round(inference_seconds, 3)
    metrics["inference_rows_per_second"] = round(len(y_pred) / max(inference_seconds, 1e-9), 1)

    importance = feature_importance(model, features, top_k=top_k)
    if importance is not None:
        metrics["top_features"] = importance["top"]
        metrics["feature_importances"] = importance["all"]

    config = {
        "run": name,
        "run_id": name,
        "target": target,
        "model": type(model).__name__,
        "params": params,
        "n_features": len(list(features)),
        "features": list(features),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "notes": notes,
    }
    if extra_config:
        config.update(extra_config)

    return save_run(
        out_dir, name, config=config, metrics=metrics, y_true=y_test, y_pred=y_pred, model=model
    )


def fit_and_save_keras(
    out_dir,
    name,
    *,
    build,
    fit,
    X_train,
    y_train,
    X_test,
    y_test,
    classes,
    config,
    parent=None,
    seed=None,
    checkpoint=True,
    predict_batch_size=4096,
) -> dict:
    """Seed, build, fit, predict, score and write a Keras run, in one statement.

    `build` takes no arguments and returns a compiled model; `fit` takes the
    model, the inputs, the integer labels and a list of callbacks and trains it.
    Both are passed in rather than written here, because for the reproduced
    baseline they belong to the baseline and this function is only the machinery
    around them.

    The seed is set immediately before `build` is called, so no code can run
    between seeding and the construction of the model. `assert_single_change`
    runs before anything is built, so a run that would confound two changes never
    starts.

    `y_train` and `y_test` are integer positions into `classes`. Predictions come
    back as class names, so `y_true.npy` and `y_pred.npy` hold the same strings
    for every run in the project whatever the task.
    """
    import keras

    assert_single_change(config, parent)

    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    classes = np.asarray(classes, dtype=str)

    if seed is not None:
        keras.utils.set_random_seed(int(seed))
    model = build()

    callbacks = []
    if checkpoint:
        # A full pass is hours of training and a Colab session can end inside
        # one. The checkpoint is written to Drive after every epoch so a dropped
        # session costs the epoch it was in rather than the whole run.
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "checkpoint.keras"), save_freq="epoch", verbose=0
            )
        )

    started = time.perf_counter()
    history = fit(model, X_train, np.asarray(y_train), callbacks)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probabilities = model.predict(X_test, batch_size=int(predict_batch_size), verbose=0)
    inference_seconds = time.perf_counter() - started

    y_pred = classes[np.asarray(probabilities).argmax(axis=1)]
    y_true = classes[np.asarray(y_test).astype("int64")]

    metrics = classification_metrics(y_true, y_pred, labels=classes.tolist())
    metrics["train_seconds"] = round(train_seconds, 3)
    metrics["inference_seconds"] = round(inference_seconds, 3)
    metrics["inference_rows_per_second"] = round(len(y_pred) / max(inference_seconds, 1e-9), 1)
    metrics["n_train"] = int(len(y_train))
    metrics["n_parameters"] = int(model.count_params())
    if history is not None and getattr(history, "history", None):
        metrics["history"] = {
            key: [float(v) for v in values] for key, values in history.history.items()
        }

    return save_run(
        out_dir, name, config=config, metrics=metrics, y_true=y_true, y_pred=y_pred, model=model
    )


def kfold_fit_and_save(
    out_dir,
    name,
    *,
    build,
    X_train,
    y_train,
    X_test,
    y_test,
    classes,
    config,
    parent=None,
    n_splits=5,
    seed=42,
    features=None,
    top_k=15,
) -> dict:
    """Cross-validate on the training partition, then fit once and score once on test.

    The folds are cut inside the training partition and never touch the test
    partition, so what they measure is how much the score moves when the training
    rows change, and the test partition is still evaluated exactly once for this
    configuration.

    `build` returns a fresh unfitted estimator each time it is called, so no fold
    inherits anything from the fold before it. The model saved at the end is the
    one fitted on the whole training partition, which is the one the test score
    belongs to.
    """
    from sklearn.model_selection import StratifiedKFold

    assert_single_change(config, parent)
    classes = np.asarray(classes, dtype=str)
    y_train = np.asarray(y_train).astype(str)
    y_test = np.asarray(y_test).astype(str)

    folds = []
    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    fold_seconds = time.perf_counter()
    for i, (fit_index, score_index) in enumerate(splitter.split(X_train, y_train), start=1):
        fold_model = build()
        fold_model.fit(X_train[fit_index], y_train[fit_index])
        if hasattr(fold_model, "n_jobs"):
            fold_model.n_jobs = 1
        predicted = fold_model.predict(X_train[score_index])
        folds.append(
            {
                "fold": i,
                "n_fit": int(len(fit_index)),
                "n_score": int(len(score_index)),
                "macro_f1": float(
                    f1_score(
                        y_train[score_index], predicted, labels=classes.tolist(),
                        average="macro", zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        y_train[score_index], predicted, labels=classes.tolist(),
                        average="weighted", zero_division=0,
                    )
                ),
                "accuracy": float(accuracy_score(y_train[score_index], predicted)),
            }
        )
        del fold_model, predicted
    fold_seconds = time.perf_counter() - fold_seconds

    model = build()
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    params = model.get_params()
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    started = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_seconds = time.perf_counter() - started

    metrics = classification_metrics(y_test, y_pred, labels=classes.tolist())
    metrics["train_seconds"] = round(train_seconds, 3)
    metrics["inference_seconds"] = round(inference_seconds, 3)
    metrics["inference_rows_per_second"] = round(len(y_pred) / max(inference_seconds, 1e-9), 1)
    metrics["n_train"] = int(len(y_train))
    metrics["cross_validation"] = {
        "n_splits": int(n_splits),
        "seconds": round(fold_seconds, 3),
        "folds": folds,
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
        "macro_f1_sd": float(np.std([f["macro_f1"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0,
        "weighted_f1_mean": float(np.mean([f["weighted_f1"] for f in folds])),
    }
    metrics["fitted_params"] = {k: str(v) for k, v in params.items()}

    if features is not None:
        importance = feature_importance(model, features, top_k=top_k)
        if importance is not None:
            metrics["top_features"] = importance["top"]
            metrics["feature_importances"] = importance["all"]

    return save_run(
        out_dir, name, config=config, metrics=metrics, y_true=y_test, y_pred=y_pred, model=model
    )


# --------------------------------------------------------------------------
# reading runs back
# --------------------------------------------------------------------------


def load_run(out_dir, name) -> dict | None:
    """A run already written to `out_dir/name`, or None if it is not there.

    `metrics.json` is the marker, because `save_run` writes it after the config
    and a directory that has it has been through the whole of a fit. What comes
    back has the same shape as what `save_run` returns, so a run read off disk and
    a run just produced go into the same tables.
    """
    run_dir = Path(out_dir) / name
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        return None
    if not config_path.exists():
        raise FileNotFoundError(
            f"{run_dir} holds metrics.json and no config.json, so it is a half-written run. "
            "Delete the directory and let it run again."
        )
    model_file = next(
        (p for p in (run_dir / "model.keras", run_dir / "model.joblib") if p.exists()), None
    )
    return {
        "name": name,
        "run_dir": str(run_dir),
        "model_file": str(model_file) if model_file else None,
        "config": json.loads(config_path.read_text()),
        "metrics": json.loads(metrics_path.read_text()),
    }


def comparison_row(run: dict) -> dict:
    """One line of the side-by-side table: what the run was, and what it scored."""
    config, metrics = run["config"], run["metrics"]
    return {
        "run_id": config["run_id"],
        "model": config["model"],
        "task": config["task"],
        "split": config["split"],
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": metrics["macro_f1"],
        "gap": metrics["weighted_f1"] - metrics["macro_f1"],
        "train_s": metrics["train_seconds"],
        "infer_s": metrics["inference_seconds"],
    }


def comparison_frame(runs):
    """Every run side by side, with the weighted-minus-macro gap on each line."""
    import pandas as pd

    frame = pd.DataFrame([comparison_row(run) for run in runs])
    numeric = ["accuracy", "weighted_f1", "macro_f1", "gap", "train_s", "infer_s"]
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{column} came out as {frame[column].dtype}, not numeric")
    return frame


def per_class_matrix(runs, *, task: str):
    """Per-class F1 for one task, one column per run.

    Averaging is what hides a class scoring zero, so the classes are the rows and
    nothing is aggregated. Classes are ordered by the worst score any run gave
    them, so the ones that fail are at the top.

    Support gets one column per split rather than one column overall, because
    runs on different splits are scored on different test partitions and a single
    support column would attach one partition's row counts to another's scores.
    """
    import pandas as pd

    chosen = [run for run in runs if run["config"]["task"] == task]
    if not chosen:
        raise ValueError(f"no run in this set has task {task!r}")

    scores = {run["config"]["run_id"]: run["metrics"]["per_class_f1"] for run in chosen}
    frame = pd.DataFrame(scores)
    frame.index.name = "label"

    support_columns = []
    for split in dict.fromkeys(run["config"]["split"] for run in chosen):
        support = next(r["metrics"]["support"] for r in chosen if r["config"]["split"] == split)
        name = f"n_test_{split}"
        frame[name] = pd.Series(support)
        support_columns.append(name)

    frame = frame.sort_values(list(scores), ascending=True).fillna(0)
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{column} came out as {frame[column].dtype}, not numeric")
    for column in support_columns:
        frame[column] = frame[column].astype("int64")
    return frame.reset_index()


def stratified_subsample(labels, *, cap, seed):
    """An index into `labels` keeping at most `cap` rows and every class's share of them.

    Each class keeps the same proportion of the whole that it had before, so the
    imbalance the corpus has is the imbalance the sample has, and a score taken on
    the sample is a score on the same problem. A class small enough to round to
    nothing keeps one row, because a class with no rows cannot be scored at all
    and losing one silently is worse than the sample being slightly larger.
    """
    labels = np.asarray(labels)
    if cap is None or len(labels) <= int(cap):
        return np.arange(len(labels))
    rng = np.random.default_rng(int(seed))
    share = float(cap) / len(labels)
    keep = []
    for value in np.unique(labels):
        where = np.flatnonzero(labels == value)
        n_keep = min(len(where), max(1, int(round(len(where) * share))))
        keep.append(rng.choice(where, size=n_keep, replace=False))
    return np.sort(np.concatenate(keep))
