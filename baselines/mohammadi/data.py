"""The preprocessing the reproduced baseline runs on its own data.

This is deliberately separate from `src/preprocessing.py`. The project's
preprocessing was built around a split that holds whole recordings apart and a
scaler fitted on that split's training partition. The reproduction is run on the
split the dataset was distributed with, and its scaler is fitted on that split's
training side. Sharing code between the two would eventually mean sharing a
decision, and the whole value of the reproduction is that it was produced the way
the original was.

What happens to a row here is the ordinary thing: read the numeric columns,
standardise each one to mean zero and unit variance using statistics taken from
training rows only, and encode the class name as an integer. Nothing is resampled,
nothing is reweighted and no row is dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PREPROCESSING = {
    "scaler": "StandardScaler",
    "scaler_fitted_on": "the training side of the split being used, and nothing else",
    "labels": "the class of the capture file the row was read from, encoded as an integer",
    "resampling": None,
    "class_weighting": None,
    "rows_dropped": None,
}


def read_block(block: dict, features, *, max_rows: int | None = None, dtype=np.float32):
    """The feature columns of one block of rows, in the order `features` gives.

    A block is a run of consecutive rows in one capture file. For the distributed
    split a block is a whole file, but the row range is honoured anyway so the
    same function can read part of one.
    """
    features = list(features)
    n_rows = int(block["n_rows"]) if max_rows is None else min(int(block["n_rows"]), int(max_rows))
    start = int(block["start"])

    frame = pd.read_csv(
        block["path"],
        usecols=features,
        skiprows=range(1, start + 1) if start else None,
        nrows=n_rows,
    )
    frame = frame[features]

    where = f"{block['file']} rows {start:,}-{start + n_rows:,}"
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            raise TypeError(f"{where}: {column} is category dtype")
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{where}: {column} is {frame[column].dtype}, not numeric")
    if len(frame) != n_rows:
        raise ValueError(f"{where}: asked for {n_rows:,} rows and read {len(frame):,}")

    values = frame.to_numpy(dtype=dtype)
    if not np.isfinite(values).all():
        column = features[int(np.argmax(~np.isfinite(values).all(axis=0)))]
        raise ValueError(f"{where}: {column} holds a value that is not finite")
    return values


def load_partition(
    blocks,
    features,
    *,
    classes,
    max_rows_per_block: int | None = None,
    dtype=np.float32,
    progress=None,
):
    """Read every block of one partition into a single array.

    The labels come back as integer positions into `classes` rather than as
    strings, because seven million strings cost more than the rows they describe.
    `classes` is passed in rather than taken from the blocks so that the training
    and test partitions encode the same class to the same integer.

    The array is allocated once at its final size from the row counts the split
    already knows, so nothing is grown row by row and the memory needed is known
    before the first file is opened.
    """
    features = list(features)
    classes = np.asarray(classes, dtype=str)
    position = {name: i for i, name in enumerate(classes.tolist())}

    planned = [
        int(b["n_rows"]) if max_rows_per_block is None else min(int(b["n_rows"]), int(max_rows_per_block))
        for b in blocks
    ]
    total = int(sum(planned))
    X = np.empty((total, len(features)), dtype=dtype)
    y = np.empty(total, dtype=np.int16)

    at = 0
    for i, (block, n_rows) in enumerate(zip(blocks, planned), start=1):
        if n_rows == 0:
            continue
        if block["label"] not in position:
            raise KeyError(f"{block['file']} has label {block['label']!r}, which is not in classes")
        values = read_block(block, features, max_rows=n_rows, dtype=dtype)
        X[at : at + n_rows] = values
        y[at : at + n_rows] = position[block["label"]]
        at += n_rows
        del values
        if progress:
            progress(i, len(blocks), block, n_rows)

    if at != total:
        raise ValueError(f"planned {total:,} rows and read {at:,}")
    return X, y


def fit_scaler(X, *, chunk: int = 500_000) -> StandardScaler:
    """Fit a StandardScaler over `X`, a chunk at a time.

    Fitting incrementally gives the same mean and variance as fitting on the
    whole array at once and does not need a second copy of several million rows
    in double precision to do it. The caller passes training rows and nothing
    else, which is the whole of the guarantee that held-out rows had no part in
    the statistics.
    """
    scaler = StandardScaler()
    for start in range(0, len(X), int(chunk)):
        scaler.partial_fit(X[start : start + int(chunk)].astype("float64", copy=False))
    return scaler


def transform_in_place(X, scaler, *, chunk: int = 500_000):
    """Standardise `X` using an already fitted scaler, writing back into `X`."""
    for start in range(0, len(X), int(chunk)):
        stop = start + int(chunk)
        X[start:stop] = scaler.transform(X[start:stop].astype("float64", copy=False)).astype(
            X.dtype, copy=False
        )
    return X


def encode_labels(labels, classes):
    """Class names to integer positions in `classes`."""
    classes = np.asarray(classes, dtype=str)
    position = {name: i for i, name in enumerate(classes.tolist())}
    return np.asarray([position[str(v)] for v in np.asarray(labels).astype(str)], dtype="int64")


def scaler_statistics(scaler, features) -> pd.DataFrame:
    """The fitted mean and scale per feature, as a table that can be printed."""
    return pd.DataFrame(
        {
            "feature": list(features),
            "mean": np.asarray(scaler.mean_, dtype="float64"),
            "scale": np.asarray(scaler.scale_, dtype="float64"),
        }
    )


def attach_paths(blocks, directories):
    """Give each block from `splits.json` the path of the file it names.

    `splits.json` records file names rather than paths, because where the corpus
    is mounted is not part of the split. `directories` is a list of places to
    look, tried in order.
    """
    directories = [Path(d) for d in directories]
    out = []
    for block in blocks:
        path = next((d / block["file"] for d in directories if (d / block["file"]).exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"{block['file']} is in the split and is not in any of "
                f"{[str(d) for d in directories]}"
            )
        out.append({**block, "path": str(path)})
    return out
