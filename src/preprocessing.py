"""Reading rows out of a split, scaling them, and cutting them into sequences.

Everything here works block by block. A block is a run of consecutive rows in
one file, which is what `src.splits` produces, and it is the unit that makes the
two rules below hold without anything having to remember them:

The scaler sees training rows only, because it is handed training blocks only.
A sequence never crosses a file boundary, because it is built inside one block.

The functions are written so that a fast pass and a full pass differ in how many
rows they read and in nothing else. Both call the same code with the same
window, the same stride and the same scaler class; the fast pass passes a cap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SEQUENCE_DTYPE = np.float32


# --------------------------------------------------------------------------
# choosing columns
# --------------------------------------------------------------------------


def select_features(all_columns, *, drop: dict) -> tuple[list[str], pd.DataFrame]:
    """Keep every column except the ones named in `drop`, which maps column to reason.

    Returns the kept list and a table of what went, so the reason is printed
    from the same object the decision was made from rather than retyped under it.
    """
    unknown = [c for c in drop if c not in list(all_columns)]
    if unknown:
        raise KeyError(f"asked to drop columns that are not in the data: {unknown}")
    kept = [c for c in all_columns if c not in drop]
    dropped = pd.DataFrame(
        [{"column": c, "reason": reason} for c, reason in drop.items()],
        columns=["column", "reason"],
    )
    return kept, dropped


def exclude_family(features, family_members) -> list[str]:
    """The feature list with one family taken out.

    The family list is intersected with the live columns rather than subtracted
    from them blindly, so a family that names a column which has already been
    dropped does not raise and does not silently remove something else.
    """
    excluded = set(features) & set(family_members)
    return [c for c in features if c not in excluded]


# --------------------------------------------------------------------------
# reading a block
# --------------------------------------------------------------------------


def assert_numeric(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    """Fail at construction, naming the column, rather than seven frames deep."""
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            raise TypeError(f"{where}: {column} is category dtype")
        if not pd.api.types.is_numeric_dtype(dtype):
            raise TypeError(f"{where}: {column} is {dtype}, not numeric")
    return frame


def read_block(block: dict, features, *, max_rows: int | None = None) -> pd.DataFrame:
    """The rows of one block, as the feature columns in a fixed order.

    `max_rows` reads the head of the block instead of all of it. The rows of a
    block are contiguous, so the head is a slice of the session rather than a
    sample of it, which is fine for checking shapes and wrong for measuring
    anything.
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
    assert_numeric(frame, where)
    if len(frame) != n_rows:
        raise ValueError(f"{where}: asked for {n_rows:,} rows and read {len(frame):,}")
    values = frame.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        column = features[int(np.argmax(~np.isfinite(values).all(axis=0)))]
        raise ValueError(f"{where}: {column} holds a value that is not finite")
    return frame


# --------------------------------------------------------------------------
# the scaler
# --------------------------------------------------------------------------


def fit_scaler(blocks, features, *, max_rows_per_block: int | None = None, progress=None):
    """Fit a StandardScaler over the blocks it is given, one block at a time.

    Fitting incrementally is what makes it possible to state where the numbers
    came from. The scaler never sees a row that was not in a block handed to
    this function, so passing training blocks only is the whole of the
    guarantee, and the table returned says which blocks those were.
    """
    features = list(features)
    scaler = StandardScaler()
    used = []
    for i, block in enumerate(blocks, start=1):
        frame = read_block(block, features, max_rows=max_rows_per_block)
        scaler.partial_fit(frame.to_numpy(dtype="float64"))
        used.append(
            {
                "partition": block["partition"],
                "label": block["label"],
                "file": block["file"],
                "start": int(block["start"]),
                "rows_read": int(len(frame)),
            }
        )
        if progress:
            progress(i, len(blocks), block)
        del frame

    table = pd.DataFrame(used)
    if not pd.api.types.is_numeric_dtype(table["rows_read"]):
        raise TypeError(f"rows_read came out as {table['rows_read'].dtype}, not numeric")
    return scaler, table


def scaler_statistics(scaler, features) -> pd.DataFrame:
    """The fitted mean and scale per feature, as a table that can be printed."""
    return pd.DataFrame(
        {
            "feature": list(features),
            "mean": np.asarray(scaler.mean_, dtype="float64"),
            "variance": np.asarray(scaler.var_, dtype="float64"),
            "scale": np.asarray(scaler.scale_, dtype="float64"),
        }
    )


# --------------------------------------------------------------------------
# records and sequences
# --------------------------------------------------------------------------

CODE_DTYPE = np.int16
ROW_DTYPE = np.int32


def sequence_starts(n_rows: int, *, window: int, stride: int, cap: int | None = None):
    """Where each window begins inside a block of `n_rows` rows."""
    if n_rows < window:
        return np.empty(0, dtype="int64")
    count = (n_rows - window) // stride + 1
    if cap is not None:
        count = min(count, int(cap))
    return np.arange(count, dtype="int64") * stride


def _codes(values) -> tuple[np.ndarray, dict]:
    """A sorted lookup array of the distinct values, and value to position."""
    lookup = np.asarray(sorted({str(v) for v in values}), dtype=str)
    return lookup, {name: i for i, name in enumerate(lookup.tolist())}


def build_partition(
    plan: pd.DataFrame,
    features,
    scaler,
    *,
    window: int,
    stride: int,
    dtype=SEQUENCE_DTYPE,
    progress=None,
) -> dict:
    """Read every block once and produce both the records and the sequences.

    `plan` is what `splits.build_plan` returns, so how many rows and how many
    windows each block contributes is known before anything is read, and both
    arrays are allocated once at their final size rather than grown.

    The records are the scaled rows themselves, in the order they appear in the
    file. The sequences are windows cut from those same rows. Reading the block
    once for both is not only cheaper, it is what makes them the same numbers:
    there is no second read that could pick up a different scaler or a different
    slice.

    Rows are ordered by their position in the file because there is no timestamp
    column in this dataset and position is the only ordering available. A window
    is therefore `window` rows as the extractor wrote them, and it carries the
    label of the file it came from, which is the class of that whole recording.

    Labels, recordings and file names are stored as integer codes against a
    lookup array of strings. Nothing is ever a pandas Categorical, and the codes
    keep a per-row label from costing more memory than the row.
    """
    features = list(features)
    plan = plan.reset_index(drop=True)
    n_features = len(features)
    total_rows = int(plan["rows_to_read"].sum())
    total_sequences = int(plan["n_sequences"].sum())

    classes, class_at = _codes(plan["label"])
    recordings, recording_at = _codes(plan["recording"])
    files, file_at = _codes(plan["file"])

    records = {
        "X": np.empty((total_rows, n_features), dtype=dtype),
        "y": np.empty(total_rows, dtype=CODE_DTYPE),
        "recording": np.empty(total_rows, dtype=CODE_DTYPE),
        "source_file": np.empty(total_rows, dtype=CODE_DTYPE),
        "row": np.empty(total_rows, dtype=ROW_DTYPE),
    }
    sequences = {
        "X": np.empty((total_sequences, int(window), n_features), dtype=dtype),
        "y": np.empty(total_sequences, dtype=CODE_DTYPE),
        "recording": np.empty(total_sequences, dtype=CODE_DTYPE),
        "source_file": np.empty(total_sequences, dtype=CODE_DTYPE),
        "start_row": np.empty(total_sequences, dtype=ROW_DTYPE),
    }

    at_row, at_sequence = 0, 0
    for i, row in enumerate(plan.itertuples(), start=1):
        block = {
            "path": row.path,
            "file": row.file,
            "start": int(row.start),
            "n_rows": int(row.n_rows),
            "partition": row.partition,
            "label": row.label,
        }
        n_read = int(row.rows_to_read)
        n_sequences = int(row.n_sequences)
        if n_read == 0:
            continue

        frame = read_block(block, features, max_rows=n_read)
        values = scaler.transform(frame.to_numpy(dtype="float64")).astype(dtype, copy=False)
        del frame

        records["X"][at_row : at_row + n_read] = values
        records["y"][at_row : at_row + n_read] = class_at[row.label]
        records["recording"][at_row : at_row + n_read] = recording_at[row.recording]
        records["source_file"][at_row : at_row + n_read] = file_at[row.file]
        records["row"][at_row : at_row + n_read] = np.arange(
            int(row.start), int(row.start) + n_read, dtype=ROW_DTYPE
        )
        at_row += n_read

        if n_sequences:
            offsets = sequence_starts(len(values), window=window, stride=stride, cap=n_sequences)
            if len(offsets) != n_sequences:
                raise ValueError(
                    f"{row.file} rows {row.start}: planned {n_sequences} windows and the "
                    f"{len(values):,} rows read give {len(offsets)}"
                )
            windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
            sequences["X"][at_sequence : at_sequence + n_sequences] = windows[offsets].transpose(
                0, 2, 1
            )
            sequences["y"][at_sequence : at_sequence + n_sequences] = class_at[row.label]
            sequences["recording"][at_sequence : at_sequence + n_sequences] = recording_at[
                row.recording
            ]
            sequences["source_file"][at_sequence : at_sequence + n_sequences] = file_at[row.file]
            sequences["start_row"][at_sequence : at_sequence + n_sequences] = offsets + int(
                row.start
            )
            at_sequence += n_sequences
            del windows
        del values

        if progress:
            progress(i, len(plan), block, n_read, n_sequences)

    if at_row != total_rows or at_sequence != total_sequences:
        raise ValueError(
            f"planned {total_rows:,} rows and {total_sequences:,} sequences, wrote "
            f"{at_row:,} and {at_sequence:,}"
        )

    return {
        "features": features,
        "window": int(window),
        "stride": int(stride),
        "classes": classes,
        "recordings": recordings,
        "files": files,
        "records": records,
        "sequences": sequences,
    }


def counts_by_class(built: dict, kind: str) -> pd.DataFrame:
    """How many rows, or how many sequences, came out per class."""
    codes = np.asarray(built[kind]["y"])
    classes = list(np.asarray(built["classes"]).tolist())
    counted = np.bincount(codes.astype("int64"), minlength=len(classes))
    frame = pd.DataFrame({"label": classes, "n": counted.astype("int64")})
    if not pd.api.types.is_numeric_dtype(frame["n"]):
        raise TypeError("counts came out non-numeric")
    return frame


def drop_feature_axis(built: dict, kind: str, keep_features) -> dict:
    """The same arrays with the feature axis cut down to `keep_features`.

    The last axis is the feature axis for both kinds, so records of shape
    (rows, features) and sequences of shape (windows, window, features) are cut
    the same way.
    """
    keep_features = list(keep_features)
    index = [built["features"].index(name) for name in keep_features]
    part = dict(built[kind])
    part["X"] = built[kind]["X"][..., index]
    part["features"] = keep_features
    return part


# --------------------------------------------------------------------------
# writing, and saying what was written
# --------------------------------------------------------------------------


def save_arrays(path: Path | str, built: dict, kind: str, part: dict | None = None) -> Path:
    """Write one partition's records, or its sequences, as a single npz.

    The lookup arrays go in with them, so a later notebook reads one file and
    has everything it needs to turn a code back into a class name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = built[kind] if part is None else part
    features = part.get("features", built["features"])
    np.savez(
        path,
        classes=np.asarray(built["classes"], dtype=str),
        recordings=np.asarray(built["recordings"], dtype=str),
        files=np.asarray(built["files"], dtype=str),
        features=np.asarray(list(features), dtype=str),
        window=np.asarray(built["window"]),
        stride=np.asarray(built["stride"]),
        **{name: value for name, value in part.items() if name != "features"},
    )
    return path


def array_digest(array) -> str:
    """A hash of an array's contents, taken before it is written.

    Hashing the array rather than the file it went into means a later notebook
    can load the npz, hash what it got, and compare, without depending on how
    the container was written.
    """
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def file_digest(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def describe_written(path: Path | str, *, arrays: dict | None = None) -> dict:
    """What was written to one path: its size, and a hash of it or of its arrays."""
    path = Path(path)
    entry = {"file": path.name, "bytes": int(path.stat().st_size)}
    if arrays is None:
        entry["sha256"] = file_digest(path)
    else:
        entry["arrays"] = {
            name: {
                "shape": list(np.shape(value)),
                "dtype": str(np.asarray(value).dtype),
                "sha256": array_digest(value),
            }
            for name, value in arrays.items()
        }
    return entry
