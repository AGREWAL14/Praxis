"""The capture-file naming rule, and the per-capture scans built on top of it.

`parse_capture` is the single place the project decides what class a file belongs
to and which capture it came from. Every notebook from NB02 onward imports it
rather than re-deriving the rule, so that a change to the naming convention is a
change in one file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .inventory import count_data_rows

# --------------------------------------------------------------------------
# the naming rule
# --------------------------------------------------------------------------


def parse_capture(filename):
    """Split a capture filename into its capture id, class label, and partition.

    `TCP_IP-DDoS-ICMP1_train.pcap.csv` -> capture ICMP1 of class DDoS-ICMP, train.

    The trailing digits are the capture number within a class, so stripping them
    gives the class. The `TCP_IP-` prefix is a grouping in the filenames rather
    than part of the class name, and `ARP_Spoofing` is the filename spelling of
    the class called Spoofing.
    """
    n = os.path.basename(filename).replace('.pcap.csv', '')
    partition = 'test' if n.endswith('_test') else 'train'
    n = re.sub(r'_(train|test)$', '', n)
    capture_id = n
    label = re.sub(r'\d+$', '', n).replace('TCP_IP-', '')
    if label == 'ARP_Spoofing':
        label = 'Spoofing'
    return {'capture_id': capture_id, 'label': label, 'partition': partition}


def group_six(label: str) -> str:
    """The 6-class grouping. Order matters: MQTT-DDoS is MQTT, not DDoS."""
    if label == "Benign":
        return "Benign"
    if label.startswith("DDoS"):
        return "DDoS"
    if label.startswith("DoS"):
        return "DoS"
    if label.startswith("MQTT-"):
        return "MQTT"
    if label.startswith("Recon-"):
        return "Recon"
    if label == "Spoofing":
        return "Spoofing"
    return "UNMAPPED"


def group_two(label: str) -> str:
    """The 2-class grouping."""
    return "Benign" if label == "Benign" else "Attack"


# --------------------------------------------------------------------------
# reading captures
# --------------------------------------------------------------------------


def read_capture(
    path: Path | str, *, frac: float = 1.0, known_rows: int | None = None, floor: int = 1000
) -> pd.DataFrame:
    """Read a capture file, or its opening `frac` of rows.

    Reading one file at a time is what makes a subsample stratified by class,
    because one file holds exactly one class. The subsample is the head of the
    file rather than a random draw from it, because stopping the reader early is
    the only version of this that is actually cheap. Rows inside a capture are in
    the order the extractor wrote them, so the head is a contiguous slice of the
    session and not a sample of it. That is fine for a screen and wrong for an
    estimate, which is why nothing quantitative is taken from it.
    """
    if frac >= 1.0:
        return pd.read_csv(path)
    total = known_rows if known_rows is not None else count_data_rows(path)
    return pd.read_csv(path, nrows=max(floor, int(round(total * frac))))


def column_stats(df: pd.DataFrame) -> dict:
    """Per column: how many distinct values it takes, and the value if only one."""
    stats = {}
    for col in df.columns:
        values = pd.unique(df[col].dropna())
        stats[col] = {
            "n_unique": int(len(values)),
            "value": values[0] if len(values) == 1 else None,
        }
    return stats


def scan_captures(paths, *, frac: float = 1.0, known_rows: dict | None = None, progress_every: int = 10):
    """One pass over the capture files, returning rows read and per-column stats."""
    scans = {}
    for i, path in enumerate(paths, start=1):
        name = Path(path).name
        df = read_capture(path, frac=frac, known_rows=(known_rows or {}).get(name))
        scans[name] = {
            "n_rows_read": int(len(df)),
            "columns": column_stats(df),
            "near": near_constancy_stats(df),
        }
        if progress_every and i % progress_every == 0:
            print(f"  scanned {i}/{len(paths)} files")
        del df
    return scans


# --------------------------------------------------------------------------
# the near-constant check
# --------------------------------------------------------------------------


def near_constancy_stats(df: pd.DataFrame, *, cardinality_cap: int = 200) -> dict:
    """Per column: rows, nulls, zeros, and the counts a modal share is built from.

    Taken from the same dataframe the constancy check already reads, so this costs
    a pass over the columns rather than a second pass over the corpus.

    Zero counts and null counts are exact, and they sum across files exactly. Value
    counts are kept in full for a column taking fewer than `cardinality_cap`
    distinct values, which makes its global modal share exact once the files are
    summed. Above the cap only the commonest value and its count are kept, so such
    a column's global modal share is a lower bound, and `exact_value_counts` says
    which of the two it is. A column dense enough to pass the cap is not a
    near-constant column, so the approximation never falls on the question being
    asked.

    The cap also bounds what the scan holds in memory. Seventy-two files and
    forty-five columns at two hundred values each is the ceiling, and a column
    above the cap contributes one entry rather than thousands.
    """
    stats = {}
    for col in df.columns:
        series = df[col]
        present = series.dropna()
        counts = present.value_counts()
        exact = len(counts) <= cardinality_cap
        stats[col] = {
            "n_rows": int(len(series)),
            "n_null": int(series.isna().sum()),
            "n_zero": int((present == 0).sum()),
            "n_unique": int(len(counts)),
            "exact_value_counts": bool(exact),
            "value_counts": (
                {_hashable(v): int(c) for v, c in counts.items()}
                if exact
                else {_hashable(counts.index[0]): int(counts.iloc[0])}
            ),
        }
    return stats


def near_constancy_report(scans: dict) -> pd.DataFrame:
    """Zero share and modal share per column, summed over every file scanned.

    A column can be near enough to constant to carry nothing while still varying,
    and the constant-column check cannot see that: it asks only whether a column
    ever moves, so one value holding 93% of the rows and one value holding 100% of
    them land on opposite sides of it. This is the same question asked as a share
    rather than as a yes or no.

    `modal_share` is computed over the non-null rows, because a null is not the
    modal value and counting it in the denominator would report a column as less
    constant than it is. `zero_share` is computed over all rows, because a zero is
    a value the model reads.
    """
    columns = sorted({c for s in scans.values() for c in s.get("near", {})})

    rows = []
    for col in columns:
        entries = [s["near"][col] for s in scans.values() if col in s.get("near", {})]
        if not entries:
            continue
        n_rows = sum(e["n_rows"] for e in entries)
        n_null = sum(e["n_null"] for e in entries)
        n_zero = sum(e["n_zero"] for e in entries)
        exact = all(e["exact_value_counts"] for e in entries)

        totals: dict = {}
        for entry in entries:
            for value, count in entry["value_counts"].items():
                totals[value] = totals.get(value, 0) + count
        modal_value, modal_count = max(totals.items(), key=lambda pair: pair[1])
        present = n_rows - n_null

        rows.append(
            {
                "column": col,
                "n_rows": int(n_rows),
                "n_null": int(n_null),
                "zero_share": n_zero / max(n_rows, 1),
                "modal_value": modal_value,
                "modal_share": modal_count / max(present, 1),
                "modal_share_exact": bool(exact),
                "n_files": len(entries),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["modal_share", "zero_share"], ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# the constant-within-capture check
# --------------------------------------------------------------------------


def capture_constant_columns(scans: dict) -> pd.DataFrame:
    """Find columns that hold one value inside a capture but differ between captures.

    A column like that carries the identity of the recording session. A model
    reading it can tell which file a row came from, which is not the same as
    telling which attack it represents.

    A capture session is stored as two files, one per side of the distributed
    split, so a value fixed for a session shows up as constant in both of them.

    Returns one row per column: in how many files it held a single value, and
    how many distinct values those single values took. A column that varies
    inside its files contributes no values at all, so `distinct_values` of zero
    means it was never constant, not that it is empty. Read the two counts
    together.
    """
    n_files = len(scans)
    columns = sorted({c for s in scans.values() for c in s["columns"]})

    rows = []
    for col in columns:
        constant_in, values = 0, []
        for scan in scans.values():
            stat = scan["columns"].get(col)
            if stat is None:
                continue
            if stat["n_unique"] <= 1:
                constant_in += 1
                if stat["value"] is not None:
                    values.append(stat["value"])
        distinct = sorted({_hashable(v) for v in values})
        rows.append(
            {
                "column": col,
                "constant_in_files": constant_in,
                "of_files": n_files,
                "constant_in_every_file": constant_in == n_files,
                "distinct_values_across_files": len(distinct),
                "values": distinct,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["distinct_values_across_files", "constant_in_files"], ascending=False
    ).reset_index(drop=True)


def constancy_report(scans: dict, recording_of: dict | None = None) -> pd.DataFrame:
    """Separate the two kinds of column that never change inside a recording.

    A recording is whatever grouping of files the caller passes in
    `recording_of`, a map from file name to recording name. The default is one
    file per recording, which is the safest reading: a file is one contiguous
    stretch of capture, whereas a group of files is only a recording if the
    grouping is right.

    Two findings come out of the same counts and they mean opposite things.

    A column holding the same value in every recording is dead weight. It
    separates nothing anywhere and can be dropped without losing information.

    A column holding one value inside each recording and a different value in
    different recordings is the dangerous kind. It is not describing the traffic,
    it is naming the session, and a model reading it can recover which recording
    a row came from rather than which attack it represents.

    `constant_in_recordings` of zero means the column moved inside every
    recording, which is what an ordinary measurement looks like.
    """
    recording_of = recording_of or {name: name for name in scans}

    recordings: dict = {}
    for name, recording in recording_of.items():
        recordings.setdefault(recording, []).append(name)

    columns = sorted({c for s in scans.values() for c in s["columns"]})

    rows = []
    for col in columns:
        constant_in, values, seen_in = 0, [], 0
        for members in recordings.values():
            stats = [scans[m]["columns"].get(col) for m in members if m in scans]
            stats = [s for s in stats if s is not None]
            if not stats:
                continue
            seen_in += 1
            if any(s["n_unique"] > 1 for s in stats):
                continue
            held = {_hashable(s["value"]) for s in stats if s["value"] is not None}
            if len(held) > 1:
                continue  # each file steady on its own, but they disagree with each other
            constant_in += 1
            values.extend(held)

        distinct = sorted(set(values))
        everywhere = constant_in == seen_in and seen_in > 0
        rows.append(
            {
                "column": col,
                "constant_in_recordings": constant_in,
                "of_recordings": seen_in,
                "distinct_values_across_recordings": len(distinct),
                "constant_everywhere": bool(everywhere and len(distinct) <= 1),
                "recording_identifying": bool(everywhere and len(distinct) > 1),
                "values": distinct,
            }
        )

    frame = pd.DataFrame(rows)
    frame["_rank"] = np.where(
        frame["recording_identifying"], 0, np.where(frame["constant_everywhere"], 1, 2)
    )
    frame = frame.sort_values(
        ["_rank", "constant_in_recordings", "distinct_values_across_recordings"],
        ascending=[True, False, False],
    )
    return frame.drop(columns="_rank").reset_index(drop=True)


def _hashable(value):
    """Round floats before de-duplicating so 1.0000000001 and 1.0 are one value."""
    if isinstance(value, (float, np.floating)):
        return round(float(value), 10)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def jsonable(obj):
    """Convert numpy scalars and paths so json.dump will accept the structure."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
