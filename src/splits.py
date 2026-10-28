"""Partitions built from capture files, and the checks that say whether they hold.

Two protocols are built here and both are kept. `shipped_blocks` is the split as
the dataset was distributed: whole files, train and test, nothing rearranged.
`two_tier_blocks` is the one the rest of the project uses. It holds whole
recordings out for the classes that were recorded more than once, and cuts the
single recording into contiguous blocks for the classes that were recorded once.

Both return the same structure, a list of blocks. A block is a run of
consecutive rows inside one file: which file it is in, where it starts, where it
stops, what class it holds and which partition it belongs to. A whole file is a
block covering all of it. Everything downstream reads blocks, so the two
protocols cannot drift apart in how they are consumed, and a sequence built
inside a block cannot cross a file boundary because a block never does.

Nothing here reads a CSV. The row counts come from the inventory, so a split can
be built, printed and checked before any data is loaded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .captures import parse_capture

# The order partitions are printed and compared in.
PARTITION_ORDER = ["train", "val", "test"]

# Files of one class are read in this order: the distributed train side first,
# then the test side, then by chunk number.
_SHIPPED_RANK = {"train": 0, "test": 1}
_TRAILING_DIGITS = re.compile(r"(\d+)$")


# --------------------------------------------------------------------------
# recordings
# --------------------------------------------------------------------------


def recording_id(filename) -> str:
    """The recording a file holds: the capture id with its partition attached.

    The chunk numbers restart in each partition, so `TCP_IP-DDoS-ICMP1_test` is
    not a second helping of `TCP_IP-DDoS-ICMP1_train`. It is a different session
    that happens to carry the number one, and dropping the partition from the
    identifier would merge two sessions into one.
    """
    meta = parse_capture(Path(filename).name)
    return f"{meta['capture_id']}_{meta['partition']}"


def chunk_number(capture_id: str) -> int:
    """The trailing number of a capture id, or 0 where it carries none."""
    found = _TRAILING_DIGITS.search(str(capture_id))
    return int(found.group(1)) if found else 0


def recording_table(paths, rows_per_file: dict) -> pd.DataFrame:
    """One row per capture file, in the order the files of a class are read.

    `order` counts from 1 within a class and is what the two-tier split holds
    recordings out by, so the assignment can be read off the table.
    """
    rows = []
    for path in paths:
        name = Path(path).name
        if name not in rows_per_file:
            raise KeyError(f"{name} has no row count in the inventory")
        meta = parse_capture(name)
        rows.append(
            {
                "file": name,
                "path": str(path),
                "label": str(meta["label"]),
                "capture_id": str(meta["capture_id"]),
                "shipped_partition": str(meta["partition"]),
                "recording": recording_id(name),
                "chunk": chunk_number(meta["capture_id"]),
                "n_rows": int(rows_per_file[name]),
            }
        )

    table = pd.DataFrame(rows)
    table["shipped_rank"] = table["shipped_partition"].map(_SHIPPED_RANK)
    table = table.sort_values(["label", "shipped_rank", "chunk", "file"]).reset_index(drop=True)
    table["order"] = table.groupby("label").cumcount() + 1

    for column in ("n_rows", "chunk", "shipped_rank", "order"):
        if not pd.api.types.is_numeric_dtype(table[column]):
            raise TypeError(f"{column} came out as {table[column].dtype}, not numeric")
    for column in ("label", "file", "recording", "capture_id", "shipped_partition"):
        if isinstance(table[column].dtype, pd.CategoricalDtype):
            raise TypeError(f"{column} is category dtype; labels and identifiers stay strings")
    return table


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


def _block(row, *, protocol: str, partition: str, start: int, stop: int, tier: str) -> dict:
    start, stop = int(start), int(stop)
    return {
        "protocol": protocol,
        "partition": str(partition),
        "label": str(row.label),
        "tier": str(tier),
        "recording": str(row.recording),
        "file": str(row.file),
        "path": str(row.path),
        "file_order": int(row.order),
        "start": start,
        "stop": stop,
        "n_rows": stop - start,
        "whole_file": bool(start == 0 and stop == int(row.n_rows)),
    }


def shipped_blocks(table: pd.DataFrame, tier_of: dict) -> list[dict]:
    """The split as distributed: every file whole, on the side its name gives."""
    return [
        _block(
            row,
            protocol="shipped",
            partition=row.shipped_partition,
            start=0,
            stop=row.n_rows,
            tier=tier_of[row.label],
        )
        for row in table.itertuples()
    ]


def two_tier_blocks(
    table: pd.DataFrame,
    tier_of: dict,
    *,
    ratios=(0.70, 0.15, 0.15),
    tier_a: str = "A",
) -> list[dict]:
    """The split the project uses, built one way for each kind of class.

    A class recorded several times keeps whole recordings apart: the earliest
    recordings train, the next-to-last validates, the last tests. Nothing from a
    recording used for training appears on either held-out side.

    A class recorded once cannot be split that way, so its single recording is
    cut into three contiguous blocks by row position. The cut is by position and
    never at random, because rows next to each other in a file can come from the
    same burst of traffic and a random split would put related rows on both
    sides of it. Where the recording was distributed as two files, the positions
    run through the first file and then the second, so a cut can land inside
    either one.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios sum to {sum(ratios)}, not 1")
    train_ratio, val_ratio = ratios[0], ratios[1]

    blocks = []
    for label, group in table.groupby("label", sort=True):
        group = group.sort_values("order")
        tier = tier_of[label]

        if tier == tier_a:
            if len(group) < 3:
                raise ValueError(
                    f"{label} is in tier {tier_a} with {len(group)} recordings; holding one out "
                    "for validation and one for test needs at least three"
                )
            rows = list(group.itertuples())
            for position, row in enumerate(rows):
                if position < len(rows) - 2:
                    partition = "train"
                elif position == len(rows) - 2:
                    partition = "val"
                else:
                    partition = "test"
                blocks.append(
                    _block(
                        row,
                        protocol="two_tier",
                        partition=partition,
                        start=0,
                        stop=row.n_rows,
                        tier=tier,
                    )
                )
            continue

        total = int(group["n_rows"].sum())
        first_cut = int(total * train_ratio)
        second_cut = int(total * (train_ratio + val_ratio))
        ranges = [
            ("train", 0, first_cut),
            ("val", first_cut, second_cut),
            ("test", second_cut, total),
        ]

        offset = 0
        for row in group.itertuples():
            for partition, low, high in ranges:
                start = max(low, offset)
                stop = min(high, offset + int(row.n_rows))
                if stop > start:
                    blocks.append(
                        _block(
                            row,
                            protocol="two_tier",
                            partition=partition,
                            start=start - offset,
                            stop=stop - offset,
                            tier=tier,
                        )
                    )
            offset += int(row.n_rows)

    return blocks


def blocks_frame(blocks) -> pd.DataFrame:
    """The blocks as a table, ordered class, then partition, then position in the recording.

    Position rather than file name, so the blocks of one class read in the order
    the rows were recorded. A class whose recording was distributed as two files
    has a validation range that ends in the first file and continues into the
    second, and sorting by name would print those two pieces the wrong way round.
    """
    frame = pd.DataFrame(blocks)
    if frame.empty:
        return frame
    frame["partition_rank"] = frame["partition"].map(
        {name: i for i, name in enumerate(PARTITION_ORDER)}
    )
    frame = frame.sort_values(
        ["label", "partition_rank", "file_order", "start"]
    ).reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(frame["n_rows"]):
        raise TypeError(f"n_rows came out as {frame['n_rows'].dtype}, not numeric")
    return frame.drop(columns="partition_rank")


def describe_blocks(blocks) -> pd.DataFrame:
    """One printable line per block: the file, the rows taken, and where they sit."""
    frame = blocks_frame(blocks)
    described = frame.assign(
        rows=frame["stop"] - frame["start"],
        range=frame.apply(
            lambda r: "whole file" if r["whole_file"] else f"rows {r['start']:,}-{r['stop']:,}",
            axis=1,
        ),
    )
    return described[["label", "tier", "partition", "recording", "file", "range", "rows"]]


# --------------------------------------------------------------------------
# what the split came out as
# --------------------------------------------------------------------------


def rows_per_partition(blocks) -> dict:
    frame = blocks_frame(blocks)
    counts = frame.groupby("partition")["n_rows"].sum()
    return {name: int(counts.get(name, 0)) for name in PARTITION_ORDER if name in set(frame["partition"])}


def class_proportions(blocks) -> pd.DataFrame:
    """Rows per class per partition, and each class's share of its partition.

    The share is what says whether the partitions hold the same mix. A class
    that is 3% of training and 12% of test is a different problem on each side.
    """
    frame = blocks_frame(blocks)
    partitions = [p for p in PARTITION_ORDER if p in set(frame["partition"])]
    counts = (
        frame.pivot_table(index="label", columns="partition", values="n_rows", aggfunc="sum")
        .reindex(columns=partitions)
        .fillna(0)
        .astype("int64")
    )
    shares = counts.divide(counts.sum(axis=0), axis=1) * 100.0
    shares.columns = [f"{name}_pct" for name in shares.columns]
    table = pd.concat([counts, shares.round(3)], axis=1)
    table.insert(0, "rows", counts.sum(axis=1))
    return table.reset_index()


def build_plan(blocks, *, window: int, stride: int, cap: int | None = None) -> pd.DataFrame:
    """What each block will contribute, worked out before any row is read.

    `n_available` is how many windows the block holds, which is arithmetic on
    its row range, so both passes agree on it even when only one of them builds
    them all. `n_sequences` is how many will actually be built, which is where a
    cap bites. `rows_used` is how many rows those windows span, and
    `rows_to_read` is how many rows to read: every row of the block when nothing
    is capped, and only the rows the windows need when something is.

    A block shorter than one window yields no windows and still holds rows.
    """
    frame = blocks_frame(blocks)
    available = ((frame["n_rows"] - window) // stride + 1).clip(lower=0)
    frame = frame.assign(n_available=available.astype("int64"))
    frame["n_sequences"] = (
        frame["n_available"] if cap is None else frame["n_available"].clip(upper=int(cap))
    )
    frame["rows_used"] = (
        (window + stride * (frame["n_sequences"] - 1)).where(frame["n_sequences"] > 0, 0)
    ).astype("int64")
    frame["rows_to_read"] = frame["n_rows"] if cap is None else frame["rows_used"]
    for column in ("n_available", "n_sequences", "rows_used", "rows_to_read"):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{column} came out as {frame[column].dtype}, not numeric")
    return frame


# --------------------------------------------------------------------------
# checking a split
# --------------------------------------------------------------------------


def validate_split(
    blocks,
    *,
    classes,
    expected_partitions,
    total_rows: int,
    rows_per_file: dict,
    disjoint_classes=(),
) -> pd.DataFrame:
    """Every check a split has to pass, each as one line of PASS or FAIL.

    `disjoint_classes` are the classes that were recorded more than once, and
    they are the only ones where holding recordings apart is possible. For the
    rest the same recording is on all three sides by construction, which is the
    limitation the notebook states rather than a failure to be caught here.
    """
    frame = blocks_frame(blocks)
    classes = sorted(str(c) for c in classes)
    expected_partitions = list(expected_partitions)
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    shared = frame[frame["label"].isin(list(disjoint_classes))]
    spread = shared.groupby("recording")["partition"].nunique()
    offenders = sorted(spread[spread > 1].index.tolist())
    add(
        f"no recording in more than one partition, {len(disjoint_classes)} classes",
        not offenders,
        f"{len(spread)} recordings checked" if not offenders else f"on both sides: {offenders}",
    )

    present = frame.groupby("label")["partition"].agg(set).to_dict()
    missing = {
        label: sorted(set(expected_partitions) - present.get(label, set()))
        for label in classes
        if set(expected_partitions) - present.get(label, set())
    }
    add(
        f"every class in all {len(expected_partitions)} partitions",
        not missing,
        f"{len(classes)} classes" if not missing else f"missing: {missing}",
    )

    counted = int(frame["n_rows"].sum())
    add(
        "row counts sum to the dataset total",
        counted == int(total_rows),
        f"{counted:,} rows across {len(frame)} blocks against {int(total_rows):,} in the inventory",
    )

    covered_ok, covered_detail = True, f"{frame['file'].nunique()} files"
    for name, group in frame.groupby("file"):
        expected = int(rows_per_file[name])
        ranges = group.sort_values("start")[["start", "stop"]].to_numpy()
        contiguous = ranges[0][0] == 0 and int(ranges[-1][1]) == expected
        contiguous = contiguous and all(
            int(ranges[i][1]) == int(ranges[i + 1][0]) for i in range(len(ranges) - 1)
        )
        if not contiguous:
            covered_ok = False
            covered_detail = f"{name} is not covered exactly once by its blocks: {ranges.tolist()}"
            break
    add("every file covered exactly once, no gaps or overlaps", covered_ok, covered_detail)

    empty = frame[frame["n_rows"] <= 0]
    add(
        "no empty block",
        empty.empty,
        f"smallest block {int(frame['n_rows'].min()):,} rows" if empty.empty else f"{len(empty)} empty",
    )

    return pd.DataFrame(checks)


def split_document(blocks, table: pd.DataFrame, *, protocol: str, ratios=None) -> dict:
    """The split written out as plain data: file lists and row ranges, by partition.

    This is what gets saved. A later notebook reads it and knows which rows of
    which file belong to which side without rebuilding any of the reasoning.
    """
    frame = blocks_frame(blocks)
    partitions = {}
    for name in PARTITION_ORDER:
        part = frame[frame["partition"] == name]
        if part.empty:
            continue
        partitions[name] = {
            "n_blocks": int(len(part)),
            "n_rows": int(part["n_rows"].sum()),
            "files": sorted(part["file"].unique().tolist()),
            "recordings": sorted(part["recording"].unique().tolist()),
            "blocks": [
                {
                    "file": row.file,
                    "label": row.label,
                    "tier": row.tier,
                    "recording": row.recording,
                    "start": int(row.start),
                    "stop": int(row.stop),
                    "n_rows": int(row.n_rows),
                    "whole_file": bool(row.whole_file),
                }
                for row in part.itertuples()
            ],
        }

    return {
        "protocol": protocol,
        "ratios": list(ratios) if ratios else None,
        "n_blocks": int(len(frame)),
        "n_rows": int(frame["n_rows"].sum()),
        "n_files": int(frame["file"].nunique()),
        "rows_per_partition": rows_per_partition(blocks),
        "recordings_per_partition": {
            name: sorted(frame.loc[frame["partition"] == name, "recording"].unique().tolist())
            for name in PARTITION_ORDER
            if name in set(frame["partition"])
        },
        "partitions": partitions,
        "row_counts_used": {
            row.file: int(row.n_rows) for row in table.itertuples()
        },
    }
