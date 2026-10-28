"""Helpers for reading and describing the raw CICIoMT2024 CSV files.

Nothing in this module trains, fits, or transforms anything. It reads headers,
describes columns, counts rows, and searches column names for particular kinds
of field. The notebook keeps the narrative and the printing; the logic lives
here so it can be reused and tested.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

# --------------------------------------------------------------------------
# config and file discovery
# --------------------------------------------------------------------------


def find_repo_root(start: Path | str | None = None) -> Path:
    """Walk up from `start` until a directory containing config/base.yaml is found."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "base.yaml").exists():
            return candidate
    raise FileNotFoundError(f"no config/base.yaml at or above {here}")


def load_config(repo_root: Path | str) -> dict:
    with open(Path(repo_root) / "config" / "base.yaml") as fh:
        return yaml.safe_load(fh)


def list_csvs(directory: Path | str) -> list[Path]:
    return sorted(Path(directory).glob("*.csv"))


_SPLIT_SUFFIXES = ("_train", "_test", "_val")


def class_name(path: Path | str) -> str:
    """Turn a capture filename into the class label it represents.

    `TCP_IP-DDoS-TCP1_train.pcap.csv` -> `TCP_IP-DDoS-TCP1`
    """
    stem = Path(path).name
    for ext in (".csv", ".pcap"):
        while stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    for suffix in _SPLIT_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def family_name(path: Path | str) -> str:
    """The coarse attack family: everything before the first hyphen of the class name."""
    return class_name(path).split("-")[0]


def split_name(path: Path | str) -> str:
    """Which distributed partition a file belongs to, read from its filename."""
    stem = Path(path).name.lower()
    for suffix in _SPLIT_SUFFIXES:
        if suffix in stem:
            return suffix.lstrip("_")
    return "unknown"


# --------------------------------------------------------------------------
# schema description
# --------------------------------------------------------------------------


def header_of(path: Path | str) -> list[str]:
    """Column names only. Reads no data rows."""
    return pd.read_csv(path, nrows=0).columns.tolist()


def count_data_rows(path: Path | str) -> int:
    """Count data rows by counting newlines, without parsing the file.

    Assumes no newline appears inside a quoted field. `verify_row_count` checks
    that assumption against pandas on one file.
    """
    total = 0
    last = b"\n"
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            total += chunk.count(b"\n")
            last = chunk[-1:]
    if last not in (b"\n", b""):
        total += 1  # final line had no trailing newline
    return max(total - 1, 0)  # drop the header


def verify_row_count(path: Path | str) -> tuple[int, int]:
    """Return (fast count, pandas count) for one file so they can be compared."""
    return count_data_rows(path), len(pd.read_csv(path))


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, nulls, distinct values, and range where numeric."""
    rows = []
    for col in df.columns:
        series = df[col]
        numeric = pd.api.types.is_numeric_dtype(series)
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "nulls": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
                "min": series.min() if numeric else "",
                "max": series.max() if numeric else "",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# searching column names
# --------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenise(name: str) -> list[str]:
    """Break a column name into lowercase words.

    `Tot size` -> ['tot', 'size'], `srcPort` -> ['src', 'port'],
    `fin_flag_number` -> ['fin', 'flag', 'number'].
    """
    spaced = _CAMEL.sub(" ", str(name))
    return [tok.lower() for tok in _NON_ALNUM.split(spaced) if tok]


def search_columns(columns, keywords) -> dict:
    """Search column names for keywords two ways.

    A whole-word hit is a column with a keyword as one of its name tokens.
    A substring hit is what a plain `keyword in name.lower()` search would
    return, which catches things like 'ts' inside 'Packets'. The two are kept
    apart so a coincidence is never mistaken for a field.
    """
    keywords = [k.lower() for k in keywords]
    word_hits, substring_hits = {}, {}
    for col in columns:
        tokens = set(tokenise(col))
        flat = str(col).lower()
        matched = [k for k in keywords if k in tokens]
        loose = [k for k in keywords if k not in tokens and k in flat]
        if matched:
            word_hits[col] = matched
        elif loose:
            substring_hits[col] = loose
    return {"word_hits": word_hits, "substring_hits": substring_hits}


TIME_KEYWORDS = ["time", "timestamp", "ts", "date", "epoch"]
ENDPOINT_KEYWORDS = [
    "src",
    "dst",
    "ip",
    "mac",
    "addr",
    "source",
    "destination",
    "port",
]


# --------------------------------------------------------------------------
# provisional feature families
# --------------------------------------------------------------------------

LABEL_KEYWORDS = ["label", "class", "attack", "target", "category"]

# Order matters: a column matching more than one family is assigned to the
# first it matches, and the alternatives are recorded for review.
FAMILY_RULES: list[tuple[str, list[str]]] = [
    (
        "timing",
        [
            "time",
            "timestamp",
            "ts",
            "date",
            "epoch",
            "duration",
            "iat",
            "interval",
            "rate",
            "srate",
            "drate",
            "delta",
            "elapsed",
            "freq",
            "frequency",
        ],
    ),
    (
        "protocol",
        [
            "protocol",
            "proto",
            "tcp",
            "udp",
            "icmp",
            "igmp",
            "arp",
            "ipv",
            "ipv4",
            "ipv6",
            "llc",
            "http",
            "https",
            "dns",
            "dhcp",
            "telnet",
            "smtp",
            "ssh",
            "irc",
            "mqtt",
            "coap",
            "ntp",
            "flag",
            "flags",
            "syn",
            "ack",
            "fin",
            "rst",
            "psh",
            "urg",
            "ece",
            "cwr",
            "port",
            "header",
        ],
    ),
    (
        "statistical",
        [
            "min",
            "max",
            "avg",
            "mean",
            "median",
            "std",
            "stddev",
            "var",
            "variance",
            "covariance",
            "sum",
            "tot",
            "total",
            "count",
            "number",
            "size",
            "length",
            "magnitue",
            "magnitude",
            "radius",
            "weight",
            "skew",
            "kurtosis",
            "entropy",
        ],
    ),
]


def assign_families(columns) -> dict:
    """Sort columns into timing / protocol / statistical / other by name alone.

    This is a guess made from spelling. Any column matching more than one rule
    is listed under `ambiguous` so it can be settled by hand.
    """
    families = {name: [] for name, _ in FAMILY_RULES}
    families["other"] = []
    labels, ambiguous = [], {}

    for col in columns:
        tokens = set(tokenise(col))
        if any(k in tokens for k in LABEL_KEYWORDS):
            labels.append(col)
            continue
        matched = [name for name, kws in FAMILY_RULES if tokens & set(kws)]
        if not matched:
            families["other"].append(col)
            continue
        families[matched[0]].append(col)
        if len(matched) > 1:
            ambiguous[col] = matched

    return {"families": families, "labels": labels, "ambiguous": ambiguous}


def render_feature_families_yaml(
    assignment: dict,
    *,
    source_file: str,
    git_sha: str,
    run_date: str,
    generated_by: str = "AG_PRAXIS_NB01_dataset_inventory.ipynb",
) -> str:
    """Render the family assignment as YAML with a header saying it is a draft."""
    body = {
        "status": "draft",
        "review_required": True,
        "generated_by": generated_by,
        "generated_from": source_file,
        "git_sha": git_sha,
        "generated_on": run_date,
        "label_columns": assignment["labels"],
        "families": assignment["families"],
        "ambiguous": assignment["ambiguous"],
    }
    header = (
        "# AG_PRAXIS — feature families (DRAFT, NEEDS MANUAL REVIEW)\n"
        "#\n"
        "# Written by NB01 from column names alone. No column was inspected for\n"
        "# what it actually measures. Anything under `ambiguous` matched more than\n"
        "# one rule and was assigned to the first match only.\n"
        "#\n"
        "# Read every line before any experiment depends on these groupings.\n"
        "# Set status: reviewed once that has been done.\n\n"
    )
    return header + yaml.safe_dump(body, sort_keys=False, default_flow_style=False)
