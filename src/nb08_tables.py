"""Rebuild NB08's consolidated tables from the committed artifacts, as CSV files.

NB08 printed three tables to cell output and wrote none of them to disk. Cell output
is not a record: it lives in one executed copy of one notebook and cannot be read by
anything else. This script re-derives the same three tables from the artifacts the run
left behind and writes them as files, so the tables exist on the same footing as the
metrics they come from.

Nothing is trained and nothing is re-scored. Every figure here is read out of a
`metrics.json` written by the run that produced it, or computed from the prediction
arrays saved beside it. No model is loaded and Keras is never imported, so this runs
on a laptop in a few seconds.

Three tables:

`records_to_threshold.csv` is the smallest observation budget at which each class
reaches F1 0.80. It was the measurement the observation hypothesis was stated in terms
of until `PREREGISTRATION.md` Amendment 12, and it is retained and reported. A class
that never reaches the threshold within fifty records has no value on that scale and is
written as "> 50", never as 50 and never as an imputed figure, under Amendment 11.

`saturation.csv` is the smallest budget within 0.02 of a class's own best score across
the four budgets. It is the measurement H2 is stated in terms of under Amendment 12,
and it answers a different question from the one above: where a class stops improving,
rather than where it becomes usable.

`mcnemar.csv` is the eleven paired tests, six between budgets and five between the
sequence parent and the class-balancing runs, with Holm-Bonferroni applied within each
family and again across all eleven.

Group medians are deliberately not computed. The median is an interpretation under a
hypothesis, it depends on a censoring rule recorded in two governance files, and the
measurement itself is the per-class budget. Anything reading these tables can take a
median; this script does not decide one for it.

No comparison against `published_cnn_19class` appears here either. That run is scored
on 1,614,182 single records from a different split, against 49,159 windows here, so no
item-level pairing exists and nothing in this file can be paired against it.

    python -m src.nb08_tables            # from the repository root
    python src/nb08_tables.py            # either way
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import significance as sg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
NB06_DIR = REPO_ROOT / "data" / "processed" / "NB06"
NB08_DIR = REPO_ROOT / "data" / "processed" / "NB08"
NB07_DIRS = (REPO_ROOT / "data" / "processed" / "NB07", REPO_ROOT / "results" / "NB07")
TABLES_DIR = NB08_DIR / "tables"

PARENT_RUN = "sequence_cnn_lstm_19class"
BUDGETS = (5, 10, 25, 50)
FULL_BUDGET = 50
BUDGET_RUNS = {5: "sequence_budget_05", 10: "sequence_budget_10", 25: "sequence_budget_25"}
INTERVENTIONS = (
    "class_weighted_loss",
    "focal_loss",
    "logit_adjustment",
    "threshold_tuning",
    "window_resampling",
)

F1_TARGET = 0.80
SATURATION_EPSILON = 0.02
ALPHA = 0.05

# Fixed by the benchmark taxonomy in PROJECT_RECORD.md Section 3, not derived from any
# measurement here. Recon-Ping_Sweep is in neither group under PREREGISTRATION.md
# Amendment 4, and so are Benign and the four MQTT flood classes, which no group names.
VOLUMETRIC = (
    "DDoS-ICMP", "DDoS-SYN", "DDoS-TCP", "DDoS-UDP",
    "DoS-ICMP", "DoS-SYN", "DoS-TCP", "DoS-UDP",
)
LOW_RATE = (
    "Recon-OS_Scan", "Recon-Port_Scan", "Recon-VulScan", "Spoofing", "MQTT-Malformed_Data",
)

# The classes whose per-class figures carry the Amendment 4 caveat: too few sequences in
# the partitions they are scored on for an F1 to take more than a few values.
THIN = ("Recon-Ping_Sweep", "Recon-VulScan", "MQTT-Malformed_Data")


# --------------------------------------------------------------------------
# reading the artifacts
# --------------------------------------------------------------------------


def find_file(filename: str, directories) -> Path:
    """The first of `directories` holding `filename`, or an error naming all of them."""
    found = next((Path(d) / filename for d in directories if (Path(d) / filename).exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"{filename} not found. Looked in: {[str(Path(d)) for d in directories]}"
        )
    return found


def load_run(run_id: str, directories, *, arrays: bool = True) -> dict:
    """One run's metrics, config and predictions, each located on its own.

    The files of a single run are not always in one directory. The sequence parent has
    its two JSON documents at the top of its notebook's artifact directory and its
    prediction arrays in a subdirectory named after the run, so a function that assumed
    one directory held all four would find none of them.
    """
    run = {
        "run_id": run_id,
        "metrics_path": find_file("metrics.json", directories),
        "config_path": find_file("config.json", directories),
    }
    run["metrics"] = json.loads(run["metrics_path"].read_text())
    run["config"] = json.loads(run["config_path"].read_text())
    if arrays:
        for which in ("y_true", "y_pred"):
            path = find_file(f"{which}.npy", directories)
            codes = np.load(path)
            if codes.dtype.kind not in "iu":
                raise TypeError(
                    f"{path} holds {codes.dtype}, not integer codes. These arrays are "
                    "positions into the label list and cannot be read as anything else."
                )
            run[f"{which}_path"] = path
            run[which] = codes.astype("int64")
    return run


def load_everything() -> dict:
    """Every run these tables are built from, and the label list they all index."""
    label_map = json.loads((NB08_DIR / "label_map.json").read_text())

    runs = {PARENT_RUN: load_run(PARENT_RUN, [NB06_DIR / PARENT_RUN, NB06_DIR])}
    for k, run_id in BUDGET_RUNS.items():
        runs[run_id] = load_run(run_id, [NB08_DIR / run_id])
    for seed in (43, 44, 45, 46):
        run_id = f"sequence_seed_{seed}"
        runs[run_id] = load_run(run_id, [NB08_DIR / run_id])
    for run_id in INTERVENTIONS:
        runs[run_id] = load_run(run_id, [d / run_id for d in NB07_DIRS])

    return {"runs": runs, "label_map": label_map, "labels": list(label_map["labels"])}


def assert_one_label_order(runs: dict, labels: list) -> None:
    """Every run names the same classes in the same order, and scores the same items.

    A code is a position, so two runs that order their labels differently mean different
    classes by the same integer and every table below would silently mix them. The check
    is against the label map rather than against each other, so agreement is with the one
    document that says what the codes mean.
    """
    disagree = {
        run_id: run["metrics"]["labels"]
        for run_id, run in runs.items()
        if list(run["metrics"]["labels"]) != labels
    }
    if disagree:
        raise ValueError(
            "these runs do not carry the label order in label_map.json, so the same code "
            f"names different classes in different runs: {sorted(disagree)}"
        )

    reference_id, reference = next(iter(runs.items()))
    for run_id, run in runs.items():
        if len(run["y_true"]) != len(reference["y_true"]):
            raise ValueError(
                f"{run_id} scores {len(run['y_true']):,} items against "
                f"{len(reference['y_true']):,} in {reference_id}. Nothing here can be paired."
            )
        if not np.array_equal(run["y_true"], reference["y_true"]):
            raise ValueError(
                f"{run_id} and {reference_id} were scored on different true labels, so they "
                "are not scored on the same items and no paired test between them is valid."
            )
        outside = run["y_pred"].min() < 0 or run["y_pred"].max() >= len(labels)
        if outside:
            raise ValueError(f"{run_id} holds a predicted code outside 0..{len(labels) - 1}")


def group_of(label: str) -> str:
    """Which group a class is in, from the taxonomy rather than from its score."""
    if label in VOLUMETRIC:
        return "volumetric"
    if label in LOW_RATE:
        return "low-rate"
    return "neither"


def per_class_by_budget(runs: dict) -> dict:
    """Per-class F1 at each of the four budgets, keyed by budget."""
    by_budget = {k: runs[run_id]["metrics"]["per_class_f1"] for k, run_id in BUDGET_RUNS.items()}
    by_budget[FULL_BUDGET] = runs[PARENT_RUN]["metrics"]["per_class_f1"]
    return {k: by_budget[k] for k in BUDGETS}


def assert_numeric(frame: pd.DataFrame, columns) -> pd.DataFrame:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{column} came out as {frame[column].dtype}, not numeric")
    return frame


# --------------------------------------------------------------------------
# the three tables
# --------------------------------------------------------------------------


def records_to_threshold(scores: dict, labels: list) -> pd.DataFrame:
    """The smallest budget at which each class reaches F1 0.80, censored above fifty.

    A class that has not reached the threshold at the largest budget on the grid is
    right-censored. Fifty is where the measurement stops, not where the class stops
    needing records, so its entry reads "> 50" and carries no number.
    """
    rows = []
    for label in labels:
        by_budget = [float(scores[k][label]) for k in BUDGETS]
        reached = next((k for k, f1 in zip(BUDGETS, by_budget) if f1 >= F1_TARGET), None)
        rows.append(
            {
                "class": label,
                "group": group_of(label),
                "reported_as": f"> {FULL_BUDGET}" if reached is None else str(reached),
                "best_f1": max(by_budget),
                "f1_at_50": by_budget[-1],
                "reaches_then_falls": bool(reached is not None and by_budget[-1] < F1_TARGET),
                "thin": label in THIN,
            }
        )
    frame = pd.DataFrame(rows)
    frame["censored"] = frame["reported_as"] == f"> {FULL_BUDGET}"
    frame = frame.sort_values(["group", "censored", "class"]).drop(columns="censored")
    return assert_numeric(frame, ["best_f1", "f1_at_50"]).reset_index(drop=True)


def saturation(scores: dict, labels: list) -> pd.DataFrame:
    """The smallest budget within 0.02 of a class's own best score across the grid.

    Each class is measured against itself, so a class that never becomes detectable
    still has a budget at which it stops improving. That is the point of the analysis
    and the reason `usable_there` is carried beside it: the two are not the same thing.
    """
    rows = []
    for label in labels:
        by_budget = {k: float(scores[k][label]) for k in BUDGETS}
        best = max(by_budget.values())
        saturates_at = next(k for k in BUDGETS if by_budget[k] >= best - SATURATION_EPSILON)
        rows.append(
            {
                "class": label,
                "group": group_of(label),
                "best_f1": best,
                "best_at_k": min(k for k in BUDGETS if by_budget[k] == best),
                "saturates_at_k": saturates_at,
                "f1_there": by_budget[saturates_at],
                "usable_there": bool(by_budget[saturates_at] >= F1_TARGET),
                "thin": label in THIN,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["saturates_at_k", "group", "class"])
    return assert_numeric(
        frame, ["best_f1", "best_at_k", "saturates_at_k", "f1_there"]
    ).reset_index(drop=True)


def mcnemar_tests(runs: dict) -> pd.DataFrame:
    """The eleven paired tests, corrected within each family and across all of them.

    The two families answer different questions, so each is corrected on its own and the
    stricter correction over all eleven is carried alongside rather than instead. A
    reader who treats them as one family reads `holm_p_all_11`.
    """
    y_true = runs[PARENT_RUN]["y_true"]
    predictions = {PARENT_RUN: runs[PARENT_RUN]["y_pred"]}
    for k, run_id in BUDGET_RUNS.items():
        predictions[run_id] = runs[run_id]["y_pred"]
    for run_id in INTERVENTIONS:
        predictions[run_id] = runs[run_id]["y_pred"]

    run_at = {**BUDGET_RUNS, FULL_BUDGET: PARENT_RUN}
    budget_pairs = [
        sg.mcnemar(
            y_true, predictions[run_at[a]], predictions[run_at[b]],
            name_a=f"k{a}", name_b=f"k{b}",
        )
        for i, a in enumerate(BUDGETS)
        for b in BUDGETS[i + 1:]
    ]
    intervention_pairs = [
        sg.mcnemar(
            y_true, predictions[PARENT_RUN], predictions[run_id],
            name_a=PARENT_RUN, name_b=run_id,
        )
        for run_id in INTERVENTIONS
    ]

    rows = [
        *sg.compare_family(budget_pairs, alpha=ALPHA, family="budget against budget"),
        *sg.compare_family(intervention_pairs, alpha=ALPHA, family="parent against intervention"),
    ]
    across_all, reject_all = sg.holm_bonferroni([row["p_value"] for row in rows], alpha=ALPHA)
    for row, value, flag in zip(rows, across_all, reject_all):
        row["holm_p_all_11"] = float(value)
        row["significant_all_11"] = bool(flag)

    frame = pd.DataFrame(rows)[
        [
            "family", "a", "b", "accuracy_a", "accuracy_b", "only_a_correct", "only_b_correct",
            "discordant", "statistic", "p_value", "holm_p", "significant_at_alpha",
            "holm_p_all_11", "favours", "method",
        ]
    ]
    return assert_numeric(
        frame,
        [
            "accuracy_a", "accuracy_b", "only_a_correct", "only_b_correct", "discordant",
            "statistic", "p_value", "holm_p", "holm_p_all_11",
        ],
    )


# --------------------------------------------------------------------------
# what PROJECT_RECORD.md Section 5 says these tables show
# --------------------------------------------------------------------------

# Quoted from PROJECT_RECORD.md v1.6, Section 5, "Observation budgets and cross-seed
# variation". Each entry is a claim in that prose paired with the value this script
# derives from the artifacts. The point is to catch the record and the artifacts drifting
# apart, in either direction, so a mismatch is printed and never corrected in place.
SECTION_5_CLAIMS = {
    "macro F1 at k=5": 0.6295,
    "macro F1 at k=10": 0.6837,
    "macro F1 at k=25": 0.7323,
    "macro F1 at k=50": 0.7138,
    "classes at F1 >= 0.80, k=5": 9,
    "classes at F1 >= 0.80, k=10": 9,
    "classes at F1 >= 0.80, k=25": 11,
    "classes at F1 >= 0.80, k=50": 10,
    "classes censored at k=50": 8,
    "low-rate records to threshold": "Recon-Port_Scan 5, MQTT-Malformed_Data 25, "
                                     "Recon-OS_Scan > 50, Recon-VulScan > 50, Spoofing > 50",
    "volumetric records to threshold": "DDoS-SYN 5, DDoS-TCP 5, DDoS-UDP 5, DoS-SYN 5, "
                                       "DoS-UDP 5, DDoS-ICMP > 50, DoS-ICMP > 50, DoS-TCP > 50",
    "low-rate classes censored": 3,
    "volumetric classes censored": 3,
    "DDoS-ICMP best F1": 0.7998,
    "Recon-Port_Scan F1 at k=5": 0.8878,
    # These two sentences state their own scope, "of the censored classes in the two named
    # groups", so the check is scoped the same way and the two match. The two censored
    # classes outside both groups, MQTT-DoS-Publish_Flood at its maximum at fifty and
    # MQTT-DDoS-Publish_Flood peaking at ten, are named in the sentence that follows them
    # and are not part of either list here.
    "censored, at their maximum at k=50": "DoS-ICMP, DoS-TCP, Recon-VulScan",
    "censored, peak earlier and score lower at 50": "DDoS-ICMP, Recon-OS_Scan, Spoofing",
    "classes saturating at k=5": 7,
    "classes saturating below F1 0.80": 8,
    "budget pairs significant": 5,
    "intervention pairs significant": 3,
    "k25 against k50": "significant, favours k25",
    "test windows": 49159,
    "parameters": 214227,
}


def check_against_project_record(runs, scores, threshold, saturated, mcnemar) -> pd.DataFrame:
    """Every Section 5 claim these artifacts can settle, and what the artifacts say."""
    censored = threshold["reported_as"] == f"> {FULL_BUDGET}"
    budget_family = mcnemar[mcnemar["family"] == "budget against budget"]
    intervention_family = mcnemar[mcnemar["family"] == "parent against intervention"]
    k25_k50 = budget_family[(budget_family["a"] == "k25") & (budget_family["b"] == "k50")].iloc[0]

    def listed(members):
        rows = threshold[threshold["class"].isin(members)]
        rows = rows.assign(sorter=rows["reported_as"].map(lambda v: 999 if v.startswith(">") else int(v)))
        rows = rows.sort_values(["sorter", "class"])
        return ", ".join(f"{r['class']} {r['reported_as']}" for _, r in rows.iterrows())

    # Scoped to censored classes inside the two named groups, which is how the Section 5
    # sentences read. "At its maximum at fifty" is the score at fifty equalling the best
    # across the grid, which is not the same as the best first occurring at fifty: a class
    # can reach its ceiling earlier and still sit on it at fifty.
    grouped_censored = threshold[
        censored.to_numpy() & threshold["group"].isin(("low-rate", "volumetric")).to_numpy()
    ]
    at_ceiling = grouped_censored["f1_at_50"] >= grouped_censored["best_f1"] - 1e-12
    at_max_at_50 = sorted(grouped_censored[at_ceiling]["class"].tolist())
    peaks_early = sorted(grouped_censored[~at_ceiling]["class"].tolist())
    run_at = {**BUDGET_RUNS, FULL_BUDGET: PARENT_RUN}

    found = {
        **{f"macro F1 at k={k}": round(float(runs[run_at[k]]["metrics"]["macro_f1"]), 4)
           for k in BUDGETS},
        **{f"classes at F1 >= 0.80, k={k}": int(sum(v >= F1_TARGET for v in scores[k].values()))
           for k in BUDGETS},
        "classes censored at k=50": int(censored.sum()),
        "low-rate records to threshold": listed(LOW_RATE),
        "volumetric records to threshold": listed(VOLUMETRIC),
        "low-rate classes censored": int((censored & threshold["group"].eq("low-rate")).sum()),
        "volumetric classes censored": int((censored & threshold["group"].eq("volumetric")).sum()),
        "DDoS-ICMP best F1": round(
            float(threshold.set_index("class").loc["DDoS-ICMP", "best_f1"]), 4
        ),
        "Recon-Port_Scan F1 at k=5": round(float(scores[5]["Recon-Port_Scan"]), 4),
        "censored, at their maximum at k=50": ", ".join(at_max_at_50),
        "censored, peak earlier and score lower at 50": ", ".join(peaks_early),
        "classes saturating at k=5": int((saturated["saturates_at_k"] == 5).sum()),
        "classes saturating below F1 0.80": int((~saturated["usable_there"]).sum()),
        "budget pairs significant": int(budget_family["significant_at_alpha"].sum()),
        "intervention pairs significant": int(intervention_family["significant_at_alpha"].sum()),
        "k25 against k50": (
            f"{'significant' if k25_k50['significant_at_alpha'] else 'not significant'}, "
            f"favours {k25_k50['favours']}"
        ),
        "test windows": int(len(runs[PARENT_RUN]["y_true"])),
        "parameters": int(runs[PARENT_RUN]["metrics"]["n_parameters"]),
    }

    rows = []
    for claim, stated in SECTION_5_CLAIMS.items():
        value = found[claim]
        agrees = (
            abs(value - stated) < 5e-5
            if isinstance(stated, float)
            else value == stated
        )
        rows.append(
            {"claim": claim, "section_5_says": stated, "artifacts_say": value, "agrees": agrees}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------


def build() -> dict:
    """Every table, from the artifacts, with nothing written."""
    loaded = load_everything()
    runs, labels = loaded["runs"], loaded["labels"]
    assert_one_label_order(runs, labels)
    scores = per_class_by_budget(runs)

    threshold = records_to_threshold(scores, labels)
    saturated = saturation(scores, labels)
    mcnemar = mcnemar_tests(runs)
    return {
        "runs": runs,
        "labels": labels,
        "scores": scores,
        "records_to_threshold": threshold,
        "saturation": saturated,
        "mcnemar": mcnemar,
        "checks": check_against_project_record(runs, scores, threshold, saturated, mcnemar),
    }


def main() -> int:
    built = build()
    runs, labels = built["runs"], built["labels"]

    print(f"{len(runs)} runs read, {len(labels)} classes, "
          f"{len(runs[PARENT_RUN]['y_true']):,} test items, one label order")
    for run_id in sorted(runs):
        run = runs[run_id]
        print(f"  {run_id:<28} {run['metrics_path'].parent}")
    print()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("records_to_threshold", "saturation", "mcnemar"):
        path = TABLES_DIR / f"{name}.csv"
        built[name].to_csv(path, index=False)
        written[name] = path

    for name in ("records_to_threshold", "saturation"):
        print("=" * 100)
        print(f"{name}.csv")
        print("=" * 100)
        print(built[name].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    print("=" * 100)
    print("mcnemar.csv")
    print("=" * 100)
    print(built["mcnemar"].to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print()

    print("=" * 100)
    print("against PROJECT_RECORD.md v1.6, Section 5")
    print("=" * 100)
    checks = built["checks"]
    print(checks.to_string(index=False))
    print()
    disagreements = checks[~checks["agrees"]]
    if len(disagreements):
        print(f"{len(disagreements)} of {len(checks)} claims disagree with the artifacts:")
        for _, row in disagreements.iterrows():
            print(f"  {row['claim']}: Section 5 says {row['section_5_says']!r}, "
                  f"the artifacts say {row['artifacts_say']!r}")
        print()
        print("Reported, not corrected. Nothing in PROJECT_RECORD.md is written by this script.")
    else:
        print(f"all {len(checks)} claims agree with the artifacts")

    print()
    for name, path in written.items():
        print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(built[name])} rows)")
    return 1 if len(disagreements) else 0


if __name__ == "__main__":
    raise SystemExit(main())
