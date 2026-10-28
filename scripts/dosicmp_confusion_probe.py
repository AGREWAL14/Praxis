"""Where do the DoS-ICMP errors go?

DoS-ICMP scores F1 0.9960 under the published single-record model and 0.3178 under
the window-based sequence model. That is a bigger fall than any other class, and the
macro-F1 figure it sits inside says nothing about what the errors are. This script
reads the saved predictions for three runs and prints, for each of them, what a
DoS-ICMP item was actually predicted as and what was predicted as DoS-ICMP.

It trains nothing. Everything here is recomputed from y_true.npy and y_pred.npy, and
where a recomputed figure disagrees with the run's own metrics.json the disagreement
is reported rather than smoothed over.

Run from anywhere:  python scripts/dosicmp_confusion_probe.py
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

RUNS = [
    {
        "name": "published_cnn_19class",
        "unit": "records",
        "metrics": REPO / "results/NB05/published_cnn_19class/metrics.json",
        "arrays": REPO / "results/NB05/published_cnn_19class",
    },
    {
        "name": "ours_cnn_19class",
        "unit": "records",
        "metrics": REPO / "results/NB05/ours_cnn_19class/metrics.json",
        "arrays": REPO / "results/NB05/ours_cnn_19class",
    },
    {
        "name": "sequence_cnn_lstm_19class",
        "unit": "windows",
        "metrics": REPO / "data/processed/NB06/metrics.json",
        "arrays": REPO / "data/processed/NB06/sequence_cnn_lstm_19class",
    },
]

FOCUS = "DoS-ICMP"
PARTNER = "DDoS-ICMP"
OUT_DIR = REPO / "data/processed/NB06d"
OUT_FILE = OUT_DIR / "dosicmp_confusion.json"

# Anything above this between a recomputed value and the stored one is reported as a
# disagreement rather than as floating-point noise.
TOLERANCE = 1e-9


def load_run(run, canonical_labels):
    """Read one run's labels and arrays, and map its arrays into the canonical order.

    Each run's own metrics.json supplies its label ordering. Nothing assumes the three
    runs agree; if one does not, its integer codes are remapped by name.
    """
    metrics = json.loads(run["metrics"].read_text())
    labels = list(metrics["labels"])
    if len(labels) != 19:
        raise AssertionError(
            f"{run['name']}: expected 19 labels, found {len(labels)}"
        )

    y_true = np.load(run["arrays"] / "y_true.npy")
    y_pred = np.load(run["arrays"] / "y_pred.npy")
    if y_true.shape != y_pred.shape:
        raise AssertionError(
            f"{run['name']}: y_true is {y_true.shape} and y_pred is {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise AssertionError(f"{run['name']}: expected 1-D arrays, got {y_true.ndim}-D")

    n_test = metrics["n_test"]
    if len(y_true) != n_test:
        raise AssertionError(
            f"{run['name']}: arrays hold {len(y_true)} items, metrics.json says "
            f"n_test is {n_test}"
        )

    if set(labels) != set(canonical_labels):
        raise AssertionError(
            f"{run['name']}: label set differs from the canonical set. "
            f"Only in this run: {sorted(set(labels) - set(canonical_labels))}. "
            f"Missing from it: {sorted(set(canonical_labels) - set(labels))}"
        )

    reordered = labels != canonical_labels
    if reordered:
        position = {name: i for i, name in enumerate(canonical_labels)}
        lookup = np.array([position[name] for name in labels], dtype=np.int64)
        y_true = lookup[y_true.astype(np.int64)]
        y_pred = lookup[y_pred.astype(np.int64)]
    else:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)

    observed = set(np.unique(np.concatenate([y_true, y_pred])).tolist())
    out_of_range = {i for i in observed if i < 0 or i >= len(canonical_labels)}
    if out_of_range:
        raise AssertionError(f"{run['name']}: codes outside the label range: {out_of_range}")

    return {
        "name": run["name"],
        "unit": run["unit"],
        "metrics": metrics,
        "own_label_order": labels,
        "reordered": reordered,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def confusion(y_true, y_pred, n_classes):
    """Counts matrix, rows are the true class and columns the predicted class."""
    flat = y_true * n_classes + y_pred
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def prf(matrix, index):
    """Precision, recall and F1 for one class, taken from the counts matrix."""
    tp = int(matrix[index, index])
    fn = int(matrix[index, :].sum()) - tp
    fp = int(matrix[:, index].sum()) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def check_against_metrics(run, class_name, recomputed):
    """Compare a recomputed precision, recall and F1 against the run's own metrics."""
    stored = {
        "precision": run["metrics"]["per_class_precision"].get(class_name),
        "recall": run["metrics"]["per_class_recall"].get(class_name),
        "f1": run["metrics"]["per_class_f1"].get(class_name),
        "support": run["metrics"]["support"].get(class_name),
    }
    comparison = {}
    disagreements = []
    for key, stored_value in stored.items():
        recomputed_value = recomputed[key]
        if stored_value is None:
            comparison[key] = {
                "recomputed": recomputed_value,
                "stored": None,
                "difference": None,
                "agrees": False,
            }
            disagreements.append(f"{class_name}: metrics.json has no {key}")
            continue
        difference = float(recomputed_value) - float(stored_value)
        agrees = abs(difference) <= TOLERANCE
        comparison[key] = {
            "recomputed": recomputed_value,
            "stored": stored_value,
            "difference": difference,
            "agrees": agrees,
        }
        if not agrees:
            disagreements.append(
                f"{class_name} {key}: recomputed {recomputed_value!r} against stored "
                f"{stored_value!r}, difference {difference:+.10g}"
            )
    return comparison, disagreements


def distribution(counts, labels, exclude=None):
    """Counts and percentages by class name, sorted descending, zeros dropped."""
    total = int(counts.sum())
    rows = []
    for i, name in enumerate(labels):
        value = int(counts[i])
        if value == 0:
            continue
        if exclude is not None and i == exclude:
            continue
        rows.append(
            {
                "class": name,
                "count": value,
                "percent_of_total": 100.0 * value / total if total else 0.0,
            }
        )
    rows.sort(key=lambda r: -r["count"])
    return rows, total


def print_distribution(title, rows, total, unit, empty_note):
    print(f"  {title}  (total {total:,} {unit})")
    if not rows:
        print(f"    {empty_note}")
        return
    print(f"    {'class':<28} {'count':>12}  {'share':>8}")
    for row in rows:
        print(f"    {row['class']:<28} {row['count']:>12,}  {row['percent_of_total']:>7.3f}%")


def main():
    canonical_metrics = json.loads((REPO / "data/processed/NB06/metrics.json").read_text())
    canonical_labels = list(canonical_metrics["labels"])
    if len(canonical_labels) != 19:
        raise AssertionError(
            f"NB06 metrics.json lists {len(canonical_labels)} labels, expected 19"
        )
    for name in (FOCUS, PARTNER):
        if name not in canonical_labels:
            raise AssertionError(f"{name} is not in the NB06 label list")

    n_classes = len(canonical_labels)
    focus = canonical_labels.index(FOCUS)
    partner = canonical_labels.index(PARTNER)

    print("=" * 78)
    print(f"{FOCUS} confusion probe")
    print("=" * 78)
    print(f"Class order taken from data/processed/NB06/metrics.json, {n_classes} classes.")
    print(f"Recomputed values are compared against each run's own metrics.json at "
          f"tolerance {TOLERANCE:g}.")
    print()

    report = {
        "canonical_label_order": canonical_labels,
        "canonical_order_from": "data/processed/NB06/metrics.json",
        "focus_class": FOCUS,
        "partner_class": PARTNER,
        "tolerance": TOLERANCE,
        "runs": {},
        "all_disagreements": [],
    }

    for spec in RUNS:
        run = load_run(spec, canonical_labels)
        y_true, y_pred = run["y_true"], run["y_pred"]
        matrix = confusion(y_true, y_pred, n_classes)
        unit = run["unit"]

        print("-" * 78)
        print(f"{run['name']}   ({len(y_true):,} test {unit})")
        print("-" * 78)
        if run["reordered"]:
            print("  Label order differs from the canonical order. Codes were remapped by")
            print("  name before anything below was computed.")
            print(f"    this run's order: {run['own_label_order']}")
        else:
            print("  Label order matches the canonical order. No remapping needed.")
        print()

        # The DoS-ICMP row: for every true DoS-ICMP item, what was predicted.
        row_all, row_total = distribution(matrix[focus, :], canonical_labels)
        print_distribution(
            f"{FOCUS} row — true {FOCUS} {unit}, by predicted class",
            row_all,
            row_total,
            unit,
            f"no true {FOCUS} {unit}",
        )
        row_errors, _ = distribution(matrix[focus, :], canonical_labels, exclude=focus)
        print()

        # The DoS-ICMP column: what else was predicted as DoS-ICMP.
        col_all, col_total = distribution(matrix[:, focus], canonical_labels)
        print_distribution(
            f"{FOCUS} column — {unit} predicted {FOCUS}, by true class",
            col_all,
            col_total,
            unit,
            f"nothing was predicted {FOCUS}",
        )
        print()

        # The same two views for the partner class, so an influx is visible.
        partner_row, partner_row_total = distribution(matrix[partner, :], canonical_labels)
        partner_col, partner_col_total = distribution(matrix[:, partner], canonical_labels)
        print_distribution(
            f"{PARTNER} column — {unit} predicted {PARTNER}, by true class",
            partner_col,
            partner_col_total,
            unit,
            f"nothing was predicted {PARTNER}",
        )
        print()

        # Per-class metrics recomputed, then checked against the stored figures.
        run_disagreements = []
        per_class = {}
        for name, index in ((FOCUS, focus), (PARTNER, partner)):
            recomputed = prf(matrix, index)
            comparison, disagreements = check_against_metrics(run, name, recomputed)
            per_class[name] = {"recomputed": recomputed, "against_metrics_json": comparison}
            run_disagreements.extend(disagreements)
            print(f"  {name} recomputed from the arrays")
            print(
                f"    tp {recomputed['tp']:,}   fp {recomputed['fp']:,}   "
                f"fn {recomputed['fn']:,}   support {recomputed['support']:,}"
            )
            print(
                f"    precision {recomputed['precision']:.6f}   "
                f"recall {recomputed['recall']:.6f}   F1 {recomputed['f1']:.6f}"
            )
            stored_f1 = run["metrics"]["per_class_f1"].get(name)
            if stored_f1 is not None:
                print(f"    metrics.json F1 {stored_f1:.6f}")
            print()

        if run_disagreements:
            print("  DISAGREEMENT with metrics.json:")
            for line in run_disagreements:
                print(f"    {line}")
        else:
            print("  Recomputed precision, recall, F1 and support agree with metrics.json")
            print(f"  for both {FOCUS} and {PARTNER}.")
        print()

        # 2x2 between the two classes, over items whose true label is one of the two.
        keep = np.isin(y_true, [focus, partner])
        pair_true, pair_pred = y_true[keep], y_pred[keep]
        pair = {}
        for true_name, true_index in ((FOCUS, focus), (PARTNER, partner)):
            selected = pair_true == true_index
            n_true = int(selected.sum())
            as_focus = int((pair_pred[selected] == focus).sum())
            as_partner = int((pair_pred[selected] == partner).sum())
            pair[true_name] = {
                "n_true": n_true,
                f"predicted_{FOCUS}": as_focus,
                f"predicted_{PARTNER}": as_partner,
                "predicted_elsewhere": n_true - as_focus - as_partner,
            }
        print(f"  {FOCUS} against {PARTNER}, over {int(keep.sum()):,} {unit} whose true")
        print("  class is one of the two")
        print(f"    {'true \\ predicted':<22} {FOCUS:>12} {PARTNER:>12} {'elsewhere':>12}")
        for true_name in (FOCUS, PARTNER):
            entry = pair[true_name]
            print(
                f"    {true_name:<22} {entry[f'predicted_{FOCUS}']:>12,} "
                f"{entry[f'predicted_{PARTNER}']:>12,} {entry['predicted_elsewhere']:>12,}"
            )
        print()

        report["runs"][run["name"]] = {
            "unit": unit,
            "n_test": int(len(y_true)),
            "own_label_order": run["own_label_order"],
            "label_order_differs_from_canonical": run["reordered"],
            "focus_row_by_predicted_class": row_all,
            "focus_row_errors_only": row_errors,
            "focus_column_by_true_class": col_all,
            "partner_row_by_predicted_class": partner_row,
            "partner_column_by_true_class": partner_col,
            "per_class": per_class,
            "focus_vs_partner_2x2": pair,
            "disagreements_with_metrics_json": run_disagreements,
            "confusion_row_counts": {
                FOCUS: {canonical_labels[i]: int(matrix[focus, i]) for i in range(n_classes)},
                PARTNER: {canonical_labels[i]: int(matrix[partner, i]) for i in range(n_classes)},
            },
            "confusion_column_counts": {
                FOCUS: {canonical_labels[i]: int(matrix[i, focus]) for i in range(n_classes)},
                PARTNER: {canonical_labels[i]: int(matrix[i, partner]) for i in range(n_classes)},
            },
        }
        report["all_disagreements"].extend(
            f"{run['name']}: {line}" for line in run_disagreements
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2))

    print("=" * 78)
    if report["all_disagreements"]:
        print(f"{len(report['all_disagreements'])} disagreement(s) with metrics.json:")
        for line in report["all_disagreements"]:
            print(f"  {line}")
    else:
        print("Every recomputed figure agrees with the metrics.json it was checked against.")
    print(f"Written to {OUT_FILE.relative_to(REPO)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
