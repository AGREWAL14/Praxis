"""Is the ICMP confusion one pair's problem or every DoS/DDoS pair's?

The ICMP probe showed that under the window-based sequence model three quarters of
true DoS-ICMP windows are predicted DDoS-ICMP, and that almost none of the errors go
anywhere else. This script asks the same question of all four DoS/DDoS pairs, and then
asks a wider one: for every class in the set, what share of its errors land on a single
other class. If concentrated single-destination error is common across the nineteen,
the ICMP result is ordinary; if it is not, the pairs are doing something particular.

Nothing is trained. Every figure is recomputed from y_true.npy and y_pred.npy, and any
disagreement with a run's own metrics.json is reported rather than smoothed over. The
loaders come from the ICMP probe so the two scripts cannot drift apart.

Run from anywhere:  python scripts/dosddos_pairs_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dosicmp_confusion_probe import (  # noqa: E402
    REPO,
    RUNS,
    TOLERANCE,
    check_against_metrics,
    confusion,
    load_run,
    prf,
)

PAIRS = [
    ("DoS-ICMP", "DDoS-ICMP"),
    ("DoS-SYN", "DDoS-SYN"),
    ("DoS-TCP", "DDoS-TCP"),
    ("DoS-UDP", "DDoS-UDP"),
]

# The record model trained on the same split as the sequence model. The summary table
# compares these two so the split is held constant and the input representation is the
# one thing that differs.
RECORD_RUN = "ours_cnn_19class"
SEQUENCE_RUN = "sequence_cnn_lstm_19class"

OUT_DIR = REPO / "data/processed/NB06d"
OUT_FILE = OUT_DIR / "dosddos_pairs.json"


def pair_analysis(matrix, labels, a_name, b_name):
    """Everything asked of one pair in one run, recomputed from the counts matrix."""
    a, b = labels.index(a_name), labels.index(b_name)

    a_support = int(matrix[a, :].sum())
    b_support = int(matrix[b, :].sum())
    a_to_b = int(matrix[a, b])
    b_to_a = int(matrix[b, a])

    n_pair_items = a_support + b_support
    stayed_a = int(matrix[a, a]) + a_to_b
    stayed_b = int(matrix[b, b]) + b_to_a
    escaped = n_pair_items - stayed_a - stayed_b

    return {
        "members": [a_name, b_name],
        "support": {a_name: a_support, b_name: b_support},
        "support_ratio_larger_over_smaller": (
            max(a_support, b_support) / min(a_support, b_support)
            if min(a_support, b_support)
            else None
        ),
        "larger_member": a_name if a_support >= b_support else b_name,
        "predicted_as_partner": {
            a_name: {
                "count": a_to_b,
                "of_true": a_support,
                "share": a_to_b / a_support if a_support else 0.0,
            },
            b_name: {
                "count": b_to_a,
                "of_true": b_support,
                "share": b_to_a / b_support if b_support else 0.0,
            },
        },
        "restricted_2x2": {
            "n_items_true_in_pair": n_pair_items,
            a_name: {
                f"predicted_{a_name}": int(matrix[a, a]),
                f"predicted_{b_name}": a_to_b,
                "predicted_outside_pair": a_support - int(matrix[a, a]) - a_to_b,
            },
            b_name: {
                f"predicted_{a_name}": b_to_a,
                f"predicted_{b_name}": int(matrix[b, b]),
                "predicted_outside_pair": b_support - int(matrix[b, b]) - b_to_a,
            },
            "escaped_pair_count": escaped,
            "escaped_pair_share": escaped / n_pair_items if n_pair_items else 0.0,
            "within_pair_confusion_count": a_to_b + b_to_a,
            "within_pair_confusion_rate": (
                (a_to_b + b_to_a) / n_pair_items if n_pair_items else 0.0
            ),
        },
        "per_class": {
            a_name: prf(matrix, a),
            b_name: prf(matrix, b),
        },
    }


def error_concentration(matrix, labels):
    """For each class, what share of its errors go to the single worst destination."""
    rows = []
    n = len(labels)
    for i, name in enumerate(labels):
        support = int(matrix[i, :].sum())
        correct = int(matrix[i, i])
        errors = support - correct
        destinations = {
            labels[j]: int(matrix[i, j]) for j in range(n) if j != i and matrix[i, j] > 0
        }
        if errors > 0 and destinations:
            top_class = max(destinations, key=lambda k: destinations[k])
            top_count = destinations[top_class]
            share = top_count / errors
        else:
            top_class, top_count, share = None, 0, None
        rows.append(
            {
                "class": name,
                "support": support,
                "correct": correct,
                "errors": errors,
                "n_distinct_error_destinations": len(destinations),
                "top_error_destination": top_class,
                "top_error_destination_count": top_count,
                "top_destination_share_of_errors": share,
                "errors_as_share_of_support": errors / support if support else None,
            }
        )
    rows.sort(
        key=lambda r: (
            r["top_destination_share_of_errors"] is not None,
            r["top_destination_share_of_errors"] or 0.0,
        ),
        reverse=True,
    )
    return rows


def main():
    canonical_metrics = json.loads((REPO / "data/processed/NB06/metrics.json").read_text())
    canonical_labels = list(canonical_metrics["labels"])
    if len(canonical_labels) != 19:
        raise AssertionError(
            f"NB06 metrics.json lists {len(canonical_labels)} labels, expected 19"
        )
    for a_name, b_name in PAIRS:
        for name in (a_name, b_name):
            if name not in canonical_labels:
                raise AssertionError(f"{name} is not in the NB06 label list")

    n_classes = len(canonical_labels)

    print("=" * 82)
    print("DoS / DDoS pair confusion across four pairs and three runs")
    print("=" * 82)
    print(f"Class order taken from data/processed/NB06/metrics.json, {n_classes} classes.")
    print(f"Every figure is recomputed from the arrays and checked against each run's own")
    print(f"metrics.json at tolerance {TOLERANCE:g}.")
    print()

    report = {
        "canonical_label_order": canonical_labels,
        "canonical_order_from": "data/processed/NB06/metrics.json",
        "pairs": [list(p) for p in PAIRS],
        "tolerance": TOLERANCE,
        "within_pair_confusion_rate_definition": (
            "over items whose true class is one of the pair, the share predicted as the "
            "other member of the pair, both directions counted together"
        ),
        "runs": {},
        "summary_table": {},
        "error_concentration_sequence_run": {},
        "all_disagreements": [],
    }

    matrices = {}
    loaded = {}

    for spec in RUNS:
        run = load_run(spec, canonical_labels)
        matrix = confusion(run["y_true"], run["y_pred"], n_classes)
        matrices[run["name"]] = matrix
        loaded[run["name"]] = run
        unit = run["unit"]

        print("-" * 82)
        print(f"{run['name']}   ({len(run['y_true']):,} test {unit})")
        print("-" * 82)
        if run["reordered"]:
            print("  Label order differs from the canonical order; codes were remapped by name.")
        else:
            print("  Label order matches the canonical order. No remapping needed.")
        print()

        run_disagreements = []
        run_pairs = {}

        for a_name, b_name in PAIRS:
            analysis = pair_analysis(matrix, canonical_labels, a_name, b_name)
            run_pairs[f"{a_name}|{b_name}"] = analysis

            support = analysis["support"]
            r2 = analysis["restricted_2x2"]
            as_partner = analysis["predicted_as_partner"]

            print(f"  {a_name}  against  {b_name}")
            print(
                f"    support: {a_name} {support[a_name]:,}   {b_name} {support[b_name]:,}"
                f"   ratio {analysis['support_ratio_larger_over_smaller']:.3f} to 1"
                f" ({analysis['larger_member']} larger)"
            )
            print(
                f"    {a_name} predicted {b_name}: {as_partner[a_name]['count']:,}"
                f" of {as_partner[a_name]['of_true']:,}"
                f"  ({100 * as_partner[a_name]['share']:.3f}%)"
            )
            print(
                f"    {b_name} predicted {a_name}: {as_partner[b_name]['count']:,}"
                f" of {as_partner[b_name]['of_true']:,}"
                f"  ({100 * as_partner[b_name]['share']:.3f}%)"
            )
            print(
                f"    restricted to the {r2['n_items_true_in_pair']:,} {unit} whose true class"
                f" is one of the two:"
            )
            print(f"      {'true \\ predicted':<24} {a_name:>14} {b_name:>14} {'outside':>10}")
            for member in (a_name, b_name):
                cell = r2[member]
                print(
                    f"      {member:<24} {cell[f'predicted_{a_name}']:>14,}"
                    f" {cell[f'predicted_{b_name}']:>14,}"
                    f" {cell['predicted_outside_pair']:>10,}"
                )
            print(
                f"      within-pair confusion {r2['within_pair_confusion_count']:,}"
                f"  ({100 * r2['within_pair_confusion_rate']:.3f}%)"
                f"   escaped the pair {r2['escaped_pair_count']:,}"
                f"  ({100 * r2['escaped_pair_share']:.3f}%)"
            )
            for member in (a_name, b_name):
                m = analysis["per_class"][member]
                comparison, disagreements = check_against_metrics(run, member, m)
                analysis.setdefault("against_metrics_json", {})[member] = comparison
                run_disagreements.extend(disagreements)
                print(
                    f"      {member:<12} precision {m['precision']:.6f}"
                    f"   recall {m['recall']:.6f}   F1 {m['f1']:.6f}"
                )
            print()

        # Check every class, not only the eight in the pairs. A disagreement anywhere
        # in the file is worth knowing about before any of it is read.
        for i, name in enumerate(canonical_labels):
            _, disagreements = check_against_metrics(run, name, prf(matrix, i))
            for line in disagreements:
                if line not in run_disagreements:
                    run_disagreements.append(line)

        if run_disagreements:
            print("  DISAGREEMENT with metrics.json:")
            for line in run_disagreements:
                print(f"    {line}")
        else:
            print("  All 19 classes: recomputed precision, recall, F1 and support agree")
            print("  with metrics.json.")
        print()

        report["runs"][run["name"]] = {
            "unit": unit,
            "n_test": int(len(run["y_true"])),
            "label_order_differs_from_canonical": run["reordered"],
            "pairs": run_pairs,
            "disagreements_with_metrics_json": run_disagreements,
        }
        report["all_disagreements"].extend(
            f"{run['name']}: {line}" for line in run_disagreements
        )

    # One table across the four pairs: the record model on the two-tier split against
    # the sequence model on the same split.
    print("=" * 82)
    print("Within-pair confusion, record model against sequence model")
    print(f"  both on the two-tier split: {RECORD_RUN} and {SEQUENCE_RUN}")
    print("  rate = of items whose true class is one of the pair, the share predicted as")
    print("  the other member")
    print("=" * 82)
    header = (
        f"{'pair':<24} {'records':>10} {'windows':>10} {'factor':>9}"
        f" {'DoS->DDoS':>11} {'DDoS->DoS':>11}"
    )
    print(header)
    print("-" * len(header))
    for a_name, b_name in PAIRS:
        key = f"{a_name}|{b_name}"
        rec = report["runs"][RECORD_RUN]["pairs"][key]
        seq = report["runs"][SEQUENCE_RUN]["pairs"][key]
        rec_rate = rec["restricted_2x2"]["within_pair_confusion_rate"]
        seq_rate = seq["restricted_2x2"]["within_pair_confusion_rate"]
        factor = seq_rate / rec_rate if rec_rate else None
        forward = seq["predicted_as_partner"][a_name]["share"]
        backward = seq["predicted_as_partner"][b_name]["share"]
        print(
            f"{a_name + ' / ' + b_name:<24} {100 * rec_rate:>9.3f}% {100 * seq_rate:>9.3f}%"
            f" {(f'{factor:.1f}x' if factor else 'n/a'):>9}"
            f" {100 * forward:>10.2f}% {100 * backward:>10.2f}%"
        )
        report["summary_table"][key] = {
            "record_run": RECORD_RUN,
            "sequence_run": SEQUENCE_RUN,
            "record_within_pair_rate": rec_rate,
            "sequence_within_pair_rate": seq_rate,
            "factor": factor,
            "sequence_dos_to_ddos_share": forward,
            "sequence_ddos_to_dos_share": backward,
            "record_dos_to_ddos_share": rec["predicted_as_partner"][a_name]["share"],
            "record_ddos_to_dos_share": rec["predicted_as_partner"][b_name]["share"],
        }
    print()
    print("  The last two columns are the sequence run only, showing which way the")
    print("  confusion runs within each pair.")
    print()

    # Error concentration across all nineteen classes, sequence run.
    concentration = error_concentration(matrices[SEQUENCE_RUN], canonical_labels)
    report["error_concentration_sequence_run"] = {
        "run": SEQUENCE_RUN,
        "definition": (
            "for each class, the share of its misclassified items that land on the single "
            "most common other class"
        ),
        "classes": concentration,
    }
    pair_members = {name for pair in PAIRS for name in pair}

    print("=" * 82)
    print(f"Error concentration across all 19 classes, {SEQUENCE_RUN}")
    print("  share = of a class's errors, the fraction landing on one single other class")
    print("=" * 82)
    header = (
        f"{'class':<26} {'errors':>8} {'dests':>6} {'top destination':<26} {'share':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in concentration:
        mark = "*" if row["class"] in pair_members else " "
        share = row["top_destination_share_of_errors"]
        share_text = f"{100 * share:.2f}%" if share is not None else "no errors"
        destination = row["top_error_destination"] or "-"
        print(
            f"{mark}{row['class']:<25} {row['errors']:>8,} {row['n_distinct_error_destinations']:>6}"
            f" {destination:<26} {share_text:>8}"
        )
    print()
    print("  * marks a member of one of the four DoS/DDoS pairs.")

    with_errors = [r for r in concentration if r["top_destination_share_of_errors"] is not None]
    pair_shares = [
        r["top_destination_share_of_errors"] for r in with_errors if r["class"] in pair_members
    ]
    other_shares = [
        r["top_destination_share_of_errors"]
        for r in with_errors
        if r["class"] not in pair_members
    ]
    summary_stats = {
        "n_classes_with_errors": len(with_errors),
        "pair_members": {
            "n": len(pair_shares),
            "median_top_destination_share": float(np.median(pair_shares)) if pair_shares else None,
            "min": min(pair_shares) if pair_shares else None,
            "max": max(pair_shares) if pair_shares else None,
        },
        "non_pair_classes": {
            "n": len(other_shares),
            "median_top_destination_share": (
                float(np.median(other_shares)) if other_shares else None
            ),
            "min": min(other_shares) if other_shares else None,
            "max": max(other_shares) if other_shares else None,
        },
        "n_classes_above_0_90": sum(
            1 for r in with_errors if r["top_destination_share_of_errors"] >= 0.90
        ),
        "n_pair_members_above_0_90": sum(
            1
            for r in with_errors
            if r["class"] in pair_members and r["top_destination_share_of_errors"] >= 0.90
        ),
    }
    report["error_concentration_sequence_run"]["summary"] = summary_stats
    print()
    print(
        f"  Pair members, n={summary_stats['pair_members']['n']}: median top-destination share"
        f" {100 * summary_stats['pair_members']['median_top_destination_share']:.2f}%"
        f" (min {100 * summary_stats['pair_members']['min']:.2f}%,"
        f" max {100 * summary_stats['pair_members']['max']:.2f}%)"
    )
    print(
        f"  Other classes, n={summary_stats['non_pair_classes']['n']}: median"
        f" {100 * summary_stats['non_pair_classes']['median_top_destination_share']:.2f}%"
        f" (min {100 * summary_stats['non_pair_classes']['min']:.2f}%,"
        f" max {100 * summary_stats['non_pair_classes']['max']:.2f}%)"
    )
    print(
        f"  Classes sending 90% or more of their errors to one class:"
        f" {summary_stats['n_classes_above_0_90']} of {summary_stats['n_classes_with_errors']},"
        f" of which {summary_stats['n_pair_members_above_0_90']} are pair members."
    )
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2))

    print("=" * 82)
    if report["all_disagreements"]:
        print(f"{len(report['all_disagreements'])} disagreement(s) with metrics.json:")
        for line in report["all_disagreements"]:
            print(f"  {line}")
    else:
        print("Every recomputed figure agrees with the metrics.json it was checked against.")
    print(f"Written to {OUT_FILE.relative_to(REPO)}")
    print("=" * 82)


if __name__ == "__main__":
    main()
