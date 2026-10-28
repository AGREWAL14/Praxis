"""Which features separate each DoS/DDoS pair after a window is averaged, and does the
model use the best of them?

The pair separability tables carry three readings of all 44 features for each of the
four pairs, and the ledger block reduces each pair to a mean. This reads the tables back
and prints what that mean summarised away: which features still separate a pair once the
fifty records of a window are averaged, how many of them there are, and whether the same
features do the work for every pair or whether each pair is separated by its own.

Then one question about the ICMP pair. Take the single feature that separates it best on
the window mean, and look at where the windows the model got wrong actually sit on that
feature. Two readings are possible and the numbers decide between them:

  - the misclassified windows sit where the feature says DoS-ICMP, and the model is
    ignoring a signal that was available to it; or
  - they sit where it says DDoS-ICMP, and they are genuinely ambiguous windows, which
    makes the feature less clean than its AUC suggests.

Trains nothing, writes no artefact. Everything is recomputed from the saved tables, the
saved windows and the saved predictions.

    python scripts/pair_feature_readout.py
    python scripts/pair_feature_readout.py --windows /path/to/sequences_test.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import eda  # noqa: E402

PAIR_KEY = "DoS-ICMP|DDoS-ICMP"
TOP_N = 10
STRONG = (0.90, 0.80)
# Below this an AUC is read as "these two groups sit in the same place on this feature".
SAME_PLACE = 0.60


def artifacts_root():
    base = yaml.safe_load((REPO / "config" / "base.yaml").read_text())
    return Path(base["paths"]["artifacts"])


def first_existing(candidates):
    return next((Path(p) for p in candidates if Path(p).exists()), None)


def quartiles(values):
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def print_quartiles(label, stats):
    if stats is None:
        print(f"    {label:<44} no windows")
        return
    print(
        f"    {label:<44} n {stats['n']:>6,}   min {stats['min']:>12.4f}"
        f"   Q1 {stats['q1']:>12.4f}   median {stats['median']:>12.4f}"
        f"   Q3 {stats['q3']:>12.4f}   max {stats['max']:>12.4f}"
    )


def feature_readout(document):
    """Part one: what separates each pair once the window is averaged."""
    pairs = document["pairs"]
    tops = {}

    for key, rows in pairs.items():
        a, b = key.split("|")
        ordered = sorted(rows, key=lambda r: -r["auc_window_mean"])
        tops[key] = [r["feature"] for r in ordered[:TOP_N]]
        above = {cut: sum(1 for r in rows if r["auc_window_mean"] > cut) for cut in STRONG}

        print("=" * 96)
        print(f"{a} against {b}")
        print("=" * 96)
        print(
            f"  features above {STRONG[0]:.2f} on the window mean: {above[STRONG[0]]} of "
            f"{len(rows)}    above {STRONG[1]:.2f}: {above[STRONG[1]]} of {len(rows)}"
        )
        print()
        print(
            f"    {'feature':<22} {'auc_window_mean':>16} {'auc_record':>12}"
            f" {'auc_window_sd':>14}"
        )
        for row in ordered[:TOP_N]:
            print(
                f"    {row['feature']:<22} {row['auc_window_mean']:>16.4f}"
                f" {row['auc_record']:>12.4f} {row['auc_window_sd']:>14.4f}"
            )
        print()

    print("=" * 96)
    print(f"Do the same features head every pair? Top {TOP_N} by window mean, compared")
    print("=" * 96)
    keys = list(tops)
    shared = set.intersection(*(set(v) for v in tops.values())) if tops else set()
    print(f"  in every pair's top {TOP_N}: {sorted(shared) if shared else 'none'}")
    print()
    print(f"    {'':<24}" + "".join(f"{k.split('|')[0]:>14}" for k in keys))
    for key in keys:
        overlaps = "".join(f"{len(set(tops[key]) & set(tops[other])):>14}" for other in keys)
        print(f"    {key.split('|')[0]:<24}{overlaps}")
    print()
    print("  Cell values are how many features the two pairs' top lists share. The diagonal")
    print(f"  is {TOP_N} by construction.")
    print()
    for key in keys:
        others = [k for k in keys if k != key]
        unique = set(tops[key]) - set().union(*(set(tops[k]) for k in others))
        print(f"  only in {key.split('|')[0]:<10}: {sorted(unique) if unique else 'nothing'}")
    print()
    return tops


def icmp_check(document, windows_path, predictions_dir, manifest):
    """Part two: where the misclassified ICMP windows sit on the best feature."""
    rows = document["pairs"][PAIR_KEY]
    best = max(rows, key=lambda r: r["auc_window_mean"])
    feature = best["feature"]
    features = list(document["features"])
    classes = list(document["classes"])
    index = features.index(feature)
    dos, ddos = classes.index("DoS-ICMP"), classes.index("DDoS-ICMP")

    with np.load(windows_path, allow_pickle=False) as npz:
        in_file = [str(v) for v in npz["features"]]
        if in_file != features:
            raise ValueError(f"{windows_path} holds different columns than the tables list")
        column = npz["X"][:, :, index].astype("float64")
        y_true_windows = npz["y"].astype("int64")

    y_true = np.load(predictions_dir / "y_true.npy").astype("int64")
    y_pred = np.load(predictions_dir / "y_pred.npy").astype("int64")
    if not (len(y_true) == len(y_pred) == len(column)):
        raise ValueError(
            f"{len(column)} windows against {len(y_true)} true and {len(y_pred)} predicted"
        )
    if not np.array_equal(y_true, y_true_windows):
        raise ValueError("the saved predictions are not over the windows in this file")

    # Back onto the scale the files hold, so the quartiles read in the column's own units.
    scaler = manifest["scaler"]
    mean, scale = float(scaler["mean"][feature]), float(scaler["scale"][feature])
    window_mean = column.mean(axis=1) * scale + mean

    correct = (y_true == dos) & (y_pred == dos)
    swapped = (y_true == dos) & (y_pred == ddos)
    reference = y_true == ddos

    groups = {
        "true DoS-ICMP, predicted DoS-ICMP": window_mean[correct],
        "true DoS-ICMP, predicted DDoS-ICMP": window_mean[swapped],
        "true DDoS-ICMP, all windows": window_mean[reference],
    }
    stats = {name: quartiles(values) for name, values in groups.items()}

    print("=" * 96)
    print(f"ICMP pair: where the misclassified windows sit on {feature}")
    print("=" * 96)
    print(
        f"  {feature} is the best of the {len(rows)} features on the window mean for this "
        f"pair, at AUC {best['auc_window_mean']:.4f}"
    )
    print(f"  (per record {best['auc_record']:.4f}, on the window spread "
          f"{best['auc_window_sd']:.4f})")
    print()
    print(f"  window-mean {feature}, on the scale the files hold")
    for name, value in stats.items():
        print_quartiles(name, value)
    print()

    swapped_values = groups["true DoS-ICMP, predicted DDoS-ICMP"].reshape(-1, 1)
    correct_values = groups["true DoS-ICMP, predicted DoS-ICMP"].reshape(-1, 1)
    reference_values = groups["true DDoS-ICMP, all windows"].reshape(-1, 1)

    against_correct = float(eda.auc_by_column(swapped_values, correct_values)[0])
    against_reference = float(eda.auc_by_column(swapped_values, reference_values)[0])

    print("  Separability of the misclassified group from each of the other two, on this")
    print("  one feature. 0.5 means it sits in the same place as that group.")
    print(f"    against the DoS-ICMP windows the model got right : {against_correct:.4f}")
    print(f"    against the true DDoS-ICMP windows               : {against_reference:.4f}")
    print()

    # The rule, stated before the answer.
    print(f"  Reading the two: a value below {SAME_PLACE:.2f} means the misclassified windows")
    print("  sit with that group on this feature.")
    looks_like_dos = against_correct < SAME_PLACE
    looks_like_ddos = against_reference < SAME_PLACE

    if looks_like_dos and not looks_like_ddos:
        verdict = (
            f"The misclassified windows sit with the DoS-ICMP windows the model got right "
            f"and apart from the true DDoS-ICMP windows. On this feature they were "
            f"separable, and the model did not use it."
        )
    elif looks_like_ddos and not looks_like_dos:
        verdict = (
            f"The misclassified windows sit with the true DDoS-ICMP windows and apart from "
            f"the DoS-ICMP windows the model got right. On this feature they look like the "
            f"class they were predicted as, so the feature is less clean than its "
            f"{best['auc_window_mean']:.4f} suggests."
        )
    elif looks_like_dos and looks_like_ddos:
        verdict = (
            "The misclassified windows sit with both groups on this feature, which means "
            "the two groups are not far apart here and this feature does not separate the "
            "two readings."
        )
    else:
        verdict = (
            "The misclassified windows sit apart from both groups on this feature, so "
            "neither reading is supported: they are their own population here."
        )
    print(f"  {verdict}")
    print()
    return {
        "feature": feature,
        "auc_window_mean": best["auc_window_mean"],
        "auc_record": best["auc_record"],
        "auc_window_sd": best["auc_window_sd"],
        "quartiles": stats,
        "auc_swapped_against_correct": against_correct,
        "auc_swapped_against_ddos_reference": against_reference,
        "reading": verdict,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artefact", type=Path, default=None,
                        help="pair_separability.json, if it is not where this looks")
    parser.add_argument("--windows", type=Path, default=None,
                        help="sequences_test.npz, if it is not on the artifacts path")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="the directory holding y_true.npy and y_pred.npy")
    args = parser.parse_args(argv)

    artifacts = artifacts_root()
    artefact = args.artefact or first_existing([
        REPO / "data" / "processed" / "NB06d" / "pair_separability.json",
        artifacts / "NB06d" / "pair_separability.json",
    ])
    if artefact is None or not artefact.exists():
        print("The pair separability tables are not on this machine.")
        print()
        print("  looked for pair_separability.json in:")
        print(f"    {REPO / 'data' / 'processed' / 'NB06d'}")
        print(f"    {artifacts / 'NB06d'}")
        print()
        print("  That file is written by AG_PRAXIS_NB06d_dosddos_pair_diagnosis.ipynb, which")
        print("  has not been run. Nothing else in the repository holds per-feature")
        print("  separability for these pairs, so there is nothing for this script to read.")
        return 2

    document = json.loads(artefact.read_text())
    print(f"read {artefact}")
    print(f"  written {document.get('generated_on')} at {document.get('git_sha')}, "
          f"{document.get('test_windows'):,} test windows, "
          f"{len(document.get('features', []))} features")
    print()

    feature_readout(document)

    windows = args.windows or first_existing([artifacts / "NB04" / "sequences_test.npz"])
    predictions = args.predictions or first_existing([
        REPO / "data" / "processed" / "NB06" / "sequence_cnn_lstm_19class",
        artifacts / "NB06" / "sequence_cnn_lstm_19class",
    ])
    manifest_path = first_existing([
        REPO / "data" / "processed" / "NB04_manifest.json",
        artifacts / "NB04" / "NB04_manifest.json",
    ])

    missing = []
    if windows is None or not Path(windows).exists():
        missing.append("sequences_test.npz")
    if predictions is None:
        missing.append("y_true.npy and y_pred.npy")
    if manifest_path is None:
        missing.append("NB04_manifest.json")
    if missing:
        print("=" * 96)
        print("ICMP check skipped")
        print("=" * 96)
        print(f"  needs {', '.join(missing)}, which is not on this machine. The window")
        print("  arrays live on Drive rather than in the repository. Pass --windows to")
        print("  point at them, or run this where they are mounted.")
        return 0

    icmp_check(document, Path(windows), Path(predictions),
               json.loads(manifest_path.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
