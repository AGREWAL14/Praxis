"""What NB11's held-out evaluation actually scored, class by class.

The held-out captures cover the eight classes recorded more than once. The other eleven
classes have no window in that set at all. `runs.classification_metrics` is called with
the full nineteen-class label list, and `f1_score` with `average="macro"` and
`zero_division=0` gives a class with no support an F1 of 0 and still divides by the number
of labels. So the recorded macro-F1 is a sum over eight classes divided by nineteen, and
reads far below what the model did on the classes that were there.

This reads the run back and prints the per-class table, the support, and the macro-F1 over
the classes actually present alongside the one that was recorded. Where the prediction
arrays are on disk it recomputes both from them rather than trusting the stored figure.

    python scripts/nb11_heldout_breakdown.py
    python scripts/nb11_heldout_breakdown.py --run /path/to/NB11/pilot_adversarial
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
RUN_ID = "pilot_adversarial"


def artifacts_root():
    base = yaml.safe_load((REPO / "config" / "base.yaml").read_text())
    return Path(base["paths"]["artifacts"])


def find_run(explicit):
    if explicit is not None:
        return Path(explicit)
    for candidate in (
        REPO / "data" / "processed" / "NB11" / RUN_ID,
        REPO / "results" / "NB11" / RUN_ID,
        artifacts_root() / "NB11" / RUN_ID,
    ):
        if (candidate / "metrics.json").exists():
            return candidate
    return None


def macro_over(f1_by_class, chosen):
    chosen = list(chosen)
    if not chosen:
        return float("nan")
    return float(np.mean([f1_by_class[c] for c in chosen]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None, help="the run directory, if it is elsewhere")
    args = parser.parse_args(argv)

    run_dir = find_run(args.run)
    if run_dir is None:
        print("The NB11 run is not on this machine.")
        print()
        print("  looked for pilot_adversarial/metrics.json in:")
        for candidate in (REPO / "data" / "processed" / "NB11",
                          REPO / "results" / "NB11",
                          artifacts_root() / "NB11"):
            print(f"    {candidate}")
        print()
        print("  It is written to Drive by the training cell. Copy the run directory into")
        print("  the repository, or pass --run with the path to it.")
        return 2

    metrics = json.loads((run_dir / "metrics.json").read_text())
    labels = list(metrics["labels"])
    support = {k: int(v) for k, v in metrics["support"].items()}
    f1 = dict(metrics["per_class_f1"])
    precision = dict(metrics["per_class_precision"])
    recall = dict(metrics["per_class_recall"])

    present = [c for c in labels if support.get(c, 0) > 0]
    absent = [c for c in labels if support.get(c, 0) == 0]

    print(f"read {run_dir}")
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    if config:
        print(f"  lambda {config.get('lambda')}   epochs {config.get('epochs')}   "
              f"seed {config.get('seed')}   train_on_held_out {config.get('train_on_held_out')}")
    print(f"  {metrics['n_test']:,} windows scored against {len(labels)} labels")
    print()

    width = max(len(c) for c in labels)
    print(f"{'class':<{width}} {'support':>9} {'precision':>10} {'recall':>9} {'F1':>9}")
    print("-" * (width + 40))
    for c in sorted(labels, key=lambda k: (-support.get(k, 0), k)):
        mark = " " if support.get(c, 0) else "*"
        print(f"{mark}{c:<{width - 1}} {support.get(c, 0):>9,} {precision.get(c, 0):>10.4f}"
              f" {recall.get(c, 0):>9.4f} {f1.get(c, 0):>9.4f}")
    print("-" * (width + 40))
    print(f"* marks a class with no window in this set. There {'is' if len(absent) == 1 else 'are'} "
          f"{len(absent)} of them.")
    print()

    recorded = float(metrics["macro_f1"])
    over_present = macro_over(f1, present)
    print(f"macro-F1 as recorded, over all {len(labels)} labels : {recorded:.4f}")
    print(f"macro-F1 over the {len(present)} classes present        : {over_present:.4f}")
    print(f"accuracy                                    : {float(metrics['accuracy']):.4f}")
    print(f"weighted F1 as recorded                     : {float(metrics['weighted_f1']):.4f}")
    print()
    if present:
        print(f"The two differ by a factor of {len(labels)}/{len(present)} = "
              f"{len(labels) / len(present):.4f}, because the absent classes each contribute")
        print("an F1 of zero to the sum and one to the denominator.")
    print()
    print("Baselines for the set actually scored, against the ones stored, which describe a")
    print(f"{len(labels)}-class problem this evaluation is not:")
    biggest = max((support[c] for c in present), default=0)
    print(f"  chance          : {1 / max(len(present), 1):.4f}   stored {float(metrics['chance_rate']):.4f}")
    print(f"  majority class  : {biggest / max(metrics['n_test'], 1):.4f}   "
          f"stored {float(metrics['majority_class_rate']):.4f}")

    # Recompute rather than trust, where the arrays are there to do it with.
    arrays = {name: run_dir / f"{name}.npy" for name in ("y_true", "y_pred")}
    if all(p.exists() for p in arrays.values()):
        from sklearn.metrics import f1_score

        y_true = np.load(arrays["y_true"]).astype("int64")
        y_pred = np.load(arrays["y_pred"]).astype("int64")
        names = np.asarray(labels, dtype=str)
        true_names, pred_names = names[y_true], names[y_pred]
        all_labels = f1_score(true_names, pred_names, labels=labels, average="macro",
                              zero_division=0)
        just_present = f1_score(true_names, pred_names, labels=present, average="macro",
                                zero_division=0)
        print()
        print("recomputed from y_true.npy and y_pred.npy")
        print(f"  over all {len(labels)} labels : {all_labels:.6f}"
              f"   {'agrees' if abs(all_labels - recorded) < 5e-7 else 'DISAGREES'} with metrics.json")
        print(f"  over the {len(present)} present   : {just_present:.6f}"
              f"   {'agrees' if abs(just_present - over_present) < 5e-7 else 'DISAGREES'}"
              " with the per-class table")
        stray = sorted(set(pred_names.tolist()) - set(present))
        print(f"  predictions landing on a class with no support: "
              f"{stray if stray else 'none'}")
    else:
        print()
        print("y_true.npy and y_pred.npy are not beside metrics.json, so the figures above")
        print("are the stored ones rather than recomputed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
