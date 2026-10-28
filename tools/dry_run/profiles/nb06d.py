"""What NB06d needs from a fixture, and what it should have written when it stops.

NB06d fits nothing. It reads the saved test windows, reads the predictions the window
model wrote when it was scored, and measures how well each feature separates the two
members of a DoS and DDoS pair before and after the fifty records of a window are
averaged. So the two things worth checking here are that its own asserts hold against a
fixture built to satisfy them, and that the inputs it will meet in Colab are actually
there and the shape it expects.

The second is what `check_scale` does. A dry run cannot read Drive, but the manifest and
the committed predictions are in the repository, and they are enough to say that the test
partition is the size the notebook will find, that all eight classes of the four pairs are
in it, and that the predictions cover the same windows. A notebook that fails on a missing
input fails after the session has started.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .nb07b import Profile

PAIRS = [("DoS-ICMP", "DDoS-ICMP"), ("DoS-TCP", "DDoS-TCP"),
         ("DoS-SYN", "DDoS-SYN"), ("DoS-UDP", "DDoS-UDP")]


@dataclass
class NB06d(Profile):
    def check_scale(self, repo_root: Path):
        """The real inputs, read off the repository rather than off the fixture."""
        manifest = json.loads(
            (repo_root / "data" / "processed" / "NB04_manifest.json").read_text())
        test = manifest["arrays"]["sequences_test"]
        by_class = test["by_class"]

        yield ("the real test partition is 49,159 windows", int(test["shape"][0]) == 49159,
               f"the manifest says {int(test['shape'][0]):,}")
        yield ("cut at window 50 and stride 25",
               (int(test["window"]), int(test["stride"])) == (50, 25),
               f"window {test['window']}, stride {test['stride']}")
        yield ("44 features, the set the notebook asserts",
               int(test["n_features"]) == 44 and len(manifest["columns"]["kept"]) == 44,
               f"{int(test['n_features'])} in the array, "
               f"{len(manifest['columns']['kept'])} kept columns")
        yield ("the scaler covers every kept column, so the values can be unscaled",
               len(manifest["scaler"]["mean"]) == 44 and len(manifest["scaler"]["scale"]) == 44,
               f"{len(manifest['scaler']['mean'])} means, "
               f"{len(manifest['scaler']['scale'])} scales")
        yield ("Number is among the kept columns, which one packet-count reading needs",
               "Number" in manifest["columns"]["kept"],
               "present" if "Number" in manifest["columns"]["kept"] else "absent")

        missing = [name for pair in PAIRS for name in pair if name not in by_class]
        yield ("all eight classes of the four pairs are in the test partition", not missing,
               "all present" if not missing else f"missing {missing}")
        if not missing:
            yield ("no pair is too thin to read",
                   all(int(by_class[name]) >= 100 for pair in PAIRS for name in pair),
                   ", ".join(f"{a} {int(by_class[a]):,}/{b} {int(by_class[b]):,}"
                             for a, b in PAIRS))

        # The predictions the notebook reads. Committed, so this is checkable here.
        where = repo_root / "data" / "processed" / "NB06" / "sequence_cnn_lstm_19class"
        arrays = {name: where / f"{name}.npy" for name in ("y_true", "y_pred")}
        absent = [str(p.relative_to(repo_root)) for p in arrays.values() if not p.exists()]
        yield ("the saved predictions are in the repository", not absent,
               "both present" if not absent else f"missing {absent}")
        if absent:
            return
        y_true = np.load(arrays["y_true"])
        y_pred = np.load(arrays["y_pred"])
        yield ("the two prediction arrays are the same length and cover the test partition",
               y_true.shape == y_pred.shape and len(y_true) == int(test["shape"][0]),
               f"{len(y_true):,} and {len(y_pred):,} against {int(test['shape'][0]):,}")

        # The notebook settles the code-to-name mapping by comparing per-class counts, and
        # that check only means something if the counts are actually distinct.
        classes = sorted(by_class)
        counts = {classes[c]: int((y_true == c).sum()) for c in range(len(classes))}
        disagreeing = {k: (v, int(by_class[k])) for k, v in counts.items()
                       if v != int(by_class[k])}
        yield ("the predictions carry the same per-class counts as the manifest",
               not disagreeing,
               "every class agrees" if not disagreeing else f"{disagreeing}")
        pair_counts = [int(by_class[name]) for pair in PAIRS for name in pair]
        yield ("the eight pair classes have distinct counts, so the mapping check bites",
               len(set(pair_counts)) == len(pair_counts),
               f"{len(set(pair_counts))} distinct counts over {len(pair_counts)} classes")

    def check_artefacts(self, shadow: Path, namespace: dict):
        """What the run left behind, read back off disk rather than off the namespace."""
        out = shadow / "artifacts" / "NB06d"
        path = out / "pair_separability.json"

        yield ("pair_separability.json was written", path.exists(),
               "present" if path.exists() else "absent")
        figures = ["NB06d_record_against_window_mean.png",
                   "NB06d_separability_lost_by_pair.png"]
        missing = [name for name in figures if not (out / name).exists()]
        yield ("both figures were written", not missing,
               "both present" if not missing else f"missing {missing}")
        if not path.exists():
            return

        document = json.loads(path.read_text())
        yield ("it records that nothing was trained", document.get("trained") == "nothing",
               str(document.get("trained")))
        yield ("the environment travels with the run",
               bool(document.get("environment", {}).get("tensorflow")),
               str(document.get("environment", {}).get("tensorflow")))

        pairs = document.get("pairs", {})
        yield ("all four pairs are in the file", len(pairs) == 4, f"{sorted(pairs)}")
        widths = {key: len(rows) for key, rows in pairs.items()}
        yield ("every pair table covers all 44 features",
               all(width == 44 for width in widths.values()), str(widths))

        # Three readings per feature, and the derived columns that read them against
        # each other. A missing column here is a KeyError in the figure cells.
        wanted = {"feature", "auc_record", "auc_window_mean", "auc_window_sd",
                  "lost_to_averaging", "recovered_by_spread", "sd_ratio"}
        first = next(iter(pairs.values()))[0] if pairs else {}
        yield ("each row carries all three readings and the differences between them",
               wanted <= set(first), f"missing {sorted(wanted - set(first))}" if first else "no rows")

        bounded = [
            (key, name, value)
            for key, rows in pairs.items()
            for row in rows
            for name in ("auc_record", "auc_window_mean", "auc_window_sd")
            for value in [row[name]]
            if value is not None and not (0.5 - 1e-9 <= float(value) <= 1.0 + 1e-9)
        ]
        yield ("every AUC is best-direction, so none falls below 0.5 or above 1.0",
               not bounded, "all in range" if not bounded else f"{bounded[:3]}")

        ordering = document.get("ordering", {})
        yield ("both orderings over the four pairs were computed and compared",
               len(ordering.get("by_separability_lost", [])) == 4
               and len(ordering.get("by_confusion_rate", [])) == 4
               and "agree" in ordering,
               f"{ordering.get('by_separability_lost')} against "
               f"{ordering.get('by_confusion_rate')}, agree={ordering.get('agree')}")

        confusion = document.get("confusion_from_saved_predictions", [])
        yield ("the confusion rate was counted for each pair from the saved predictions",
               len(confusion) == 4 and all("confusion rate" in row for row in confusion),
               ", ".join(f"{row['pair']} {row['confusion rate']:.3f}" for row in confusion))

        packets = document.get("packets_per_row", [])
        yield ("a packet-count reading was attempted for all 19 classes", len(packets) == 19,
               f"{len(packets)} classes")
        yield ("the packet-count method records whether the two groups separated",
               "separates_the_pair_classes" in document.get("packets_per_row_method", {}),
               str(document.get("packets_per_row_method", {}).get("separates_the_pair_classes")))

        for name in ("TABLES", "CROSS", "CONFUSION", "PACKETS", "SPREAD_TABLE"):
            yield (f"{name} exists in the namespace after the run", name in namespace,
                   "yes" if name in namespace else "the cell that defines it did not run")


PROFILE = NB06d(
    name="nb06d",
    notebook="AG_PRAXIS_NB06d_dosddos_pair_diagnosis.ipynb",
    fast_env="0",
    training_path="none",
    max_traces=0,
    scale_asserts=[],
    fixture={"classes": 19, "features": 44, "window": 50, "stride": 25,
             "train_windows_per_capture": 24, "test_windows_per_capture": 12,
             "val_windows_per_capture": 12, "needs_parent_model": False,
             "needs_parent_predictions": True},
)
