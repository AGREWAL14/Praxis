"""What NB08b needs from a fixture, and what it should have written when it stops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .nb07b import Profile


@dataclass
class NB08b(Profile):
    def check_scale(self, repo_root: Path):
        manifest = json.loads((repo_root / "data" / "processed" / "NB04_manifest.json").read_text())
        test = int(manifest["arrays"]["sequences_test"]["shape"][0])
        yield ("the real test partition is 49,159 windows", test == 49159,
               f"NB04_manifest.json says {test:,}")
        for k in (5, 10, 25):
            path = repo_root / "data" / "processed" / "NB08" / f"sequence_budget_{k:02d}" / "metrics.json"
            metrics = json.loads(path.read_text())
            yield (f"the real k={k} run was scored on the same windows",
                   int(metrics["n_test"]) == test,
                   f"{int(metrics['n_test']):,} windows, macro-F1 {metrics['macro_f1']:.4f}")
        parent = json.loads((repo_root / "data" / "processed" / "NB06" / "metrics.json").read_text())
        yield ("k=50 is the parent, not an NB08 run", int(parent["n_test"]) == test,
               f"NB06/metrics.json, macro-F1 {parent['macro_f1']:.4f}")

    def check_artefacts(self, shadow: Path, namespace: dict):
        out = shadow / "artifacts" / "NB08b"
        run = out / "adaptive_prefix_19class"
        five = ("config.json", "metrics.json", "y_true.npy", "y_pred.npy", "model.keras")
        missing = [name for name in five if not (run / name).exists()]
        yield ("the run wrote its five files", not missing,
               "all five present" if not missing else f"missing {missing}")

        if (run / "metrics.json").exists():
            metrics = json.loads((run / "metrics.json").read_text())
            wanted = ["macro_f1", "weighted_f1", "accuracy", "per_class_f1", "n_test",
                      "n_parameters", "train_seconds", "labels", "prefix_lengths"]
            absent = [key for key in wanted if key not in metrics]
            yield ("metrics.json carries every key the ledger block reads", not absent,
                   "all present" if not absent else f"missing {absent}")
            drawn = metrics.get("prefix_lengths", {})
            yield ("the prefix lengths drawn were recorded",
                   1 <= int(drawn.get("min", 0)) and int(drawn.get("max", 0)) <= 50,
                   f"{drawn.get('n_batches', 0)} batches, {drawn.get('min')} to "
                   f"{drawn.get('max')}, mean {drawn.get('mean')}")
            yield ("the lengths were drawn per epoch, not once",
                   len(drawn.get("per_epoch", [])) == 10,
                   f"{len(drawn.get('per_epoch', []))} epochs recorded")

        summary = out / "adaptive_earliness.json"
        if summary.exists():
            document = json.loads(summary.read_text())
            yield ("the sweep covers all twelve taus", len(document.get("tau_grid", [])) == 12,
                   f"{len(document.get('tau_grid', []))} values")
            yield ("one curve row per tau", len(document.get("curve", [])) == 12,
                   f"{len(document.get('curve', []))} rows")
            yield ("per-class rows are twelve taus by nineteen classes",
                   len(document.get("per_class", [])) == 12 * 19,
                   f"{len(document.get('per_class', []))} rows")
        else:
            yield ("adaptive_earliness.json was written", False, "it is not there")

        yield ("the figure was written", (out / "NB08b_earliness_curve.png").exists(),
               "NB08b_earliness_curve.png")
        yield ("no figure number was baked into the notebook", "FIGURE_NUMBER" not in namespace,
               "numbering is left to the chapter")


PROFILE = NB08b(
    name="nb08b",
    notebook="AG_PRAXIS_NB08b_adaptive_earliness.ipynb",
    fast_env="0",
    training_path="keras_fit",
    max_traces=2,
    scale_asserts=['assert len(Y_TEST) == 49159'],
    fixture={"classes": 19, "features": 44, "window": 50, "stride": 25,
             "train_windows_per_capture": 24, "test_windows_per_capture": 12,
             "val_windows_per_capture": 12, "needs_parent_model": False},
)
