"""What NB09a needs from a fixture, and what it should have written when it stops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .nb07b import Profile


@dataclass
class NB09a(Profile):
    def check_scale(self, repo_root: Path):
        manifest = json.loads((repo_root / "data" / "processed" / "NB04_manifest.json").read_text())
        seq = int(manifest["arrays"]["sequences_test"]["shape"][0])
        rec = int(manifest["arrays"]["records_test"]["shape"][0])
        yield ("the real sequence test partition", seq == 49159, f"{seq:,} windows")
        yield ("the real record test partition", rec == 1229711, f"{rec:,} records")
        yield ("the timing-excluded slice keeps 40",
               len(manifest["timing_excluded_slice"]["keep"]) == 40,
               f"{len(manifest['timing_excluded_slice']['keep'])} features, dropping "
               f"{manifest['timing_excluded_slice']['drop']}")
        forest = json.loads((repo_root / "results" / "NB05" / "forest_19class" / "config.json").read_text())
        yield ("the forest parent is NB05's, not NB03's",
               int(forest["n_estimators"]) == 100 and int(forest["min_samples_leaf"]) == 20,
               f"{forest['n_estimators']} trees, min leaf {forest['min_samples_leaf']}, "
               f"cap {forest['train_rows_cap']:,}")

    def check_artefacts(self, shadow: Path, namespace: dict):
        out = shadow / "artifacts" / "NB09a"
        five = ("config.json", "metrics.json", "y_true.npy", "y_pred.npy")
        for seed in (42, 43, 44, 45, 46):
            d = out / f"seq_timing_excluded_seed_{seed}"
            missing = [n for n in five if not (d / n).exists()]
            yield (f"seed {seed} wrote its files", not missing,
                   "present" if not missing else f"missing {missing}")

        forest = out / "forest_timing_excluded"
        yield ("the forest wrote its files", (forest / "metrics.json").exists(),
               "present" if (forest / "metrics.json").exists() else "absent")
        if (forest / "metrics.json").exists():
            m = json.loads((forest / "metrics.json").read_text())
            cv = m.get("cross_validation", {})
            yield ("the forest actually cross-validated, as its config claims",
                   int(cv.get("n_splits", 0)) == 5 and len(cv.get("folds", [])) == 5,
                   f"{cv.get('n_splits')} splits, {len(cv.get('folds', []))} folds recorded")

        import numpy as np

        per_seed = sorted(out.glob("attributions_seed_*.npz"))
        yield ("one attribution file per seed, written as each pass finished",
               len(per_seed) == 5, f"{len(per_seed)} files: {[q.name for q in per_seed]}")
        for path in per_seed:
            with np.load(path, allow_pickle=False) as f:
                table, classes, features = f["attributions"], f["classes"], f["features"]
                n = int(f["nsamples"])
            yield (f"{path.name} is nineteen classes by forty features, nsamples recorded",
                   table.shape == (19, 40) and len(classes) == 19 and len(features) == 40
                   and n == 50, f"{table.shape}, nsamples {n}")
            yield (f"{path.name} holds no negative value, the aggregation being absolute",
                   bool((table >= 0).all()), "yes")

        forest_path = out / "attributions_forest.npz"
        yield ("the forest wrote its own attribution file", forest_path.exists(),
               "present" if forest_path.exists() else "absent")
        yield ("no single combined attributions.npz, which the per-seed writes replace",
               not (out / "attributions.npz").exists(), "absent as intended")

        doc = out / "attributions.json"
        if doc.exists():
            d = json.loads(doc.read_text())
            env = d.get("environment", {})
            yield ("the environment records shap and scipy, not just tensorflow and keras",
                   {"shap", "scipy", "tensorflow", "keras", "accelerator"} <= set(env),
                   f"records {sorted(env)}")
            yield ("the background was drawn once and the document says so",
                   d.get("background", {}).get("drawn_once") is True, "recorded")
            yield ("nsamples is recorded and is Amendment 19's value",
                   d.get("nsamples") == 50, str(d.get("nsamples")))
            yield ("the resume rule is recorded",
                   d.get("resume", {}).get("enabled") is True, "recorded")
        else:
            yield ("attributions.json was written", False, "absent")


PROFILE = NB09a(
    name="nb09a",
    notebook="AG_PRAXIS_NB09a_shap_attributions.ipynb",
    fast_env="0",
    training_path="keras_fit",
    max_traces=2,
    scale_asserts=[
        'assert len(SEQ_TEST_Y) == 49159 and len(REC_TEST_Y) == 1229711',
    ],
    fixture={"classes": 19, "features": 44, "window": 50, "stride": 25,
             "train_windows_per_capture": 24, "test_windows_per_capture": 12,
             "val_windows_per_capture": 12, "needs_parent_model": False,
             "needs_records": True, "needs_forest_parent": True},
)
