"""What NB07b needs from a fixture, and what it should have written when it stops.

NB07b is withdrawn under `DECISIONS.md` 2026-08-11 and will not be run. This profile is
kept anyway, because it is the only thing that exercises the group-DRO machinery in
`src/interventions.py` and `src/sequence.py`. Running it is how that code is known still
to work: the compiled step traces, the weights stay a distribution over 45 groups, the
trajectory is written once an epoch, and the five files land. Delete this and that code
has nothing calling it at all.

A profile is a description, not a script. It says which notebook to run, what the
fixture has to contain for that notebook's own asserts to hold, which asserts
cannot hold against a small fixture and are checked against the real metadata
instead, and what the run directory should look like afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Profile:
    name: str
    notebook: str
    fast_env: str = "0"
    training_path: str = "custom"          # "custom" or "keras_fit"
    max_traces: int = 2
    scale_asserts: list = field(default_factory=list)
    fixture: dict = field(default_factory=dict)

    def check_scale(self, repo_root: Path):
        raise NotImplementedError

    def check_artefacts(self, shadow: Path, namespace: dict):
        raise NotImplementedError


@dataclass
class NB07b(Profile):
    def check_scale(self, repo_root: Path):
        """The asserts the fixture cannot satisfy, read off the real metadata instead."""
        manifest = json.loads((repo_root / "data" / "processed" / "NB04_manifest.json").read_text())
        parent = json.loads((repo_root / "data" / "processed" / "NB06" / "metrics.json").read_text())

        test = int(manifest["arrays"]["sequences_test"]["shape"][0])
        train = int(manifest["arrays"]["sequences_train"]["shape"][0])
        yield ("the real test partition is 49,159 windows", test == 49159,
               f"NB04_manifest.json says {test:,}")
        yield ("the parent was scored on the same", int(parent["n_test"]) == test,
               f"NB06/metrics.json says {int(parent['n_test']):,}")
        yield ("the real train partition matches the manifest", train == 249061,
               f"NB04_manifest.json says {train:,}")
        yield ("the parent's parameter count", int(parent["n_parameters"]) == 214227,
               f"NB06/metrics.json says {int(parent['n_parameters']):,}")

    def check_artefacts(self, shadow: Path, namespace: dict):
        """What the run left behind, read back off disk rather than off the namespace."""
        out = shadow / "artifacts" / "NB07b"
        five = ("config.json", "metrics.json", "y_true.npy", "y_pred.npy", "model.keras")

        for run in ("loop_check_batch_share", "capture_invariant_dro"):
            directory = out / run
            missing = [name for name in five if not (directory / name).exists()]
            yield (f"{run} wrote its five files", not missing,
                   "all five present" if not missing else f"missing {missing}")

        path = out / "capture_invariant_dro" / "metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text())
            # The ledger block is an f-string over these. A key missing here is a
            # KeyError in the last cell of a paid session.
            wanted = ["macro_f1", "weighted_f1", "accuracy", "per_class_f1", "n_test",
                      "n_parameters", "train_seconds", "labels", "group_dro", "history"]
            absent = [key for key in wanted if key not in metrics]
            yield ("metrics.json carries every key the ledger block reads", not absent,
                   "all present" if not absent else f"missing {absent}")

            dro = metrics.get("group_dro", {})
            trajectory = dro.get("trajectory", [])
            yield ("the weight trajectory was written, one entry an epoch", len(trajectory) == 10,
                   f"{len(trajectory)} entries")
            if trajectory:
                weights = trajectory[-1].get("weights", {})
                total = sum(weights.values())
                yield ("the weights are a distribution", abs(total - 1.0) < 1e-6,
                       f"they sum to {total:.6f} over {len(weights)} groups")
            yield ("every group's batch count was recorded",
                   len(dro.get("batches_each_group_appeared_in", {})) == 45,
                   f"{len(dro.get('batches_each_group_appeared_in', {}))} groups")

        summary = out / "duration_ablation.json"
        yield ("no stray artefact from another notebook", not summary.exists(),
               "clean" if not summary.exists() else "NB03b's summary is in NB07b's directory")

        for name, value in (("CHECK_PASSES", bool), ("PROBE_TABLE", None)):
            yield (f"{name} exists in the namespace after the run", name in namespace,
                   "yes" if name in namespace else "the cell that defines it did not run")


PROFILE = NB07b(
    name="nb07b",
    notebook="AG_PRAXIS_NB07b_capture_invariant.ipynb",
    fast_env="0",
    training_path="custom",
    max_traces=2,
    # The only place this dry run knowingly diverges from the notebook. Approved as
    # exactly these three; anything else failing is a real failure.
    scale_asserts=[
        'assert len(TEST["y"]) == 49159',
        'assert len(TRAIN["y"]) == int(MANIFEST["arrays"]["sequences_train"]["shape"][0])',
    ],
    fixture={
        "classes": 19,
        "features": 44,
        "window": 50,
        "stride": 25,
        # 8 classes recorded more than once, holding 34 captures between them, and 11
        # recorded once. 45 in training, 57 in the corpus, so 12 are never seen.
        "captures_per_multi_class": [8, 3, 3, 8, 3, 3, 3, 3],
        "single_capture_classes": 11,
        "corpus_captures": 57,
        "train_windows_per_capture": 24,
        "test_windows_per_capture": 12,
        "val_windows_per_capture": 12,
        "needs": [
            "data/processed/NB04_manifest.json",
            "data/processed/NB06/config.json",
            "data/processed/NB06/metrics.json",
            "data/processed/NB03_verdict.json",
            "data/processed/NB08/sequence_budget_{05,10,25}/metrics.json",
            "results/NB07/{five runs}/metrics.json",
            "artifacts/NB04/sequences_{train,val,test}.npz",
            "artifacts/NB06/sequence_cnn_lstm_19class/model.keras",
        ],
    },
)
