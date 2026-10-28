"""What NB09b needs from a fixture, and what it should have written when it stops.

NB09b calls the openFDA device event API. A dry run that reached the live service would
not be a dry run, and it would spend requests against a limit of a thousand a day per IP.
So `urllib.request.urlopen` is stubbed for the duration, and the canned responses cover
the three paths that can silently produce a wrong number: an ordinary count, a 404, which
is what openFDA returns for a search matching nothing and is the branch most likely to
fire for real given the enumeration was fixed without probing FDA's vocabulary, and an
overlap where one record carries two generic names, so the de-duplicated union comes out
below the sum of the per-term counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .nb07b import Profile

# term -> record count. "sleep monitor" is the vocabulary miss. The union is set below the
# sum on purpose: pulse oximeter and electrocardiograph share records in this fixture.
CANNED_COUNTS = {
    "pulse+oximeter": 1200, "electrocardiograph": 800, "heart+rate+monitor": 300,
    "electromyograph": 40, "galvanic+skin+response+device": 5, "sleep+monitor": None,
    "infant+monitor": 90, "personal+emergency+response+system": 60,
}
CANNED_UNION = 2100          # below the 2,495 the per-term counts sum to
CANNED_KEYWORD = 3


@dataclass
class NB09b(Profile):
    def check_scale(self, repo_root: Path):
        gt = repo_root / "config" / "stride_ground_truth.yaml"
        import yaml

        d = yaml.safe_load(gt.read_text())
        yield ("the ground truth holds 18 attack classes", len(d["classes"]) == 18,
               f"{len(d['classes'])} classes, Benign excluded")
        yield ("the majority-class baseline in the file matches its own entries",
               abs(d["majority_class_baseline"]["fraction"] - 12 / 18) < 1e-3,
               f"{d['majority_class_baseline']['fraction']}")
        m = yaml.safe_load((repo_root / "config" / "capec_stride.yaml").read_text())["mapping"]
        yield ("every CAPEC-to-STRIDE row is verified against its source",
               all(v["verified_against_source"] for v in m.values()),
               f"{len(m)} rows, none unverified")

    def check_artefacts(self, shadow: Path, namespace: dict):
        path = shadow / "artifacts" / "NB09b" / "threat_mapping.json"
        yield ("threat_mapping.json was written", path.exists(),
               "present" if path.exists() else "absent")
        if not path.exists():
            return
        d = json.loads(path.read_text())

        yield ("both model families were evaluated", len(d["h3"]) == 2,
               ", ".join(r["model"] for r in d["h3"]))
        for r in d["h3"]:
            yield (f"{r['model'][:34]}: assignment and agreement counted separately",
                   "n_assigned" in r and "n_agree" in r,
                   f"{r['n_assigned']} assigned, {r['n_agree']} agree of 18")
            yield (f"{r['model'][:34]}: agreement never exceeds assignment",
                   r["n_agree"] <= r["n_assigned"], f"{r['n_agree']} <= {r['n_assigned']}")
        yield ("the majority-class baseline travels with the result",
               abs(d["majority_class_baseline"] - 12 / 18) < 1e-3, str(d["majority_class_baseline"]))
        yield ("the pass mark is 13 of 18", d["pass_mark"] == 13 and d["n_attack_classes"] == 18,
               f"{d['pass_mark']} of {d['n_attack_classes']}")

        tau = d["kendall_tau"]
        yield ("tau covers every class over ten seed pairs",
               len(tau["per_class"]) == 19 and tau["pairs"] == 10,
               f"{len(tau['per_class'])} classes, {tau['pairs']} pairs")
        yield ("tau carries no pass mark", tau.get("no_pass_mark") is True, "recorded")

        maude = d["maude"]
        zero = [r["generic_name"] for r in maude["per_generic_name"] if r["records"] == 0]
        yield ("a term returning nothing is visible rather than absorbed", bool(zero),
               f"zero for {zero}" if zero else "no term returned zero in the fixture")
        summed = sum(r["records"] for r in maude["per_generic_name"])
        yield ("the denominator is de-duplicated, not the sum of the per-term counts",
               maude["denominator"] < summed,
               f"union {maude['denominator']:,} against a sum of {summed:,}")
        yield ("the numerator cannot exceed the denominator",
               maude["numerator"] <= maude["denominator"],
               f"{maude['numerator']:,} of {maude['denominator']:,}")
        yield ("counts are labelled as device-category events, not as cyber-caused harm",
               "never as cyber-caused harm" in maude.get("reported_as", ""),
               maude.get("reported_as", ""))


PROFILE = NB09b(
    name="nb09b",
    notebook="AG_PRAXIS_NB09b_threat_mapping.ipynb",
    fast_env="0",
    training_path="keras_fit",
    max_traces=2,
    scale_asserts=[],
    fixture={"classes": 19, "features": 44, "window": 50, "stride": 25,
             "train_windows_per_capture": 24, "test_windows_per_capture": 12,
             "val_windows_per_capture": 12, "needs_parent_model": False,
             "needs_nb09a_artifacts": True, "stub_openfda": True,
             "canned_counts": CANNED_COUNTS, "canned_union": CANNED_UNION,
             "canned_keyword": CANNED_KEYWORD},
)
