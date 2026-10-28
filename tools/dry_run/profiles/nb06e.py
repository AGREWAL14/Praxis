"""What NB06e needs from a fixture, and what it should have written when it stops.

NB06e fits nothing. It reads the saved test records, the saved test windows, and the
predictions both models wrote when they were scored, maps every window to its fifty
records, and votes the forest's per-record predictions into a window prediction so the two
models can be compared item by item. So what is worth checking here is that its alignment
asserts hold against a fixture built the way the real arrays are built, and that the inputs
it will meet in Colab are there and the size it expects.

The second is what `check_scale` does. A dry run cannot read Drive, but the manifest and
both sets of committed predictions are in the repository, and together they say that the
test partition is the 49,159 windows and 1,229,711 records the notebook pins, that the
forest was scored on the record count and the sequence model on the window count, and that
the label encoding the notebook searches for exists. A notebook that fails on a missing
input fails after the session has started.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .nb07b import Profile

COMBINER_CLASSES = ["Recon-VulScan", "Recon-OS_Scan", "MQTT-DDoS-Publish_Flood",
                    "Spoofing", "MQTT-Malformed_Data"]


@dataclass
class NB06e(Profile):
    def check_scale(self, repo_root: Path):
        """The real inputs, read off the repository rather than off the fixture."""
        manifest = json.loads(
            (repo_root / "data" / "processed" / "NB04_manifest.json").read_text())
        windows = manifest["arrays"]["sequences_test"]
        records = manifest["arrays"]["records_test"]

        n_windows = int(windows["shape"][0])
        n_records = int(records["shape"][0])
        yield ("the real test partition is 49,159 windows", n_windows == 49159,
               f"the manifest says {n_windows:,}")
        yield ("and 1,229,711 records", n_records == 1229711,
               f"the manifest says {n_records:,}")
        yield ("cut at window 50 and stride 25",
               (int(windows["window"]), int(windows["stride"])) == (50, 25),
               f"window {windows['window']}, stride {windows['stride']}")

        # 49,159 windows of 50 records at stride 25 cannot cover more rows than the
        # partition holds. If they could, the mapping the notebook builds is impossible
        # before it reads anything.
        yield ("the windows cannot cover more records than the partition holds",
               n_windows * 50 >= n_windows and (n_windows - 1) * 25 + 50 <= n_records,
               f"{(n_windows - 1) * 25 + 50:,} rows spanned against {n_records:,} records")

        forest = repo_root / "results" / "NB05" / "forest_19class"
        sequence = repo_root / "data" / "processed" / "NB06" / "sequence_cnn_lstm_19class"
        wanted = {
            "forest y_true": forest / "y_true.npy",
            "forest y_pred": forest / "y_pred.npy",
            "forest config": forest / "config.json",
            "sequence y_true": sequence / "y_true.npy",
            "sequence y_pred": sequence / "y_pred.npy",
        }
        absent = [name for name, path in wanted.items() if not path.exists()]
        yield ("both models' saved predictions are in the repository", not absent,
               "all present" if not absent else f"missing {absent}")

        found = sorted((repo_root / "data" / "processed" / "NB08").rglob("label_map.json"))
        yield ("the label encoding NB08 wrote is findable under data/processed/NB08",
               bool(found),
               str(found[0].relative_to(repo_root)) if found else "nowhere under NB08")
        if absent or not found:
            return

        labels = [str(name) for name in json.loads(found[0].read_text())["labels"]]
        config = json.loads(wanted["forest config"].read_text())
        yield ("the forest's config lists the same 19 classes in the same order",
               [str(c) for c in config["observed"]["classes"]] == labels and len(labels) == 19,
               f"{len(labels)} classes, identical order"
               if [str(c) for c in config["observed"]["classes"]] == labels else "they differ")

        forest_true = np.load(wanted["forest y_true"])
        forest_pred = np.load(wanted["forest y_pred"])
        sequence_true = np.load(wanted["sequence y_true"])
        sequence_pred = np.load(wanted["sequence y_pred"])
        yield ("the forest was scored on the records, one prediction per record",
               forest_true.shape == forest_pred.shape and len(forest_true) == n_records,
               f"{len(forest_true):,} against {n_records:,}")
        yield ("the sequence model was scored on the windows, one prediction per window",
               sequence_true.shape == sequence_pred.shape and len(sequence_true) == n_windows,
               f"{len(sequence_true):,} against {n_windows:,}")

        # The notebook takes the two encodings as the same one and asserts it. The
        # per-class counts of the saved labels are checkable here without Drive.
        classes = sorted(windows["by_class"])
        window_counts = {classes[c]: int((sequence_true == c).sum()) for c in range(len(classes))}
        record_counts = {classes[c]: int((forest_true == c).sum()) for c in range(len(classes))}
        disagreeing = {
            name: (window_counts[name], int(windows["by_class"][name]),
                   record_counts[name], int(records["by_class"][name]))
            for name in classes
            if window_counts[name] != int(windows["by_class"][name])
            or record_counts[name] != int(records["by_class"][name])
        }
        yield ("both sets of saved labels carry the manifest's per-class counts",
               not disagreeing,
               "every class agrees in both" if not disagreeing else str(disagreeing))

        missing = [name for name in COMBINER_CLASSES if name not in windows["by_class"]]
        yield ("the five Amendment 6 classes the combiner routes on are in the test windows",
               not missing,
               ", ".join(f"{name} {int(windows['by_class'][name]):,}" for name in COMBINER_CLASSES)
               if not missing else f"missing {missing}")
        yield ("DoS-ICMP, reported on its own line under the combiner, is in the test windows",
               "DoS-ICMP" in windows["by_class"],
               f"{int(windows['by_class'].get('DoS-ICMP', 0)):,} windows")

    def check_artefacts(self, shadow: Path, namespace: dict):
        """What the run left behind, read back off disk rather than off the namespace."""
        path = shadow / "artifacts" / "NB06e" / "forest_window_comparison.json"
        yield ("forest_window_comparison.json was written", path.exists(),
               "present" if path.exists() else "absent")
        if not path.exists():
            return

        document = json.loads(path.read_text())
        yield ("it records that nothing was trained", document.get("trained") == "nothing",
               str(document.get("trained")))
        yield ("it records that no hypothesis is tested",
               document.get("tests_no_hypothesis") is True,
               str(document.get("tests_no_hypothesis")))
        yield ("the environment travels with the run",
               bool(document.get("environment", {}).get("tensorflow")),
               str(document.get("environment", {}).get("tensorflow")))
        # The shadow root is a temporary directory and not a git repository, so the sha
        # comes back empty here. What is checkable is that the field was written at all.
        yield ("the git sha travels with the run", "git_sha" in document,
               f"recorded as {document.get('git_sha')!r}; empty under a fixture, which is "
               "not a repository")

        checks = document.get("assertions", [])
        failed = [c for c in checks if c.get("outcome") != "pass"]
        yield ("every assertion outcome is recorded and every one passed",
               bool(checks) and not failed, f"{len(checks)} checks, {len(failed)} failures")
        named = {c["check"] for c in checks}
        wanted = [
            "the forest's saved labels are the test records, element for element",
            "the sequence model's saved labels are the test windows, element for element",
            "every window maps to exactly fifty records",
            "the fifty records' labels all equal the window's label",
        ]
        absent = [name for name in wanted if name not in named]
        yield ("the four alignment checks the design names are among them", not absent,
               "all four recorded" if not absent else f"missing {absent}")

        vote = document.get("vote", {})
        yield ("the tie rule is stated in the file", "argmax" in str(vote.get("tie_rule", "")),
               str(vote.get("tie_rule"))[:70])
        yield ("the tie count was reported", "windows_decided_by_a_tie" in vote,
               str(vote.get("windows_decided_by_a_tie")))
        histogram = vote.get("votes_for_the_winner_histogram", {})
        yield ("the distribution of vote purity was reported",
               bool(histogram) and bool(vote.get("purity_bands")),
               f"{len(histogram)} distinct winning vote counts, "
               f"{len(vote.get('purity_bands', []))} bands")
        # A fixture window is forced to an exact 25/25 split, so the tie branch has to
        # have been taken. A tie count of zero here means the branch was never exercised.
        yield ("the fixture's constructed tie was found",
               int(vote.get("windows_decided_by_a_tie", 0)) >= 1,
               f"{vote.get('windows_decided_by_a_tie')} tied windows")

        per_class = document.get("per_class", [])
        yield ("all 19 classes are in the per-class table", len(per_class) == 19,
               f"{len(per_class)} rows")
        columns = {"class", "test_windows", "f1_forest_at_window", "f1_sequence",
                   "f1_oracle_union", "f1_combined"}
        first = per_class[0] if per_class else {}
        yield ("each row carries all four F1 columns and the support",
               columns <= set(first),
               f"missing {sorted(columns - set(first))}" if first else "no rows")

        macro = document.get("macro_f1", {})
        yield ("macro-F1 is recorded for all four",
               {"forest_at_window", "sequence", "oracle_union", "combined"} <= set(macro),
               str({k: round(v, 4) for k, v in macro.items()}))
        yield ("the oracle union is at least as good as either model, being a ceiling",
               macro.get("oracle_union", 0) >= max(macro.get("forest_at_window", 0),
                                                   macro.get("sequence", 0)) - 1e-9,
               f"oracle {macro.get('oracle_union', 0):.4f} against forest "
               f"{macro.get('forest_at_window', 0):.4f} and sequence "
               f"{macro.get('sequence', 0):.4f}")

        four_way = document.get("four_way", [])
        totals = document.get("four_way_totals", {})
        yield ("the four-way count covers all 19 classes", len(four_way) == 19,
               f"{len(four_way)} rows")
        yield ("the four boxes account for every window once",
               sum(totals.values()) == document.get("n_test_windows"),
               f"{sum(totals.values()):,} against {document.get('n_test_windows'):,}")

        test = document.get("mcnemar", {})
        yield ("the McNemar implementation that ran is named",
               bool(test.get("implementation")), str(test.get("implementation")))
        yield ("whether a correction was applied is recorded",
               bool(test.get("correction")), str(test.get("correction"))[:70])
        yield ("it is one test at alpha 0.05 with no correction family",
               test.get("alpha") == 0.05 and "one test" in str(test.get("family", "")),
               f"alpha {test.get('alpha')}, family: {test.get('family')}")
        yield ("the test used the discordant windows the four-way count found",
               test.get("discordant") == totals.get("forest only", -1)
               + totals.get("sequence only", -2),
               f"{test.get('discordant')} against "
               f"{totals.get('forest only')} + {totals.get('sequence only')}")

        oracle = document.get("oracle_union", {})
        yield ("the oracle union is labelled as a ceiling",
               "ceiling" in str(oracle.get("status", "")), str(oracle.get("status"))[:70])

        combiner = document.get("combiner", {})
        yield ("the combiner is the five Amendment 6 classes and nothing else",
               combiner.get("classes") == COMBINER_CLASSES, str(combiner.get("classes")))
        yield ("the rule is recorded as chosen from the published run, not from this notebook",
               "Amendment 6" in str(combiner.get("chosen_from", "")),
               str(combiner.get("chosen_from"))[:70])
        yield ("it is labelled post-hoc and carrying no hypothesis",
               "post-hoc" in str(combiner.get("status", ""))
               and "no hypothesis rests on it" in str(combiner.get("status", "")),
               str(combiner.get("status"))[:70])
        yield ("the combiner actually routed windows to the sequence model, so both branches ran",
               int(combiner.get("windows_routed_to_the_sequence_model", 0)) >= 1,
               f"{combiner.get('windows_routed_to_the_sequence_model')} windows routed")
        dos_icmp = combiner.get("dos_icmp", {})
        yield ("DoS-ICMP under the combiner is reported on its own",
               {"f1_combined", "f1_forest_at_window", "f1_sequence"} <= set(dos_icmp),
               str({k: round(v, 4) if isinstance(v, float) else v
                    for k, v in dos_icmp.items()}))

        alignment = document.get("alignment", {})
        yield ("the mapping is recorded as arithmetic rather than a search",
               "no search" in str(alignment.get("method", "")),
               str(alignment.get("method"))[:70])
        yield ("the overlap at stride 25 is counted rather than assumed away",
               "records_serving_more_than_one_window" in alignment,
               f"{alignment.get('records_serving_more_than_one_window')} records serve "
               f"more than one window, at most "
               f"{alignment.get('most_windows_one_record_serves')} each")

        for name in ("MEMBERS", "VOTES", "FOREST_WINDOW", "SIDE_BY_SIDE", "FOUR_WAY",
                     "MCNEMAR", "ORACLE", "COMBINED"):
            yield (f"{name} exists in the namespace after the run", name in namespace,
                   "yes" if name in namespace else "the cell that defines it did not run")


PROFILE = NB06e(
    name="nb06e",
    notebook="AG_PRAXIS_NB06e_forest_window_comparison.ipynb",
    fast_env="0",
    training_path="none",
    max_traces=0,
    scale_asserts=[
        'assert len(WINDOW_Y) == 49159, "the test partition is not the 49,159 windows '
        'this compares on"',
        'assert len(RECORD_Y) == 1229711, "the test records are not the 1,229,711 rows '
        'the forest was scored on"',
    ],
    fixture={"classes": 19, "features": 44, "window": 50, "stride": 25,
             "train_windows_per_capture": 24, "test_windows_per_capture": 12,
             "val_windows_per_capture": 12, "needs_parent_model": False,
             "needs_parent_predictions": False, "needs_forest_parent": True,
             "needs_aligned_records": True},
)
