"""What NB08c needs from a fixture, and what it should have written when it stops.

NB08c fits nothing. It reads the saved test records, inverts the NB04 scaler on the IAT and
Rate columns, measures what unit IAT is in from the product of the two, and converts the
NB08 budgets into elapsed time under two readings of what a record summarises. So what is
worth checking here is that the conversion runs against arrays and tables built the way the
real ones are built, that both branches of the unit rule are exercised, and that the inputs
it will meet in Colab are there and the shape it expects.

Both branches of the unit rule matter enough to run separately. This profile builds records
whose IAT and Rate multiply to one power of ten, so the rule resolves a unit and every
figure comes out in seconds, and its budgets make the two readings disagree about which
group takes longer. `nb08c_unresolved` builds the same notebook against a product spread
over four decades, so the rule resolves nothing and the figures stay in IAT units, and its
budgets make the two readings agree. Neither run alone covers the notebook.

`check_scale` is what a dry run can say about the real data without reading Drive. The
manifest, the NB08 tables and the group definitions are all in the repository, and together
they say that the records are 1,229,711 rows of 44 columns, that IAT and Rate are among
those columns with a mean and a scale each, that the saturation table holds all nineteen
classes on the grid NB08 used, and that the group listings the notebook hard-codes are the
groups the table itself records.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .nb07b import Profile

VOLUMETRIC = ["DDoS-ICMP", "DDoS-SYN", "DDoS-TCP", "DDoS-UDP",
              "DoS-ICMP", "DoS-SYN", "DoS-TCP", "DoS-UDP"]
LOW_RATE = ["Recon-OS_Scan", "Recon-Port_Scan", "Recon-VulScan", "Spoofing",
            "MQTT-Malformed_Data"]
EXCLUDED = ["Recon-Ping_Sweep"]
GRID = {5, 10, 25, 50}


@dataclass
class NB08c(Profile):
    expect_concentrated: bool = True
    expect_agreement: bool = False

    def check_scale(self, repo_root: Path):
        """The real inputs, read off the repository rather than off the fixture."""
        manifest = json.loads(
            (repo_root / "data" / "processed" / "NB04_manifest.json").read_text())
        records = manifest["arrays"]["records_test"]
        kept = [str(name) for name in manifest["columns"]["kept"]]
        scaler = manifest["scaler"]

        yield ("the real test partition is 1,229,711 records",
               int(records["shape"][0]) == 1229711, f"the manifest says {records['shape'][0]:,}")
        yield ("of 44 columns, which is what the notebook asserts",
               int(records["shape"][1]) == 44 and len(kept) == 44,
               f"array {records['shape'][1]}, kept list {len(kept)}")

        named = [name for name in ("IAT", "Rate") if name in kept]
        yield ("IAT and Rate are both in the kept column order", len(named) == 2,
               ", ".join(f"{name} at index {kept.index(name)}" for name in named)
               if named else "neither is there")
        scaled = all(name in scaler["mean"] and float(scaler["scale"].get(name, 0)) > 0
                     for name in ("IAT", "Rate"))
        yield ("the scaler carries a mean and a non-zero scale for each of them", scaled,
               f"IAT mean {scaler['mean'].get('IAT')} scale {scaler['scale'].get('IAT')}, "
               f"Rate mean {scaler['mean'].get('Rate')} scale {scaler['scale'].get('Rate')}")

        tables = repo_root / "data" / "processed" / "NB08" / "tables"
        saturation_csv = tables / "saturation.csv"
        threshold_csv = tables / "records_to_threshold.csv"
        yield ("the NB08 saturation table is in the repository", saturation_csv.exists(),
               str(saturation_csv.relative_to(repo_root)) if saturation_csv.exists()
               else f"nothing at {saturation_csv}")
        # The notebook restricts itself to saturation when this is absent and records that
        # it did. Whether it is present decides which of those two paths Colab will take.
        yield ("the records-to-threshold table is there too, so the run converts both",
               threshold_csv.exists(),
               str(threshold_csv.relative_to(repo_root)) if threshold_csv.exists()
               else "absent; the run would restrict itself to saturation")
        if not saturation_csv.exists():
            return

        with saturation_csv.open() as handle:
            rows = list(csv.DictReader(handle))
        classes = [row["class"] for row in rows]
        yield ("it holds one row for each of the nineteen classes", len(rows) == 19,
               f"{len(rows)} rows")
        yield ("every class in it is one the manifest counted in the test records",
               set(classes) <= set(records["by_class"]),
               f"unmatched {sorted(set(classes) - set(records['by_class']))}")

        budgets = {row["class"]: int(row["saturates_at_k"]) for row in rows}
        off_grid = {name: k for name, k in budgets.items() if k not in GRID}
        yield ("every saturation budget is on the {5, 10, 25, 50} grid", not off_grid,
               "all on the grid" if not off_grid else str(off_grid))

        # The notebook hard-codes both groups from PROJECT_RECORD.md Section 3 rather than
        # reading the table's group column. If the two ever disagreed, the notebook would
        # be comparing a group the project no longer defines that way.
        marked = {row["class"]: row["group"] for row in rows}
        wrong = {name: marked.get(name) for name in VOLUMETRIC if marked.get(name) != "volumetric"}
        yield ("the eight classes the notebook calls volumetric are the ones the table marks",
               not wrong, "all eight agree" if not wrong else str(wrong))
        wrong = {name: marked.get(name) for name in LOW_RATE if marked.get(name) != "low-rate"}
        yield ("the five it calls low-rate are the ones the table marks", not wrong,
               "all five agree" if not wrong else str(wrong))
        yield ("Recon-Ping_Sweep is in the table and is not among the low-rate classes",
               EXCLUDED[0] in marked and marked.get(EXCLUDED[0]) != "low-rate",
               f"marked {marked.get(EXCLUDED[0])!r}; excluded from the median under "
               "PREREGISTRATION.md Amendment 4")

    def check_artefacts(self, shadow: Path, namespace: dict):
        """What the run left behind, read back off disk rather than off the namespace."""
        path = shadow / "artifacts" / "NB08c" / "observation_time.json"
        yield ("observation_time.json was written", path.exists(),
               "present" if path.exists() else "absent")
        if not path.exists():
            return

        document = json.loads(path.read_text())
        yield ("it records that nothing was trained", document.get("trained") == "nothing",
               str(document.get("trained")))
        yield ("it records that no hypothesis is tested",
               document.get("tests_no_hypothesis") is True,
               str(document.get("tests_no_hypothesis")))
        yield ("it records that H2 does not move",
               "H2 is measured on saturation in records" in str(document.get("status", "")),
               str(document.get("status"))[:80])
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

        caveats = document.get("caveats", [])
        wanted = ["timestamp", "provenance", "packet"]
        absent = [word for word in wanted if not any(word in c for c in caveats)]
        yield ("the three caveats are carried in the file", len(caveats) == 3 and not absent,
               f"{len(caveats)} caveats" if not absent else f"none mentions {absent}")

        unit = document.get("unit_evidence", {})
        yield ("the unit rule is recorded, and recorded as fixed before the measurement",
               "90 percent" in str(unit.get("rule", ""))
               and unit.get("rule_fixed_before_the_measurement") is True,
               str(unit.get("rule"))[:80])
        yield ("the sample is the fixed 100,000 rows at seed 42",
               unit.get("sample_seed") == 42 and int(unit.get("sample_rows", 0)) > 0,
               f"{unit.get('sample_rows')} rows at seed {unit.get('sample_seed')}; the "
               "fixture holds fewer than 100,000 records, so the sample is all of them")
        yield ("the quantiles of the product are recorded",
               len(unit.get("quantiles_of_iat_times_rate", {})) == 9,
               f"{len(unit.get('quantiles_of_iat_times_rate', {}))} quantiles")
        yield ("products that are not positive are counted rather than dropped",
               "non_positive_products" in unit, str(unit.get("non_positive_products")))
        yield (f"the fixture's product {'concentrates' if self.expect_concentrated else 'does not concentrate'}, "
               "which is the branch this profile is here to exercise",
               unit.get("concentrates") is self.expect_concentrated,
               f"{unit.get('share_within_half_a_decade')} within half a decade of "
               f"10^{unit.get('median_power_of_ten')}, mark {unit.get('concentration_mark')}")
        if self.expect_concentrated:
            yield ("a unit is implied and the figures are carried in seconds",
                   unit.get("implied_iat_unit") not in (None, "unresolved")
                   and unit.get("time_figures_reported_in") == "seconds"
                   and unit.get("seconds_per_iat_unit"),
                   f"{unit.get('implied_iat_unit')}, "
                   f"{unit.get('seconds_per_iat_unit')} seconds per IAT unit")
        else:
            yield ("no unit is invented and the figures stay in IAT units",
                   unit.get("implied_iat_unit") == "unresolved"
                   and unit.get("time_figures_reported_in") == "IAT units"
                   and unit.get("seconds_per_iat_unit") is None,
                   f"{unit.get('implied_iat_unit')}, reported in "
                   f"{unit.get('time_figures_reported_in')}")

        per_class = document.get("per_class_iat", [])
        columns = {"class", "iat_median_raw", "iat_mean_raw", "test_records"}
        yield ("the per-class table holds all 19 classes with a median, a mean and a count",
               len(per_class) == 19 and columns <= set(per_class[0] if per_class else {}),
               f"{len(per_class)} rows, columns "
               f"{sorted(per_class[0]) if per_class else 'none'}")

        saturation = document.get("saturation", [])
        columns = {"class", "group", "budget_records", "packets_per_record",
                   "median_iat_raw", "reading_a", "reading_b"}
        yield ("both readings are recorded for every class",
               len(saturation) == 19 and columns <= set(saturation[0] if saturation else {}),
               f"{len(saturation)} rows, missing "
               f"{sorted(columns - set(saturation[0])) if saturation else 'no rows'}")
        off = [row["class"] for row in saturation
               if abs(row["reading_b"] - row["reading_a"] * row["packets_per_record"]) > 1e-6
               * max(1.0, abs(row["reading_b"]))]
        yield ("reading B is reading A times the class's packets per record", not off,
               "every class" if not off else f"not for {off}")

        windows = document.get("extraction_windows", {}).get("packets_per_record", {})
        hundreds = sorted(name for name, packets in windows.items() if packets == 100)
        yield ("exactly the eight DDoS and DoS classes carry the 100-packet window",
               hundreds == sorted(VOLUMETRIC), str(hundreds))
        yield ("the MQTT flood classes are in the 10-packet remainder, not matched on name",
               all(windows.get(name) == 10 for name in windows if name.startswith("MQTT")),
               ", ".join(f"{name} {windows[name]}" for name in sorted(windows)
                         if name.startswith("MQTT")))

        exclusion = document.get("exclusion", {})
        yield ("the exclusion applied is Recon-Ping_Sweep from the low-rate median",
               exclusion.get("class") == EXCLUDED
               and "low-rate median" in str(exclusion.get("applied_to", "")),
               f"{exclusion.get('class')} from {exclusion.get('applied_to')}")
        yield ("and what it rests on is named", "Amendment 4" in str(exclusion.get("basis", "")),
               str(exclusion.get("basis"))[:70])
        yield ("the low-rate side of the median is the group without the excluded class",
               document.get("groups", {}).get("low_rate_used_in_the_median")
               == [name for name in LOW_RATE if name not in EXCLUDED],
               str(document.get("groups", {}).get("low_rate_used_in_the_median")))

        comparison = document.get("group_comparison", {})
        both = all(key in comparison for key in ("reading_a", "reading_b"))
        yield ("a group comparison is recorded for each reading separately", both,
               "both present" if both else f"only {sorted(comparison)}")
        directions = {key: comparison.get(key, {}).get("direction") for key in
                      ("reading_a", "reading_b")}
        yield ("each carries the direction of the difference in words",
               all(directions.values()), str(directions))
        yield (f"the fixture's budgets make the two readings "
               f"{'agree' if self.expect_agreement else 'disagree'}, which is the branch "
               "this profile is here to exercise",
               comparison.get("readings_agree") is self.expect_agreement,
               str(comparison.get("statement"))[:100])
        yield ("no ratio and no pass mark are computed",
               "H2 states neither" in str(comparison.get("no_ratio", ""))
               and not any("ratio" in str(v) or "pass" in str(v)
                           for key in ("reading_a", "reading_b")
                           for v in comparison.get(key, {})),
               str(comparison.get("no_ratio")))

        threshold = document.get("records_to_threshold")
        yield ("the records-to-threshold table was converted, and its censoring carried",
               threshold is not None and any(row.get("censored") for row in threshold)
               and document.get("restricted_to_saturation") is False,
               f"{len(threshold)} rows, "
               f"{sum(1 for row in threshold if row.get('censored'))} of them lower bounds"
               if threshold else "not converted")
        yield ("why no group median is computed on it is recorded",
               "not determinate" in str(document.get("records_to_threshold_note", "")),
               str(document.get("records_to_threshold_note"))[:100])

        for name in ("IAT", "RATE", "PRODUCT", "UNIT_EVIDENCE", "PER_CLASS",
                     "SATURATION_TIME", "COMPARISON", "AGREEMENT", "ledger"):
            yield (f"{name} exists in the namespace after the run", name in namespace,
                   "yes" if name in namespace else "the cell that defines it did not run")


FIXTURE = {"classes": 19, "features": 44, "window": 50, "stride": 25,
           "train_windows_per_capture": 24, "test_windows_per_capture": 12,
           "val_windows_per_capture": 12, "needs_parent_model": False,
           "needs_parent_predictions": False, "needs_records": True,
           "needs_nb08_tables": True}

PROFILE = NB08c(
    name="nb08c",
    notebook="AG_PRAXIS_NB08c_observation_time.ipynb",
    fast_env="0",
    training_path="none",
    max_traces=0,
    scale_asserts=[],
    # A budget of 25 records for the low-rate classes against 5 for the volumetric ones puts
    # the low-rate group ahead on reading A, and the hundredfold extraction window puts the
    # volumetric group ahead on reading B. The two readings disagree, which is the case the
    # notebook has to state rather than resolve.
    fixture={**FIXTURE, "iat_evidence": "concentrated",
             "saturation_budgets": {"volumetric": 5, "low_rate": 25, "neither": 10}},
    expect_concentrated=True,
    expect_agreement=False,
)
