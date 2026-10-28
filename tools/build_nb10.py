"""Builds the results consolidation notebook. Standard library only.

Run once from anywhere:  python tools/build_nb10.py
It writes notebooks/AG_PRAXIS_NB10_results_consolidation.ipynb."""

import json, os

C = []
def md(s):
    C.append({"cell_type": "markdown", "metadata": {},
              "source": s.strip("\n").splitlines(keepends=True)})
def code(s):
    C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md("""
# Results consolidation

This notebook builds the tables and figures the results chapter is written from. It
computes nothing. Every number it writes has already been produced by an earlier run and
is sitting in this repository; the job here is to read those files, arrange them into
tables a reader can follow, draw three figures, and check every value against what the
project record and the results ledger already say.

That last part matters more than it sounds. Figures get quoted in prose, then the prose
gets edited, and the two drift apart. The check at the end reads each value this notebook
is about to hand to the chapter and compares it against the recorded figure. If any of
them disagree the notebook stops and says which one.

It needs no GPU and no accelerator of any kind. It reads JSON, CSV and small arrays, and
it runs on a laptop in under a minute. There is no model here, no training, no prediction,
and nothing that touches the network.

Filenames are descriptive. Numbering figures and tables happens when the chapter is
written, not here, because the numbering depends on the order things end up in the prose.
""")

code(r'''
# Setup. Paths, output locations, and the small helpers everything else uses.

import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Find the repository root by walking up until the project record is beside us.
def find_root(start: Path) -> Path:
    p = start.resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "PROJECT_RECORD.md").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find PROJECT_RECORD.md above " + str(p) +
        ". Run this notebook from inside the repository."
    )

ROOT = find_root(Path.cwd())
PROC = ROOT / "data" / "processed"
RES  = ROOT / "results"
CFG  = ROOT / "config"

OUT        = RES / "chapter4"
OUT_TABLES = OUT / "tables"
OUT_FIGS   = OUT / "figures"
for d in (OUT, OUT_TABLES, OUT_FIGS):
    d.mkdir(parents=True, exist_ok=True)

# Figures are drawn the same way every time so two runs produce the same file.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.hashsalt": "ag_praxis",
})

CAPTIONS = {}   # filename -> caption text
MANIFEST = []   # one record per emitted file, with the artifacts behind it
EMITTED  = {}   # named values this notebook hands to the chapter, for the check at the end

def render_cell(v):
    """One value as the string the table will show, decided here rather than by a library.

    Shortest round-trip decimal, never scientific notation, empty for a missing value.
    """
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, (float, np.floating)):
        return np.format_float_positional(float(v), trim="-")
    return str(v)


def emit_table(df, name, caption, sources, note=None):
    """Write a table as CSV and as Markdown, and record where it came from.

    The Markdown is rendered explicitly rather than left to library defaults. tabulate
    infers alignment and number formatting from the values handed to it, and that
    inference differs between versions, so the same frame rendered on two machines gave
    two files with identical values and different columns. Formatting every cell here and
    turning the inference off makes the output a function of the data alone.
    """
    csv_path = OUT_TABLES / (name + ".csv")
    md_path  = OUT_TABLES / (name + ".md")
    df.to_csv(csv_path, index=False)
    shown = df.apply(lambda col: col.map(render_cell))
    align = ["right" if pd.api.types.is_numeric_dtype(df[c]) else "left" for c in df.columns]
    lines = [shown.to_markdown(index=False, colalign=align, disable_numparse=True)]
    if note:
        lines += ["", note]
    md_path.write_text("\n".join(lines) + "\n")
    CAPTIONS[name] = caption
    MANIFEST.append({"file": name, "kind": "table", "rows": int(len(df)),
                     "caption": caption, "note": note, "sources": sources})
    print("table  ", name, " rows:", len(df))
    return df

def emit_figure(fig, name, caption, sources, note=None):
    """Write a figure as PNG and PDF, and record where it came from.

    The metadata arguments suppress the fields the writers stamp from the environment
    rather than from the figure: the PNG's software string, and the PDF's creator,
    producer and creation date. Without the last of those the three PDFs changed on every
    re-run with no content change, which is churn the consistency check cannot tell from a
    real difference.
    """
    png = OUT_FIGS / (name + ".png")
    pdf = OUT_FIGS / (name + ".pdf")
    fig.savefig(png, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(pdf, bbox_inches="tight",
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    CAPTIONS[name] = caption
    MANIFEST.append({"file": name, "kind": "figure", "rows": None,
                     "caption": caption, "note": note, "sources": sources})
    print("figure ", name)

def record(key, value):
    """Register a value the chapter will quote, so the check at the end can test it."""
    EMITTED[key] = value
    return value

print("repository:", ROOT)
print("writing to:", OUT)
''')

md("""
## The files this reads

Everything below is already in the repository. The cell lists each file, checks it is
there, and stops if anything is missing. A missing file found here is a much cheaper
problem than one found six cells later.
""")

code(r'''
# Every input, checked before anything is read.

INPUTS = {
    # single-record baselines
    "published_cnn":      RES / "NB05" / "published_cnn_19class" / "metrics.json",
    "ours_cnn":           RES / "NB05" / "ours_cnn_19class" / "metrics.json",
    "forest":             RES / "NB05" / "forest_19class" / "metrics.json",
    "forest_leaf1":       RES / "NB05" / "forest_19class_dadkhah_leaf1" / "metrics.json",
    # sequence model, seed 42, the full 50-record window
    "sequence":           PROC / "NB06" / "metrics.json",
    # class balancing
    "balance_weighted":   RES / "NB07" / "class_weighted_loss" / "metrics.json",
    "balance_focal":      RES / "NB07" / "focal_loss" / "metrics.json",
    "balance_logit":      RES / "NB07" / "logit_adjustment" / "metrics.json",
    "balance_threshold":  RES / "NB07" / "threshold_tuning" / "metrics.json",
    "balance_resampling": RES / "NB07" / "window_resampling" / "metrics.json",
    # observation budgets and seeds
    "budget_05":          PROC / "NB08" / "sequence_budget_05" / "metrics.json",
    "budget_10":          PROC / "NB08" / "sequence_budget_10" / "metrics.json",
    "budget_25":          PROC / "NB08" / "sequence_budget_25" / "metrics.json",
    "seed_43":            PROC / "NB08" / "sequence_seed_43" / "metrics.json",
    "seed_44":            PROC / "NB08" / "sequence_seed_44" / "metrics.json",
    "seed_45":            PROC / "NB08" / "sequence_seed_45" / "metrics.json",
    "seed_46":            PROC / "NB08" / "sequence_seed_46" / "metrics.json",
    "saturation":         PROC / "NB08" / "tables" / "saturation.csv",
    "records_threshold":  PROC / "NB08" / "tables" / "records_to_threshold.csv",
    "mcnemar":            PROC / "NB08" / "tables" / "mcnemar.csv",
    # separability
    "pairwise_auc":       PROC / "pairwise_auc.csv",
    # provenance
    "provenance":         PROC / "NB03_verdict.json",
    "duration_ablation":  PROC / "NB03b" / "duration_ablation.json",
    # timing-excluded models and the threat mapping
    "attributions":       PROC / "NB09a" / "attributions.json",
    "attr_seed_42":       PROC / "NB09a" / "attributions_seed_42.npz",
    "attr_forest":        PROC / "NB09a" / "attributions_forest.npz",
    "threat_mapping":     PROC / "NB09b" / "threat_mapping.json",
    "capec_map":          CFG / "shap_capec_map.yaml",
    "capec_stride":       CFG / "capec_stride.yaml",
}

missing = {k: str(v) for k, v in INPUTS.items() if not v.exists()}
if missing:
    for k, v in missing.items():
        print("MISSING", k, "->", v)
    raise FileNotFoundError(str(len(missing)) + " input(s) missing. Nothing was written.")

def load_json(key):
    return json.loads(INPUTS[key].read_text())

def load_csv(key):
    return pd.read_csv(INPUTS[key])

print("all", len(INPUTS), "inputs present")
''')

code(r'''
# Loaders. Every run's metrics file has the same shape, so one reader covers all of them.

def per_class_f1(key) -> pd.Series:
    m = load_json(key)
    return pd.Series(m["per_class_f1"], name=key)

def support(key) -> pd.Series:
    m = load_json(key)
    return pd.Series(m["support"], name=key)

def headline(key) -> dict:
    m = load_json(key)
    return {
        "accuracy": m["accuracy"],
        "weighted_f1": m["weighted_f1"],
        "macro_f1": m["macro_f1"],
        "n_test": m["n_test"],
    }

# Classes, in the order every metrics file uses.
CLASSES = load_json("published_cnn")["labels"]

# Fixed by the benchmark taxonomy, not derived from any measurement here.
VOLUMETRIC = ["DDoS-ICMP", "DDoS-SYN", "DDoS-TCP", "DDoS-UDP",
              "DoS-ICMP", "DoS-SYN", "DoS-TCP", "DoS-UDP"]
LOW_RATE   = ["Recon-OS_Scan", "Recon-Port_Scan", "Recon-VulScan",
              "Spoofing", "MQTT-Malformed_Data"]

# The five classes the published single-record model scores below the detection floor.
HYPOTHESIS_CLASSES = ["Recon-VulScan", "Recon-OS_Scan", "MQTT-DDoS-Publish_Flood",
                      "Spoofing", "MQTT-Malformed_Data"]

# Classes resting on too few test sequences for a per-class figure to be interpretable.
THIN = {"Recon-Ping_Sweep": 4, "Recon-VulScan": 18, "MQTT-Malformed_Data": 40}

FLOOR = 0.50            # the detection floor the recovered-and-lost claim is measured on
RELIABLE = 0.80         # the level a class is called reliably detected at

def thin_mark(c):
    return "yes" if c in THIN else ""

print(len(CLASSES), "classes;", len(VOLUMETRIC), "volumetric,", len(LOW_RATE), "low-rate")
''')

md("""
## Detection: single records against sequences

The first objective asks whether reading a run of records rather than one at a time
improves the classes a single-record model handles worst. Four models are being compared
and they are not scored on the same items, so the first table carries the unit each was
measured on alongside its score. A window of fifty records and a single record are not the
same thing, and the two splits hold out different rows.
""")

code(r'''
# Headline comparison across the four models.

rows = [
    ("Published convolutional model, single records, shipped split", "records", "published_cnn"),
    ("Published convolutional model, single records, capture-disjoint split", "records", "ours_cnn"),
    ("Random forest, single records, capture-disjoint split", "records", "forest"),
    ("Convolutional encoder with a recurrent layer, 50-record windows, capture-disjoint split", "windows", "sequence"),
]

df = pd.DataFrame([{
    "model": label,
    "test unit": unit,
    "test items": headline(key)["n_test"],
    "accuracy": round(headline(key)["accuracy"], 4),
    "weighted F1": round(headline(key)["weighted_f1"], 4),
    "macro F1": round(headline(key)["macro_f1"], 4),
    "weighted minus macro": round(headline(key)["weighted_f1"] - headline(key)["macro_f1"], 4),
} for label, unit, key in rows])

record("macro_published", headline("published_cnn")["macro_f1"])
record("macro_ours", headline("ours_cnn")["macro_f1"])
record("macro_forest", headline("forest")["macro_f1"])
record("macro_sequence", headline("sequence")["macro_f1"])

emit_table(
    df,
    "model_comparison_nineteen_class",
    "Nineteen-class results for four models. The rows are scored on different items and "
    "different held-out partitions, so they are four models' own scores rather than a "
    "paired comparison.",
    sources=[str(INPUTS[k].relative_to(ROOT)) for _, _, k in rows],
    note="The gap between weighted and macro F1 is the aggregate that a class-imbalanced "
         "task hides behind: accuracy and weighted F1 read high while the smallest classes "
         "are undetected.",
)
''')

code(r'''
# The five classes the published model scores below the detection floor.

f1 = pd.DataFrame({
    "published": per_class_f1("published_cnn"),
    "single-record convolution": per_class_f1("ours_cnn"),
    "single-record forest": per_class_f1("forest"),
    "sequence": per_class_f1("sequence"),
})
seq_support = support("sequence")

df = f1.loc[HYPOTHESIS_CLASSES].round(4).reset_index().rename(columns={"index": "class"})
df["test sequences"] = [int(seq_support[c]) for c in HYPOTHESIS_CLASSES]
df["thin"] = [thin_mark(c) for c in HYPOTHESIS_CLASSES]
df["reaches the floor under the sequence model"] = [
    "yes" if f1.loc[c, "sequence"] >= FLOOR else "no" for c in HYPOTHESIS_CLASSES]

n_recovered = int((f1.loc[HYPOTHESIS_CLASSES, "sequence"] >= FLOOR).sum())
record("n_recovered", n_recovered)

emit_table(
    df,
    "hypothesis_classes_by_model",
    "The five classes the published single-record model scores below F1 0.50, under each "
    "of the four models. Four of the five reach the floor when records are read as "
    "sequences.",
    sources=["results/NB05", "data/processed/NB06/metrics.json"],
    note="Two of the four recovered classes rest on very few test sequences — 18 and 40 — "
         "so the direction is more interpretable than the magnitude.",
)
''')

code(r'''
# All nineteen classes, the published model against the sequence model.

d = pd.DataFrame({
    "published": f1["published"],
    "sequence": f1["sequence"],
})
d["difference"] = d["sequence"] - d["published"]
d = d.sort_values("difference")

df = d.round(4).reset_index().rename(columns={"index": "class"})
df["test sequences"] = [int(seq_support[c]) for c in d.index]
df["group"] = ["volumetric" if c in VOLUMETRIC else "low-rate" if c in LOW_RATE else ""
               for c in d.index]
df["thin"] = [thin_mark(c) for c in d.index]

gains  = d.loc[d["difference"] > 0, "difference"].sum()
losses = d.loc[d["difference"] < 0, "difference"].sum()
crossed_up   = [c for c in d.index if f1.loc[c, "published"] < FLOOR <= f1.loc[c, "sequence"]]
crossed_down = [c for c in d.index if f1.loc[c, "sequence"] < FLOOR <= f1.loc[c, "published"]]
five_worst   = d["difference"].nsmallest(5)
vol_share    = five_worst.sum() / losses

record("gains_sum", gains)
record("losses_sum", losses)
record("net_sum", gains + losses)
record("n_crossed_up", len(crossed_up))
record("n_crossed_down", len(crossed_down))
record("five_worst_sum", five_worst.sum())
record("volumetric_loss_share", vol_share)

emit_table(
    df,
    "per_class_f1_single_record_against_sequence",
    "Per-class F1 for all nineteen classes, the published single-record model against the "
    "model reading fifty-record windows, sorted by the difference between them.",
    sources=["results/NB05/published_cnn_19class/metrics.json",
             "data/processed/NB06/metrics.json"],
    note=("Nine classes gain and ten lose. The gains sum to " + format(gains, "+.4f") +
          " and the losses to " + format(losses, "+.4f") + ", a net of " +
          format(gains + losses, "+.4f") + " across nineteen classes. " +
          str(len(crossed_up)) + " classes cross above F1 0.50 and " +
          str(len(crossed_down)) + " crosses below it. The five largest losses are all "
          "volumetric classes and account for " + format(vol_share, ".0%") +
          " of the total negative movement."),
)

print("crossed up:", crossed_up)
print("crossed down:", crossed_down)
''')

code(r'''
# Figure: where each class sits against the detection floor, before and after.

order = f1["sequence"].sort_values().index.tolist()
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(7.0, 6.4))
for i, c in enumerate(order):
    a, b = f1.loc[c, "published"], f1.loc[c, "sequence"]
    ax.plot([a, b], [i, i], color="0.6", linewidth=1.0, zorder=1)
    ax.scatter([a], [i], s=26, facecolors="white", edgecolors="0.25",
               linewidths=1.0, zorder=2)
    ax.scatter([b], [i], s=26, color="0.15", zorder=3)

ax.axvline(FLOOR, color="0.1", linestyle="--", linewidth=1.0, zorder=0)
ax.text(FLOOR, len(order) - 0.2, " detection floor", fontsize=8, va="top", color="0.1")

ax.set_yticks(y)
ax.set_yticklabels([c + (" *" if c in THIN else "") for c in order])
ax.set_xlim(-0.03, 1.03)
ax.set_xlabel("F1")
ax.set_ylabel("")
ax.grid(axis="y", visible=False)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
           markeredgecolor="0.25", label="single records"),
    Line2D([], [], marker="o", linestyle="none", color="0.15",
           label="fifty-record windows"),
], loc="lower right", frameon=False, fontsize=8)

emit_figure(
    fig,
    "detection_floor_crossings_by_class",
    "Per-class F1 under the published single-record model and under the model reading "
    "fifty-record windows, with the F1 0.50 detection floor marked. Classes marked with "
    "an asterisk rest on too few test sequences for the magnitude to be interpreted.",
    sources=["results/NB05/published_cnn_19class/metrics.json",
             "data/processed/NB06/metrics.json"],
)
''')

code(r'''
# Class balancing: five interventions, one at a time, against the untreated model.

interventions = [
    ("none (untreated)",            "sequence"),
    ("class-weighted loss",         "balance_weighted"),
    ("focal loss",                  "balance_focal"),
    ("logit adjustment",            "balance_logit"),
    ("threshold tuning",            "balance_threshold"),
    ("window resampling",           "balance_resampling"),
]

rows = []
for label, key in interventions:
    p = per_class_f1(key)
    row = {"intervention": label}
    for c in HYPOTHESIS_CLASSES:
        row[c] = round(float(p[c]), 4)
    row["volumetric mean"] = round(float(p[VOLUMETRIC].mean()), 4)
    row["macro F1"] = round(headline(key)["macro_f1"], 4)
    row["classes at or above the floor, of five"] = int((p[HYPOTHESIS_CLASSES] >= FLOOR).sum())
    rows.append(row)

df = pd.DataFrame(rows)
record("balancing_best_macro", df["macro F1"].max())
record("balancing_floor_counts", df["classes at or above the floor, of five"].tolist())

emit_table(
    df,
    "class_balancing_interventions",
    "Five treatments of class imbalance applied one at a time to the sequence model, on "
    "the five classes the published model fails, with the volumetric mean and macro F1 "
    "alongside. All rows are scored on the same test windows.",
    sources=["results/NB07", "data/processed/NB06/metrics.json"],
    note="No treatment raises the count of the five reaching the floor. One class does not "
         "respond to any of them, moving across a range narrower than a tenth of the "
         "floor, which points at how separable its traffic is rather than at how rare it is.",
)
''')

code(r'''
# Cross-seed variation, and the one class whose regression the seeds were run to test.

seed_keys = {42: "sequence", 43: "seed_43", 44: "seed_44", 45: "seed_45", 46: "seed_46"}
rows = []
for s, key in seed_keys.items():
    rows.append({
        "seed": s,
        "macro F1": round(headline(key)["macro_f1"], 4),
        "DoS-ICMP F1": round(float(per_class_f1(key)["DoS-ICMP"]), 4),
    })
df = pd.DataFrame(rows)

vals = np.array([headline(k)["macro_f1"] for k in seed_keys.values()])
mean, sd1, sd0 = vals.mean(), vals.std(ddof=1), vals.std(ddof=0)
record("seed_mean", mean)
record("seed_sd_ddof1", sd1)
record("seed_sd_ddof0", sd0)

df.loc[len(df)] = {"seed": "mean", "macro F1": round(mean, 4), "DoS-ICMP F1": ""}
df.loc[len(df)] = {"seed": "standard deviation", "macro F1": round(sd1, 4), "DoS-ICMP F1": ""}

emit_table(
    df,
    "cross_seed_macro_f1",
    "Macro F1 across five training seeds for the sequence model at the full fifty-record "
    "window, with the one class that falls below the detection floor shown alongside.",
    sources=["data/processed/NB06/metrics.json", "data/processed/NB08"],
    note="The class stays below the floor at every seed, so its fall is a property of "
         "window-based sequencing rather than of the seed the first run used. What causes "
         "it is not established here.",
)
print("mean", round(mean, 4), "sd(ddof=1)", round(sd1, 4), "sd(ddof=0)", round(sd0, 4))
''')

code(r'''
# Paired significance tests.

mc = load_csv("mcnemar")
df = mc[["family", "a", "b", "accuracy_a", "accuracy_b", "discordant",
         "statistic", "p_value", "holm_p", "significant_at_alpha", "favours"]].copy()
df.columns = ["family", "first", "second", "accuracy of the first", "accuracy of the second",
              "items the two disagree on", "statistic", "p", "p after correction",
              "significant at 0.05", "favours"]
for c in ["accuracy of the first", "accuracy of the second"]:
    df[c] = df[c].round(4)
for c in ["statistic"]:
    df[c] = df[c].round(2)
for c in ["p", "p after correction"]:
    df[c] = df[c].map(lambda v: format(v, ".3g"))

n_budget = int(((mc["family"] == "budget against budget") & mc["significant_at_alpha"]).sum())
n_interv = int(((mc["family"] == "parent against intervention") & mc["significant_at_alpha"]).sum())
record("mcnemar_budget_significant", n_budget)
record("mcnemar_intervention_significant", n_interv)

emit_table(
    df,
    "paired_significance_tests",
    "Paired tests between observation budgets and between the untreated model and each "
    "balancing treatment, with a correction applied within each family and again across "
    "all eleven.",
    sources=["data/processed/NB08/tables/mcnemar.csv"],
    note="No test is computed against the published single-record model. It is scored on "
         "different units and a different held-out partition, so no item-level pairing "
         "between the two exists and no substitute was computed.",
)
''')

code(r'''
# Separability: the class pairs one feature cannot tell apart.

auc = load_csv("pairwise_auc")
below = auc[auc["best_auc"] < 0.90].sort_values("best_auc")
df = below[["class_a", "class_b", "same_family", "best_feature", "best_auc",
            "runner_up_feature", "runner_up_auc"]].copy()
df.columns = ["first class", "second class", "same attack family", "best single feature",
              "best AUC", "next best feature", "next best AUC"]
df["best AUC"] = df["best AUC"].round(4)
df["next best AUC"] = df["next best AUC"].round(4)
df["same attack family"] = df["same attack family"].map({True: "yes", False: ""})

record("n_pairs_below_090", int(len(below)))
record("n_pairs_total", int(len(auc)))

emit_table(
    df,
    "class_pairs_below_auc_090",
    "The class pairs no single feature separates at AUC 0.90, out of all "
    + str(len(auc)) + " pairs.",
    sources=["data/processed/pairwise_auc.csv"],
    note="These are the pairs where the difficulty is in the traffic rather than in the "
         "class sizes, which is the reading the balancing results point to for the one "
         "class that does not respond to any treatment.",
)
''')

md("""
## Observation: how much traffic each class needs

The second objective asks how much has to be observed before a class stops improving, and
whether that differs between the two groups. Saturation is the smallest budget within a
fixed tolerance of a class's own best score across the grid, so it marks where a class
stops improving and not where it becomes reliably detectable. The grid takes four values,
which limits how finely any comparison between the two group medians can be read.
""")

code(r'''
# Saturation per class, and the two group medians.

sat = load_csv("saturation")
df = sat.copy()
df["group"] = df["group"].replace({"neither": ""})
df = df[["class", "group", "saturates_at_k", "f1_there", "best_f1", "best_at_k", "thin"]]
df.columns = ["class", "group", "saturates at", "F1 there", "best F1", "best at", "thin"]
df["F1 there"] = df["F1 there"].round(4)
df["best F1"] = df["best F1"].round(4)
df["thin"] = df["thin"].map({True: "yes", False: ""})
df["below the reliable level"] = np.where(df["F1 there"] < RELIABLE, "yes", "")
df = df.sort_values(["group", "saturates at", "class"])

med_low = float(sat.loc[sat["group"] == "low-rate", "saturates_at_k"].median())
med_vol = float(sat.loc[sat["group"] == "volumetric", "saturates_at_k"].median())
n_below = int((sat["f1_there"] < RELIABLE).sum())
n_at_five = int((sat["saturates_at_k"] == 5).sum())

record("median_low_rate", med_low)
record("median_volumetric", med_vol)
record("n_saturating_below_reliable", n_below)
record("n_saturating_at_five", n_at_five)

emit_table(
    df,
    "saturation_budget_by_class",
    "The smallest number of observed records at which each class comes within 0.02 of its "
    "own best score across the budget grid, with the score it reaches there.",
    sources=["data/processed/NB08/tables/saturation.csv"],
    note=("Low-rate median " + format(med_low, ".0f") + " records against volumetric " +
          format(med_vol, ".0f") + ". " + str(n_at_five) + " of nineteen classes stop "
          "improving at the smallest budget, and " + str(n_below) + " saturate at an F1 "
          "below 0.80, so saturation marks where a class stops improving rather than where "
          "it becomes reliably detectable. The grid takes only four values, so a median can "
          "land on one of them or midway between two, and a finer grid would move the "
          "ratio between the groups."),
)
''')

code(r'''
# Records to threshold, kept as a reported result alongside saturation.

rt = load_csv("records_threshold")
df = rt.copy()
df["group"] = df["group"].replace({"neither": ""})
df = df[["class", "group", "reported_as", "best_f1", "f1_at_50", "reaches_then_falls", "thin"]]
df.columns = ["class", "group", "records to reach F1 0.80", "best F1", "F1 at fifty records",
              "reaches the level then falls", "thin"]
df["best F1"] = df["best F1"].round(4)
df["F1 at fifty records"] = df["F1 at fifty records"].round(4)
for c in ["reaches the level then falls", "thin"]:
    df[c] = df[c].map({True: "yes", False: ""})
df = df.sort_values(["group", "class"])

n_censored = int((rt["reported_as"].astype(str) == "> 50").sum())
record("n_censored", n_censored)

emit_table(
    df,
    "records_to_reach_the_reliable_level",
    "How many observed records each class needs to reach F1 0.80, with classes that do not "
    "reach it within fifty records reported as exceeding the grid rather than as fifty.",
    sources=["data/processed/NB08/tables/records_to_threshold.csv"],
    note=(str(n_censored) + " of nineteen classes do not reach the level within fifty "
          "records. Because three of five low-rate classes are among them the low-rate "
          "median is not determinate on this scale, which is why saturation rather than "
          "this measurement carries the group comparison. Two situations sit behind the "
          "same marker: some of these classes are at their best score at fifty records and "
          "others peaked earlier and fell."),
)
''')

code(r'''
# Macro F1 and the count of reliably detected classes at each budget.

budgets = [(5, "budget_05"), (10, "budget_10"), (25, "budget_25"), (50, "sequence")]
rows = []
for k, key in budgets:
    p = per_class_f1(key)
    rows.append({
        "records observed": k,
        "accuracy": round(headline(key)["accuracy"], 4),
        "macro F1": round(headline(key)["macro_f1"], 4),
        "classes at or above F1 0.80": int((p >= RELIABLE).sum()),
    })
df = pd.DataFrame(rows)

record("budget_macros", df["macro F1"].tolist())
record("budget_reliable_counts", df["classes at or above F1 0.80"].tolist())

emit_table(
    df,
    "macro_f1_by_observation_budget",
    "Overall results at each observation budget. Every budget is scored on the same test "
    "windows and uses the same model size, so the only thing varying is how many records "
    "of each window the model sees.",
    sources=["data/processed/NB08", "data/processed/NB06/metrics.json"],
    note="The curve is read as flat from twenty-five records rather than declining: the "
         "spread across seeds at fifty records is wider than the gap between the two.",
)
''')

code(r'''
# Figure: how each class responds to being given more records.

ks = [5, 10, 25, 50]
per_budget = pd.DataFrame({k: per_class_f1(key) for k, key in budgets})
sat_at = load_csv("saturation").set_index("class")["saturates_at_k"].to_dict()

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

def spread(values, gap=0.035):
    """Nudge label positions apart so class names do not sit on top of each other."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out

for ax, (title, group) in zip(axes[:2], [("Low-rate classes", LOW_RATE),
                                         ("Volumetric classes", VOLUMETRIC)]):
    finals = []
    for c in group:
        vals = [per_budget.loc[c, k] for k in ks]
        finals.append(vals[-1])
        ax.plot(ks, vals, marker="o", markersize=3.5, linewidth=1.1,
                color="0.35" if c not in THIN else "0.65",
                linestyle="-" if c not in THIN else "--")
        s = sat_at.get(c)
        if s in ks:
            ax.scatter([s], [per_budget.loc[c, s]], s=34, facecolors="none",
                       edgecolors="0.05", linewidths=1.1, zorder=4)
    for c, y_end, y_label in zip(group, finals, spread(finals)):
        ax.annotate(c + (" *" if c in THIN else ""), xy=(ks[-1], y_end),
                    xytext=(ks[-1] * 1.16, y_label), fontsize=6.5, va="center",
                    color="0.2", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", linewidth=0.4, color="0.7",
                                    shrinkA=1, shrinkB=1))
    ax.axhline(RELIABLE, color="0.1", linestyle="--", linewidth=0.9)
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlim(4.4, 165); ax.set_ylim(-0.03, 1.06)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("records observed")
axes[0].set_ylabel("F1")

ax = axes[2]
macro = [headline(key)["macro_f1"] for _, key in budgets]
ax.plot(ks, macro, marker="o", markersize=4, color="0.15", linewidth=1.3)
ax.errorbar([50], [EMITTED["seed_mean"]], yerr=[EMITTED["seed_sd_ddof1"]],
            fmt="s", markersize=4, color="0.45", capsize=3, linewidth=1.1)
ax.annotate("spread across\nfive seeds", (50, EMITTED["seed_mean"]),
            textcoords="offset points", xytext=(-8, 16), fontsize=6.5,
            ha="right", color="0.3")
ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_xlim(4.4, 70); ax.set_ylim(0.55, 0.82)
ax.set_title("All classes together", fontsize=9)
ax.set_xlabel("records observed"); ax.set_ylabel("macro F1")

fig.tight_layout()
emit_figure(
    fig,
    "f1_by_observation_budget",
    "Per-class F1 across the four observation budgets, separated into the two groups, with "
    "each class's saturation point ringed and the reliable level marked; and overall macro "
    "F1 across the same budgets with the spread across five training seeds at the largest "
    "budget. Dashed lines and asterisks mark classes resting on too few test sequences to "
    "interpret.",
    sources=["data/processed/NB08", "data/processed/NB06/metrics.json",
             "data/processed/NB08/tables/saturation.csv"],
)
''')

md("""
## Explanations, attack patterns and surveillance

The third objective asks whether the features a model relies on can be resolved to a
documented attack pattern, and from there to a threat category, through a fixed mapping
written before any model was trained. Because the features that identify the recording
session are also the strongest timing features, the models explained here are trained
without them, and the first table records what that costs.
""")

code(r'''
# What excluding the four timing columns costs each model.

attr = load_json("attributions")
seq_seed_f1 = {int(s): v["macro_f1"] for s, v in attr["sequence_runs"].items()}
seq40 = np.array(list(seq_seed_f1.values()))

df = pd.DataFrame([
    {"model": "Random forest, single records",
     "with all forty-four features": round(headline("forest")["macro_f1"], 4),
     "with the four timing features removed": round(attr["forest_run"]["macro_f1"], 4),
     "fall": round(headline("forest")["macro_f1"] - attr["forest_run"]["macro_f1"], 4),
     "test unit": "records"},
    {"model": "Convolutional encoder with a recurrent layer, fifty-record windows",
     "with all forty-four features": round(headline("sequence")["macro_f1"], 4),
     "with the four timing features removed": round(float(seq40.mean()), 4),
     "fall": round(headline("sequence")["macro_f1"] - float(seq40.mean()), 4),
     "test unit": "windows"},
])

record("forest_40", attr["forest_run"]["macro_f1"])
record("sequence_40_mean", float(seq40.mean()))
record("sequence_40_sd", float(seq40.std(ddof=1)))

emit_table(
    df,
    "cost_of_excluding_the_timing_features",
    "Macro F1 for both models with and without the four timing features, which are the "
    "features that identify the recording session. The sequence figure is the mean across "
    "five training seeds.",
    sources=["data/processed/NB09a/attributions.json",
             "results/NB05/forest_19class/metrics.json",
             "data/processed/NB06/metrics.json"],
    note="The two rows are scored on different units. Removing the same four columns costs "
         "the forest an order of magnitude more than it costs the sequence model. The "
         "forest is used here only as an exactness check on the attribution method, which "
         "holds whatever the model scores, but a reader told it is the exact comparator "
         "should also be told what it scores.",
)
''')

code(r'''
# The mapping, class by class, for each model.

import yaml

tm = load_json("threat_mapping")
seq_arm = [a for a in tm["h3"] if a["model"].startswith("sequence")][0]
for_arm = [a for a in tm["h3"] if a["model"].startswith("forest")][0]
K_FEATURES = int(tm["k"])

capec_names = yaml.safe_load(INPUTS["capec_stride"].read_text())["mapping"]

def pattern_label(capec_id):
    """The CAPEC id with the pattern's name, from the correspondence file."""
    entry = capec_names.get(int(capec_id)) or capec_names.get(str(capec_id)) or {}
    name = entry.get("name")
    return "CAPEC-" + str(capec_id) + (" · " + name if name else "")

def mapping_table(arm, name, caption, note):
    df = pd.DataFrame([{
        "class": r["class"],
        "five highest-ranked features": r["top_features"],
        "attack pattern": pattern_label(r["capec"]),
        "category resolved": r["assigned"],
        "category documented for the class": r["truth"],
        "agrees": "yes" if r["agrees"] else "",
        "thin": thin_mark(r["class"]),
    } for r in arm["per_class"]])
    return emit_table(df, name, caption,
                      sources=["data/processed/NB09b/threat_mapping.json",
                               "config/capec_stride.yaml"], note=note)

def ranking_table(arm, npz_key, name, caption, note):
    """The ten features the mapping reads for each class, with their weights.

    Recomputed from the attribution matrix, because threat_mapping.json stores only the
    highest five and stores them as a string. The assert below is the reason that is safe:
    the recomputed top five reproduce the stored string for every attack class, so this is
    the same ranking the mapping read, extended to the full ten.
    """
    z = np.load(INPUTS[npz_key], allow_pickle=False)
    A = z["attributions"]
    classes = [str(c) for c in z["classes"]]
    features = [str(f) for f in z["features"]]
    rows = []
    for r in arm["per_class"]:
        i = classes.index(r["class"])
        order = np.argsort(-A[i])[:K_FEATURES]
        assert [features[j] for j in order[:5]] == \
               [s.strip() for s in r["top_features"].split(",")], \
               "the recomputed ranking does not match the stored top five for " + r["class"]
        mass = float(A[i][order].sum())
        for rank, j in enumerate(order, start=1):
            rows.append({
                "class": r["class"],
                "rank": rank,
                "feature": features[j],
                "mean absolute attribution": round(float(A[i][j]), 6),
                "share of the ten": round(float(A[i][j]) / mass, 4) if mass else None,
            })
    return emit_table(pd.DataFrame(rows), name, caption,
                      sources=["data/processed/NB09a/" + INPUTS[npz_key].name,
                               "data/processed/NB09b/threat_mapping.json"], note=note)

mapping_table(
    seq_arm,
    "attack_pattern_and_category_by_class",
    "Each attack class, the five features its explanations rank highest, the attack "
    "pattern the mapping resolves from the ten it reads, the threat category that pattern "
    "carries, and the category documented for that class in the benchmark paper.",
    "The feature column shows the highest five. The mapping reads ten, and all ten are in "
    "`feature_ranking_by_class` with their attribution weights. Every class reaches an "
    "attack pattern. Half reach the category documented for them. One class rests on a "
    "single mapping entry — the one linking the address-resolution protocol to identity "
    "spoofing — and one is the only route to its category and rests on forty test "
    "sequences, so neither should be read as evidence about the category.",
)

ranking_table(
    seq_arm, "attr_seed_42",
    "feature_ranking_by_class",
    "The ten features the mapping reads for each attack class, in rank order, with the "
    "mean absolute attribution behind each and its share of the ten.",
    "The share column is what says how much any assignment rests on one feature. Where "
    "the ten are near-flat, no single feature carries the pattern the class resolves to. "
    "The ranking is recomputed from the attribution matrix rather than read from "
    "threat_mapping.json, which stores only the highest five; the recomputed five "
    "reproduce that string for all eighteen classes.",
)

mapping_table(
    for_arm,
    "attack_pattern_and_category_by_class_exact_attributions",
    "The same mapping applied to the single-record forest, whose attributions are exact "
    "rather than approximate. Used as a check on the attribution method, not as a test of "
    "the mapping.",
    "This model agrees with the documented category less often than the approximate one, "
    "which rules out approximation error as the explanation for the disagreements.",
)

ranking_table(
    for_arm, "attr_forest",
    "feature_ranking_by_class_exact_attributions",
    "The same ranking for the forest, whose attributions are exact rather than "
    "approximate.",
    "Read against `feature_ranking_by_class`, this shows whether the two models rank the "
    "same features or reach the same pattern by different routes.",
)
''')

code(r'''
# The headline mapping result, with the things it has to be read against.

import yaml
cap_map = yaml.safe_load(INPUTS["capec_map"].read_text())

def count_refusals(doc):
    if isinstance(doc, dict):
        if "no_entry" in doc:
            v = doc["no_entry"]
            return len(v) if hasattr(v, "__len__") else None
        for v in doc.values():
            if isinstance(v, dict) and "no_entry" in v:
                w = v["no_entry"]
                return len(w) if hasattr(w, "__len__") else None
    return None

n_refused = count_refusals(cap_map)
if n_refused is None:
    raise KeyError("Could not find the refused-feature list in the mapping rules. "
                   "Top-level keys are: " + str(list(cap_map)))
n_features = len(attr["features"])

PASS_MARK = 15          # 80% of eighteen, as the project record states it
pass_mark_in_artifact = tm.get("pass_mark")

df = pd.DataFrame([
    {"measure": "Attack classes reaching an attack pattern",
     "value": str(seq_arm["n_assigned"]) + " of " + str(tm["n_attack_classes"]),
     "as a proportion": round(seq_arm["n_assigned"] / tm["n_attack_classes"], 3),
     "compared against": "a pass mark of " + str(PASS_MARK) + " of " + str(tm["n_attack_classes"])},
    {"measure": "Features the mapping declines to map",
     "value": str(n_refused) + " of " + str(n_features),
     "as a proportion": round(n_refused / n_features, 3),
     "compared against": "a class whose highest-ranked features all fell here would reach nothing"},
    {"measure": "Classes whose resolved category matches the documented one",
     "value": str(seq_arm["n_agree"]) + " of " + str(tm["n_attack_classes"]),
     "as a proportion": round(seq_arm["proportion"], 3),
     "compared against": "a majority-class baseline of " + format(tm["majority_class_baseline"], ".3f")},
    {"measure": "The same, for the model with exact attributions",
     "value": str(for_arm["n_agree"]) + " of " + str(tm["n_attack_classes"]),
     "as a proportion": round(for_arm["proportion"], 3),
     "compared against": "the same baseline"},
])

record("n_assigned", seq_arm["n_assigned"])
record("n_agree_sequence", seq_arm["n_agree"])
record("n_agree_forest", for_arm["n_agree"])
record("n_refused_features", n_refused)

emit_table(
    df,
    "attack_pattern_resolution_summary",
    "How far the mapping reaches and how often it reaches the right place. Reaching no "
    "attack pattern and reaching the wrong category are different failures and are counted "
    "separately.",
    sources=["data/processed/NB09b/threat_mapping.json", "config/shap_capec_map.yaml"],
    note=("The mapping declines " + str(n_refused) + " of " + str(n_features) + " features, "
          "so a class whose highest-ranked features all landed among them would have "
          "resolved to nothing and counted against the result. That every class resolved is "
          "therefore a measurement rather than an artifact of a table built to cover "
          "everything. Twelve of the eighteen classes are denial of service, so a rule that "
          "read nothing and answered denial of service every time would match the majority "
          "baseline; the agreement figure has to be read against it."),
)

if pass_mark_in_artifact != PASS_MARK:
    print("NOTE: the mapping artifact records a pass mark of", pass_mark_in_artifact,
          "and this notebook reports", PASS_MARK, "-- the artifact predates the restatement "
          "of the criterion. The result clears both.")
''')

code(r'''
# How much the explanations move between training seeds.

tau = tm["kendall_tau"]
df = pd.DataFrame([{
    "class": r["class"],
    "mean rank agreement": round(r["mean tau"], 4),
    "lowest pair": round(r["min"], 4),
    "highest pair": round(r["max"], 4),
    "thin": thin_mark(r["class"]),
} for r in tau["per_class"]]).sort_values("mean rank agreement")

record("tau_mean", tau["mean_across_classes"])

emit_table(
    df,
    "attribution_stability_by_class",
    "Rank agreement between the highest-ranked features of models trained with different "
    "seeds, over all ten pairs of five seeds, per class.",
    sources=["data/processed/NB09b/threat_mapping.json"],
    note=("Mean across the nineteen classes " + format(tau["mean_across_classes"], ".4f") +
          ". No pass mark applies to this measurement; it is reported alongside the "
          "mapping so a reader can see how stable the explanations the mapping reads "
          "actually are."),
)
''')

code(r'''
# Figure: which features each class's explanations rest on.

with np.load(INPUTS["attr_seed_42"]) as z:
    keys = list(z.keys())
    A = z[keys[0]]
if A.shape != (len(attr["classes"]), len(attr["features"])):
    raise ValueError("Attribution table is " + str(A.shape) + ", expected "
                     + str((len(attr["classes"]), len(attr["features"]))))

# Scale each class to its own maximum so classes can be compared by shape, not size.
S = A / np.where(A.max(axis=1, keepdims=True) == 0, 1, A.max(axis=1, keepdims=True))
order = [attr["classes"].index(c) for c in sorted(attr["classes"])]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
im = ax.imshow(S[order], aspect="auto", cmap="Greys", vmin=0, vmax=1)
ax.set_xticks(range(len(attr["features"])))
ax.set_xticklabels(attr["features"], rotation=90, fontsize=6.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([sorted(attr["classes"])[i] + (" *" if sorted(attr["classes"])[i] in THIN else "")
                    for i in range(len(order))], fontsize=7)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cb.set_label("attribution, scaled within each class", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

emit_figure(
    fig,
    "attribution_weight_by_class_and_feature",
    "How much each of the forty remaining features contributes to each class's "
    "explanations, scaled within each class so the pattern rather than the magnitude is "
    "comparable across rows. The four timing features are absent because the model is "
    "trained without them.",
    sources=["data/processed/NB09a/attributions_seed_42.npz"],
)
''')

code(r'''
# Adverse event reports for the device categories in the testbed.

maude = tm["maude"]
by_device = pd.DataFrame(maude["per_generic_name"])
by_device.columns = ["device category", "reports returned"]
by_device = by_device.sort_values("reports returned", ascending=False)

kw = pd.DataFrame(maude["per_keyword"])
matched = kw[kw["records"] > 0]

emit_table(
    by_device,
    "adverse_event_reports_by_device_category",
    "Adverse event reports held in the public postmarket surveillance database for the "
    "device categories represented in the testbed, over the window searched.",
    sources=["data/processed/NB09b/threat_mapping.json"],
    note=("Four of the eight categories return nothing at all, which is a finding about "
          "how the database names devices rather than about those devices being safe. Of "
          + str(maude["denominator"]) + " reports across the eight categories, "
          + str(maude["numerator"]) + " mentions any of the fifteen attack-related terms "
          "searched, a share of " + format(maude["share"], ".4f") + ". That count is "
          "equally consistent with such events being rare and with the database having no "
          "category in which to record them, and it cannot distinguish the two. It is "
          "reported as adverse events in these device categories mentioning these terms, "
          "never as harm caused by an attack."),
)

record("maude_denominator", maude["denominator"])
record("maude_numerator", maude["numerator"])
print("terms matching anything:", matched["keyword"].tolist())
''')

md("""
## Where the features come from

The explanation stage runs on a model trained without the timing features. These tables
are why. They also stand as a result in their own right: the features carry information
about which recording session a row came from, and that is not removable by dropping any
single column.
""")

code(r'''
# Identifying the recording session, by feature set, with and without the header field.

prov = load_json("provenance")
abl = load_json("duration_ablation")
runs = {r["name"]: r for r in prov["capture_identification"]}
ablr = {r["name"]: r for r in abl["runs"]}

rows = [
    ("All forty-four features", 44, 0.02, runs["capture_all_features"]["accuracy"],
     ablr["all_minus_duration"]["accuracy"]),
    ("Timing features only", 4, 0.02, runs["capture_family_timing"]["accuracy"],
     ablr["timing_minus_duration"]["accuracy"]),
    ("Protocol features only", 28, 0.02, runs["capture_family_protocol"]["accuracy"], None),
    ("Statistical features only", 12, 0.02, runs["capture_family_statistical"]["accuracy"], None),
    ("The five features named by published work", 5, 0.02, runs["capture_prior_five"]["accuracy"], None),
    ("The header field alone", 1, 0.02, None, ablr["duration_only"]["accuracy"]),
    ("All features, attack class held fixed", 43, prov["within_class_mean_chance"],
     prov["within_class_mean_accuracy"], ablr["within_class_mean_minus_duration"]["accuracy"]),
]

df = pd.DataFrame([{
    "feature set": label,
    "features": n,
    "chance": round(ch, 4),
    "accuracy with the header field": round(a, 4) if a is not None else "",
    "accuracy without it": round(b, 4) if b is not None else "",
    "difference": round(b - a, 4) if (a is not None and b is not None) else "",
} for label, n, ch, a, b in rows])

record("capture_all", runs["capture_all_features"]["accuracy"])
record("capture_held_fixed", prov["within_class_mean_accuracy"])
record("capture_timing", runs["capture_family_timing"]["accuracy"])
record("capture_held_fixed_no_header", ablr["within_class_mean_minus_duration"]["accuracy"])
record("capture_all_no_header", ablr["all_minus_duration"]["accuracy"])
record("header_alone", ablr["duration_only"]["accuracy"])

emit_table(
    df,
    "recording_session_identification_by_feature_set",
    "How well a model names which recording session a row came from, by feature set, with "
    "and without the one column in the timing group that turns out to be a header field "
    "rather than a measure of time. The last row holds the attack class fixed, so telling "
    "one attack from another is no help.",
    sources=["data/processed/NB03_verdict.json",
             "data/processed/NB03b/duration_ablation.json"],
    note="Four timing features identify the session better than all forty-four together. "
         "The result with the attack class held fixed does not depend on the header field "
         "and rises slightly when it is removed, so the finding cannot be dismissed as an "
         "artifact of one badly chosen column. That column alone identifies the session a "
         "little above chance rather than at chance. Every figure here is a single run at "
         "one seed with no replicates.",
)
''')

code(r'''
# The same, class by class, with the attack held fixed.

wc = {r["name"]: r for r in abl["within_class_runs"]}
rows = []
for c in prov["classes"]:
    key_a = "capture_within_" + c.replace("-", "_")
    key_b = "within_" + c.replace("-", "_") + "_minus_duration"
    a = runs[key_a]["accuracy"]; b = wc[key_b]["accuracy"]
    rows.append({
        "class": c,
        "recordings": prov["recordings_per_class"][c],
        "chance": round(1.0 / prov["recordings_per_class"][c], 4),
        "accuracy with the header field": round(a, 4),
        "accuracy without it": round(b, 4),
        "difference": round(b - a, 4),
    })
df = pd.DataFrame(rows)

emit_table(
    df,
    "recording_session_identification_by_class",
    "Identifying which recording a row came from with the attack class held fixed, for the "
    "eight classes recorded more than once, with and without the header field.",
    sources=["data/processed/NB03_verdict.json",
             "data/processed/NB03b/duration_ablation.json"],
    note="Removing the header field raises the result on five of the eight classes and "
         "lowers it on three, one of them by a ten-thousandth.",
)
''')

code(r'''
# The published benchmark figures, as the anchor both this work and the reproduced model cite.

published = [
    ("Logistic regression", 0.432),
    ("AdaBoost", 0.141),
    ("Deep neural network", 0.522),
    ("Random forest", 0.551),
]
rows = [{
    "model": name,
    "F1": v,
    "split": "by capture file",
    "averaging": "not stated in the source",
} for name, v in published]

rows.append({
    "model": "Random forest, the published configuration, on a capture-disjoint split",
    "F1": round(headline("forest_leaf1")["macro_f1"], 4),
    "split": "by recording session",
    "averaging": "macro",
})
rows.append({
    "model": "Random forest, the leaf setting chosen here, on a capture-disjoint split",
    "F1": round(headline("forest")["macro_f1"], 4),
    "split": "by recording session",
    "averaging": "macro",
})
df = pd.DataFrame(rows)

record("forest_leaf1_macro", headline("forest_leaf1")["macro_f1"])

emit_table(
    df,
    "published_benchmark_anchor",
    "The originating benchmark's own published figures for this dataset, and the same "
    "forest configuration evaluated under a split that holds recording sessions apart.",
    sources=["Dadkhah et al. (2024), Table 7 and Section 5",
             "results/NB05/forest_19class_dadkhah_leaf1/metrics.json",
             "results/NB05/forest_19class/metrics.json"],
    note="Three things differ between the published figures and the runs here and none is "
         "controlled: the averaging method is not stated anywhere in the source, so the "
         "published figures may not be macro at all; the split is by capture file there "
         "and by recording session here; and the feature count is unsettled, with "
         "thirty-nine listed and forty-five shipped against the forty-four used here. This "
         "is a reference point showing where the work sits under a stricter protocol, not "
         "a controlled comparison between models.",
)
''')

md("""
## The check

Every value above that the chapter will quote is compared here against the figure already
recorded in the project record and the results ledger. This is the point of the notebook
as much as the tables are: it is the last place a number that drifted can be caught before
it reaches the prose.

One divergence is expected and is reported rather than resolved. The threat mapping
artifact carries the pass mark that was in force when it ran, and the criterion was
restated afterwards. The result clears both, so nothing downstream moves, but the notebook
reports the older figure rather than quietly discarding it.
""")

code(r'''
# Compare every quoted value against what the record already states.

EXPECTED = {
    "macro_published":              (0.7110, 5e-4),
    "macro_ours":                   (0.7356, 5e-4),
    "macro_forest":                 (0.8418, 5e-4),
    "macro_sequence":               (0.7138, 5e-4),
    "forest_leaf1_macro":           (0.8680, 5e-4),
    "n_recovered":                  (4, 0),
    "gains_sum":                    (2.0566, 5e-4),
    "losses_sum":                   (-2.0033, 5e-4),
    "net_sum":                      (0.0533, 5e-4),
    "n_crossed_up":                 (4, 0),
    "n_crossed_down":               (1, 0),
    "five_worst_sum":               (-1.7662, 5e-4),
    "volumetric_loss_share":        (0.88, 5e-3),
    "balancing_best_macro":         (0.7699, 5e-4),
    "seed_mean":                    (0.7373, 5e-4),
    "seed_sd_ddof1":                (0.0233, 5e-4),
    "mcnemar_budget_significant":   (5, 0),
    "mcnemar_intervention_significant": (3, 0),
    "n_pairs_below_090":            (11, 0),
    "n_pairs_total":                (171, 0),
    "median_low_rate":              (25, 0),
    "median_volumetric":            (15, 0),
    "n_saturating_below_reliable":  (8, 0),
    "n_saturating_at_five":         (7, 0),
    "n_censored":                   (8, 0),
    "budget_macros":                ([0.6295, 0.6837, 0.7323, 0.7138], 5e-4),
    "budget_reliable_counts":       ([9, 9, 11, 10], 0),
    "forest_40":                    (0.5910, 5e-4),
    "sequence_40_mean":             (0.6901, 5e-4),
    "sequence_40_sd":               (0.0236, 5e-4),
    "n_assigned":                   (18, 0),
    "n_agree_sequence":             (9, 0),
    "n_agree_forest":               (6, 0),
    "n_refused_features":           (20, 0),
    "tau_mean":                     (0.4560, 5e-4),
    "maude_denominator":            (829, 0),
    "maude_numerator":              (1, 0),
    "capture_all":                  (0.8010, 5e-4),
    "capture_held_fixed":           (0.8280, 5e-4),
    "capture_timing":               (0.9301, 5e-4),
    "capture_all_no_header":        (0.7495, 5e-4),
    "capture_held_fixed_no_header": (0.8504, 5e-4),
    "header_alone":                 (0.0299, 5e-4),
}

def agrees(got, want, tol):
    if isinstance(want, list):
        return (len(got) == len(want)
                and all(abs(float(g) - float(w)) <= tol for g, w in zip(got, want)))
    return abs(float(got) - float(want)) <= tol

failures, checked = [], 0
for key, (want, tol) in EXPECTED.items():
    if key not in EMITTED:
        failures.append((key, "not produced", want))
        continue
    checked += 1
    if not agrees(EMITTED[key], want, tol):
        failures.append((key, EMITTED[key], want))

print("checked", checked, "of", len(EXPECTED), "values")
if failures:
    for key, got, want in failures:
        print("  MISMATCH", key, "produced", got, "recorded", want)
    raise AssertionError(str(len(failures)) + " value(s) disagree with the record. "
                         "Resolve before any of this reaches the chapter.")
print("all", checked, "values agree with the record")

if pass_mark_in_artifact != PASS_MARK:
    print("known divergence: mapping artifact records pass mark",
          pass_mark_in_artifact, "and the record states", PASS_MARK,
          "- the artifact predates the restatement; the result clears both")
''')

code(r'''
# Write the captions and the record of what came from where.

lines = ["# Captions", "",
         "Descriptive captions for the tables and figures. Numbering is added when the",
         "chapter is written, not here.", ""]
for item in MANIFEST:
    lines += ["## " + item["file"], "", item["caption"], ""]
    if item["note"]:
        lines += ["Note. " + item["note"], ""]
(OUT / "captions.md").write_text("\n".join(lines) + "\n")

manifest = {
    "written_on": str(date.today()),
    "tables": sum(1 for m in MANIFEST if m["kind"] == "table"),
    "figures": sum(1 for m in MANIFEST if m["kind"] == "figure"),
    "values_checked_against_the_record": len(EXPECTED),
    "known_divergences": ([] if pass_mark_in_artifact == PASS_MARK else [{
        "what": "pass mark for attack-pattern resolution",
        "in the artifact": pass_mark_in_artifact,
        "in the record": PASS_MARK,
        "handling": "the record's figure is reported; the artifact predates the restatement",
    }]),
    "items": MANIFEST,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

print(manifest["tables"], "tables and", manifest["figures"], "figures written to", OUT)
''')

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

here = os.path.dirname(os.path.abspath(__file__))
root = here
while root != os.path.dirname(root):
    if os.path.exists(os.path.join(root, "PROJECT_RECORD.md")):
        break
    root = os.path.dirname(root)
target_dir = os.path.join(root, "notebooks")
if not os.path.isdir(target_dir):
    target_dir = here
out = os.path.join(target_dir, "AG_PRAXIS_NB10_results_consolidation.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
print("cells:", len(C))
