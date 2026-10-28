"""Step 2 — confidence gating by attribution concentration.

Exploratory. Reads only committed artefacts, writes only under explore/h3/.
Descriptive: every observed concentration value is used as a threshold. Nothing is
smoothed, binned, or selected.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

mapping = json.loads((ROOT / "data/processed/NB09b/threat_mapping.json").read_text())
arm = [a for a in mapping["h3"] if a["model"].startswith("sequence")][0]
rows = {r["class"]: r for r in arm["per_class"]}

rank = pd.read_csv(ROOT / "results/chapter4/tables/feature_ranking_by_class.csv")
rank.columns = [c.strip() for c in rank.columns]

# Which ranking table is the sequence arm? Verify against the artefact's stored
# top_features rather than trusting the filename.
def top5(df, cls):
    s = df[df["class"] == cls].sort_values("rank").head(5)
    return ", ".join(s["feature"].tolist())


exact = pd.read_csv(ROOT / "results/chapter4/tables/feature_ranking_by_class_exact_attributions.csv")
exact.columns = [c.strip() for c in exact.columns]
m_plain = sum(top5(rank, c) == rows[c]["top_features"] for c in rows)
m_exact = sum(top5(exact, c) == rows[c]["top_features"] for c in rows)
print(f"top-5 reproduction vs artefact: feature_ranking_by_class {m_plain}/18, "
      f"exact_attributions {m_exact}/18")
assert m_plain == len(rows), (
    f"feature_ranking_by_class does not reproduce the sequence arm's top_features "
    f"({m_plain}/18); exact_attributions gives {m_exact}/18"
)

# Concentration per class, from the share column (share of the ten).
conc = []
for cls, g in rank.groupby("class"):
    g = g.sort_values("rank")
    shares = g["share of the ten"].tolist()
    assert len(shares) == 10, f"{cls} has {len(shares)} rows, expected 10"
    tot = sum(shares)
    assert abs(tot - 1.0) < 0.02, f"{cls} shares sum to {tot:.4f}"
    conc.append(
        {
            "class": cls,
            "top1_share": shares[0],
            "top3_share": sum(shares[:3]),
            "agrees": rows[cls]["agrees"],
            "assigned": rows[cls]["assigned"],
            "truth": rows[cls]["truth"],
            "dos": rows[cls]["truth"] == "Denial of Service",
        }
    )

df = pd.DataFrame(conc).sort_values("top1_share", ascending=False).reset_index(drop=True)

lines = []
w = lines.append
w("STEP 2 — confidence gating by attribution concentration")
w("=" * 92)
w(f"artefact: threat_mapping.json git_sha {mapping['git_sha']}; ranking table "
  f"results/chapter4/tables/feature_ranking_by_class.csv")
w(f"top-5 reproduction check vs artefact top_features: {m_plain} of 18 classes")
w("")
w("Per-class concentration (sorted by top-1 share)")
w("-" * 92)
w(f"  {'class':<26}{'top1':>8}{'top3':>8}  {'agrees':<8}{'documented':<24}{'assigned'}")
for _, r in df.iterrows():
    w(f"  {r['class']:<26}{r['top1_share']:>8.4f}{r['top3_share']:>8.4f}  "
      f"{'yes' if r['agrees'] else 'NO':<8}{r['truth']:<24}{r['assigned']}")
w("")

curves = {}
for metric in ("top1_share", "top3_share"):
    w(f"Precision-coverage sweep on {metric}")
    w("-" * 92)
    w(f"  {'threshold':>10}{'kept':>7}{'coverage':>10}{'agree':>7}{'precision':>11}   dropped classes")
    thresholds = sorted(df[metric].unique())
    pts = []
    for t in thresholds:
        kept = df[df[metric] >= t]
        n_kept = len(kept)
        n_agree = int(kept["agrees"].sum())
        prec = n_agree / n_kept if n_kept else float("nan")
        pts.append({"threshold": float(t), "kept": n_kept, "coverage": n_kept / len(df),
                    "agree": n_agree, "precision": prec})
        dropped = len(df) - n_kept
        w(f"  {t:>10.4f}{n_kept:>7}{n_kept/len(df):>10.4f}{n_agree:>7}{prec:>11.4f}"
          f"   {dropped} dropped")
    curves[metric] = pts
    w("")
    # Does precision rise as the gate tightens? Report the raw sequence, no fit.
    precs = [p["precision"] for p in pts]
    w(f"  precision as threshold rises: {' '.join(f'{p:.3f}' for p in precs)}")
    w(f"  first (all 18 kept) = {precs[0]:.4f};  last (strictest) = {precs[-1]:.4f}")
    mono = all(b >= a - 1e-12 for a, b in zip(precs, precs[1:]))
    w(f"  monotone non-decreasing: {mono}")
    w("")

w("Concentration of agreeing vs disagreeing classes")
w("-" * 92)
for metric in ("top1_share", "top3_share"):
    a = df[df["agrees"]][metric]
    d = df[~df["agrees"]][metric]
    w(f"  {metric}")
    w(f"    agree    n={len(a):<3} min {a.min():.4f}  median {a.median():.4f}  max {a.max():.4f}")
    w(f"    disagree n={len(d):<3} min {d.min():.4f}  median {d.median():.4f}  max {d.max():.4f}")
w("")
w("Same, split by documented DoS / non-DoS")
w("-" * 92)
for metric in ("top1_share", "top3_share"):
    w(f"  {metric}")
    for label, sub in (("DoS", df[df["dos"]]), ("non-DoS", df[~df["dos"]])):
        w(f"    {label:<8} n={len(sub):<3} median {sub[metric].median():.4f}  "
          f"agreement {int(sub['agrees'].sum())}/{len(sub)}")

text = "\n".join(lines)
print(text)
(OUT / "step2_gating.txt").write_text(text + "\n")
df.to_csv(OUT / "step2_concentration.csv", index=False)
json.dump(curves, (OUT / "step2_curves.json").open("w"), indent=1)

# Figure: precision-coverage, both metrics. Points only, no fitted line.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, metric in zip(axes, ("top1_share", "top3_share")):
    pts = curves[metric]
    cov = [p["coverage"] for p in pts]
    prec = [p["precision"] for p in pts]
    ax.plot(cov, prec, marker="o", color="#31688e", linewidth=1.2, markersize=5)
    ax.axhline(9 / 18, color="#999999", linestyle="--", linewidth=1,
               label="ungated agreement (0.500)")
    ax.axhline(12 / 18, color="#c44e52", linestyle=":", linewidth=1.2,
               label="all-DoS baseline (0.667)")
    ax.set_xlabel(f"coverage — fraction of 18 classes still assigned")
    ax.set_ylabel("precision — fraction agreeing")
    ax.set_title(f"gate on {metric.replace('_', ' ')}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower left")
fig.suptitle("H3 confidence gating — precision against coverage (18 classes, descriptive)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "step2_precision_coverage.png", dpi=200)

# Second figure: concentration vs agreement, per class.
fig2, ax = plt.subplots(figsize=(7.5, 4.6))
for agrees, colour, label in ((True, "#2b8a3e", "agrees"), (False, "#c44e52", "disagrees")):
    sub = df[df["agrees"] == agrees]
    ax.scatter(sub["top1_share"], sub["top3_share"], c=colour, s=55, label=label,
               edgecolor="white", linewidth=0.8, zorder=3)
ax.set_xlabel("top-1 share of the ten")
ax.set_ylabel("top-3 share of the ten")
ax.set_title("Attribution concentration against agreement, per class")
ax.grid(alpha=0.25, linewidth=0.5)
ax.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(OUT / "step2_concentration_scatter.png", dpi=200)
print(f"\nwrote figures to {OUT}")
