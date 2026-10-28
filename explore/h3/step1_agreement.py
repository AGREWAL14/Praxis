"""Step 1 — recompute H3 agreement properly, on the sequence-model arm.

Exploratory. Reads only committed artefacts, writes only under explore/h3/.
Nothing here feeds the gate, the manifest, or any governance file.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

mapping = json.loads((ROOT / "data/processed/NB09b/threat_mapping.json").read_text())
truth_cfg = yaml.safe_load((ROOT / "config/stride_ground_truth.yaml").read_text())

arm = [a for a in mapping["h3"] if a["model"].startswith("sequence")][0]
rows = arm["per_class"]

# Verify the artefact against the ground-truth file rather than trusting either alone.
gt = {k: v["stride"] for k, v in truth_cfg["classes"].items()}
mismatched_truth = [r["class"] for r in rows if gt.get(r["class"]) != r["truth"]]
assert not mismatched_truth, f"truth disagrees with config: {mismatched_truth}"
assert set(gt) == {r["class"] for r in rows}, "class sets differ"

# Recompute agreement rather than reading n_agree.
recomputed_agree = sum(1 for r in rows if r["assigned"] == r["truth"])
stored_agree = arm["n_agree"]

n = len(rows)
assigned = [r["assigned"] for r in rows]
truth = [r["truth"] for r in rows]
cats = sorted(set(assigned) | set(truth))

# Cohen's kappa, two raters (the rule, the documented category) over n classes.
po = recomputed_agree / n
pa = Counter(assigned)
pt = Counter(truth)
pe = sum((pa[c] / n) * (pt[c] / n) for c in cats)
kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")

# Split: DoS classes vs non-DoS, by the DOCUMENTED category, not the assigned one.
dos = [r for r in rows if r["truth"] == "Denial of Service"]
non_dos = [r for r in rows if r["truth"] != "Denial of Service"]


def rate(group):
    k = sum(1 for r in group if r["agrees"])
    return k, len(group), (k / len(group) if group else float("nan"))


# Confusion: documented -> assigned.
confusion = defaultdict(Counter)
for r in rows:
    confusion[r["truth"]][r["assigned"]] += 1

lines = []
w = lines.append

w("STEP 1 — H3 agreement, sequence model seed 42")
w("=" * 78)
w(f"artefact: data/processed/NB09b/threat_mapping.json (git_sha {mapping['git_sha']}, "
  f"generated {mapping['generated_on']})")
w(f"arm: {arm['model']}")
w("")
w("Raw agreement")
w("-" * 78)
w(f"  recomputed from per_class : {recomputed_agree} of {n} = {po:.4f}")
w(f"  stored n_agree            : {stored_agree} of {arm['n_assigned']} = {arm['proportion']:.4f}")
w(f"  agree                     : {'yes' if recomputed_agree == stored_agree else 'NO — MISMATCH'}")
w(f"  majority-class baseline   : {mapping['majority_class_baseline']} "
  f"(all-DoS rule = 12 of 18)")
w(f"  pass mark                 : {mapping['pass_mark']} of {mapping['n_attack_classes']}")
w("")
w("Cohen's kappa")
w("-" * 78)
w(f"  observed agreement  po = {po:.4f}")
w(f"  expected agreement  pe = {pe:.4f}")
w(f"  kappa                  = {kappa:.4f}")
w("")
w("  marginals")
w(f"    {'category':<24}{'assigned':>10}{'documented':>12}")
for c in cats:
    w(f"    {c:<24}{pa[c]:>10}{pt[c]:>12}")
w("")
w("Agreement split by documented category")
w("-" * 78)
for name, group in (("DoS classes", dos), ("non-DoS classes", non_dos)):
    k, tot, frac = rate(group)
    w(f"  {name:<18} {k:>2} of {tot:<2} = {frac:.4f}")
w("")
for name, group in (("DoS classes", dos), ("non-DoS classes", non_dos)):
    w(f"  {name}:")
    for r in sorted(group, key=lambda r: r["class"]):
        flag = "agree " if r["agrees"] else "MISS  "
        w(f"    {flag}{r['class']:<26} assigned {r['assigned']:<22} "
          f"documented {r['truth']:<22} CAPEC-{r['capec']}")
    w("")
w("Confusion — documented category to assigned category")
w("-" * 78)
w(f"  {'documented':<24}{'assigned':<24}{'n':>3}")
for t in sorted(confusion):
    for a in sorted(confusion[t]):
        mark = "" if t == a else "   <- disagreement"
        w(f"  {t:<24}{a:<24}{confusion[t][a]:>3}{mark}")
w("")
w("Direction of the disagreements")
w("-" * 78)
misses = [r for r in rows if not r["agrees"]]
dirn = Counter((r["truth"], r["assigned"]) for r in misses)
for (t, a), c in sorted(dirn.items(), key=lambda kv: -kv[1]):
    w(f"  {c:>2}x  {t}  ->  {a}")
w("")
w(f"  total disagreements: {len(misses)} of {n}")
into = Counter(r["assigned"] for r in misses)
w(f"  categories the disagreements land on: "
  + ", ".join(f"{k} ({v})" for k, v in sorted(into.items(), key=lambda kv: -kv[1])))
w("")
w("CAPEC ids behind the assignments")
w("-" * 78)
capec_by_assigned = defaultdict(Counter)
for r in rows:
    capec_by_assigned[r["assigned"]][r["capec"]] += 1
for a in sorted(capec_by_assigned):
    w(f"  {a:<24}" + ", ".join(f"CAPEC-{c} x{n_}" for c, n_ in sorted(capec_by_assigned[a].items())))

text = "\n".join(lines)
print(text)
(OUT / "step1_agreement.txt").write_text(text + "\n")

json.dump(
    {
        "n": n,
        "recomputed_agree": recomputed_agree,
        "stored_agree": stored_agree,
        "po": po,
        "pe": pe,
        "kappa": kappa,
        "dos": dict(zip(("k", "n", "frac"), rate(dos))),
        "non_dos": dict(zip(("k", "n", "frac"), rate(non_dos))),
        "confusion": {t: dict(c) for t, c in confusion.items()},
    },
    (OUT / "step1_agreement.json").open("w"),
    indent=1,
)
