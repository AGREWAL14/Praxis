"""Do the artefacts, the notebooks, the ledger and the record still agree?

Each of these has already cost this project time: a criterion changed in the record while
an artefact kept the old one, a figure quoted from memory into a ledger entry, a
regenerated file landing one directory up from where everything reads it, an executed copy
picking up a dated name against the convention. None fails a test suite, because none is a
bug in code; they are disagreements between files that are each well-formed on their own.

Run from the repo root, no arguments. One line per check, non-zero exit if any fails.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"
MAPPING = PROC / "NB09b" / "threat_mapping.json"
MANIFEST = REPO / "results" / "chapter4" / "manifest.json"
LEDGER = REPO / "RESULTS_LEDGER.md"
RUNS = REPO / "runs"

EXPECTED_GATE = 43
EXECUTED = re.compile(r"^AG_PRAXIS_NB\d+[a-z]?_[a-z0-9_]+_executed\.ipynb$")



def notebook_constant(path, name):
    """The literal a notebook assigns to `name`, or None."""
    cells = json.loads(path.read_text())["cells"]
    pattern = re.compile(rf"^\s*{name}\s*=\s*(\d+)")
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for line in "".join(cell["source"]).split("\n"):
            found = pattern.match(line)
            if found:
                return int(found.group(1))
    return None


def latest_ledger_entry(prefix):
    """The text of the last entry whose heading starts with `prefix`."""
    text = LEDGER.read_text()
    starts = [m.start() for m in re.finditer(rf"^### {re.escape(prefix)}", text, re.M)]
    if not starts:
        return None
    after = re.search(r"^### ", text[starts[-1] + 4:], re.M)
    return text[starts[-1]: starts[-1] + 4 + after.start()] if after else text[starts[-1]:]


def check_pass_mark():
    artefact = json.loads(MAPPING.read_text())["pass_mark"]
    seen = {"artefact": artefact}
    for label, name in (("NB09b", "AG_PRAXIS_NB09b_threat_mapping.ipynb"),
                        ("NB10", "AG_PRAXIS_NB10_results_consolidation.ipynb")):
        seen[label] = notebook_constant(REPO / "notebooks" / name, "PASS_MARK")
    if len(set(seen.values())) == 1 and artefact is not None:
        return True, f"all three read {artefact}"
    return False, ", ".join(f"{k} {v}" for k, v in seen.items())


def check_ledger_figures():
    entry = latest_ledger_entry("NB09b")
    if entry is None:
        return False, "no NB09b entry in the ledger"
    d = json.loads(MAPPING.read_text())
    forest = next(r for r in d["h3"] if "forest" in r["model"].lower())
    seq = next(r for r in d["h3"] if "forest" not in r["model"].lower())
    maude = d["maude"]

    wanted = [
        ("CAPEC resolution", rf"{seq['n_assigned']} of {d['n_attack_classes']} classes resolved"),
        ("semantic agreement, sequence",
         rf"semantic agreement, sequence \| {seq['n_agree']} of {d['n_attack_classes']}"),
        ("semantic agreement, forest",
         rf"semantic agreement, forest \| {forest['n_agree']} of {d['n_attack_classes']}"),
        ("MAUDE count", rf"MAUDE keyword matches \| {maude['numerator']},"),
        ("MAUDE denominator", rf"MAUDE denominator \| {maude['denominator']} reports"),
        ("MAUDE share", rf"share of {maude['numerator'] / maude['denominator']:.6f}"),
    ]
    missing = [label for label, pattern in wanted if not re.search(pattern, entry)]
    if missing:
        return False, "ledger does not state " + "; ".join(missing)
    return True, f"{len(wanted)} figures agree with the latest NB09b entry"


def check_manifest():
    if not MANIFEST.exists():
        return False, f"{MANIFEST.relative_to(REPO)} is missing"
    m = json.loads(MANIFEST.read_text())
    divergences = m.get("known_divergences", [])
    gate = m.get("values_checked_against_the_record")
    problems = []
    if divergences:
        problems.append(f"{len(divergences)} known divergence(s): "
                        + "; ".join(str(x.get("what", x)) for x in divergences))
    if gate != EXPECTED_GATE:
        problems.append(f"gate count {gate}, expected {EXPECTED_GATE}")
    ok = not problems
    return ok, "; ".join(problems) if problems else f"no divergences, gate {gate} of {EXPECTED_GATE}"


def check_executed_names():
    stray = sorted(p.name for p in RUNS.glob("*.ipynb") if not EXECUTED.match(p.name))
    if stray:
        return False, "not on the undated pattern: " + ", ".join(stray)
    return True, f"{len(list(RUNS.glob('*.ipynb')))} executed copies, all undated"


def check_processed_layout():
    """No artefact appears both in its notebook subdirectory and loose above it.

    A regenerated file downloaded one directory up leaves the stale copy sitting where
    everything reads it by path, and both are well-formed, so nothing else notices. The
    top-level files PROJECT_RECORD.md section 9 documents are left alone; what is caught
    is a name existing in two places at once.
    """
    shadows = []
    for subdir in sorted(PROC.glob("NB*")):
        if not subdir.is_dir():
            continue
        for artefact in sorted(subdir.iterdir()):
            if artefact.is_file() and (PROC / artefact.name).is_file():
                shadows.append(f"{artefact.name} in both {subdir.name}/ and data/processed/")
    if shadows:
        return False, "; ".join(shadows)
    return True, "no artefact is shadowed by a loose copy above its subdirectory"


CHECKS = [
    ("pass mark agrees across artefact and both notebooks", check_pass_mark),
    ("threat_mapping.json agrees with the latest NB09b ledger entry", check_ledger_figures),
    ("chapter4 manifest is clean and the gate is complete", check_manifest),
    ("executed copies follow CLAUDE.md section 7", check_executed_names),
    ("no artefact is shadowed by a loose copy", check_processed_layout),
]


def main():
    failures = 0
    width = max(len(name) for name, _ in CHECKS)
    for name, run in CHECKS:
        try:
            ok, detail = run()
        except Exception as error:
            ok, detail = False, f"{type(error).__name__}: {error}"
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print()
    print(f"{len(CHECKS) - failures} of {len(CHECKS)} checks pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
