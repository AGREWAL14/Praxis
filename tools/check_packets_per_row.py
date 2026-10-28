"""
Settles what the Number column measures.

Two checks, both on the raw CSVs. Nothing here is scaled, split, or read back
through a transform.

Check A. What Number holds, per class, as recorded.
Check B. Whether Number is the divisor the window statistics were computed over,
         tested as Tot sum / AVG against Number row by row.

Writes packets_per_row_check.json and prints a summary.
Trains nothing. Reads seven columns.
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd

# Leave as None to search Drive for the files. Set it to a directory to skip
# the search.
CSV_DIR = None

SEARCH_ROOT = "/content/drive/MyDrive"
OUT_DIR = "/content/drive/MyDrive/AG_PRAXIS_artifacts/checks"

COLS = ["Number", "Tot sum", "AVG", "Tot size", "Std", "Variance"]
VOLUMETRIC = {
    "DDoS-ICMP", "DDoS-SYN", "DDoS-TCP", "DDoS-UDP",
    "DoS-ICMP", "DoS-SYN", "DoS-TCP", "DoS-UDP",
}
ROWS_PER_FILE = None   # set to an integer to sample, None reads every row
RTOL = 1e-6


def parse_capture(filename):
    n = os.path.basename(filename)
    n = n.replace(".pcap.csv", "").replace(".csv", "")
    partition = "test" if n.endswith("_test") else "train"
    n = re.sub(r"_(train|test)$", "", n)
    capture_id = n
    label = re.sub(r"\d+$", "", n).replace("TCP_IP-", "")
    if label == "ARP_Spoofing":
        label = "Spoofing"
    return {"capture_id": capture_id, "label": label, "partition": partition}


def find_csv_dir():
    """Return the directory holding the most .csv files under SEARCH_ROOT."""
    if not os.path.isdir(SEARCH_ROOT):
        raise SystemExit(f"{SEARCH_ROOT} is not mounted. Run drive.mount first.")
    print(f"searching {SEARCH_ROOT} for the capture CSVs ...")
    counts = {}
    for dirpath, dirnames, filenames in os.walk(SEARCH_ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        n = sum(1 for f in filenames if f.endswith(".csv"))
        if n:
            counts[dirpath] = n
    if not counts:
        raise SystemExit(f"no .csv files anywhere under {SEARCH_ROOT}")
    for d, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:>4} csv  {d}")
    best = max(counts, key=counts.get)
    print(f"\nusing {best}\n")
    return best


csv_dir = CSV_DIR or find_csv_dir()
files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
print(f"{len(files)} files found in {csv_dir}")
if len(files) != 72:
    print("WARNING: expected 72 files")

header = pd.read_csv(files[0], nrows=0)
missing = [c for c in COLS if c not in header.columns]
if missing:
    raise SystemExit(f"columns absent from the header: {missing}\n"
                     f"header reads: {list(header.columns)}")

frames = []
for i, path in enumerate(files, 1):
    meta = parse_capture(path)
    df = pd.read_csv(path, usecols=COLS, nrows=ROWS_PER_FILE)
    df["label"] = meta["label"]
    df["capture_id"] = meta["capture_id"]
    frames.append(df)
    print(f"  [{i:2d}/{len(files)}] {os.path.basename(path):45s} {len(df):>9,} rows")

data = pd.concat(frames, ignore_index=True)
del frames
print(f"\n{len(data):,} rows read, {data['label'].nunique()} classes")
print(f"labels: {sorted(data['label'].unique())}\n")

# ---------------------------------------------------------------- check A

print("Check A. Number as recorded, per class\n")
rows_a = []
for label, g in data.groupby("label", sort=True):
    n = g["Number"]
    counts = n.value_counts().head(3)
    is_int = bool(np.all(np.isclose(n.values, np.round(n.values), rtol=0, atol=1e-9)))
    rows_a.append({
        "label": label,
        "group": "volumetric" if label in VOLUMETRIC else "other",
        "rows": int(len(n)),
        "min": float(n.min()),
        "median": float(n.median()),
        "max": float(n.max()),
        "distinct_values": int(n.nunique()),
        "all_integers": is_int,
        "top_3_values": {str(k): int(v) for k, v in counts.items()},
    })

a = pd.DataFrame(rows_a).sort_values(["group", "label"])
print(a[["label", "group", "min", "median", "max",
         "distinct_values", "all_integers"]].to_string(index=False))

vol = a.loc[a["group"] == "volumetric", "median"]
oth = a.loc[a["group"] == "other", "median"]
vol_med = float(vol.median()) if len(vol) else float("nan")
oth_med = float(oth.median()) if len(oth) else float("nan")
print(f"\n  volumetric median of medians: {vol_med}")
print(f"  other      median of medians: {oth_med}")
if oth_med:
    print(f"  ratio: {vol_med / oth_med:.4f}")

# ---------------------------------------------------------------- check B

print("\n\nCheck B. Is Number the divisor the window statistics used?\n")
print("If each row's statistics were computed over Number packets, then")
print("Tot sum / AVG should equal Number.\n")

valid = (data["AVG"] != 0).values
derived = np.full(len(data), np.nan)
derived[valid] = (data["Tot sum"].values[valid] / data["AVG"].values[valid])
data["derived_n"] = derived

rows_b = []
for label, g in data.groupby("label", sort=True):
    ok = g["derived_n"].notna().values
    if ok.sum() == 0:
        rows_b.append({
            "label": label,
            "group": "volumetric" if label in VOLUMETRIC else "other",
            "rows_testable": 0,
            "rows_avg_zero": int(len(g)),
            "match_rate": None,
        })
        continue
    d = g["derived_n"].values[ok]
    n = g["Number"].values[ok]
    match = np.isclose(d, n, rtol=RTOL, atol=1e-6)
    rows_b.append({
        "label": label,
        "group": "volumetric" if label in VOLUMETRIC else "other",
        "rows_testable": int(ok.sum()),
        "rows_avg_zero": int((~ok).sum()),
        "match_rate": float(match.mean()),
        "derived_median": float(np.median(d)),
        "number_median": float(np.median(n)),
    })

b = pd.DataFrame(rows_b).sort_values(["group", "label"])
print(b[["label", "group", "rows_testable", "match_rate",
         "derived_median", "number_median"]].to_string(index=False))

ok_all = data["derived_n"].notna().values
overall = float(np.isclose(data["derived_n"].values[ok_all],
                           data["Number"].values[ok_all],
                           rtol=RTOL, atol=1e-6).mean())
print(f"\n  overall match rate: {overall:.6f}")

# ---------------------------------------------------------------- verdict

if overall > 0.99:
    verdict = ("Number is the divisor the window statistics were computed over, "
               "so it is the packet count for the row.")
elif overall < 0.01:
    verdict = ("Number is not the divisor the window statistics were computed "
               "over, so this column does not establish the packet count.")
else:
    verdict = (f"Mixed: {overall:.4f} of rows match. Read the per-class rates "
               f"before concluding anything.")

print("\n" + verdict)
print("\nCheck A says what the column holds. Check B says whether it is a "
      "packet count at all. Neither tests the published claim about extraction "
      "windows; only counting the pcaps does that.")

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "packets_per_row_check.json")
with open(out, "w") as f:
    json.dump({
        "csv_dir": csv_dir,
        "files_read": len(files),
        "rows_read": int(len(data)),
        "rows_per_file_cap": ROWS_PER_FILE,
        "rtol": RTOL,
        "check_a_number_as_recorded": rows_a,
        "check_b_derived_divisor": rows_b,
        "check_b_overall_match_rate": overall,
        "volumetric_median_of_medians": vol_med,
        "other_median_of_medians": oth_med,
        "verdict": verdict,
    }, f, indent=2)

print(f"\nwritten to {out}")
