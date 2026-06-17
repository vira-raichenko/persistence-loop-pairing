#!/usr/bin/env python3
"""
H-threading: split a single network's loop persistence pairs into "threaded"
vs "not changed" by comparing against the UNION network's diagram, and write the
files that classify_loops.py consumes.

A loop pair that exists in the single network's diagram but has NO matching pair
in the union diagram is "threaded" (it was destroyed by threading with the other
network). Matching is done in the shared cropped/scaled coordinate frame:

    candidates = union pairs whose death point is within RADIUS of this pair's
    exact      = a candidate with death coord within DEATH_EPS AND death value
                 within DEATH_EPS                              -> not changed
    soft       = a candidate with birth value within BIRTH_EPS AND death value
                 within SOFT_DEATH                             -> not changed
    otherwise                                                  -> threaded

Per single network <name>, writes into OUT_ROOT/<name>/:
    <name>_Hb.csv   <name>_Hb_not_changed.csv     threaded / kept births
    <name>_Hd.csv   <name>_Hd_not_changed.csv     threaded / kept deaths
    <name>_Hbd.csv  <name>_Hbd_not_changed.csv    threaded / kept birth+death
    labeled_all.npy                               copy of the pairing output

The Hb/Hd/Hbd CSVs are ';'-separated (the format classify_loops.py reads).

Usage:
    python separate_diagrams.py            # batch over networks/ (auto-detect)
    python separate_diagrams.py --single networks/persistence_diagrams/C1_bd.csv \
        --union networks/persistence_diagrams/C1_2_bd.csv --name C1 \
        --pairing-npy networks/pairing_results/C1/labeled_birth_loop_local.npy \
        --out-dir networks/threading/C1
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

# ======================== CONFIG ========================
# Defaults are relative to this script so the tool runs from a fresh clone.

SCRIPT_DIR = Path(__file__).resolve().parent
NETWORKS_DIR = SCRIPT_DIR / "networks"
PD_DIR = NETWORKS_DIR / "persistence_diagrams"
PAIRING_DIR = NETWORKS_DIR / "pairing_results"
OUT_ROOT = NETWORKS_DIR / "threading"

# Matching tolerances. Coordinates are in the cropped/scaled voxel frame;
# birth/death values are persistence values (unchanged by cropping).
RADIUS = 6.0       # death-coordinate radius to gather union candidates
DEATH_EPS = 0.5    # exact match: death coordinate + death value tolerance
SOFT_DEATH = 1.5   # soft match: death value tolerance
BIRTH_EPS = 0.5    # soft match: birth value tolerance

# ======================== CSV I/O ========================


def load_bd_csv(path):
    """Load a birth/death CSV -> list of [birth, death, xb, yb, zb, xd, yd, zd]."""
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0] == "birth":
                continue
            try:
                rows.append([float(v) for v in row[:8]])
            except ValueError:
                continue
    return rows


def write_semicolon_csv(path, header, rows):
    """Write rows with a ';' delimiter (the format classify_loops.py reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


# ======================== THREADING ========================


def death_xyz(r):
    return (r[5], r[6], r[7])


def classify_pairs(single_rows, union_rows):
    """Split single_rows into (threaded, not_changed) against union_rows."""
    threaded, not_changed = [], []

    for r in single_rows:
        r_birth_val, r_death_val = r[0], r[1]
        r_death = death_xyz(r)

        candidates = [s for s in union_rows
                      if math.dist(r_death, death_xyz(s)) <= RADIUS]

        # 1) exact: same death location AND same death value
        exact = any(
            math.dist(r_death, death_xyz(s)) <= DEATH_EPS
            and abs(r_death_val - s[1]) <= DEATH_EPS
            for s in candidates
        )
        # 2) soft: same birth value AND close death value
        soft = any(
            abs(r_birth_val - s[0]) <= BIRTH_EPS
            and abs(r_death_val - s[1]) <= SOFT_DEATH
            for s in candidates
        )

        (not_changed if (exact or soft) else threaded).append(r)

    return threaded, not_changed


def hb_rows(rows):
    return [[r[2], r[3], r[4], round(abs(r[0]), 1)] for r in rows]


def hd_rows(rows):
    return [[r[5], r[6], r[7], round(abs(r[1]), 1)] for r in rows]


def hbd_rows(rows):
    return [[r[2], r[3], r[4], r[5], r[6], r[7],
             round(abs(r[0]), 1), round(abs(r[1]), 1)] for r in rows]


def run_one(single_csv, union_csv, name, pairing_npy, out_dir):
    """Run H-threading for one single network and write classify's input files."""
    single_rows = load_bd_csv(single_csv)
    union_rows = load_bd_csv(union_csv)

    threaded, not_changed = classify_pairs(single_rows, union_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    hb = ["x_b", "y_b", "z_b", "birth"]
    hd = ["x_d", "y_d", "z_d", "death"]
    hbd = ["x_b", "y_b", "z_b", "x_d", "y_d", "z_d", "birth", "death"]

    write_semicolon_csv(out_dir / f"{name}_Hb.csv", hb, hb_rows(threaded))
    write_semicolon_csv(out_dir / f"{name}_Hd.csv", hd, hd_rows(threaded))
    write_semicolon_csv(out_dir / f"{name}_Hbd.csv", hbd, hbd_rows(threaded))
    write_semicolon_csv(out_dir / f"{name}_Hb_not_changed.csv", hb, hb_rows(not_changed))
    write_semicolon_csv(out_dir / f"{name}_Hd_not_changed.csv", hd, hd_rows(not_changed))
    write_semicolon_csv(out_dir / f"{name}_Hbd_not_changed.csv", hbd, hbd_rows(not_changed))

    # labeled_all.npy: the pairing output, which classify_loops.py reads.
    if pairing_npy is not None and Path(pairing_npy).exists():
        np.save(out_dir / "labeled_all.npy", np.load(pairing_npy, allow_pickle=False))
        labeled_note = "labeled_all.npy"
    else:
        labeled_note = "labeled_all.npy SKIPPED (pairing .npy not found)"

    print(f"  {name}: single={len(single_rows)} union={len(union_rows)} "
          f"-> threaded={len(threaded)} not_changed={len(not_changed)}  "
          f"[{labeled_note}]  ({out_dir})")


# ======================== AUTO-DETECT DRIVER ========================


def _numbers(stem):
    return [int(n) for n in re.findall(r"\d+", stem)]


def discover_jobs(pd_dir, pairing_dir, out_root):
    """
    Find single diagrams (C<k>_bd.csv) and pair each with a union diagram
    (C<i>_<j>_bd.csv) whose label set contains k.
    """
    singles, unions = [], []
    for fp in sorted(pd_dir.glob("*_bd.csv")):
        stem = fp.name[:-len("_bd.csv")]
        (unions if len(_numbers(stem)) >= 2 else singles).append((stem, fp))

    jobs = []
    for name, single_fp in singles:
        k = _numbers(name)[0]
        union = next((fp for ustem, fp in unions if k in _numbers(ustem)), None)
        if union is None:
            print(f"  [skip] {name}: no union diagram contains network {k}")
            continue
        pairing_npy = pairing_dir / name / "labeled_birth_loop_local.npy"
        jobs.append((single_fp, union, name, pairing_npy, out_root / name))
    return jobs


def main():
    ap = argparse.ArgumentParser(description="H-threading: split single vs union diagrams.")
    ap.add_argument("--single", type=Path, help="Single-network bd CSV.")
    ap.add_argument("--union", type=Path, help="Union-network bd CSV.")
    ap.add_argument("--name", type=str, help="Network name (e.g. C1).")
    ap.add_argument("--pairing-npy", type=Path, help="labeled_birth_loop_local.npy for the single net.")
    ap.add_argument("--out-dir", type=Path, help="Output folder.")
    ap.add_argument("--pd-dir", type=Path, default=PD_DIR)
    ap.add_argument("--pairing-dir", type=Path, default=PAIRING_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    if args.single and args.union and args.name:
        out_dir = args.out_dir or (args.out_root / args.name)
        print("H-threading (single run):")
        run_one(args.single, args.union, args.name, args.pairing_npy, out_dir)
        return

    jobs = discover_jobs(args.pd_dir, args.pairing_dir, args.out_root)
    if not jobs:
        raise RuntimeError(f"No single/union diagram pairs found in {args.pd_dir}")
    print(f"H-threading (batch): {len(jobs)} network(s)")
    for single_fp, union_fp, name, pairing_npy, out_dir in jobs:
        run_one(single_fp, union_fp, name, pairing_npy, out_dir)
    print(f"\nDone. Outputs under {args.out_root}")


if __name__ == "__main__":
    main()