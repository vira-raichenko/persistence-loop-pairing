#!/usr/bin/env python3
"""
Classify each loop as "threaded" or "not threaded" using the H-threading output,
and write consolidated loop geometry plus a per-loop summary.

A loop label is THREADED iff any of its birth points appears (within +-BIRTH_FUZZ
voxels) in the threaded-birth set <name>_Hb.csv produced by separate_diagrams.py.

Inputs per network <name>:
    networks/threading/<name>/labeled_all.npy   pairing output rows
    networks/threading/<name>/<name>_Hb.csv      threaded births (';'-separated)
    networks/cycles/<name>/net_cycle_<label>.poly   loop geometry
    networks/persistence_diagrams/<name>_bd.csv  (for the death persistence value)

Outputs into networks/classified/<name>/:
    Threaded_<name>.poly  Not_<name>.poly          loops with edges, colored by label
    labeled_birth_loop_threaded.npy / _rest.npy    [[label, label], ...]
    loops_summary_<name>.csv                        label, death_value, length, death, class

Coordinate note: labeled_all and Hb are already in the SAME (cropped/scaled)
voxel frame, so NO scaling is applied for matching. Loop lengths are reported in
micrometres via VOXEL_SIZE (set it to (1, 1, 1) to report voxel lengths).

Usage:
    python classify_loops.py                 # batch over networks/threading/*
    python classify_loops.py --name C1
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils import read_poly_into_list

# ======================== CONFIG ========================

SCRIPT_DIR = Path(__file__).resolve().parent
NETWORKS_DIR = SCRIPT_DIR / "networks"
THREADING_DIR = NETWORKS_DIR / "threading"
CYCLES_DIR = NETWORKS_DIR / "cycles"
PD_DIR = NETWORKS_DIR / "persistence_diagrams"
OUT_ROOT = NETWORKS_DIR / "classified"

BIRTH_FUZZ = 1                       # +-voxels when matching a birth to the Hb set
VOXEL_SIZE = (1.4, 0.332, 0.332)     # per-axis voxel size (um) for loop lengths

# ======================== I/O ========================


def load_bd_csv(path):
    """Load a comma-separated birth/death CSV -> list of 8-float rows."""
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


def load_hb_births(path):
    """Load <name>_Hb.csv (';'-separated x_b;y_b;z_b;birth) -> list of int triples."""
    triples = []
    with open(path) as f:
        for row in csv.reader(f, delimiter=";"):
            if not row or row[0] == "x_b":
                continue
            try:
                triples.append((int(float(row[0])), int(float(row[1])), int(float(row[2]))))
            except (ValueError, IndexError):
                continue
    return triples


def fuzzy_set(triples, fuzz):
    """Expand integer coordinates to include all neighbours within +-fuzz."""
    out = set()
    rng = range(-fuzz, fuzz + 1)
    for x, y, z in triples:
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    out.add((x + dx, y + dy, z + dz))
    return out


def write_loops_with_edges(loops, out_prefix):
    """
    Write loops (dict[label -> ordered list of (x, y, z)]) to one .poly with
    POINTS colored by label and POLYS edges connecting consecutive points,
    closing each loop.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_prefix}.poly", "w") as f:
        f.write("POINTS\n")
        loop_start, pid = {}, 1
        for lab, pts in loops.items():
            loop_start[lab] = pid
            for x, y, z in pts:
                f.write(f"{pid}: {x} {y} {z} c(0, 0, {lab}, 1)\n")
                pid += 1
        f.write("POLYS\n")
        eid = 1
        for lab, pts in loops.items():
            n = len(pts)
            if n < 2:
                continue
            start = loop_start[lab]
            for j in range(n):
                a = start + j
                b = start + (j + 1) % n          # close last -> first
                f.write(f"{eid}: {a} {b}\n")
                eid += 1
        f.write("END\n")


# ======================== CLASSIFICATION ========================


def loop_length_um(pts, voxel_size):
    """Polyline length in micrometres (anisotropic voxel sizes applied)."""
    if len(pts) < 2:
        return 0.0
    arr = np.asarray(pts, dtype=float) * np.asarray(voxel_size)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def classify_one(name, threading_dir, cycles_dir, diagram_csv, out_dir,
                 voxel_size, birth_fuzz):
    """Classify one network's loops and write geometry + summary."""
    net_dir = threading_dir / name
    labeled = np.load(net_dir / "labeled_all.npy", allow_pickle=False)
    hb_set = fuzzy_set(load_hb_births(net_dir / f"{name}_Hb.csv"), birth_fuzz)
    diagram = load_bd_csv(diagram_csv) if Path(diagram_csv).exists() else []

    labels_to_births = defaultdict(set)
    labels_to_deaths = defaultdict(set)
    death_value = {}
    for r in labeled:
        lab = int(r[3])
        if lab < 0:                              # unpaired birth
            continue
        labels_to_births[lab].add((int(r[0]), int(r[1]), int(r[2])))
        labels_to_deaths[lab].add((int(r[5]), int(r[6]), int(r[7])))
        bid = int(r[8])                          # 1-based row index into the diagram
        if lab not in death_value and 1 <= bid <= len(diagram):
            death_value[lab] = diagram[bid - 1][1]

    threaded = {lab for lab, births in labels_to_births.items()
                if any(b in hb_set for b in births)}
    rest = set(labels_to_births) - threaded

    # Load loop geometry for labels that have a .poly.
    def load_geom(labels):
        geom = {}
        for lab in sorted(labels):
            poly = cycles_dir / name / f"net_cycle_{lab}.poly"
            if poly.exists():
                geom[lab] = read_poly_into_list(str(poly))
        return geom

    all_threaded = load_geom(threaded)
    all_not = load_geom(rest)

    out_dir.mkdir(parents=True, exist_ok=True)
    if all_threaded:
        write_loops_with_edges(all_threaded, out_dir / f"Threaded_{name}")
    if all_not:
        write_loops_with_edges(all_not, out_dir / f"Not_{name}")

    np.save(out_dir / "labeled_birth_loop_threaded.npy",
            np.array([[lab, lab] for lab in sorted(all_threaded)], dtype=object))
    np.save(out_dir / "labeled_birth_loop_rest.npy",
            np.array([[lab, lab] for lab in sorted(all_not)], dtype=object))

    # loops_summary
    def rep_death(lab):
        ds = labels_to_deaths.get(lab)
        return sorted(ds)[0] if ds else (None, None, None)

    summary = []
    for cls, geom in (("th", all_threaded), ("no", all_not)):
        for lab in sorted(geom):
            dx, dy, dz = rep_death(lab)
            summary.append({
                "label": lab,
                "death_value": death_value.get(lab),
                "length": loop_length_um(geom[lab], voxel_size),
                "d_x": dx, "d_y": dy, "d_z": dz,
                "class": cls,
            })

    summary_path = out_dir / f"loops_summary_{name}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "death_value", "length",
                                          "d_x", "d_y", "d_z", "class"])
        w.writeheader()
        w.writerows(summary)

    print(f"  {name}: labels={len(labels_to_births)} "
          f"threaded={len(threaded)} not={len(rest)} | "
          f"geometry threaded={len(all_threaded)} not={len(all_not)}  ({out_dir})")


# ======================== DRIVER ========================


def discover_names(threading_dir):
    return sorted(d.name for d in threading_dir.iterdir()
                  if d.is_dir() and (d / "labeled_all.npy").exists())


def main():
    ap = argparse.ArgumentParser(description="Classify loops as threaded / not threaded.")
    ap.add_argument("--name", type=str, help="Single network name (e.g. C1).")
    ap.add_argument("--threading-dir", type=Path, default=THREADING_DIR)
    ap.add_argument("--cycles-dir", type=Path, default=CYCLES_DIR)
    ap.add_argument("--pd-dir", type=Path, default=PD_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    names = [args.name] if args.name else discover_names(args.threading_dir)
    if not names:
        raise RuntimeError(f"No <name>/labeled_all.npy found under {args.threading_dir}")

    print(f"Classifying {len(names)} network(s):")
    for name in names:
        classify_one(
            name=name,
            threading_dir=args.threading_dir,
            cycles_dir=args.cycles_dir,
            diagram_csv=args.pd_dir / f"{name}_bd.csv",
            out_dir=args.out_root / name,
            voxel_size=VOXEL_SIZE,
            birth_fuzz=BIRTH_FUZZ,
        )
    print(f"\nDone. Outputs under {args.out_root}")


if __name__ == "__main__":
    main()