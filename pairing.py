#!/usr/bin/env python3
"""
Persistence Pair to Loop Cycle Matching Algorithm.

Matches persistence diagram birth/death pairs to geometric loop cycle
representatives extracted from a minimum cycle basis of a network skeleton.

Algorithm described in:
    "Topological Analysis of Multi-Network Architecture in the Pancreas"
    Raichenko, Maaruf, Nyeng, Evans (2026)
    Supporting Information, Section 1.2

The algorithm proceeds in four stages:
    1. Candidate Harvesting  - spatial queries around birth and death points
    2. Loop Scoring           - spherical arc length, center distance, min death distance
    3. Primary Selection      - threshold-based rule (Eq. 2 in the supplement)
    4. Collision Resolution   - limited reassignment + strict final cleanup

Usage:
    # First, prepare the precomputed data:
    python prepare_data.py --cycles-dir example/cycles/loops

    # Then run pairing:
    python pairing.py --persistence example/cycles/persistence.csv \
                      --cycles-dir example/cycles/loops \
                      --output example/cycles/loops/labeled_birth_loop_local.npy
"""

from pathlib import Path
from collections import defaultdict
import os
import re
import csv
import sys
import signal
import pickle
import time
import numpy as np
from scipy.spatial import cKDTree
from functools import lru_cache

from utils import read_poly_into_list, calculate_barycenter

# ======================== ALGORITHM PARAMETERS ========================
# See Table 1 in the supplement for descriptions.

# Candidate harvesting
K_NEAR_VOXELS = 768          # k-NN neighbors for birth-side harvest
R_BIRTH_INIT = 3.0           # initial radius for birth augmentation
R_BIRTH_MAX = 128.0          # maximum radius for birth augmentation
R_GROWTH = 1.5               # multiplicative radius growth
MIN_CANDIDATES = 20          # minimum candidate labels to harvest
DEATH_RADIUS_PAD = 25        # offset added to |death_scalar| for death sphere
DEATH_RADIUS_MIN = 3.0       # minimum death sphere radius
R_DEATH_FALLBACK = 96.0      # fallback radius when death_scalar is missing

# Loop scoring / primary selection
SA_FLOOR = 5.0               # Omega_floor: SA qualification threshold (rad)
SA_THRESHOLD = 0.0           # fallback threshold for SA-first branch
HARD_CONSTRAINT_EPS = 20.0   # epsilon for hard constraint: md <= |delta| + eps

# Collision resolution
RESOLVE_COLLISIONS = True
COLLISION_MOVE_MAX_CENTER_DELTA = 18.0   # Delta_max
COLLISION_MAX_PASSES = 2
COLLISION_REQUIRE_SA_FLOOR = True

# Same-death resolution
RESOLVE_SAME_DEATH = True
DEATH_KEY_ROUND = 3          # decimal places for grouping deaths
PREFER_UNUSED_WITHIN_DEATH_GROUP = True

# Strict final cleanup
STRICT_FINAL_CLEANUP = True
UNPAIRED_LABEL_VALUE = -1.0

# Pre-filter
BIRTH_TO_LOOP_MAX_DIST = 200.0
MIN_FILTERED_TO_KEEP = 20

# Reports
WRITE_REPORTS = True
TOPK_IN_REPORT = 12

# ======================== GEOMETRY HELPERS ========================


def best_fit_plane(pts):
    """Fit a plane to a point cloud via SVD. Returns (center, normal, ex, ey)."""
    c = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[-1]
    ex = Vt[0]
    ey = Vt[1]
    return c, n, ex, ey


def spherical_arc_length(loop_pts, x):
    """
    Compute the spherical arc length Omega of a loop as seen from point x.

    Projects each edge of the loop onto the unit sphere centered at x,
    and sums the geodesic arc lengths. See Eq. (1) in the supplement.

    Returns: Omega in radians (can exceed 2*pi).
    """
    v = loop_pts - x
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-15)
    ang_sum = 0.0
    n = len(v)
    for i in range(n):
        a = v[i]
        b = v[(i + 1) % n]
        cross = np.linalg.norm(np.cross(a, b))
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        ang_sum += np.arctan2(cross, 1.0 + dot)
    return 2.0 * ang_sum


# ======================== CSV I/O ========================


def load_persistence_csv(filepath):
    """
    Load a persistence diagram CSV.

    Expected columns: birth, death, x_b, y_b, z_b, x_d, y_d, z_d

    Returns:
        births: dict {id: [bx, by, bz]}
        deaths: dict {id: [dx, dy, dz]}
        death_scalars: dict {id: float or None}
    """
    data_birth, data_death, data_death_scalar = {}, {}, {}
    i = 1
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if not row or row[0] == 'birth':
                continue
            vals = [None] * 8
            for j in range(min(len(row), 8)):
                vals[j] = row[j]
            try:
                bx = float(vals[2])
                by = float(vals[3])
                bz = float(vals[4])
                dx = float(vals[5])
                dy = float(vals[6])
                dz = float(vals[7])
            except Exception:
                continue
            data_birth[i] = [bx, by, bz]
            data_death[i] = [dx, dy, dz]
            ds = None
            try:
                if vals[1] is not None and vals[1] != '':
                    ds = float(vals[1])
            except Exception:
                ds = None
            data_death_scalar[i] = ds
            i += 1
    return data_birth, data_death, data_death_scalar


# ======================== CHECKPOINT I/O ========================

N_COLS = 9


def rows_to_float2d(rows, ncols=N_COLS):
    arr = np.full((len(rows), ncols), np.nan, dtype=float)
    for i, r in enumerate(rows):
        r = list(r)
        if len(r) < ncols:
            r += [np.nan] * (ncols - len(r))
        elif len(r) > ncols:
            r = r[:ncols]
        arr[i] = np.asarray(r, dtype=float)
    return arr


def save_checkpoint(path, rows):
    arr = rows_to_float2d(rows)
    np.save(path, arr)


def try_load_checkpoint(path):
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr.astype(float, copy=False).tolist()
    except Exception:
        return None


def dedup_by_birth_id(rows):
    if not rows:
        return rows
    last_idx = {}
    for i, r in enumerate(rows):
        if len(r) >= 9:
            try:
                bid = int(float(r[8]))
            except Exception:
                continue
            last_idx[bid] = i
    keep = set(last_idx.values())
    return [rows[i] for i in range(len(rows)) if i in keep]


# ======================== REPORT HELPERS ========================


def fmt_xyz(xyz):
    return f"({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"


def open_report(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", buffering=1)


# ======================== COLLISION RESOLUTION ========================
# Algorithm 1 in the supplement.


def build_label_to_births(assigned_label_by_birth):
    lab2bs = defaultdict(list)
    for b, lab in assigned_label_by_birth.items():
        lab2bs[int(lab)].append(int(b))
    return lab2bs


def _decision_block_lines(b, birth_meta, pref_lists, centers_dict):
    """Produce a decision block for a birth, used in reports."""
    meta = birth_meta[b]
    prefs = pref_lists.get(b, [])
    lines = []
    if not prefs:
        lines.append(f"\n  birth_id={b}\n    (no pref list stored)\n")
        return lines
    chosen = prefs[0]
    second = prefs[1] if len(prefs) > 1 else None
    center_xyz = centers_dict.get(chosen["label"], np.array([np.nan]*3))
    lines.append(f"\nlabel_center={fmt_xyz(center_xyz)}\n\n")
    lines.append(f"  birth_id={b}\n")
    lines.append(f"    birth={fmt_xyz(meta['birth'])}   death={fmt_xyz(meta['death'])}\n")
    lines.append(f"    rule={meta['used_rule']}  hard_ok={meta['hard_ok']}  "
                 f"hard_thresh={meta['hard_thresh']:.3f}  candidates_union={meta['candidates_union']}\n")
    lines.append(f"    CHOSEN label={chosen['label']}  sa={chosen['sa']:.6f}  "
                 f"center_dist={chosen['center_dist']:.3f}  min_d={chosen['min_death_dist']:.3f}\n")
    if second is not None:
        lines.append(f"    SECOND: label={second['label']}  sa={second['sa']:.6f}  "
                     f"center_dist={second['center_dist']:.3f}  min_d={second['min_death_dist']:.3f}\n")
    lines.append("    TOP-K:\n")
    lines.append("      k  label        sa      center_dist  min_death_dist\n")
    for k_i, t in enumerate(prefs):
        lines.append(f"     {k_i:02d}  {t['label']:6d}  {t['sa']:10.6f}  "
                     f"{t['center_dist']:11.3f}  {t['min_death_dist']:13.3f}\n")
    return lines


def resolve_collisions_limited(birth_ids, pref_lists, assigned_label_by_birth,
                               max_passes, max_center_delta, sa_floor,
                               require_sa_floor, report_lines,
                               birth_meta, centers_dict):
    """
    Label Collision Resolution (Algorithm 1 in supplement).

    For each label claimed by multiple births:
      1. Find keeper (birth that cannot be moved)
      2. Move non-keepers to next-best available alternative
         satisfying SA >= sa_floor and delta_cd <= max_center_delta
    """
    assigned = dict(assigned_label_by_birth)

    for p in range(1, max_passes + 1):
        lab2bs = build_label_to_births(assigned)
        conflicts = {lab: bs for lab, bs in lab2bs.items()
                     if lab != int(UNPAIRED_LABEL_VALUE) and len(bs) > 1}
        if not conflicts:
            report_lines.append(f"\n[collisions] pass {p}: none\n")
            return assigned

        report_lines.append(f"\n[collisions] pass {p}: {len(conflicts)} collided labels\n")

        for lab in sorted(conflicts.keys()):
            bs = sorted(conflicts[lab])

            # Determine movability of each birth
            move_options = []
            for b in bs:
                prefs = pref_lists.get(b, [])
                if not prefs:
                    move_options.append((False, float("inf"), None, b))
                    continue
                cur_cd = prefs[0]["center_dist"]
                best_alt = None
                for k_idx, t in enumerate(prefs[1:], start=1):
                    if require_sa_floor and t["sa"] < sa_floor:
                        continue
                    delta = t["center_dist"] - cur_cd
                    if delta > max_center_delta:
                        continue
                    if int(t["label"]) == lab:
                        continue
                    best_alt = (k_idx, t, delta)
                    break
                if best_alt is None:
                    move_options.append((False, float("inf"), None, b))
                else:
                    move_options.append((True, best_alt[2], best_alt, b))

            # Select keeper
            if any(not mo[0] for mo in move_options):
                move_options_sorted = sorted(move_options, key=lambda x: (x[0], x[1], x[3]))
                keeper = move_options_sorted[0][3]
            else:
                keeper = max(move_options, key=lambda x: x[1])[3]

            # Move non-keepers
            taken_labels = set(int(v) for v in assigned.values()
                               if int(v) != int(UNPAIRED_LABEL_VALUE))
            taken_labels.discard(lab)

            for b in bs:
                if b == keeper:
                    assigned[b] = lab
                    continue
                prefs = pref_lists.get(b, [])
                if not prefs:
                    assigned[b] = lab
                    continue
                cur_cd = prefs[0]["center_dist"]
                moved = False
                for k_idx, t in enumerate(prefs[1:], start=1):
                    if require_sa_floor and t["sa"] < sa_floor:
                        continue
                    delta = t["center_dist"] - cur_cd
                    if delta > max_center_delta:
                        continue
                    cand_lab = int(t["label"])
                    if cand_lab == lab:
                        continue
                    if cand_lab not in taken_labels:
                        report_lines.append(
                            f"  MOVE birth_id={b}: {lab} -> {cand_lab} "
                            f"(k={k_idx}, delta_cd={delta:.3f}, sa={t['sa']:.6f})\n"
                        )
                        assigned[b] = cand_lab
                        taken_labels.add(cand_lab)
                        moved = True
                        break
                if not moved:
                    assigned[b] = lab

            assigned[keeper] = lab

    return assigned


def resolve_same_death_collisions(processed_births, assigned_map, birth_meta,
                                  pref_lists, min_dist_fn, report_lines,
                                  ndigits, prefer_unused, sa_floor,
                                  require_sa_for_move):
    """
    Resolve collisions where multiple births share the same death point.

    Keep the birth whose birth point is closest to the loop voxels;
    move others to alternatives if possible, otherwise mark unpaired.
    """
    assigned = dict(assigned_map)

    # Group births by rounded death coordinates
    death2bs = defaultdict(list)
    for b in processed_births:
        if b not in birth_meta:
            continue
        dx, dy, dz = birth_meta[b]["death"]
        dk = (round(dx, ndigits), round(dy, ndigits), round(dz, ndigits))
        death2bs[dk].append(b)

    same_death_groups = {k: sorted(v) for k, v in death2bs.items() if len(v) > 1}
    report_lines.append(f"[same-death] groups_found={len(same_death_groups)}\n")

    if not same_death_groups:
        return assigned

    changes = 0
    unpaired_count = 0

    for dk in sorted(same_death_groups.keys()):
        bs = same_death_groups[dk]

        # Find keeper: birth closest to its assigned loop
        best_keeper = None
        best_keeper_dist = float('inf')
        for b in bs:
            lab = int(assigned.get(b, UNPAIRED_LABEL_VALUE))
            if lab == int(UNPAIRED_LABEL_VALUE):
                continue
            q_birth = np.array(birth_meta[b].get("q_birth", birth_meta[b]["birth"]), dtype=float)
            d, _ = min_dist_fn(lab, q_birth)
            if d < best_keeper_dist:
                best_keeper_dist = d
                best_keeper = b

        if best_keeper is None:
            continue

        # Try to move others
        group_taken = set(int(assigned.get(b, UNPAIRED_LABEL_VALUE)) for b in bs
                          if int(assigned.get(b, UNPAIRED_LABEL_VALUE)) != int(UNPAIRED_LABEL_VALUE))

        for b in bs:
            if b == best_keeper:
                continue
            prefs = pref_lists.get(b, [])
            moved = False
            for t in prefs[1:]:
                cand_lab = int(t["label"])
                if require_sa_for_move and t["sa"] < sa_floor:
                    continue
                if prefer_unused and cand_lab in group_taken:
                    continue
                if cand_lab not in set(int(v) for v in assigned.values()):
                    assigned[b] = cand_lab
                    group_taken.add(cand_lab)
                    changes += 1
                    moved = True
                    break
            if not moved:
                assigned[b] = float(UNPAIRED_LABEL_VALUE)
                unpaired_count += 1

    report_lines.append(f"[same-death] moves={changes}, unpaired={unpaired_count}\n")
    return assigned


def strict_finalize(processed_births, assigned_map, pref_lists, report_lines,
                    sa_floor):
    """
    Strict Final Cleanup: for any remaining collided label, keep one winner
    and mark all others as unpaired.
    """
    assigned = dict(assigned_map)
    lab2bs = build_label_to_births(assigned)
    conflicts = {lab: bs for lab, bs in lab2bs.items()
                 if lab != int(UNPAIRED_LABEL_VALUE) and len(bs) > 1}

    report_lines.append(f"[strict-final] collided labels: {len(conflicts)}\n")
    if not conflicts:
        return assigned

    dropped = 0
    for lab in sorted(conflicts.keys()):
        bs = sorted(conflicts[lab])

        # Score each birth on this label
        scored = []
        for b in bs:
            prefs = pref_lists.get(b, [])
            entry = next((t for t in prefs if int(t["label"]) == lab), None)
            if entry is None:
                scored.append((b, -np.inf, np.inf))
            else:
                scored.append((b, entry["sa"], entry["center_dist"]))

        has_qualified = any(s[1] >= sa_floor for s in scored)
        if has_qualified:
            scored.sort(key=lambda s: (s[2], -s[1], s[0]))
        else:
            scored.sort(key=lambda s: (-s[1], s[2], s[0]))

        winner = scored[0][0]
        for b, _, _ in scored[1:]:
            assigned[b] = float(UNPAIRED_LABEL_VALUE)
            dropped += 1

    report_lines.append(f"[strict-final] dropped={dropped}\n")
    return assigned


# ======================== PRIMARY SELECTION ========================
# Eq. (2) in the supplement.


def choose_best_candidate(eligible):
    """
    Primary selection rule (Eq. 2).

    If any candidate has SA >= SA_FLOOR, select the one with minimum
    center distance among those. Otherwise, fall back to maximum SA.

    Each element of eligible is a tuple:
        (label, sa, cdist, plane_dist, birth_plane_dist, n_pts, min_death_dist)

    Returns: (sorted_eligible, used_rule_string)
    """
    sa_qualified = [t for t in eligible if t[1] >= SA_FLOOR]

    if sa_qualified:
        sa_qualified_sorted = sorted(sa_qualified, key=lambda t: (t[2], -t[1], t[6], t[0]))
        chosen = sa_qualified_sorted[0]
        rest = sorted([t for t in eligible if t is not chosen],
                      key=lambda t: (t[2], -t[1], t[6], t[0]))
        return [chosen] + rest, f"SA>={SA_FLOOR:g} -> min-center (count={len(sa_qualified)})"

    max_sa = max(t[1] for t in eligible)
    if max_sa >= SA_THRESHOLD:
        return (sorted(eligible, key=lambda t: (-t[1], t[2], t[3], t[4], t[6], t[0])),
                f"SA-first (threshold {SA_THRESHOLD:.3f} sr)")
    else:
        return (sorted(eligible, key=lambda t: (t[2], -t[1], t[3], t[4], t[6], t[0])),
                f"nearest-center fallback (all SA < {SA_THRESHOLD:.3f} sr)")


# ======================== MAIN ========================


def main():
    persistence_csv = "/home/vira/persistence-loop-pairing/data/cycles_bd.csv"
    cycles_dir = Path("/home/vira/persistence-loop-pairing/cycles")
    output_path = cycles_dir / "labeled_birth_loop_local.npy"

    # Load persistence diagram
    print(f"Loading persistence diagram from {persistence_csv} ...")
    births, deaths, death_scalar = load_persistence_csv(persistence_csv)
    print(f"  {len(births)} persistence pairs loaded.")

    # Build labeled volume and voxel dictionary directly from .poly files
    print(f"Building labeled volume from {cycles_dir} ...")
    dict_labels_for_voxels = {}
    all_coords = []
    centers_dict = {}

    for entry in sorted(os.listdir(cycles_dir)):
        if not (entry.startswith("net_cycle_") and entry.endswith(".poly")):
            continue
        lab = int(re.findall(r'\d+', entry)[0])
        pts = read_poly_into_list(str(cycles_dir / entry))
        centers_dict[lab] = np.array(calculate_barycenter(pts), dtype=float)
        for x, y, z in pts:
            key = (x, y, z)
            all_coords.append(key)
            if key in dict_labels_for_voxels:
                dict_labels_for_voxels[key].append(lab)
            else:
                dict_labels_for_voxels[key] = [lab]

    if not centers_dict:
        raise RuntimeError("No net_cycle_*.poly files found")
    if not all_coords:
        raise RuntimeError("No loop voxels found in .poly files")

    coords_arr = np.array(all_coords)
    max_xyz = coords_arr.max(axis=0)
    vol_shape = (int(max_xyz[0]) + 2, int(max_xyz[1]) + 2, int(max_xyz[2]) + 2)
    just_b_w_loops = np.zeros(vol_shape, dtype=np.uint8)
    for x, y, z in all_coords:
        just_b_w_loops[x, y, z] = 1

    # Build KD-tree over loop voxels
    loop_voxels = np.argwhere(just_b_w_loops > 0)
    tree_loops = cKDTree(loop_voxels)
    print(f"  loop voxels: {loop_voxels.shape[0]}, volume shape: {vol_shape}")
    print(f"  loop centers loaded: {len(centers_dict)}")

    centers_arr = np.array([centers_dict[k] for k in sorted(centers_dict)], dtype=float)
    labels_arr = np.array(sorted(centers_dict.keys()))

    # Cached helpers
    @lru_cache(maxsize=1024)
    def get_voxels_for_label(lab):
        coords = [key for key, labs in dict_labels_for_voxels.items() if lab in labs]
        if not coords:
            return np.empty((0, 3), dtype=float)
        return np.array(coords, dtype=float)

    def min_dist_to_label_voxels(lab, q):
        pts = get_voxels_for_label(lab)
        if pts.size == 0:
            return np.inf, 0
        d = np.linalg.norm(pts - q[None, :], axis=1)
        return float(d.min()), int(pts.shape[0])

    @lru_cache(maxsize=4096)
    def get_loop_points(label):
        pts = read_poly_into_list(str(cycles_dir / f"net_cycle_{label}.poly"))
        return np.array(pts, dtype=float)

    # SIGTERM handling
    shutting_down = {"flag": False}

    def _term_handler(signum, frame):
        shutting_down["flag"] = True
    signal.signal(signal.SIGTERM, _term_handler)

    # Reports
    decisions_path = cycles_dir / "decisions_all.txt"
    collisions_path = cycles_dir / "collisions_report.txt"
    f_dec = open_report(decisions_path) if WRITE_REPORTS else None

    # State
    pref_lists = {}
    birth_meta = {}
    chosen_label_noninj = {}
    list_of_labeled_birth = []

    # Resume from checkpoint if exists
    if output_path.exists():
        loaded = try_load_checkpoint(output_path)
        if loaded:
            list_of_labeled_birth = dedup_by_birth_id(loaded)
            print(f"  resumed {len(list_of_labeled_birth)} rows from checkpoint.")
    processed_ids = {int(r[8]) for r in list_of_labeled_birth if len(r) >= 9}

    # ---- Per-birth processing ----
    def process_birth(b):
        if b not in deaths:
            return False

        bx, by, bz = births[b]
        dx, dy, dz = deaths[b]
        birth_pt = np.array([bx, by, bz], dtype=float)
        death_pt = np.array([dx, dy, dz], dtype=float)
        q_birth = np.array([bx - 0.5, by - 0.5, bz - 0.5], dtype=float)
        q_death = np.array([dx - 0.5, dy - 0.5, dz - 0.5], dtype=float)

        # Death sphere radius
        ds_val = death_scalar.get(b, None)
        if ds_val is None or not np.isfinite(ds_val):
            r_death = float(R_DEATH_FALLBACK)
        else:
            r_death = float(max(abs(ds_val) + DEATH_RADIUS_PAD, DEATH_RADIUS_MIN))

        # Hard constraint threshold
        if ds_val is not None and np.isfinite(ds_val):
            hard_thresh = float(abs(ds_val) + HARD_CONSTRAINT_EPS)
        else:
            hard_thresh = float(R_DEATH_FALLBACK + HARD_CONSTRAINT_EPS)

        # ---- Candidate Harvesting (Section 1.2.2) ----
        # Birth-side: k-NN + iterative radius expansion
        birth_labels = set()
        k = min(K_NEAR_VOXELS, loop_voxels.shape[0])
        dists_knn, idxs_knn = tree_loops.query(q_birth, k=k)
        dists_knn_arr = np.atleast_1d(dists_knn)
        idxs_knn = np.atleast_1d(idxs_knn)
        kNN_radius = float(np.max(dists_knn_arr)) if dists_knn_arr.size else 0.0

        for idx_lv in idxs_knn:
            key = tuple(loop_voxels[int(idx_lv)])
            labs = dict_labels_for_voxels.get(key, [])
            if labs:
                birth_labels.update(labs)

        if len(birth_labels) < MIN_CANDIDATES:
            r = max(R_BIRTH_INIT, kNN_radius + 1.0)
            while r <= R_BIRTH_MAX and len(birth_labels) < MIN_CANDIDATES:
                idxs = tree_loops.query_ball_point(q_birth, r=r)
                for idx_lv in idxs:
                    key = tuple(loop_voxels[int(idx_lv)])
                    labs = dict_labels_for_voxels.get(key, [])
                    if labs:
                        birth_labels.update(labs)
                r *= R_GROWTH

        # Death-side: radius search
        death_labels = set()
        idxs_death = tree_loops.query_ball_point(q_death, r=r_death)
        for idx_lv in idxs_death:
            key = tuple(loop_voxels[int(idx_lv)])
            labs = dict_labels_for_voxels.get(key, [])
            if labs:
                death_labels.update(labs)

        # Union
        candidates = birth_labels | death_labels
        if not candidates:
            return False

        # ---- Loop Scoring (Section 1.2.3) ----
        def _cand_sort_key(lab_i):
            lid = int(lab_i)
            c = centers_dict.get(lid)
            cdist = float(np.linalg.norm(c - death_pt)) if c is not None else float('inf')
            mdist, _ = min_dist_to_label_voxels(lid, q_birth)
            return (cdist, mdist, lid)

        ordered_candidates = sorted(candidates, key=_cand_sort_key)

        ranked = []
        for lab in ordered_candidates:
            lab_i = int(lab)
            try:
                pts = get_loop_points(lab_i)
            except Exception:
                continue
            if pts.shape[0] < 3:
                continue

            c, n, ex, ey = best_fit_plane(pts)
            plane_dist = abs(np.dot(death_pt - c, n))
            birth_plane_dist = abs(np.dot(birth_pt - c, n))

            try:
                sa = spherical_arc_length(pts, death_pt)
            except Exception:
                sa = -np.inf

            center = centers_dict.get(lab_i)
            cdist = np.linalg.norm(center - death_pt) if center is not None else np.inf

            try:
                min_death_dist = float(np.min(np.linalg.norm(pts - death_pt[None, :], axis=1)))
            except Exception:
                min_death_dist = float('inf')

            ranked.append((lab_i, sa, cdist, plane_dist, birth_plane_dist,
                           int(pts.shape[0]), min_death_dist))

        if not ranked:
            return False

        # ---- Primary Selection (Section 1.2.4) ----
        # Hard constraint: md(L, d) <= |delta| + epsilon
        eligible = [t for t in ranked if t[6] <= hard_thresh]
        hard_ok = True
        if not eligible:
            hard_ok = False
            min_over_all = min(ranked, key=lambda t: t[6])
            print(f"[viol] birth_id={b}: no candidate within hard threshold "
                  f"({hard_thresh:.1f}). closest={min_over_all[6]:.1f}")
            eligible = ranked

        eligible_sorted, used_rule = choose_best_candidate(eligible)
        best_label = int(eligible_sorted[0][0])
        chosen_label_noninj[int(b)] = best_label

        # Store preferences for collision resolution
        topk = eligible_sorted[:min(TOPK_IN_REPORT, len(eligible_sorted))]
        prefs = []
        for t in topk:
            lab_i = int(t[0])
            bdist, _ = min_dist_to_label_voxels(lab_i, q_birth)
            prefs.append({
                "label": lab_i,
                "sa": float(t[1]),
                "center_dist": float(t[2]),
                "min_death_dist": float(t[6]),
                "birth_vox_dist": float(bdist),
            })
        pref_lists[int(b)] = prefs

        birth_meta[int(b)] = {
            "birth_id": int(b),
            "birth": (float(bx), float(by), float(bz)),
            "death": (float(dx), float(dy), float(dz)),
            "q_birth": (float(q_birth[0]), float(q_birth[1]), float(q_birth[2])),
            "used_rule": used_rule,
            "hard_ok": hard_ok,
            "hard_thresh": float(hard_thresh),
            "candidates_union": int(len(candidates)),
            "ds": None if (ds_val is None or not np.isfinite(ds_val)) else float(ds_val),
        }

        # Write decision report
        if WRITE_REPORTS and f_dec is not None:
            first = prefs[0]
            second = prefs[1] if len(prefs) > 1 else None
            center_xyz = centers_dict.get(first["label"], np.array([np.nan]*3))
            f_dec.write(f"\nlabel_center={fmt_xyz(center_xyz)}\n\n")
            f_dec.write(f"  birth_id={int(b)}\n")
            f_dec.write(f"    birth={fmt_xyz(birth_meta[int(b)]['birth'])}   "
                        f"death={fmt_xyz(birth_meta[int(b)]['death'])}\n")
            f_dec.write(f"    rule={used_rule}  hard_ok={hard_ok}  "
                        f"hard_thresh={hard_thresh:.3f}  candidates={len(candidates)}\n")
            f_dec.write(f"    CHOSEN label={first['label']}  sa={first['sa']:.6f}  "
                        f"center_dist={first['center_dist']:.3f}  "
                        f"min_d={first['min_death_dist']:.3f}\n")
            if second is not None:
                f_dec.write(f"    SECOND: label={second['label']}  sa={second['sa']:.6f}  "
                            f"center_dist={second['center_dist']:.3f}\n")
            f_dec.write("    TOP-K:\n")
            for k_i, t in enumerate(prefs):
                f_dec.write(f"     {k_i:02d}  {t['label']:6d}  {t['sa']:10.6f}  "
                            f"{t['center_dist']:11.3f}  {t['min_death_dist']:13.3f}\n")

        return True

    # ---- PASS 1 ----
    print(f"\n{'#'*30}  PASS 1  {'#'*30}")
    t0 = time.perf_counter()

    keys_to_process = sorted(births.keys())
    births_filtered = []
    for b in keys_to_process:
        if b in processed_ids:
            continue
        bx, by, bz = births[b]
        dist, _ = tree_loops.query([bx - 0.5, by - 0.5, bz - 0.5], k=1)
        if dist < BIRTH_TO_LOOP_MAX_DIST:
            births_filtered.append(b)

    if len(births_filtered) < MIN_FILTERED_TO_KEEP:
        births_filtered = [b for b in keys_to_process if b not in processed_ids]

    processed_now = []
    for idx, b in enumerate(births_filtered, 1):
        if shutting_down["flag"]:
            save_checkpoint(output_path, dedup_by_birth_id(list_of_labeled_birth))
            sys.exit(0)
        ok = process_birth(b)
        if ok:
            processed_now.append(int(b))
        if idx % 50 == 0:
            print(f"  processed {idx}/{len(births_filtered)} ...")

    t1 = time.perf_counter()
    print(f"  computed preferences for {len(processed_now)} births in {t1 - t0:.2f}s")

    if WRITE_REPORTS and f_dec is not None:
        f_dec.close()
        f_dec = None

    if not processed_now:
        print("No births matched any loops. Nothing to output.")
        sys.exit(0)

    # ---- Collision Resolution (Section 1.2.5) ----
    assigned = {b: chosen_label_noninj[b] for b in processed_now}
    collision_lines = ["COLLISIONS REPORT\n"]

    def collisions_summary(amap):
        lab2bs = build_label_to_births(amap)
        return {lab: bs for lab, bs in lab2bs.items()
                if lab != int(UNPAIRED_LABEL_VALUE) and len(bs) > 1}

    conflicts0 = collisions_summary(assigned)
    print(f"  initial collisions: {len(conflicts0)} labels")

    # Step 1: Limited collision resolution (Algorithm 1)
    if RESOLVE_COLLISIONS and conflicts0:
        assigned = resolve_collisions_limited(
            birth_ids=processed_now, pref_lists=pref_lists,
            assigned_label_by_birth=assigned,
            max_passes=COLLISION_MAX_PASSES,
            max_center_delta=COLLISION_MOVE_MAX_CENTER_DELTA,
            sa_floor=SA_FLOOR, require_sa_floor=COLLISION_REQUIRE_SA_FLOOR,
            report_lines=collision_lines, birth_meta=birth_meta,
            centers_dict=centers_dict
        )

    # Step 2: Same-death resolution
    if RESOLVE_SAME_DEATH:
        assigned = resolve_same_death_collisions(
            processed_births=processed_now, assigned_map=assigned,
            birth_meta=birth_meta, pref_lists=pref_lists,
            min_dist_fn=min_dist_to_label_voxels, report_lines=collision_lines,
            ndigits=DEATH_KEY_ROUND, prefer_unused=PREFER_UNUSED_WITHIN_DEATH_GROUP,
            sa_floor=SA_FLOOR, require_sa_for_move=True
        )

    # Step 3: Strict final cleanup
    if STRICT_FINAL_CLEANUP:
        assigned = strict_finalize(
            processed_births=processed_now, assigned_map=assigned,
            pref_lists=pref_lists, report_lines=collision_lines,
            sa_floor=SA_FLOOR
        )

    conflicts_final = collisions_summary(assigned)
    collision_lines.append(f"\n[final] remaining collisions: {len(conflicts_final)}\n")

    if WRITE_REPORTS:
        with open_report(collisions_path) as f_col:
            f_col.write("".join(collision_lines))

    # ---- Write output ----
    existing = {}
    for i, r in enumerate(list_of_labeled_birth):
        if len(r) >= 9:
            try:
                existing[int(float(r[8]))] = i
            except Exception:
                pass

    for b in processed_now:
        meta = birth_meta[b]
        lab = assigned.get(b, chosen_label_noninj[b])
        if lab is None:
            lab = float(UNPAIRED_LABEL_VALUE)
        lab_int = int(lab) if float(lab) != float(UNPAIRED_LABEL_VALUE) else int(UNPAIRED_LABEL_VALUE)

        if lab_int != int(UNPAIRED_LABEL_VALUE) and lab_int in centers_dict:
            center = centers_dict[lab_int]
            cdist = float(np.linalg.norm(center - np.array(meta["death"], dtype=float)))
        else:
            cdist = float("nan")

        new_row = [
            float(meta["birth"][0]), float(meta["birth"][1]), float(meta["birth"][2]),
            float(lab_int), float(cdist),
            float(meta["death"][0]), float(meta["death"][1]), float(meta["death"][2]),
            float(b)
        ]
        if b in existing:
            list_of_labeled_birth[existing[b]] = new_row
        else:
            list_of_labeled_birth.append(new_row)

    save_checkpoint(output_path, dedup_by_birth_id(list_of_labeled_birth))

    t2 = time.perf_counter()
    paired = [b for b in processed_now
              if int(assigned.get(b, UNPAIRED_LABEL_VALUE)) != int(UNPAIRED_LABEL_VALUE)]
    print(f"\n  births processed: {len(processed_now)}, paired: {len(paired)}, "
          f"unpaired: {len(processed_now) - len(paired)}")
    print(f"  total time: {t2 - t0:.2f}s")
    print(f"  output: {output_path}")
    if WRITE_REPORTS:
        print(f"  reports: {decisions_path}, {collisions_path}")

    # ---- Write visualization .poly (death point <-> loop point edges) ----
    vis_poly_path = cycles_dir / "matching_vis.poly"
    point_idx = 1
    points_lines = []
    edges = []

    for b in processed_now:
        lab = int(assigned.get(b, UNPAIRED_LABEL_VALUE))
        if lab == int(UNPAIRED_LABEL_VALUE):
            continue
        dx, dy, dz = deaths[b]
        try:
            loop_pts = get_loop_points(lab)
        except Exception:
            continue
        if loop_pts.shape[0] == 0:
            continue
        # Pick the loop point closest to the death point
        death_pt = np.array([dx, dy, dz], dtype=float)
        dists = np.linalg.norm(loop_pts - death_pt[None, :], axis=1)
        closest_idx = int(np.argmin(dists))
        lx, ly, lz = loop_pts[closest_idx]

        # Death point (red)
        points_lines.append(f"{point_idx}: {dx} {dy} {dz} c(1, 0, 0, 1)")
        death_pidx = point_idx
        point_idx += 1
        # Loop point (green)
        points_lines.append(f"{point_idx}: {lx} {ly} {lz} c(0, 1, 0, 1)")
        loop_pidx = point_idx
        point_idx += 1
        edges.append((death_pidx, loop_pidx))

    with open(vis_poly_path, 'w') as f:
        f.write("POINTS\n")
        for line in points_lines:
            f.write(line + "\n")
        f.write("POLYS\n")
        for j, (a, b_idx) in enumerate(edges, 1):
            f.write(f"{j}: {a} {b_idx}\n")
        f.write("END\n")

    print(f"  matching visualization: {vis_poly_path} ({len(edges)} edges)")


if __name__ == "__main__":
    main()
