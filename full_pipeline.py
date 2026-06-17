#!/usr/bin/env python3
"""
Run the full loop-analysis pipeline end to end:

    skeletons + persistence diagrams
        |  1. extract_basis.py     -> cycles/<net>/net_cycle_*.poly
        |  2. pairing.py           -> pairing_results/<net>/labeled_birth_loop_local.npy
        |  3. separate_diagrams.py -> threading/<net>/  (Hb/Hd/Hbd + labeled_all.npy)
        |  4. classify_loops.py    -> classified/<net>/ (Threaded/Not + loops_summary)
        v

Inputs live under NETWORKS_DIR:
    <networks>/skeletons/<net>_best.poly        one skeleton per network
    <networks>/persistence_diagrams/<net>_bd.csv  matching diagram(s) + the union

All outputs are written back under <networks>/. Each stage is invoked through its
own CLI, so any stage can still be run on its own.

Usage:
    python full_pipeline.py
    python full_pipeline.py --networks-dir /path/to/networks
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_stage(title, script, *cli_args):
    """Invoke one pipeline stage as a subprocess, streaming its output."""
    print(f"\n{'='*70}\n# {title}\n{'='*70}")
    cmd = [sys.executable, str(SCRIPT_DIR / script), *map(str, cli_args)]
    print("  $ " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run basis -> pairing -> threading -> classify.")
    ap.add_argument("--networks-dir", type=Path, default=SCRIPT_DIR / "networks",
                    help="Folder holding skeletons/ and persistence_diagrams/; outputs go here too.")
    ap.add_argument("--skeleton-dir", type=Path, default=None,
                    help="Override skeleton folder (default: <networks>/skeletons).")
    ap.add_argument("--pd-dir", type=Path, default=None,
                    help="Override diagram folder (default: <networks>/persistence_diagrams).")
    args = ap.parse_args()

    nd = args.networks_dir
    skeleton_dir = args.skeleton_dir or (nd / "skeletons")
    pd_dir = args.pd_dir or (nd / "persistence_diagrams")
    cycles_dir = nd / "cycles"
    pairing_dir = nd / "pairing_results"
    threading_dir = nd / "threading"
    classified_dir = nd / "classified"

    if not skeleton_dir.is_dir():
        raise SystemExit(f"Skeleton folder not found: {skeleton_dir}")
    if not pd_dir.is_dir():
        raise SystemExit(f"Diagram folder not found: {pd_dir}")

    # 1. Basis extraction (batch over all skeletons).
    run_stage("1/4  Basis extraction", "extract_basis.py",
              "--skeleton-dir", skeleton_dir, "--output-root", cycles_dir)

    # 2. Pairing (one run per network; pairing.py is single-network).
    networks = sorted(d.name for d in cycles_dir.iterdir() if d.is_dir())
    if not networks:
        raise SystemExit(f"No loop folders produced under {cycles_dir}")
    print(f"\n{'='*70}\n# 2/4  Pairing  ({len(networks)} network(s): {', '.join(networks)})\n{'='*70}")
    for net in networks:
        diagram = pd_dir / f"{net}_bd.csv"
        if not diagram.exists():
            print(f"  [skip] {net}: no diagram {diagram.name}")
            continue
        # Clear any old checkpoint so pairing recomputes against the fresh loops.
        checkpoint = pairing_dir / net / "labeled_birth_loop_local.npy"
        if checkpoint.exists():
            checkpoint.unlink()
        run_stage(f"2/4  Pairing — {net}", "pairing.py",
                  "--cycles-dir", cycles_dir / net,
                  "--persistence", diagram,
                  "--results-dir", pairing_dir / net)

    # 3. H-threading separation (batch; auto-pairs each single net with its union).
    run_stage("3/4  Diagram separation (H-threading)", "separate_diagrams.py",
              "--pd-dir", pd_dir, "--pairing-dir", pairing_dir, "--out-root", threading_dir)

    # 4. Loop classification (batch).
    run_stage("4/4  Loop classification", "classify_loops.py",
              "--threading-dir", threading_dir, "--cycles-dir", cycles_dir,
              "--pd-dir", pd_dir, "--out-root", classified_dir)

    print(f"\n{'='*70}\nPipeline complete. Results under {nd}:")
    print(f"  cycles/<net>/            loop basis")
    print(f"  pairing_results/<net>/   birth->loop matching")
    print(f"  threading/<net>/         threaded vs not-changed diagrams")
    print(f"  classified/<net>/        Threaded/Not loops + loops_summary")
    print('='*70)


if __name__ == "__main__":
    main()