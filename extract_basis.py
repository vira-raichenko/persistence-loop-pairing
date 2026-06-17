#!/usr/bin/env python3
"""
Extract a minimum cycle basis (the set of independent loops) from each network
skeleton and save the loops, one folder per network.

For every ``*.poly`` skeleton in SKELETON_DIR:
  1. Parse it into a NetworkX graph (nodes carry their 3D position).
  2. Reduce the graph while preserving its cycle space (prune leaves, suppress
     degree-2 nodes) and keep a mapping back to the original node paths.
  3. Compute a minimum cycle basis per connected component of the reduced graph.
  4. Expand each basis cycle back to original node positions, ordered so the
     points trace the loo
     p, and write it as a .poly point list.

Outputs (under OUTPUT_ROOT/<network>/):
    net_cycle_<i>.poly   one loop each (point list, ready for pairing.py)
    all_cycles.poly      every loop concatenated (for quick visualization)

where <network> is the skeleton file name with any trailing "_best" removed
(C1_best.poly -> C1).

Usage:
    python extract_basis.py
    python extract_basis.py --skeleton-dir DIR --output-root DIR
"""

import argparse
from collections import deque
from pathlib import Path

import networkx as nx

# ======================== CONFIG ========================
# Defaults are relative to this script so the tool runs from a fresh clone.
# Override either path with the CLI flags below.

SCRIPT_DIR = Path(__file__).resolve().parent

NETWORKS_DIR = SCRIPT_DIR / "networks"
SKELETON_DIR = NETWORKS_DIR / "skeletons"
OUTPUT_ROOT = NETWORKS_DIR / "cycles"

MIN_LOOP_NODES = 3   # ignore degenerate "loops" with fewer than 3 nodes

# ======================== .poly I/O ========================


def read_poly_into_graph(filepath):
    """
    Parse a .poly skeleton into an undirected graph.

    POINTS lines ("idx: x y z c(...)") become nodes with a 'pos' attribute;
    POLYS lines ("idx: a b") become edges between node indices.
    """
    vertices = {}
    edges = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("POINTS", "POLYS", "END")):
                continue
            parts = [p for p in line.split(" ") if p]
            if not parts or not parts[0].endswith(":"):
                continue
            idx = int(parts[0][:-1])
            if len(parts) >= 4:   # vertex: idx + x y z (+ color)
                try:
                    vertices[idx] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    continue
                except ValueError:
                    pass
            if len(parts) >= 3:   # edge: idx + a b
                try:
                    edges.append((int(parts[1]), int(parts[2])))
                except ValueError:
                    pass

    graph = nx.Graph()
    for v, pos in vertices.items():
        graph.add_node(v, pos=pos)
    for a, b in edges:
        if a != b:
            graph.add_edge(a, b)
    return graph


def write_loop_poly(filepath, graph, loop_nodes):
    """Write one ordered loop (list of node IDs) as a .poly point list."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("POINTS\n")
        i = 1
        for v in loop_nodes:
            pos = graph.nodes[v].get("pos")
            if pos is None:
                continue
            x, y, z = pos
            f.write(f"{i}: {x} {y} {z} c(0, 0, 0, 1)\n")
            i += 1
        f.write("POLYS\nEND\n")


def write_all_loops_poly(filepath, graph, loops):
    """Write every loop's points into a single combined .poly point list."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("POINTS\n")
        i = 1
        for loop in loops:
            for v in loop:
                pos = graph.nodes[v].get("pos")
                if pos is None:
                    continue
                x, y, z = pos
                f.write(f"{i}: {x} {y} {z} c(0, 0, 0, 1)\n")
                i += 1
        f.write("POLYS\nEND\n")


# ======================== GRAPH REDUCTION ========================


def _path_lookup(mapping, u, v):
    """Directed original-node path from u to v; default to [u, v] if none."""
    if (u, v) in mapping:
        return mapping[(u, v)]
    if (v, u) in mapping:
        return list(reversed(mapping[(v, u)]))
    return [u, v]


def augment_with_mapping(graph):
    """
    Reduce the graph while preserving its cycle space.

    Returns (reduced_graph, mapping) where mapping[(u, v)] is the original-node
    path that a reduced edge (u, v) stands for, so cycles on the reduced graph
    can be expanded back to the original nodes.
    """
    G = graph.copy()
    mapping = {}

    # (1) Prune leaves (degree <= 1) — they cannot lie on any cycle.
    dq = deque([n for n, d in G.degree() if d <= 1])
    seen = set(dq)
    while dq:
        n = dq.popleft()
        if n not in G:
            continue
        nbrs = list(G.neighbors(n))
        G.remove_node(n)
        for u in nbrs:
            if u in G and G.degree(u) <= 1 and u not in seen:
                dq.append(u)
                seen.add(u)

    # (2) Suppress degree-2 nodes, recording the bypassed path, without
    #     collapsing cycles (skip if the two neighbors are already adjacent).
    while True:
        changed = False
        for v in list(G.nodes()):
            if v not in G or G.degree(v) != 2:
                continue
            u, w = list(G.neighbors(v))
            if u == w:
                G.remove_node(v)
                changed = True
                continue
            if G.has_edge(u, w):
                continue
            uw_path = _path_lookup(mapping, u, v) + _path_lookup(mapping, v, w)[1:]
            G.add_edge(u, w)
            mapping[(u, w)] = uw_path
            mapping[(w, u)] = list(reversed(uw_path))
            G.remove_node(v)
            changed = True
        if not changed:
            break

    return G, mapping


def minimum_cycle_basis_by_component(G):
    """Compute the minimum cycle basis per connected component (for speed)."""
    basis = []
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        if H.number_of_edges() < H.number_of_nodes():
            continue   # trees / isolated nodes carry no cycle
        basis.extend(nx.minimum_cycle_basis(H))
    return basis


def order_cycle_nodes(G, cycle):
    """
    Reorder a minimum-cycle-basis cycle (an UNORDERED node set) into traversal
    order by walking edges of G, so that consecutive nodes are adjacent. Falls
    back to the given order if the induced subgraph is not a clean cycle.
    """
    nodes = set(cycle)
    H = G.subgraph(nodes)
    start = next(iter(nodes))
    order = [start]
    prev, cur = None, start
    while len(order) < len(nodes):
        nbrs = [w for w in H.neighbors(cur) if w in nodes and w != prev and w not in order]
        if not nbrs:
            break
        nxt = nbrs[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order if len(order) == len(nodes) else list(cycle)


def expand_cycle_to_original_nodes(cycle_nodes, mapping):
    """Expand an ordered reduced cycle back to a sequence of original nodes."""
    if not cycle_nodes:
        return []
    expanded = []
    n = len(cycle_nodes)
    for i in range(n):
        u = cycle_nodes[i]
        v = cycle_nodes[(i + 1) % n]
        seg = _path_lookup(mapping, u, v)
        expanded.extend(seg if i == 0 else seg[1:])
    return expanded


# ======================== PER-NETWORK DRIVER ========================


def network_name(skeleton_path):
    """C1_best.poly -> C1 ; otherwise the file stem unchanged."""
    stem = skeleton_path.stem
    return stem[:-5] if stem.endswith("_best") else stem


def extract_network(skeleton_path, out_dir):
    """Extract the loop basis for one skeleton and write loops to out_dir."""
    graph = read_poly_into_graph(skeleton_path)
    if graph.number_of_nodes() == 0:
        print(f"  {skeleton_path.name}: empty, skipped")
        return 0

    reduced, mapping = augment_with_mapping(graph)
    basis = minimum_cycle_basis_by_component(reduced)

    loops = []
    for cyc in basis:
        ordered = order_cycle_nodes(reduced, cyc)
        expanded = expand_cycle_to_original_nodes(ordered, mapping)
        if len(expanded) >= MIN_LOOP_NODES:
            loops.append(expanded)

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("net_cycle_*.poly"):   # clear a previous run
        stale.unlink()
    for i, loop in enumerate(loops):
        write_loop_poly(out_dir / f"net_cycle_{i}.poly", graph, loop)
    write_all_loops_poly(out_dir / "all_cycles.poly", graph, loops)

    print(f"  {skeleton_path.name}: nodes={graph.number_of_nodes()} "
          f"edges={graph.number_of_edges()} -> {len(loops)} loops  ({out_dir})")
    return len(loops)


def main():
    ap = argparse.ArgumentParser(description="Extract a loop basis per skeleton.")
    ap.add_argument("--skeleton-dir", type=Path, default=SKELETON_DIR)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()

    skeleton_files = sorted(args.skeleton_dir.glob("*.poly"))
    if not skeleton_files:
        raise RuntimeError(f"No .poly skeletons in {args.skeleton_dir}")

    print(f"Extracting loop bases for {len(skeleton_files)} network(s):")
    total = 0
    for fp in skeleton_files:
        out_dir = args.output_root / network_name(fp)
        total += extract_network(fp, out_dir)

    print(f"\nDone. {total} loops total under {args.output_root}")


if __name__ == "__main__":
    main()