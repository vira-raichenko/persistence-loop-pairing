# Loop Basis Extraction, Pairing, Threading and Classification

Code for the loop-analysis pipeline used in:

> **Topological Analysis of Multi-Network Architecture in the Pancreas**
> Raichenko, Maaruf, Nyeng, Evans (2026)
> https://www.biorxiv.org/content/10.64898/2026.03.02.708973v1

Given the **skeleton** of a network and its **persistence diagram**, this code
extracts the geometric loops of the network, pairs them with the persistence
birth/death points, and classifies each loop as *threaded* or *not threaded*
using the persistence information of the individual networks and their union.

## Overview

The pipeline runs in four stages:

```
skeletons + persistence diagrams
      │  1. basis extraction      (extract_basis.py)
      ▼
loop cycles  net_cycle_<i>.poly
      │  2. pairing               (pairing.py)
      ▼
birth → loop matching  labeled_birth_loop_local.npy
      │  3. threading separation  (separate_diagrams.py)
      ▼
threaded vs not-changed diagrams
      │  4. classification        (classify_loops.py)
      ▼
threaded / not-threaded loops + per-loop summary
```

1. **Basis extraction** — reads a network skeleton and computes a minimum cycle
   basis, i.e. the set of independent geometric loops of the network. Each loop
   is written to its own `.poly` file.
2. **Pairing** — matches every persistence birth/death pair to the loop it
   represents.
3. **Threading separation** — compares a single network's persistence diagram
   against the diagram of the **union** of two networks. A loop that exists in
   the single network but disappears in the union is *threaded* (it was
   destroyed by threading with the other network); a loop that survives is *not
   changed*.
4. **Classification** — labels each loop as *threaded* or *not threaded* and
   writes the loop geometry plus a per-loop summary (death value, length, class).

## Persistence diagrams (input)

The persistence diagrams are precomputed from the voxelised network images with
[**diamorse**](https://github.com/AppliedMathematicsANU/diamorse). For each
sample you need:

- one diagram **per individual network** (e.g. `C1_bd.csv`, `C2_bd.csv`), and
- one diagram for **the union of the two networks** (e.g. `C1_2_bd.csv`).

The union diagram is what stage 3 compares each single network against to decide
which loops are threaded.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ with `numpy`, `scipy` and `networkx`.

## Usage

Place your inputs under a `networks/` folder:

```
networks/
  skeletons/              <name>_best.poly        one skeleton per network
  persistence_diagrams/   <name>_bd.csv           one diagram per network
                          C<i>_<j>_bd.csv          + the union diagram(s)
```

Run the whole pipeline:

```bash
python full_pipeline.py
```

or run any stage on its own:

```bash
python extract_basis.py
python pairing.py            --cycles-dir networks/cycles/C1 \
                             --persistence networks/persistence_diagrams/C1_bd.csv \
                             --results-dir networks/pairing_results/C1
python separate_diagrams.py
python classify_loops.py
```

## Output

All outputs are written back under `networks/`:

| Folder | Contents |
|--------|----------|
| `cycles/<name>/` | **All the loops** of the network — the minimum cycle basis. `net_cycle_<i>.poly` is one loop each, `all_cycles.poly` is every loop combined. |
| `pairing_results/<name>/` | The birth→loop matching (`labeled_birth_loop_local.npy`), a visualisation (`matching_vis.poly`) and decision/collision reports. |
| `threading/<name>/` | The persistence diagram split into threaded vs not-changed (`<name>_Hb/Hd/Hbd*.csv`). |
| `classified/<name>/` | **Classified loops** based on the persistence information: `Threaded_<name>.poly` and `Not_<name>.poly` (loop geometry coloured by label), the threaded/rest label lists, and `loops_summary_<name>.csv` (per loop: label, death value, length, representative death, class). |

## Input formats

### Skeleton `.poly`

```
POINTS
1: x y z c(...)
...
POLYS
1: a b
...
END
```

`POINTS` are the skeleton vertices; `POLYS` are edges between vertex indices.
The optional `c(...)` colour field is ignored.

### Persistence diagram CSV

```
birth,death,x_b,y_b,z_b,x_d,y_d,z_d
1.6,27.2,58.8,431.3,276.1,122.5,310.9,204.5
```

`birth`/`death` are the scalar persistence values; `x_b,y_b,z_b` and
`x_d,y_d,z_d` are the 3D coordinates of the birth and death critical points.

## License

MIT
