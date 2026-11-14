"""Fig 3 v2 (2026-09-28) -- data extraction for the editable PowerPoint version.

Writes figures/Fig3_v2_data.json with exactly the numbers drawn in Fig 3 v1
(make_fig3_frontier.py): for the 16-configuration set and the four condition
subsets (interior 50 / 20 / 5 %, near-boundary 20 %), the per-method mean V and
mean feasible regret, 95 % bootstrap CIs over seeds (B = 5000, RandomState(0),
same call order as v1), and the non-dominated frontier.
The PowerPoint itself is drawn by make_fig3_v2_pptx.js from this file.

Usage: python make_fig3_v2_data.py [--summary PATH] [--out PATH]
"""

import argparse
import json
import os

import numpy as np

import figstyle as FS
from make_fig3_frontier import METHODS, SEED, boot_mean, frontier, seed_matrix

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PANELS = [("all16", lambda p: True), ("int50", lambda p: "_f050_int" in p), ("int20", lambda p: "_f020_int" in p),
          ("int05", lambda p: "_f005_int" in p), ("bnd20", lambda p: "_f020_bnd" in p)]


def panel(S, probs, methods):
    rng = np.random.RandomState(SEED)
    pts, rows = [], {}
    for m in methods:
        Vm = seed_matrix(S, probs, m, "V_per_seed")
        Rm = seed_matrix(S, probs, m, "regret_per_seed")
        v0, r0 = float(Vm.mean()), float(Rm.mean())
        bv, br = boot_mean(Vm, Rm, rng)
        vlo, vhi = np.percentile(bv, [2.5, 97.5])
        rlo, rhi = np.percentile(br, [2.5, 97.5])
        rows[m] = {"V": v0, "r": r0, "V_lo": float(vlo), "V_hi": float(vhi), "r_lo": float(rlo), "r_hi": float(rhi),
                   "n_seeds": int(Vm.shape[0]), "n_configs": len(probs)}
        pts.append((v0, r0, m))
    fr = frontier(pts)
    return {"configs": probs, "methods": rows, "frontier": [{"V": v, "r": r, "method": m} for v, r, m in fr]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "tier1", "summary.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(CODE_DIR), "figures", "Fig3_v2_data.json"))
    args = ap.parse_args()
    S = json.load(open(args.summary))
    probs = list(S["per_problem"])
    methods = [m for m in METHODS if all(m in S["per_problem"][p]["methods"] for p in probs)]
    out = {"source": os.path.relpath(args.summary, os.path.dirname(CODE_DIR)),
           "methods": [{"key": m, "label": FS.LABELS[m]} for m in methods],
           "bootstrap": {"B": 5000, "seed": SEED},
           "panels": {key: panel(S, [p for p in probs if sel(p)], methods) for key, sel in PANELS}}
    json.dump(out, open(args.out, "w"), indent=1)
    for key, P in out["panels"].items():
        print(key, {m: (round(d["V"], 2), round(d["r"], 4)) for m, d in P["methods"].items()})


if __name__ == "__main__":
    main()
