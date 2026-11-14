"""Fig 8 v2 (2026-09-29) -- data extraction for the editable PowerPoint version.

Same numbers as Fig 8 v1 (make_fig8_map.py): per cell of the 2 x 2 map (record clustered /
uniform) x (optimum interior / near boundary), the per-configuration violations of each
method, the cell mean, and the mean regret; the kNN-gate variants for the clustered /
near-boundary cell.  Writes figures/Fig8_v2_data.json.  The figure itself is drawn natively
by make_fig8_v2_pptx.js.

Usage: python make_fig8_v2_data.py [--res DIR] [--out PATH]
"""

import argparse
import json
import os

import numpy as np

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
CLASS_COLOR = {"DALIv2LE": "#2E75B6", "DALIv2": "#2E75B6", "LEonly": "#2E75B6", "SCBO": "#1E8C8C",
               "TuRBO": "#7B4FA6", "GA": "#7B4FA6", "BO": "#7B4FA6", "Random": "#7B4FA6", "cTuRBOknown": "#8A6D00"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(CODE_DIR), "figures", "Fig8_v2_data.json"))
    args = ap.parse_args()
    load = lambda b: json.load(open(os.path.join(args.res, b, "summary.json")))["per_problem"]
    T1, U, UB, G = load("tier1"), load("h3_uniform"), load("h3_uniform_bnd"), load("gate")
    cells = {"ci": (T1, [p for p in T1 if p.endswith("_int")]), "ui": (U, list(U)),
             "cb": (T1, [p for p in T1 if p.endswith("_bnd")]), "ub": (UB, list(UB))}
    rng = np.random.RandomState(0)
    out = {"methods": [{"key": m, "label": FS.LABELS[m], "color": CLASS_COLOR[m]} for m in METHODS], "cells": {}}
    for key, (P, probs) in cells.items():
        cell = {"configs": probs, "n_seeds": int(P[probs[0]]["methods"]["DALIv2LE"]["n_seeds"]), "methods": {}}
        for m in METHODS:
            if m not in P[probs[0]]["methods"]:
                cell["methods"][m] = None
                continue
            vals = [float(P[p]["methods"][m]["V_mean"]) for p in probs]
            reg = [float(P[p]["methods"][m]["regret_mean"]) for p in probs]
            cell["methods"][m] = {"V_per_config": vals, "V_mean": float(np.mean(vals)), "regret_mean": float(np.mean(reg)),
                                  "jitter": [float(j) for j in rng.uniform(-0.2, 0.2, size=len(vals))]}
        out["cells"][key] = cell
    gate_probs = [f"{g}_f020_bnd" for g in ("g1", "g2", "g3", "g4")]
    out["gate"] = {}
    for m in ("DALIv2LEgate", "LEonlygate"):
        vals = [float(G[p]["methods"][m]["V_mean"]) for p in gate_probs]
        out["gate"][m] = {"base": m.replace("gate", ""), "V_per_config": vals, "V_mean": float(np.mean(vals)),
                          "jitter": [float(j) for j in rng.uniform(-0.2, 0.2, size=len(vals))], "configs": gate_probs}
    json.dump(out, open(args.out, "w"), indent=1)
    for key, c in out["cells"].items():
        print(key, {m: (round(d["V_mean"], 1) if d else None) for m, d in c["methods"].items()})
    print("gate", {m: round(d["V_mean"], 1) for m, d in out["gate"].items()})


if __name__ == "__main__":
    main()
