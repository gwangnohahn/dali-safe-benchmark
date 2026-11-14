"""Fig 2 v8 (2026-09-28) -- data extraction for the editable PowerPoint version.

Writes figures/Fig2_v8_data.json with exactly the numbers drawn in Fig 2 v7
(make_fig2_heatmap.py): per-configuration mean V (16 x 9), Holm-significance
marks against DALI+LE, and the interior / near-boundary mean V per method
with 95 % bootstrap CIs over seeds (B = 5000, RandomState(0), as in v7).
The PowerPoint itself is drawn by make_fig2_v8_pptx.js from this file.

Usage: python make_fig2_v8_data.py [--summary PATH] [--out PATH]
"""

import argparse
import json
import os

import numpy as np

import figstyle as FS
from make_fig2_heatmap import METHODS, config_order, load_grid

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "tier1", "summary.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(CODE_DIR), "figures", "Fig2_v8_data.json"))
    args = ap.parse_args()
    S = json.load(open(args.summary))
    rows = config_order(list(S["per_problem"]))
    cols = [m for m in METHODS if all(m in S["per_problem"][r]["methods"] for r in rows)]
    ref = S["config"]["reference"]
    V, marks = load_grid(S, rows, cols, "V_mean", "V")   # V: configs x methods; marks "*" higher, "+" lower

    rng = np.random.RandomState(0)
    B = 5000
    bars = {}
    for loc in ("int", "bnd"):
        sub = [r for r in rows if r.split("_")[2] == loc]
        bars[loc] = {}
        for m in cols:
            Vm = np.array([S["per_problem"][r]["methods"][m]["V_per_seed"] for r in sub]).T
            n = Vm.shape[0]
            idx = rng.randint(0, n, size=(B, n))
            boot = Vm[idx].mean(axis=(1, 2))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            bars[loc][m] = {"mean": float(Vm.mean()), "lo": float(lo), "hi": float(hi), "n_seeds": int(n), "n_configs": len(sub)}

    out = {
        "source": os.path.relpath(args.summary, os.path.dirname(CODE_DIR)),
        "reference": ref,
        "configs": [{"name": r, "geometry": r.split("_")[0], "fraction_pct": int(r.split("_")[1][1:]),
                     "location": r.split("_")[2]} for r in rows],
        "methods": [{"key": m, "label": FS.LABELS[m], "color": FS.COLORS[m]} for m in cols],
        "V_mean": [[float(v) for v in row] for row in V],          # configs x methods
        "marks": marks,                                            # configs x methods: "*" / "+" / ""
        "bars": bars,
        "bootstrap": {"B": B, "seed": 0},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out, "configs", len(rows), "methods", len(cols))


if __name__ == "__main__":
    main()
