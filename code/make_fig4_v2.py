"""Fig 4 v2 (2026-09-28) -- data + plot-area images for the editable PowerPoint version.

Same numbers as Fig 4 v1 (make_fig4_survival.py): pooled Kaplan-Meier curves with
Greenwood 95 % CI (a), cumulative-violation mean curves with 95 % CI (b), and the
violations within the first 20 experiments per configuration (c); main block,
16 configurations x 20 seeds = 320 runs per method.

This script writes
  figures/Fig4_v2_data.json        numbers + the axes geometry of the panel images
  figures/Fig4_v2_panel_a.png      plot area of (a): curves, bands, ticks (no titles / labels)
  figures/Fig4_v2_panel_b.png      plot area of (b)
Everything else (panel titles, axis titles, direct labels, panel (c) bars/dots, footnote)
is drawn as editable shapes by make_fig4_v2_pptx.js.

Style: four class colours (Record-based / Constraint learning / Constraint-unaware /
Reference) as in Fig 1-3 v2; methods within a class differ by line style.

Usage: python make_fig4_v2.py [--summary PATH] [--figdir DIR]
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as FS
from make_fig4_survival import METHODS, km_greenwood

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_COLOR = {"DALIv2LE": "#2E75B6", "DALIv2": "#2E75B6", "LEonly": "#2E75B6", "SCBO": "#1E8C8C",
               "TuRBO": "#7B4FA6", "GA": "#7B4FA6", "BO": "#7B4FA6", "Random": "#7B4FA6", "cTuRBOknown": "#8A6D00"}
LINE = {"DALIv2LE": ("-", 2.4), "DALIv2": ("--", 1.8), "LEonly": (":", 2.0), "SCBO": ("-", 2.0),
        "TuRBO": ("-", 2.0), "GA": ("--", 1.8), "BO": (":", 2.0), "Random": ("-.", 1.8), "cTuRBOknown": (":", 1.4)}
LINE_NAME = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot"}
# axes geometry inside each panel image (inches): figure size and axes rectangle
GEOM = {"a": {"fig": (4.75, 3.55), "ax": (0.55, 0.38, 4.10, 3.10)},
        "b": {"fig": (4.75, 3.55), "ax": (0.55, 0.38, 4.10, 3.10)}}


def panel_axes(key):
    fw, fh = GEOM[key]["fig"]
    x0, y0, w, h = GEOM[key]["ax"]
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#808080")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="#595959", labelsize=8, width=0.8, length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily("DejaVu Sans")
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "tier1", "summary.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    args = ap.parse_args()
    S = json.load(open(args.summary))
    probs = list(S["per_problem"])
    methods = [m for m in METHODS if all(m in S["per_problem"][p]["methods"] for p in probs)]
    t_max = 100
    out = {"source": os.path.relpath(args.summary, os.path.dirname(CODE_DIR)), "n_configs": len(probs),
           "methods": [{"key": m, "label": FS.LABELS[m], "color": CLASS_COLOR[m], "line": LINE_NAME[LINE[m][0]]} for m in methods],
           "geom": GEOM, "a": {}, "b": {}, "c": {}}

    # ---------------- (a) pooled KM
    fig, ax = panel_axes("a")
    for m in methods:
        T, E = [], []
        for p in probs:
            r = S["per_problem"][p]["methods"][m]
            T += r["T1_per_seed"]
            E += r["T1_event_per_seed"]
        s, lo, hi = km_greenwood(np.array(T), np.array(E, bool), t_max)
        ls, lw = LINE[m]
        col = CLASS_COLOR[m] if m != "cTuRBOknown" else "black"
        ax.step(np.arange(0, t_max + 1), [1.0] + s, where="post", color=col, ls=ls, lw=lw, zorder=3 if m == "DALIv2LE" else 2)
        if m != "cTuRBOknown":
            ax.fill_between(np.arange(0, t_max + 1), [1.0] + lo, [1.0] + hi, step="post", color=CLASS_COLOR[m], alpha=0.10, lw=0)
        out["a"][m] = {"S_end": float(s[-1]), "n_runs": len(T), "S": [float(v) for v in s]}
    ax.set_xlim(0, t_max); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    out["a"]["xlim"], out["a"]["ylim"] = [0, t_max], [0, 1.02]
    fig.savefig(os.path.join(args.figdir, "Fig4_v2_panel_a.png"), dpi=400)
    plt.close(fig)

    # ---------------- (b) cumulative violations
    fig, ax = panel_axes("b")
    cyc = np.arange(1, 21) * 5
    for m in methods:
        M = np.array([S["per_problem"][p]["methods"][m]["V_curve_mean"] for p in probs])
        SD = np.array([S["per_problem"][p]["methods"][m]["V_curve_sd"] or [0] * 20 for p in probs])
        n = np.array([S["per_problem"][p]["methods"][m]["n_seeds"] for p in probs])
        mu = M.mean(axis=0)
        se = np.sqrt((SD ** 2 / n[:, None]).sum(axis=0)) / len(probs)
        ls, lw = LINE[m]
        col = CLASS_COLOR[m] if m != "cTuRBOknown" else "black"
        ax.plot(np.r_[0, cyc], np.r_[0, mu], color=col, ls=ls, lw=lw, zorder=3 if m == "DALIv2LE" else 2)
        if m != "cTuRBOknown":
            ax.fill_between(cyc, mu - 1.96 * se, mu + 1.96 * se, color=CLASS_COLOR[m], alpha=0.13, lw=0)
        out["b"][m] = {"V_end": float(mu[-1]), "V_curve": [float(v) for v in mu], "ci_halfwidth_end": float(1.96 * se[-1])}
    ax.set_xlim(0, t_max); ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 20, 40, 60, 80, 100])
    out["b"]["xlim"], out["b"]["ylim"] = [0, t_max], [0, 100]
    fig.savefig(os.path.join(args.figdir, "Fig4_v2_panel_b.png"), dpi=400)
    plt.close(fig)

    # ---------------- (c) first 20 experiments (numbers only; drawn natively in pptx)
    rng = np.random.RandomState(0)
    for m in methods:
        vals = np.array([S["per_problem"][p]["methods"][m]["V_curve_mean"][3] for p in probs])
        jit = rng.uniform(-0.22, 0.22, size=len(vals))
        out["c"][m] = {"mean": float(vals.mean()), "per_config": [float(v) for v in vals], "jitter": [float(j) for j in jit]}

    path = os.path.join(args.figdir, "Fig4_v2_data.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    for m in methods:
        print(f"{FS.LABELS[m]:16s} S_end {out['a'][m]['S_end']:.2f}  V_end {out['b'][m]['V_end']:.1f}  first20 {out['c'][m]['mean']:.1f}")


if __name__ == "__main__":
    main()
