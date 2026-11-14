"""Fig 8 -- applicability map: where does staying inside the record keep
experiments safe?  A 2 x 2 map (record clustered vs uniform) x (best condition
interior vs on the boundary).  Each cell: violations in 100 experiments per
algorithm (bar = mean over the cell's configurations, dots = configurations),
the exploratory kNN gate as an extra row in the boundary/clustered cell, and a
one-line performance verdict.

Data: tier1 (interior 12 / boundary 4, 20 seeds), h3_uniform (4, 10 seeds),
h3_uniform_bnd (4, 10 seeds), gate (4 boundary-20 % configurations, 10 seeds).
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
TINT = {"ci": "#eef7ee", "cb": "#fdf3e3", "ui": "#fdf3e3", "ub": "#fbe9e7"}


def load(block):
    return json.load(open(os.path.join(RES, block, "summary.json")))["per_problem"]


def cell_values(P, probs, m):
    if m not in P[probs[0]]["methods"]:
        return None
    return np.array([P[p]["methods"][m]["V_mean"] for p in probs])


def cell_regret(P, probs, m):
    if m not in P[probs[0]]["methods"]:
        return None
    return float(np.mean([P[p]["methods"][m]["regret_mean"] for p in probs]))


VERDICT_COLOR = {"Safe": "#2e8b57", "Mostly safe": "#d9962b", "Safety erodes": "#d9962b", "Not safe": "#c0392b"}


def draw_cell(ax, P, probs, title, verdict, tint, extra=None, xlabel=True):
    rows = list(METHODS)
    if extra:
        rows = ["DALIv2LE", "DALIv2LEgate", "DALIv2", "LEonly", "LEonlygate", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
    ax.set_facecolor(tint)
    rng = np.random.RandomState(0)
    for i, m in enumerate(rows):
        base = m.replace("gate", "")
        if m.endswith("gate"):
            vals = extra[m]
            ax.barh(i, vals.mean(), height=0.66, facecolor="white", edgecolor=FS.COLORS[base], lw=1.6, hatch="....", zorder=2)
        else:
            vals = cell_values(P, probs, m)
            if vals is None:
                ax.text(0.0, i, "not run", va="center", ha="left", fontsize=13.5, color="0.6", style="italic")
                continue
            hollow = m == "DALIv2"
            ax.barh(i, vals.mean(), height=0.66, facecolor="white" if hollow else FS.COLORS[m],
                    edgecolor=FS.COLORS[m] if hollow else "black", lw=1.6 if hollow else 0.6,
                    hatch="////" if hollow else None, zorder=2)
        jit = rng.uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(np.maximum(vals, 0), i + jit, s=20, color="black", alpha=0.55, zorder=3, lw=0)
        ax.text(max(vals.max(), vals.mean()) * 1.18 + 0.15, i, f"{vals.mean():.1f}" if vals.mean() < 10 else f"{vals.mean():.0f}",
                va="center", ha="left", fontsize=15, fontweight="bold" if m in ("DALIv2LE", "LEonly", "SCBO") else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([("+ kNN gate" if m.endswith("gate") else FS.LABELS[m]) for m in rows], fontsize=16)
    for t, m in zip(ax.get_yticklabels(), rows):
        if m.endswith("gate"):
            t.set_color(FS.COLORS[m.replace("gate", "")])
            t.set_fontsize(14)
            t.set_style("italic")
        elif cell_values(P, probs, m) is None:
            t.set_color("0.6")
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.55)
    ax.set_xlim(-0.2, 260)
    ax.set_xticks([0, 1, 3, 10, 30, 100])
    ax.set_xticklabels(["0", "1", "3", "10", "30", "100"], fontsize=15)
    ax.minorticks_off()
    if xlabel:
        ax.set_xlabel("Violations in 100 experiments", fontsize=17)
    FS.clean_axes(ax, xbound=(0, 100))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # verdict chip + two-line summary above the cell
    key = title.split(" —")[0]
    ax.text(0.0, 1.12, f"  {title}  ", transform=ax.transAxes, ha="left", va="bottom", fontsize=18, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=VERDICT_COLOR[key], ec="none"))
    ax.text(0.0, 1.025, verdict, transform=ax.transAxes, ha="left", va="bottom", fontsize=14.5, color="0.25", linespacing=1.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stem", default="Fig8_applicability_map")
    args = ap.parse_args()
    T1, U, UB, G = load("tier1"), load("h3_uniform"), load("h3_uniform_bnd"), load("gate")
    ci = [p for p in T1 if p.endswith("_int")]
    cb = [p for p in T1 if p.endswith("_bnd")]
    ui, ub = list(U), list(UB)
    gate_probs = [f"{g}_f020_bnd" for g in ("g1", "g2", "g3", "g4")]
    extra = {"DALIv2LEgate": np.array([G[p]["methods"]["DALIv2LEgate"]["V_mean"] for p in gate_probs]),
             "LEonlygate": np.array([G[p]["methods"]["LEonlygate"]["V_mean"] for p in gate_probs])}

    def r_line(P, probs):
        r = {m: cell_regret(P, probs, m) for m in METHODS if cell_regret(P, probs, m) is not None}
        return r

    FS.main_rc()
    fig = plt.figure(figsize=(20.0, 17.5))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.50, left=0.15, right=0.965, top=0.855, bottom=0.075)
    cells = [
        (0, 0, T1, ci, "ci", "Safe",
         "In-distribution ≈ 0 · constraint-learning ≈ 2 · constraint-unaware 8–79\nOptimum reached by every in-distribution method, SCBO and TuRBO (r ≤ 0.002)", None),
        (0, 1, U, ui, "ui", "Safety erodes",
         "In-distribution 7–24 · SCBO 8 · TuRBO 25\nNo method reaches the optimum (r = 0.35–0.71, even with the true constraint)", None),
        (1, 0, T1, cb, "cb", "Mostly safe — one failure mode",
         "In-distribution 3–8 except non-convex G2: DALI+LE 54, DALI 88, LE-only 6\nkNN gate: G2 54 → 13 with no loss of performance", extra),
        (1, 1, UB, ub, "ub", "Not safe",
         "In-distribution 9–34 · SCBO 11 · TuRBO 28 · SCBO safest in 3 of 4 geometries\nNo method reaches the optimum (r = 0.42–0.72)", None),
    ]
    for r, c, P, probs, key, title, verdict, ex in cells:
        ax = fig.add_subplot(gs[r, c])
        draw_cell(ax, P, probs, title, verdict, TINT[key], extra=ex, xlabel=(r == 1))
    # row / column headers
    fig.text(0.31, 0.945, "Record clustered around the best condition", ha="center", va="center", fontsize=21, fontweight="bold")
    fig.text(0.31, 0.918, "Tier 1: 85 % spread + 15 % near the best · 20 seeds", ha="center", va="center", fontsize=15, color="0.35")
    fig.text(0.80, 0.945, "Record spread uniformly over the feasible set", ha="center", va="center", fontsize=21, fontweight="bold")
    fig.text(0.80, 0.918, "exploratory · 10 seeds", ha="center", va="center", fontsize=15, color="0.35")
    fig.text(0.025, 0.665, "Best condition\ninterior", ha="center", va="center", fontsize=21, fontweight="bold", rotation=90, linespacing=1.2)
    fig.text(0.05, 0.665, "12 configurations (4 geometries × 50 / 20 / 5 %)", ha="center", va="center", fontsize=14, color="0.35", rotation=90)
    fig.text(0.025, 0.245, "Best condition\non the boundary", ha="center", va="center", fontsize=21, fontweight="bold", rotation=90, linespacing=1.2)
    fig.text(0.05, 0.245, "4 configurations (4 geometries × 20 %)", ha="center", va="center", fontsize=14, color="0.35", rotation=90)
    fig.text(0.5, 0.012, "Bar = mean over the cell's configurations, dots = configurations; hatched blue = DALI (no LE), dotted = exploratory kNN gate",
             ha="center", va="bottom", fontsize=14, color="0.35")
    FS.savefig(fig, args.figdir, args.stem, dpi=args.dpi)
    # numbers used in the text
    for name, P, probs in (("clustered/interior", T1, ci), ("clustered/boundary", T1, cb), ("uniform/interior", U, ui), ("uniform/boundary", UB, ub)):
        print(name, {m: (round(float(cell_values(P, probs, m).mean()), 1), round(cell_regret(P, probs, m), 3))
                     for m in METHODS if cell_values(P, probs, m) is not None})


if __name__ == "__main__":
    main()
