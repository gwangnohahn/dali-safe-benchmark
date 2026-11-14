"""Fig 6 -- where compliance breaks (H4) and where it does not.

(a) optimum in the interior vs on the boundary of the feasible set (ratio 0.20,
    four geometries): cumulative violations V per method, paired bars.
(b) boundary configurations for every feasible ratio (0.50 / 0.20 / 0.05), one
    panel per geometry: V vs ratio (Tier 1 + Tier 2).  G2 x boundary is the one
    case in which DALI fails and the LE refiner only halves the damage.
(c) warm-start contamination 0 / 10 / 30 % infeasible points (ratio 0.20,
    interior): V averaged over the four geometries, markers = geometries.
(d) warm start 200 -> 50 points (G2, G3, ratio 0.20, interior): V and feasible
    regret before/after.
(e) exploratory (not pre-registered): uniform warm start instead of the
    record clustered around c0 (ratio 0.20, interior, 10 seeds): V and regret.

Data: tier1, tier2, h4_contam, h4_warm, h3_uniform summary.json.
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")
GEOMS = ["g1", "g2", "g3", "g4"]
GEOM_LABEL = {"g1": "G1 convex", "g2": "G2 non-convex", "g3": "G3 disconnected", "g4": "G4 thin band"}
GEOM_MARK = {"g1": "o", "g2": "s", "g3": "^", "g4": "D"}
M_A = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA"]
M_ALL = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
M_C = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO"]


def load(block):
    return json.load(open(os.path.join(RES, block, "summary.json")))["per_problem"]


def stat(P, p, m, key):
    return P[p]["methods"][m][key]


def bar_kw(m, hollow=False):
    """DALI (no LE) is hatched so that it differs from DALI+LE (same hue)."""
    kw = dict(facecolor="white" if hollow else FS.COLORS[m], edgecolor=FS.COLORS[m] if hollow else "black",
              lw=1.8 if hollow else 0.6)
    if m == "DALIv2":
        kw.update(hatch="////", edgecolor=FS.COLORS[m] if hollow else "white", lw=1.8 if hollow else 0.0)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stem", default="Fig6_limits")
    args = ap.parse_args()
    T1, T2, C, W, U = load("tier1"), load("tier2"), load("h4_contam"), load("h4_warm"), load("h3_uniform")

    FS.main_rc()
    fig = plt.figure(figsize=(22.0, 19.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.85, 1.0], hspace=0.60, wspace=0.42,
                          left=0.07, right=0.975, top=0.955, bottom=0.13)
    gs0 = gs[0, :].subgridspec(1, 4, width_ratios=[2.1, 0.95, 0.7, 0.7], wspace=0.45)

    def dumbbell_panel(ax, groups, methods, get_ref, get_new, ls_of=lambda m: "-"):
        nm = len(methods)
        w = 0.82 / nm
        for gi, g in enumerate(groups):
            for k, m in enumerate(methods):
                xx = gi + (k - (nm - 1) / 2) * w
                FS.dumbbell(ax, xx, get_ref(g, m), get_new(g, m), m, ls=ls_of(m))
        ax.set_xticks(range(len(groups)))
        ax.set_xlim(-0.5, len(groups) - 0.5)
        for gi in range(1, len(groups)):
            ax.axvline(gi - 0.5, color="0.85", lw=0.8, zorder=0)

    dali_ls = lambda m: "--" if m == "DALIv2" else "-"      # DALI (no LE) dashed: same hue as DALI+LE

    # ------------------------------------------------------------ (a) interior -> boundary
    ax = fig.add_subplot(gs0[0, 0])
    dumbbell_panel(ax, GEOMS, M_A,
                   lambda g, m: stat(T1, f"{g}_f020_int", m, "V_mean"),
                   lambda g, m: stat(T1, f"{g}_f020_bnd", m, "V_mean"), dali_ls)
    ax.set_xticklabels([GEOM_LABEL[g].replace(" ", "\n", 1) for g in GEOMS], fontsize=16)
    ax.set_yscale("symlog", linthresh=10, linscale=0.9)
    ax.set_ylim(-0.6, 130)
    ax.set_yticks([0, 5, 10, 30, 100])
    ax.set_yticklabels(["0", "5", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_ylabel("Violations in\n100 experiments")
    FS.subtitle(ax, "Feasible fraction 20 %\n○ optimum interior  →  ● optimum on the boundary")
    FS.clean_axes(ax)
    FS.panel_label(ax, "a", dx=-0.16, dy=1.10)

    # ------------------------------------------------------------ (b) contamination
    ax = fig.add_subplot(gs0[0, 1])
    levels = [0, 10, 30]
    for m in M_C:
        vals = np.array([[stat(T1 if c == 0 else C, f"{g}_f020_int" + ("" if c == 0 else f"_c{c}"), m, "V_mean")
                          for c in levels] for g in GEOMS])
        kw = FS.style_line(m)
        for k in ("marker", "markevery", "ms", "markerfacecolor", "markeredgecolor", "markeredgewidth"):
            kw.pop(k, None)
        ax.plot(levels, vals.mean(axis=0), **kw)
        for gi, g in enumerate(GEOMS):
            ax.scatter(levels, vals[gi], marker=GEOM_MARK[g], s=48, color=FS.COLORS[m], alpha=0.6, zorder=4)
    ax.set_xticks(levels)
    ax.set_xticklabels(["0", "10", "30"])
    ax.set_xlabel("Contamination (%)")
    ax.set_ylabel("Violations in\n100 experiments")
    ax.set_ylim(-0.5, 16)
    ax.set_yticks([0, 4, 8, 12, 16])
    FS.subtitle(ax, "Contaminated record\ninterior 20 %; markers: geometries")
    FS.clean_axes(ax)
    FS.panel_label(ax, "b", dx=-0.30, dy=1.10)

    # ------------------------------------------------------------ (c) record 200 -> 50: V and r
    items = ["g2", "g3"]
    ms_c = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO"]
    for j, (key, ylab, ylim, yt) in enumerate((("V_mean", "Violations in\n100 experiments", (-0.3, 8), [0, 2, 4, 6, 8]),
                                              ("regret_mean", "Feasible regret, r", (-0.004, 0.12), [0, 0.04, 0.08, 0.12]))):
        ax = fig.add_subplot(gs0[0, 2 + j])
        dumbbell_panel(ax, items, ms_c,
                       lambda g, m: stat(T1, f"{g}_f020_int", m, key),
                       lambda g, m: stat(W, f"{g}_f020_int_w50", m, key), dali_ls)
        ax.set_xticklabels(["G2", "G3"])
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        FS.subtitle(ax, ("Record 200 → 50 points\ninterior 20 %") if j == 0 else "\n")
        FS.clean_axes(ax)
        if j == 0:
            FS.panel_label(ax, "c", dx=-0.45, dy=1.10)

    # ------------------------------------------------------------ (d) boundary x feasible fraction
    ratios = [0.50, 0.20, 0.05]
    for gi, g in enumerate(GEOMS):
        ax = fig.add_subplot(gs[1, gi])
        ends, txts, cols = [], [], []
        for m in M_ALL:
            vals = [stat(T1 if r == 0.20 else T2, f"{g}_f{int(r * 100):03d}_bnd", m, "V_mean") for r in ratios]
            kw = FS.style_line(m)
            for k in ("marker", "markevery", "ms", "markerfacecolor", "markeredgecolor", "markeredgewidth", "label"):
                kw.pop(k, None)
            kw["lw"] = {"DALIv2LE": 3.4, "DALIv2": 2.2}.get(m, 2.0)
            if m == "DALIv2":
                kw["ls"] = "--"
            ax.plot(range(3), vals, **kw)
            ends.append(vals[-1])
            txts.append(FS.LABELS[m])
            cols.append(FS.COLORS[m])
        FS.end_labels(ax, 2, ends, txts, cols, min_sep=7.5, x_text=2.12, y_max=100, y_min=0, fs=15)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["50", "20", "5"])
        ax.set_xlim(-0.15, 3.3)
        ax.set_xlabel("Feasible fraction of the box (%)")
        ax.set_ylim(-2, 100)
        geom = GEOM_LABEL[g].replace(" ", " · ", 1)
        FS.subtitle(ax, ("Boundary, feasible fraction 50 / 20 / 5 % (Tier 1 + Tier 2)\n" + geom) if gi == 0 else ("\n" + geom))
        FS.clean_axes(ax, xbound=(0, 2))
        if gi == 0:
            ax.set_ylabel("Violations in\n100 experiments")
            FS.panel_label(ax, "d", dx=-0.34, dy=1.09)

    # ------------------------------------------------------------ (e)(f) clustered -> uniform record
    ms_e = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "cTuRBOknown"]
    for col, (key, ylab, ylim, yt) in enumerate((("V_mean", "Violations in\n100 experiments", (-1, 45), [0, 10, 20, 30, 40]),
                                                ("regret_mean", "Feasible regret, r", (-0.02, 0.9), [0, 0.2, 0.4, 0.6, 0.8]))):
        ax = fig.add_subplot(gs[2, 2 * col:2 * col + 2])
        dumbbell_panel(ax, GEOMS, ms_e,
                       lambda g, m: stat(T1, f"{g}_f020_int", m, key),
                       lambda g, m: stat(U, f"{g}_f020_int_uniform", m, key), dali_ls)
        ax.set_xticklabels([GEOM_LABEL[g] for g in GEOMS])
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        FS.subtitle(ax, "Exploratory, interior 20 %\n○ record clustered around the optimum  →  ● uniform record")
        ax.set_facecolor("#f7f7f7")
        FS.clean_axes(ax)
        FS.panel_label(ax, "e" if col == 0 else "f", dx=-0.13, dy=1.10)

    # ------------------------------------------------------------ shared legend
    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, markerfacecolor=FS.COLORS[m], markeredgecolor="black",
                          markeredgewidth=0.6, label=FS.LABELS[m]) for m in M_ALL]
    handles[1] = plt.Line2D([], [], marker="o", ls="--", lw=2, color=FS.COLORS["DALIv2"], ms=11,
                            markerfacecolor=FS.COLORS["DALIv2"], markeredgecolor="black", markeredgewidth=0.6,
                            label="DALI (dashed connector / line)")
    fig.legend(handles=handles, loc="lower center", ncol=9, bbox_to_anchor=(0.5, 0.045), frameon=False,
               fontsize=17, columnspacing=1.2, handletextpad=0.5)
    enc = [plt.Line2D([], [], marker="o", ls="", ms=12, markerfacecolor="white", markeredgecolor="0.3",
                      markeredgewidth=2.2, label="reference condition (○)"),
           plt.Line2D([], [], marker="o", ls="", ms=9, markerfacecolor="0.3", markeredgecolor="black",
                      markeredgewidth=0.6, label="changed condition (●, arrow head)")]
    enc += [plt.Line2D([], [], marker=GEOM_MARK[g], ls="", ms=9, color="0.3", label=GEOM_LABEL[g] + " (b)")
            for g in GEOMS]
    fig.legend(handles=enc, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), frameon=False,
               fontsize=17, columnspacing=1.2, handletextpad=0.5)
    FS.savefig(fig, args.figdir, args.stem, dpi=args.dpi)


if __name__ == "__main__":
    main()
