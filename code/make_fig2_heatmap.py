"""Fig 2 -- constraint suite and main result (paper 2).

(a) heat map of cumulative violations V (mean over seeds) for 16 Tier-1
    configurations x methods, colour only; cells significantly different from
    DALI+LE (Holm-adjusted MWU p < 0.05) are marked (filled: method higher,
    open: method lower), (b) mean V per method, interior vs boundary
    configurations, with 95 % bootstrap CI over seeds.
v7 (2026-09-19): the 2-D sections of the four geometries (former panel a) moved to
Fig 1b; they are still written as FigS_suite_sections_f020 for the SI.
Also writes the SI figures FigS_tier1_V_heatmap (same as b with the numbers),
FigS_tier1_regret_heatmap (feasible regret) and FigS_tier2_heatmap (Tier 2).

Data: results/constraint_suite/tier1/summary.json (analyze_constraint_suite.py).
Usage: python make_fig2_heatmap.py [--summary PATH] [--figdir ../figures] [--dpi 300]
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import figstyle as FS
import problems_constraint_suite as CS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
GEOM_LABEL = {"g1": "G1 convex", "g2": "G2 non-convex", "g3": "G3 disconnected", "g4": "G4 thin band"}


def config_order(names):
    def key(n):
        c = CS.parse_name(n) if hasattr(CS, "parse_name") else None
        g, r, loc = n.split("_")[:3]
        return (g, 0 if loc == "int" else 1, -int(r[1:]))
    return sorted(names, key=key)


def row_label(n):
    g, r, loc = n.split("_")[:3]
    return f"{g.upper()}  {int(r[1:]) / 100:.2f}  {'interior' if loc == 'int' else 'boundary'}"


def heat(ax, M, ann, marks, cmap, vmin, vmax, rows, cols, cbar_label, fmt, log=False, numbers=True,
         fontsize=10.5):
    norm = LogNorm(vmin=max(vmin, 1e-4), vmax=vmax) if log else None
    im = ax.imshow(M, cmap=cmap, vmin=None if log else vmin, vmax=None if log else vmax,
                   norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = (np.log(v + 1e-4) - np.log(max(vmin, 1e-4))) / (np.log(vmax) - np.log(max(vmin, 1e-4))) > 0.6 if log \
                else (v - vmin) / (vmax - vmin) > 0.6
            if numbers:
                ax.text(j, i, fmt(v) + marks[i][j], ha="center", va="center", fontsize=fontsize,
                        color="white" if dark else "black")
            elif marks[i][j]:
                ax.text(j, i, marks[i][j], ha="center", va="center", fontsize=18, fontweight="bold",
                        color="white" if dark else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([FS.LABELS[m] for m in cols], rotation=35, ha="right", fontsize=14)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row_label(r) for r in rows], fontsize=13)
    ax.tick_params(length=0)
    # geometry separators
    g = [r.split("_")[0] for r in rows]
    for i in range(1, len(rows)):
        if g[i] != g[i - 1]:
            ax.axhline(i - 0.5, color="black", lw=1.4)
    for j in range(1, len(cols)):
        ax.axvline(j - 0.5, color="white", lw=1.0)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(cbar_label, fontsize=14)
    cb.ax.tick_params(labelsize=13)
    return im


def draw_sections(axes, ratio="f020"):
    for ax, gkey in zip(axes, CS.GEOMETRY_KEYS):
        p_int = CS.ConstrainedSuiteProblem(gkey, ratio, "int")
        p_bnd = CS.ConstrainedSuiteProblem(gkey, ratio, "bnd")
        half, nn = FS.section_extent(gkey, 220)   # G3 plane needs half = 1.6 (2026-09-23)
        (e1, e2, l1, l2), t, mask_int, _ = CS.section_grid(p_int, n=nn, half=half)
        # evaluate the boundary configuration on the same plane
        P = p_int.c0[None, :] + np.meshgrid(t, t)[0].ravel()[:, None] * e1 + np.meshgrid(t, t)[1].ravel()[:, None] * e2
        inside = np.all((P >= 0) & (P <= 1), axis=1)
        mask_bnd = p_bnd.feasible(P).astype(float)
        mask_bnd[~inside] = np.nan
        o1 = float(p_int.c0 @ e1) if np.count_nonzero(e1) == 1 else 0.0
        o2 = float(p_int.c0 @ e2) if np.count_nonzero(e2) == 1 else 0.0
        t1, t2 = t + o1, t + o2
        FS.draw_feasible_mask(ax, t1, t2, mask_int)
        ax.contour(t1, t2, np.nan_to_num(mask_bnd.reshape(len(t), len(t)), nan=0.0), levels=[0.5],
                   colors=["#1a7a1a"], linestyles="--", linewidths=1.6, zorder=3)
        ax.scatter([o1], [o2], **FS.C0_MARKER)
        pad = 0.0 if (o1 or o2) else 0.04
        FS.set_section_window(ax, t1, t2, mask_int, pad)
        ax.set_title(GEOM_LABEL[gkey], fontsize=15)
        ax.set_xlabel(l1.replace("distance from $c_0$ toward ball B", "toward ball B"), fontsize=15)
        ax.set_ylabel(l2.replace("perpendicular (in subspace)", "perpendicular"), fontsize=15)
        ax.tick_params(labelsize=11)


def load_grid(S, rows, cols, key, cmp_metric):
    M = np.array([[S["per_problem"][r]["methods"][m][key] for m in cols] for r in rows])
    marks = []
    for r in rows:
        mm = []
        for m in cols:
            c = S["per_problem"][r]["comparisons"].get(m)
            if c is None or cmp_metric not in c:
                mm.append("")
            else:
                cc = c[cmp_metric]
                mm.append("*" if cc["ref_sig_lower"] else ("+" if cc["ref_sig_higher"] else ""))
        marks.append(mm)
    return M, marks


CMAP_V = LinearSegmentedColormap.from_list("v", ["#ffffff", "#fcbba1", "#cb181d", "#67000d"])
CMAP_R = LinearSegmentedColormap.from_list("r", ["#ffffff", "#c6dbef", "#2171b5", "#08306b"])


def heat_V(ax, S, rows, cols, ref, numbers=True):
    V, markV = load_grid(S, rows, cols, "V_mean", "V")
    heat(ax, V, None, markV, CMAP_V, 0, 100, rows, cols, "cumulative violations V (of 100)",
         lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}", numbers=numbers)
    if numbers:
        ax.set_title(f"cumulative violations V, mean over seeds;  '*' higher than {FS.LABELS[ref]} (Holm p < 0.05), "
                     f"'+' lower", fontsize=13)
    else:
        ax.set_title("cumulative violations in 100 experiments (mean of 20 seeds)", fontsize=15)
        ax.text(0.0, -0.24, f"*  higher than {FS.LABELS[ref]}      +  lower than {FS.LABELS[ref]}      (Holm-adjusted p < 0.05; values in Fig. S2)",
                transform=ax.transAxes, ha="left", va="top", fontsize=13, color="0.25")


def heat_R(ax, S, rows, cols, ref):
    R, markR = load_grid(S, rows, cols, "regret_mean", "best_f")
    vmax_r = float(np.nanmax(R))
    heat(ax, np.maximum(R, 5e-4), None, markR, CMAP_R, 5e-4, vmax_r, rows, cols,
         "feasible regret r = best_f − f* (log colour scale)",
         lambda v: f"{v:.3f}" if v < 0.01 else f"{v:.2f}", log=True)
    ax.set_title(f"feasible simple regret, mean over seeds;  '*' worse than {FS.LABELS[ref]} (Holm p < 0.05), '+' better",
                 fontsize=12.5)


def method_means(ax, S, rows, cols, rng):
    """(c) mean V per method, interior vs boundary configurations, 95 % bootstrap CI over seeds."""
    B = 5000
    groups = [("int", "interior optimum (12 configs)", False), ("bnd", "boundary optimum (4 configs)", True)]
    w = 0.38
    for k, (loc, lab, filled) in enumerate(groups):
        sub = [r for r in rows if r.split("_")[2] == loc]
        for i, m in enumerate(cols):
            Vm = np.array([S["per_problem"][r]["methods"][m]["V_per_seed"] for r in sub]).T   # seeds x configs
            n = Vm.shape[0]
            idx = rng.randint(0, n, size=(B, n))
            boot = Vm[idx].mean(axis=(1, 2))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mu = Vm.mean()
            x = i + (k - 0.5) * w
            ax.bar(x, mu, width=w * 0.95, facecolor=FS.COLORS[m] if filled else "white",
                   edgecolor="black" if filled else FS.COLORS[m], lw=0.6 if filled else 1.8, zorder=3,
                   hatch="////" if (m == "DALIv2") else None)
            ax.errorbar(x, mu, yerr=[[mu - lo], [hi - mu]], color="black", lw=0.9, capsize=2, zorder=4)
            ax.text(x, hi + 2.0, f"{mu:.0f}" if mu >= 10 else f"{mu:.1f}", ha="center", va="bottom", fontsize=11.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([FS.LABELS[m] for m in cols], rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("mean V (of 100)", fontsize=15)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title("mean over configurations, 95 % bootstrap CI over seeds", fontsize=15)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="0.3", lw=1.8, label="optimum in the interior (12 configurations)"),
                       Patch(facecolor="0.55", edgecolor="black", lw=0.6, label="optimum on the boundary (4 configurations)")],
              loc="upper left", fontsize=13, frameon=True, edgecolor="0.7")
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)



# ---------------------------------------------------------------------------
# v3 layout (2026-09-15): narrower heat map with grouped row labels, horizontal
# method-mean bars beside it, larger type.
# ---------------------------------------------------------------------------
GEOM_SHORT = {"g1": "G1\nconvex", "g2": "G2\nnon-convex", "g3": "G3\ndisconnected", "g4": "G4\nthin band"}
COL_LABEL = {"DALIv2LE": "DALI\n+LE", "DALIv2": "DALI", "LEonly": "LE\nonly", "SCBO": "SCBO", "TuRBO": "TuRBO", "GA": "GA",
             "BO": "BO", "Random": "Random", "cTuRBOknown": "cTuRBO\n(known)"}


def inner_label(n):
    g, r, loc = n.split("_")[:3]
    return f"{int(r[1:])} %  ·  {'inside' if loc == 'int' else 'edge'}"


def heat_v3(ax, S, rows, cols, ref):
    V, marks = load_grid(S, rows, cols, "V_mean", "V")
    im = ax.imshow(V, cmap=CMAP_V, vmin=0, vmax=100, aspect=0.62)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            mk = marks[i][j]
            if not mk:
                continue
            dark = V[i, j] > 60
            if mk == "*":      # method worse than the reference
                ax.scatter([j], [i], s=70, marker="o", facecolor="white" if dark else "black", edgecolor="none", zorder=5)
            else:              # method better than the reference
                ax.scatter([j], [i], s=70, marker="o", facecolor="none", edgecolor="white" if dark else "black", lw=1.8, zorder=5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([COL_LABEL[m].replace("\n", " ") for m in cols], fontsize=15, rotation=40, ha="left",
                       rotation_mode="anchor")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([inner_label(r) for r in rows], fontsize=14)
    ax.tick_params(length=0, pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # white grid between cells
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color="white", lw=2.0)
    for i in range(len(rows) + 1):
        ax.axhline(i - 0.5, color="white", lw=2.0)
    # geometry groups: bracket + label at the far left, grey tint on the edge rows
    g = [r.split("_")[0] for r in rows]
    starts = [i for i in range(len(rows)) if i == 0 or g[i] != g[i - 1]] + [len(rows)]
    for a, b in zip(starts[:-1], starts[1:]):
        ax.axhline(a - 0.5, color="black", lw=2.2, xmin=0, xmax=1)
        ax.annotate("", xy=(-0.40, a - 0.42), xytext=(-0.40, b - 0.58), xycoords=("axes fraction", "data"),
                    textcoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", lw=2.0, color="0.2", shrinkA=0, shrinkB=0))
        ax.text(-0.43, (a + b - 1) / 2, GEOM_SHORT[g[a]], ha="right", va="center", fontsize=15, fontweight="bold",
                transform=ax.get_yaxis_transform())
    for i, r in enumerate(rows):
        if r.split("_")[2] == "bnd":
            ax.get_yticklabels()[i].set_fontweight("bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.55)
    cb.set_label("violations in 100 experiments", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    cb.outline.set_visible(False)
    return im


def method_means_h(ax, S, rows, cols, rng):
    """(c) horizontal bars, methods top-to-bottom in the heat-map column order."""
    from matplotlib.patches import Patch
    B = 5000
    h = 0.38
    for k, (loc, filled) in enumerate((("int", False), ("bnd", True))):
        sub = [r for r in rows if r.split("_")[2] == loc]
        for i, m in enumerate(cols):
            Vm = np.array([S["per_problem"][r]["methods"][m]["V_per_seed"] for r in sub]).T
            n = Vm.shape[0]
            idx = rng.randint(0, n, size=(B, n))
            boot = Vm[idx].mean(axis=(1, 2))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mu = Vm.mean()
            y = i + (k - 0.5) * h
            ax.barh(y, mu, height=h * 0.92, facecolor=FS.COLORS[m] if filled else "white",
                    edgecolor="black" if filled else FS.COLORS[m], lw=0.6 if filled else 2.0, zorder=3,
                    hatch="////" if m == "DALIv2" else None)
            ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], color="black", lw=1.0, capsize=2.5, zorder=4)
            ax.text(max(hi, mu) + 1.5, y, f"{mu:.0f}" if mu >= 10 else f"{mu:.1f}", va="center", ha="left", fontsize=13)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([FS.LABELS[m] for m in cols], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("mean violations in 100 experiments", fontsize=15)
    ax.legend(handles=[Patch(facecolor="white", edgecolor="0.3", lw=2.0, label="optimum inside (12 configurations)"),
                       Patch(facecolor="0.55", edgecolor="black", lw=0.6, label="optimum at the edge (4 configurations)")],
              loc="upper right", fontsize=13, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def heat_v4(ax, S, rows, cols, ref):
    """Transposed heat map: rows = methods, columns = 16 configurations grouped by geometry."""
    V, marks = load_grid(S, rows, cols, "V_mean", "V")
    V, marks = V.T, [list(x) for x in zip(*marks)]
    im = ax.imshow(V, cmap=CMAP_V, vmin=0, vmax=100, aspect=1.15)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            mk = marks[i][j]
            if not mk:
                continue
            dark = V[i, j] > 60
            if mk == "*":
                ax.scatter([j], [i], s=75, marker="o", facecolor="white" if dark else "black", edgecolor="none", zorder=5)
            else:
                ax.scatter([j], [i], s=75, marker="o", facecolor="none", edgecolor="white" if dark else "black", lw=2.0, zorder=5)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([FS.LABELS[m] for m in cols],
                       fontsize=16)
    ax.set_xticks(range(len(rows)))
    xl = []
    for r in rows:
        g, rt, loc = r.split("_")[:3]
        xl.append(f"{int(rt[1:])}" + ("\nboundary" if loc == "bnd" else ""))
    ax.set_xticklabels(xl, fontsize=15)
    for i, r in enumerate(rows):
        if r.split("_")[2] == "bnd":
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.tick_params(length=0, pad=5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j in range(len(rows) + 1):
        ax.axvline(j - 0.5, color="white", lw=2.0)
    for i in range(len(cols) + 1):
        ax.axhline(i - 0.5, color="white", lw=2.0)
    g = [r.split("_")[0] for r in rows]
    starts = [i for i in range(len(rows)) if i == 0 or g[i] != g[i - 1]] + [len(rows)]
    for a, b in zip(starts[:-1], starts[1:]):
        if a > 0:
            ax.axvline(a - 0.5, color="black", lw=2.2)
        ax.annotate("", xy=(a - 0.42, 1.04), xytext=(b - 0.58, 1.04), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=2.0, color="0.2", shrinkA=0, shrinkB=0))
        ax.text((a + b - 1) / 2, 1.07, GEOM_SHORT[g[a]], ha="center", va="bottom", fontsize=16,
                fontweight="bold", transform=ax.get_xaxis_transform(), linespacing=1.1)
    ax.set_xlabel("Feasible fraction of the box (%)", fontsize=18, labelpad=8)
    return im


def method_means_h2(ax, S, rows, cols, rng):
    """(c) horizontal bars aligned with the transposed heat-map rows (same method order, shared y)."""
    from matplotlib.patches import Patch
    B = 5000
    h = 0.40
    for k, (loc, filled) in enumerate((("int", False), ("bnd", True))):
        sub = [r for r in rows if r.split("_")[2] == loc]
        for i, m in enumerate(cols):
            Vm = np.array([S["per_problem"][r]["methods"][m]["V_per_seed"] for r in sub]).T
            n = Vm.shape[0]
            idx = rng.randint(0, n, size=(B, n))
            boot = Vm[idx].mean(axis=(1, 2))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mu = Vm.mean()
            y = i + (k - 0.5) * h
            ax.barh(y, mu, height=h * 0.86, facecolor=FS.COLORS[m] if filled else "white",
                    edgecolor="black" if filled else FS.COLORS[m], lw=0.6 if filled else 2.0, zorder=3,
                    hatch="////" if m == "DALIv2" else None)
            if hi - lo > 0.5:
                ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], color="black", lw=1.0, capsize=2.5, zorder=4)
            ax.text(max(hi, mu) + 2.0, y, f"{mu:.0f}" if mu >= 10 else f"{mu:.1f}", va="center", ha="left", fontsize=15)
    ax.set_ylim(len(cols) - 0.5, -0.5)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([FS.LABELS[m] for m in cols],
                       fontsize=16)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=13, length=3)
    ax.set_xlabel("Violations in 100 experiments", fontsize=18, labelpad=8)
    ax.grid(axis="x", alpha=0.25, lw=0.8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return [Patch(facecolor="white", edgecolor="0.3", lw=2.0, label=FS.SUBSET["int12"]),
            Patch(facecolor="0.55", edgecolor="black", lw=0.6, label=FS.SUBSET["bnd4"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "tier1", "summary.json"))
    ap.add_argument("--tier2", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "tier2", "summary.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stem", default="Fig2_heatmap")
    args = ap.parse_args()
    S = json.load(open(args.summary))
    rows = config_order(list(S["per_problem"]))
    cols = [m for m in METHODS if all(m in S["per_problem"][r]["methods"] for r in rows)]
    ref = S["config"]["reference"]

    # ---------------- main Fig 2 (v7, 2026-09-19): (a) V heat map, (b) method means.
    # The 2-D sections of the geometries moved to Fig 1b.
    FS.main_rc()
    fig = plt.figure(figsize=(22.0, 10.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.04, left=0.08, right=0.98, top=0.86, bottom=0.14)
    ax_b = fig.add_subplot(gs[0, 0])
    im = heat_v4(ax_b, S, rows, cols, ref)
    FS.panel_label(ax_b, "a", dx=-0.17, dy=1.16)
    ax_c = fig.add_subplot(gs[0, 1])
    handles_c = method_means_h2(ax_c, S, rows, cols, np.random.RandomState(0))
    FS.panel_label(ax_c, "b", dx=-0.36, dy=1.16)
    ax_c.legend(handles=handles_c, loc="upper left", bbox_to_anchor=(0.0, -0.10), fontsize=15, frameon=False, ncol=1)
    cb = fig.colorbar(im, ax=ax_b, orientation="horizontal", fraction=0.035, pad=0.17, shrink=0.42, anchor=(0.0, 1.0))
    cb.set_label("Violations in 100 experiments", fontsize=16)
    cb.ax.tick_params(labelsize=13)
    cb.outline.set_visible(False)
    ax_b.text(1.0, -0.21, f"●  more than {FS.LABELS[ref]}    ○  fewer than {FS.LABELS[ref]}\n(Holm-adjusted p < 0.05; values in Fig. S2)",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=14, color="0.25", linespacing=1.5)
    fig.canvas.draw()
    pb = ax_b.get_position()
    ax_c.set_position([pb.x1 + 0.13, pb.y0, 0.30, pb.height])
    ax_c.set_ylim(len(cols) - 0.5, -0.5)
    FS.savefig(fig, args.figdir, args.stem, dpi=args.dpi)

    # ---------------- SI: the four 2-D sections alone (former Fig 2a), kept for the SI
    fig = plt.figure(figsize=(22.0, 6.0))
    gs_a = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.1], wspace=0.5, left=0.05, right=0.98, top=0.85, bottom=0.15)
    axes_a = [fig.add_subplot(gs_a[0, k]) for k in range(4)]
    draw_sections(axes_a)
    for ax, gk in zip(axes_a, CS.GEOMETRY_KEYS):
        ax.set_title(GEOM_SHORT[gk].replace("\n", " · "), fontsize=18, fontweight="bold")
    ax_leg = fig.add_subplot(gs_a[0, 4])
    ax_leg.axis("off")
    ax_leg.legend(handles=FS.legend_handles_constraint()[:1]
                  + [plt.Line2D([], [], color="#1a7a1a", ls="--", lw=1.6, label="feasible set when the\noptimum is on the boundary"),
                     plt.Line2D([], [], marker="*", ls="", ms=15, color="black", label="optimum (fixed)")],
                  loc="center left", frameon=False, fontsize=14, labelspacing=1.2)
    FS.savefig(fig, args.figdir, "FigS_suite_sections_f020", dpi=args.dpi)

    # ---------------- SI: Tier-1 V heat map WITH the numbers (former main panel b)
    fig = plt.figure(figsize=(19.2, 8.0))
    ax = fig.add_subplot(111)
    heat_V(ax, S, rows, cols, ref, numbers=True)
    FS.savefig(fig, args.figdir, "FigS_tier1_V_heatmap", dpi=args.dpi)

    # ---------------- SI: Tier-1 regret heat map (former Fig 2c)
    fig = plt.figure(figsize=(19.2, 7.6))
    ax = fig.add_subplot(111)
    heat_R(ax, S, rows, cols, ref)
    FS.savefig(fig, args.figdir, "FigS_tier1_regret_heatmap", dpi=args.dpi)

    # ---------------- SI: Tier-2 (boundary x ratio 0.50 / 0.05) V and regret heat maps
    if os.path.exists(args.tier2):
        S2 = json.load(open(args.tier2))
        rows2 = config_order(list(S2["per_problem"]))
        cols2 = [m for m in METHODS if all(m in S2["per_problem"][r]["methods"] for r in rows2)]
        fig, axes = plt.subplots(2, 1, figsize=(19.2, 11.0), gridspec_kw={"hspace": 0.6})
        heat_V(axes[0], S2, rows2, cols2, S2["config"]["reference"])
        FS.panel_label(axes[0], "a", dx=-0.12, dy=1.02)
        heat_R(axes[1], S2, rows2, cols2, S2["config"]["reference"])
        FS.panel_label(axes[1], "b", dx=-0.12, dy=1.02)
        FS.savefig(fig, args.figdir, "FigS_tier2_heatmap", dpi=args.dpi)


if __name__ == "__main__":
    main()
