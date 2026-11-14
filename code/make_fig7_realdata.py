"""Fig 7 -- real-process datasets with a data-derived feasibility oracle (H5).

Columns: SFS (steel fatigue, 5-D), NOMC (methane conversion, 4-D), CCPP (power
plant, 4-D); oracle threshold q99 (main).  Rows: (top) cumulative violations per
cycle, mean and 95 % CI over 10 seeds; (bottom) improvement of the feasible best
over the warm-start best, in the original objective units (positive = better;
dotted line = the best value in the recorded data, i.e. f*).

Data: realdata/summary.json + summary_metrics_per_run.csv, realdata_params.json.
Feasible fraction of the box (q99 oracle) is reported in the caption / table.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "realdata")
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
# (problem key, dataset subtitle (top row), quantity subtitle (bottom row))
COLS = [("real_sfs_q99", "SFS — steel fatigue (5-D)", "Fatigue objective (reduction)"),
        ("real_nomc_q99", "NOMC — methane conversion (4-D)", "Conversion gain"),
        ("real_ccpp_q99", "CCPP — power plant (4-D)", "Power gain (MW)")]
T_MAX = 100      # data end
X_MAX = 150      # panel end; the extra room holds the direct curve labels
LABEL_FS = 17
LABEL_SEP_IN = 0.30   # minimum vertical separation between end labels (inches)


def line_kw(m):
    """figstyle line style without the DALI+LE markers (curves are labelled directly)."""
    kw = FS.style_line(m)
    for k in ("marker", "markevery", "ms", "markerfacecolor", "markeredgecolor", "markeredgewidth"):
        kw.pop(k, None)
    kw.pop("label", None)
    return kw


def stack_labels(ys, min_sep, lo, hi):
    """Shift label positions so that consecutive labels (sorted by y) are at least
    min_sep apart and all lie in [lo, hi] (requires min_sep <= (hi-lo)/(n-1));
    returns positions in the original order (ties keep input order)."""
    order = np.argsort(ys, kind="stable")
    pos = np.array(ys, float)[order]
    n = len(pos)
    for _ in range(2):
        pos[0] = max(pos[0], lo)                      # push up from the bottom
        for i in range(1, n):
            pos[i] = max(pos[i], pos[i - 1] + min_sep)
        pos[-1] = min(pos[-1], hi)                    # push down from the top
        for i in range(n - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - min_sep)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def end_labels(fig, ax, ys, texts, colors, x_end=T_MAX, x_text=T_MAX + 4, fs=LABEL_FS):
    """Direct labels at the right end of curves, stacked so they do not overlap,
    with a thin grey leader line where a label had to be moved.  The minimum
    separation is LABEL_SEP_IN inches converted to data units of this axis."""
    y0, y1 = ax.get_ylim()
    ax_h_in = ax.get_position().height * fig.get_size_inches()[1]
    min_sep = (y1 - y0) * LABEL_SEP_IN / ax_h_in
    # the whole stack must fit inside [y0, y1] with half a label of margin
    min_sep = min(min_sep, ((y1 - y0) - min_sep) / max(len(ys) - 1, 1))
    pos = stack_labels(ys, min_sep, y0 + 0.5 * min_sep, y1 - 0.5 * min_sep)
    for y, yp, t, c in zip(ys, pos, texts, colors):
        if abs(yp - y) > 1e-9:
            ax.plot([x_end, x_text - 0.4 * (x_text - x_end)], [y, yp], color="0.6", lw=0.8,
                    clip_on=False, zorder=1)
        ax.text(x_text, yp, t, color=c, fontsize=fs, va="center", ha="left", clip_on=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES)
    ap.add_argument("--params", default=os.path.join(CODE_DIR, "realdata_params.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stem", default="Fig7_realdata")
    args = ap.parse_args()
    S = json.load(open(os.path.join(args.results, "summary.json")))
    R = pd.read_csv(os.path.join(args.results, "summary_metrics_per_run.csv"))
    PR = json.load(open(args.params))   # feasible fractions: reported in the caption / table
    cyc = np.arange(1, 21) * 5
    FS.main_rc()

    fig = plt.figure(figsize=(24.0, 11.0))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30,
                          left=0.065, right=0.985, top=0.93, bottom=0.085)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])

    for k, (p, title, ylab) in enumerate(COLS):
        P = S["per_problem"][p]
        methods = [m for m in METHODS if m in P["methods"]]
        # (top) cumulative violations
        ax = axes[0, k]
        ends, txts, cols = [], [], []
        for m in methods:
            r = P["methods"][m]
            mu = np.array(r["V_curve_mean"])
            sd = np.array(r["V_curve_sd"] or [0] * 20)
            se = sd / np.sqrt(r["n_seeds"])
            ax.plot(cyc, mu, **line_kw(m))
            ax.fill_between(cyc, mu - 1.96 * se, mu + 1.96 * se, color=FS.COLORS[m], alpha=0.12, lw=0)
            ends.append(mu[-1])
            txts.append(FS.LABELS[m])
            cols.append(FS.COLORS[m])
        ax.set_xlim(0, X_MAX)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        end_labels(fig, ax, ends, txts, cols)
        ax.set_xlabel("Experiments")
        if k == 0:
            ax.set_ylabel("Cumulative violations")
        FS.subtitle(ax, title, pad=10)
        FS.clean_axes(ax, xbound=(0, T_MAX))
        FS.panel_label(ax, "abc"[k], dx=-0.19 if k == 0 else -0.13, dy=1.04)

        # (bottom) improvement over the warm-start best
        ax = axes[1, k]
        ends, txts, cols = [], [], []
        for m in methods:
            r = P["methods"][m]
            f0 = R.loc[(R["problem"] == p) & (R["method"] == m), "f0_min"].mean()
            imp = f0 - np.array(r["best_f_curve_mean"])
            ax.plot(cyc, imp, **line_kw(m))
            ends.append(imp[-1])
            txts.append(FS.LABELS[m])
            cols.append(FS.COLORS[m])
        f0_ref = R.loc[(R["problem"] == p) & (R["method"] == "DALIv2LE"), "f0_min"].mean()
        gap = f0_ref - P["f_star"]                     # best value in the full data set, relative to the warm start
        top = max(ends)
        if gap > 3.0 * top:
            # the reference is far above every curve (NOMC): keep the curves readable and
            # quote the reference as an off-scale note instead of drawing the line
            ax.set_ylim(-0.02 * top, 1.45 * top)
            ax.text(T_MAX - 2, 1.40 * top, f"empirical $\\hat f^*$ = {gap:.2f} (off scale)", fontsize=15,
                    va="top", ha="right", color="0.3")
        else:
            ax.axhline(gap, xmax=T_MAX / X_MAX, color="0.3", ls=":", lw=1.4)
            ax.text(T_MAX - 2, gap, "empirical $\\hat f^*$", fontsize=15, va="top", ha="right", color="0.3")
        ax.axhline(0, xmax=T_MAX / X_MAX, color="black", lw=0.8)
        ax.set_xlim(0, X_MAX)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(*ax.get_ylim())   # freeze the autoscaled y range before placing labels
        end_labels(fig, ax, ends, txts, cols)
        ax.set_xlabel("Experiments")
        if k == 0:
            ax.set_ylabel("Improvement over\nwarm-start best")
        FS.subtitle(ax, ylab, pad=10)
        FS.clean_axes(ax, xbound=(0, T_MAX))
        FS.panel_label(ax, "def"[k], dx=-0.19 if k == 0 else -0.13, dy=1.04)

    FS.savefig(fig, args.figdir, args.stem, dpi=args.dpi)
    # numbers for the text
    for p, _, _ in COLS:
        P = S["per_problem"][p]
        print(p, "box_feasible_frac", PR[p]["box_feasible_frac"],
              {m: (round(P["methods"][m]["V_mean"], 1), round(P["methods"][m]["improvement_over_warm_mean"], 3))
               for m in METHODS})


if __name__ == "__main__":
    main()
