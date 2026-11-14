"""SI figures, v2 style (2026-09-29): one script, eight figures, renumbered S1-S8.

Same numbers as the v1 SI figures (make_fig2_heatmap.py, make_si_figs.py,
make_fig3_frontier.py, make_si_extra.py); only the presentation changes, following the
class-colour / line-style / marker conventions of the main-text figures v2
(figstyle_v2.py).  PNG (300 dpi) + PDF only; no PowerPoint version.

Numbering (2026-09-29, second pass): the former Fig S1 (2-D sections with axis ticks)
duplicated Fig 1b of the main text and was dropped; every later figure moved up by one.
The former S7 (LE sensitivity, never run) had already been removed in the first pass.

  Numbering of 2026-10-06 (order of first citation in the main text):
  S1  FigS1_heatmaps_v2                 (a, b) main block V and regret with numbers, (c, d) boundary 50 / 5 % block
  S2  FigS2_safety_cost_v2              regret cost per avoided violation per configuration (2.3)
  S3  FigS3_survival_per_config_v2      per-configuration Kaplan-Meier survival (2.4)
  S4  FigS4_candidate_feasibility_per_cycle_v2  feasible fraction of raw candidates per cycle (2.5)
  S5  FigS5_record_conditions_v2        the four record conditions of 2.6 (actual warm starts, G3 20 % interior) -- new
  S6  FigS6_realdata_threshold_sensitivity_v2   real-data oracle threshold q95 / q99 / q99.9 (2.7)
  S7  FigS7_gate_v2                     kNN gate, exploratory (2.8)
  S8  FigS8_dim2_cbranin2_v2            2-D constrained Branin spot check (Methods 4.2)
  S9  FigS9_curves_per_config_v2        (a) cumulative violations, (b) feasible regret per configuration (SI)
  --  FigS0_suite_sections_v2           the dropped sections figure, still drawable with --only sections

Usage: python make_si_figs_v2.py [--res DIR] [--figdir DIR] [--dpi 300] [--only S2,S3] [--pptx]

--pptx (2026-10-01): instead of the PNG/PDF, write <stem>_pptx_base.png (the figure with all
editable text hidden; data untouched) + <stem>_pptx.json (positions of the editable text, legends
and colour strips) for make_si_pptx.js.  Grid figures S2, S3, S6 use thinner lines (LW_GRID) since
2026-10-01 (author: lines too thick to tell apart).
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

import figstyle as FS
import figstyle_v2 as V2

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
GEOM_TITLE = {"g1": "G1  convex", "g2": "G2  non-convex", "g3": "G3  disconnected", "g4": "G4  thin band"}
CMAP_V = LinearSegmentedColormap.from_list("v", ["#ffffff", "#fcbba1", "#cb181d", "#67000d"])   # as Fig 2 v8
CMAP_R = LinearSegmentedColormap.from_list("r", ["#ffffff", "#e3e3e3", "#a6a6a6", "#4d4d4d"])   # neutral for regret
LEG_H = 0.30
PX_MODE = False            # set by --pptx
PENDING_LEGENDS = []       # legend specs collected before the figure is saved (pptx mode)
LW_GRID = 0.65             # line-width factor for the 4 x 4 grid figures (S2, S3, S6)


def thin(kw, f=LW_GRID):
    kw = dict(kw)
    kw["lw"] = kw["lw"] * f
    return kw


def tag(artist):
    """Mark a text artist as editable in the PowerPoint export."""
    artist.set_gid("px")
    return artist


def legend(fig, entries, x, y, w, h, fs=7.5, sample=0.28, gap=0.10):
    if PX_MODE:
        PENDING_LEGENDS.append((entries, x, y, w, h, fs, sample, gap))
    else:
        V2.legend_box(fig, entries, x, y, w, h, fs=fs, sample=sample, gap=gap)


def save(fig, figdir, stem, dpi):
    global PENDING_LEGENDS
    if PX_MODE:
        from si_pptx_assets import PptxAssets
        px = PptxAssets(fig, stem)
        for args in PENDING_LEGENDS:
            px.legend(*args)
        PENDING_LEGENDS = []
        px.collect()
        px.export(figdir, dpi=max(dpi, 400))
    else:
        FS.savefig(fig, figdir, stem, dpi=dpi)


def config_order(names):
    return sorted(names, key=lambda n: (n.split("_")[0], 0 if n.split("_")[2] == "int" else 1, -int(n.split("_")[1][1:])))


def load(res, block):
    return json.load(open(os.path.join(res, block, "summary.json")))


def method_cols(S, rows):
    return [m for m in METHODS if all(m in S["per_problem"][r]["methods"] for r in rows)]


def grid_axes(fig, nrow, ncol, left, right, bottom, top, wgap, hgap):
    """Axes grid laid out in inches (left/right/bottom/top = margins)."""
    fw, fh = fig.get_size_inches()
    pw = (fw - left - right - (ncol - 1) * wgap) / ncol
    ph = (fh - bottom - top - (nrow - 1) * hgap) / nrow
    axes = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            x = left + j * (pw + wgap)
            y = fh - top - (i + 1) * ph - i * hgap
            axes[i, j] = fig.add_axes([x / fw, y / fh, pw / fw, ph / fh])
    return axes


def method_lines_legend(fig, y, methods=METHODS, fs=7.0, lw_scale=1.0):
    ent = []
    prev = None
    for m in methods:
        if prev is not None and V2.CLASS_OF[m] != prev:
            ent.append("sep")
        ent.append((plt.Line2D([], [], label=FS.LABELS[m], **thin(V2.line_kw(m), lw_scale)), FS.LABELS[m]))
        prev = V2.CLASS_OF[m]
    legend(fig, ent, None, y, None, LEG_H, fs=fs, sample=0.26, gap=0.09)


# ====================================================================== S1 sections
def fig_s1(res, figdir, dpi):
    import make_fig2_heatmap as F2
    fig = plt.figure(figsize=(7.2, 2.15))
    axes = grid_axes(fig, 1, 4, left=0.42, right=1.55, bottom=0.42, top=0.28, wgap=0.42, hgap=0)
    F2.draw_sections(list(axes[0]))
    for ax, gk in zip(axes[0], ["g1", "g2", "g3", "g4"]):
        ax.set_title(GEOM_TITLE[gk], fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xlabel(ax.get_xlabel(), fontsize=7.5)
        ax.set_ylabel(ax.get_ylabel(), fontsize=7.5)
        ax.tick_params(labelsize=6.5, colors=V2.MUTED, width=0.8, length=2.5)
        for sp in ax.spines.values():
            sp.set_color(V2.AXIS)
            sp.set_linewidth(0.8)
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([(fw - 1.45) / fw, 0.45 / fh, 1.4 / fw, 1.3 / fh])
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.3)
    ax.axis("off")
    ax.add_patch(Rectangle((0.0, 0.0), 1.4, 1.3, facecolor="white", edgecolor="#BFBFBF", lw=0.8))
    ax.add_patch(Rectangle((0.1, 0.98), 0.28, 0.14, facecolor=FS.FEAS_COLOR, alpha=0.3, edgecolor=FS.FEAS_COLOR, lw=1.2))
    ax.text(0.46, 1.05, "feasible set (hidden)", fontsize=7, va="center")
    ax.plot([0.1, 0.38], [0.66, 0.66], color="#1a7a1a", ls="--", lw=1.2)
    ax.text(0.46, 0.66, "feasible set when the\noptimum is near the\nboundary", fontsize=7, va="center", linespacing=1.15)
    ax.plot([0.24], [0.22], marker="*", ms=9, color="black", ls="")
    ax.text(0.46, 0.22, "optimum (fixed)", fontsize=7, va="center")
    FS.savefig(fig, figdir, "FigS0_suite_sections_v2", dpi=dpi)


# ====================================================================== S2 heat maps
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


def heat2(ax, M, marks, cmap, vmin, vmax, rows, cols, fmt, log=False, row_labels=True, fs=5.8):
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
    im = ax.imshow(M, cmap=cmap, vmin=None if log else vmin, vmax=None if log else vmax, norm=norm, aspect="auto",
                   interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            frac = (np.log(v) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)) if log else (v - vmin) / (vmax - vmin)
            ax.text(j, i, fmt(v) + marks[i][j], ha="center", va="center", fontsize=fs,
                    color="white" if frac > 0.6 else V2.INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([FS.LABELS[m].replace(" (", "\n(") for m in cols], fontsize=6.3)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([V2.config_label(r, short=True) for r in rows] if row_labels else [], fontsize=6.5)
    g = [r.split("_")[0] for r in rows]
    for i in range(1, len(rows)):
        if g[i] != g[i - 1]:
            ax.axhline(i - 0.5, color=V2.INK, lw=0.9)
    for j in range(1, len(cols)):
        ax.axvline(j - 0.5, color="white", lw=0.8)
    for sp in ax.spines.values():
        sp.set_color(V2.AXIS)
        sp.set_linewidth(0.6)
    V2.class_band_x(ax, cols, y0=1.0 + 0.30 / (0.19 * len(rows)) * 0.9, hgt=0.012 * 16 / len(rows))
    return im


def fig_s2(res, figdir, dpi):
    S1, S2 = load(res, "tier1"), load(res, "tier2")
    rows1, rows2 = config_order(list(S1["per_problem"])), config_order(list(S2["per_problem"]))
    cols = method_cols(S1, rows1)
    cw, rh = 0.32, 0.19
    W = 1.02 + 2 * 9 * cw + 0.28
    hdr = 0.42
    H = 0.30 + hdr + 16 * rh + 0.62 + hdr + 8 * rh + 0.55
    fig = plt.figure(figsize=(W, H))
    fmt_v = lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}"
    fmt_r = lambda v: f"{v:.3f}" if v < 0.01 else f"{v:.2f}"
    R1, _ = load_grid(S1, rows1, cols, "regret_mean", "best_f")
    R2, _ = load_grid(S2, rows2, cols, "regret_mean", "best_f")
    rmin, rmax = 5e-4, float(max(np.nanmax(R1), np.nanmax(R2)))
    ims = {}
    y_top = H - 0.30 - hdr
    for k, (S, rows, ytop, letters, n) in enumerate([(S1, rows1, y_top, "ab", 16), (S2, rows2, y_top - 16 * rh - 0.62 - hdr, "cd", 8)]):
        ref = S["config"]["reference"]
        V, mV = load_grid(S, rows, cols, "V_mean", "V")
        R, mR = load_grid(S, rows, cols, "regret_mean", "best_f")
        for c, (M, marks, cmap, vmin, vmax, fmt, log, key) in enumerate([
                (V, mV, CMAP_V, 0, 100, fmt_v, False, "V"), (np.maximum(R, rmin), mR, CMAP_R, rmin, rmax, fmt_r, True, "R")]):
            x0 = 1.02 + c * (9 * cw + 0.28)
            ax = fig.add_axes([x0 / W, (ytop - n * rh) / H, 9 * cw / W, n * rh / H])
            ims[key] = heat2(ax, M, marks, cmap, vmin, vmax, rows, cols, fmt, log=log, row_labels=(c == 0))
            block = "main block (20 seeds)" if k == 0 else "near-boundary 50 % / 5 % block (10 seeds)"
            ttl = f"Violations in 100 experiments, {block}" if c == 0 else "Feasible regret, same block"
            V2.title(ax, letters[c], ttl, dx=-0.02 if c else -0.37, dy=1.0 + (hdr + 0.05) / (n * rh))
    # colour bars under (c) and (d)
    for c, (key, lab) in enumerate([("V", "Violations in 100 experiments (cell mean)"), ("R", "Feasible regret r (log colour scale)")]):
        x0 = 1.02 + c * (9 * cw + 0.28)
        cax = fig.add_axes([(x0 + 0.6) / W, 0.28 / H, (9 * cw - 1.2) / W, 0.09 / H])
        cb = fig.colorbar(ims[key], cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=6.8, labelpad=2)
        tag(cb.ax.xaxis.label)
        cb.ax.tick_params(labelsize=6, length=2, width=0.6, colors=V2.MUTED)
        cb.outline.set_visible(False)
        cb.ax.minorticks_off()
    save(fig, figdir, "FigS1_heatmaps_v2", dpi)


# ====================================================================== S3 survival per configuration
def fig_s3(res, figdir, dpi):
    S = load(res, "tier1")
    rows = config_order(list(S["per_problem"]))
    fig = plt.figure(figsize=(7.2, 6.6))
    axes = grid_axes(fig, 4, 4, left=0.55, right=0.12, bottom=0.78, top=0.22, wgap=0.22, hgap=0.34)
    for ax, p in zip(axes.ravel(), rows):
        for m in METHODS:
            s = S["per_problem"][p]["methods"][m]["km_survival"]
            ax.step(np.arange(0, len(s) + 1), [1.0] + list(s), where="post", **thin(V2.line_kw(m)))
        ax.set_title(V2.config_label(p), fontsize=7.5, pad=3)
        tag(ax.title)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.02, 1.04)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 50, 100])
        V2.axes(ax)
    for i in range(4):
        for j in range(4):
            if i < 3:
                axes[i, j].tick_params(labelbottom=False)
            if j > 0:
                axes[i, j].tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("Experiment index n", fontsize=7.5)
        tag(ax.xaxis.label)
    tag(fig.text(0.13 / 7.2, 0.5 + 0.28 / 6.6, "Fraction of runs with no violation up to n", rotation=90, va="center", ha="center", fontsize=8))
    method_lines_legend(fig, 0.10, lw_scale=LW_GRID)
    save(fig, figdir, "FigS3_survival_per_config_v2", dpi)


# ====================================================================== S4 curves per configuration
def fig_s4(res, figdir, dpi):
    S = load(res, "tier1")
    rows = config_order(list(S["per_problem"]))
    cyc = np.arange(1, 21) * 5
    fig = plt.figure(figsize=(7.2, 9.4))
    fw, fh = 7.2, 9.4
    top_a, top_b = 0.30, 0.30
    blockH = (fh - 0.78 - 0.45 - top_a - top_b) / 2
    ph = (blockH - 3 * 0.30) / 4
    pw = (fw - 0.6 - 0.12 - 3 * 0.22) / 4
    axes = {}
    for b, (key, ytop) in enumerate([("a", fh - top_a), ("b", fh - top_a - blockH - 0.45 - top_b)]):
        for i in range(4):
            for j in range(4):
                x = 0.6 + j * (pw + 0.22)
                y = ytop - (i + 1) * ph - i * 0.30
                axes[key, i, j] = fig.add_axes([x / fw, y / fh, pw / fw, ph / fh])
        V2.title(axes[key, 0, 0], key, "Cumulative violations" if key == "a" else "Feasible regret, log scale",
                 dx=-0.28, dy=1.0 + 0.22 / ph)
    for k, p in enumerate(rows):
        i, j = divmod(k, 4)
        fs = S["per_problem"][p]["f_star"]
        for m in METHODS:
            r = S["per_problem"][p]["methods"][m]
            axes["a", i, j].plot(cyc, r["V_curve_mean"], **thin(V2.line_kw(m)))
            axes["b", i, j].plot(cyc, np.maximum(np.array(r["best_f_curve_mean"]) - fs, 3e-4), **thin(V2.line_kw(m)))
        for key in "ab":
            ax = axes[key, i, j]
            ax.set_title(V2.config_label(p), fontsize=7.0, pad=2.5)
            tag(ax.title)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            V2.axes(ax)
            if i < 3:
                ax.tick_params(labelbottom=False)
            if j > 0:
                ax.tick_params(labelleft=False)
        axes["a", i, j].set_ylim(0, 100)
        axes["a", i, j].set_yticks([0, 50, 100])
        axes["b", i, j].set_yscale("log")
        axes["b", i, j].set_ylim(3e-4, 0.3)
        axes["b", i, j].set_yticks([1e-3, 1e-2, 1e-1])
        axes["b", i, j].minorticks_off()
    for j in range(4):
        axes["b", 3, j].set_xlabel("Experiments", fontsize=7.5)
        axes["a", 3, j].set_xlabel("Experiments", fontsize=7.5)
        tag(axes["b", 3, j].xaxis.label)
        tag(axes["a", 3, j].xaxis.label)
    ya = (fh - top_a - blockH / 2) / fh
    yb = (0.78 + blockH / 2) / fh
    tag(fig.text(0.14 / fw, ya, "Violations so far", rotation=90, va="center", ha="center", fontsize=8))
    tag(fig.text(0.14 / fw, yb, "Best feasible f − f*", rotation=90, va="center", ha="center", fontsize=8))
    method_lines_legend(fig, 0.10, lw_scale=LW_GRID)
    save(fig, figdir, "FigS9_curves_per_config_v2", dpi)


# ====================================================================== S5 cbranin2
def fig_s5(res, figdir, dpi):
    S = load(res, "dim2")
    d = S["per_problem"]["cbranin2"]
    cyc = np.arange(1, 21) * 5
    fig = plt.figure(figsize=(7.2, 2.75))
    fw, fh = 7.2, 2.75
    ax_a = fig.add_axes([0.55 / fw, 0.95 / fh, 2.0 / fw, 1.5 / fh])
    ax_b = fig.add_axes([3.15 / fw, 0.95 / fh, 2.0 / fw, 1.5 / fh])
    ax_c = fig.add_axes([6.05 / fw, 0.95 / fh, 1.0 / fw, 1.5 / fh])
    for m in METHODS:
        r = d["methods"][m]
        ax_a.plot(cyc, r["V_curve_mean"], **V2.line_kw(m))
        ax_b.plot(cyc, r["best_f_curve_mean"], **V2.line_kw(m))
    ax_b.axhline(d["f_star"], color="#9A9A9A", ls=(0, (1, 2)), lw=1.0, zorder=1)
    tag(ax_b.text(100, d["f_star"], " f*", fontsize=7, va="center", ha="left", color=V2.MUTED))
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Experiments", fontsize=7.5)
        tag(ax.xaxis.label)
        V2.axes(ax)
    ax_a.set_ylabel("Violations so far", fontsize=7.5)
    ax_b.set_ylabel("Best feasible f", fontsize=7.5)
    tag(ax_a.yaxis.label)
    tag(ax_b.yaxis.label)
    V2.title(ax_a, "a", "Cumulative violations", dx=-0.25, dy=1.04)
    V2.title(ax_b, "b", "Best feasible value", dx=-0.25, dy=1.04)
    # (c) violations per method, horizontal bars with 95 % CI
    ms = METHODS[::-1]
    vals = [d["methods"][m]["V_mean"] for m in ms]
    cis = [d["methods"][m]["V_ci95"] for m in ms]
    y = np.arange(len(ms))
    ax_c.barh(y, vals, height=0.62, color=[V2.CLASS_COLOR[m] for m in ms], edgecolor=V2.INK, lw=0.4)
    ax_c.errorbar(vals, y, xerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
                  fmt="none", ecolor=V2.INK, elinewidth=0.7, capsize=1.5)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([FS.LABELS[m] for m in ms], fontsize=6.8)
    ax_c.set_ylim(-0.6, len(ms) - 0.4)
    ax_c.set_xlim(0, 100)
    ax_c.set_xticks([0, 50, 100])
    ax_c.set_xlabel("Violations in 100\nexperiments", fontsize=7.5)
    tag(ax_c.xaxis.label)
    V2.axes(ax_c)
    ax_c.tick_params(axis="y", length=0)
    V2.title(ax_c, "c", "Violations, mean and 95 % CI", dx=-1.05, dy=1.04)
    method_lines_legend(fig, 0.08)
    save(fig, figdir, "FigS8_dim2_cbranin2_v2", dpi)


# ====================================================================== S6 real-data threshold sensitivity
def fig_s6(res, figdir, dpi):
    Sr, Ss = load(res, "realdata"), load(res, "realdata_sens")
    ds = [("sfs", "SFS"), ("nomc", "NOMC"), ("ccpp", "CCPP")]
    qs = ["q95", "q99", "q999"]
    ms = METHODS[::-1]
    fig = plt.figure(figsize=(7.2, 6.4))
    axes = grid_axes(fig, 2, 3, left=0.95, right=0.12, bottom=0.78, top=0.42, wgap=0.28, hgap=0.62)
    bh = 0.24
    for k, (dk, dl) in enumerate(ds):
        for row, key in enumerate(("V_mean", "improvement_over_warm_mean")):
            ax = axes[row, k]
            for j, q in enumerate(qs):
                P = (Sr if q == "q99" else Ss)["per_problem"][f"real_{dk}_{q}"]["methods"]
                vals = [P[m][key] for m in ms]
                yy = np.arange(len(ms)) + (1 - j) * bh
                for m, y, v in zip(ms, yy, vals):
                    c = V2.CLASS_COLOR[m]
                    if q == "q95":
                        ax.barh(y, v, height=bh * 0.92, color=V2.light(c, 0.6), edgecolor=V2.light(c, 0.6), lw=0.3)
                    elif q == "q99":
                        ax.barh(y, v, height=bh * 0.92, color=c, edgecolor=V2.INK, lw=0.4)
                    else:
                        ax.barh(y, v, height=bh * 0.92, color="white", edgecolor=c, lw=0.9)
            ax.set_yticks(np.arange(len(ms)))
            ax.set_yticklabels([FS.LABELS[m] for m in ms] if k == 0 else [], fontsize=7)
            ax.set_ylim(-0.6, len(ms) - 0.3)
            ax.tick_params(axis="y", length=0)
            V2.axes(ax)
            if row == 0:
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])
                ax.set_title(dl, fontsize=9, fontweight="bold", pad=5)
                tag(ax.title)
            if k == 0:
                for yy0 in range(len(ms) - 1):
                    ax.axhline(yy0 + 0.5, color="#E3E3E3", lw=0.5, zorder=0)
            else:
                for yy0 in range(len(ms) - 1):
                    ax.axhline(yy0 + 0.5, color="#E3E3E3", lw=0.5, zorder=0)
    for k in range(3):
        axes[0, k].set_xlabel("Violations in 100 experiments", fontsize=7.5)
        axes[1, k].set_xlabel("Improvement over warm-start best", fontsize=7.5)
        tag(axes[0, k].xaxis.label)
        tag(axes[1, k].xaxis.label)
    V2.title(axes[0, 0], "a", "Violations", dx=-0.62, dy=1.08)
    V2.title(axes[1, 0], "b", "Improvement over the warm-start best", dx=-0.62, dy=1.02)
    ent = [(("#BFBFBF", "#BFBFBF", None), "q95 (strictest: smallest feasible set)"), "sep",
           (("#595959", V2.INK, None), "q99 (main)"), "sep",
           (("white", "#595959", None), "q99.9 (loosest)")]
    legend(fig, ent, None, 0.10, None, LEG_H, fs=7.0, sample=0.26, gap=0.09)
    save(fig, figdir, "FigS6_realdata_threshold_sensitivity_v2", dpi)


# ====================================================================== S7 candidate feasibility per cycle
def fig_s7(res, figdir, dpi):
    S = load(res, "tier1")
    rows = config_order(list(S["per_problem"]))
    cyc = np.arange(1, 21)
    fig = plt.figure(figsize=(7.2, 6.6))
    axes = grid_axes(fig, 4, 4, left=0.55, right=0.12, bottom=0.78, top=0.22, wgap=0.22, hgap=0.34)
    for ax, p in zip(axes.ravel(), rows):
        ratio = int(p.split("_")[1][1:]) / 100
        for m in ("DALIv2", "DALIv2LE", "SCBO"):
            c = S["per_problem"][p]["methods"][m].get("cand_feasible_curve_mean")
            if c:
                ax.plot(cyc, c, **thin(V2.line_kw(m)))
        ax.axhline(ratio, color="#9A9A9A", ls=(0, (1, 2)), lw=1.0, zorder=1)
        ax.set_title(V2.config_label(p), fontsize=7.5, pad=3)
        tag(ax.title)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([1, 10, 20])
        ax.set_ylim(-0.02, 1.04)
        ax.set_yticks([0, 0.5, 1])
        V2.axes(ax)
    for i in range(4):
        for j in range(4):
            if i < 3:
                axes[i, j].tick_params(labelbottom=False)
            if j > 0:
                axes[i, j].tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("Cycle", fontsize=7.5)
        tag(ax.xaxis.label)
    tag(fig.text(0.13 / 7.2, 0.5 + 0.28 / 6.6, "Feasible fraction of raw candidates", rotation=90, va="center", ha="center", fontsize=8))
    ent = [(plt.Line2D([], [], label=FS.LABELS[m], **thin(V2.line_kw(m))), FS.LABELS[m]) for m in ("DALIv2LE", "DALIv2")]
    ent += ["sep", (plt.Line2D([], [], label="SCBO", **thin(V2.line_kw("SCBO"))), "SCBO")]
    ent += ["sep", (plt.Line2D([], [], color="#9A9A9A", ls=":", lw=1.0), "uniform-in-box candidates (= feasible fraction)")]
    legend(fig, ent, None, 0.10, None, LEG_H, fs=7.0, sample=0.26, gap=0.09)
    save(fig, figdir, "FigS4_candidate_feasibility_per_cycle_v2", dpi)


# ====================================================================== S8 safety cost
def fig_s8(res, figdir, dpi):
    S = load(res, "tier1")
    probs = config_order(list(S["per_problem"]))
    fig = plt.figure(figsize=(7.2, 4.9))
    fw, fh = 7.2, 4.9
    ax = fig.add_axes([1.35 / fw, 0.95 / fh, 5.7 / fw, 3.7 / fh])
    x = np.arange(len(probs))
    w = 0.36
    for k, m in enumerate(("DALIv2LE", "SCBO")):
        for i, p in enumerate(probs):
            c = S["per_problem"][p]["safety_cost"].get(m, {}).get("cost_per_avoided_violation")
            y = x[i] + (k - 0.5) * w
            if c is None:
                tag(ax.text(0, y, "  n/a", va="center", fontsize=6.5, color=V2.MUTED))
            else:
                ax.barh(y, c, height=w * 0.92, color=V2.CLASS_COLOR[m], edgecolor=V2.INK, lw=0.4)
    ax.axvline(0, color=V2.AXIS, lw=0.8)
    g = [p.split("_")[0] for p in probs]
    for i in range(1, len(probs)):
        if g[i] != g[i - 1]:
            ax.axhline(i - 0.5, color="#E3E3E3", lw=0.6, zorder=0)
    ax.set_yticks(x)
    ax.set_yticklabels([V2.config_label(p) for p in probs], fontsize=7)
    ax.set_ylim(len(probs) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("symlog", linthresh=1e-3, linscale=1.2)
    ax.set_xlim(-1.2e-2, 4e-3)
    ax.set_xticks([-1e-2, -3e-3, -1e-3, 0, 1e-3, 3e-3])
    ax.set_xticklabels(["−0.01", "−0.003", "−0.001", "0", "0.001", "0.003"])
    ax.minorticks_off()
    ax.set_xlabel("Regret cost per avoided violation, κ (symlog; negative = lower regret and fewer violations)", fontsize=7.5)
    tag(ax.xaxis.label)
    V2.axes(ax)
    ax.spines["left"].set_visible(False)
    ent = [(plt.Line2D([], [], marker="s", ls="", ms=7, markerfacecolor=V2.CLASS_COLOR["DALIv2LE"], markeredgecolor=V2.INK, markeredgewidth=0.4), "DALI+LE"), "sep",
           (plt.Line2D([], [], marker="s", ls="", ms=7, markerfacecolor=V2.CLASS_COLOR["SCBO"], markeredgecolor=V2.INK, markeredgewidth=0.4), "SCBO")]
    legend(fig, ent, None, 0.10, None, LEG_H, fs=7.0, sample=0.26, gap=0.09)
    save(fig, figdir, "FigS2_safety_cost_v2", dpi)


# ====================================================================== S9 kNN gate (exploratory)
GATE_PROBS = ["g2_f050_bnd", "g2_f020_bnd", "g2_f005_bnd", "g1_f020_bnd", "g3_f020_bnd", "g4_f020_bnd", "g2_f020_int"]


def fig_s9(res, figdir, dpi):
    G = load(res, "gate")["per_problem"]
    fig = plt.figure(figsize=(7.2, 3.4))
    fw, fh = 7.2, 3.4
    ax_a = fig.add_axes([0.55 / fw, 1.15 / fh, 2.85 / fw, 1.85 / fh])
    ax_b = fig.add_axes([4.15 / fw, 1.15 / fh, 2.85 / fw, 1.85 / fh])
    pairs = [("DALIv2LE", "DALIv2LEgate"), ("LEonly", "LEonlygate")]
    for ax, key in zip((ax_a, ax_b), ("V_mean", "regret_mean")):
        for gi, p in enumerate(GATE_PROBS):
            M = G[p]["methods"]
            for k, (m0, m1) in enumerate(pairs):
                V2.dumbbell(ax, gi + (k - 0.5) * 0.34, M[m0][key], M[m1][key], m0)
            V2.marker(ax, gi, M["SCBO"][key], "SCBO", size=26)
        ax.set_xticks(range(len(GATE_PROBS)))
        ax.set_xticklabels([V2.config_label(p, short=True) for p in GATE_PROBS], rotation=32, ha="right",
                           rotation_mode="anchor", fontsize=6.8)
        ax.set_xlim(-0.6, len(GATE_PROBS) - 0.4)
        for gi in range(1, len(GATE_PROBS)):
            ax.axvline(gi - 0.5, color="#E3E3E3", lw=0.6, zorder=0)
        V2.axes(ax)
        ax.tick_params(axis="x", length=0)
    ax_a.set_yscale("symlog", linthresh=10, linscale=0.9)
    ax_a.set_ylim(-0.6, 130)
    ax_a.set_yticks([0, 5, 10, 30, 100])
    ax_a.set_yticklabels(["0", "5", "10", "30", "100"])
    ax_a.minorticks_off()
    ax_a.set_ylabel("Violations in 100 experiments", fontsize=7.5)
    tag(ax_a.yaxis.label)
    ax_b.set_ylim(-0.0005, 0.02)
    ax_b.set_yticks([0, 0.005, 0.01, 0.015, 0.02])
    ax_b.set_ylabel("Feasible regret, r", fontsize=7.5)
    tag(ax_b.yaxis.label)
    V2.title(ax_a, "a", "Violations, without gate → with kNN gate", dx=-0.17, dy=1.04)
    V2.title(ax_b, "b", "Feasible regret, same runs", dx=-0.19, dy=1.04)
    grey = "#595959"
    ent = [(V2.marker_handle("DALIv2LE"), "DALI+LE"), (V2.marker_handle("LEonly"), "LE-only"), "sep",
           (V2.marker_handle("SCBO"), "SCBO (reference)"), "sep",
           (plt.Line2D([], [], marker="o", ls="", ms=6, markerfacecolor=V2.light(grey), markeredgecolor=V2.light(grey)), "without gate"),
           (plt.Line2D([], [], marker="o", ls="", ms=6, markerfacecolor=grey, markeredgecolor="black", markeredgewidth=0.4), "with gate")]
    legend(fig, ent, None, 0.08, None, LEG_H, fs=7.0, sample=0.26, gap=0.09)
    save(fig, figdir, "FigS7_gate_v2", dpi)



# ====================================================================== S5 (2026-10-06) record conditions of Section 2.6
REC_CONDS = [("a", "Default record (200 points)", dict()),
             ("b", "Contaminated record (30 %)", dict(contamination=0.3)),
             ("c", "Reduced record (50 points)", dict(n_warm=50)),
             ("d", "Uniform record (exploratory)", dict(coverage="uniform"))]
REC_STYLE = {"a": ("#222222", "-"), "b": ("#222222", "--"), "c": ("#9A9A9A", "-"), "d": ("#2E75B6", "-")}


def fig_records(res, figdir, dpi, gkey="g3", seed=0):
    """Actual warm starts (seed 0) of the four record conditions compared in Section 2.6,
    for one configuration (default G3, 20 %, interior: all four blocks were run on it).
    Top: projection onto the section plane through c0 (feasibility judged in 6-D: red ring =
    infeasible point).  Bottom: exact 6-D distributions -- distance to the optimum and margin g."""
    import problems_constraint_suite as CS
    from make_fig1_v13 import section_data
    t1, t2, masks, o1, o2, (l1, l2) = section_data(gkey)
    base = CS.ConstrainedSuiteProblem(gkey, "f020", "int")
    e1, e2, _, _ = CS.section_plane(base)
    fig = plt.figure(figsize=(7.2, 4.9))
    fw, fh = 7.2, 4.9
    top = grid_axes(fig, 1, 4, left=0.45, right=0.12, bottom=2.45, top=0.42, wgap=0.22, hgap=0)[0]
    data = {}
    for ax, (letter, title, kw) in zip(top, REC_CONDS):
        prob = CS.ConstrainedSuiteProblem(gkey, "f020", "int", **kw)
        X, f, feas = prob.warm_start_labeled(seed)
        data[letter] = (X, f, feas, prob.name)
        FS.draw_feasible_mask(ax, t1, t2, masks["int20"])
        P = (X - base.c0) @ np.vstack([e1, e2]).T + np.array([o1, o2])
        ax.scatter(P[feas, 0], P[feas, 1], s=5, color="#555555", lw=0, zorder=4)
        if (~feas).any():
            ax.scatter(P[~feas, 0], P[~feas, 1], s=14, facecolors="none", edgecolors="#C8322B", lw=0.8, zorder=5)
        ax.scatter([o1], [o2], marker="*", s=90, color="black", zorder=6, lw=0)
        pad = 0.0 if (o1 or o2) else 0.04
        FS.set_section_window(ax, t1, t2, masks["int20"], pad)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(V2.AXIS); sp.set_linewidth(0.8)
        V2.title(ax, letter, title, dx=0.0, dy=1.03, fs_text=7.5)
        ax.set_xlabel(l1.replace("distance from $c_0$ toward ball B", "toward ball B"), fontsize=7)
        tag(ax.xaxis.label)
        if letter == "a":
            ax.set_ylabel(l2.replace("perpendicular (in subspace)", "perpendicular"), fontsize=7)
            tag(ax.yaxis.label)
        n_in = int(feas.sum()); n_out = int((~feas).sum())
        tag(ax.text(0.03, 0.03, f"{n_in} feasible" + (f", {n_out} infeasible" if n_out else ""), transform=ax.transAxes,
                    fontsize=6.5, color=V2.MUTED, va="bottom", ha="left"))
    # bottom: exact 6-D distributions
    ax_e = fig.add_axes([0.55 / fw, 0.95 / fh, 2.85 / fw, 1.25 / fh])
    ax_f = fig.add_axes([4.0 / fw, 0.95 / fh, 2.85 / fw, 1.25 / fh])
    for letter, title, _ in REC_CONDS:
        X, f, feas, name = data[letter]
        col, ls = REC_STYLE[letter]
        d = np.linalg.norm(X - base.c0, axis=1)
        g = base.g(X)
        for ax, v in ((ax_e, d), (ax_f, g)):
            xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", color=col, ls=ls, lw=1.3)
    ax_f.axvline(0, color="#C8322B", ls=":", lw=0.9)
    for ax in (ax_e, ax_f):
        V2.axes(ax); ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1])
    ax_e.set_xlabel("Distance from the optimum, ||x − c₀||", fontsize=7.5); tag(ax_e.xaxis.label)
    ax_f.set_xlabel("Constraint margin g(x) (> 0 = feasible)", fontsize=7.5); tag(ax_f.xaxis.label)
    ax_e.set_ylabel("Fraction of record points", fontsize=7.5); tag(ax_e.yaxis.label)
    V2.title(ax_e, "e", "Distance to the optimum (6-D)", dx=-0.17, dy=1.04)
    V2.title(ax_f, "f", "Constraint margin (6-D)", dx=-0.13, dy=1.04)
    ent = []
    for k, (letter, title, _) in enumerate(REC_CONDS):
        col, ls = REC_STYLE[letter]
        ent.append((plt.Line2D([], [], color=col, ls=ls, lw=1.3), title.replace(" (exploratory)", "")))
    ent += ["sep", (plt.Line2D([], [], marker="o", ls="", ms=5, markerfacecolor="white", markeredgecolor="#C8322B", markeredgewidth=0.8), "infeasible record point")]
    legend(fig, ent, None, 0.10, None, LEG_H, fs=7.0, sample=0.26, gap=0.09)
    json.dump({k: {"name": v[3], "n": int(len(v[0])), "n_infeasible": int((~v[2]).sum()),
                   "dist_mean": float(np.linalg.norm(v[0] - base.c0, axis=1).mean()),
                   "margin_min": float(base.g(v[0]).min())} for k, v in data.items()},
              open(os.path.join(figdir, "FigS5_record_conditions_v2_data.json"), "w"), indent=1)
    save(fig, figdir, "FigS5_record_conditions_v2", dpi)

# key = current SI number; function names keep the first-pass numbering (fig_s2 = heat maps, ...)
# 2026-10-06: renumbered in the order of first citation in the main text; S5 = new record-condition figure
FIGS = {"S1": fig_s2, "S2": fig_s8, "S3": fig_s3, "S4": fig_s7, "S5": fig_records, "S6": fig_s6, "S7": fig_s9, "S8": fig_s5, "S9": fig_s4}
EXTRA = {"sections": fig_s1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--only", default="")
    ap.add_argument("--pptx", action="store_true", help="write the editable-PowerPoint assets instead of PNG/PDF")
    args = ap.parse_args()
    global PX_MODE
    PX_MODE = args.pptx
    V2.rc(7.5)
    keys = args.only.split(",") if args.only else list(FIGS)
    for k in keys:
        (FIGS.get(k) or EXTRA[k])(args.res, args.figdir, args.dpi)
        plt.close("all")


if __name__ == "__main__":
    main()
