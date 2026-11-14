"""Shared figure style for the DALI safety paper (paper 2).

Inherits the paper-1 manuscript style (make_revision_figs.py / Figure_final):
fixed method colours, DALI emphasised (thick line + hollow marker), large
fonts, bold panel letters, +/- sd bands, shared boxed legend, white
background, 300 dpi PNG + vector PDF.  Every make_fig*.py imports ONLY this
module for styling so that all figures read as one set.

Constraint conventions (paper-1 FigR1): feasible region = green fill
(alpha 0.13) with green boundary line; infeasible proposals = red outline;
warm-start data = small grey dots; c0 / optimum = star.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- methods
# keys = method names used in the result CSVs (paper-1 names kept: DALIv2,
# DALIv2LE, LEonly, TuRBO, GA, BO, Random; new: SCBO, cTuRBOknown)
COLORS = {
    "DALIv2":      "#1f77b4",   # DALI (paper-1 colour)
    "DALIv2LE":    "#1f77b4",   # DALI + LE refiner (same hue, marker + thick)
    "LEonly":      "#9ecae1",   # diffusion removed (paper-1 "DALI base" slot)
    "TuRBO":       "#9467bd",
    "GA":          "#d62728",
    "BO":          "#ff7f0e",
    "Random":      "#7f7f7f",
    "SCBO":        "#17becf",   # constraint-learning baseline (new)
    "cTuRBOknown": "#000000",   # oracle upper reference (new)
    # H3 ablation controls (Fig 5d only)
    "VerifRandom": "#8c564b",   # diffusion removed: uniform candidates + verifier
    "DALInoclip":  "#1f77b4",   # DALI without the clip step (DALI hue, 'x' marker)
}
LABELS = {
    "DALIv2": "DALI", "DALIv2LE": "DALI+LE", "LEonly": "LE-only",
    "TuRBO": "TuRBO", "GA": "GA", "BO": "BO", "Random": "Random",
    "SCBO": "SCBO", "cTuRBOknown": "cTuRBO (known)",
    "VerifRandom": "verifier + uniform", "DALInoclip": "DALI (no clip)",
}
# scatter markers (Fig 3, Fig 5d, Fig 6)
MARKERS = {"DALIv2LE": "o", "DALIv2": "o", "LEonly": "s", "SCBO": "D", "TuRBO": "^", "GA": "v",
           "BO": "P", "Random": "X", "cTuRBOknown": "*", "VerifRandom": "h", "DALInoclip": "x"}
METHOD_ORDER = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO",
                "Random", "cTuRBOknown"]
LW = {"DALIv2LE": 3.0, "DALIv2": 2.3}

# constraint / geometry conventions
FEAS_COLOR = "#2ca02c"
FEAS_ALPHA = 0.13
FEAS_EDGE_LW = 2.0
INFEAS_EDGE = "#d62728"
WARM_COLOR = "0.45"
C0_MARKER = dict(marker="*", s=230, c="black", zorder=7, linewidths=0)
OUTSIDE_COLOR = "0.92"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13.5,
    "axes.linewidth": 1.1, "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------- main-text conventions (v2, 2026-09-16)
# One vocabulary for "which runs does this panel show".  Tier 1 = 4 geometries x
# 4 conditions (feasible fraction 50 / 20 / 5 % with the optimum interior, and
# 20 % with the optimum on the boundary) = 16 configurations x 20 seeds.
SUBSET = {
    "all16": "All 16 configurations",
    "int50": "Interior, feasible fraction 50 %",
    "int20": "Interior, feasible fraction 20 %",
    "int05": "Interior, feasible fraction 5 %",
    "bnd20": "Boundary, feasible fraction 20 %",
    "int12": "Interior (12 configurations)",
    "bnd4":  "Boundary (4 configurations)",
    "tier2": "Boundary, feasible fraction 50 % and 5 % (Tier 2)",
}
MAIN_RC = {"font.size": 18, "axes.titlesize": 19, "axes.labelsize": 20,
           "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 17}
PANEL_FS = 26


def main_rc():
    """Font sizes for the main-text figures (Fig 2-7)."""
    plt.rcParams.update(MAIN_RC)


def clean_axes(ax, xbound=None):
    """Nature-style axes: no top/right spines, no grid, 1.1-pt spines, 6-pt ticks."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    if xbound is not None:
        ax.spines["bottom"].set_bounds(*xbound)
    ax.grid(False)
    ax.tick_params(width=1.1, length=6)


def subtitle(ax, key_or_text, pad=8):
    """Left-aligned panel subtitle in the fixed vocabulary (SUBSET key or free text)."""
    ax.set_title(SUBSET.get(key_or_text, key_or_text), loc="left", fontsize=MAIN_RC["axes.titlesize"], pad=pad)


def end_labels(ax, x_end, ys, texts, colors, min_sep, x_text, y_max=None, y_min=None, fs=17):
    """Direct labels at the right end of curves (sorted by end value, pushed apart by
    min_sep in data units); a thin grey leader marks labels that had to move."""
    ys = np.asarray(ys, float)
    order = np.argsort(ys)
    pos = ys[order].copy()
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < min_sep:
            pos[i] = pos[i - 1] + min_sep
    if y_max is not None and pos.max() > y_max:
        pos -= pos.max() - y_max
    if y_min is not None and pos.min() < y_min:
        pos += y_min - pos.min()
    out = np.empty_like(pos)
    out[order] = pos
    for y, yp, t, c in zip(ys, out, texts, colors):
        if abs(yp - y) > 1e-9:
            ax.plot([x_end, x_text - 0.4 * (x_text - x_end)], [y, yp], color="0.6", lw=0.8, clip_on=False, zorder=1)
        ax.text(x_text, yp, t, color=c, fontsize=fs, va="center", ha="left", clip_on=False)


def dumbbell(ax, x, v_ref, v_new, m, s_ref=120, s_new=70, lw=2.0, ls="-"):
    """Paired-condition glyph: hollow marker at the reference value, filled marker at
    the changed value, joined by a line with an arrowhead pointing at the change."""
    c = COLORS[m]
    if abs(v_new - v_ref) > 1e-9:
        ax.annotate("", xy=(x, v_new), xytext=(x, v_ref),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, ls=ls, shrinkA=5, shrinkB=5,
                                    mutation_scale=12), zorder=3)
    ax.scatter([x], [v_ref], s=s_ref, marker="o", facecolors="white", edgecolors=c, linewidths=2.2, zorder=4)
    ax.scatter([x], [v_new], s=s_new, marker="o", facecolors=c, edgecolors="black", linewidths=0.6, zorder=5)


# ---------------------------------------------------------------- helpers
def panel_label(ax, letter, dx=-0.10, dy=1.06, fontsize=None):
    """Bold lower-case panel letter (Nature style: 'a', not '(a)')."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=fontsize or PANEL_FS, fontweight="bold", va="bottom", ha="left")


def style_line(m):
    """Line kwargs for method m (paper-1 conventions)."""
    kw = dict(color=COLORS[m], lw=LW.get(m, 1.9), label=LABELS[m])
    if m == "DALIv2LE":
        kw.update(marker="o", markevery=2, ms=6, markerfacecolor="white",
                  markeredgecolor=COLORS[m], markeredgewidth=1.4, zorder=6)
    if m == "DALIv2":
        kw.update(ls="-", zorder=5)
    if m == "LEonly":
        kw.update(ls="--")
    if m == "cTuRBOknown":
        kw.update(ls=":", lw=1.8)
    return kw


def band(ax, x, M, m, value_transform=None):
    """Mean line + (+/- sd) band for a (n_seeds, n_iter) matrix."""
    mu, sd = M.mean(axis=0), M.std(axis=0, ddof=1)
    if value_transform:
        mu = value_transform(mu)
    ax.plot(x, mu, **style_line(m))
    ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.13, lw=0)
    return mu


def shared_legend(fig, methods, y=-0.13, ncol=None, extra_handles=()):
    handles = [plt.Line2D([], [], **style_line(m)) for m in methods]
    handles += list(extra_handles)
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles),
               bbox_to_anchor=(0.5, y), frameon=True, edgecolor="0.7",
               fancybox=True, borderpad=0.7, columnspacing=1.6)


def legend_handles_constraint():
    """Legend proxies for the constraint conventions."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=FEAS_COLOR, alpha=0.35, edgecolor=FEAS_COLOR, lw=1.5,
              label="feasible region (hidden)"),
        plt.Line2D([], [], marker="o", ls="", ms=4, color=WARM_COLOR,
                   label="warm-start data (feasible in 6-D; projected)"),
        plt.Line2D([], [], marker="*", ls="", ms=13, color="black",
                   label="$c_0$ (optimum)"),
        plt.Line2D([], [], marker="s", ls="", markersize=7, markerfacecolor="0.75",
                   markeredgecolor=INFEAS_EDGE, markeredgewidth=1.4,
                   label="red outline = infeasible"),
    ]


def draw_feasible_mask(ax, t1, t2, mask, outside=None):
    """Fill the feasible set (mask==1) green with a green boundary; grey out
    points outside the box (mask==-1) if present."""
    m = np.asarray(mask, dtype=float)
    if (m < 0).any():
        ax.contourf(t1, t2, (m < 0).astype(float), levels=[0.5, 1.5],
                    colors=[OUTSIDE_COLOR], zorder=0)
    feas = (m > 0.5).astype(float)
    ax.contourf(t1, t2, feas, levels=[0.5, 1.5], colors=[FEAS_COLOR],
                alpha=FEAS_ALPHA, zorder=1)
    ax.contour(t1, t2, feas, levels=[0.5], colors=[FEAS_COLOR],
               linewidths=FEAS_EDGE_LW, zorder=2)


def savefig(fig, outdir, stem, dpi=300, pdf=True, bbox_inches="tight"):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{stem}.png")
    fig.savefig(png, dpi=dpi, bbox_inches=bbox_inches)
    if pdf:
        fig.savefig(os.path.join(outdir, f"{stem}.pdf"), bbox_inches=bbox_inches)
    print(f"wrote {png}")
    return png


# --- 2-D section windows (added 2026-09-23; fixes the truncated G3 section) ---------
# The G3 section plane is rotated (c0 -> centre of ball B); the box corner that
# holds ball B lies 1.34 from c0, beyond the default half-width 1.0 of
# problems_constraint_suite.section_grid.  Axis-aligned planes never need more
# than 1.0.
SECTION_HALF = {"g3": 1.6}


def section_extent(gkey, n):
    """Half-width and grid size to use for section_grid so the plane covers the
    whole box; the grid step is kept the same as for half = 1.0."""
    half = SECTION_HALF.get(gkey, 1.0)
    return half, int(round(n * half))


def set_section_window(ax, t1, t2, mask, pad=0.0):
    """Square, equal-aspect window covering every grid point inside the box
    (mask >= 0).  Padding outside the grid is filled with the outside colour."""
    m = np.asarray(mask)
    ins = np.argwhere(m >= 0)
    x0, x1 = t1[ins[:, 1].min()] - pad, t1[ins[:, 1].max()] + pad
    y0, y1 = t2[ins[:, 0].min()] - pad, t2[ins[:, 0].max()] + pad
    s = max(x1 - x0, y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ax.set_xlim(cx - s / 2, cx + s / 2)
    ax.set_ylim(cy - s / 2, cy + s / 2)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    if (m < 0).any():
        # outside the grid: outside colour; inside the box: white (drawn explicitly,
        # since draw_feasible_mask only paints the mask == -1 region)
        ax.set_facecolor(OUTSIDE_COLOR)
        ax.contourf(t1, t2, (m >= 0).astype(float), levels=[0.5, 1.5], colors=["white"], zorder=0.5)
