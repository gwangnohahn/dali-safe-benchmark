"""Shared v2 figure style (2026-09-29) for matplotlib-drawn figures of paper 2.

Mirrors the conventions of the editable main-text figures (Fig 1-8 v2 / v8 / v13):
  * four class colours -- Record-based blue, Constraint learning teal,
    Constraint-unaware purple, Reference (cTuRBO) tan -- with light class bands;
  * methods within a class differ by line style (solid DALI+LE / TuRBO / SCBO,
    dashed DALI / GA, dotted LE-only / BO, dash-dot Random, black dotted cTuRBO)
    or by marker (DALI+LE filled circle, DALI hollow circle, LE-only square,
    SCBO diamond, TuRBO / GA triangles, BO plus, Random cross, cTuRBO hollow star);
  * Arial (Liberation Sans as the metric-compatible fallback), thin grey axes,
    no grids, bold panel letter + short plain title, boxed legend with dotted
    class separators and no class names.
Used by make_si_figs_v2.py.  figstyle.py (v1 conventions) is left untouched.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import figstyle as FS

LABELS = FS.LABELS
CLASS_COLOR = {"DALIv2LE": "#2E75B6", "DALIv2": "#2E75B6", "LEonly": "#2E75B6", "SCBO": "#1E8C8C",
               "TuRBO": "#7B4FA6", "GA": "#7B4FA6", "BO": "#7B4FA6", "Random": "#7B4FA6", "cTuRBOknown": "#8A6D00",
               "DALIv2LEgate": "#2E75B6", "LEonlygate": "#2E75B6"}
BAND = {"DALIv2LE": "#E4EEF8", "SCBO": "#E2F3F1", "TuRBO": "#EFE8F5", "cTuRBOknown": "#FFF4D6"}
CLASSES = [["DALIv2LE", "DALIv2", "LEonly"], ["SCBO"], ["TuRBO", "GA", "BO", "Random"], ["cTuRBOknown"]]
CLASS_OF = {m: grp[0] for grp in CLASSES for m in grp}
LINE = {"DALIv2LE": ("-", 1.9), "DALIv2": ("--", 1.4), "LEonly": (":", 1.6), "SCBO": ("-", 1.5),
        "TuRBO": ("-", 1.5), "GA": ("--", 1.4), "BO": (":", 1.6), "Random": ("-.", 1.4), "cTuRBOknown": (":", 1.1)}
LINE_COLOR = dict(CLASS_COLOR, cTuRBOknown="#000000")     # reference curve is black dotted (Fig 4 / 6 / 7 v2)
MARK = {"DALIv2LE": "o", "DALIv2": "o", "LEonly": "s", "SCBO": "D", "TuRBO": "^", "GA": "v", "BO": "P",
        "Random": "X", "cTuRBOknown": "*"}
HOLLOW = {"DALIv2", "cTuRBOknown"}
INK, MUTED, AXIS = "#222222", "#595959", "#808080"


def _font():
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Arial", "Liberation Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "sans-serif"


def rc(base=8.0):
    fam = _font()
    plt.rcParams.update({
        "font.family": fam,
        "font.size": base, "axes.titlesize": base + 1, "axes.labelsize": base + 0.5,
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5, "legend.fontsize": base,
        "axes.linewidth": 0.8, "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
        "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
    })


def light(c, f=0.55):
    """Tint towards white by fraction f."""
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def axes(ax, ybound=None, xbound=None):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(0.8)
    if ybound is not None:
        ax.spines["left"].set_bounds(*ybound)
    if xbound is not None:
        ax.spines["bottom"].set_bounds(*xbound)
    ax.tick_params(colors=MUTED, width=0.8, length=3)
    ax.grid(False)


def title(ax, letter, text="", dx=0.0, dy=1.02, fs_letter=11, fs_text=8.5, ha="left"):
    """Bold panel letter followed by a short plain title (Fig 2-8 v2 convention)."""
    t = ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs_letter, fontweight="bold", va="bottom", ha=ha)
    t.set_gid("px")          # editable in the PowerPoint export (si_pptx_assets.py)
    if text:
        a = ax.annotate(" " + text, xy=(1, 0), xycoords=t, xytext=(0, 0), textcoords="offset points",
                        fontsize=fs_text, va="bottom", ha="left", annotation_clip=False)
        a.set_gid("px")


def line_kw(m):
    ls, lw = LINE[m]
    return dict(color=LINE_COLOR[m], ls=ls, lw=lw, solid_capstyle="round", dash_capstyle="round",
                zorder=6 if m == "DALIv2LE" else 4)


def marker(ax, x, y, m, size=30, state="new"):
    """Method marker; state 'ref' = light tint (dumbbell reference), 'new' = full colour."""
    c = CLASS_COLOR[m]
    col = light(c) if state == "ref" else c
    s = size * (2.0 if m == "cTuRBOknown" else 1.0)
    if m in HOLLOW:
        ax.scatter([x], [y], marker=MARK[m], s=s, facecolors="white", edgecolors=col, linewidths=1.3, zorder=5)
    else:
        ax.scatter([x], [y], marker=MARK[m], s=s, facecolors=col, edgecolors="black" if state == "new" else col,
                   linewidths=0.4, zorder=5)


def dumbbell(ax, x, v_ref, v_new, m):
    """Reference (light) -> changed (full colour) with an arrow head, as Fig 6 v2."""
    c = CLASS_COLOR[m]
    if abs(v_new - v_ref) > 1e-12:
        ax.annotate("", xy=(x, v_new), xytext=(x, v_ref),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, shrinkA=3.5, shrinkB=3.5, mutation_scale=7),
                    zorder=3)
    marker(ax, x, v_ref, m, state="ref")
    marker(ax, x, v_new, m, state="new")


def marker_handle(m, state="new", ms=6):
    c = CLASS_COLOR[m]
    col = light(c) if state == "ref" else c
    if m in HOLLOW:
        return plt.Line2D([], [], marker=MARK[m], ls="", ms=ms * (1.5 if m == "cTuRBOknown" else 1),
                          markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.3, label=LABELS[m])
    return plt.Line2D([], [], marker=MARK[m], ls="", ms=ms, markerfacecolor=col,
                      markeredgecolor="black" if state == "new" else col, markeredgewidth=0.4, label=LABELS[m])


def line_handle(m):
    return plt.Line2D([], [], label=LABELS[m], **line_kw(m))


def text_width(label, fs):
    """Rendered width of a label in inches (TextPath with the figure font), replacing the earlier
    per-character estimate that left empty space at the right of the legend boxes."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    bb = TextPath((0, 0), label, size=fs, prop=FontProperties(family=_font())).get_extents()
    return bb.width / 72.0 + fs * 0.0012


def legend_width(entries, fs=7.5, sample=0.28, gap=0.10, pad=0.10):
    w = 2 * pad
    for e in entries:
        w += gap if e == "sep" else sample + 0.05 + text_width(e[1], fs) + gap
    return w - gap


def legend_box(fig, entries, x, y, w, h, fs=7.5, sample=0.28, gap=0.10, pad=0.10):
    """Boxed legend laid out in figure inches: entries = list of (handle, label) or
    the string 'sep' for a dotted vertical separator (class boundary).  x, y, w, h are
    the box rectangle in inches from the bottom-left corner of the figure; x = None
    centres the box, w = None fits it to the entries."""
    fw, fh = fig.get_size_inches()
    if w is None:
        w = legend_width(entries, fs, sample, gap, pad)
    if x is None:
        x = (fw - w) / 2
    ax = fig.add_axes([x / fw, y / fh, w / fw, h / fh])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.add_patch(Rectangle((0.004, 0.004), w - 0.008, h - 0.008, facecolor="white", edgecolor="#BFBFBF",
                           lw=0.8, zorder=0))
    cx, cy = pad, h / 2
    for e in entries:
        if e == "sep":
            ax.plot([cx + gap / 2, cx + gap / 2], [0.12 * h, 0.88 * h], color="#9A9A9A", ls=(0, (1, 2)), lw=0.8)
            cx += gap
            continue
        handle, label = e
        hx = cx + sample / 2
        if isinstance(handle, plt.Line2D) and handle.get_linestyle() != "None" and handle.get_marker() in ("", "None", None):
            ax.plot([cx, cx + sample], [cy, cy], color=handle.get_color(), ls=handle.get_linestyle(),
                    lw=handle.get_linewidth(), solid_capstyle="round", dash_capstyle="round")
        elif isinstance(handle, plt.Line2D):
            ax.plot([hx], [cy], marker=handle.get_marker(), ms=handle.get_markersize(), ls="",
                    markerfacecolor=handle.get_markerfacecolor(), markeredgecolor=handle.get_markeredgecolor(),
                    markeredgewidth=handle.get_markeredgewidth())
        else:   # patch-like: (facecolor, edgecolor, hatch)
            fc, ec, hatch = handle
            ax.add_patch(Rectangle((cx + 0.02, cy - 0.06), sample - 0.04, 0.12, facecolor=fc, edgecolor=ec,
                                   lw=0.6, hatch=hatch))
        ax.text(cx + sample + 0.05, cy, label, fontsize=fs, va="center", ha="left")
        cx += sample + 0.05 + text_width(label, fs) + gap
    return ax


def method_entries(methods, kind="line"):
    """Legend entries for a method list with 'sep' between classes."""
    out = []
    prev = None
    for m in methods:
        cls = CLASS_OF[m]
        if prev is not None and cls != prev:
            out.append("sep")
        out.append((line_handle(m) if kind == "line" else marker_handle(m), LABELS[m]))
        prev = cls
    return out


def class_band_x(ax, methods, y0=1.0, hgt=0.03):
    """Thin class-colour strip above column headers (axes fraction)."""
    n = len(methods)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and CLASS_OF[methods[j + 1]] == CLASS_OF[methods[i]]:
            j += 1
        r = ax.add_patch(Rectangle((i / n + 0.004, y0), (j - i + 1) / n - 0.008, hgt, transform=ax.transAxes,
                                   facecolor=CLASS_COLOR[methods[i]], edgecolor="none", clip_on=False))
        r.set_gid("px_band")
        i = j + 1


def config_label(n, short=False):
    g, r, loc = n.split("_")[:3]
    pct = f"{int(r[1:])} %"
    where = "interior" if loc == "int" else "near bnd." if short else "near boundary"
    return f"{g.upper()}, {pct}, {where}"
