"""Fig 6 v2 (2026-09-28) -- data + plot-area images for the PowerPoint version.

Same numbers as Fig 6 v1 (make_fig6_limits.py).  Panels:
  a  interior 20 % -> near-boundary 20 %, V per geometry (dumbbells, symlog y)
  b  contamination 0 / 10 / 30 % (interior 20 %): mean over geometries (lines) + geometries (grey dots)
  c  record 200 -> 50 points (G2, G3, interior 20 %): V only (regret change is quoted in the text)
  d  near-boundary optimum, feasible fraction 50 / 20 / 5 %, one image per geometry (lines; numbers
     at the line ends are added in PowerPoint)
  e, f  clustered -> uniform record (interior 20 %, exploratory): V and regret (dumbbells)
Encoding (as Fig 3/4 v2): class colours; marker shape = method; line style = method within a
class (solid DALI+LE/TuRBO, dashed DALI/GA, dotted LE-only/BO, dash-dot Random); dumbbell:
light marker = reference condition -> full-colour marker = changed condition, arrow head.

Writes figures/Fig6_v2_data.json and figures/Fig6_v2_panel_{a,b,c,d_g1..g4,e,f}.png.
Usage: python make_fig6_v2.py [--res DIR] [--figdir DIR]
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
GEOMS = ["g1", "g2", "g3", "g4"]
M_ALL = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
M_A = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA"]
M_C = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO"]
M_E = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "cTuRBOknown"]
CLASS_COLOR = {"DALIv2LE": "#2E75B6", "DALIv2": "#2E75B6", "LEonly": "#2E75B6", "SCBO": "#1E8C8C",
               "TuRBO": "#7B4FA6", "GA": "#7B4FA6", "BO": "#7B4FA6", "Random": "#7B4FA6", "cTuRBOknown": "#8A6D00"}
MARK = {"DALIv2LE": "o", "DALIv2": "o", "LEonly": "s", "SCBO": "D", "TuRBO": "^", "GA": "v", "BO": "P", "Random": "X", "cTuRBOknown": "*"}
LINE = {"DALIv2LE": "-", "DALIv2": "--", "LEonly": ":", "SCBO": "-", "TuRBO": "-", "GA": "--", "BO": ":", "Random": "-.", "cTuRBOknown": ":"}
LINE_NAME = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot"}
HOLLOW = {"DALIv2", "cTuRBOknown"}
GEOM = {"a": {"fig": (4.30, 2.90), "ax": (0.55, 0.45, 3.65, 2.30)},
        "b": {"fig": (2.70, 2.90), "ax": (0.55, 0.45, 2.05, 2.30)},
        "c": {"fig": (2.20, 2.90), "ax": (0.55, 0.45, 1.55, 2.30)},
        "d": {"fig": (2.55, 2.50), "ax": (0.50, 0.42, 1.95, 1.95)},
        "e": {"fig": (5.95, 2.90), "ax": (0.60, 0.45, 5.25, 2.30)}}


def light(c, f=0.55):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def axes_for(key):
    fw, fh = GEOM[key]["fig"]
    x0, y0, w, h = GEOM[key]["ax"]
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#808080")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="#595959", labelsize=7.5, width=0.8, length=3)
    return fig, ax


def glyph(ax, x, y, m, state, size=38):
    """method marker; state 'ref' = light, 'new' = full colour"""
    c = CLASS_COLOR[m]
    col = light(c) if state == "ref" else c
    s = size * (1.9 if m == "cTuRBOknown" else 1.0)
    if m in HOLLOW:
        ax.scatter([x], [y], marker=MARK[m], s=s, facecolors="white", edgecolors=col, linewidths=1.6, zorder=5)
    else:
        ax.scatter([x], [y], marker=MARK[m], s=s, facecolors=col, edgecolors="black" if state == "new" else col, linewidths=0.5, zorder=5)


def dumbbell(ax, x, v_ref, v_new, m):
    c = CLASS_COLOR[m]
    if abs(v_new - v_ref) > 1e-9:
        ax.annotate("", xy=(x, v_new), xytext=(x, v_ref),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, ls=LINE[m] if m == "DALIv2" else "-",
                                    shrinkA=4, shrinkB=4, mutation_scale=8), zorder=3)
    glyph(ax, x, v_ref, m, "ref")
    glyph(ax, x, v_new, m, "new")


def dumbbell_panel(ax, groups, methods, get_ref, get_new):
    nm = len(methods)
    w = 0.82 / nm
    vals = {}
    for gi, g in enumerate(groups):
        for k, m in enumerate(methods):
            xx = gi + (k - (nm - 1) / 2) * w
            r, n = get_ref(g, m), get_new(g, m)
            vals[f"{g}|{m}"] = {"ref": float(r), "new": float(n)}
            dumbbell(ax, xx, r, n, m)
    ax.set_xticks(range(len(groups)))
    ax.set_xlim(-0.5, len(groups) - 0.5)
    for gi in range(1, len(groups)):
        ax.axvline(gi - 0.5, color="#DDDDDD", lw=0.8, zorder=0)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    args = ap.parse_args()
    load = lambda b: json.load(open(os.path.join(args.res, b, "summary.json")))["per_problem"]
    T1, T2, C, W, U = load("tier1"), load("tier2"), load("h4_contam"), load("h4_warm"), load("h3_uniform")
    stat = lambda P, p, m, k: P[p]["methods"][m][k]
    GL = {"g1": "G1 Convex", "g2": "G2 Non-convex", "g3": "G3 Disconnected", "g4": "G4 Thin band"}
    out = {"methods": [{"key": m, "label": FS.LABELS[m], "color": CLASS_COLOR[m], "line": LINE_NAME[LINE[m]]} for m in M_ALL],
           "geom": GEOM}

    # ---- (a)
    fig, ax = axes_for("a")
    out["a"] = dumbbell_panel(ax, GEOMS, M_A, lambda g, m: stat(T1, f"{g}_f020_int", m, "V_mean"),
                              lambda g, m: stat(T1, f"{g}_f020_bnd", m, "V_mean"))
    ax.set_xticklabels([GL[g] for g in GEOMS])
    ax.set_yscale("symlog", linthresh=10, linscale=0.9)
    ax.set_ylim(-0.6, 130); ax.set_yticks([0, 5, 10, 30, 100]); ax.set_yticklabels(["0", "5", "10", "30", "100"]); ax.minorticks_off()
    fig.savefig(os.path.join(args.figdir, "Fig6_v2_panel_a.png"), dpi=400); plt.close(fig)

    # ---- (b)
    fig, ax = axes_for("b")
    levels = [0, 10, 30]
    out["b"] = {}
    for m in M_C:
        vals = np.array([[stat(T1 if c == 0 else C, f"{g}_f020_int" + ("" if c == 0 else f"_c{c}"), m, "V_mean") for c in levels] for g in GEOMS])
        ax.plot(levels, vals.mean(axis=0), color=CLASS_COLOR[m], ls=LINE[m], lw=2.2 if m == "DALIv2LE" else 1.7, zorder=3)
        for gi in range(4):
            ax.scatter(levels, vals[gi], s=9, color="#9A9A9A", alpha=0.7, zorder=2, lw=0)
        out["b"][m] = {"mean": [float(v) for v in vals.mean(axis=0)], "per_geometry": vals.tolist()}
    ax.set_xticks(levels); ax.set_xlim(-2, 32)
    ax.set_ylim(-0.5, 16); ax.set_yticks([0, 4, 8, 12, 16])
    fig.savefig(os.path.join(args.figdir, "Fig6_v2_panel_b.png"), dpi=400); plt.close(fig)

    # ---- (c) V only
    fig, ax = axes_for("c")
    out["c"] = dumbbell_panel(ax, ["g2", "g3"], M_C, lambda g, m: stat(T1, f"{g}_f020_int", m, "V_mean"),
                              lambda g, m: stat(W, f"{g}_f020_int_w50", m, "V_mean"))
    out["c_regret"] = {f"{g}|{m}": {"ref": float(stat(T1, f"{g}_f020_int", m, "regret_mean")), "new": float(stat(W, f"{g}_f020_int_w50", m, "regret_mean"))}
                       for g in ("g2", "g3") for m in M_C}
    ax.set_xticklabels(["G2", "G3"])
    ax.set_ylim(-0.3, 8); ax.set_yticks([0, 2, 4, 6, 8])
    fig.savefig(os.path.join(args.figdir, "Fig6_v2_panel_c.png"), dpi=400); plt.close(fig)

    # ---- (d) one image per geometry
    ratios = [0.50, 0.20, 0.05]
    out["d"] = {"ratios": ratios, "ylim": [-2, 100], "xlim": [-0.15, 2.15]}
    for g in GEOMS:
        fig, ax = axes_for("d")
        out["d"][g] = {}
        for m in M_ALL:
            vals = [stat(T1 if r == 0.20 else T2, f"{g}_f{int(r * 100):03d}_bnd", m, "V_mean") for r in ratios]
            col = CLASS_COLOR[m] if m != "cTuRBOknown" else "black"
            ax.plot(range(3), vals, color=col, ls=LINE[m], lw={"DALIv2LE": 2.4, "cTuRBOknown": 1.2}.get(m, 1.7), zorder=3 if m == "DALIv2LE" else 2)
            out["d"][g][m] = [float(v) for v in vals]
        ax.set_xticks(range(3)); ax.set_xticklabels(["50", "20", "5"]); ax.set_xlim(-0.15, 2.15)
        ax.set_ylim(-2, 100); ax.set_yticks([0, 25, 50, 75, 100])
        fig.savefig(os.path.join(args.figdir, f"Fig6_v2_panel_d_{g}.png"), dpi=400); plt.close(fig)

    # ---- (e)(f)
    for key, k2, ylim, yt in (("e", "V_mean", (-1, 45), [0, 10, 20, 30, 40]), ("f", "regret_mean", (-0.02, 0.9), [0, 0.2, 0.4, 0.6, 0.8])):
        fig, ax = axes_for("e")
        out[key] = dumbbell_panel(ax, GEOMS, M_E, lambda g, m: stat(T1, f"{g}_f020_int", m, k2),
                                  lambda g, m: stat(U, f"{g}_f020_int_uniform", m, k2))
        ax.set_xticklabels([GL[g] for g in GEOMS])
        ax.set_ylim(*ylim); ax.set_yticks(yt)
        fig.savefig(os.path.join(args.figdir, f"Fig6_v2_panel_{key}.png"), dpi=400); plt.close(fig)

    path = os.path.join(args.figdir, "Fig6_v2_data.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    print("(a) DALI+LE int->bnd:", {g: (round(out['a'][f'{g}|DALIv2LE']['ref'], 1), round(out['a'][f'{g}|DALIv2LE']['new'], 1)) for g in GEOMS})
    print("(d) G2 DALI:", out["d"]["g2"]["DALIv2"], " (e) DALI+LE uniform V:", [round(out['e'][f'{g}|DALIv2LE']['new'], 1) for g in GEOMS])


if __name__ == "__main__":
    main()
