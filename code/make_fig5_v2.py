"""Fig 5 v2 (2026-09-28) -- data + plot-area images for the PowerPoint version.

Same numbers as Fig 5 v1 (make_fig5_mechanism.py).  Writes
  figures/Fig5_v2_data.json          numbers + axes geometry of the images
  figures/Fig5_v2_panel_b_g{1..4}.png  (b) margin g(x) of proposals, interior 20 %, one image per
                                       geometry; HORIZONTAL quantile boxes, rows = methods
  figures/Fig5_v2_panel_c.png          (c) d_kNN per run (12 interior configs x 20 seeds), rows = methods,
                                       log axis, boxes + points
  figures/Fig5_v2_panel_d.png          (d) ablation scatter (5 configurations x 11 methods)
Panel (a) (bars), the shared method column, titles, headers, annotations and legends are
drawn as editable shapes by make_fig5_v2_pptx.js.

Style: class colours (Record-based / Constraint learning / Constraint-unaware / Reference);
in (d) marker shapes distinguish methods (as Fig 3 v2), the two ablation controls keep their
own markers (verifier + uniform: brown hexagon; DALI (no clip): blue x).

Usage: python make_fig5_v2.py [--tier1 DIR] [--ablation PATH] [--figdir DIR]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

import figstyle as FS
from make_fig5_mechanism import ABL, METHODS, config_order, ratio_of

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_COLOR = {"DALIv2LE": "#2E75B6", "DALIv2": "#2E75B6", "LEonly": "#2E75B6", "DALInoclip": "#2E75B6",
               "VerifRandom": "#8C564B", "SCBO": "#1E8C8C",
               "TuRBO": "#7B4FA6", "GA": "#7B4FA6", "BO": "#7B4FA6", "Random": "#7B4FA6", "cTuRBOknown": "#8A6D00"}
MARK = {"DALIv2LE": "o", "DALIv2": "o", "LEonly": "s", "DALInoclip": "x", "VerifRandom": "h", "SCBO": "D",
        "TuRBO": "^", "GA": "v", "BO": "P", "Random": "X", "cTuRBOknown": "*"}
HOLLOW = {"DALIv2", "cTuRBOknown"}
GEOM = {"b": {"fig": (1.75, 2.95), "ax": (0.08, 0.32, 1.60, 2.55)},
        "c": {"fig": (2.30, 2.95), "ax": (0.10, 0.32, 2.10, 2.55)},
        "d": {"fig": (9.90, 3.20), "ax": (0.50, 0.42, 9.30, 2.68)}}   # (d) widened 2026-09-28 (user request)
LABEL = {**FS.LABELS, "VerifRandom": "verifier + uniform", "DALInoclip": "DALI (no clip)"}


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


def hbox(ax, i, q, color, hollow=False, height=0.62):
    """horizontal quantile box at row i: whiskers 5-95 %, box 25-75 %, line = median"""
    q05, q25, q50, q75, q95 = q
    ax.plot([q05, q25], [i, i], color=color, lw=1.2, zorder=3)
    ax.plot([q75, q95], [i, i], color=color, lw=1.2, zorder=3)
    ax.add_patch(Rectangle((q25, i - height / 2), q75 - q25, height, facecolor="white" if hollow else color,
                           edgecolor=color, lw=1.4 if hollow else 0.8, alpha=1.0 if hollow else 0.85, zorder=4))
    ax.plot([q50, q50], [i - height / 2, i + height / 2], color=color if hollow else "black", lw=1.6, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    RES = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")
    ap.add_argument("--tier1", default=os.path.join(RES, "tier1"))
    ap.add_argument("--ablation", default=os.path.join(RES, "h3_ablation", "summary.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    args = ap.parse_args()
    S = json.load(open(os.path.join(args.tier1, "summary.json")))
    R = pd.read_csv(os.path.join(args.tier1, "summary_metrics_per_run.csv"))
    A = json.load(open(args.ablation))
    rows = config_order(list(S["per_problem"]))
    methods = [m for m in METHODS if all(m in S["per_problem"][p]["methods"] for p in rows)]
    n = len(methods)
    out = {"methods": [{"key": m, "label": LABEL[m], "color": CLASS_COLOR[m]} for m in methods], "geom": GEOM}

    # ---------------- (a) candidate feasible fraction (numbers only; bars drawn natively)
    out["a"] = {"configs": [{"name": p, "geometry": p.split("_")[0], "fraction_pct": int(p.split("_")[1][1:]),
                             "location": p.split("_")[2], "uniform": ratio_of(p)} for p in rows],
                "series": {m: [S["per_problem"][p]["methods"][m].get("cand_feasible_mean") for p in rows]
                           for m in ("DALIv2", "DALIv2LE", "SCBO")}}

    # ---------------- (b) margin boxes, interior 20 %, one image per geometry (rows = methods)
    out["b"] = {}
    for g in ("g1", "g2", "g3", "g4"):
        p = f"{g}_f020_int"
        fig, ax = axes_for("b")
        qs = {}
        for i, m in enumerate(methods):
            r = S["per_problem"][p]["methods"][m]
            q = (r["g_q05"], r["g_q25"], r["g_median"], r["g_q75"], r["g_q95"])
            qs[m] = [float(v) for v in q]
            hbox(ax, i, q, CLASS_COLOR[m], hollow=(m == "DALIv2"))
        ax.axvline(0, color=FS.INFEAS_EDGE, lw=1.2, ls="--", zorder=2)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        lo = min(v[0] for v in qs.values()); hi = max(v[4] for v in qs.values())
        pad = 0.06 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        fig.savefig(os.path.join(args.figdir, f"Fig5_v2_panel_b_{g}.png"), dpi=400)
        plt.close(fig)
        out["b"][g] = {"config": p, "quantiles": qs, "xlim": [float(lo - pad), float(hi + pad)]}

    # ---------------- (c) d_kNN per run, interior 12 configs, rows = methods, log axis
    R["loc"] = R["problem"].str.split("_").str[2]
    Ri = R[R["loc"] == "int"]
    fig, ax = axes_for("c")
    cq = {}
    for i, m in enumerate(methods):
        v = Ri.loc[Ri["method"] == m, "dknn_mean"].values
        q = np.percentile(v, [5, 25, 50, 75, 95])
        cq[m] = {"quantiles": [float(x) for x in q], "n_runs": int(len(v))}
        hbox(ax, i, q, CLASS_COLOR[m], hollow=(m == "DALIv2"))
        ax.scatter(v, np.full(len(v), i) + np.random.RandomState(i).uniform(-0.28, 0.28, len(v)), s=4,
                   color="black", alpha=0.25, zorder=6, lw=0)
    ax.set_xscale("log")
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    fig.savefig(os.path.join(args.figdir, "Fig5_v2_panel_c.png"), dpi=400)
    plt.close(fig)
    out["c"] = {"per_method": cq, "n_configs": int(Ri["problem"].nunique())}

    # ---------------- (d) ablation scatter
    abl_probs = [p for p in A["per_problem"] if "VerifRandom" in A["per_problem"][p]["methods"]]
    fig, ax = axes_for("d")
    pts, xs, ys = [], [], []
    for m in ABL:
        for p in abl_probs:
            r = A["per_problem"][p]["methods"].get(m)
            if r is None:
                continue
            x, y = r["dknn_mean"], r["V_mean"]
            xs.append(x); ys.append(y)
            pts.append({"method": m, "config": p, "dknn": float(x), "V": float(y), "cand_feasible": r.get("cand_feasible_mean")})
            ctrl = m in ("VerifRandom", "DALInoclip")
            hollow = m in HOLLOW
            col = CLASS_COLOR[m]
            if m == "DALInoclip":
                ax.scatter([x], [y], marker="x", s=70, color=col, linewidths=2.0, zorder=6)
            else:
                ax.scatter([x], [y], marker=MARK[m], s=95 if ctrl else (75 if m != "cTuRBOknown" else 190),
                           facecolors="white" if hollow else col, edgecolors=col if hollow else "black",
                           linewidths=1.6 if hollow else 0.6, alpha=1.0 if (ctrl or m.startswith("DALI") or m == "LEonly") else 0.8,
                           zorder=6 if ctrl else 4)
    rho, pv = spearmanr(xs, ys)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 6.0); ax.set_ylim(-3, 100)
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 3, 5]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "3", "5"])
    ax.minorticks_off()
    fig.savefig(os.path.join(args.figdir, "Fig5_v2_panel_d.png"), dpi=400)
    plt.close(fig)
    vr = [A["per_problem"][p]["methods"]["VerifRandom"] for p in abl_probs]
    out["d"] = {"configs": abl_probs, "points": pts, "spearman_rho": float(rho), "spearman_p": float(pv),
                "xlim": [0.15, 6.0], "ylim": [-3, 100],
                "verif_uniform_mean": {"dknn": float(np.mean([r["dknn_mean"] for r in vr])),
                                       "V": float(np.mean([r["V_mean"] for r in vr])),
                                       "cand_feasible": float(np.mean([r["cand_feasible_mean"] for r in vr]))},
                "labels": {m: LABEL[m] for m in ABL}, "colors": {m: CLASS_COLOR[m] for m in ABL}}

    path = os.path.join(args.figdir, "Fig5_v2_data.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path, "| (d) points", len(pts), "rho %.2f p %.1e" % (rho, pv))


if __name__ == "__main__":
    main()
