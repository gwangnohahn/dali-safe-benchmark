"""Fig 7 v2 (2026-09-28) -- data + plot-area images for the PowerPoint version.

Same numbers as Fig 7 v1 (make_fig7_realdata.py) for the violation curves; the bottom row
changes from improvement curves to the FINAL improvement over the warm-start best per method
(mean over 10 seeds, 95 % bootstrap CI over seeds), drawn natively in PowerPoint as bars
beside a shared method column (as Fig 2 / 4 / 5).

Writes figures/Fig7_v2_data.json and figures/Fig7_v2_panel_{sfs,nomc,ccpp}.png
(cumulative violations, mean and 95 % CI over seeds; class colours, line style = method).
Usage: python make_fig7_v2.py [--results DIR] [--params PATH] [--figdir DIR]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
CLASS_COLOR = {"DALIv2LE": "#2E75B6", "DALIv2": "#2E75B6", "LEonly": "#2E75B6", "SCBO": "#1E8C8C",
               "TuRBO": "#7B4FA6", "GA": "#7B4FA6", "BO": "#7B4FA6", "Random": "#7B4FA6", "cTuRBOknown": "#8A6D00"}
LINE = {"DALIv2LE": "-", "DALIv2": "--", "LEonly": ":", "SCBO": "-", "TuRBO": "-", "GA": "--", "BO": ":", "Random": "-.", "cTuRBOknown": ":"}
LINE_NAME = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot"}
COLS = [("real_sfs_q99", "sfs", "SFS · steel fatigue (5-D)", "Fatigue objective (reduction)"),
        ("real_nomc_q99", "nomc", "NOMC · methane conversion (4-D)", "C₂ yield gain"),
        ("real_ccpp_q99", "ccpp", "CCPP · power plant (4-D)", "Power gain (MW)")]
GEOM = {"top": {"fig": (3.85, 2.70), "ax": (0.55, 0.42, 3.20, 2.20)}}
B, SEED = 5000, 0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "realdata"))
    ap.add_argument("--params", default=os.path.join(CODE_DIR, "realdata_params.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    args = ap.parse_args()
    S = json.load(open(os.path.join(args.results, "summary.json")))
    R = pd.read_csv(os.path.join(args.results, "summary_metrics_per_run.csv"))
    PR = json.load(open(args.params))
    cyc = np.arange(1, 21) * 5
    rng = np.random.RandomState(SEED)
    out = {"methods": [{"key": m, "label": FS.LABELS[m], "color": CLASS_COLOR[m], "line": LINE_NAME[LINE[m]]} for m in METHODS],
           "geom": GEOM, "datasets": []}
    for p, key, title, unit in COLS:
        P = S["per_problem"][p]
        methods = [m for m in METHODS if m in P["methods"]]
        # ---- top: cumulative violations
        fig, ax = axes_for("top")
        V_end = {}
        for m in methods:
            r = P["methods"][m]
            mu = np.array(r["V_curve_mean"]); sd = np.array(r["V_curve_sd"] or [0] * 20); se = sd / np.sqrt(r["n_seeds"])
            col = CLASS_COLOR[m] if m != "cTuRBOknown" else "black"
            ax.plot(np.r_[0, cyc], np.r_[0, mu], color=col, ls=LINE[m], lw={"DALIv2LE": 2.4, "cTuRBOknown": 1.2}.get(m, 1.7), zorder=3 if m == "DALIv2LE" else 2)
            if m != "cTuRBOknown":
                ax.fill_between(cyc, mu - 1.96 * se, mu + 1.96 * se, color=CLASS_COLOR[m], alpha=0.10, lw=0)
            V_end[m] = float(mu[-1])
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 20, 40, 60, 80, 100])
        fig.savefig(os.path.join(args.figdir, f"Fig7_v2_panel_{key}.png"), dpi=400); plt.close(fig)
        # ---- bottom: final improvement over the warm-start best (per run), mean + bootstrap CI
        imp = {}
        for m in methods:
            sub = R[(R["problem"] == p) & (R["method"] == m)]
            v = (sub["f0_min"] - sub["best_f"]).values
            idx = rng.randint(0, len(v), size=(B, len(v)))
            boot = v[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            imp[m] = {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(v)),
                      "check_summary_mean": float(P["methods"][m].get("improvement_over_warm_mean", np.nan))}
        f0_ref = float(R.loc[(R["problem"] == p) & (R["method"] == "DALIv2LE"), "f0_min"].mean())
        gap = f0_ref - PR[p]["f_star"]
        out["datasets"].append({"key": key, "problem": p, "title": title, "unit": unit, "V_end": V_end, "improvement": imp,
                                "fstar_gap": float(gap), "box_feasible_frac": PR[p]["box_feasible_frac"], "n_seeds": int(P["methods"]["DALIv2LE"]["n_seeds"]),
                                "xlim": [0, 100], "ylim": [0, 100]})
        print(key, "V_end", {m: round(v, 1) for m, v in V_end.items()}, "| imp", {m: round(d["mean"], 2) for m, d in imp.items()}, "| f*gap", round(gap, 2))
    path = os.path.join(args.figdir, "Fig7_v2_data.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
