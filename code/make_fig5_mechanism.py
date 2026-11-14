"""Fig 5 -- mechanism: compliance comes from in-distribution proposal generation (H3).

(a) feasible fraction of the raw candidate pool per configuration (all 16
    Tier-1 configurations): DALI candidates before verifier/clip (1,600 per
    cycle), SCBO trust-region candidates before its constraint filter, and the
    uniform-in-box reference (= the feasible ratio of the configuration).
(b) constraint margin g(x) of the proposed points (ratio 0.20, interior, one
    panel per geometry): box = quartiles, whiskers = 5-95 %, line = median,
    pooled over 20 seeds x 100 proposals; g < 0 is a violation.
(c) mean kNN(5) distance of the proposals to the warm-start data (z-scored
    units), one value per run, 12 interior configurations x 20 seeds.
(d) H3 ablation: V against mean kNN distance for every (configuration,
    method) pair in the five ablation configurations; the two controls
    (verifier + uniform candidates, DALI without clip) are highlighted.

Data: tier1/summary.json + summary_metrics_per_run.csv, h3_ablation/summary.json.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
ABL = ["DALIv2LE", "DALIv2", "LEonly", "VerifRandom", "DALInoclip", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
GEOM_LABEL = {"g1": "G1 convex", "g2": "G2 non-convex", "g3": "G3 disconnected", "g4": "G4 thin band"}


def config_order(names):
    return sorted(names, key=lambda n: (n.split("_")[0], 0 if n.split("_")[2] == "int" else 1, -int(n.split("_")[1][1:])))


def row_label(n):
    g, r, loc = n.split("_")[:3]
    return f"{g.upper()} · {int(r[1:]) / 100:.2f} · {'int' if loc == 'int' else 'bnd'}"


def ratio_of(n):
    return int(n.split("_")[1][1:]) / 100


def quantile_box(ax, i, q, color, width=0.62, hollow=False):
    q05, q25, q50, q75, q95 = q
    ax.plot([i, i], [q05, q25], color=color, lw=1.4, zorder=3)
    ax.plot([i, i], [q75, q95], color=color, lw=1.4, zorder=3)
    ax.add_patch(Rectangle((i - width / 2, q25), width, q75 - q25,
                           facecolor="white" if hollow else color, edgecolor=color, lw=1.8 if hollow else 1.0,
                           alpha=1.0 if hollow else 0.85, zorder=4))
    ax.plot([i - width / 2, i + width / 2], [q50, q50], color="black" if not hollow else color, lw=2.0, zorder=5)


TREND_LINE = False


def isotonic_fit(x, y):
    """Non-decreasing least-squares fit (pool-adjacent-violators); returns the
    sorted x and the fitted values (a step function drawn through the pooled blocks)."""
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks = [[ys[i], 1, i, i] for i in range(len(ys))]      # mean, weight, first, last
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m, w = blocks[i], blocks[i + 1]
            merged = [(m[0] * m[1] + w[0] * w[1]) / (m[1] + w[1]), m[1] + w[1], m[2], w[3]]
            blocks[i:i + 2] = [merged]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.empty_like(ys)
    for mean, _, a, b in blocks:
        fit[a:b + 1] = mean
    return xs, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", default=os.path.join(RES, "tier1"))
    ap.add_argument("--ablation", default=os.path.join(RES, "h3_ablation", "summary.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stem", default="Fig5_mechanism")
    args = ap.parse_args()
    S = json.load(open(os.path.join(args.tier1, "summary.json")))
    R = pd.read_csv(os.path.join(args.tier1, "summary_metrics_per_run.csv"))
    A = json.load(open(args.ablation))
    rows = config_order(list(S["per_problem"]))
    methods = [m for m in METHODS if all(m in S["per_problem"][p]["methods"] for p in rows)]

    FS.main_rc()
    LABEL_FS = 15          # value labels over bars / in-panel notes (never < 15)
    TICK_FS = 16           # rotated category tick labels
    fig = plt.figure(figsize=(22.0, 19.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.85, 1.05], hspace=0.62, wspace=0.34,
                          left=0.075, right=0.985, top=0.955, bottom=0.135)

    # ---------------------------------------------------------------- (a)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(rows))
    w = 0.26
    series = [("DALIv2", "DALI"), ("DALIv2LE", "DALI+LE"), ("SCBO", "SCBO")]
    for k, (m, lab) in enumerate(series):
        vals = np.array([S["per_problem"][p]["methods"][m].get("cand_feasible_mean") or np.nan for p in rows])
        hollow = m == "DALIv2"
        ax.bar(x + (k - 1) * w, vals, width=w, facecolor="white" if hollow else FS.COLORS[m],
               edgecolor=FS.COLORS[m] if hollow else "black", lw=1.8 if hollow else 0.6, label=lab, zorder=3)
        # no per-bar value labels (all values in Table S1; G2 x boundary = 0.06 / 0.07 is quoted in the text)
    ratios = np.array([ratio_of(p) for p in rows])
    ax.scatter(x, ratios, marker="_", s=900, color="black", lw=2.4, zorder=6,
               label="Uniform in box (= feasible fraction)")
    ax.set_xticks(x)
    xl = []
    for p in rows:
        g, rt, loc = p.split("_")[:3]
        xl.append(f"{int(rt[1:])}" + ("\nboundary" if loc == "bnd" else ""))
    ax.set_xticklabels(xl, fontsize=TICK_FS)
    for i, p in enumerate(rows):
        if p.split("_")[2] == "bnd":
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlabel("Feasible fraction of the box (%)", labelpad=6)
    # geometry group headers (same device as Fig 2b)
    GEOM_NAME = {"g1": "G1 · convex", "g2": "G2 · non-convex", "g3": "G3 · disconnected", "g4": "G4 · thin band"}
    starts = [i for i in range(len(rows)) if i == 0 or rows[i].split("_")[0] != rows[i - 1].split("_")[0]] + [len(rows)]
    for a, b in zip(starts[:-1], starts[1:]):
        ax.text((a + b - 1) / 2, 1.12, GEOM_NAME[rows[a].split("_")[0]], ha="center", va="bottom", fontsize=17,
                fontweight="bold")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_ylabel("Feasible fraction\nof raw candidates")
    for i in range(1, len(rows)):
        if rows[i].split("_")[0] != rows[i - 1].split("_")[0]:
            ax.axvline(i - 0.5, color="black", lw=1.0, ymax=0.92)
    FS.clean_axes(ax)
    FS.subtitle(ax, "all16")
    ax.legend(loc="upper right", ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(1.0, 1.14),
              columnspacing=2.0, handlelength=1.8)
    FS.panel_label(ax, "a", dx=-0.06, dy=1.01)

    # ---------------------------------------------------------------- (b)
    for k, g in enumerate(["g1", "g2", "g3", "g4"]):
        ax = fig.add_subplot(gs[1, k])
        p = f"{g}_f020_int"
        for i, m in enumerate(methods):
            r = S["per_problem"][p]["methods"][m]
            q = (r["g_q05"], r["g_q25"], r["g_median"], r["g_q75"], r["g_q95"])
            quantile_box(ax, i, q, FS.COLORS[m], hollow=(m == "DALIv2"))
        ax.axhline(0, color=FS.INFEAS_EDGE, lw=1.4, ls="--", zorder=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([FS.LABELS[m] for m in methods], rotation=55, ha="right", fontsize=TICK_FS,
                           rotation_mode="anchor")
        geom = GEOM_LABEL[g].replace(" ", " · ", 1)
        FS.subtitle(ax, (FS.SUBSET["int20"] + "\n" + geom) if k == 0 else geom)
        if k == 0:
            ax.set_ylabel("Margin $g(x)$ of proposals")
            FS.panel_label(ax, "b", dx=-0.42, dy=1.01)
        FS.clean_axes(ax)
        ax.set_xlim(-0.7, len(methods) - 0.3)
        lo = min(S["per_problem"][p]["methods"][m]["g_q05"] for m in methods)
        hi = max(S["per_problem"][p]["methods"][m]["g_q95"] for m in methods)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))

    # ---------------------------------------------------------------- (c)
    ax = fig.add_subplot(gs[2, :2])
    R["loc"] = R["problem"].str.split("_").str[2]
    Ri = R[R["loc"] == "int"]
    data = [Ri.loc[Ri["method"] == m, "dknn_mean"].values for m in methods]
    for i, (m, v) in enumerate(zip(methods, data)):
        q = np.percentile(v, [5, 25, 50, 75, 95])
        quantile_box(ax, i, q, FS.COLORS[m], hollow=(m == "DALIv2"))
        ax.scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-0.28, 0.28, len(v)), v, s=6, color="black",
                   alpha=0.25, zorder=6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([FS.LABELS[m] for m in methods], rotation=40, ha="right", fontsize=TICK_FS,
                       rotation_mode="anchor")
    ax.set_xlim(-0.7, len(methods) - 0.3)
    ax.set_ylabel("Distance to record, $d_{\\mathrm{kNN}}$")
    FS.subtitle(ax, "int12")
    FS.clean_axes(ax)
    FS.panel_label(ax, "c", dx=-0.13, dy=1.01)

    # ---------------------------------------------------------------- (d)
    ax = fig.add_subplot(gs[2, 2:])
    abl_probs = [p for p in A["per_problem"] if "VerifRandom" in A["per_problem"][p]["methods"]]
    xs, ys = [], []
    for m in ABL:
        px, py = [], []
        for p in abl_probs:
            r = A["per_problem"][p]["methods"].get(m)
            if r is None:
                continue
            px.append(r["dknn_mean"])
            py.append(r["V_mean"])
        xs += px
        ys += py
        ctrl = m in ("VerifRandom", "DALInoclip")
        hollow = m in ("DALIv2", "cTuRBOknown")
        ax.scatter(px, py, marker=FS.MARKERS[m], s=210 if ctrl else (170 if m != "cTuRBOknown" else 420),
                   facecolors="white" if hollow else FS.COLORS[m],
                   edgecolors=FS.COLORS[m] if (hollow or m == "DALInoclip") else "black",
                   linewidths=2.6 if (m == "DALInoclip" or hollow) else 0.8,
                   alpha=1.0 if ctrl or m.startswith("DALI") or m == "LEonly" else 0.75,
                   zorder=6 if ctrl else 4, label=FS.LABELS[m])
    rho, pv = spearmanr(xs, ys)
    # monotone (isotonic) trend: pool-adjacent-violators on log d; the rank-based
    # statistic (Spearman) is what the fit visualises -- no linear model is implied
    if TREND_LINE:   # off by default (decision 09-16): the two clusters and rho say it; a line invites a model
        xf, yf = isotonic_fit(np.log(np.array(xs)), np.array(ys))
        ax.plot(np.exp(xf), yf, color="0.35", lw=2.2, ls="-", zorder=3, label="monotone trend (isotonic fit)")
    ax.set_xscale("log")
    ax.set_xlabel("Distance to record, $d_{\\mathrm{kNN}}$")
    ax.set_ylabel("Violations in 100 experiments")
    FS.subtitle(ax, "H3 ablation (5 configurations)")
    ax.text(0.97, 0.06, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=17)
    FS.clean_axes(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=17, frameon=False,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.6, handletextpad=0.5)
    ax.set_ylim(-3, 100)
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 3, 5])
    ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "3", "5"])
    ax.minorticks_off()
    # annotate the boundary configuration (the only one where DALI / DALI+LE fail)
    for m, off in (("DALIv2", (16, -5)), ("DALIv2LE", (16, -5))):
        r = A["per_problem"]["g2_f020_bnd"]["methods"].get(m)
        if r:
            ax.annotate("G2, boundary 20 %", (r["dknn_mean"], r["V_mean"]), textcoords="offset points",
                        xytext=off, fontsize=LABEL_FS, color="0.25", va="center")
    # annotate the verifier + uniform control with its raw-candidate feasible fraction
    # (= the feasible fraction of the box, 0.20: the verifier alone does not make candidates safe)
    vr = [A["per_problem"][p]["methods"]["VerifRandom"] for p in abl_probs]
    cf = np.mean([r["cand_feasible_mean"] for r in vr])
    ax.annotate(f"verifier + uniform:\n{cf * 100:.0f} % of raw candidates feasible\n(= feasible fraction of the box)",
                (float(np.mean([r["dknn_mean"] for r in vr])), float(np.mean([r["V_mean"] for r in vr]))),
                textcoords="data", xytext=(0.78, 70), fontsize=LABEL_FS, color="0.25", ha="center", va="center",
                linespacing=1.2, arrowprops=dict(arrowstyle="-", color="0.5", lw=1.0, shrinkB=12))
    FS.panel_label(ax, "d", dx=-0.16, dy=1.01)

    FS.savefig(fig, args.figdir, args.stem, dpi=args.dpi)
    # numbers used in the caption / text
    out = {"cand_feas_DALI_interior_min": float(np.nanmin([S["per_problem"][p]["methods"]["DALIv2"]["cand_feasible_mean"]
                                                           for p in rows if "_int" in p])),
           "spearman_rho_d": float(rho), "spearman_p_d": float(pv), "n_points_d": len(xs)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
