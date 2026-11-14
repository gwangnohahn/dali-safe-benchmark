"""SI items added on 2026-09-16 (exploratory blocks of transfers #6 and #7).

Fig S10  kNN gate (exploratory): DALI+LE -> DALI+LE + gate and LE-only -> LE-only
         + gate on the 7 gate configurations; V (symlog) and feasible regret;
         SCBO as reference marker.  Data: gate/summary.json (gate methods) and
         tier1 / tier2 (ungated methods, same configurations).
Table S8 navigation block (exploratory): 6 problems x 6 methods -- V, regret,
         best_f, seeds that reached the global basin, and DALI+LE vs LE-only
         Holm p / Cliff's delta.  Data: nav/summary*.csv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")
GATE_PROBS = ["g2_f050_bnd", "g2_f020_bnd", "g2_f005_bnd", "g1_f020_bnd", "g3_f020_bnd", "g4_f020_bnd", "g2_f020_int"]
LABEL = {"g2_f050_bnd": "G2 boundary 50 %", "g2_f020_bnd": "G2 boundary 20 %", "g2_f005_bnd": "G2 boundary 5 %",
         "g1_f020_bnd": "G1 boundary 20 %", "g3_f020_bnd": "G3 boundary 20 %", "g4_f020_bnd": "G4 boundary 20 %",
         "g2_f020_int": "G2 interior 20 %"}


def load(block):
    return json.load(open(os.path.join(RES, block, "summary.json")))["per_problem"]


def fig_s10(figdir, dpi):
    G = load("gate")
    FS.main_rc()
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5), gridspec_kw={"wspace": 0.25, "left": 0.06, "right": 0.985,
                                                                   "top": 0.86, "bottom": 0.30})
    pairs = [("DALIv2LE", "DALIv2LEgate"), ("LEonly", "LEonlygate")]
    for ax, (key, ylab) in zip(axes, (("V_mean", "Violations in 100 experiments"), ("regret_mean", "Feasible regret, r"))):
        for gi, p in enumerate(GATE_PROBS):
            M = G[p]["methods"]
            for k, (m0, m1) in enumerate(pairs):
                x = gi + (k - 0.5) * 0.36
                FS.dumbbell(ax, x, M[m0][key], M[m1][key], m0)
            ax.scatter([gi + 0.0], [M["SCBO"][key]], marker="D", s=90, facecolor=FS.COLORS["SCBO"], edgecolor="black",
                       lw=0.6, zorder=6)
        ax.set_xticks(range(len(GATE_PROBS)))
        ax.set_xticklabels([LABEL[p] for p in GATE_PROBS], rotation=35, ha="right", rotation_mode="anchor", fontsize=16)
        ax.set_xlim(-0.6, len(GATE_PROBS) - 0.4)
        for gi in range(1, len(GATE_PROBS)):
            ax.axvline(gi - 0.5, color="0.85", lw=0.8, zorder=0)
        ax.set_ylabel(ylab)
        FS.clean_axes(ax)
        if key == "V_mean":
            ax.set_yscale("symlog", linthresh=10, linscale=0.9)
            ax.set_ylim(-0.6, 130)
            ax.set_yticks([0, 5, 10, 30, 100])
            ax.set_yticklabels(["0", "5", "10", "30", "100"])
            ax.minorticks_off()
            FS.subtitle(ax, "○ without gate  →  ● with kNN gate (exploratory)")
            FS.panel_label(ax, "a", dx=-0.10, dy=1.04)
        else:
            ax.set_ylim(-0.001, 0.02)
            FS.subtitle(ax, "Same runs, feasible regret")
            FS.panel_label(ax, "b", dx=-0.13, dy=1.04)
    h = [plt.Line2D([], [], marker="o", ls="", ms=11, markerfacecolor=FS.COLORS["DALIv2LE"], markeredgecolor="black", label="DALI+LE"),
         plt.Line2D([], [], marker="o", ls="", ms=11, markerfacecolor=FS.COLORS["LEonly"], markeredgecolor="black", label="LE-only"),
         plt.Line2D([], [], marker="D", ls="", ms=10, markerfacecolor=FS.COLORS["SCBO"], markeredgecolor="black", label="SCBO (reference)"),
         plt.Line2D([], [], marker="o", ls="", ms=12, markerfacecolor="white", markeredgecolor="0.3", markeredgewidth=2.2, label="without gate"),
         plt.Line2D([], [], marker="o", ls="", ms=9, markerfacecolor="0.3", markeredgecolor="black", label="with gate")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=17)
    FS.savefig(fig, figdir, "FigS10_gate", dpi=dpi)


def table_s8(figdir):
    N = load("nav")
    st = pd.read_csv(os.path.join(RES, "nav", "summary_table_S1_stats.csv"))
    per = pd.read_csv(os.path.join(RES, "nav", "summary_metrics_per_run.csv"))
    # "reached the global basin": best_f below a problem-specific threshold half-way
    # between the local and global minimum (navigation_params.json)
    P = json.load(open(os.path.join(CODE_DIR, "navigation_params.json")))["configs"]
    rows = []
    order = ["ncamel2", "ncamel2_hole", "nhart3", "nhart3_hole", "nhart6", "nhart6_hole"]
    methods = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "cTuRBOknown"]
    for p in order:
        base = p.replace("_hole", "")
        thr = 0.5 * (P[base]["f_star"] + P[base]["f_local"])
        for m in methods:
            r = N[p]["methods"][m]
            sub = per[(per.problem == p) & (per.method == m)]
            reached = int((sub.best_f < thr).sum())
            s = st[(st.problem == p) & (st.method == m) & (st.metric == "best_f")]
            rows.append(dict(problem=p, method=m, V=round(r["V_mean"], 1), best_f=round(r["best_f_mean"], 4),
                             regret=round(r["regret_mean"], 4), improvement_over_warm=round(r.get("improvement_over_warm_mean", np.nan), 4),
                             reached_global_basin=f"{reached}/{len(sub)}",
                             p_holm_vs_DALIplusLE=(round(float(s.p_holm.iloc[0]), 4) if len(s) else ""),
                             cliffs_delta=(float(s.cliffs_delta.iloc[0]) if len(s) else "")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(figdir, "TableS8_navigation.csv"), index=False)
    md = ["# Table S8 | Navigation block (exploratory, not pre-registered)", "",
          "Six constrained multimodal problems (2-, 3-, 6-D); `_hole` = record drawn outside a ball around the optimum. ",
          "V = violations in 100 experiments; regret = best feasible f − f*; reached = seeds whose best f lies below the midpoint ",
          "between the local and the global minimum; p and δ = Holm-adjusted Mann–Whitney and Cliff's δ of best_f vs DALI+LE. ",
          "Caveat: the ball of the `_hole` variant removes ~64 of 200 record points in ncamel2 but only ~23 (nhart3) and ~2 (nhart6), ",
          "so only ncamel2_hole realises a record that has never visited the basin of the optimum.", "",
          "| problem | method | V | best f | regret | improvement over warm | reached | p (Holm) | δ |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.problem} | {FS.LABELS[r.method]} | {r.V} | {r.best_f} | {r.regret} | {r.improvement_over_warm} | {r.reached_global_basin} | {r.p_holm_vs_DALIplusLE} | {r.cliffs_delta} |")
    open(os.path.join(figdir, "TableS8_navigation.md"), "w").write("\n".join(md) + "\n")
    print(df.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    fig_s10(args.figdir, args.dpi)
    table_s8(args.figdir)


if __name__ == "__main__":
    main()
