"""SI tables S3-S6 (paper 2), from the cached parameters and the summary.json files.

Table S3  constraint-suite parameters and verification (suite_verification.csv)
Table S4  real-data oracle parameters (realdata_params.json)
Table S5  pre-registered hypotheses: verdict table (tier1 / h4 / realdata grids)
Table S6  auxiliary OLS coefficients V ~ method + geometry + ratio + location (no inference)

Writes CSVs to figures/ and one markdown file figures/SI_tables_S3-S6.md.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")
UNAWARE = ["TuRBO", "GA", "BO", "Random"]


def load(block):
    return json.load(open(os.path.join(RES, block, "summary.json")))


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_s3(figdir):
    V = pd.read_csv(os.path.join(RES, "suite_verification.csv"))
    T = V[["name", "tier", "geometry", "ratio_target", "location", "theta", "shift", "ratio_calib", "ratio_verify",
           "c0_margin", "g_at_c0", "f_star", "g3_gap"]].copy()
    T["theta_meaning"] = T["geometry"].map({"g1": "ball radius", "g2": "sine threshold tau", "g3": "ball radius (3-D subspace)",
                                            "g4": "band half-width eps"})
    T.to_csv(os.path.join(figdir, "TableS3_suite_parameters.csv"), index=False)
    return T


def table_s4(figdir):
    P = json.load(open(os.path.join(CODE_DIR, "realdata_params.json")))
    rows = []
    for k, v in P.items():
        if not isinstance(v, dict) or "tau" not in v:
            continue
        ds, q = k.replace("real_", "").split("_")
        rows.append(dict(config=k, dataset=ds.upper(), threshold=q, n_ref=v["n_ref"], tau=v["tau"],
                         box_feasible_frac=v["box_feasible_frac"], f_star=v["f_star"], mc_n=v["mc_n"]))
    T = pd.DataFrame(rows).sort_values(["dataset", "threshold"])
    T.to_csv(os.path.join(figdir, "TableS4_realdata_oracle.csv"), index=False)
    return T


def table_s5(figdir):
    S1, C, W, R, A, U = load("tier1"), load("h4_contam"), load("h4_warm"), load("realdata"), load("h3_ablation"), load("h3_uniform")
    G = S1["grid"]
    rows = []

    def add(h, item, value, verdict, note=""):
        rows.append(dict(hypothesis=h, item=item, value=value, verdict=verdict, note=note))

    # H1
    add("H1", "configurations where DALI+LE has lower V than all four constraint-unaware baselines (Holm p<0.05)",
        f"{G['H1_wins_all4']}/{G['n_tier1']}", "not rejected (pre-registered rejection: < 12 wins); exactly at the threshold",
        "non-wins: " + ", ".join(p for p, w in G["H1_wins_by_problem"].items() if not w))
    for m, d in G["H1_sign_test"].items():
        add("H1", f"sign test, V(DALI+LE) < V({m}) over 16 configurations", f"{d['n_ref_lower']}/16, p = {d['p_two_sided']:.2g}", "supported")
    # H1b
    for m, d in G["H1b_pooled_logrank"].items():
        add("H1b", f"pooled log-rank on time to first violation, DALI+LE vs {m} (320 runs each)",
            f"chi2 = {d['chi2']:.1f}, p = {d['p']:.2g}, KM median {d['km_median']}",
            "no difference (p > 0.05)" if d["p"] > 0.05 else ("DALI+LE survives longer" if d["E1"] > d["O1"] else "DALI+LE survives shorter"))
    add("H1b", "pre-registered rejection: pooled log-rank vs SCBO non-significant",
        f"p = {G['H1b_pooled_logrank']['SCBO']['p']:.2g}", "not rejected")
    # H2
    for m in ("DALIv2LE", "SCBO", "LEonly", "DALIv2"):
        d = G["H2_cost_per_avoided_violation"].get(m)
        if d:
            vals = [v for v in d["values"] if v is not None]
            add("H2", f"regret cost per avoided violation vs best unaware baseline, {m} (median over configs)",
                f"{d['median']:.2g} (n = {len(vals)})", "cost ~ 0" if abs(d["median"]) < 1e-3 else "cost > 0")
    n_imp = sum(int(S1["per_problem"][p]["reference_vs_warm_start"]["p_one_sided"] < 0.05) for p in S1["config"]["problems"])
    add("H2", "pre-registered rejection: DALI+LE final feasible best not significantly better than the warm-start best (one-sided sign test)",
        f"significant improvement in {n_imp}/16 configurations", "not rejected" if n_imp == 16 else "check")
    # H3 (pre-registered: diffusion contributes beyond LE) -- LE-only vs DALI+LE, V and best_f
    rep = [f"{g}_f020_int" for g in ("g1", "g2", "g3", "g4")]
    n_ns = sum(int(S1["per_problem"][p]["comparisons"]["LEonly"]["V"]["p_holm"] >= 0.05) for p in rep)
    add("H3", "pre-registered rejection: LE-only V ~ DALI+LE V (non-significant) in >= 3 of the 4 representative configs (ratio 0.20, interior)",
        f"non-significant in {n_ns}/4 (both V = 0.0 in all four)", "REJECTED as stated; the ablation below supports the mechanism claim, not the diffusion-vs-LE claim")
    n_le_worse = n_le_better = 0
    for p in S1["config"]["problems"]:
        c = S1["per_problem"][p]["comparisons"]["LEonly"]["V"]
        n_le_worse += int(c["ref_sig_lower"])
        n_le_better += int(c["ref_sig_higher"])
    add("H3", "LE-only vs DALI+LE, V (Holm p<0.05): configs where LE-only is worse / better",
        f"{n_le_worse} / {n_le_better} of 16", "not supported (LE-only is not worse; better on the G2 boundary)")
    n_f_worse = sum(int(S1["per_problem"][p]["comparisons"]["LEonly"]["best_f"]["ref_sig_lower"]) for p in S1["config"]["problems"])
    n_f_better = sum(int(S1["per_problem"][p]["comparisons"]["LEonly"]["best_f"]["ref_sig_higher"]) for p in S1["config"]["problems"])
    add("H3", "LE-only vs DALI+LE, best_f (Holm p<0.05): configs where LE-only is worse / better",
        f"{n_f_worse} / {n_f_better} of 16", "no navigation advantage of the generator on the suite")
    abl = [p for p in A["per_problem"] if "VerifRandom" in A["per_problem"][p]["methods"]]
    vr = np.mean([A["per_problem"][p]["methods"]["VerifRandom"]["V_mean"] for p in abl])
    nc_int = np.mean([A["per_problem"][p]["methods"]["DALInoclip"]["V_mean"] for p in abl if p.endswith("_int")])
    add("H3 (ablation)", "verifier + uniform candidates (generator removed), mean V over 5 configs", f"{vr:.1f}",
        "generator is what keeps proposals feasible")
    add("H3 (ablation)", "DALI without clip, mean V over the 4 interior configs", f"{nc_int:.1f}", "clip is not the cause")
    # H4
    for g in ("g1", "g2", "g3", "g4"):
        v0 = S1["per_problem"][f"{g}_f020_int"]["methods"]["DALIv2LE"]["V_mean"]
        v10 = C["per_problem"][f"{g}_f020_int_c10"]["methods"]["DALIv2LE"]["V_mean"]
        v30 = C["per_problem"][f"{g}_f020_int_c30"]["methods"]["DALIv2LE"]["V_mean"]
        add("H4 contamination", f"DALI+LE V at 0 / 10 / 30 % infeasible warm-start points, {g} 0.20 interior",
            f"{v0:.1f} / {v10:.1f} / {v30:.1f}", "no degradation")
    for g in ("g2", "g3"):
        v0 = S1["per_problem"][f"{g}_f020_int"]["methods"]["DALIv2LE"]
        v50 = W["per_problem"][f"{g}_f020_int_w50"]["methods"]["DALIv2LE"]
        add("H4 warm-start size", f"DALI+LE V / regret at 200 vs 50 warm-start points, {g} 0.20 interior",
            f"{v0['V_mean']:.1f} / {v0['regret_mean']:.4f}  vs  {v50['V_mean']:.1f} / {v50['regret_mean']:.4f}", "V unchanged, regret slightly higher")
    for g in ("g1", "g2", "g3", "g4"):
        vi = S1["per_problem"][f"{g}_f020_int"]["methods"]
        vb = S1["per_problem"][f"{g}_f020_bnd"]["methods"]
        add("H4 boundary optimum", f"V interior -> boundary (ratio 0.20), {g}: DALI+LE / DALI / LE-only / SCBO",
            " / ".join(f"{vi[m]['V_mean']:.1f}->{vb[m]['V_mean']:.1f}" for m in ("DALIv2LE", "DALIv2", "LEonly", "SCBO")),
            "G2: DALI fails, LE halves it, LE-only unaffected" if g == "g2" else "moderate increase")
    # H5
    for p, d in R["per_problem"].items():
        ref = d["methods"]["DALIv2LE"]["V_mean"]
        lower = [m for m in UNAWARE if d["methods"][m]["V_mean"] > ref]
        sig = [m for m in UNAWARE if d["comparisons"][m]["V"]["ref_sig_lower"]]
        add("H5", f"{p}: DALI+LE V = {ref:.1f}; unaware baselines with higher mean V / Holm-significant (10 seeds)",
            f"{len(lower)}/4 / {len(sig)}/4 ({', '.join(sig)})",
            "ordering holds; significant vs all four" if len(sig) == 4 else "ordering holds; not significant vs all four")
        add("H5", f"{p}: DALI+LE vs SCBO V", f"{ref:.1f} vs {d['methods']['SCBO']['V_mean']:.1f} (Holm p = {d['comparisons']['SCBO']['V']['p_holm']:.2g})", "")
    n_strict = sum(int(all(d["comparisons"][m]["V"]["ref_sig_lower"] for m in UNAWARE)) for d in R["per_problem"].values())
    add("H5", "pre-registered rejection: H1 (significant vs all four unaware baselines) fails in >= 2 of 3 datasets",
        f"strict H1 holds in {n_strict}/3 datasets (mean ordering holds in 3/3)",
        "REJECTED under the strict reading (10 seeds; see per-dataset rows)" if n_strict < 2 else "not rejected")
    # exploratory
    for g in ("g1", "g2", "g3", "g4"):
        u = U["per_problem"][f"{g}_f020_int_uniform"]["methods"]
        add("exploratory: uniform warm start", f"{g} 0.20 interior, V: DALI+LE / LE-only / SCBO / TuRBO",
            " / ".join(f"{u[m]['V_mean']:.1f}" for m in ("DALIv2LE", "LEonly", "SCBO", "TuRBO")), "not pre-registered")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(figdir, "TableS5_hypothesis_verdicts.csv"), index=False)
    return T


def table_s6(figdir):
    G = load("tier1")["grid"]["aux_ols_V"]
    T = pd.DataFrame([dict(term=k, coefficient=v) for k, v in G["coef"].items()])
    T.to_csv(os.path.join(figdir, "TableS6_aux_ols.csv"), index=False)
    return T, G["note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    args = ap.parse_args()
    s3 = table_s3(args.figdir)
    s4 = table_s4(args.figdir)
    s5 = table_s5(args.figdir)
    s6, note6 = table_s6(args.figdir)
    md = ["# SI tables S3-S6 (generated by make_si_tables.py)\n",
          "## Table S3. Constraint-suite parameters and verification\n",
          "theta = calibrated size parameter (see theta_meaning); shift = diagonal shift of the G2 boundary configurations; "
          "ratio_calib = Monte-Carlo feasible ratio at calibration (200k points, seed 123); ratio_verify = independent check "
          "(seed 0); c0_margin = distance of c0 to the constraint boundary in g units; g3_gap = gap between the two balls.\n",
          md_table(s3), "",
          "## Table S4. Real-data feasibility oracle\n",
          "tau = threshold on the mean kNN5 distance (z-scored units) at the given leave-one-out quantile of the reference set "
          "(n_ref unique rows); box_feasible_frac = fraction of the bounding box that the oracle accepts (200k uniform points); "
          "f_star = empirical best of the minimised objective over data rows and feasible box samples.\n",
          md_table(s4), "",
          "## Table S5. Pre-registered hypotheses: verdicts\n", md_table(s5), "",
          "## Table S6. Auxiliary OLS, V ~ method + geometry + ratio + location (Tier 1, per-run)\n",
          note6 + ". Reference levels: method = BO, geometry = g1, location = bnd; ratio is numeric (0.05-0.50).\n",
          md_table(s6)]
    open(os.path.join(args.figdir, "SI_tables_S3-S6.md"), "w").write("\n".join(md))
    print("rows: S3", len(s3), "S4", len(s4), "S5", len(s5), "S6", len(s6))


if __name__ == "__main__":
    main()
