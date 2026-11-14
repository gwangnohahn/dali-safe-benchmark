"""Analysis of constraint-suite runs -> summary.json (+ aggregate CSVs) (paper 2, W1-5).

Implements the PRE-REGISTERED metrics and tests of plan v3_final section 3
(nothing else is computed by default; anything added later must be labelled
"exploratory").

Per run (method, problem, seed), from <method>_<problem>_seed<seed>.csv and
its _proposals.csv:
  V          cumulative number of infeasible proposals over the 100 evaluations
  T1         evaluation index of the first violation (1..100); no violation ->
             right-censored at n_evals (event = 0)
  best_f     final best objective over FEASIBLE evaluations (paper-1 column)
  r          feasible simple regret = best_f - f*  (f* from the suite params;
             for problems without f* the empirical minimum over all runs of the
             problem is used and flagged)
  margins    summary of the proposal margins g (column `g`, or recomputed from
             the problem when it exposes g)
  d_kNN      mean standardised kNN5 distance of proposals to the warm start
             (warm start regenerated with prob.warm_start(seed))
  cand_feas  mean feasible fraction of the raw candidate pool (cand_feasible_frac)

Per (problem, method): mean, sd, bootstrap 95 % CI (B=10000, seed 0) of V,
best_f, r; per-seed values; Kaplan-Meier curve of T1 (in-house estimator,
cross-checked against lifelines when importable); cumulative-violation and
best_f curves per cycle (mean, sd).

Tests (reference method = DALIv2LE unless --reference):
  * per problem, reference vs every other method: two-sided Mann-Whitney U on
    V and on best_f, Holm correction within problem x metric, Cliff's delta
    (delta < 0 <=> reference smaller), bootstrap CI of the mean difference.
  * T1: log-rank reference vs each method per problem, and pooled over the
    Tier-1 problems present (all seeds stacked).
  * grid level (H1): number of Tier-1 problems in which the reference beats
    ALL FOUR constraint-unaware baselines (TuRBO, GA, BO, Random) on V
    (Holm p < 0.05 and delta < 0); per-baseline sign test over problems on the
    sign of the mean-V difference.
  * H2: safety cost (r_method - r_bestbaseline) / (V_bestbaseline - V_method)
    per problem for DALIv2LE, DALIv2, SCBO, best baseline = lowest mean r among
    the unaware baselines; improvement of the reference over the warm-start
    best (one-sided Wilcoxon on best_f_final - f0_min).
  * H3: reference vs LEonly is part of the per-problem comparisons.
  * auxiliary: OLS coefficients of V ~ method + geometry + ratio + location
    (dummy coding, no inference).

Usage:
  python analyze_constraint_suite.py --results ../results/constraint_suite/tier1
  python analyze_constraint_suite.py --results ../results/constraint_suite/tier1,../results/constraint_suite/tier2 --out ../results/constraint_suite/summary_tier12.json
  python analyze_constraint_suite.py --results ref_results/track_C,ref_results/track_C_LE --reference DALIv2LE --out scratch/ref_summary.json
Outputs: summary.json, table_S1_stats.csv (all p / delta / CI), metrics_per_run.csv.
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest, wilcoxon
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UNAWARE = ["TuRBO", "GA", "BO", "Random"]
METHOD_ORDER = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown",
                "VerifRandom", "DALInoclip", "DALIv2LEgate", "LEonlygate"]
B_BOOT, BOOT_SEED, ALPHA = 10_000, 0, 0.05
KNN_K = 5


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def cliffs_delta(a, b):
    """P(a > b) - P(a < b); negative <=> a tends to be smaller."""
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    return float(((a > b).mean() - (a < b).mean()))


def boot_ci(x, stat=np.mean, B=B_BOOT, seed=BOOT_SEED):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(B, len(x)))
    s = stat(x[idx], axis=1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def boot_ci_diff(a, b, B=B_BOOT, seed=BOOT_SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.RandomState(seed)
    ia = rng.randint(0, len(a), size=(B, len(a)))
    ib = rng.randint(0, len(b), size=(B, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def mwu(a, b):
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:  # all identical
        return 1.0


def km_curve(times, events, t_max):
    """Kaplan-Meier survival S(t) for t = 1..t_max (probability of NO violation
    up to and including evaluation t)."""
    times = np.asarray(times, float)
    events = np.asarray(events, bool)
    S, surv = 1.0, []
    for t in range(1, t_max + 1):
        at_risk = np.sum(times >= t)
        d = np.sum((times == t) & events)
        if at_risk > 0:
            S *= 1.0 - d / at_risk
        surv.append(S)
    return surv


def km_median(times, events, t_max):
    s = km_curve(times, events, t_max)
    for t, v in enumerate(s, start=1):
        if v <= 0.5:
            return t
    return None  # not reached


def logrank(times1, events1, times2, events2):
    """Two-group log-rank test (chi-square, 1 df); in-house implementation."""
    from scipy.stats import chi2
    t1, e1 = np.asarray(times1, float), np.asarray(events1, bool)
    t2, e2 = np.asarray(times2, float), np.asarray(events2, bool)
    grid = np.unique(np.concatenate([t1[e1], t2[e2]]))
    O1 = E1 = Var = 0.0
    for t in grid:
        n1, n2 = np.sum(t1 >= t), np.sum(t2 >= t)
        d1, d2 = np.sum((t1 == t) & e1), np.sum((t2 == t) & e2)
        n, d = n1 + n2, d1 + d2
        if n < 2:
            continue
        E1 += d * n1 / n
        Var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
        O1 += d1
    if Var <= 0:
        return {"chi2": 0.0, "p": 1.0, "O1": float(O1), "E1": float(E1)}
    stat = (O1 - E1) ** 2 / Var
    return {"chi2": float(stat), "p": float(chi2.sf(stat, 1)), "O1": float(O1), "E1": float(E1)}


def logrank_lifelines(times1, events1, times2, events2):
    try:
        from lifelines.statistics import logrank_test
    except ImportError:
        return None
    r = logrank_test(times1, times2, event_observed_A=events1, event_observed_B=events2)
    return float(r.p_value)


# ---------------------------------------------------------------------------
# problem registry (f*, warm start, g) -- optional, degrades gracefully
# ---------------------------------------------------------------------------
def build_registry(need):
    reg = {}
    try:
        import problems_synthetic as SP
        reg.update(SP.build_problems())
    except Exception as e:  # pragma: no cover
        warnings.warn(f"problems_synthetic unavailable: {e}")
    try:
        import problems_constraint_suite as CS
        if any(re.match(r"g[1-4]_f\d{3}_(int|bnd)", p) for p in need):
            reg.update(CS.build_suite("all"))
            reg.update(CS.build_suite("h4"))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"problems_constraint_suite unavailable: {e}")
    uni = [p for p in need if p.endswith("_uniform")]
    if uni:
        try:
            import problems_constraint_suite as CS
            reg.update(CS.build_suite([p[:-len("_uniform")] for p in uni], coverage="uniform"))
        except Exception as e:  # pragma: no cover
            warnings.warn(f"uniform-coverage problems unavailable: {e}")
    nav = [p for p in need if p.split("_")[0] in ("ncamel2", "nhart3", "nhart6")]
    if nav:
        try:
            import problems_navigation as NAV
            reg.update(NAV.build_nav_problems())
        except Exception as e:  # pragma: no cover
            warnings.warn(f"problems_navigation unavailable: {e}")
    real = [p for p in need if p.startswith("real_")]
    if real:
        try:
            import feasibility_oracle_real as FR
            qk = sorted({p.split("_")[-1] for p in real})
            reg.update(FR.build_real_problems(q_keys=tuple(qk)))
        except Exception as e:  # pragma: no cover
            warnings.warn(f"feasibility_oracle_real unavailable: {e}")
    return reg


def parse_config(pname):
    m = re.match(r"(g[1-4])_f(\d{3})_(int|bnd)(?:_(.*))?$", pname)
    if not m:
        return {"geometry": None, "ratio": None, "location": None, "variant": None}
    return {"geometry": m.group(1), "ratio": int(m.group(2)) / 100.0,
            "location": m.group(3), "variant": m.group(4)}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
RUN_RE = re.compile(r"^(?P<method>[A-Za-z0-9]+)_(?P<problem>.+)_seed(?P<seed>\d+)\.csv$")


def find_runs(dirs):
    runs = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            base = os.path.basename(f)
            if base.endswith("_proposals.csv") or base.startswith("_"):
                continue
            m = RUN_RE.match(base)
            if not m:
                continue
            key = (m["problem"], m["method"], int(m["seed"]))
            runs[key] = f
    return runs


def per_run_metrics(path, prob, reg_ok, warm_cache):
    df = pd.read_csv(path).sort_values("iteration")
    pfile = path[:-4] + "_proposals.csv"
    props = pd.read_csv(pfile) if os.path.exists(pfile) else None
    method, pname, seed = df["method"].iloc[0], df["problem"].iloc[0], int(df["seed"].iloc[0])
    out = {"problem": pname, "method": method, "seed": seed,
           "n_cycles": int(len(df)), "V": int(df["n_infeasible"].sum()),
           "best_f": float(df["best_f"].iloc[-1]),
           "V_curve": df["n_infeasible"].cumsum().astype(int).tolist(),
           "best_f_curve": df["best_f"].astype(float).tolist(),
           "cycle_time_sec": float(df["cycle_time_sec"].sum())}
    for col in ("cand_feasible_frac", "cand_le_feasible_frac"):
        if col in df.columns and df[col].notna().any():
            out[col.replace("_frac", "_mean")] = float(np.nanmean(df[col]))
            out[col + "_curve"] = [None if pd.isna(v) else float(v) for v in df[col]]
    # T1 from proposals (evaluation resolution); fallback to cycle resolution
    if props is not None and "in_box" in props.columns:
        out["in_box_frac"] = float(props["in_box"].astype(str).str.lower().isin(["true", "1"]).mean())
    if props is not None and "feasible" in props.columns:
        feas = props["feasible"].astype(str).str.lower().isin(["true", "1"]).values
        n_eval = len(feas)
        inf_idx = np.where(~feas)[0]
        out["n_evals"] = int(n_eval)
        out["T1"] = int(inf_idx[0] + 1) if len(inf_idx) else int(n_eval)
        out["T1_event"] = bool(len(inf_idx))
    else:
        n_eval = int(df["n_infeasible"].size * 5)
        cyc = df.index[df["n_infeasible"] > 0]
        out["n_evals"] = n_eval
        out["T1"] = int(df["iteration"].iloc[cyc[0]] * 5) if len(cyc) else n_eval
        out["T1_event"] = bool(len(cyc))
    # margins
    if props is not None:
        X = props[[c for c in props.columns if re.match(r"^x\d+$", c)]].values
        g = None
        if "g" in props.columns:
            g = props["g"].values.astype(float)
        elif prob is not None and hasattr(prob, "g"):
            g = np.asarray(prob.g(X), float)
        if g is not None:
            out["g_mean"], out["g_median"], out["g_min"] = float(g.mean()), float(np.median(g)), float(g.min())
            out["g_frac_neg"] = float((g < 0).mean())
            out["g_q05"], out["g_q25"], out["g_q75"], out["g_q95"] = [float(v) for v in np.percentile(g, [5, 25, 75, 95])]
        # d_kNN to the warm start (standardised by warm-start mean/sd)
        if prob is not None and reg_ok:
            key = (pname, seed)
            if key not in warm_cache:
                try:
                    X0, f0 = prob.warm_start(seed)
                    warm_cache[key] = (X0, f0)
                except Exception as e:
                    warm_cache[key] = None
                    warnings.warn(f"warm_start failed for {key}: {e}")
            if warm_cache[key] is not None:
                X0, f0 = warm_cache[key]
                mu, sd = X0.mean(axis=0), X0.std(axis=0) + 1e-12
                nn = NearestNeighbors(n_neighbors=KNN_K).fit((X0 - mu) / sd)
                d, _ = nn.kneighbors((X - mu) / sd)
                dk = d.mean(axis=1)
                out["dknn_mean"], out["dknn_median"] = float(dk.mean()), float(np.median(dk))
                out["f0_min"] = float(f0.min())
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate(runs_df, reference, tier1_names):
    problems = list(dict.fromkeys(runs_df["problem"]))
    summary = {"per_problem": {}, "grid": {}, "config": {
        "reference": reference, "alpha": ALPHA, "bootstrap_B": B_BOOT, "bootstrap_seed": BOOT_SEED,
        "unaware_baselines": UNAWARE, "knn_k": KNN_K,
        "tests": "MWU two-sided + Holm within problem x metric; Cliff's delta; bootstrap CI; KM + log-rank",
        "note": "pre-registered metrics/tests of plan v3_final section 3"}}
    table_rows = []
    pooled = {}   # method -> (times, events) over Tier-1 problems for pooled log-rank

    for pname in problems:
        dp = runs_df[runs_df["problem"] == pname]
        methods = [m for m in METHOD_ORDER if m in set(dp["method"])] + \
                  [m for m in dict.fromkeys(dp["method"]) if m not in METHOD_ORDER]
        f_star = dp["f_star"].iloc[0]
        f_star_source = dp["f_star_source"].iloc[0]
        t_max = int(dp["n_evals"].max())
        P = {"config": parse_config(pname), "f_star": float(f_star), "f_star_source": f_star_source,
             "n_evals": t_max, "methods": {}, "comparisons": {}, "logrank": {}, "safety_cost": {}}
        per_m = {}
        for m in methods:
            dm = dp[dp["method"] == m].sort_values("seed")
            V, bf, r = dm["V"].values.astype(float), dm["best_f"].values, dm["regret"].values
            times, events = dm["T1"].values, dm["T1_event"].values.astype(bool)
            per_m[m] = dm
            rec = {"n_seeds": int(len(dm)), "seeds": dm["seed"].astype(int).tolist(),
                   "V_mean": float(V.mean()), "V_sd": float(V.std(ddof=1)) if len(V) > 1 else 0.0,
                   "V_ci95": boot_ci(V), "V_per_seed": V.astype(int).tolist(),
                   "best_f_mean": float(bf.mean()), "best_f_sd": float(bf.std(ddof=1)) if len(bf) > 1 else 0.0,
                   "best_f_ci95": boot_ci(bf), "best_f_per_seed": [float(v) for v in bf],
                   "regret_mean": float(r.mean()), "regret_ci95": boot_ci(r),
                   "regret_per_seed": [float(v) for v in r],
                   "T1_per_seed": times.astype(int).tolist(), "T1_event_per_seed": events.tolist(),
                   "T1_censored_frac": float(1 - events.mean()),
                   "T1_km_median": km_median(times, events, t_max),
                   "km_survival": km_curve(times, events, t_max),
                   "V_curve_mean": np.vstack(dm["V_curve"].values).mean(axis=0).tolist(),
                   "V_curve_sd": np.vstack(dm["V_curve"].values).std(axis=0, ddof=1).tolist() if len(dm) > 1 else None,
                   "best_f_curve_mean": np.vstack(dm["best_f_curve"].values).mean(axis=0).tolist(),
                   "mean_cycle_time_sec": float(dm["cycle_time_sec"].mean() / dm["n_cycles"].iloc[0])}
            for col in ("cand_feasible_mean", "cand_le_feasible_mean", "g_mean", "g_median", "g_min",
                        "g_frac_neg", "g_q05", "g_q25", "g_q75", "g_q95", "dknn_mean", "dknn_median"):
                if col in dm.columns and dm[col].notna().any():
                    rec[col] = float(np.nanmean(dm[col]))
            if "f0_min" in dm.columns and dm["f0_min"].notna().all():
                # improvement over the warm-start best (sign-safe alternative to regret
                # when f* is only empirical, e.g. real datasets)
                imp = dm["f0_min"].values - bf
                rec["improvement_over_warm_mean"] = float(imp.mean())
                rec["improvement_over_warm_ci95"] = boot_ci(imp)
            if "in_box_frac" in dm.columns and dm["in_box_frac"].notna().any():
                rec["in_box_frac"] = float(np.nanmean(dm["in_box_frac"]))
            if "cand_feasible_frac_curve" in dm.columns and dm["cand_feasible_frac_curve"].notna().any():
                curves = [c for c in dm["cand_feasible_frac_curve"] if isinstance(c, list)]
                if curves:
                    arr = np.array([[np.nan if v is None else v for v in c] for c in curves], float)
                    rec["cand_feasible_curve_mean"] = np.nanmean(arr, axis=0).tolist()
            P["methods"][m] = rec
            if pname in tier1_names:
                pooled.setdefault(m, ([], []))
                pooled[m][0].extend(times.tolist())
                pooled[m][1].extend(events.tolist())

        # reference vs others
        if reference in per_m:
            ref = per_m[reference]
            for metric in ("V", "best_f"):
                comps = []
                for m in methods:
                    if m == reference:
                        continue
                    a, b = ref[metric].values.astype(float), per_m[m][metric].values.astype(float)
                    comps.append({"method": m, "metric": metric, "n_ref": len(a), "n": len(b),
                                  "mean_ref": float(a.mean()), "mean": float(b.mean()),
                                  "p_raw": mwu(a, b), "cliffs_delta": cliffs_delta(a, b),
                                  "diff_ci95": boot_ci_diff(a, b)})
                adj = holm_adjust([c["p_raw"] for c in comps])
                for c, ph in zip(comps, adj):
                    c["p_holm"] = ph
                    c["ref_sig_lower"] = bool(ph < ALPHA and c["cliffs_delta"] < 0)
                    c["ref_sig_higher"] = bool(ph < ALPHA and c["cliffs_delta"] > 0)
                    P["comparisons"].setdefault(c["method"], {})[metric] = {k: v for k, v in c.items() if k not in ("method", "metric")}
                    table_rows.append({"problem": pname, "reference": reference, **c})
            # log-rank on T1
            for m in methods:
                if m == reference:
                    continue
                lr = logrank(ref["T1"], ref["T1_event"].astype(bool), per_m[m]["T1"], per_m[m]["T1_event"].astype(bool))
                ll = logrank_lifelines(ref["T1"], ref["T1_event"].astype(bool), per_m[m]["T1"], per_m[m]["T1_event"].astype(bool))
                if ll is not None:
                    lr["p_lifelines"] = ll
                P["logrank"][m] = lr
            # H2: improvement over the warm-start best (one-sided Wilcoxon)
            if "f0_min" in ref.columns and ref["f0_min"].notna().all():
                d = ref["best_f"].values - ref["f0_min"].values
                try:
                    p_imp = float(wilcoxon(d, alternative="less").pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    p_imp = float("nan")
                P["reference_vs_warm_start"] = {"mean_improvement": float(-d.mean()), "p_one_sided": p_imp,
                                                "n_improved": int((d < 0).sum()), "n": int(len(d))}
        # H2: safety cost relative to the best unaware baseline
        un = [m for m in UNAWARE if m in per_m]
        if un:
            best_b = min(un, key=lambda m: P["methods"][m]["regret_mean"])
            rb, Vb = P["methods"][best_b]["regret_mean"], P["methods"][best_b]["V_mean"]
            P["safety_cost"]["best_baseline"] = best_b
            for m in ("DALIv2LE", "DALIv2", "LEonly", "SCBO"):
                if m in per_m:
                    dV = Vb - P["methods"][m]["V_mean"]
                    dr = P["methods"][m]["regret_mean"] - rb
                    P["safety_cost"][m] = {"delta_regret": float(dr), "violations_avoided": float(dV),
                                           "cost_per_avoided_violation": float(dr / dV) if dV > 0 else None}
        summary["per_problem"][pname] = P

    # ---- grid level (H1, H1b) over Tier-1 problems present
    t1 = [p for p in problems if p in tier1_names]
    G = {"tier1_problems_present": t1, "n_tier1": len(t1)}
    if t1 and reference:
        wins, per_baseline = [], {b: [] for b in UNAWARE}
        for p in t1:
            comps = summary["per_problem"][p]["comparisons"]
            ok = all(b in comps and comps[b]["V"]["ref_sig_lower"] for b in UNAWARE)
            wins.append(ok)
            for b in UNAWARE:
                if b in comps:
                    per_baseline[b].append(comps[b]["V"]["mean"] - comps[b]["V"]["mean_ref"])
        G["H1_wins_all4"] = int(sum(wins))
        G["H1_wins_by_problem"] = dict(zip(t1, [bool(w) for w in wins]))
        G["H1_sign_test"] = {}
        for b, diffs in per_baseline.items():
            diffs = np.asarray(diffs)
            if len(diffs):
                k = int((diffs > 0).sum())
                G["H1_sign_test"][b] = {"n": int(len(diffs)), "n_ref_lower": k,
                                        "p_two_sided": float(binomtest(k, len(diffs), 0.5).pvalue)}
        # pooled log-rank
        G["H1b_pooled_logrank"] = {}
        if reference in pooled:
            for m, (tm, ev) in pooled.items():
                if m == reference:
                    continue
                lr = logrank(pooled[reference][0], pooled[reference][1], tm, ev)
                ll = logrank_lifelines(pooled[reference][0], pooled[reference][1], tm, ev)
                if ll is not None:
                    lr["p_lifelines"] = ll
                lr["km_median_ref"] = km_median(np.array(pooled[reference][0]), np.array(pooled[reference][1], bool), 100)
                lr["km_median"] = km_median(np.array(tm), np.array(ev, bool), 100)
                G["H1b_pooled_logrank"][m] = lr
        # H2 summary
        costs = {m: [summary["per_problem"][p]["safety_cost"].get(m, {}).get("cost_per_avoided_violation")
                     for p in t1] for m in ("DALIv2LE", "SCBO")}
        G["H2_cost_per_avoided_violation"] = {m: {"values": v, "median": float(np.nanmedian([x for x in v if x is not None]))
                                                  if any(x is not None for x in v) else None} for m, v in costs.items()}
        # auxiliary OLS: V ~ method + geometry + ratio + location
        sub = runs_df[runs_df["problem"].isin(t1)].copy()
        cfg = sub["problem"].map(parse_config)
        sub["geometry"] = [c["geometry"] for c in cfg]
        sub["ratio"] = [c["ratio"] for c in cfg]
        sub["location"] = [c["location"] for c in cfg]
        Xd = pd.get_dummies(sub[["method", "geometry", "location"]].astype(str), drop_first=True).astype(float)
        Xd["ratio"] = sub["ratio"].astype(float).values
        Xd.insert(0, "intercept", 1.0)
        beta, *_ = np.linalg.lstsq(Xd.values, sub["V"].values.astype(float), rcond=None)
        G["aux_ols_V"] = {"note": "auxiliary, dummy coding (reference levels dropped), no inference",
                          "coef": dict(zip(Xd.columns, [float(b) for b in beta]))}
    summary["grid"] = G
    return summary, pd.DataFrame(table_rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="comma-separated result directories (union)")
    ap.add_argument("--out", default=None, help="summary.json path (default <first results dir>/summary.json)")
    ap.add_argument("--reference", default="DALIv2LE")
    ap.add_argument("--no-dknn", action="store_true", help="skip warm-start regeneration (d_kNN, f0_min)")
    args = ap.parse_args()

    dirs = [d for d in args.results.split(",") if d]
    runs = find_runs(dirs)
    if not runs:
        raise SystemExit("no run CSVs found")
    problems = sorted({k[0] for k in runs})
    reg = build_registry(problems)
    try:
        import problems_constraint_suite as CS
        tier1 = set(CS.TIER1)
    except Exception:
        tier1 = set()

    warm_cache, rows = {}, []
    for (pname, method, seed), path in sorted(runs.items()):
        prob = reg.get(pname)
        rows.append(per_run_metrics(path, prob, reg_ok=not args.no_dknn, warm_cache=warm_cache))
    df = pd.DataFrame(rows)
    # f*: suite params, else empirical minimum over all runs of the problem (flagged)
    fs, src = [], []
    for pname in df["problem"]:
        prob = reg.get(pname)
        f = getattr(prob, "f_star", None) if prob is not None else None
        if f is None:
            f = float(df.loc[df["problem"] == pname, "best_f"].min())
            src.append("empirical_min_over_runs")
        else:
            src.append(getattr(prob, "f_star_source", "suite_params"))
        fs.append(float(f))
    df["f_star"], df["f_star_source"] = fs, src
    df["regret"] = df["best_f"] - df["f_star"]

    summary, table = aggregate(df, args.reference, tier1)
    summary["config"]["results_dirs"] = dirs
    summary["config"]["n_runs"] = int(len(df))
    summary["config"]["problems"] = problems
    summary["config"]["methods"] = sorted(set(df["method"]))

    out = args.out or os.path.join(dirs[0], "summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    stem = os.path.splitext(out)[0]
    table.to_csv(stem + "_table_S1_stats.csv", index=False)
    scalar_cols = [c for c in df.columns if not c.endswith("_curve")]
    df[scalar_cols].to_csv(stem + "_metrics_per_run.csv", index=False)
    print(f"wrote {out} (+ _table_S1_stats.csv, _metrics_per_run.csv); {len(df)} runs, "
          f"{len(problems)} problems, {len(summary['config']['methods'])} methods")

    # console digest
    for pname, P in summary["per_problem"].items():
        print(f"\n[{pname}] f*={P['f_star']:.4f} ({P['f_star_source']})")
        for m, r in P["methods"].items():
            c = P["comparisons"].get(m, {})
            pv = c.get("V", {}).get("p_holm")
            pb = c.get("best_f", {}).get("p_holm")
            print(f"  {m:12s} n={r['n_seeds']:2d} V={r['V_mean']:6.1f} [{r['V_ci95'][0]:.1f},{r['V_ci95'][1]:.1f}]"
                  f"  best_f={r['best_f_mean']:8.4f}  r={r['regret_mean']:.4f}  T1med={r['T1_km_median']}"
                  + (f"  pV={pv:.3g} pF={pb:.3g}" if pv is not None else "")
                  + (f"  cand_feas={r['cand_feasible_mean']:.2f}" if 'cand_feasible_mean' in r else ""))
    G = summary["grid"]
    if G.get("n_tier1"):
        print(f"\nGrid (Tier 1, n={G['n_tier1']}): H1 wins vs all 4 = {G.get('H1_wins_all4')}; "
              f"pooled log-rank vs SCBO p = {G.get('H1b_pooled_logrank', {}).get('SCBO', {}).get('p')}")


if __name__ == "__main__":
    main()
