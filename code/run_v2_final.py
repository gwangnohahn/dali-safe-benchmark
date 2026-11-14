"""DALI v2 final SO experiment + ablation orchestrator (prompt 7, experiments 1 & 3).

Reuses run_dali_v2.run_dali_v2() with module on/off flags. v1 code/results untouched.

Filename conventions (so each parallel process writes only its own files; summaries
are built once at the end by a single --summary-only call to avoid overwrite races):
  SO-full  (outdir v2_final/so):        DALIv2_<ds>_seed<seed>.csv
  ablation (outdir v2_final/ablation):  V2verif_<ds>_seed<seed>.csv  (verifier only)
                                        V2pi_<ds>_seed<seed>.csv     (adaptive pi only)
  ablation 'full'  reuses v2_final/so;  ablation 'v1' reuses results/so_runs.

Run examples:
  python run_v2_final.py --experiment so       --datasets steel_fatigue
  python run_v2_final.py --experiment ablation  --datasets power_plant --variants verif,pi
  python run_v2_final.py --experiment so       --summary-only
  python run_v2_final.py --experiment ablation  --summary-only
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from run_so_experiments import DATASETS, COMMON, DEFAULT_SEEDS, load_problem, auc_of
from run_dali_v2 import run_dali_v2, V2

ABLATION_VARIANTS = {
    # label -> (use_verifier, use_adaptive_pi, file_prefix, source_dir_key)
    "full":  (True,  True,  "DALIv2",  "so"),       # reuse v2_final/so
    "verif": (True,  False, "V2verif", "ablation"),
    "pi":    (False, True,  "V2pi",    "ablation"),
    "v1":    (False, False, "DALI",    "so_runs"),   # reuse results/so_runs
}


def run_one(prefix, outdir, ds_name, seed, trials, use_verifier, use_adaptive_pi):
    tag = f"{prefix}_{ds_name}_seed{seed}"
    out_path = os.path.join(outdir, tag + ".csv")
    if os.path.exists(out_path):
        print(f"[skip] {tag} (exists)", flush=True)
        return
    print(f"[run ] {tag} (verifier={use_verifier}, adaptive_pi={use_adaptive_pi}) ...", flush=True)
    comp_all, perf_all, oracle = PROBLEMS[ds_name]
    run_t0 = time.perf_counter()
    rows = run_dali_v2(ds_name, DATASETS[ds_name], seed, comp_all, perf_all, oracle,
                       trials, use_verifier=use_verifier, use_adaptive_pi=use_adaptive_pi)
    wall = time.perf_counter() - run_t0
    df = pd.DataFrame(rows)
    df.insert(0, "method", prefix)
    df.insert(1, "dataset", ds_name)
    df.insert(2, "seed", seed)
    df = df[["method", "dataset", "seed", "iteration", "best_value", "cycle_time_sec"]]
    df.to_csv(out_path, index=False)
    print(f"       done in {wall:.1f}s (final best {rows[-1]['best_value']:.4f})", flush=True)


PROBLEMS = {}


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["so", "ablation"], required=True)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--variants", default="verif,pi",
                    help="ablation arms to actually run (full/v1 are reused).")
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--base", default="../results/v2_final")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    trials = args.trials
    so_dir = os.path.join(args.base, "so")
    abl_dir = os.path.join(args.base, "ablation")
    os.makedirs(so_dir, exist_ok=True)
    os.makedirs(abl_dir, exist_ok=True)

    if args.experiment == "so":
        datasets = ([d for d in args.datasets.split(",") if d]
                    or ["steel_fatigue", "methane_conv", "power_plant"])
        if not args.summary_only:
            for ds in datasets:
                PROBLEMS[ds] = load_problem(DATASETS[ds])
            for ds in datasets:
                for seed in seeds:
                    run_one("DALIv2", so_dir, ds, seed, trials, True, True)
        build_so_summary(so_dir, datasets, seeds, trials)

    else:  # ablation
        datasets = ([d for d in args.datasets.split(",") if d]
                    or ["methane_conv", "power_plant"])
        variants = [v for v in args.variants.split(",") if v]
        if not args.summary_only:
            for ds in datasets:
                PROBLEMS[ds] = load_problem(DATASETS[ds])
            for ds in datasets:
                for v in variants:
                    uv, ua, prefix, _ = ABLATION_VARIANTS[v]
                    for seed in seeds:
                        run_one(prefix, abl_dir, ds, seed, trials, uv, ua)
        build_ablation_summary(args.base, datasets, seeds, trials)


# ---------------------------------------------------------------------------
# Summaries (built once, single process)
# ---------------------------------------------------------------------------
def _stats(paths_for_seeds, trials):
    finals, aucs, cyc = [], [], []
    for path in paths_for_seeds:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).sort_values("iteration")
        finals.append(float(df["best_value"].iloc[-1]))
        aucs.append(auc_of(df["best_value"].values))
        cyc.append(float(df["cycle_time_sec"].sum()))
    if not finals:
        return None, None
    n = len(finals)
    return {
        "n_seeds": n,
        "final_best_mean": float(np.mean(finals)),
        "final_best_std": float(np.std(finals, ddof=1)) if n > 1 else 0.0,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
        "mean_cycle_time_sec": float(np.mean(cyc) / trials),
    }, finals


def build_so_summary(so_dir, datasets, seeds, trials):
    so_runs = "../results/so_runs"
    baselines = "../results/baselines"
    sources = {  # method -> (dir, filename pattern)
        "DALIv2": (so_dir, "DALIv2_{ds}_seed{s}.csv"),
        "DALI_v1": (so_runs, "DALI_{ds}_seed{s}.csv"),
        "BO": (so_runs, "BO_{ds}_seed{s}.csv"),
        "Random": (so_runs, "Random_{ds}_seed{s}.csv"),
        "TuRBO": (baselines, "TuRBO_{ds}_seed{s}.csv"),
        "SAASBO": (baselines, "SAASBO_{ds}_seed{s}.csv"),
        "GA": (baselines, "GA_{ds}_seed{s}.csv"),
    }
    summary = {"config": {**COMMON, **V2, "trials": trials, "seeds": seeds},
               "results": {}, "v2_vs_each_mannwhitney": {}}
    for ds in datasets:
        summary["results"][ds] = {}
        finals_by_method = {}
        for method, (base, pat) in sources.items():
            st, finals = _stats(
                [os.path.join(base, pat.format(ds=ds, s=s)) for s in seeds], trials)
            if st:
                summary["results"][ds][method] = st
                finals_by_method[method] = finals
        summary["v2_vs_each_mannwhitney"][ds] = {}
        a = finals_by_method.get("DALIv2")
        if a:
            for method, b in finals_by_method.items():
                if method == "DALIv2" or not b:
                    continue
                try:
                    U, p = mannwhitneyu(a, b, alternative="two-sided")
                    summary["v2_vs_each_mannwhitney"][ds][method] = {
                        "p_two_sided": float(p),
                        "v2_mean": float(np.mean(a)), f"{method}_mean": float(np.mean(b)),
                        "v2_minus_other": float(np.mean(a) - np.mean(b)),
                    }
                except ValueError as e:
                    summary["v2_vs_each_mannwhitney"][ds][method] = {"error": str(e)}
    out = os.path.join(so_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


def build_ablation_summary(base, datasets, seeds, trials):
    so_dir = os.path.join(base, "so")
    abl_dir = os.path.join(base, "ablation")
    dirmap = {"so": so_dir, "ablation": abl_dir, "so_runs": "../results/so_runs"}
    summary = {"variants": {k: {"use_verifier": v[0], "use_adaptive_pi": v[1]}
                            for k, v in ABLATION_VARIANTS.items()},
               "results": {}, "pairwise_vs_v1_mannwhitney": {}}
    for ds in datasets:
        summary["results"][ds] = {}
        finals_by_variant = {}
        for label, (uv, ua, prefix, srckey) in ABLATION_VARIANTS.items():
            base_dir = dirmap[srckey]
            st, finals = _stats(
                [os.path.join(base_dir, f"{prefix}_{ds}_seed{s}.csv") for s in seeds], trials)
            if st:
                summary["results"][ds][label] = st
                finals_by_variant[label] = finals
        summary["pairwise_vs_v1_mannwhitney"][ds] = {}
        v1 = finals_by_variant.get("v1")
        if v1:
            for label, b in finals_by_variant.items():
                if label == "v1" or not b:
                    continue
                try:
                    U, p = mannwhitneyu(b, v1, alternative="two-sided")
                    summary["pairwise_vs_v1_mannwhitney"][ds][label] = {
                        "p_two_sided": float(p),
                        f"{label}_mean": float(np.mean(b)), "v1_mean": float(np.mean(v1)),
                        "delta_vs_v1": float(np.mean(b) - np.mean(v1)),
                    }
                except ValueError as e:
                    summary["pairwise_vs_v1_mannwhitney"][ds][label] = {"error": str(e)}
    out = os.path.join(abl_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
