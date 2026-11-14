"""Warm-start size sweep (reviewer point 9): N_init in {20, 50, 100}.

Reuses the exact DALI / BO / Random implementations from run_so_experiments.py,
only varying the warm-start size. N_init=200 results come from results/so_runs.

Per-run CSV: method, dataset, seed, n_init, iteration, best_value, cycle_time_sec
  (file: <method>_<dataset>_n<init>_seed<seed>.csv — existing files are skipped,
   so interrupted sweeps resume by re-running the same command)
A failed run writes <tag>.FAILED.txt and the sweep continues (no global abort).

summary.json: per n_init x dataset x method final best / AUC stats and
DALI-vs-BO Mann-Whitney p-values, including the n_init=200 reference.
"""

import argparse
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import run_so_experiments as so
from run_so_experiments import (
    DATASETS, COMMON, DEFAULT_SEEDS, load_problem, auc_of,
)

N_INIT_LEVELS = [20, 50, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="DALI,BO,Random")
    ap.add_argument("--datasets", default="steel_fatigue,methane_conv,power_plant")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--n-inits", default=",".join(map(str, N_INIT_LEVELS)))
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default="../results/warmstart_sweep")
    ap.add_argument("--ref-dir", default="../results/so_runs",
                    help="n_init=200 reference results")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    datasets = [d for d in args.datasets.split(",") if d]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    n_inits = [int(n) for n in args.n_inits.split(",") if n != ""]
    trials = args.trials
    os.makedirs(args.outdir, exist_ok=True)

    if not args.summary_only:
        problems = {}
        for ds_name in datasets:
            print(f"[load] {ds_name} ...", flush=True)
            problems[ds_name] = load_problem(DATASETS[ds_name])

        for ds_name in datasets:
            comp_all, perf_all, oracle = problems[ds_name]
            for n_init in n_inits:
                for method in methods:
                    for seed in seeds:
                        tag = f"{method}_{ds_name}_n{n_init}_seed{seed}"
                        out_path = os.path.join(args.outdir, tag + ".csv")
                        fail_path = os.path.join(args.outdir, tag + ".FAILED.txt")
                        if os.path.exists(out_path):
                            print(f"[skip] {tag} (exists)", flush=True)
                            continue
                        if os.path.exists(fail_path):
                            print(f"[skip] {tag} (previously failed)", flush=True)
                            continue
                        print(f"[run ] {tag} ...", flush=True)
                        # runners read COMMON at call time; override warm-start size
                        so.COMMON["initial_train_size"] = n_init
                        run_t0 = time.perf_counter()
                        try:
                            rows, _ = so.METHODS[method](
                                ds_name, DATASETS[ds_name], seed,
                                comp_all, perf_all, oracle, trials,
                            )
                        except Exception:
                            err = traceback.format_exc()
                            with open(fail_path, "w", encoding="utf-8") as f:
                                f.write(err)
                            print(f"       FAILED ({err.splitlines()[-1]}) — continuing",
                                  flush=True)
                            continue
                        finally:
                            so.COMMON["initial_train_size"] = 200  # restore
                        wall = time.perf_counter() - run_t0
                        df = pd.DataFrame(rows)
                        df.insert(0, "method", method)
                        df.insert(1, "dataset", ds_name)
                        df.insert(2, "seed", seed)
                        df.insert(3, "n_init", n_init)
                        df = df[["method", "dataset", "seed", "n_init", "iteration",
                                 "best_value", "cycle_time_sec"]]
                        df.to_csv(out_path, index=False)
                        print(f"       done in {wall:.1f}s "
                              f"(final best {rows[-1]['best_value']:.4f})", flush=True)

    build_summary(args.outdir, args.ref_dir, methods, datasets, seeds, n_inits, trials)


def collect(outdir, method, ds_name, n_init, seeds, trials, ref_dir=None):
    """Gather finals/aucs; n_init=200 reads from ref_dir with the so_runs naming."""
    finals, aucs, cyc = [], [], []
    for seed in seeds:
        if n_init == 200 and ref_dir is not None:
            path = os.path.join(ref_dir, f"{method}_{ds_name}_seed{seed}.csv")
        else:
            path = os.path.join(outdir, f"{method}_{ds_name}_n{n_init}_seed{seed}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).sort_values("iteration")
        finals.append(float(df["best_value"].iloc[-1]))
        aucs.append(auc_of(df["best_value"].values))
        cyc.append(float(df["cycle_time_sec"].sum()))
    return finals, aucs, cyc


def build_summary(outdir, ref_dir, methods, datasets, seeds, n_inits, trials):
    levels = sorted(set(n_inits + [200]))
    summary = {"config": {"n_init_levels": levels, "trials": trials, "seeds": seeds,
                          "note": "n_init=200 stats taken from results/so_runs"},
               "results": {}, "dali_vs_bo_mannwhitney": {}}

    for ds_name in datasets:
        summary["results"][ds_name] = {}
        summary["dali_vs_bo_mannwhitney"][ds_name] = {}
        for n_init in levels:
            key = f"n_init={n_init}"
            summary["results"][ds_name][key] = {}
            per_method_finals = {}
            for method in methods:
                finals, aucs, cyc = collect(outdir, method, ds_name, n_init, seeds,
                                            trials, ref_dir)
                if not finals:
                    continue
                n = len(finals)
                per_method_finals[method] = finals
                summary["results"][ds_name][key][method] = {
                    "n_seeds": n,
                    "final_best_mean": float(np.mean(finals)),
                    "final_best_std": float(np.std(finals, ddof=1)) if n > 1 else 0.0,
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
                    "total_wall_clock_sec": float(np.sum(cyc)),
                }
            a = per_method_finals.get("DALI")
            b = per_method_finals.get("BO")
            if a and b:
                try:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                    summary["dali_vs_bo_mannwhitney"][ds_name][key] = {
                        "U": float(stat), "p_two_sided": float(p),
                        "dali_final_mean": float(np.mean(a)),
                        "bo_final_mean": float(np.mean(b)),
                        "dali_minus_bo": float(np.mean(a) - np.mean(b)),
                    }
                except ValueError as e:
                    summary["dali_vs_bo_mannwhitney"][ds_name][key] = {"error": str(e)}

    out = os.path.join(outdir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
