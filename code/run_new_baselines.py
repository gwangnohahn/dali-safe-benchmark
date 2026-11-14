"""New single-objective baselines for DALI resubmission (reviewer-requested).

Methods:
  - TuRBO   : trust-region BO (BoTorch, TuRBO-1 style), batch q=5 via Thompson sampling
  - SAASBO  : fully Bayesian SAAS GP (BoTorch) + qLogEI, batch q=5
  - GA      : conventional surrogate-based optimization — XGBoost surrogate refit each
              cycle + pymoo GA proposes candidates, top-B (deduplicated) evaluated by oracle

Protocol identical to run_so_experiments.py (same warm start, oracle, objective,
20 cycles, B=5, seeds). Per-run CSV columns: method, dataset, seed, iteration,
best_value, cycle_time_sec. Summary compares against DALI/BO/Random in results/so_runs.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize

from run_so_experiments import (
    DATASETS, COMMON, DEFAULT_SEEDS,
    objective_value, load_problem, warm_start_indices, auc_of,
    save_setup_times, load_setup_times,
)

DTYPE = torch.double
DEVICE = torch.device("cpu")


def normalize_x(X_raw, lb, ub):
    return (X_raw - lb) / (ub - lb)


def denormalize_x(X_norm, lb, ub):
    return X_norm * (ub - lb) + lb


# ---------------------------------------------------------------------------
# TuRBO-1 (BoTorch tutorial implementation, batch Thompson sampling)
# ---------------------------------------------------------------------------
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def update_turbo_state(state, Y_next):
    if Y_next.max().item() > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, Y_next.max().item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def turbo_generate_batch(state, model, X, Y, batch_size, n_candidates, seed_offset):
    # Trust region centered at best observed point, scaled by GP lengthscales
    x_center = X[Y.argmax(), :].clone()
    try:
        ls = model.covar_module.lengthscale
        if ls is None:
            raise AttributeError
        weights = ls.squeeze().detach()
    except AttributeError:
        weights = torch.ones(X.shape[-1], dtype=DTYPE)
    if weights.ndim == 0:
        weights = weights.unsqueeze(0)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    # Perturbation-based candidate generation (TuRBO tutorial)
    dim = X.shape[-1]
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed_offset)
    pert = sobol.draw(n_candidates).to(dtype=DTYPE)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),))] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():
        X_next = thompson_sampling(X_cand, num_samples=batch_size)
    return X_next


def run_turbo(ds_name, ds_cfg, seed, comp_all, perf_all, oracle, trials):
    np.random.seed(seed)
    torch.manual_seed(seed)
    direction, weights = ds_cfg["direction"], ds_cfg["weights"]

    lb = comp_all.min(axis=0)
    ub = comp_all.max(axis=0)
    dim = len(ds_cfg["cond_idx"])

    idx_init = warm_start_indices(comp_all.shape[0], COMMON["initial_train_size"], seed)
    X_raw = comp_all[idx_init].copy()
    scores = np.array([objective_value(p, direction, weights) for p in perf_all[idx_init]])

    X = torch.tensor(normalize_x(X_raw, lb, ub), dtype=DTYPE)
    Y = torch.tensor(scores, dtype=DTYPE).unsqueeze(-1)

    state = TurboState(dim=dim, batch_size=COMMON["batch_size"])
    state.best_value = float(Y.max())
    best = float(Y.max())
    rows = []

    for t in range(trials):
        t0 = time.perf_counter()
        if state.restart_triggered:
            # restart trust region, keep data (standard TuRBO restart)
            state = TurboState(dim=dim, batch_size=COMMON["batch_size"])
            state.best_value = float(Y.max())

        train_Y = Y
        model = SingleTaskGP(X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        n_cand = min(5000, max(2000, 200 * dim))
        X_next = turbo_generate_batch(
            state, model, X, Y.squeeze(-1), COMMON["batch_size"], n_cand,
            seed_offset=seed * 1000 + t,
        )

        X_next_raw = denormalize_x(X_next.numpy(), lb, ub)
        perf_preds = oracle.predict(X_next_raw)
        y_next = np.array([objective_value(p, direction, weights) for p in perf_preds])

        Y_next = torch.tensor(y_next, dtype=DTYPE).unsqueeze(-1)
        state = update_turbo_state(state, Y_next)
        X = torch.cat([X, X_next], dim=0)
        Y = torch.cat([Y, Y_next], dim=0)
        best = max(best, float(y_next.max()))

        cycle_time = time.perf_counter() - t0
        rows.append({"iteration": t + 1, "best_value": best, "cycle_time_sec": cycle_time})

    return rows, 0.0


# ---------------------------------------------------------------------------
# SAASBO (fully Bayesian SAAS GP + qLogEI)
# ---------------------------------------------------------------------------
SAAS_NUTS = {"warmup_steps": 256, "num_samples": 128, "thinning": 16}


def run_saasbo(ds_name, ds_cfg, seed, comp_all, perf_all, oracle, trials):
    np.random.seed(seed)
    torch.manual_seed(seed)
    direction, weights = ds_cfg["direction"], ds_cfg["weights"]

    lb = comp_all.min(axis=0)
    ub = comp_all.max(axis=0)
    dim = len(ds_cfg["cond_idx"])
    bounds = torch.stack([torch.zeros(dim, dtype=DTYPE), torch.ones(dim, dtype=DTYPE)])

    idx_init = warm_start_indices(comp_all.shape[0], COMMON["initial_train_size"], seed)
    X_raw = comp_all[idx_init].copy()
    scores = np.array([objective_value(p, direction, weights) for p in perf_all[idx_init]])

    X = torch.tensor(normalize_x(X_raw, lb, ub), dtype=DTYPE)
    Y = torch.tensor(scores, dtype=DTYPE).unsqueeze(-1)
    best = float(Y.max())
    rows = []

    for t in range(trials):
        t0 = time.perf_counter()
        model = SaasFullyBayesianSingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=SAAS_NUTS["warmup_steps"],
            num_samples=SAAS_NUTS["num_samples"],
            thinning=SAAS_NUTS["thinning"],
            disable_progbar=True,
        )
        acqf = qLogExpectedImprovement(model=model, best_f=Y.max())
        X_next, _ = optimize_acqf(
            acqf, bounds=bounds, q=COMMON["batch_size"],
            num_restarts=10, raw_samples=256,
        )

        X_next_raw = denormalize_x(X_next.detach().numpy(), lb, ub)
        perf_preds = oracle.predict(X_next_raw)
        y_next = np.array([objective_value(p, direction, weights) for p in perf_preds])

        X = torch.cat([X, X_next.detach()], dim=0)
        Y = torch.cat([Y, torch.tensor(y_next, dtype=DTYPE).unsqueeze(-1)], dim=0)
        best = max(best, float(y_next.max()))

        cycle_time = time.perf_counter() - t0
        rows.append({"iteration": t + 1, "best_value": best, "cycle_time_sec": cycle_time})
        print(f"      [SAASBO] cycle {t+1}/{trials}: {cycle_time:.1f}s, best {best:.4f}",
              flush=True)

    return rows, 0.0


# ---------------------------------------------------------------------------
# GA-based surrogate optimization (XGBoost surrogate + pymoo GA)
# ---------------------------------------------------------------------------
class SurrogateProblem(Problem):
    def __init__(self, surrogate, direction, weights, lb, ub):
        super().__init__(n_var=len(lb), n_obj=1, xl=lb, xu=ub)
        self.surrogate = surrogate
        self.direction = direction
        self.weights = weights

    def _evaluate(self, X, out, *args, **kwargs):
        perf = self.surrogate.predict(X)
        if perf.ndim == 1:
            perf = perf.reshape(-1, 1)
        scores = perf * np.asarray(self.direction, dtype=float)
        scores = scores @ np.asarray(self.weights, dtype=float)
        out["F"] = -scores.reshape(-1, 1)  # pymoo minimizes


def run_ga(ds_name, ds_cfg, seed, comp_all, perf_all, oracle, trials):
    np.random.seed(seed)
    direction, weights = ds_cfg["direction"], ds_cfg["weights"]

    lb = comp_all.min(axis=0)
    ub = comp_all.max(axis=0)

    idx_init = warm_start_indices(comp_all.shape[0], COMMON["initial_train_size"], seed)
    X_obs = comp_all[idx_init].copy()
    P_obs = perf_all[idx_init].copy()

    best = max(objective_value(p, direction, weights) for p in P_obs)
    rows = []

    for t in range(trials):
        t0 = time.perf_counter()
        # surrogate refit on all observed data (same arch as oracle)
        surrogate = MultiOutputRegressor(
            XGBRegressor(objective="reg:squarederror", n_estimators=200, max_depth=4,
                         random_state=seed, n_jobs=-1, verbosity=0)
        )
        surrogate.fit(X_obs, P_obs)

        problem = SurrogateProblem(surrogate, direction, weights, lb, ub)
        algo = PymooGA(pop_size=50, eliminate_duplicates=True)
        res = pymoo_minimize(problem, algo, ("n_gen", 30), seed=seed * 1000 + t,
                             verbose=False)

        # top-B distinct candidates from final population
        pop_X = res.pop.get("X")
        pop_F = res.pop.get("F").flatten()
        order = np.argsort(pop_F)  # ascending = best surrogate score first
        chosen = []
        for i in order:
            x = pop_X[i]
            if all(np.linalg.norm(x - c) > 1e-8 for c in chosen):
                chosen.append(x)
            if len(chosen) == COMMON["batch_size"]:
                break
        chosen = np.array(chosen)

        perf_preds = oracle.predict(chosen)
        for perf_pred in perf_preds:
            best = max(best, objective_value(perf_pred, direction, weights))
        X_obs = np.vstack([X_obs, chosen])
        P_obs = np.vstack([P_obs, perf_preds])

        cycle_time = time.perf_counter() - t0
        rows.append({"iteration": t + 1, "best_value": best, "cycle_time_sec": cycle_time})

    return rows, 0.0


METHODS = {"TuRBO": run_turbo, "SAASBO": run_saasbo, "GA": run_ga}


# ---------------------------------------------------------------------------
# Orchestration (same CSV/summary format as run_so_experiments)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="TuRBO,GA")
    ap.add_argument("--datasets", default="steel_fatigue,methane_conv,power_plant")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default="../results/baselines")
    ap.add_argument("--dali-dir", default="../results/so_runs",
                    help="Existing DALI/BO results for statistical comparison.")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    datasets = [d for d in args.datasets.split(",") if d]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    trials = args.trials
    os.makedirs(args.outdir, exist_ok=True)

    if not args.summary_only:
        problems = {}
        for ds_name in datasets:
            print(f"[load] {ds_name} ...", flush=True)
            problems[ds_name] = load_problem(DATASETS[ds_name])

        for ds_name in datasets:
            comp_all, perf_all, oracle = problems[ds_name]
            for method in methods:
                for seed in seeds:
                    tag = f"{method}_{ds_name}_seed{seed}"
                    out_path = os.path.join(args.outdir, tag + ".csv")
                    if os.path.exists(out_path):
                        print(f"[skip] {tag} (exists)", flush=True)
                        continue
                    print(f"[run ] {tag} ...", flush=True)
                    run_t0 = time.perf_counter()
                    rows, _ = METHODS[method](
                        ds_name, DATASETS[ds_name], seed,
                        comp_all, perf_all, oracle, trials,
                    )
                    wall = time.perf_counter() - run_t0
                    df = pd.DataFrame(rows)
                    df.insert(0, "method", method)
                    df.insert(1, "dataset", ds_name)
                    df.insert(2, "seed", seed)
                    df = df[["method", "dataset", "seed", "iteration", "best_value", "cycle_time_sec"]]
                    df.to_csv(out_path, index=False)
                    print(f"       done in {wall:.1f}s (final best {rows[-1]['best_value']:.4f})",
                          flush=True)

    build_summary(args.outdir, args.dali_dir, methods, datasets, seeds, trials)


def collect_stats(outdir, method, ds_name, seeds, trials):
    finals, aucs, cyc_times = [], [], []
    for seed in seeds:
        path = os.path.join(outdir, f"{method}_{ds_name}_seed{seed}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).sort_values("iteration")
        finals.append(float(df["best_value"].iloc[-1]))
        aucs.append(auc_of(df["best_value"].values))
        cyc_times.append(float(df["cycle_time_sec"].sum()))
    if not finals:
        return None
    n = len(finals)
    return {
        "n_seeds": n,
        "final_best_mean": float(np.mean(finals)),
        "final_best_std": float(np.std(finals, ddof=1)) if n > 1 else 0.0,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
        "total_wall_clock_sec": float(np.sum(cyc_times)),
        "mean_cycle_time_sec": float(np.mean(cyc_times) / trials),
        "_finals": finals,
    }


def build_summary(outdir, dali_dir, methods, datasets, seeds, trials):
    summary = {"config": {**COMMON, "trials": trials, "seeds": seeds,
                          "saasbo_nuts": SAAS_NUTS},
               "results": {}, "dali_vs_new_mannwhitney": {}}

    for ds_name in datasets:
        summary["results"][ds_name] = {}
        # new baselines from this outdir
        for method in methods:
            st = collect_stats(outdir, method, ds_name, seeds, trials)
            if st:
                finals = st.pop("_finals")
                summary["results"][ds_name][method] = st
                summary["results"][ds_name][method]["final_best_per_seed"] = dict(
                    zip(map(str, seeds[:len(finals)]), finals))
        # reference methods from so_runs
        for ref in ("DALI", "BO", "Random"):
            st = collect_stats(dali_dir, ref, ds_name, seeds, trials)
            if st:
                st.pop("_finals")
                summary["results"][ds_name][f"{ref} (ref)"] = st

        # Mann-Whitney: DALI vs each new baseline
        dali_st = collect_stats(dali_dir, "DALI", ds_name, seeds, trials)
        if dali_st:
            a = dali_st["_finals"]
            summary["dali_vs_new_mannwhitney"][ds_name] = {}
            for method in methods:
                st = collect_stats(outdir, method, ds_name, seeds, trials)
                if not st:
                    continue
                b = st["_finals"]
                try:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                    summary["dali_vs_new_mannwhitney"][ds_name][method] = {
                        "U": float(stat), "p_two_sided": float(p),
                        "dali_final_mean": float(np.mean(a)),
                        f"{method.lower()}_final_mean": float(np.mean(b)),
                    }
                except ValueError as e:
                    summary["dali_vs_new_mannwhitney"][ds_name][method] = {"error": str(e)}

    out = os.path.join(outdir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
