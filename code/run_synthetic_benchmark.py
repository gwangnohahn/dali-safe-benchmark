"""Synthetic benchmark runner for the DALI revision (Tracks A/B/C).

Reviewer mapping:
  Track A (standard analytical functions)          -> R2-2
  Track B (sparsity x warm-start coverage control) -> R3-2, R2-1
  Track C (known feasibility constraints)          -> R3-3

Protocol (identical to the emulator experiments wherever applicable):
  warm start 200 (problem-specific sampler), 20 cycles x batch 5 = 100 oracle
  evaluations, seeds = DEFAULT_SEEDS (10), per-cycle wall-clock recorded.

Differences vs the emulator runners -- all deliberate, all documented for the paper:
  * oracle = the analytical function itself (ground truth; no emulator, no
    circularity).
  * search/clip bounds = the problem's KNOWN box [lb, ub] for every method
    (for the emulator datasets the box was estimated from data ranges; here the
    box is the prescribed domain, the exact analogue of "coarse physical limits").
  * GA surrogate (XGBoost) no longer shares a model family with the oracle
    (the oracle is analytic), which removes the model-match advantage it had in
    the emulator benchmarks.
  * Constrained problems (Track C): NO method is given the constraint. Warm-start
    data are feasible-only (mimicking a database of successful experiments).
    An infeasible proposal counts as a FAILED experiment: it consumes budget and
    returns a death-penalty observation f_pen = max(f_warm) + 0.1*range(f_warm)
    (the same value for every method); it never improves the reported best, which
    is tracked over FEASIBLE evaluations only. Violations are recorded per cycle.

Outputs (results/synthetic_benchmark/<track>/):
  <method>_<problem>_seed<seed>.csv :
      method, problem, seed, iteration, best_value, best_f, n_infeasible, cycle_time_sec
      (best_value = -best_f keeps the pipeline's maximize convention; best_f is the
       raw minimized objective over feasible evaluations, the value to report.)
  <method>_<problem>_seed<seed>_proposals.csv :
      iteration, x0..x{D-1}, f_true, feasible   (every evaluated point, for the
      density/trajectory analyses answering R2-1)
  summary.json : per problem/method final best_f mean+/-std, AUC, total violations,
      Mann-Whitney U DALIv2-vs-each-method with Holm correction within problem.

Usage:
  python run_synthetic_benchmark.py --track A            # all Track A problems
  python run_synthetic_benchmark.py --track B --methods DALIv2,BO,TuRBO,GA
  python run_synthetic_benchmark.py --problems branin2 --methods DALIv2 --seeds 43 --trials 2
  python run_synthetic_benchmark.py --track A --summary-only
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from skopt import Optimizer
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern

from ml.model import FCNN, CondDiffu
from util.data import Dataset

from run_so_experiments import (
    COMMON, DEFAULT_SEEDS, auc_of, comp_uncertainty_knn, greedy_diverse_batch,
)
from run_dali_v2 import (
    V2, adaptive_pi, greedy_select_v2, train_verifier_ensemble, verifier_predict,
)
import problems_synthetic as SP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

TRACKS = {"A": SP.TRACK_A, "B": SP.TRACK_B, "C": SP.TRACK_C}
BASELINE_ORDER = ["DALIbase", "BO", "TuRBO", "SAASBO", "GA", "Random"]


# ---------------------------------------------------------------------------
# Shared evaluation bookkeeping
# ---------------------------------------------------------------------------
class EvalTracker:
    """Constraint-aware evaluation wrapper shared by every method."""

    def __init__(self, prob, f_warm):
        self.prob = prob
        f_warm = np.asarray(f_warm, dtype=float)
        rng = float(f_warm.max() - f_warm.min())
        self.penalty_f = float(f_warm.max() + 0.1 * (rng if rng > 0 else 1.0))
        self.best_f = float(f_warm.min())  # warm start is feasible by construction
        self.proposals = []

    def evaluate(self, X, iteration):
        """Return observed values f_obs (death penalty on infeasible) and log truth."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f_true = np.asarray(self.prob.evaluate(X), dtype=float)
        if self.prob.constrained:
            feas = np.asarray(self.prob.feasible(X), dtype=bool)
        else:
            feas = np.ones(X.shape[0], dtype=bool)
        f_obs = np.where(feas, f_true, self.penalty_f)
        if feas.any():
            self.best_f = min(self.best_f, float(f_true[feas].min()))
        for i in range(X.shape[0]):
            row = {"iteration": iteration}
            row.update({f"x{d}": float(X[i, d]) for d in range(X.shape[1])})
            row["f_true"] = float(f_true[i])
            row["feasible"] = bool(feas[i])
            self.proposals.append(row)
        return f_obs, int((~feas).sum())

    def row(self, iteration, n_infeasible, cycle_time):
        return {
            "iteration": iteration,
            "best_value": -self.best_f,   # pipeline maximize convention
            "best_f": self.best_f,        # raw minimize value (report this)
            "n_infeasible": n_infeasible,
            "cycle_time_sec": cycle_time,
        }


# ---------------------------------------------------------------------------
# DALI (diffusion inverse model + generate-then-filter); v2 modules by flags
# ---------------------------------------------------------------------------
def run_dali_syn(prob, seed, trials, use_verifier=True, use_adaptive_pi=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_cond, n_perf = prob.dim, 1
    clip_lb, clip_ub = prob.lb, prob.ub

    X0, f0 = prob.warm_start(seed)
    tracker = EvalTracker(prob, f0)
    perf_train = f0.reshape(-1, 1)
    comp_train = X0

    x_mean = perf_train.mean(axis=0, keepdims=True)
    x_std = perf_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = comp_train.mean(axis=0, keepdims=True)
    y_std = comp_train.std(axis=0, keepdims=True) + 1e-8
    dataset_train = Dataset(
        torch.tensor((perf_train - x_mean) / x_std, dtype=torch.float),
        torch.tensor((comp_train - y_mean) / y_std, dtype=torch.float),
        x_mean, x_std, y_mean, y_std,
    )
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    var_stat = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    pred_model = FCNN(n_perf, n_cond, dataset_train).to(device)
    opt_pred = torch.optim.AdamW(pred_model.parameters(), lr=5e-4, weight_decay=1e-6)
    loss_func = torch.nn.MSELoss()
    for _ in range(200):
        pred_model.fit(loader_train, opt_pred, loss_func)

    diffu_model = CondDiffu(n_perf, n_cond, num_steps=200,
                            pred_model=pred_model, var_stat=var_stat).to(device)
    opt_diffu = torch.optim.AdamW(diffu_model.parameters(), lr=5e-4, weight_decay=5e-6)
    for _ in range(200):
        diffu_model.fit(loader_train, opt_diffu)

    verifiers = []
    if use_verifier:
        for i in range(V2["n_verifiers"]):
            torch.manual_seed(seed * 100 + i)
            m = FCNN(n_cond, n_perf, dataset_train).to(device)
            verifiers.append((m, torch.optim.AdamW(m.parameters(), lr=5e-4,
                                                   weight_decay=1e-6)))
        torch.manual_seed(seed)
        train_verifier_ensemble(verifiers, dataset_train.y.to(device),
                                dataset_train.x.to(device), epochs=200)

    rows = []
    for trial in range(trials):
        t0 = time.perf_counter()

        # 1) sample target performances from the observed range (s = -f, higher better)
        perf_raw = dataset_train.x.numpy() * x_std + x_mean
        lb_f, ub_f = perf_raw.min(axis=0), perf_raw.max(axis=0)
        targets_raw = np.random.uniform(lb_f, ub_f,
                                        size=(COMMON["num_target_candidates"], n_perf))
        pi = adaptive_pi(trial, trials) if use_adaptive_pi \
            else COMMON["candidate_score_percentile"]
        scores_tmp = -targets_raw[:, 0]
        threshold = np.percentile(scores_tmp, pi)
        targets_raw = targets_raw[scores_tmp >= threshold]
        if targets_raw.shape[0] == 0:
            targets_raw = np.random.uniform(lb_f, ub_f, size=(10, n_perf))

        # 2) diffusion generates conditions, clipped to the known box
        all_comps = []
        for cand_x in targets_raw:
            cand_norm = (cand_x - x_mean) / x_std
            cand_tensor = torch.tensor(cand_norm, dtype=torch.float32).reshape(1, -1)
            for _ in range(COMMON["num_comp_samples"]):
                comp_norm = diffu_model.predict(cand_tensor).numpy()[0]
                comp_raw = (comp_norm * y_std + y_mean)[0]
                all_comps.append(np.clip(comp_raw, clip_lb, clip_ub))
        all_comps = np.array(all_comps)
        comps_norm = (all_comps - y_mean) / y_std

        # 3) filter/selection: verifier ensemble (v2) or kNN+diversity (base)
        if use_verifier:
            cand_tensor = torch.tensor(comps_norm, dtype=torch.float32, device=device)
            ens_pred_norm = verifier_predict(verifiers, cand_tensor)
            ens_obj = -(ens_pred_norm * x_std + x_mean)[..., 0]     # (E, N): s = -f
            P_score, U_score = ens_obj.mean(axis=0), ens_obj.std(axis=0)
            N = comps_norm.shape[0]
            keep_n = max(COMMON["batch_size"], int(np.ceil(N * V2["verifier_keep_frac"])))
            keep_idx = np.argsort(-P_score)[:keep_n]
            sel_local = greedy_select_v2(
                comps_norm[keep_idx], P_score[keep_idx], U_score[keep_idx],
                COMMON["batch_size"], V2["w_perf"], COMMON["alpha_uncertainty"])
            sel_idx = keep_idx[sel_local]
        else:
            train_norm = dataset_train.y.numpy()
            U = comp_uncertainty_knn(comps_norm, train_norm, k=COMMON["knn_k"])
            sel_idx = greedy_diverse_batch(comps_norm, U, COMMON["batch_size"],
                                           COMMON["alpha_uncertainty"])

        # 4) oracle evaluates the selected batch (death penalty if infeasible)
        X_sel = all_comps[sel_idx]
        f_obs, n_inf = tracker.evaluate(X_sel, trial + 1)
        for x_raw, f_o in zip(X_sel, f_obs):
            perf_norm = (np.array([[f_o]]) - x_mean) / x_std
            comp_norm = (x_raw.reshape(1, -1) - y_mean) / y_std
            dataset_train.x = torch.cat(
                [dataset_train.x, torch.tensor(perf_norm, dtype=torch.float)], dim=0)
            dataset_train.y = torch.cat(
                [dataset_train.y, torch.tensor(comp_norm, dtype=torch.float)], dim=0)

        # 5) retrain
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64,
                                                   shuffle=True)
        for _ in range(50):
            pred_model.fit(loader_train, opt_pred, loss_func)
        for _ in range(50):
            diffu_model.fit(loader_train, opt_diffu)
        if use_verifier:
            train_verifier_ensemble(verifiers, dataset_train.y.to(device),
                                    dataset_train.x.to(device), epochs=50)

        rows.append(tracker.row(trial + 1, n_inf, time.perf_counter() - t0))

    return rows, tracker.proposals


def run_dali_v2_syn(prob, seed, trials):
    return run_dali_syn(prob, seed, trials, use_verifier=True, use_adaptive_pi=True)


def run_dali_base_syn(prob, seed, trials):
    return run_dali_syn(prob, seed, trials, use_verifier=False, use_adaptive_pi=False)


# ---------------------------------------------------------------------------
# BO (skopt + GP Matern 2.5), same construction as the emulator runner
# ---------------------------------------------------------------------------
def run_bo_syn(prob, seed, trials):
    np.random.seed(seed)
    X0, f0 = prob.warm_start(seed)
    tracker = EvalTracker(prob, f0)

    scaler = MinMaxScaler().fit(np.vstack([prob.lb, prob.ub]))
    space = [Real(0.0, 1.0, name=f"n_x{i}") for i in range(prob.dim)]
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                   alpha=1e-10, n_restarts_optimizer=5,
                                   random_state=seed)
    opt = Optimizer(space, base_estimator=gpr, random_state=seed,
                    n_initial_points=0, initial_point_generator="random")
    opt.tell(scaler.transform(X0).tolist(), f0.tolist())  # skopt minimizes f

    rows = []
    for t in range(trials):
        t0 = time.perf_counter()
        cands_norm = opt.ask(n_points=COMMON["batch_size"])
        cands_raw = scaler.inverse_transform(np.asarray(cands_norm))
        f_obs, n_inf = tracker.evaluate(cands_raw, t + 1)
        opt.tell(cands_norm, f_obs.tolist())
        rows.append(tracker.row(t + 1, n_inf, time.perf_counter() - t0))
    return rows, tracker.proposals


# ---------------------------------------------------------------------------
# Random search in the known box
# ---------------------------------------------------------------------------
def run_random_syn(prob, seed, trials):
    rng = np.random.RandomState(seed)
    X0, f0 = prob.warm_start(seed)
    tracker = EvalTracker(prob, f0)
    rows = []
    for t in range(trials):
        t0 = time.perf_counter()
        cands = prob.lb + rng.rand(COMMON["batch_size"], prob.dim) * (prob.ub - prob.lb)
        _, n_inf = tracker.evaluate(cands, t + 1)
        rows.append(tracker.row(t + 1, n_inf, time.perf_counter() - t0))
    return rows, tracker.proposals


# ---------------------------------------------------------------------------
# TuRBO-1 and SAASBO (BoTorch), adapted from run_new_baselines with box bounds
# ---------------------------------------------------------------------------
def run_turbo_syn(prob, seed, trials):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from run_new_baselines import TurboState, update_turbo_state, turbo_generate_batch

    np.random.seed(seed)
    torch.manual_seed(seed)
    X0, f0 = prob.warm_start(seed)
    tracker = EvalTracker(prob, f0)
    lb, ub = prob.lb, prob.ub

    X = torch.tensor((X0 - lb) / (ub - lb), dtype=DTYPE)
    Y = torch.tensor(-f0, dtype=DTYPE).unsqueeze(-1)  # maximize s = -f
    state = TurboState(dim=prob.dim, batch_size=COMMON["batch_size"])
    state.best_value = float(Y.max())

    rows = []
    for t in range(trials):
        t0 = time.perf_counter()
        if state.restart_triggered:
            state = TurboState(dim=prob.dim, batch_size=COMMON["batch_size"])
            state.best_value = float(Y.max())
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        n_cand = min(5000, max(2000, 200 * prob.dim))
        X_next = turbo_generate_batch(state, model, X, Y.squeeze(-1),
                                      COMMON["batch_size"], n_cand,
                                      seed_offset=seed * 1000 + t)
        X_next_raw = X_next.numpy() * (ub - lb) + lb
        f_obs, n_inf = tracker.evaluate(X_next_raw, t + 1)
        Y_next = torch.tensor(-f_obs, dtype=DTYPE).unsqueeze(-1)
        state = update_turbo_state(state, Y_next)
        X = torch.cat([X, X_next], dim=0)
        Y = torch.cat([Y, Y_next], dim=0)
        rows.append(tracker.row(t + 1, n_inf, time.perf_counter() - t0))
    return rows, tracker.proposals


def run_saasbo_syn(prob, seed, trials):
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from run_new_baselines import SAAS_NUTS

    np.random.seed(seed)
    torch.manual_seed(seed)
    X0, f0 = prob.warm_start(seed)
    tracker = EvalTracker(prob, f0)
    lb, ub = prob.lb, prob.ub
    bounds = torch.stack([torch.zeros(prob.dim, dtype=DTYPE),
                          torch.ones(prob.dim, dtype=DTYPE)])

    X = torch.tensor((X0 - lb) / (ub - lb), dtype=DTYPE)
    Y = torch.tensor(-f0, dtype=DTYPE).unsqueeze(-1)

    rows = []
    for t in range(trials):
        t0 = time.perf_counter()
        model = SaasFullyBayesianSingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=SAAS_NUTS["warmup_steps"],
            num_samples=SAAS_NUTS["num_samples"], thinning=SAAS_NUTS["thinning"],
            disable_progbar=True)
        acqf = qLogExpectedImprovement(model=model, best_f=Y.max())
        X_next, _ = optimize_acqf(acqf, bounds=bounds, q=COMMON["batch_size"],
                                  num_restarts=10, raw_samples=256)
        X_next_raw = X_next.detach().numpy() * (ub - lb) + lb
        f_obs, n_inf = tracker.evaluate(X_next_raw, t + 1)
        X = torch.cat([X, X_next.detach()], dim=0)
        Y = torch.cat([Y, torch.tensor(-f_obs, dtype=DTYPE).unsqueeze(-1)], dim=0)
        rows.append(tracker.row(t + 1, n_inf, time.perf_counter() - t0))
        print(f"      [SAASBO] cycle {t+1}/{trials}: "
              f"{rows[-1]['cycle_time_sec']:.1f}s best_f {tracker.best_f:.4f}",
              flush=True)
    return rows, tracker.proposals


# ---------------------------------------------------------------------------
# GA + XGBoost surrogate (surrogate family is now DIFFERENT from the oracle)
# ---------------------------------------------------------------------------
def run_ga_syn(prob, seed, trials):
    from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.optimize import minimize as pymoo_minimize

    np.random.seed(seed)
    X0, f0 = prob.warm_start(seed)
    tracker = EvalTracker(prob, f0)
    X_obs, F_obs = X0.copy(), f0.copy()

    class _SurrogateProblem(PymooProblem):
        def __init__(self, surrogate):
            super().__init__(n_var=prob.dim, n_obj=1, xl=prob.lb, xu=prob.ub)
            self.surrogate = surrogate

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = self.surrogate.predict(X).reshape(-1, 1)  # minimize f

    rows = []
    for t in range(trials):
        t0 = time.perf_counter()
        surrogate = XGBRegressor(objective="reg:squarederror", n_estimators=200,
                                 max_depth=4, random_state=seed, n_jobs=-1,
                                 verbosity=0)
        surrogate.fit(X_obs, F_obs)
        algo = PymooGA(pop_size=50, eliminate_duplicates=True)
        res = pymoo_minimize(_SurrogateProblem(surrogate), algo, ("n_gen", 30),
                             seed=seed * 1000 + t, verbose=False)
        pop_X, pop_F = res.pop.get("X"), res.pop.get("F").flatten()
        chosen = []
        for i in np.argsort(pop_F):
            x = pop_X[i]
            if all(np.linalg.norm(x - c) > 1e-8 for c in chosen):
                chosen.append(x)
            if len(chosen) == COMMON["batch_size"]:
                break
        chosen = np.array(chosen)
        f_obs, n_inf = tracker.evaluate(chosen, t + 1)
        X_obs = np.vstack([X_obs, chosen])
        F_obs = np.concatenate([F_obs, f_obs])
        rows.append(tracker.row(t + 1, n_inf, time.perf_counter() - t0))
    return rows, tracker.proposals


METHODS = {
    "DALIv2": run_dali_v2_syn,
    "DALIbase": run_dali_base_syn,
    "BO": run_bo_syn,
    "TuRBO": run_turbo_syn,
    "SAASBO": run_saasbo_syn,
    "GA": run_ga_syn,
    "Random": run_random_syn,
}


# ---------------------------------------------------------------------------
# Statistics: Mann-Whitney with Holm correction (addresses weakness M1)
# ---------------------------------------------------------------------------
def holm_adjust(pvals):
    """Holm step-down adjusted p-values (family = the given list)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_all(problems, methods, seeds, trials, outdir):
    registry = SP.build_problems()
    for pname in problems:
        prob = registry[pname]
        for method in methods:
            for seed in seeds:
                tag = f"{method}_{pname}_seed{seed}"
                out_path = os.path.join(outdir, tag + ".csv")
                if os.path.exists(out_path):
                    print(f"[skip] {tag} (exists)", flush=True)
                    continue
                print(f"[run ] {tag} ...", flush=True)
                run_t0 = time.perf_counter()
                rows, proposals = METHODS[method](prob, seed, trials)
                wall = time.perf_counter() - run_t0
                df = pd.DataFrame(rows)
                df.insert(0, "method", method)
                df.insert(1, "problem", pname)
                df.insert(2, "seed", seed)
                df.to_csv(out_path, index=False)
                pd.DataFrame(proposals).to_csv(
                    os.path.join(outdir, tag + "_proposals.csv"), index=False)
                print(f"       done in {wall:.1f}s "
                      f"(best_f {rows[-1]['best_f']:.4f}, "
                      f"violations {sum(r['n_infeasible'] for r in rows)})", flush=True)


def build_summary(problems, methods, seeds, trials, outdir):
    summary = {"config": {**COMMON, **V2, "trials": trials, "seeds": seeds,
                          "protocol_notes": "oracle=analytic ground truth; bounds=known box; "
                                            "Track C death penalty on infeasible proposals"},
               "results": {}, "dali_vs_baselines": {}}
    finals = {}
    for pname in problems:
        summary["results"][pname] = {}
        for method in methods:
            fs, aucs, cyc, viol = [], [], [], []
            for seed in seeds:
                path = os.path.join(outdir, f"{method}_{pname}_seed{seed}.csv")
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path).sort_values("iteration")
                fs.append(float(df["best_f"].iloc[-1]))
                aucs.append(auc_of(df["best_value"].values))
                cyc.append(float(df["cycle_time_sec"].sum()))
                viol.append(int(df["n_infeasible"].sum()))
            if not fs:
                continue
            n = len(fs)
            finals[(pname, method)] = fs
            summary["results"][pname][method] = {
                "n_seeds": n,
                "final_best_f_mean": float(np.mean(fs)),
                "final_best_f_std": float(np.std(fs, ddof=1)) if n > 1 else 0.0,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
                "mean_cycle_time_sec": float(np.mean(cyc) / trials),
                "total_infeasible_mean": float(np.mean(viol)),
                "final_best_f_per_seed": dict(zip(map(str, seeds[:n]), fs)),
            }

        # DALIv2 vs every other available method, Holm-corrected within problem
        a = finals.get((pname, "DALIv2"))
        if not a:
            continue
        comps, raw_p = [], []
        for method in BASELINE_ORDER:
            b = finals.get((pname, method))
            if not b:
                continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                p_better = mannwhitneyu(a, b, alternative="less")[1]  # lower f = better
            except ValueError:
                continue
            comps.append((method, float(p), float(p_better),
                          float(np.mean(a)), float(np.mean(b))))
            raw_p.append(float(p))
        if comps:
            adj = holm_adjust(raw_p)
            summary["dali_vs_baselines"][pname] = {
                m: {"p_two_sided": p, "p_holm": adj[i], "p_dali_better": pb,
                    "dali_best_f_mean": am, f"{m.lower()}_best_f_mean": bm}
                for i, (m, p, pb, am, bm) in enumerate(comps)
            }

    out = os.path.join(outdir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="", choices=["", "A", "B", "C"])
    ap.add_argument("--problems", default="", help="comma list; overrides --track")
    ap.add_argument("--methods", default="DALIv2,DALIbase,BO,TuRBO,SAASBO,GA,Random")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default="")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.problems:
        problems = [p for p in args.problems.split(",") if p]
        track = "custom"
    elif args.track:
        problems = TRACKS[args.track]
        track = args.track
    else:
        raise SystemExit("Specify --track A|B|C or --problems ...")
    methods = [m for m in args.methods.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    outdir = args.outdir or f"../results/synthetic_benchmark/track_{track}"
    os.makedirs(outdir, exist_ok=True)

    if not args.summary_only:
        run_all(problems, methods, seeds, args.trials, outdir)
    build_summary(problems, methods, seeds, args.trials, outdir)


if __name__ == "__main__":
    main()
