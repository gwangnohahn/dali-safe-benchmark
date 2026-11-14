"""Unified single-objective (SO) optimization experiment runner for DALI resubmission.

Runs three methods on three datasets with a common, fair protocol:
  - DALI    : diffusion-based active learning (refactor of exec_active_learning_mod_sp.py)
  - BO       : skopt Optimizer + Gaussian Process (Matern 2.5), faithful to exec_mobo_baseline_*.py
  - Random   : uniform random sampling in condition space within global bounds

Fairness guarantees (identical across methods for a given seed):
  - Same XGBoost oracle (MultiOutputRegressor, n_estimators=200, max_depth=4, oracle_seed=0),
    fitted on the full dataset, used ONLY to evaluate selected conditions.
  - Same warm-start set of 200 points (np.random.choice after np.random.seed(seed)).
  - Same condition search bounds (global min/max of each condition variable).
  - Same scalar objective_value (higher is better).

Output (DALI Resubmit/results/so_runs/):
  - per-run CSV:  <method>_<dataset>_seed<seed>.csv
        columns = method, dataset, seed, iteration, best_value, cycle_time_sec
  - summary.json: per method/dataset final best (mean+/-std), AUC (mean+/-std),
        total wall-clock, DALI-vs-BO Mann-Whitney U p-value.

Code notation note (legacy tensor names, see CLAUDE.md):
  dataset.x stores performance (physical y);  dataset.y stores conditions (physical x).
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from skopt import Optimizer
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern

from ml.model import FCNN, CondDiffu
from util.data import Dataset

# ---------------------------------------------------------------------------
# Dataset registry (indices / objective extracted from the original scripts)
# ---------------------------------------------------------------------------
DATASETS = {
    "steel_fatigue": {
        "file": "dataset/steel_fatigue.xlsx",
        "perf_idx": [13, 14, 15, 19, 20, 23, 24, 25],   # performance (y)
        "cond_idx": [16, 17, 18, 21, 22],               # conditions (x)
        "direction": [0, 0, 0, 0, 0, 0, 0, -1],          # minimize last (fatigue)
        "weights":   [0, 0, 0, 0, 0, 0, 0, 1],
    },
    "methane_conv": {
        "file": "dataset/methane_conv.xlsx",
        "perf_idx": [2, 5, 12, 14, 15],
        "cond_idx": [0, 1, 3, 4],
        "direction": [0, 0, 1, 0, 0],                    # maximize index 2
        "weights":   [0, 0, 1, 0, 0],
    },
    "power_plant": {
        "file": "dataset/power_plant.xlsx",
        "perf_idx": [4],
        "cond_idx": [0, 1, 2, 3],
        "direction": [1],                                # maximize power
        "weights":   [1],
    },
}

# Common protocol (paper settings)
COMMON = {
    "initial_train_size": 200,
    "trials": 20,
    "batch_size": 5,
    "num_target_candidates": 200,
    "num_comp_samples": 8,
    "alpha_uncertainty": 0.7,
    "candidate_score_percentile": 80,
    "knn_k": 5,
    "oracle_seed": 0,
}

DEFAULT_SEEDS = [43, 77, 777, 34, 1004, 0, 1, 2, 3, 7]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def objective_value(perf, direction, weights):
    """Scalarize a performance vector; higher is better."""
    perf = np.asarray(perf, dtype=float).flatten()
    perf = perf * np.asarray(direction, dtype=float)
    return float(np.dot(perf, np.asarray(weights, dtype=float)))


def load_problem(ds_cfg):
    """Load data, fit oracle, return arrays + oracle + bounds."""
    df = pd.read_excel(ds_cfg["file"])
    comp_all = df.iloc[:, ds_cfg["cond_idx"]].values.astype(float)   # conditions (x)
    perf_all = df.iloc[:, ds_cfg["perf_idx"]].values.astype(float)   # performance (y)

    oracle = MultiOutputRegressor(
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=4,
            random_state=COMMON["oracle_seed"],
            n_jobs=-1,
            verbosity=0,
        )
    )
    oracle.fit(comp_all, perf_all)
    return comp_all, perf_all, oracle


def warm_start_indices(n_total, size, seed):
    """Deterministic warm-start selection shared by every method for this seed."""
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size, replace=False)


def comp_uncertainty_knn(cands_norm, train_norm, k):
    k = min(k, max(1, train_norm.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nbrs.fit(train_norm)
    dists, _ = nbrs.kneighbors(cands_norm, return_distance=True)
    return dists.mean(axis=1)


def greedy_diverse_batch(cands_norm, unc, batch_size, alpha_uncertainty):
    N = cands_norm.shape[0]
    selected = []
    remaining = np.arange(N)
    U = unc.copy()
    if np.nanmax(U) > np.nanmin(U):
        U = (U - np.nanmin(U)) / (np.nanmax(U) - np.nanmin(U))
    else:
        U = np.zeros_like(U)
    D_cache = np.zeros(N, dtype=float)

    def min_dist_to_selected(idx):
        if not selected:
            return 0.0
        v = cands_norm[idx]
        ss = cands_norm[selected]
        return float(np.min(np.linalg.norm(ss - v, axis=1)))

    for _ in range(batch_size):
        if not selected:
            D_norm = np.zeros_like(U)
        else:
            for i in remaining:
                D_cache[i] = min_dist_to_selected(i)
            D = D_cache[remaining]
            if np.max(D) > np.min(D):
                D_norm_vec = (D - np.min(D)) / (np.max(D) - np.min(D))
            else:
                D_norm_vec = np.zeros_like(D)
            D_norm = np.zeros_like(U)
            D_norm[remaining] = D_norm_vec
        score = alpha_uncertainty * U + (1.0 - alpha_uncertainty) * D_norm
        best_idx_global = remaining[np.argmax(score[remaining])]
        selected.append(best_idx_global)
        remaining = remaining[remaining != best_idx_global]
        if len(remaining) == 0:
            break
    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# Method: DALI
# ---------------------------------------------------------------------------
def run_dali(ds_name, ds_cfg, seed, comp_all, perf_all, oracle, trials):
    np.random.seed(seed)
    torch.manual_seed(seed)

    direction, weights = ds_cfg["direction"], ds_cfg["weights"]
    n_perf = len(ds_cfg["perf_idx"])

    # clip bounds (0% margin, as in original)
    clip_lb = comp_all.min(axis=0)
    clip_ub = comp_all.max(axis=0)

    # warm start
    idx_train = warm_start_indices(comp_all.shape[0], COMMON["initial_train_size"], seed)
    perf_train = perf_all[idx_train]
    comp_train = comp_all[idx_train]

    x_mean = perf_train.mean(axis=0, keepdims=True)
    x_std = perf_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = comp_train.mean(axis=0, keepdims=True)
    y_std = comp_train.std(axis=0, keepdims=True) + 1e-8

    perf_train_norm = (perf_train - x_mean) / x_std
    comp_train_norm = (comp_train - y_mean) / y_std

    dataset_train = Dataset(
        torch.tensor(perf_train_norm, dtype=torch.float),
        torch.tensor(comp_train_norm, dtype=torch.float),
        x_mean, x_std, y_mean, y_std,
    )
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    var_stat = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    setup_t0 = time.perf_counter()
    pred_model = FCNN(dataset_train.x.shape[1], dataset_train.y.shape[1], dataset_train).to(device)
    opt_pred = torch.optim.AdamW(pred_model.parameters(), lr=5e-4, weight_decay=1e-6)
    loss_func = torch.nn.MSELoss()
    for _ in range(200):
        pred_model.fit(loader_train, opt_pred, loss_func)

    diffu_model = CondDiffu(
        dataset_train.x.shape[1], dataset_train.y.shape[1],
        num_steps=200, pred_model=pred_model, var_stat=var_stat,
    ).to(device)
    opt_diffu = torch.optim.AdamW(diffu_model.parameters(), lr=5e-4, weight_decay=5e-6)
    for _ in range(200):
        diffu_model.fit(loader_train, opt_diffu)
    setup_time = time.perf_counter() - setup_t0

    # running best initialized from warm start
    best = max(objective_value(p, direction, weights) for p in perf_train)
    rows = []

    for trial in range(trials):
        t0 = time.perf_counter()

        perf_train_raw = dataset_train.x.numpy() * x_std + x_mean
        lb = perf_train_raw.min(axis=0)
        ub = perf_train_raw.max(axis=0)
        targets_raw = np.random.uniform(lb, ub, size=(COMMON["num_target_candidates"], n_perf))

        scores_tmp = np.array([objective_value(c, direction, weights) for c in targets_raw])
        threshold = np.percentile(scores_tmp, COMMON["candidate_score_percentile"])
        targets_raw = targets_raw[scores_tmp >= threshold]

        all_comps = []
        for cand_x in targets_raw:
            cand_norm = (cand_x - x_mean) / x_std
            cand_tensor = torch.tensor(cand_norm, dtype=torch.float32).reshape(1, -1)
            for _ in range(COMMON["num_comp_samples"]):
                comp_norm = diffu_model.predict(cand_tensor).numpy()[0]
                comp_raw = (comp_norm * y_std + y_mean)[0]
                comp_raw = np.clip(comp_raw, clip_lb, clip_ub)
                all_comps.append(comp_raw)
        all_comps = np.array(all_comps)

        train_comps_raw = dataset_train.y.numpy() * y_std + y_mean
        train_comps_raw = train_comps_raw.reshape(-1, train_comps_raw.shape[-1])
        train_comps_norm = (train_comps_raw - y_mean) / y_std
        comps_norm = (all_comps - y_mean) / y_std
        U = comp_uncertainty_knn(comps_norm, train_comps_norm, k=COMMON["knn_k"])

        sel_idx = greedy_diverse_batch(
            comps_norm, U, COMMON["batch_size"], COMMON["alpha_uncertainty"]
        )

        for idx_local in sel_idx:
            comp_raw = all_comps[idx_local]
            pred_perf = oracle.predict(comp_raw.reshape(1, -1))[0]
            obj = objective_value(pred_perf, direction, weights)
            best = max(best, obj)
            # append measured point to training set
            perf_norm = (pred_perf - x_mean) / x_std
            comp_norm = (comp_raw - y_mean) / y_std
            dataset_train.x = torch.cat([dataset_train.x, torch.tensor(perf_norm, dtype=torch.float)], dim=0)
            dataset_train.y = torch.cat([dataset_train.y, torch.tensor(comp_norm, dtype=torch.float)], dim=0)

        # retrain
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
        for _ in range(50):
            pred_model.fit(loader_train, opt_pred, loss_func)
        for _ in range(50):
            diffu_model.fit(loader_train, opt_diffu)

        cycle_time = time.perf_counter() - t0
        rows.append({"iteration": trial + 1, "best_value": best, "cycle_time_sec": cycle_time})

    return rows, setup_time


# ---------------------------------------------------------------------------
# Method: BO (skopt + GP), faithful to exec_mobo_baseline_*.py
# ---------------------------------------------------------------------------
def run_bo(ds_name, ds_cfg, seed, comp_all, perf_all, oracle, trials):
    direction, weights = ds_cfg["direction"], ds_cfg["weights"]
    np.random.seed(seed)

    idx_init = warm_start_indices(comp_all.shape[0], COMMON["initial_train_size"], seed)
    init_comps_raw = comp_all[idx_init]
    init_perfs_raw = perf_all[idx_init]

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x.fit(init_comps_raw)
    global_lb_norm = scaler_x.transform([comp_all.min(axis=0)])[0]
    global_ub_norm = scaler_x.transform([comp_all.max(axis=0)])[0]
    space = [Real(global_lb_norm[i], global_ub_norm[i], name=f"n_comp_{i}")
             for i in range(len(ds_cfg["cond_idx"]))]

    setup_t0 = time.perf_counter()
    gpr_model = GaussianProcessRegressor(
        kernel=Matern(nu=2.5), normalize_y=True, alpha=1e-10,
        n_restarts_optimizer=5, random_state=seed,
    )
    opt = Optimizer(
        space, base_estimator=gpr_model, random_state=seed,
        n_initial_points=0, initial_point_generator="random",
    )
    init_comps_norm = scaler_x.transform(init_comps_raw)
    init_losses = [-objective_value(p, direction, weights) for p in init_perfs_raw]
    opt.tell(init_comps_norm.tolist(), init_losses)
    setup_time = time.perf_counter() - setup_t0

    best = max(objective_value(p, direction, weights) for p in init_perfs_raw)
    rows = []

    for t in range(trials):
        t0 = time.perf_counter()
        cands_norm = opt.ask(n_points=COMMON["batch_size"])
        cands_raw = scaler_x.inverse_transform(cands_norm)
        perf_preds = oracle.predict(cands_raw)
        losses = []
        for perf_pred in perf_preds:
            score = objective_value(perf_pred, direction, weights)
            losses.append(-score)
            best = max(best, score)
        opt.tell(cands_norm, losses)
        cycle_time = time.perf_counter() - t0
        rows.append({"iteration": t + 1, "best_value": best, "cycle_time_sec": cycle_time})

    return rows, setup_time


# ---------------------------------------------------------------------------
# Method: Random sampling in condition space (same bounds/oracle as BO)
# ---------------------------------------------------------------------------
def run_random(ds_name, ds_cfg, seed, comp_all, perf_all, oracle, trials):
    direction, weights = ds_cfg["direction"], ds_cfg["weights"]
    rng = np.random.RandomState(seed)

    idx_init = warm_start_indices(comp_all.shape[0], COMMON["initial_train_size"], seed)
    init_perfs_raw = perf_all[idx_init]

    lb = comp_all.min(axis=0)
    ub = comp_all.max(axis=0)
    n_cond = len(ds_cfg["cond_idx"])

    best = max(objective_value(p, direction, weights) for p in init_perfs_raw)
    rows = []
    for t in range(trials):
        t0 = time.perf_counter()
        cands_raw = rng.uniform(lb, ub, size=(COMMON["batch_size"], n_cond))
        perf_preds = oracle.predict(cands_raw)
        for perf_pred in perf_preds:
            best = max(best, objective_value(perf_pred, direction, weights))
        cycle_time = time.perf_counter() - t0
        rows.append({"iteration": t + 1, "best_value": best, "cycle_time_sec": cycle_time})

    return rows, 0.0


METHODS = {"DALI": run_dali, "BO": run_bo, "Random": run_random}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def auc_of(best_values):
    """Area under the best-so-far curve over iterations (dx=1)."""
    return float(np.trapezoid(best_values, dx=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="DALI,BO,Random")
    ap.add_argument("--datasets", default="steel_fatigue,methane_conv,power_plant")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default="../results/so_runs")
    ap.add_argument("--summary-only", action="store_true",
                    help="Recompute summary.json from existing per-run CSVs only.")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    datasets = [d for d in args.datasets.split(",") if d]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    trials = args.trials
    os.makedirs(args.outdir, exist_ok=True)

    meta = {"setup_time": {}}  # keyed "method|dataset|seed"

    if not args.summary_only:
        # cache loaded problems / oracles per dataset (oracle is seed-independent)
        problems = {}
        for ds_name in datasets:
            print(f"[load] {ds_name} ...", flush=True)
            problems[ds_name] = load_problem(DATASETS[ds_name])

        for ds_name in datasets:
            comp_all, perf_all, oracle = problems[ds_name]
            for method in methods:
                for seed in seeds:
                    tag = f"{method}_{ds_name}_seed{seed}"
                    print(f"[run ] {tag} ...", flush=True)
                    run_t0 = time.perf_counter()
                    rows, setup_time = METHODS[method](
                        ds_name, DATASETS[ds_name], seed,
                        comp_all, perf_all, oracle, trials,
                    )
                    wall = time.perf_counter() - run_t0
                    df = pd.DataFrame(rows)
                    df.insert(0, "method", method)
                    df.insert(1, "dataset", ds_name)
                    df.insert(2, "seed", seed)
                    df = df[["method", "dataset", "seed", "iteration", "best_value", "cycle_time_sec"]]
                    df.to_csv(os.path.join(args.outdir, tag + ".csv"), index=False)
                    meta["setup_time"][f"{method}|{ds_name}|{seed}"] = setup_time
                    save_setup_times(args.outdir, meta["setup_time"])
                    print(f"       done in {wall:.1f}s (setup {setup_time:.1f}s, "
                          f"final best {rows[-1]['best_value']:.4f})", flush=True)

    meta["setup_time"] = load_setup_times(args.outdir)
    build_summary(args.outdir, methods, datasets, seeds, trials, meta)


def save_setup_times(outdir, setup_times):
    """One file per process so parallel runs never clobber each other."""
    path = os.path.join(outdir, f"setup_times_{os.getpid()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(setup_times, f, indent=2)


def load_setup_times(outdir):
    import glob
    merged = {}
    for path in sorted(glob.glob(os.path.join(outdir, "setup_times_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            merged.update(json.load(f))
    return merged


def build_summary(outdir, methods, datasets, seeds, trials, meta):
    # gather per-run trajectories from CSVs
    summary = {"config": {**COMMON, "trials": trials, "seeds": seeds}, "results": {}}
    per_method_dataset_finalbest = {}  # (method,dataset)->list final best per seed

    total_wall_by_method = {m: 0.0 for m in methods}
    grand_total_wall = 0.0

    for ds_name in datasets:
        summary["results"][ds_name] = {}
        for method in methods:
            finals, aucs, cyc_times, setups = [], [], [], []
            for seed in seeds:
                path = os.path.join(outdir, f"{method}_{ds_name}_seed{seed}.csv")
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path).sort_values("iteration")
                finals.append(float(df["best_value"].iloc[-1]))
                aucs.append(auc_of(df["best_value"].values))
                cyc = df["cycle_time_sec"].sum()
                cyc_times.append(float(cyc))
                stime = meta.get("setup_time", {}).get(f"{method}|{ds_name}|{seed}", 0.0)
                setups.append(float(stime))
            if not finals:
                continue
            wall = float(np.sum(cyc_times) + np.sum(setups))
            total_wall_by_method[method] += wall
            entry = {
                "n_seeds": len(finals),
                "final_best_mean": float(np.mean(finals)),
                "final_best_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "total_wall_clock_sec": wall,
                "mean_cycle_time_sec": float(np.mean(cyc_times) / trials),
                "mean_setup_time_sec": float(np.mean(setups)),
                "final_best_per_seed": dict(zip(map(str, seeds[:len(finals)]), finals)),
            }
            summary["results"][ds_name][method] = entry
            per_method_dataset_finalbest[(method, ds_name)] = finals

    # DALI vs BO Mann-Whitney U per dataset (on final best across seeds)
    summary["dali_vs_bo_mannwhitney"] = {}
    for ds_name in datasets:
        a = per_method_dataset_finalbest.get(("DALI", ds_name))
        b = per_method_dataset_finalbest.get(("BO", ds_name))
        if a and b and len(a) > 0 and len(b) > 0:
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                p_greater = mannwhitneyu(a, b, alternative="greater")[1]
                summary["dali_vs_bo_mannwhitney"][ds_name] = {
                    "U": float(stat), "p_two_sided": float(p),
                    "p_dali_greater": float(p_greater),
                    "dali_final_mean": float(np.mean(a)), "bo_final_mean": float(np.mean(b)),
                }
            except ValueError as e:
                summary["dali_vs_bo_mannwhitney"][ds_name] = {"error": str(e)}

    summary["total_wall_clock_by_method_sec"] = total_wall_by_method
    summary["grand_total_wall_clock_sec"] = float(sum(total_wall_by_method.values()))

    out = os.path.join(outdir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
