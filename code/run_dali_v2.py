"""DALI v2: Verifier-ensemble filter + Adaptive percentile schedule.

This is a NEW file. It does NOT modify or read v1 code paths destructively; v1
(run_so_experiments.run_dali / exec_active_learning_mod_sp.py) and its results in
results/so_runs are preserved and reused as the comparison baseline.

Two improvements over v1 (both ON together for this quick check):

  Module 1 - Verifier ensemble filter
    Before the condition-space (comp./cond.) selection, an ensemble of 5 FCNNs is
    trained on the accumulated data in the FORWARD direction (condition x -> performance y),
    i.e. the same direction as the oracle but a DELIBERATELY DIFFERENT model family.
    The oracle here is XGBoost; the verifier is a neural net (FCNN). Using a different
    family avoids the "model-match" bias that would inflate scores if the verifier and
    oracle shared an inductive bias (cf. the GA-surrogate baseline, which shares XGBoost
    with the oracle and benefits unfairly).
      - ensemble MEAN prediction -> performance verification score: candidates whose
        predicted objective is low (won't reach the target performance) are filtered out
        with a hard cut before selection.
      - ensemble VARIANCE -> principled uncertainty estimate, REPLACING the v1 kNN
        distance heuristic (reviewer point 5).

  Module 2 - Adaptive percentile pi
    Instead of a fixed pi=80, pi increases linearly across cycles (pi=60 -> pi=90),
    shifting from exploration (loose target filter early) to exploitation (tight late).
    Addresses the reviewer remark that a fixed percentile is weaker than a BO acquisition.

Final selection score (greedy, batch B):
    score = w_perf * P_norm + (1 - w_perf) * (alpha * U_norm + (1 - alpha) * D_norm)
  with w_perf=0.5, alpha=0.7 (v1 alpha reused), P=verifier perf score, U=ensemble std,
  D=diversity (greedy max-min in condition space).

Output: results/v2_quickcheck/ (or --outdir), same CSV format as run_so_experiments
(method label "DALIv2"). Columns: method, dataset, seed, iteration, best_value, cycle_time_sec.
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

from ml.model import FCNN, CondDiffu
from util.data import Dataset

from run_so_experiments import (
    DATASETS, COMMON, DEFAULT_SEEDS, objective_value, load_problem,
    warm_start_indices, auc_of, greedy_diverse_batch, comp_uncertainty_knn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V2 = {
    "n_verifiers": 5,        # FCNN ensemble size
    "verifier_keep_frac": 0.5,   # hard-filter: keep top this fraction by verifier perf score
    "w_perf": 0.5,           # weight of verifier perf score vs (uncertainty+diversity)
    "pi_start": 60.0,        # adaptive percentile schedule
    "pi_end": 90.0,
}


def adaptive_pi(trial, trials):
    if trials <= 1:
        return V2["pi_end"]
    return V2["pi_start"] + (V2["pi_end"] - V2["pi_start"]) * trial / (trials - 1)


def greedy_select_v2(cands_norm, P, U, batch_size, w_perf, alpha):
    """Greedy max-min selection combining perf score P, uncertainty U, diversity D."""
    def norm01(v):
        v = np.asarray(v, dtype=float)
        if np.nanmax(v) > np.nanmin(v):
            return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
        return np.zeros_like(v)

    Pn = norm01(P)
    Un = norm01(U)
    N = cands_norm.shape[0]
    selected = []
    remaining = np.arange(N)
    D_cache = np.zeros(N, dtype=float)

    def min_dist_to_selected(idx):
        if not selected:
            return 0.0
        d = np.linalg.norm(cands_norm[selected] - cands_norm[idx], axis=1)
        return float(np.min(d))

    static = w_perf * Pn + (1.0 - w_perf) * alpha * Un  # perf + uncertainty part
    for _ in range(min(batch_size, N)):
        if not selected:
            D_norm = np.zeros(N, dtype=float)
        else:
            for i in remaining:
                D_cache[i] = min_dist_to_selected(i)
            D = D_cache[remaining]
            D_norm = np.zeros(N, dtype=float)
            if np.max(D) > np.min(D):
                D_norm[remaining] = (D - np.min(D)) / (np.max(D) - np.min(D))
        score = static + (1.0 - w_perf) * (1.0 - alpha) * D_norm
        best = remaining[np.argmax(score[remaining])]
        selected.append(best)
        remaining = remaining[remaining != best]
        if len(remaining) == 0:
            break
    return np.array(selected, dtype=int)


def train_verifier_ensemble(models_opts, X_cond, Y_perf, epochs):
    """Train/continue the forward-direction FCNN ensemble (condition -> performance)."""
    loss_func = torch.nn.MSELoss()
    for model, opt in models_opts:
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_func(model(X_cond), Y_perf)
            loss.backward()
            opt.step()


def verifier_predict(models_opts, X_cond):
    """Return per-model normalized performance predictions, shape (n_models, N, perf_dim)."""
    preds = []
    for model, _ in models_opts:
        model.eval()
        with torch.no_grad():
            preds.append(model(X_cond).cpu().numpy())
    return np.stack(preds, axis=0)


def run_dali_v2(ds_name, ds_cfg, seed, comp_all, perf_all, oracle, trials,
                use_verifier=True, use_adaptive_pi=True):
    """DALI v2 run. With use_verifier=use_adaptive_pi=False this reduces exactly to v1
    (kNN uncertainty + greedy diversity, fixed pi=80) for the ablation 'both off' arm."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    direction, weights = ds_cfg["direction"], ds_cfg["weights"]
    n_perf = len(ds_cfg["perf_idx"])
    n_cond = len(ds_cfg["cond_idx"])

    clip_lb = comp_all.min(axis=0)
    clip_ub = comp_all.max(axis=0)

    idx_train = warm_start_indices(comp_all.shape[0], COMMON["initial_train_size"], seed)
    perf_train = perf_all[idx_train]
    comp_train = comp_all[idx_train]

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

    # --- diffusion guidance predictor (perf -> cond), as in v1 ---
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

    # --- Module 1: forward verifier ensemble (cond -> perf), DIFFERENT family from oracle ---
    def cond_perf_tensors():
        Xc = dataset_train.y.to(device)            # normalized conditions (input)
        Yp = dataset_train.x.to(device)            # normalized performance (target)
        return Xc, Yp

    verifiers = []
    if use_verifier:
        for i in range(V2["n_verifiers"]):
            torch.manual_seed(seed * 100 + i)
            m = FCNN(n_cond, n_perf, dataset_train).to(device)
            verifiers.append((m, torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-6)))
        torch.manual_seed(seed)
        Xc, Yp = cond_perf_tensors()
        train_verifier_ensemble(verifiers, Xc, Yp, epochs=200)

    best = max(objective_value(p, direction, weights) for p in perf_train)
    rows = []

    for trial in range(trials):
        t0 = time.perf_counter()

        # 1) target sampling from current performance range
        perf_train_raw = dataset_train.x.numpy() * x_std + x_mean
        lb = perf_train_raw.min(axis=0)
        ub = perf_train_raw.max(axis=0)
        targets_raw = np.random.uniform(lb, ub, size=(COMMON["num_target_candidates"], n_perf))

        # 2) Module 2: adaptive percentile cut on target performance (or fixed pi=80)
        pi = adaptive_pi(trial, trials) if use_adaptive_pi else COMMON["candidate_score_percentile"]
        scores_tmp = np.array([objective_value(c, direction, weights) for c in targets_raw])
        threshold = np.percentile(scores_tmp, pi)
        targets_raw = targets_raw[scores_tmp >= threshold]
        if targets_raw.shape[0] == 0:
            targets_raw = np.random.uniform(lb, ub, size=(10, n_perf))

        # 3) diffusion generates conditions, clipped to physical bounds
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
        comps_norm = (all_comps - y_mean) / y_std

        if use_verifier:
            # 4) Module 1: verifier ensemble -> perf score (mean) + uncertainty (std)
            cand_cond_tensor = torch.tensor(comps_norm, dtype=torch.float32, device=device)
            ens_pred_norm = verifier_predict(verifiers, cand_cond_tensor)   # (E, N, perf_dim)
            # denormalize performance predictions (perf normalized with x_mean / x_std)
            ens_pred_raw = ens_pred_norm * x_std + x_mean                   # (E, N, perf_dim)
            dir_arr = np.asarray(direction, dtype=float)
            w_arr = np.asarray(weights, dtype=float)
            ens_obj = (ens_pred_raw * dir_arr) @ w_arr                      # (E, N)
            P_score = ens_obj.mean(axis=0)                                  # (N,)
            U_score = ens_obj.std(axis=0)                                   # (N,)

            # hard filter: keep top fraction by verifier perf score
            N = comps_norm.shape[0]
            keep_n = max(COMMON["batch_size"], int(np.ceil(N * V2["verifier_keep_frac"])))
            keep_idx = np.argsort(-P_score)[:keep_n]
            sel_local = greedy_select_v2(
                comps_norm[keep_idx], P_score[keep_idx], U_score[keep_idx],
                COMMON["batch_size"], V2["w_perf"], COMMON["alpha_uncertainty"],
            )
            sel_idx = keep_idx[sel_local]
        else:
            # v1 path: kNN uncertainty + greedy diversity (no perf verification)
            train_comps_raw = (dataset_train.y.numpy() * y_std + y_mean).reshape(-1, comp_all.shape[1])
            train_comps_norm = (train_comps_raw - y_mean) / y_std
            U = comp_uncertainty_knn(comps_norm, train_comps_norm, k=COMMON["knn_k"])
            sel_idx = greedy_diverse_batch(
                comps_norm, U, COMMON["batch_size"], COMMON["alpha_uncertainty"])

        # 5) oracle evaluates the selected batch only
        for idx_local in sel_idx:
            comp_raw = all_comps[idx_local]
            pred_perf = oracle.predict(comp_raw.reshape(1, -1))[0]
            best = max(best, objective_value(pred_perf, direction, weights))
            perf_norm = (pred_perf - x_mean) / x_std
            comp_norm = (comp_raw - y_mean) / y_std
            dataset_train.x = torch.cat(
                [dataset_train.x, torch.tensor(perf_norm, dtype=torch.float)], dim=0)
            dataset_train.y = torch.cat(
                [dataset_train.y, torch.tensor(comp_norm, dtype=torch.float)], dim=0)

        # 6) retrain all models
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
        for _ in range(50):
            pred_model.fit(loader_train, opt_pred, loss_func)
        for _ in range(50):
            diffu_model.fit(loader_train, opt_diffu)
        if use_verifier:
            Xc, Yp = cond_perf_tensors()
            train_verifier_ensemble(verifiers, Xc, Yp, epochs=50)

        cycle_time = time.perf_counter() - t0
        rows.append({"iteration": trial + 1, "best_value": best, "cycle_time_sec": cycle_time})

    return rows


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="methane_conv,power_plant")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS[:5])))
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--n-inits", default="",
                    help="If set (e.g. '20,50'), run warm-start comparison mode "
                         "with n-tagged files and compare vs warmstart_sweep v1.")
    ap.add_argument("--outdir", default="../results/v2_quickcheck")
    ap.add_argument("--v1-dir", default="../results/so_runs")
    ap.add_argument("--v1-warmstart-dir", default="../results/warmstart_sweep")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    n_inits = [int(n) for n in args.n_inits.split(",") if n != ""]
    trials = args.trials
    os.makedirs(args.outdir, exist_ok=True)

    # ---- warm-start comparison mode (N_init varied) ----
    if n_inits:
        if not args.summary_only:
            problems = {ds: load_problem(DATASETS[ds]) for ds in datasets}
            for ds_name in datasets:
                comp_all, perf_all, oracle = problems[ds_name]
                for n_init in n_inits:
                    for seed in seeds:
                        tag = f"DALIv2_{ds_name}_n{n_init}_seed{seed}"
                        out_path = os.path.join(args.outdir, tag + ".csv")
                        if os.path.exists(out_path):
                            print(f"[skip] {tag} (exists)", flush=True)
                            continue
                        print(f"[run ] {tag} ...", flush=True)
                        COMMON["initial_train_size"] = n_init
                        run_t0 = time.perf_counter()
                        try:
                            rows = run_dali_v2(ds_name, DATASETS[ds_name], seed,
                                               comp_all, perf_all, oracle, trials)
                        finally:
                            COMMON["initial_train_size"] = 200
                        wall = time.perf_counter() - run_t0
                        df = pd.DataFrame(rows)
                        df.insert(0, "method", "DALIv2")
                        df.insert(1, "dataset", ds_name)
                        df.insert(2, "seed", seed)
                        df.insert(3, "n_init", n_init)
                        df = df[["method", "dataset", "seed", "n_init", "iteration",
                                 "best_value", "cycle_time_sec"]]
                        df.to_csv(out_path, index=False)
                        print(f"       done in {wall:.1f}s "
                              f"(final best {rows[-1]['best_value']:.4f})", flush=True)
        build_summary_warmstart(args.outdir, args.v1_warmstart_dir,
                                datasets, seeds, n_inits, trials)
        return

    # ---- original warm-start-200 quick check ----
    if not args.summary_only:
        problems = {ds: load_problem(DATASETS[ds]) for ds in datasets}
        for ds_name in datasets:
            comp_all, perf_all, oracle = problems[ds_name]
            for seed in seeds:
                tag = f"DALIv2_{ds_name}_seed{seed}"
                out_path = os.path.join(args.outdir, tag + ".csv")
                if os.path.exists(out_path):
                    print(f"[skip] {tag} (exists)", flush=True)
                    continue
                print(f"[run ] {tag} ...", flush=True)
                run_t0 = time.perf_counter()
                rows = run_dali_v2(ds_name, DATASETS[ds_name], seed,
                                   comp_all, perf_all, oracle, trials)
                wall = time.perf_counter() - run_t0
                df = pd.DataFrame(rows)
                df.insert(0, "method", "DALIv2")
                df.insert(1, "dataset", ds_name)
                df.insert(2, "seed", seed)
                df = df[["method", "dataset", "seed", "iteration", "best_value", "cycle_time_sec"]]
                df.to_csv(out_path, index=False)
                print(f"       done in {wall:.1f}s (final best {rows[-1]['best_value']:.4f})",
                      flush=True)

    build_summary(args.outdir, args.v1_dir, datasets, seeds, trials)


def build_summary_warmstart(outdir, v1_ws_dir, datasets, seeds, n_inits, trials):
    """Compare v2 (n-tagged) against v1 results in warmstart_sweep, per N_init."""
    summary = {"config": {**COMMON, **V2, "trials": trials, "seeds": seeds,
                          "n_inits": n_inits},
               "results": {}, "v2_vs_v1_mannwhitney": {}}
    for ds_name in datasets:
        summary["results"][ds_name] = {}
        summary["v2_vs_v1_mannwhitney"][ds_name] = {}
        for n_init in n_inits:
            key = f"n_init={n_init}"
            summary["results"][ds_name][key] = {}
            per = {}
            sources = [
                ("DALIv2", outdir, f"DALIv2_{ds_name}_n{n_init}_seed{{s}}.csv"),
                ("DALI_v1", v1_ws_dir, f"DALI_{ds_name}_n{n_init}_seed{{s}}.csv"),
            ]
            for label, base, pat in sources:
                finals, aucs, traj = [], [], []
                for seed in seeds:
                    path = os.path.join(base, pat.format(s=seed))
                    if not os.path.exists(path):
                        continue
                    df = pd.read_csv(path).sort_values("iteration")
                    finals.append(float(df["best_value"].iloc[-1]))
                    aucs.append(auc_of(df["best_value"].values))
                    traj.append(df["best_value"].values.tolist())
                if not finals:
                    continue
                n = len(finals)
                per[label] = finals
                # mean convergence curve
                L = min(len(t) for t in traj)
                mean_curve = np.mean([t[:L] for t in traj], axis=0).tolist()
                summary["results"][ds_name][key][label] = {
                    "n_seeds": n,
                    "final_best_mean": float(np.mean(finals)),
                    "final_best_std": float(np.std(finals, ddof=1)) if n > 1 else 0.0,
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
                    "mean_convergence_curve": mean_curve,
                    "final_best_per_seed": dict(zip(map(str, seeds[:n]), finals)),
                }
            a, b = per.get("DALIv2"), per.get("DALI_v1")
            if a and b:
                try:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                    p_g = mannwhitneyu(a, b, alternative="greater")[1]
                    summary["v2_vs_v1_mannwhitney"][ds_name][key] = {
                        "U": float(stat), "p_two_sided": float(p),
                        "p_v2_greater": float(p_g),
                        "v2_mean": float(np.mean(a)), "v1_mean": float(np.mean(b)),
                        "v2_minus_v1": float(np.mean(a) - np.mean(b)),
                    }
                except ValueError as e:
                    summary["v2_vs_v1_mannwhitney"][ds_name][key] = {"error": str(e)}

    out = os.path.join(outdir, "summary_warmstart.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


def build_summary(outdir, v1_dir, datasets, seeds, trials):
    summary = {"config": {**COMMON, **V2, "trials": trials, "seeds": seeds},
               "results": {}, "v2_vs_v1_mannwhitney": {}}
    for ds_name in datasets:
        summary["results"][ds_name] = {}
        per = {}
        for label, base in (("DALIv2", outdir), ("DALI_v1", v1_dir)):
            finals, aucs, cyc = [], [], []
            for seed in seeds:
                fn = (f"DALIv2_{ds_name}_seed{seed}.csv" if label == "DALIv2"
                      else f"DALI_{ds_name}_seed{seed}.csv")
                path = os.path.join(base, fn)
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path).sort_values("iteration")
                finals.append(float(df["best_value"].iloc[-1]))
                aucs.append(auc_of(df["best_value"].values))
                cyc.append(float(df["cycle_time_sec"].sum()))
            if not finals:
                continue
            n = len(finals)
            per[label] = finals
            summary["results"][ds_name][label] = {
                "n_seeds": n,
                "final_best_mean": float(np.mean(finals)),
                "final_best_std": float(np.std(finals, ddof=1)) if n > 1 else 0.0,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
                "mean_cycle_time_sec": float(np.mean(cyc) / trials),
                "final_best_per_seed": dict(zip(map(str, seeds[:n]), finals)),
            }
        a, b = per.get("DALIv2"), per.get("DALI_v1")
        if a and b:
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                p_g = mannwhitneyu(a, b, alternative="greater")[1]
                summary["v2_vs_v1_mannwhitney"][ds_name] = {
                    "U": float(stat), "p_two_sided": float(p), "p_v2_greater": float(p_g),
                    "v2_mean": float(np.mean(a)), "v1_mean": float(np.mean(b)),
                    "v2_minus_v1": float(np.mean(a) - np.mean(b)),
                }
            except ValueError as e:
                summary["v2_vs_v1_mannwhitney"][ds_name] = {"error": str(e)}

    out = os.path.join(outdir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
