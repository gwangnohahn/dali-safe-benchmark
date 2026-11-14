"""EXPLORATORY (not part of the approved revision benchmark).

Local-Exploitation (LE) module for DALI, Track B only.
Requested 2026-08-29: keep the diffusion model and the whole v2 pipeline
untouched; swap only (part of) the batch-SELECTION stage for a local
exploitation module. Goal: understand whether the Track B losses are fully
explained by the missing last-mile refinement, and whether a module-level
swap closes the gap.

Design (module-level, no change to CondDiffu / FCNN / training):
  * Trust region: axis-aligned box around the best feasible x observed so
    far, half-width r * (ub - lb) per dim, clipped to the known bounds.
    r starts at R0=0.10; after a cycle that improves best_f: r *= 1.6
    (cap R_MAX=0.2); otherwise r /= 1.6 (floor R_MIN=0.005).
  * Local candidates: N_LOCAL=200 uniform samples in the trust box.
  * Scoring: the SAME v2 verifier ensemble (unchanged training) ranks the
    local candidates by predicted s = -f; the top b_local are selected.
    ("The same filter, pointed at trust-region candidates instead of
    diffusion candidates.")
  * DALIv2LE  : b_local=2 local + 3 from the unchanged DALIv2
                generate-then-filter path (diffusion navigation kept).
  * LEonly    : b_local=5, diffusion never trained or sampled. Control that
                isolates predictor-guided local search without generation —
                answers "does the diffusion part add anything here?".
Everything else (warm start 200, 20 cycles x batch 5, seeds, adaptive pi,
verifier config, retrain epochs, CSV schema) is identical to run_dali_syn.

Usage:
  python run_dali_le.py --seeds 43 --trials 3 --problems gm6_mid_ctail   # smoke
  python run_dali_le.py --seeds 43,77                                    # split
Outputs -> ../results/synthetic_benchmark/track_B_LE/  (separate dir; the
approved track_B results and summaries are never touched).
"""

import argparse
import os

import numpy as np
import torch
import torch.utils.data

import run_synthetic_benchmark as RB
from run_synthetic_benchmark import (
    COMMON, V2, EvalTracker, adaptive_pi, greedy_select_v2,
    train_verifier_ensemble, verifier_predict, device,
)
from ml.model import FCNN, CondDiffu
from util.data import Dataset
import problems_synthetic as SP

N_LOCAL = 200
R0, R_MAX, R_MIN, R_FACTOR = 0.10, 0.20, 0.005, 1.6


def _local_candidates(x_best, r, lb, ub, rng):
    half = r * (ub - lb)
    lo = np.clip(x_best - half, lb, ub)
    hi = np.clip(x_best + half, lb, ub)
    return lo + rng.rand(N_LOCAL, lb.shape[0]) * (hi - lo)


def run_dali_le_syn(prob, seed, trials, b_local, use_diffusion):
    """Same skeleton as RB.run_dali_syn (v2 path); selection stage modified."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng_local = np.random.RandomState(seed + 20260829)
    n_cond, n_perf = prob.dim, 1
    clip_lb, clip_ub = prob.lb, prob.ub
    b_gen = COMMON["batch_size"] - b_local
    assert b_gen >= 0
    if not use_diffusion:
        assert b_gen == 0, "LEonly must not need generated candidates"

    X0, f0 = prob.warm_start(seed)
    tracker = EvalTracker(prob, f0)
    x_best = X0[int(np.argmin(f0))].copy()
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

    pred_model = opt_pred = loss_func = diffu_model = opt_diffu = None
    if use_diffusion:
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
    for i in range(V2["n_verifiers"]):
        torch.manual_seed(seed * 100 + i)
        m = FCNN(n_cond, n_perf, dataset_train).to(device)
        verifiers.append((m, torch.optim.AdamW(m.parameters(), lr=5e-4,
                                               weight_decay=1e-6)))
    torch.manual_seed(seed)
    train_verifier_ensemble(verifiers, dataset_train.y.to(device),
                            dataset_train.x.to(device), epochs=200)

    def verifier_score(comps_raw):
        comps_norm = (comps_raw - y_mean) / y_std
        cand_tensor = torch.tensor(comps_norm, dtype=torch.float32, device=device)
        ens_pred_norm = verifier_predict(verifiers, cand_tensor)
        ens_obj = -(ens_pred_norm * x_std + x_mean)[..., 0]  # (E, N): s = -f
        return comps_norm, ens_obj.mean(axis=0), ens_obj.std(axis=0)

    r = R0
    rows = []
    import time
    for trial in range(trials):
        t0 = time.perf_counter()
        picks = []

        # --- generative part: unchanged DALIv2 generate-then-filter, b_gen picks
        if b_gen > 0:
            perf_raw = dataset_train.x.numpy() * x_std + x_mean
            lb_f, ub_f = perf_raw.min(axis=0), perf_raw.max(axis=0)
            targets_raw = np.random.uniform(lb_f, ub_f,
                                            size=(COMMON["num_target_candidates"], n_perf))
            pi = adaptive_pi(trial, trials)
            scores_tmp = -targets_raw[:, 0]
            threshold = np.percentile(scores_tmp, pi)
            targets_raw = targets_raw[scores_tmp >= threshold]
            if targets_raw.shape[0] == 0:
                targets_raw = np.random.uniform(lb_f, ub_f, size=(10, n_perf))
            all_comps = []
            for cand_x in targets_raw:
                cand_norm = (cand_x - x_mean) / x_std
                cand_tensor = torch.tensor(cand_norm, dtype=torch.float32).reshape(1, -1)
                for _ in range(COMMON["num_comp_samples"]):
                    comp_norm = diffu_model.predict(cand_tensor).numpy()[0]
                    comp_raw = (comp_norm * y_std + y_mean)[0]
                    all_comps.append(np.clip(comp_raw, clip_lb, clip_ub))
            all_comps = np.array(all_comps)
            comps_norm, P_score, U_score = verifier_score(all_comps)
            N = comps_norm.shape[0]
            keep_n = max(b_gen, int(np.ceil(N * V2["verifier_keep_frac"])))
            keep_idx = np.argsort(-P_score)[:keep_n]
            sel_local = greedy_select_v2(
                comps_norm[keep_idx], P_score[keep_idx], U_score[keep_idx],
                b_gen, V2["w_perf"], COMMON["alpha_uncertainty"])
            picks.append(all_comps[keep_idx[sel_local]])

        # --- LE module: trust-region candidates scored by the same verifiers
        local_raw = _local_candidates(x_best, r, clip_lb, clip_ub, rng_local)
        _, P_loc, _ = verifier_score(local_raw)
        picks.append(local_raw[np.argsort(-P_loc)[:b_local]])

        X_sel = np.vstack(picks)
        best_before = tracker.best_f
        f_obs, n_inf = tracker.evaluate(X_sel, trial + 1)

        # trust-region bookkeeping (feasible best only; Track B is unconstrained)
        f_true_batch = np.asarray(prob.evaluate(X_sel), dtype=float)
        feas = (np.asarray(prob.feasible(X_sel), dtype=bool)
                if prob.constrained else np.ones(len(X_sel), dtype=bool))
        if feas.any() and f_true_batch[feas].min() < best_before:
            x_best = X_sel[feas][int(np.argmin(f_true_batch[feas]))].copy()
            r = min(r * R_FACTOR, R_MAX)
        else:
            r = max(r / R_FACTOR, R_MIN)

        for x_raw, f_o in zip(X_sel, f_obs):
            perf_norm = (np.array([[f_o]]) - x_mean) / x_std
            comp_norm = (x_raw.reshape(1, -1) - y_mean) / y_std
            dataset_train.x = torch.cat(
                [dataset_train.x, torch.tensor(perf_norm, dtype=torch.float)], dim=0)
            dataset_train.y = torch.cat(
                [dataset_train.y, torch.tensor(comp_norm, dtype=torch.float)], dim=0)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64,
                                                   shuffle=True)
        if use_diffusion:
            for _ in range(50):
                pred_model.fit(loader_train, opt_pred, loss_func)
            for _ in range(50):
                diffu_model.fit(loader_train, opt_diffu)
        train_verifier_ensemble(verifiers, dataset_train.y.to(device),
                                dataset_train.x.to(device), epochs=50)

        rows.append(tracker.row(trial + 1, n_inf, time.perf_counter() - t0))

    return rows, tracker.proposals


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=",".join(SP.TRACK_B))
    ap.add_argument("--methods", default="DALIv2LE,LEonly")
    ap.add_argument("--seeds", default=",".join(map(str, RB.DEFAULT_SEEDS)))
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default="../results/synthetic_benchmark/track_B_LE")
    args = ap.parse_args()

    RB.METHODS["DALIv2LE"] = lambda p, s, t: run_dali_le_syn(p, s, t, b_local=2,
                                                             use_diffusion=True)
    RB.METHODS["LEonly"] = lambda p, s, t: run_dali_le_syn(p, s, t, b_local=5,
                                                           use_diffusion=False)

    problems = [p for p in args.problems.split(",") if p]
    methods = [m for m in args.methods.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    os.makedirs(args.outdir, exist_ok=True)
    RB.run_all(problems, methods, seeds, args.trials, args.outdir)


if __name__ == "__main__":
    main()
