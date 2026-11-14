"""EXPLORATORY (not pre-registered): DALI+LE with an in-distribution GATE.

Motivation (paper 2, Fig 5d): the kNN distance of a proposal to the record
predicts violations (Spearman 0.70), and the one failure of DALI (G2 x
boundary optimum) comes from generated candidates that leave the record in a
biased direction.  The gate turns that diagnostic into a control:

  * reference set R_t = recorded conditions whose observed outcome is at least
    as good as the worst warm-start outcome (i.e. the penalised failures are
    left out -- a performance criterion, not a feasibility label; the loop
    still never sees a constraint model or a failure label);
  * every candidate (diffusion pool AND LE trust-region pool) is compared with
    the LOCAL density of the record: let d_c = mean kNN5 distance of the
    candidate to R_t and d_j = leave-one-out kNN5 distance of the candidate's
    nearest record point j; the candidate is kept iff d_c <= GATE_RATIO * d_j
    (a local-outlier-factor style test: "no farther from the record than the
    record points there are from each other").  A single global quantile of
    the record's LOO distances does NOT work here: the 85 % uniform part of
    the warm start makes it ~2 (z-scored units) while the failures at the G2
    boundary sit at 0.27-0.9 and the feasible proposals at 0.17-0.32 (checked
    on seed 43).  The reference set grows every cycle -> the admissible region
    expands only where experiments have succeeded;
  * if fewer than b_gen diffusion candidates pass, the missing slots are
    filled from the (gated) LE trust-region pool instead -- the generator is
    never forced to contribute out-of-distribution points; the LE pool keeps a
    closest-candidates fallback so the batch size never changes.

Everything else is byte-identical to run_dali_le.py (this file is that file
with the gate inserted at the two candidate pools).  Method names in the suite
runner: DALIv2LEgate (b_local=2, diffusion on), LEonlygate (b_local=5).
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


GATE_RATIO = 1.0   # candidate kNN5 distance allowed relative to the local record spacing
GATE_K = 5


def _knn_mean(A, B, k, loo=False):
    """mean distance from each row of A to its k nearest rows of B (z-scored inputs)."""
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    if loo:
        np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, :k].mean(axis=1)


def knn_gate(cands_raw, ref_raw, y_mean, y_std, n_min, ratio=GATE_RATIO, k=GATE_K, fallback=True):
    """Keep candidates that are no farther from the record than the record is
    from itself locally.  Returns (kept candidates, median local tau, fraction kept)."""
    C = (cands_raw - y_mean) / y_std
    R = (ref_raw - y_mean) / y_std
    loo = _knn_mean(R, R, k, loo=True)
    D = np.sqrt(((C[:, None, :] - R[None, :, :]) ** 2).sum(axis=2))
    d = np.sort(D, axis=1)[:, :k].mean(axis=1)
    tau_c = ratio * loo[np.argmin(D, axis=1)]
    tau = float(np.median(tau_c))
    ok = d <= tau_c
    if fallback and ok.sum() < n_min:          # LE pool: never starve the batch
        ok = np.zeros(len(d), dtype=bool)
        ok[np.argsort(d)[:n_min]] = True
    return cands_raw[ok], tau, float(ok.mean())


def run_dali_gate_syn(prob, seed, trials, b_local, use_diffusion):
    """run_dali_le_syn + kNN gate on both candidate pools (see module docstring)."""
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
    f_ref_max = float(np.max(f0))          # reference set: outcomes <= worst warm-start outcome
    rec_X, rec_f = [X0.copy()], [np.asarray(f0, float).copy()]
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
            ref = np.vstack(rec_X)[np.concatenate(rec_f) <= f_ref_max]
            all_comps, tau, frac_gen = knn_gate(all_comps, ref, y_mean, y_std, n_min=b_gen, fallback=False)
            b_gen_eff = min(b_gen, len(all_comps))
            if b_gen_eff > 0:
                comps_norm, P_score, U_score = verifier_score(all_comps)
                N = comps_norm.shape[0]
                keep_n = max(b_gen_eff, int(np.ceil(N * V2["verifier_keep_frac"])))
                keep_idx = np.argsort(-P_score)[:keep_n]
                sel_local = greedy_select_v2(
                    comps_norm[keep_idx], P_score[keep_idx], U_score[keep_idx],
                    b_gen_eff, V2["w_perf"], COMMON["alpha_uncertainty"])
                picks.append(all_comps[keep_idx[sel_local]])
        else:
            b_gen_eff = 0
        b_local_eff = COMMON["batch_size"] - b_gen_eff   # LE fills whatever the gate removed

        # --- LE module: trust-region candidates scored by the same verifiers
        local_raw = _local_candidates(x_best, r, clip_lb, clip_ub, rng_local)
        ref = np.vstack(rec_X)[np.concatenate(rec_f) <= f_ref_max]
        local_raw, tau, frac_loc = knn_gate(local_raw, ref, y_mean, y_std, n_min=b_local_eff)
        _, P_loc, _ = verifier_score(local_raw)
        picks.append(local_raw[np.argsort(-P_loc)[:b_local_eff]])

        X_sel = np.vstack(picks)
        best_before = tracker.best_f
        f_obs, n_inf = tracker.evaluate(X_sel, trial + 1)
        rec_X.append(np.asarray(X_sel, float))
        rec_f.append(np.asarray(f_obs, float))

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


def run_dali_le_gate_syn(prob, seed, trials):
    return run_dali_gate_syn(prob, seed, trials, b_local=2, use_diffusion=True)


def run_le_only_gate_syn(prob, seed, trials):
    return run_dali_gate_syn(prob, seed, trials, b_local=5, use_diffusion=False)
