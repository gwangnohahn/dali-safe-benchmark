"""H3 ablations for the constraint suite (paper 2, plan section 1 / Fig 5(d)).

Two controls that isolate what in the DALI pipeline produces constraint
compliance.  Both are verbatim copies of paper-1 run_synthetic_benchmark.
run_dali_syn (v2 path: verifier ensemble + adaptive pi) with ONE change each;
paper-1 files are not modified.

  VerifRandom  "verifier + random": the diffusion generator is removed and the
               candidate pool is the same NUMBER of points drawn UNIFORMLY in
               the known box; the verifier ensemble filters/ranks them exactly
               as in DALIv2.  If compliance came from the verifier (or from the
               clip), this control would comply too.
  DALInoclip   DALIv2 with np.clip(comp_raw, lb, ub) removed: generated points
               outside the box are proposed as they are.  If compliance came
               from clipping to the box, this control would violate more.  The
               proposals CSV column `in_box` records box membership.

Everything else (warm start, 20 x 5, verifier config, retrain epochs, adaptive
pi, seeds, CSV schema via SuiteTracker) is identical to DALIv2.
"""

import time

import numpy as np
import torch
import torch.utils.data

import run_synthetic_benchmark as RB
from run_synthetic_benchmark import (COMMON, V2, adaptive_pi, greedy_select_v2,
                                     train_verifier_ensemble, verifier_predict, device)
from ml.model import FCNN, CondDiffu
from util.data import Dataset


def _run_dali_variant(prob, seed, trials, mode):
    """mode: 'verif_random' | 'noclip'.  Structure mirrors RB.run_dali_syn
    (use_verifier=True, use_adaptive_pi=True)."""
    assert mode in ("verif_random", "noclip")
    use_diffusion = mode == "noclip"
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_cond, n_perf = prob.dim, 1
    clip_lb, clip_ub = prob.lb, prob.ub
    rng_unif = np.random.RandomState(seed + 20260914)

    X0, f0 = prob.warm_start(seed)
    tracker = RB.EvalTracker(prob, f0)   # SuiteTracker after suite_common.use_suite_tracker()
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
        verifiers.append((m, torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-6)))
    torch.manual_seed(seed)
    train_verifier_ensemble(verifiers, dataset_train.y.to(device),
                            dataset_train.x.to(device), epochs=200)

    rows = []
    for trial in range(trials):
        t0 = time.perf_counter()
        # 1) targets (identical to DALIv2)
        perf_raw = dataset_train.x.numpy() * x_std + x_mean
        lb_f, ub_f = perf_raw.min(axis=0), perf_raw.max(axis=0)
        targets_raw = np.random.uniform(lb_f, ub_f, size=(COMMON["num_target_candidates"], n_perf))
        pi = adaptive_pi(trial, trials)
        scores_tmp = -targets_raw[:, 0]
        threshold = np.percentile(scores_tmp, pi)
        targets_raw = targets_raw[scores_tmp >= threshold]
        if targets_raw.shape[0] == 0:
            targets_raw = np.random.uniform(lb_f, ub_f, size=(10, n_perf))

        # 2) candidate pool
        n_pool = targets_raw.shape[0] * COMMON["num_comp_samples"]
        if mode == "verif_random":
            all_comps = clip_lb + rng_unif.rand(n_pool, n_cond) * (clip_ub - clip_lb)
            tracker.log_candidates(trial + 1, all_comps, slot="gen")
        else:  # noclip: diffusion samples, NOT clipped to the box
            all_comps = []
            for cand_x in targets_raw:
                cand_norm = (cand_x - x_mean) / x_std
                cand_tensor = torch.tensor(cand_norm, dtype=torch.float32).reshape(1, -1)
                for _ in range(COMMON["num_comp_samples"]):
                    comp_norm = diffu_model.predict(cand_tensor).numpy()[0]
                    comp_raw = (comp_norm * y_std + y_mean)[0]
                    all_comps.append(comp_raw)            # <-- the only change vs DALIv2
            all_comps = np.array(all_comps)
        comps_norm = (all_comps - y_mean) / y_std

        # 3) verifier filter + greedy selection (identical to DALIv2)
        cand_tensor = torch.tensor(comps_norm, dtype=torch.float32, device=device)
        ens_pred_norm = verifier_predict(verifiers, cand_tensor)
        ens_obj = -(ens_pred_norm * x_std + x_mean)[..., 0]
        P_score, U_score = ens_obj.mean(axis=0), ens_obj.std(axis=0)
        N = comps_norm.shape[0]
        keep_n = max(COMMON["batch_size"], int(np.ceil(N * V2["verifier_keep_frac"])))
        keep_idx = np.argsort(-P_score)[:keep_n]
        sel_local = greedy_select_v2(comps_norm[keep_idx], P_score[keep_idx], U_score[keep_idx],
                                     COMMON["batch_size"], V2["w_perf"], COMMON["alpha_uncertainty"])
        sel_idx = keep_idx[sel_local]

        # 4) evaluate (death penalty on infeasible), 5) retrain (identical)
        X_sel = all_comps[sel_idx]
        f_obs, n_inf = tracker.evaluate(X_sel, trial + 1)
        for x_raw, f_o in zip(X_sel, f_obs):
            perf_norm = (np.array([[f_o]]) - x_mean) / x_std
            comp_norm = (x_raw.reshape(1, -1) - y_mean) / y_std
            dataset_train.x = torch.cat([dataset_train.x, torch.tensor(perf_norm, dtype=torch.float)], dim=0)
            dataset_train.y = torch.cat([dataset_train.y, torch.tensor(comp_norm, dtype=torch.float)], dim=0)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
        if use_diffusion:
            for _ in range(50):
                pred_model.fit(loader_train, opt_pred, loss_func)
            for _ in range(50):
                diffu_model.fit(loader_train, opt_diffu)
        train_verifier_ensemble(verifiers, dataset_train.y.to(device), dataset_train.x.to(device), epochs=50)
        rows.append(tracker.row(trial + 1, n_inf, time.perf_counter() - t0))
    return rows, tracker.proposals


def run_verif_random_syn(prob, seed, trials):
    return _run_dali_variant(prob, seed, trials, "verif_random")


def run_dali_noclip_syn(prob, seed, trials):
    return _run_dali_variant(prob, seed, trials, "noclip")


METHODS = {"VerifRandom": run_verif_random_syn, "DALInoclip": run_dali_noclip_syn}
