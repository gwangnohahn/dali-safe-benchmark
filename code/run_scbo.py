"""SCBO and cTuRBO-known on the paper-1 TuRBO skeleton (paper 2, W1-2).

Both methods reuse the paper-1 TuRBO-1 loop (run_synthetic_benchmark.run_turbo_syn
/ run_new_baselines.TurboState): unit-box normalisation, one SingleTaskGP with
Standardize refitted every cycle, trust region centred at the best point and
scaled by GP lengthscales, Sobol perturbation candidates with the tutorial's
per-coordinate perturbation mask, batch Thompson sampling, success/failure
counters with doubling/halving of the region and restart on collapse.  Only the
CANDIDATE SELECTION and the definition of "best point" change, so the three
runs TuRBO / SCBO / cTuRBO-known differ solely in how much they know about the
constraint:

  TuRBO (paper 1)   no constraint information; death-penalty value only.
  SCBO              after each evaluation the signed margin m(x) = g(x) is
                    observed and a second GP (constraint GP, c = -g <= 0
                    feasible) is fitted; candidates are chosen by constrained
                    Thompson sampling (botorch ConstrainedMaxPosteriorSampling,
                    Eriksson & Poloczek 2021): among candidates whose sampled
                    constraint is satisfied take the best sampled objective;
                    if none, the smallest sampled violation.  Region centre =
                    best FEASIBLE point; success = a feasible improvement.
  cTuRBO-known      the true g(x) is known: candidates are filtered by
                    prob.feasible() before Thompson sampling; never proposes an
                    infeasible point.  Upper reference.

Protocol identical to paper 1 (warm start 200 feasible-only, 20 cycles x batch
5, death penalty via the tracker, seeds), CSV schema = paper 1 + `g` column in
proposals + `cand_n`/`cand_feasible_frac` per cycle (see suite_common).

Information given to SCBO (deliberately generous -- "SCBO-favourable"):
  * warm-start margins are observed too (WARM_MARGINS=True), i.e. the record
    contains the constraint value of every past experiment;
  * the constraint is observed component-wise (prob.g_vector): one GP per
    smooth component -- G1/G2/G3 one GP on g, G4 (thin band |h| <= eps) two
    GPs on the one-sided limits eps -+ h, the way a two-sided spec limit would
    be modelled in practice (a single GP on the kinked eps - |h| is much worse);
  * the objective GP is trained on FEASIBLE observations only
    (INFEASIBLE_OBJECTIVE="exclude"): a failed experiment yields no objective
    measurement, so the penalty value is not fed to the GP.  "penalty" feeds
    the death-penalty value instead (paper-1 TuRBO behaviour).

Usage (smoke test):
  python run_scbo.py --problem g2_f020_int --seed 43 --trials 20 --outdir scratch/scbo
  python run_scbo.py --method cTuRBOknown --problem g1_f005_bnd --seed 43
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from suite_common import SuiteTracker, write_run, DTYPE, COMMON
from run_new_baselines import TurboState

SCBO_CFG = {
    "warm_margins": True,              # SCBO observes m(x) of the warm-start records
    "infeasible_objective": "exclude",  # "exclude" | "penalty"
    "n_cand_factor": 200,              # n_cand = clip(200*dim, 2000, 5000)  (paper-1 TuRBO)
    "known_extra_rounds": 10,          # cTuRBO-known: extra candidate rounds if too few feasible
}


# ---------------------------------------------------------------------------
# GP helpers (paper-1 choices: SingleTaskGP + Standardize, MLL fit)
# ---------------------------------------------------------------------------
def _fit_gp(X, Y):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _tr_candidates(state, model, x_center, n_candidates, seed_offset):
    """Trust-region candidate pool, identical to run_new_baselines.turbo_generate_batch
    up to the selection step (which is done by the caller)."""
    dim = x_center.shape[-1]
    try:
        ls = model.covar_module.lengthscale
        if ls is None:
            raise AttributeError
        weights = ls.squeeze().detach()
    except AttributeError:
        weights = torch.ones(dim, dtype=DTYPE)
    if weights.ndim == 0:
        weights = weights.unsqueeze(0)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    sobol = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed_offset)
    pert = sobol.draw(n_candidates).to(dtype=DTYPE)
    pert = tr_lb + (tr_ub - tr_lb) * pert
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),))] = 1
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]
    return X_cand


def _update_state_feasible(state, y_next_feas):
    """TuRBO counters driven by FEASIBLE improvements only (SCBO rule).
    y_next_feas: tensor of objective values (maximise) of the feasible points
    of the new batch (may be empty)."""
    improved = (y_next_feas.numel() > 0 and
                y_next_feas.max().item() > state.best_value + 1e-3 * abs(state.best_value))
    if improved:
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
    if y_next_feas.numel() > 0:
        state.best_value = max(state.best_value, y_next_feas.max().item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


# ---------------------------------------------------------------------------
# Main loop (mode = "scbo" | "known")
# ---------------------------------------------------------------------------
def run_tr_constrained(prob, seed, trials, mode="scbo", cfg=None):
    from botorch.models import ModelListGP
    from botorch.generation.sampling import MaxPosteriorSampling, ConstrainedMaxPosteriorSampling

    cfg = {**SCBO_CFG, **(cfg or {})}
    assert mode in ("scbo", "known")
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch = COMMON["batch_size"]
    lb, ub = prob.lb, prob.ub

    X0, f0 = prob.warm_start(seed)
    tracker = SuiteTracker(prob, f0)
    feas0 = np.asarray(prob.feasible(X0), dtype=bool)   # all True unless contaminated

    # data store (numpy): unit-box X, observed objective (penalty applied by the
    # protocol), true feasibility flag, observed margin (SCBO only)
    Xn = (X0 - lb) / (ub - lb)
    F_obs = np.where(feas0, f0, tracker.penalty_f)
    FEAS = feas0.copy()
    # observed constraint components (SCBO): (n, k) array, feasible <=> all >= 0;
    # k = 1 except the thin band (two one-sided limits).  Warm-start margins are
    # observed if WARM_MARGINS, else NaN (the constraint GPs then start from the
    # first evaluated batch).
    gvec = getattr(prob, "g_vector", lambda X: np.asarray(prob.g(X))[:, None])
    G = None
    if mode == "scbo":
        G0 = np.asarray(gvec(X0), dtype=float)
        G = G0 if cfg["warm_margins"] else np.full_like(G0, np.nan)

    def obj_training_set():
        if cfg["infeasible_objective"] == "exclude" or mode == "known":
            idx = np.where(FEAS)[0]
        else:
            idx = np.arange(len(F_obs))
        X = torch.tensor(Xn[idx], dtype=DTYPE)
        Y = torch.tensor(-F_obs[idx], dtype=DTYPE).unsqueeze(-1)   # maximise s = -f
        return X, Y

    def best_feasible_center():
        idx = np.where(FEAS)[0]
        j = idx[np.argmin(F_obs[idx])]
        return torch.tensor(Xn[j], dtype=DTYPE)

    state = TurboState(dim=prob.dim, batch_size=batch)
    state.best_value = float(-F_obs[FEAS].min())
    n_cand = int(min(5000, max(2000, cfg["n_cand_factor"] * prob.dim)))

    rows = []
    for t in range(trials):
        t0 = time.perf_counter()
        if state.restart_triggered:
            state = TurboState(dim=prob.dim, batch_size=batch)
            state.best_value = float(-F_obs[FEAS].min())

        X_obj, Y_obj = obj_training_set()
        model = _fit_gp(X_obj, Y_obj)
        x_center = best_feasible_center()
        X_cand = _tr_candidates(state, model, x_center, n_cand, seed_offset=seed * 1000 + t)
        X_cand_raw = X_cand.numpy() * (ub - lb) + lb
        tracker.log_candidates(t + 1, X_cand_raw)

        if mode == "scbo":
            ok = ~np.isnan(G[:, 0])
            if ok.sum() >= 2:
                Xc = torch.tensor(Xn[ok], dtype=DTYPE)
                c_models = [_fit_gp(Xc, torch.tensor(-G[ok, k], dtype=DTYPE).unsqueeze(-1))
                            for k in range(G.shape[1])]                       # c = -g <= 0
                sampler = ConstrainedMaxPosteriorSampling(
                    model=model, constraint_model=ModelListGP(*c_models), replacement=False)
            else:  # no margin observed yet: plain Thompson sampling this cycle
                sampler = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = sampler(X_cand, num_samples=batch)
        else:  # known constraint: keep only truly feasible candidates
            feas_c = np.asarray(prob.feasible(X_cand_raw), dtype=bool)
            pool = X_cand[torch.tensor(feas_c)]
            rounds = 0
            while pool.shape[0] < batch and rounds < cfg["known_extra_rounds"]:
                rounds += 1
                more = _tr_candidates(state, model, x_center, n_cand,
                                      seed_offset=seed * 1000 + t + 100_000 * rounds)
                fm = np.asarray(prob.feasible(more.numpy() * (ub - lb) + lb), dtype=bool)
                pool = torch.cat([pool, more[torch.tensor(fm)]], dim=0)
            if pool.shape[0] < batch:
                # last resort: the centre itself is feasible; jitter inside the region
                extra = x_center.expand(batch - pool.shape[0], prob.dim).clone()
                extra = torch.clamp(extra + 1e-3 * state.length * torch.randn_like(extra), 0.0, 1.0)
                fe = np.asarray(prob.feasible(extra.numpy() * (ub - lb) + lb), dtype=bool)
                extra[~torch.tensor(fe)] = x_center
                pool = torch.cat([pool, extra], dim=0)
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = sampler(pool, num_samples=batch)

        X_next_raw = X_next.numpy() * (ub - lb) + lb
        f_obs, n_inf = tracker.evaluate(X_next_raw, t + 1)
        feas_next = np.asarray(prob.feasible(X_next_raw), dtype=bool)

        Xn = np.vstack([Xn, X_next.numpy()])
        F_obs = np.concatenate([F_obs, f_obs])
        FEAS = np.concatenate([FEAS, feas_next])
        if mode == "scbo":
            G = np.vstack([G, np.asarray(gvec(X_next_raw), dtype=float)])

        y_feas = torch.tensor(-f_obs[feas_next], dtype=DTYPE)
        state = _update_state_feasible(state, y_feas)
        rows.append(tracker.row(t + 1, n_inf, time.perf_counter() - t0))
    return rows, tracker.proposals


def run_scbo_syn(prob, seed, trials):
    return run_tr_constrained(prob, seed, trials, mode="scbo")


def run_cturbo_known_syn(prob, seed, trials):
    return run_tr_constrained(prob, seed, trials, mode="known")


METHODS = {"SCBO": run_scbo_syn, "cTuRBOknown": run_cturbo_known_syn}


# ---------------------------------------------------------------------------
# CLI (single-run smoke test; the full sweep is run_constraint_suite.py)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="SCBO", choices=list(METHODS))
    ap.add_argument("--problem", default="g2_f020_int")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default=None, help="write paper-1-style CSVs here")
    args = ap.parse_args()

    import problems_constraint_suite as CS
    import problems_synthetic as SP
    reg = {**SP.build_problems(), **CS.build_suite("all")}
    prob = reg[args.problem]
    t0 = time.perf_counter()
    rows, proposals = METHODS[args.method](prob, args.seed, args.trials)
    wall = time.perf_counter() - t0
    V = sum(r["n_infeasible"] for r in rows)
    print(f"{args.method} {prob.name} seed {args.seed}: best_f {rows[-1]['best_f']:.4f} "
          f"(f* {getattr(prob, 'f_star', float('nan')):.4f}), violations {V}, "
          f"cand feasible frac (mean) {np.nanmean([r['cand_feasible_frac'] for r in rows]):.3f}, "
          f"{wall:.1f}s")
    if args.outdir:
        write_run(args.outdir, args.method, prob, args.seed, rows, proposals)


if __name__ == "__main__":
    main()
