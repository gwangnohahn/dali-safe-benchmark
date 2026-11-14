"""Shared bookkeeping for the constraint-suite runners (paper 2).

SuiteTracker extends the paper-1 EvalTracker (run_synthetic_benchmark.py)
without modifying it:
  * proposals CSV gains a `g` column (signed constraint margin m(x), when the
    problem exposes g) -- Fig 5(b), SCBO diagnostics;
  * per-cycle rows gain `cand_n` / `cand_feasible_frac`: feasibility of the
    method's raw candidate pool BEFORE any filtering/clipping (Fig 5(a), H3;
    cannot be reconstructed afterwards);
  * `first_violation_iter` (T1 helper; -1 = no violation in the run).

Everything the paper-1 tracker wrote (death penalty, best over feasible
evaluations, 8-column run CSV, proposals with x/f_true/feasible) is unchanged,
so the new CSVs are the old schema plus extra columns.

Method-name conventions for result files (paper-1 names kept where they
exist): DALIv2, DALIv2LE, LEonly, TuRBO, GA, BO, Random, SCBO, cTuRBOknown.
"""

import os
import time

import numpy as np
import pandas as pd

from run_synthetic_benchmark import EvalTracker, DTYPE, COMMON  # noqa: F401
from run_so_experiments import DEFAULT_SEEDS  # noqa: F401

NEW_SEEDS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 47]
ALL_SEEDS = DEFAULT_SEEDS + NEW_SEEDS

METHOD_NAMES = ["DALIv2", "DALIv2LE", "LEonly", "TuRBO", "GA", "BO", "Random",
                "SCBO", "cTuRBOknown"]


ACTIVE_TRACKER = None   # the tracker of the run in progress (hooks write here)


class SuiteTracker(EvalTracker):
    """EvalTracker + margin column + candidate-stage feasibility logging.

    Candidate pools are logged in two slots:
      "gen": the method's primary raw candidate pool of the cycle
             (DALI: every diffusion sample BEFORE clipping/filtering, collected
             by the generator hook; TuRBO/SCBO/cTuRBO-known: trust-region pool)
      "le" : LE-box candidates (DALIv2LE, LEonly), collected by the LE hook.
    Per-cycle row columns: cand_n, cand_feasible_frac, cand_le_n, cand_le_feasible_frac.
    Methods with no visible pool (BO, GA, Random) leave them NaN.
    proposals also get `in_box` (inside the known box; only the no-clip
    ablation can propose outside it)."""

    def __init__(self, prob, f_warm):
        global ACTIVE_TRACKER
        super().__init__(prob, f_warm)
        self.has_g = hasattr(prob, "g")
        self.first_violation_iter = -1
        self._cand = {"gen": {}, "le": {}}     # slot -> iteration -> (n, frac)
        self._buf = {"gen": [], "le": []}      # hook buffers for the current cycle
        self.n_infeasible_cum = 0
        ACTIVE_TRACKER = self

    def evaluate(self, X, iteration):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # flush hook buffers: everything generated since the last evaluation
        # belongs to this cycle
        for slot in ("gen", "le"):
            if self._buf[slot]:
                self.log_candidates(iteration, np.vstack(self._buf[slot]), slot=slot)
                self._buf[slot] = []
        n_before = len(self.proposals)
        f_obs, n_inf = super().evaluate(X, iteration)
        in_box = np.all((X >= self.prob.lb[None, :]) & (X <= self.prob.ub[None, :]), axis=1)
        g = np.asarray(self.prob.g(X), dtype=float) if self.has_g else None
        for i, row in enumerate(self.proposals[n_before:]):
            if g is not None:
                row["g"] = float(g[i])
            row["in_box"] = bool(in_box[i])
        if n_inf > 0 and self.first_violation_iter < 0:
            self.first_violation_iter = int(iteration)
        self.n_infeasible_cum += int(n_inf)
        return f_obs, n_inf

    def log_candidates(self, iteration, X_cand, slot="gen"):
        """Record the feasible fraction of a raw candidate pool of a cycle
        (pool BEFORE filtering/clipping)."""
        X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))
        if X_cand.shape[0] == 0:
            return
        feas = np.asarray(self.prob.feasible(X_cand), dtype=bool)
        self._cand[slot][int(iteration)] = (int(X_cand.shape[0]), float(feas.mean()))

    def buffer_candidates(self, X, slot="gen"):
        """Called by the hooks; X raw coordinates (any shape (k, D))."""
        self._buf[slot].append(np.atleast_2d(np.asarray(X, dtype=float)))

    def row(self, iteration, n_infeasible, cycle_time):
        r = super().row(iteration, n_infeasible, cycle_time)
        n, frac = self._cand["gen"].get(int(iteration), (0, float("nan")))
        r["cand_n"], r["cand_feasible_frac"] = n, frac
        n, frac = self._cand["le"].get(int(iteration), (0, float("nan")))
        r["cand_le_n"], r["cand_le_feasible_frac"] = n, frac
        return r


# ---------------------------------------------------------------------------
# Candidate-stage hooks (no change to paper-1 code: monkey-patched wrappers)
# ---------------------------------------------------------------------------
_HOOKS_INSTALLED = False


def install_candidate_hooks():
    """Wrap (1) CondDiffu.predict -> every diffusion sample, denormalised with
    the model's var_stat, goes to the active tracker's "gen" buffer (this is
    the raw generator output BEFORE np.clip and before the verifier filter);
    (2) run_dali_le._local_candidates -> LE-box pool to the "le" buffer.
    Idempotent; harmless when no SuiteTracker is active."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    from ml.model import CondDiffu
    import run_dali_le

    _orig_predict = CondDiffu.predict

    def predict_logged(self, x):
        out = _orig_predict(self, x)
        tr = ACTIVE_TRACKER
        if tr is not None and self.var_stat is not None:
            raw = out.numpy() * self.var_stat["y_std"] + self.var_stat["y_mean"]
            tr.buffer_candidates(raw, slot="gen")
        return out

    _orig_local = run_dali_le._local_candidates

    def local_logged(x_best, r, lb, ub, rng):
        X = _orig_local(x_best, r, lb, ub, rng)
        if ACTIVE_TRACKER is not None:
            ACTIVE_TRACKER.buffer_candidates(X, slot="le")
        return X

    CondDiffu.predict = predict_logged
    run_dali_le._local_candidates = local_logged
    _HOOKS_INSTALLED = True


def use_suite_tracker():
    """Make the paper-1 runners construct SuiteTracker instead of EvalTracker
    (they look the name up in their module globals at call time)."""
    import run_synthetic_benchmark as RB
    import run_dali_le
    RB.EvalTracker = SuiteTracker
    run_dali_le.EvalTracker = SuiteTracker


# ---------------------------------------------------------------------------
# Interface gaps of paper-1 code, closed at runtime (paper-1 files untouched)
# ---------------------------------------------------------------------------
_PATCHED = set()


def add_paper1_margins():
    """Give the paper-1 constrained problems a signed margin g(X) so that SCBO
    (constraint observation) and the margin logging work on cbranin2 / cgm6:
      cbranin2: g = sqrt(R2) - ||x - CENTER||   (distance to the disk boundary)
      cgm6    : g = mean_i sin(2 pi x_i) - C2_TAU  (paper-1 constraint value)
    Both are consistent with the classes' feasible() (g >= 0 <=> feasible)."""
    if "margins" in _PATCHED:
        return
    import problems_synthetic as SP

    def g_branin(self, X):
        X = np.atleast_2d(X)
        return np.sqrt(self.R2) - np.linalg.norm(X - self.CENTER[None, :], axis=1)

    def g_gm6(self, X):
        X = np.atleast_2d(X)
        return np.sin(2 * np.pi * X).mean(axis=1) - SP.C2_TAU

    SP.ConstrainedBranin.g = g_branin
    SP.ConstrainedGM6.g = g_gm6
    _PATCHED.add("margins")


def patch_bo_scaler():
    """run_bo_syn normalises the warm start with MinMaxScaler fitted on [lb, ub];
    a point lying exactly on the box boundary can map to 1 + 2.2e-16 and skopt
    then rejects it ("Not all points are within the bounds of the space").
    Real datasets derive lb/ub from the data, so boundary rows do occur.  This
    replaces the scaler used by run_bo_syn with one whose transform() is
    clipped to [0, 1] (numerically a no-op: <= 1 ulp)."""
    if "bo_scaler" in _PATCHED:
        return
    import run_synthetic_benchmark as RB
    from sklearn.preprocessing import MinMaxScaler

    class ClippedMinMaxScaler(MinMaxScaler):
        def transform(self, X):
            return np.clip(super().transform(X), 0.0, 1.0)

    RB.MinMaxScaler = ClippedMinMaxScaler
    _PATCHED.add("bo_scaler")


def apply_runtime_patches():
    use_suite_tracker()
    install_candidate_hooks()
    add_paper1_margins()
    patch_bo_scaler()


def write_run(outdir, method, prob, seed, rows, proposals, wall=None):
    """Write <method>_<problem>_seed<seed>.csv (+ _proposals.csv) exactly as
    paper-1 run_all() does (method/problem/seed columns first)."""
    os.makedirs(outdir, exist_ok=True)
    tag = f"{method}_{prob.name}_seed{seed}"
    df = pd.DataFrame(rows)
    df.insert(0, "method", method)
    df.insert(1, "problem", prob.name)
    df.insert(2, "seed", seed)
    df.to_csv(os.path.join(outdir, tag + ".csv"), index=False)
    pd.DataFrame(proposals).to_csv(os.path.join(outdir, tag + "_proposals.csv"), index=False)
    if wall is not None:
        print(f"       done in {wall:.1f}s (best_f {rows[-1]['best_f']:.4f}, "
              f"violations {sum(r['n_infeasible'] for r in rows)})", flush=True)
    return tag


def run_exists(outdir, method, prob_name, seed):
    return os.path.exists(os.path.join(outdir, f"{method}_{prob_name}_seed{seed}.csv"))


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0
