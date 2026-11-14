"""Real-process datasets with a data-derived hidden feasibility oracle (paper 2, W1-3).

Plan v3_final section 2.4 (H5): the three paper-1 datasets (SFS steel fatigue,
NOMC methane conversion, CCPP power plant) are wrapped as SyntheticProblem-
compatible problems so that run_constraint_suite.py runs all nine methods on
them unchanged.

Objective (paper-1 real-data protocol, run_so_experiments.py):
  the XGBoost oracle (MultiOutputRegressor, 200 trees, depth 4, seed 0) fitted
  on the whole dataset predicts the performance vector; the scalarised
  objective (direction x weights) is MAXIMISED in paper 1, so here
  f(x) = -objective(oracle(x)) is minimised (runner convention s = -f).
Warm start: the same 200 data rows as paper 1 (run_so_experiments.
  warm_start_indices(n, 200, seed)), with their MEASURED objective values
  (paper-1 real-data protocol; proposals are scored by the oracle).  Rows the
  oracle deems infeasible are replaced by resampling with the same RNG
  (practically never happens, see verify()).

Hidden feasibility oracle ("outside the recorded operating region = failure"):
  z-score every condition variable with the full-data mean/sd; reference set
  R = unique standardised rows; d(x) = mean distance from x to its 5 nearest
  neighbours in R; threshold tau = q-th percentile (q = 0.99 main; 0.95, 0.999
  sensitivity) of the leave-one-out kNN5 distances of R itself; g(x) = tau - d(x),
  feasible <=> g >= 0.  Data rows are in R, so they are feasible by construction.
  Optional extra constraints (e.g. a NOMC domain rule) can be ANDed via
  extra_constraints=[callable(X)->g_array].

f* (regret reference): no ground truth; the empirical minimum of f over the
  data rows and over 200k uniform box points that are feasible
  (f_star_source = "empirical").

Names: real_<sfs|nomc|ccpp>_q99 (main), _q95 / _q999 (sensitivity).
Usage: python feasibility_oracle_real.py --verify
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems_synthetic import SyntheticProblem, WARM_N
from run_so_experiments import DATASETS, load_problem, warm_start_indices, objective_value

SHORT = {"steel_fatigue": "sfs", "methane_conv": "nomc", "power_plant": "ccpp"}
KNN_K = 5
QUANTILES = {"q95": 0.95, "q99": 0.99, "q999": 0.999}
FSTAR_MC_N, FSTAR_MC_SEED = 200_000, 0

_DATA_CACHE = {}


def _load(ds_name):
    """Cached (comp_all, perf_all, oracle) -- the oracle is seed independent."""
    if ds_name not in _DATA_CACHE:
        cfg = DATASETS[ds_name]
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = dict(cfg, file=os.path.join(here, cfg["file"]))
        _DATA_CACHE[ds_name] = load_problem(cfg)
    return _DATA_CACHE[ds_name]


class KNNDensityOracle:
    """d(x) = mean kNN_k distance to the standardised unique data rows;
    tau = quantile q of the leave-one-out kNN_k distances of those rows."""

    def __init__(self, X_data, q=0.99, k=KNN_K):
        X_data = np.asarray(X_data, float)
        self.mu = X_data.mean(axis=0)
        self.sd = X_data.std(axis=0) + 1e-12
        self.R = np.unique(((X_data - self.mu) / self.sd).round(10), axis=0)
        self.k, self.q = k, q
        self._nn = NearestNeighbors(n_neighbors=k).fit(self.R)
        loo = NearestNeighbors(n_neighbors=k + 1).fit(self.R)
        d, _ = loo.kneighbors(self.R)
        self.loo_dist = d[:, 1:].mean(axis=1)
        self.tau = float(np.quantile(self.loo_dist, q))
        self.n_ref = int(len(self.R))

    def dist(self, X):
        Z = (np.atleast_2d(np.asarray(X, float)) - self.mu) / self.sd
        d, _ = self._nn.kneighbors(Z)
        return d.mean(axis=1)

    def g(self, X):
        return self.tau - self.dist(X)

    def feasible(self, X):
        return self.g(X) >= 0.0


class RealDataProblem(SyntheticProblem):
    """Paper-1 real dataset + XGB oracle + hidden kNN-density feasibility."""

    constrained = True

    def __init__(self, ds_name, q_key="q99", extra_constraints=(), n_warm=WARM_N):
        assert ds_name in DATASETS and q_key in QUANTILES
        comp_all, perf_all, oracle = _load(ds_name)
        lb, ub = comp_all.min(axis=0), comp_all.max(axis=0)
        super().__init__(f"real_{SHORT[ds_name]}_{q_key}", comp_all.shape[1], lb, ub)
        self.ds_name, self.q_key = ds_name, q_key
        self.cfg = DATASETS[ds_name]
        self.comp_all, self.perf_all, self.oracle = comp_all, perf_all, oracle
        self.n_warm = int(n_warm)
        self.knn = KNNDensityOracle(comp_all, q=QUANTILES[q_key])
        self.extra_constraints = list(extra_constraints)
        # measured objective of every data row (paper-1 warm-start values)
        self.f_data = -np.array([objective_value(p, self.cfg["direction"], self.cfg["weights"])
                                 for p in perf_all])
        self.f_star = None          # computed lazily (MC), see compute_f_star()
        self.f_star_source = "empirical"
        self.config = {"dataset": ds_name, "q": QUANTILES[q_key], "tau": self.knn.tau,
                       "knn_k": KNN_K, "n_ref": self.knn.n_ref, "dim": self.dim,
                       "n_extra_constraints": len(self.extra_constraints)}

    # -- objective ------------------------------------------------------------
    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        P = self.oracle.predict(X)
        return -np.array([objective_value(p, self.cfg["direction"], self.cfg["weights"]) for p in P])

    # -- hidden constraint ----------------------------------------------------
    def g_vector(self, X):
        cols = [self.knn.g(X)] + [np.asarray(c(X), float) for c in self.extra_constraints]
        return np.stack(cols, axis=1)

    def g(self, X):
        return self.g_vector(X).min(axis=1)

    margin = g

    def feasible(self, X):
        return self.g(X) >= 0.0

    # -- warm start (paper-1 rows, measured values) ---------------------------
    def warm_start_labeled(self, seed, n=None):
        n = self.n_warm if n is None else int(n)
        idx = warm_start_indices(len(self.comp_all), n, seed)
        feas = self.feasible(self.comp_all[idx])
        if not feas.all():   # replace infeasible rows with further draws (same RNG stream)
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(self.comp_all))
            pool = [i for i in order if i not in set(idx) and self.feasible(self.comp_all[i:i + 1])[0]]
            idx = np.array([i for i, f in zip(idx, feas) if f] + pool[: int((~feas).sum())])
        X0 = self.comp_all[idx].copy()
        return X0, self.f_data[idx].copy(), np.ones(len(idx), dtype=bool), idx

    def warm_start(self, seed):
        X0, f0, _, _ = self.warm_start_labeled(seed)
        return X0, f0

    # -- regret reference -----------------------------------------------------
    def compute_f_star(self, n=FSTAR_MC_N, seed=FSTAR_MC_SEED):
        rng = np.random.RandomState(seed)
        B = self.lb + rng.rand(n, self.dim) * (self.ub - self.lb)
        B = B[self.feasible(B)]
        cands = [float(self.f_data.min()), float(self.evaluate(self.comp_all).min())]
        if len(B):
            cands.append(float(self.evaluate(B).min()))
        self.f_star = float(min(cands))
        self.known_min = self.f_star
        self.config["f_star"] = self.f_star
        self.config["box_feasible_frac"] = float(len(B) / n)
        return self.f_star

    def set_f_star(self, f_star, box_feasible_frac=None):
        self.f_star = float(f_star)
        self.known_min = self.f_star
        self.config["f_star"] = self.f_star
        if box_feasible_frac is not None:
            self.config["box_feasible_frac"] = float(box_feasible_frac)


PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realdata_params.json")


def build_real_problems(q_keys=("q99",), datasets=tuple(DATASETS), with_f_star=True,
                        params_path=PARAMS_PATH):
    """f* (200k-point MC, ~1-2 min for CCPP) is cached in realdata_params.json so
    that worker processes do not recompute it; the cache is keyed by problem
    name and stores tau / n_ref for a consistency check."""
    import json
    cache = {}
    if with_f_star and os.path.exists(params_path):
        with open(params_path) as fh:
            cache = json.load(fh)
    probs, dirty = {}, False
    for ds in datasets:
        for qk in q_keys:
            p = RealDataProblem(ds, qk)
            if with_f_star:
                c = cache.get(p.name)
                if c and abs(c["tau"] - p.knn.tau) < 1e-9 and c["n_ref"] == p.knn.n_ref:
                    p.set_f_star(c["f_star"], c.get("box_feasible_frac"))
                else:
                    p.compute_f_star()
                    cache[p.name] = {"tau": p.knn.tau, "n_ref": p.knn.n_ref, "f_star": p.f_star,
                                     "box_feasible_frac": p.config["box_feasible_frac"],
                                     "mc_n": FSTAR_MC_N, "mc_seed": FSTAR_MC_SEED}
                    dirty = True
            probs[p.name] = p
    if dirty:
        with open(params_path, "w") as fh:
            json.dump(cache, fh, indent=1)
    return probs


REAL_MAIN = [f"real_{SHORT[d]}_q99" for d in DATASETS]
REAL_SENS = [f"real_{SHORT[d]}_{q}" for d in DATASETS for q in ("q95", "q999")]


def verify(seeds=(43, 77, 11)):
    rows = []
    for ds in DATASETS:
        for qk in QUANTILES:
            p = RealDataProblem(ds, qk)
            p.compute_f_star()
            warm_ok = []
            same_as_v1 = []
            for s in seeds:
                X0, f0, fl, idx = p.warm_start_labeled(s)
                warm_ok.append(bool(p.feasible(X0).all()))
                same_as_v1.append(bool(np.array_equal(np.sort(idx), np.sort(warm_start_indices(len(p.comp_all), WARM_N, s)))))
            rows.append({"problem": p.name, "dim": p.dim, "n_rows": len(p.comp_all), "n_ref": p.knn.n_ref,
                         "tau": p.knn.tau, "box_feasible_frac": p.config["box_feasible_frac"],
                         "data_rows_feasible_frac": float(p.feasible(p.comp_all).mean()),
                         "f_star": p.f_star, "f_data_min": float(p.f_data.min()),
                         "warm_all_feasible": all(warm_ok), "warm_idx_same_as_paper1": all(same_as_v1),
                         "warm_f0_min_seed43": float(p.warm_start(43)[1].min())})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify()
