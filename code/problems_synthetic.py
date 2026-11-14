"""Synthetic benchmark problems for the DALI revision (Digital Discovery major revision).

Addresses reviewer requests:
  R2-2 : standard analytical test functions (Track A)
  R2-1 / R3-2 : controlled landscape properties -- sparsity of the high-performing
                region x warm-start coverage of it (Track B)
  R3-3 : problems with KNOWN feasibility constraints, so the fraction of infeasible
         proposals can be measured directly (Track C)

Conventions
-----------
* All problems are MINIMIZATION of a scalar f(x) over a box [lb, ub]^D.
  (The runner converts to the pipeline's maximize convention via s = -f.)
* Oracle = the analytical function itself (ground truth). No emulator anywhere,
  which removes the circularity concern raised for the emulator benchmarks.
* Warm start = 200 points drawn by the problem's own sampler (LHS for Tracks A/C-in-box;
  coverage-controlled samplers for Track B; feasible-only rejection sampling for Track C).
* Every random draw is seeded; landscape geometry (peak centers) is FIXED across
  seeds/runs via a dedicated landscape RNG so that only the warm start and method
  stochasticity vary between seeds.

Track B landscape: negative Gaussian mixture on [0,1]^6
  f(x) = -sum_i h_i * exp(-||x - c_i||^2 / (2 s_i^2)),  global peak h_0 = 1,
  7 decoy peaks with h in [0.5, 0.8], s_decoy = 0.15.
  Sparsity levels set the global-peak width s_0: broad 0.25 / mid 0.08 / sparse 0.025.

  Coverage modes control the warm-start data in FUNCTION-VALUE terms (radial
  placement r = s_0 * sqrt(-2 ln u) puts a point exactly at fraction u of the
  global-peak height, which stays interpretable in any dimension):
    c-in   : 85% uniform + 15% local with u ~ U(0.30, 0.95)
             (the best warm-start sample reaches ~95% of the optimum: the
              high-performing tail EXISTS in the data, though it is rare)
    c-tail : 95% uniform + 5%  local with u ~ U(0.05, 0.40)
             (data approaches the peak but never enters it: best local sample
              only ~40% of the optimum)
    c-out  : uniform, EXCLUDING the ball ||x - c_0|| < 2.5 s_0
             (peak lies outside the data support; peak contribution at the
              exclusion boundary is < 5% of its height -- the boundary case
              where DALI is EXPECTED to struggle, reported honestly)
"""

import numpy as np
from scipy.stats import qmc

WARM_N = 200
LANDSCAPE_SEED = 7  # fixes peak geometry for Track B/C2 across ALL runs


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class SyntheticProblem:
    """name, dim, lb, ub, evaluate(X)->f (minimize), warm_start(seed)->(X0, f0).

    Constrained problems additionally implement feasible(X)->bool array and set
    self.constrained = True; their warm_start returns feasible points only.
    """

    constrained = False
    known_min = None  # optional: known global minimum value (for regret reporting)

    def __init__(self, name, dim, lb, ub):
        self.name = name
        self.dim = dim
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)

    def evaluate(self, X):
        raise NotImplementedError

    def feasible(self, X):
        return np.ones(np.atleast_2d(X).shape[0], dtype=bool)

    def _lhs(self, seed, n=WARM_N):
        sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
        u = sampler.random(n)
        return self.lb + u * (self.ub - self.lb)

    def warm_start(self, seed):
        X0 = self._lhs(seed)
        return X0, self.evaluate(X0)


# ---------------------------------------------------------------------------
# Track A -- standard analytical functions (all minimize)
# ---------------------------------------------------------------------------
class Branin2(SyntheticProblem):
    known_min = 0.397887

    def __init__(self):
        super().__init__("branin2", 2, [-5.0, 0.0], [10.0, 15.0])

    def evaluate(self, X):
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
        return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


class Ackley(SyntheticProblem):
    known_min = 0.0

    def __init__(self, dim=6):
        super().__init__(f"ackley{dim}", dim, [-32.768] * dim, [32.768] * dim)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        d = X.shape[1]
        s1 = np.sqrt((X**2).sum(axis=1) / d)
        s2 = np.cos(2 * np.pi * X).sum(axis=1) / d
        return -20.0 * np.exp(-0.2 * s1) - np.exp(s2) + 20.0 + np.e


class Rastrigin(SyntheticProblem):
    known_min = 0.0

    def __init__(self, dim=6):
        super().__init__(f"rastrigin{dim}", dim, [-5.12] * dim, [5.12] * dim)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        return 10.0 * X.shape[1] + (X**2 - 10.0 * np.cos(2 * np.pi * X)).sum(axis=1)


class Hartmann6(SyntheticProblem):
    known_min = -3.32237

    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

    def __init__(self):
        super().__init__("hartmann6", 6, [0.0] * 6, [1.0] * 6)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        inner = ((X[:, None, :] - self.P[None, :, :]) ** 2 * self.A[None, :, :]).sum(axis=2)
        return -(self.ALPHA[None, :] * np.exp(-inner)).sum(axis=1)


class Levy(SyntheticProblem):
    known_min = 0.0

    def __init__(self, dim=10):
        super().__init__(f"levy{dim}", dim, [-10.0] * dim, [10.0] * dim)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        w = 1.0 + (X - 1.0) / 4.0
        term1 = np.sin(np.pi * w[:, 0]) ** 2
        term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
        wi = w[:, :-1]
        term2 = ((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2)).sum(axis=1)
        return term1 + term2 + term3


# ---------------------------------------------------------------------------
# Track B -- Gaussian-mixture landscape with controlled sparsity x coverage
# ---------------------------------------------------------------------------
SPARSITY_SIGMA = {"broad": 0.25, "mid": 0.08, "sparse": 0.025}
COVERAGE_MODES = ("cin", "ctail", "cout")
N_DECOYS = 7
DECOY_SIGMA = 0.15


def _landscape(dim=6, feasible_fn=None):
    """Fixed peak geometry: global center c0 + decoy centers/heights.

    If feasible_fn is given, c0 is placed at the first feasible draw (Track C2)."""
    rng = np.random.RandomState(LANDSCAPE_SEED)
    while True:
        c0 = 0.15 + 0.7 * rng.rand(dim)  # keep away from box edges
        if feasible_fn is None or feasible_fn(c0[None, :])[0]:
            break
    decoys, heights = [], []
    while len(decoys) < N_DECOYS:
        c = rng.rand(dim)
        if np.linalg.norm(c - c0) > 0.35:  # decoys well-separated from global peak
            decoys.append(c)
            heights.append(0.5 + 0.3 * rng.rand())
    return c0, np.array(decoys), np.array(heights)


class GaussianMixture6(SyntheticProblem):
    """f(x) = -[h0*exp(-||x-c0||^2/2s0^2) + sum decoys]; minimize; min ~= -1 at c0."""

    known_min = -1.0  # approximately (decoy overlap is negligible by construction)

    def __init__(self, sparsity, coverage, dim=6, feasible_fn=None, name=None):
        assert sparsity in SPARSITY_SIGMA and coverage in COVERAGE_MODES
        super().__init__(name or f"gm6_{sparsity}_{coverage}", dim,
                         [0.0] * dim, [1.0] * dim)
        self.sparsity, self.coverage = sparsity, coverage
        self.s0 = SPARSITY_SIGMA[sparsity]
        self.c0, self.decoys, self.heights = _landscape(dim, feasible_fn)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        v = np.exp(-((X - self.c0[None, :]) ** 2).sum(axis=1) / (2 * self.s0**2))
        for c, h in zip(self.decoys, self.heights):
            v = v + h * np.exp(-((X - c[None, :]) ** 2).sum(axis=1) / (2 * DECOY_SIGMA**2))
        return -v

    # -- coverage-controlled warm start (function-value based) --------------
    CIN_U = (0.30, 0.95)    # local samples sit at 30..95% of the peak height
    CTAIL_U = (0.05, 0.40)  # local samples sit at  5..40% of the peak height
    R_EXCL_FACTOR = 2.5     # cout: exclude ||x-c0|| < 2.5*s0 (peak contrib <5%)

    def _local_at_value(self, rng, n, u_lo, u_hi):
        """Points at radius r = s0*sqrt(-2 ln u), u~U(u_lo,u_hi): the global-peak
        term equals exactly u there, in any dimension (box-truncated resampling)."""
        out = []
        while len(out) < n:
            u = u_lo + (u_hi - u_lo) * rng.rand()
            r = self.s0 * np.sqrt(-2.0 * np.log(u))
            z = rng.randn(self.dim)
            x = self.c0 + r * z / (np.linalg.norm(z) + 1e-12)
            if np.all(x >= self.lb) and np.all(x <= self.ub) and self._ok(x):
                out.append(x)
        return np.array(out)

    def warm_start(self, seed):
        rng = np.random.RandomState(seed)
        if self.coverage == "cin":
            n_loc = int(round(0.15 * WARM_N))
            X = np.vstack([self._uniform_ok(rng, WARM_N - n_loc),
                           self._local_at_value(rng, n_loc, *self.CIN_U)])
        elif self.coverage == "ctail":
            n_loc = int(round(0.05 * WARM_N))
            X = np.vstack([self._uniform_ok(rng, WARM_N - n_loc),
                           self._local_at_value(rng, n_loc, *self.CTAIL_U)])
        else:  # cout: uniform, excluding the peak's support ball
            r_excl = self.R_EXCL_FACTOR * self.s0
            out = []
            while len(out) < WARM_N:
                x = self.lb + rng.rand(self.dim) * (self.ub - self.lb)
                if np.linalg.norm(x - self.c0) >= r_excl and self._ok(x):
                    out.append(x)
            X = np.array(out)
        rng.shuffle(X)
        return X, self.evaluate(X)

    # hooks so the constrained subclass can force feasible-only warm starts
    def _ok(self, x):
        return True

    def _uniform_ok(self, rng, n):
        out = []
        while len(out) < n:
            x = self.lb + rng.rand(self.dim) * (self.ub - self.lb)
            if self._ok(x):
                out.append(x)
        return np.array(out)


# ---------------------------------------------------------------------------
# Track C -- known feasibility constraints
# ---------------------------------------------------------------------------
class ConstrainedBranin(Branin2):
    """Branin + disk constraint (x1-2.5)^2 + (x2-7.5)^2 <= 18  (~25% of the box).

    All three unconstrained global minima are INFEASIBLE, so the constrained
    optimum sits inside/on the disk; methods ignorant of the constraint will
    be drawn toward infeasible basins."""

    constrained = True
    known_min = None  # constrained optimum estimated numerically in the runner
    CENTER = np.array([2.5, 7.5])
    R2 = 18.0

    def __init__(self):
        super().__init__()
        self.name = "cbranin2"

    def feasible(self, X):
        X = np.atleast_2d(X)
        return ((X - self.CENTER[None, :]) ** 2).sum(axis=1) <= self.R2

    def warm_start(self, seed):
        rng = np.random.RandomState(seed)
        out = []
        while len(out) < WARM_N:
            x = self.lb + rng.rand(self.dim) * (self.ub - self.lb)
            if self.feasible(x[None, :])[0]:
                out.append(x)
        X0 = np.array(out)
        return X0, self.evaluate(X0)


def _c2_feasible(X):
    """Nonconvex feasibility for Track C2: mean_i sin(2 pi x_i) >= TAU."""
    X = np.atleast_2d(X)
    return np.sin(2 * np.pi * X).mean(axis=1) >= C2_TAU


# TAU calibrated so that ~20% of the unit box is feasible (MC, fixed seed):
#   P[mean_i sin(2 pi x_i) >= 0.24] ~= 0.20 for D=6  (see calibrate_c2_tau below)
C2_TAU = 0.24


def calibrate_c2_tau(target=0.20, dim=6, n=200_000, seed=123):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, dim)
    g = np.sin(2 * np.pi * X).mean(axis=1)
    return float(np.quantile(g, 1.0 - target))


class ConstrainedGM6(GaussianMixture6):
    """Track B mid-sparsity landscape + nonconvex known constraint; the global
    peak c0 is placed at a feasible location (landscape RNG rejection)."""

    constrained = True

    def __init__(self):
        super().__init__("mid", "cin", feasible_fn=_c2_feasible, name="cgm6")

    def feasible(self, X):
        return _c2_feasible(X)

    def _ok(self, x):
        return _c2_feasible(x[None, :])[0]

    def warm_start(self, seed):
        # feasible-only: uniform rejection + the c-in local fraction (also
        # feasible; _ok() enforces the constraint inside _local_at_value)
        rng = np.random.RandomState(seed)
        n_loc = int(round(0.15 * WARM_N))
        X = np.vstack([self._uniform_ok(rng, WARM_N - n_loc),
                       self._local_at_value(rng, n_loc, *self.CIN_U)])
        rng.shuffle(X)
        return X, self.evaluate(X)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def build_problems():
    probs = {}
    # Track A
    for p in (Branin2(), Ackley(6), Rastrigin(6), Hartmann6(), Levy(10)):
        probs[p.name] = p
    # Track B (3 sparsity x 3 coverage)
    for sp in SPARSITY_SIGMA:
        for cov in COVERAGE_MODES:
            p = GaussianMixture6(sp, cov)
            probs[p.name] = p
    # Track C
    for p in (ConstrainedBranin(), ConstrainedGM6()):
        probs[p.name] = p
    return probs


TRACK_A = ["branin2", "ackley6", "rastrigin6", "hartmann6", "levy10"]
TRACK_B = [f"gm6_{sp}_{cov}" for sp in ("broad", "mid", "sparse")
           for cov in ("cin", "ctail", "cout")]
TRACK_C = ["cbranin2", "cgm6"]


if __name__ == "__main__":
    # sanity report: bounds, warm-start stats, feasible fractions, sparsity volumes
    np.set_printoptions(precision=4, suppress=True)
    print("C2_TAU calibration check:", calibrate_c2_tau())
    for name, p in build_problems().items():
        X0, f0 = p.warm_start(seed=43)
        line = (f"{name:16s} D={p.dim:2d}  f0 min/med {f0.min():9.4f}/{np.median(f0):9.4f}")
        if p.constrained:
            rng = np.random.RandomState(0)
            Xu = p.lb + rng.rand(100_000, p.dim) * (p.ub - p.lb)
            line += f"  feasible_frac={p.feasible(Xu).mean():.3f}"
        if isinstance(p, GaussianMixture6):
            rng = np.random.RandomState(0)
            Xu = p.lb + rng.rand(200_000, p.dim) * (p.ub - p.lb)
            fu = p.evaluate(Xu)
            thr = -0.8  # within 80% of global peak height
            line += f"  top_frac(f<-0.8)={np.mean(fu < thr):.2e}  dist_min_c0={np.min(np.linalg.norm(X0 - p.c0, axis=1)):.3f}"
        print(line)
