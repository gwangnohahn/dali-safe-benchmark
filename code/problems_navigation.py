"""EXPLORATORY navigation block (not pre-registered; plan v3_final does not
contain it).  Question: when does the diffusion generator help navigation
beyond the LE trust region?  Paper 1 found a significant diffusion
contribution only on the 2-D constrained Branin, where the record covers the
feasible set and the optimum's basin differs from the warm-start best.  This
module builds three constrained multimodal problems in 2 / 3 / 6 dimensions
that reproduce that regime, each in two warm-start variants:

  <name>        200 uniform feasible points (record spread over the feasible set;
                paper-1 cbranin2 protocol)
  <name>_hole   200 uniform feasible points drawn OUTSIDE a ball of radius
                0.2 (2-D) / 0.3 (3-D, 6-D) x range around x* (the record covers
                the feasible set but has never visited the best region; the
                warm-start best therefore lies in another basin or on the slope)

Problems (minimisation, feasible <=> g(x) >= 0):
  ncamel2   six-hump camel on [-3,3]x[-2,2]; disk centred (1.2,-0.7), r=1.25 -->
            contains one global minimum (0.0898,-0.7126) near its edge and the
            local minimum (1.7036,-0.7961) well inside; excludes the other global minimum.
  nhart3    Hartmann-3 on [0,1]^3; ball centred (0.30,0.35,0.50), r=0.55 -->
            contains the global minimum (0.1146,0.5556,0.8525) and the local
            minimum (0.3687,0.1176,0.2676).
  nhart6    Hartmann-6 on [0,1]^6; ball centred at the midpoint of the global
            minimum and the second-best local minimum, r=0.65.

f* (regret reference) is calibrated once (Monte-Carlo over the feasible set +
L-BFGS-B refinement from the best starts) and cached in navigation_params.json
together with the feasible fraction of the box.  All methods of the suite runner
work unchanged (g, g_vector, feasible, warm_start, lb/ub, f_star).
"""

import json
import os

import numpy as np
from scipy.optimize import minimize

from problems_synthetic import SyntheticProblem, Hartmann6, WARM_N

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(CODE_DIR, "navigation_params.json")
HOLE_RADIUS = {"ncamel2": 0.20, "nhart3": 0.30, "nhart6": 0.30}   # fraction of the box range, around x*
MC_SEED, MC_N = 0, 200_000

NAV_NAMES = ["ncamel2", "nhart3", "nhart6"]
NAV_VARIANTS = [n for n in NAV_NAMES] + [n + "_hole" for n in NAV_NAMES]


class BallConstrainedProblem(SyntheticProblem):
    """Box problem + ball (disk) constraint; uniform-feasible warm start, optionally
    excluding a ball around the optimum (`hole`)."""

    constrained = True
    centre = None
    radius = None
    local_min = None     # a local (non-global) minimiser inside the ball

    def __init__(self, name, dim, lb, ub, hole=False):
        super().__init__(name + ("_hole" if hole else ""), dim, lb, ub)
        self.hole = bool(hole)
        self.x_star = None
        self.f_star = None
        self.f_star_source = "nav_params"
        self.box_feasible_frac = None

    # -- constraint --------------------------------------------------------
    def g(self, X):
        X = np.atleast_2d(X)
        return self.radius - np.sqrt(((X - self.centre[None, :]) ** 2).sum(axis=1))

    margin = g

    def g_vector(self, X):
        return self.g(X)[:, None]

    def feasible(self, X):
        return self.g(X) >= 0

    # -- warm start --------------------------------------------------------
    def in_hole(self, X):
        X = np.atleast_2d(X)
        r = HOLE_RADIUS[self.name.replace("_hole", "")] * (self.ub - self.lb)
        return (((X - self.x_star[None, :]) / r[None, :]) ** 2).sum(axis=1) <= 1.0

    def _uniform_feasible(self, rng, n):
        out, got = [], 0
        while got < n:
            X = self.lb + rng.rand(4096, self.dim) * (self.ub - self.lb)
            ok = self.feasible(X)
            if self.hole:
                ok &= ~self.in_hole(X)
            X = X[ok]
            out.append(X)
            got += len(X)
        return np.vstack(out)[:n]

    def warm_start(self, seed):
        rng = np.random.RandomState(seed)
        X = self._uniform_feasible(rng, WARM_N)
        return X, self.evaluate(X)


class NavCamel2(BallConstrainedProblem):
    centre = np.array([1.2, -0.7])
    radius = 1.25
    local_min = np.array([1.7036, -0.7961])

    def __init__(self, hole=False):
        super().__init__("ncamel2", 2, [-3.0, -2.0], [3.0, 2.0], hole)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        x, y = X[:, 0], X[:, 1]
        return (4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2


class NavHartmann3(BallConstrainedProblem):
    centre = np.array([0.30, 0.35, 0.50])
    radius = 0.55
    local_min = np.array([0.36872, 0.11756, 0.26757])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
    ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

    def __init__(self, hole=False):
        super().__init__("nhart3", 3, [0.0] * 3, [1.0] * 3, hole)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        inner = ((X[:, None, :] - self.P[None, :, :]) ** 2 * self.A[None, :, :]).sum(axis=2)
        return -(self.ALPHA[None, :] * np.exp(-inner)).sum(axis=1)


class NavHartmann6(BallConstrainedProblem):
    GLOBAL = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    local_min = np.array([0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381])   # refined below
    centre = 0.5 * (GLOBAL + local_min)
    radius = 0.65

    def __init__(self, hole=False):
        super().__init__("nhart6", 6, [0.0] * 6, [1.0] * 6, hole)
        self._h6 = Hartmann6()

    def evaluate(self, X):
        return self._h6.evaluate(X)


CLASSES = {"ncamel2": NavCamel2, "nhart3": NavHartmann3, "nhart6": NavHartmann6}


# ---------------------------------------------------------------------------
# f* calibration
# ---------------------------------------------------------------------------
def _refine(prob, x0):
    pen = lambda x: prob.evaluate(x[None, :])[0] + 1e3 * max(0.0, -prob.g(x[None, :])[0]) ** 2
    r = minimize(pen, x0, method="L-BFGS-B", bounds=list(zip(prob.lb, prob.ub)))
    x = np.clip(r.x, prob.lb, prob.ub)
    return x, float(prob.evaluate(x[None, :])[0]), bool(prob.feasible(x[None, :])[0])


def calibrate(verbose=True):
    rng = np.random.RandomState(MC_SEED)
    out = {"meta": {"mc_seed": MC_SEED, "mc_n": MC_N, "hole_radius": HOLE_RADIUS}, "configs": {}}
    for name, cls in CLASSES.items():
        p = cls()
        X = p.lb + rng.rand(MC_N, p.dim) * (p.ub - p.lb)
        feas = p.feasible(X)
        frac = float(feas.mean())
        Xf = X[feas]
        f = p.evaluate(Xf)
        starts = Xf[np.argsort(f)[:30]]
        best = (None, np.inf)
        for x0 in list(starts) + [p.local_min]:
            x, fx, ok = _refine(p, x0)
            if ok and fx < best[1]:
                best = (x, fx)
        xl, fl, okl = _refine(p, p.local_min)
        # local-minimum sanity: the refined local minimiser must be feasible and worse than f*
        rec = {"f_star": best[1], "x_star": best[0].tolist(), "box_feasible_frac": frac,
               "local_min": xl.tolist(), "f_local": fl, "local_feasible": okl,
               "f_mc_min": float(f.min()), "g_at_x_star": float(p.g(best[0][None, :])[0]),
               "g_at_local": float(p.g(xl[None, :])[0])}
        out["configs"][name] = rec
        if verbose:
            print(f"{name}: f*={best[1]:.5f} at {np.round(best[0], 4)} (g={rec['g_at_x_star']:.3f}); "
                  f"local f={fl:.5f} feasible={okl} (g={rec['g_at_local']:.3f}); box feasible {frac:.4f}")
    json.dump(out, open(PARAMS_PATH, "w"), indent=1)
    return out


def load_params(auto=True):
    if not os.path.exists(PARAMS_PATH):
        if not auto:
            raise FileNotFoundError(PARAMS_PATH)
        return calibrate(verbose=False)
    return json.load(open(PARAMS_PATH))


def build_nav_problems(names=None, params=None):
    params = params or load_params()
    names = names or NAV_VARIANTS
    reg = {}
    for n in names:
        base, hole = (n[:-5], True) if n.endswith("_hole") else (n, False)
        if base not in CLASSES:
            continue
        p = CLASSES[base](hole=hole)
        rec = params["configs"][base]
        p.x_star = np.asarray(rec["x_star"])
        p.f_star = float(rec["f_star"])
        p.known_min = p.f_star
        p.box_feasible_frac = rec["box_feasible_frac"]
        p.local_min = np.asarray(rec["local_min"])
        reg[p.name] = p
    return reg


def verify(seed=43):
    """Print warm-start facts per variant: best warm value, distance of the warm
    best to x* and to the local minimum, fraction of the record inside the trap."""
    P = load_params()
    for n, p in build_nav_problems().items():
        X, f = p.warm_start(seed)
        i = int(np.argmin(f))
        rec = P["configs"][n.replace("_hole", "")]
        xs, xl = np.asarray(rec["x_star"]), np.asarray(rec["local_min"])
        print(f"{n:14s} warm best f={f[i]:.4f} (f*={rec['f_star']:.4f}, local={rec['f_local']:.4f}); "
              f"|xbest-x*|={np.linalg.norm(X[i]-xs):.3f} |xbest-xlocal|={np.linalg.norm(X[i]-xl):.3f}; "
              f"all feasible={bool(p.feasible(X).all())}; in hole={int(p.in_hole(X).sum())}")


if __name__ == "__main__":
    calibrate()
    verify()
