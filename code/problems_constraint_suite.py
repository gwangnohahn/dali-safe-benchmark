"""Synthetic constraint suite for the DALI safety paper (paper 2, W1-1).

Purpose
-------
Systematic benchmark of *hidden* feasibility constraints: 4 constraint
geometries x 3 feasible-volume ratios x 2 optimum locations = 24 configurations
(plan v3_final, section 2.1).  Every configuration shares ONE objective
landscape -- the paper-1 Track-B 6-D Gaussian mixture at mid sparsity
(s0 = 0.08, 7 decoys, fixed centre c0 from LANDSCAPE_SEED) -- and differs only
in the constraint g(x) >= 0 that is imposed on top of it.

The runner treats these problems exactly like paper-1 Track C:
  * no method is told g(x); an infeasible proposal is a failed experiment
    (death penalty f_pen = max(f_warm) + 0.1 * range(f_warm), assigned by the
    runner's EvalTracker), the reported best is tracked over feasible points;
  * warm start = 200 feasible-only points (rejection sampling; the paper-1
    cgm6 "c-in" mixture: 85 % uniform + 15 % local at 30..95 % of the peak).

Interface compatibility (problems_synthetic.SyntheticProblem)
-------------------------------------------------------------
name, dim, lb, ub, constrained=True, known_min, evaluate(X) -> f (minimise),
feasible(X) -> bool, warm_start(seed) -> (X0, f0).  Additions used by the new
runners:
  g(X)                     signed constraint margin, feasible <=> g >= 0
                           (SCBO observes g at evaluated points; cTuRBO-known
                           gets the function itself)
  warm_start_labeled(seed, n=None, contamination=None)
                           -> (X0, f0, feasible_flags) for the H4 variants
  f_star, config           constrained optimum value and the config record

Factor 1 -- constraint geometry (all in the unit box [0,1]^6, c0 fixed)
  G1 convex        g = rho - ||x - a||
  G2 non-convex    g = mean_i sin(2 pi (x_i - delta_i)) - tau     (paper-1 cgm6 form,
                   shifted by delta; shifting leaves the feasible volume exactly
                   invariant because the box is one full period)
  G3 disconnected  union of two balls in the 3-D subspace S of the coordinates
                   in which c0 is most off-centre (the other 3 coordinates are
                   unconstrained):  g = max( rho - ||x_S - a||, rho - ||x_S - b|| ),
                   c0 in ball A (centre a), ball B centred at the far corner of
                   the S-cube, separated by a gap; the warm start is drawn from
                   ball A only ("two allowed operating regions in three of the
                   six variables").
                   [Refinement of the plan text "union of two balls": with c0
                   fixed in [0.15,0.85]^6 two disjoint FULL 6-D balls cannot
                   reach a feasible ratio of 0.5 (equal split needs rho ~ 0.71
                   while the far box corner is only 1.68 from c0), and two
                   parallel slabs give interior margins < 0.1 at ratios 0.2/0.05.
                   Two balls in a 3-D subspace satisfy every ratio and margin
                   target. Decided 2026-09-09.]
  G4 thin band     g = eps - |h(x)|,  h(x) = (x_j - q_j) - sum_{i != j} (x_i - q_i)^2
                   (quadratic paraboloid sheet; both sides of the band are
                   infeasible at every ratio)

Factor 2 -- feasible ratio  {0.50, 0.20, 0.05}: the size parameter theta
  (rho / tau / w / eps) is found by bisection on a fixed-seed uniform Monte
  Carlo sample (CALIB_N points); every geometry's feasible set is nested in
  theta, so the ratio is monotone and bisection is exact up to MC noise.

Factor 3 -- optimum location.  c0 never moves; the constraint is translated.
  interior : c0 at the deepest point of the set (G1/G3: centre; G2: sine
             maximum; G4: mid-surface).  Target margin >= 0.1 -- NOT attainable
             for G4 at ratio 0.05 (band half-thickness eps ~ 0.033): the actual
             margin is recorded in the verification report instead.
  boundary : constraint shifted along a fixed direction so that the distance
             from c0 to the constraint boundary is BOUNDARY_MARGIN = 0.01
             (G1/G3/G4 analytic; G2 by 1-D root finding on a numerical
             distance-to-boundary estimate).  The shift direction is chosen so
             that the feasible set can still grow to any ratio: the ball (G1),
             ball A (G3) and the band (G4) extend AWAY from the nearest box
             faces / from ball B, and the boundary near c0 faces the near box
             corner (G1), ball B (G3) or +x_j (G4).

Naming:  <geometry>_<ratio>_<location>[_<variant>]
         g2_f020_int, g4_f005_bnd, g1_f050_int_c10 (10 % contaminated warm
         start), g3_f020_int_w50 (50-point warm start)
Tier 1 (16): 4 geometries x 3 ratios x interior + 4 geometries x f020 x boundary
Tier 2 (8):  4 geometries x {f050, f005} x boundary

Calibrated parameters are stored in constraint_suite_params.json next to this
file (generated once with --calibrate and committed, so server and analysis
sessions use identical numbers).  --verify re-checks every configuration with
an independent MC sample and writes results/constraint_suite/suite_verification.csv.
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

from problems_synthetic import (GaussianMixture6, WARM_N, LANDSCAPE_SEED,
                                _landscape)

# ---------------------------------------------------------------------------
# Suite constants
# ---------------------------------------------------------------------------
DIM = 6
GEOMETRY_KEYS = ("g1", "g2", "g3", "g4")
RATIOS = {"f050": 0.50, "f020": 0.20, "f005": 0.05}
LOCATIONS = ("int", "bnd")

INTERIOR_MARGIN_MIN = 0.10   # plan: c0 margin >= 0.1 * range (range = 1)
BOUNDARY_MARGIN = 0.01       # plan: margin ~ 0.01
BOUNDARY_MARGIN_TOL = 0.005  # verification tolerance on the boundary margin

CALIB_SEED, CALIB_N = 123, 200_000     # bisection sample
VERIFY_SEED, VERIFY_N = 0, 200_000     # independent verification sample
RATIO_TOL = 0.02                       # plan: MC ratio within +/- 0.02
BISECT_ITERS = 40

G3_SUBSPACE_DIM = 3  # G3 balls live in the 3 coordinates where c0 is most off-centre
G3_GAP_MIN = 0.05    # minimum gap between the two balls (verified)
G4_CURVATURE = 1.0   # quadratic coefficient of the G4 sheet

COVERAGE_MODES = ("cin", "uniform")
CIN_LOCAL_FRAC = 0.15

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(CODE_DIR, "constraint_suite_params.json")
RESULTS_DIR = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")


# ---------------------------------------------------------------------------
# Fixed landscape and fixed directions (deterministic, no extra RNG)
# ---------------------------------------------------------------------------
def base_landscape():
    """Unconstrained paper-1 landscape geometry (c0, decoys, heights).

    c0 is NOT re-placed inside the constraint (unlike paper-1 cgm6): the plan
    fixes c0 and translates the constraint instead."""
    return _landscape(DIM, feasible_fn=None)


def fixed_directions(c0):
    """v: unit vector from c0 towards the farthest box corner (boundary-shift
    direction: the feasible set extends towards the far corner, the boundary
    near c0 faces the near corner); j: axis whose c0 coordinate is closest to
    0.5 (G4 sheet normal, most room on both sides)."""
    corner = np.where(c0 < 0.5, 1.0, 0.0)
    v = corner - c0
    v = v / np.linalg.norm(v)
    j = int(np.argmin(np.abs(c0 - 0.5)))
    return v, j


def g3_subspace(c0, k=G3_SUBSPACE_DIM):
    """Coordinates in which c0 is most off-centre (largest |c0_i - 0.5|)."""
    return np.argsort(-np.abs(c0 - 0.5))[:k]


# ---------------------------------------------------------------------------
# Constraint geometries
# ---------------------------------------------------------------------------
class ConstraintGeometry:
    """g(X) >= 0 is feasible. theta = size parameter (calibrated), location in
    {'int','bnd'}, shift = extra scalar used by geometries whose boundary
    placement is solved numerically (G2)."""

    key = None
    label = None
    THETA_BRACKET = (0.0, 3.0)

    def __init__(self, c0, theta, location, shift=0.0):
        assert location in LOCATIONS
        self.c0 = np.asarray(c0, dtype=float)
        self.theta = float(theta)
        self.location = location
        self.shift = float(shift)
        self.v, self.j = fixed_directions(self.c0)
        self._setup()

    def _setup(self):
        pass

    def g(self, X):
        raise NotImplementedError

    def feasible(self, X):
        return self.g(X) >= 0.0

    def g_vector(self, X):
        """(n, k) array of constraint components, feasible <=> all >= 0, with
        g(X) = min over components.  Used by constraint-learning baselines
        (SCBO fits one GP per component).  Default: the single scalar g; G4
        overrides with the two smooth one-sided limits of the band."""
        return self.g(X)[:, None]

    def warm_region(self, X):
        """Region the feasible-only warm start is drawn from (G3: window A)."""
        return self.feasible(X)

    def describe(self):
        return {"geometry": self.key, "label": self.label, "theta": self.theta,
                "location": self.location, "shift": self.shift}


class SphereConstraint(ConstraintGeometry):
    """G1 convex: ||x - a|| <= rho.  interior a = c0; boundary a = c0 + (rho-0.01) v
    (ball extends towards the far corner; nearest boundary point c0 - 0.01 v)."""
    key, label = "g1", "convex (ball)"

    def _setup(self):
        self.rho = self.theta
        if self.location == "int":
            self.a = self.c0.copy()
        else:
            self.a = self.c0 + (self.rho - BOUNDARY_MARGIN) * self.v

    def g(self, X):
        X = np.atleast_2d(X)
        return self.rho - np.linalg.norm(X - self.a[None, :], axis=1)

    def describe(self):
        d = super().describe()
        d.update({"rho": self.rho, "center": self.a.tolist()})
        return d


class SineConstraint(ConstraintGeometry):
    """G2 non-convex: mean_i sin(2 pi (x_i - delta_i)) >= tau (paper-1 cgm6 form).

    interior: delta_i = c0_i - 1/4 puts c0 at the sine maximum (g = 1 - tau).
    boundary: delta = delta_int + s * 1 (diagonal shift); s solved so that the
    distance from c0 to the boundary is BOUNDARY_MARGIN (see solve_g2_shift)."""
    key, label = "g2", "non-convex (sine)"
    THETA_BRACKET = (-1.0, 1.0)

    def _setup(self):
        self.tau = self.theta
        self.delta = self.c0 - 0.25 + self.shift

    def g(self, X):
        X = np.atleast_2d(X)
        return np.sin(2.0 * np.pi * (X - self.delta[None, :])).mean(axis=1) - self.tau

    def describe(self):
        d = super().describe()
        d.update({"tau": self.tau, "delta": self.delta.tolist()})
        return d


class TwoBallConstraint(ConstraintGeometry):
    """G3 disconnected: two balls of radius rho in the 3-D coordinate subspace S
    (g3_subspace); the remaining coordinates are free.

    A: ||x_S - a|| <= rho, c0 inside.  B: ||x_S - b|| <= rho, b = far corner of
    the S-cube (fixed => feasible set nested in rho).  u = unit vector from b to
    c0_S.  interior a = c0_S; boundary a = c0_S + (rho - 0.01) u (A moves away
    from B; the nearest boundary point c0_S - 0.01 u faces B).  Warm start: A only."""
    key, label = "g3", "disconnected (two balls, 3-D subspace)"
    THETA_BRACKET = (0.0, 2.0)

    def _setup(self):
        self.S = g3_subspace(self.c0)
        self.rho = self.theta
        cS = self.c0[self.S]
        self.b = np.where(cS < 0.5, 1.0, 0.0)
        self.u = cS - self.b
        self.u = self.u / np.linalg.norm(self.u)
        if self.location == "int":
            self.a = cS.copy()
        else:
            self.a = cS + (self.rho - BOUNDARY_MARGIN) * self.u

    def g_components(self, X):
        XS = np.atleast_2d(X)[:, self.S]
        return (self.rho - np.linalg.norm(XS - self.a[None, :], axis=1),
                self.rho - np.linalg.norm(XS - self.b[None, :], axis=1))

    def g(self, X):
        gA, gB = self.g_components(X)
        return np.maximum(gA, gB)

    def warm_region(self, X):
        return self.g_components(X)[0] >= 0.0

    @property
    def gap(self):
        return float(np.linalg.norm(self.a - self.b) - 2.0 * self.rho)

    def describe(self):
        d = super().describe()
        d.update({"rho": self.rho, "subspace": self.S.tolist(), "center_A": self.a.tolist(),
                  "center_B": self.b.tolist(), "gap": self.gap})
        return d


class BandConstraint(ConstraintGeometry):
    """G4 thin band: |h(x)| <= eps, h(x) = (x_j - q_j) - sum_{i!=j} (x_i - q_i)^2.

    interior q = c0 (c0 on the mid-surface h = 0); boundary q_j = c0_j - (eps - 0.01)
    so the upper surface h = +eps passes through x_j = c0_j + 0.01 (nearest
    point: the gradient there is exactly e_j; off-axis the surface bends to
    larger x_j).  The band then grows towards -x_j as eps grows, so any ratio
    is attainable."""
    key, label = "g4", "thin band (quadratic sheet)"
    THETA_BRACKET = (0.0, 2.0)

    def _setup(self):
        self.eps = self.theta
        self.q = self.c0.copy()
        if self.location == "bnd":
            self.q[self.j] -= self.eps - BOUNDARY_MARGIN
        self._others = [i for i in range(DIM) if i != self.j]

    def h(self, X):
        X = np.atleast_2d(X)
        d = X[:, self._others] - self.q[self._others][None, :]
        return (X[:, self.j] - self.q[self.j]) - G4_CURVATURE * (d ** 2).sum(axis=1)

    def g(self, X):
        return self.eps - np.abs(self.h(X))

    def g_vector(self, X):
        """Two smooth one-sided limits: eps - h >= 0 and eps + h >= 0
        (min of the two = eps - |h| = g)."""
        h = self.h(X)
        return np.stack([self.eps - h, self.eps + h], axis=1)

    def describe(self):
        d = super().describe()
        d.update({"eps": self.eps, "q": self.q.tolist(), "axis_j": self.j})
        return d


GEOMETRIES = {"g1": SphereConstraint, "g2": SineConstraint,
              "g3": TwoBallConstraint, "g4": BandConstraint}


# ---------------------------------------------------------------------------
# Numerical distance from c0 to the constraint boundary (inside the box)
# ---------------------------------------------------------------------------
def boundary_distance(geom, n_dirs=4096, r_max=1.0, n_scan=200, seed=11):
    """Ray-cast estimate of min ||x - c0|| over the boundary {g = 0} within the
    box: random unit directions + axes + diagonals + v + (-)gradient, first
    sign change along each ray (scan) refined by bisection.  Upper bound on
    the true distance; tight for the smooth boundaries used here."""
    c0 = geom.c0
    rng = np.random.RandomState(seed)
    D = rng.randn(n_dirs, DIM)
    eye = np.eye(DIM)
    diag = np.ones(DIM) / np.sqrt(DIM)
    h = 1e-5
    grad = np.array([(geom.g(c0 + h * eye[i])[0] - geom.g(c0 - h * eye[i])[0]) / (2 * h)
                     for i in range(DIM)])
    extra = [eye, -eye, diag[None, :], -diag[None, :], geom.v[None, :], -geom.v[None, :]]
    if np.linalg.norm(grad) > 0:
        extra.append((-grad / np.linalg.norm(grad))[None, :])
    D = np.vstack([D] + extra)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    r = np.linspace(0.0, r_max, n_scan + 1)[1:]
    P = c0[None, None, :] + r[None, :, None] * D[:, None, :]
    inside = np.all((P >= 0.0) & (P <= 1.0), axis=2)
    gv = geom.g(P.reshape(-1, DIM)).reshape(D.shape[0], n_scan)
    cross = (gv < 0.0) & inside
    has = cross.any(axis=1)
    if not has.any():
        return float("inf"), None
    k = np.argmax(cross, axis=1)[has]
    Dh = D[has]
    lo = np.where(k > 0, r[np.maximum(k - 1, 0)], 0.0)
    hi = r[k]
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        gm = geom.g(c0[None, :] + mid[:, None] * Dh)
        neg = gm < 0.0
        hi = np.where(neg, mid, hi)
        lo = np.where(neg, lo, mid)
    dist = 0.5 * (lo + hi)
    i = int(np.argmin(dist))
    return float(dist[i]), Dh[i]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _mc_sample(seed, n):
    return np.random.RandomState(seed).rand(n, DIM)


def calibrate_theta(geom_cls, c0, ratio, location, X, shift=0.0):
    """Bisection on theta so that the MC feasible fraction equals `ratio`
    (feasible sets are nested in theta for every geometry: growing in theta for
    G1/G3/G4, shrinking for G2 where theta = tau)."""
    lo, hi = geom_cls.THETA_BRACKET

    def frac_at(t):
        return float(geom_cls(c0, t, location, shift).feasible(X).mean())

    increasing = frac_at(hi) >= frac_at(lo)
    if not (min(frac_at(lo), frac_at(hi)) <= ratio <= max(frac_at(lo), frac_at(hi))):
        raise ValueError(f"{geom_cls.key} {location}: ratio {ratio} outside the "
                         f"attainable range [{frac_at(lo):.3f}, {frac_at(hi):.3f}]")
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if (frac_at(mid) < ratio) == increasing:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    return theta, frac_at(theta)


def solve_g2_shift(c0, tau, target=BOUNDARY_MARGIN, iters=30):
    """G2 boundary: diagonal shift s in (0, s_b), s_b = arccos(tau)/(2 pi) where
    g(c0) = 0; bisection on the numerical distance-to-boundary d(s) = target."""
    s_b = np.arccos(np.clip(tau, -1.0, 1.0)) / (2.0 * np.pi)
    lo, hi = 0.0, s_b
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d, _ = boundary_distance(SineConstraint(c0, tau, "bnd", shift=mid))
        if d > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_f_star(landscape_prob, geom, X_mc):
    """Constrained optimum value: min over feasible MC points, then Nelder-Mead
    from c0 and from the best MC point with an infeasible/out-of-box penalty."""
    feas = geom.feasible(X_mc)
    f_mc = landscape_prob.evaluate(X_mc[feas])
    best_mc = X_mc[feas][int(np.argmin(f_mc))]
    lb, ub = landscape_prob.lb, landscape_prob.ub

    def obj(x):
        if np.any(x < lb) or np.any(x > ub) or not geom.feasible(x[None, :])[0]:
            return 10.0
        return float(landscape_prob.evaluate(x[None, :])[0])

    cands = [float(f_mc.min())]
    for x0 in (geom.c0, best_mc):
        if obj(x0) < 10.0:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000})
            cands.append(float(res.fun))
    return float(min(cands))


def calibrate_all(verbose=True):
    """Calibrate every geometry x ratio x location; returns the params dict."""
    c0, _, _ = base_landscape()
    v, j = fixed_directions(c0)
    X_cal = _mc_sample(CALIB_SEED, CALIB_N)
    landscape = GaussianMixture6("mid", "cin", dim=DIM)  # same c0 (no feasible_fn)
    assert np.allclose(landscape.c0, c0)
    params = {"meta": {
        "landscape_seed": LANDSCAPE_SEED, "c0": c0.tolist(), "v": v.tolist(), "axis_j": j,
        "calib_seed": CALIB_SEED, "calib_n": CALIB_N, "bisect_iters": BISECT_ITERS,
        "interior_margin_min": INTERIOR_MARGIN_MIN, "boundary_margin": BOUNDARY_MARGIN,
        "g3_subspace": g3_subspace(c0).tolist(), "g4_curvature": G4_CURVATURE,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
        "configs": {}}
    for gkey in GEOMETRY_KEYS:
        cls = GEOMETRIES[gkey]
        for rkey, ratio in RATIOS.items():
            for loc in LOCATIONS:
                name = f"{gkey}_{rkey}_{loc}"
                t0 = time.time()
                shift = 0.0
                if gkey == "g2":
                    # volume is shift-invariant: calibrate tau once, then place
                    theta, frac = calibrate_theta(cls, c0, ratio, "int", X_cal)
                    if loc == "bnd":
                        shift = solve_g2_shift(c0, theta)
                else:
                    theta, frac = calibrate_theta(cls, c0, ratio, loc, X_cal)
                geom = cls(c0, theta, loc, shift)
                dist, _ = boundary_distance(geom)
                f_star = estimate_f_star(landscape, geom, X_cal)
                rec = {"geometry": gkey, "ratio_key": rkey, "ratio_target": ratio,
                       "location": loc, "theta": float(theta), "shift": float(shift),
                       "ratio_calib": frac, "c0_margin": dist,
                       "g_at_c0": float(geom.g(c0[None, :])[0]),
                       "g_std_uniform": float(np.std(geom.g(X_cal))),
                       "f_star": f_star, "f_c0": float(landscape.evaluate(c0[None, :])[0]),
                       "geom": geom.describe()}
                params["configs"][name] = rec
                if verbose:
                    print(f"  {name:14s} theta={theta:.5f} shift={shift:.5f} "
                          f"ratio={frac:.4f} margin={dist:.4f} f*={f_star:.4f} "
                          f"({time.time() - t0:.1f}s)", flush=True)
    return params


def load_params(path=PARAMS_PATH, auto_calibrate=True):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    if not auto_calibrate:
        raise FileNotFoundError(path)
    print(f"[constraint_suite] {path} not found -> calibrating (deterministic)")
    params = calibrate_all()
    with open(path, "w") as fh:
        json.dump(params, fh, indent=1)
    return params


# ---------------------------------------------------------------------------
# Problem class (runner-facing)
# ---------------------------------------------------------------------------
class ConstrainedSuiteProblem(GaussianMixture6):
    """Paper-1 mid-sparsity GM6 landscape + one suite constraint.

    coverage      'cin' (default; paper-1 cgm6 warm start: 85 % uniform-feasible
                  rejection + 15 % local at 30..95 % of the peak, feasible-only)
                  or 'uniform' (pure rejection).
    n_warm        warm-start size (200; H4 variant 50).
    contamination fraction of the warm start replaced by uniformly drawn
                  INFEASIBLE points (H4 variant 0.1 / 0.3).  Their objective
                  value is the analytic f(x) (as if the record kept the
                  measurement); the runner may substitute a penalty value.
                  Flags are returned by warm_start_labeled()."""

    constrained = True

    def __init__(self, gkey, rkey, loc, params=None, coverage="cin",
                 n_warm=WARM_N, contamination=0.0):
        assert gkey in GEOMETRIES and rkey in RATIOS and loc in LOCATIONS
        assert coverage in COVERAGE_MODES
        params = params or load_params()
        base = f"{gkey}_{rkey}_{loc}"
        rec = params["configs"][base]
        suffix = ""
        if contamination > 0:
            suffix += f"_c{int(round(100 * contamination)):02d}"
        if n_warm != WARM_N:
            suffix += f"_w{n_warm}"
        if coverage != "cin":
            suffix += f"_{coverage}"
        super().__init__("mid", "cin", dim=DIM, feasible_fn=None, name=base + suffix)
        assert np.allclose(self.c0, np.asarray(params["meta"]["c0"]))
        self.geom = GEOMETRIES[gkey](self.c0, rec["theta"], loc, rec["shift"])
        self.gkey, self.rkey, self.loc = gkey, rkey, loc
        self.ratio_target = RATIOS[rkey]
        self.coverage_mode = coverage
        self.n_warm = int(n_warm)
        self.contamination = float(contamination)
        self.f_star = float(rec["f_star"])
        self.known_min = self.f_star
        self.config = dict(rec, name=self.name, coverage=coverage, n_warm=self.n_warm,
                           contamination=self.contamination)

    # -- constraint --------------------------------------------------------
    def g(self, X):
        return self.geom.g(X)

    margin = g

    def g_vector(self, X):
        return self.geom.g_vector(X)

    def feasible(self, X):
        return self.geom.feasible(X)

    def _ok(self, x):
        return bool(self.geom.warm_region(x[None, :])[0])

    def _uniform_ok(self, rng, n, region=None):
        """Vectorised rejection sampling from the warm region (or `region`)."""
        region = region or self.geom.warm_region
        out = []
        got = 0
        while got < n:
            X = self.lb + rng.rand(4096, self.dim) * (self.ub - self.lb)
            X = X[region(X)]
            out.append(X)
            got += len(X)
        return np.vstack(out)[:n]

    # -- warm start --------------------------------------------------------
    def warm_start_labeled(self, seed, n=None, contamination=None):
        rng = np.random.RandomState(seed)
        n = self.n_warm if n is None else int(n)
        cont = self.contamination if contamination is None else float(contamination)
        if self.coverage_mode == "cin":
            n_loc = int(round(CIN_LOCAL_FRAC * n))
            X = np.vstack([self._uniform_ok(rng, n - n_loc),
                           self._local_at_value(rng, n_loc, *self.CIN_U)])
        else:
            X = self._uniform_ok(rng, n)
        feas = np.ones(n, dtype=bool)
        k = int(round(cont * n))
        if k > 0:
            idx = rng.choice(n, k, replace=False)
            X[idx] = self._uniform_ok(rng, k, region=lambda Z: ~self.geom.feasible(Z))
            feas[idx] = False
        perm = rng.permutation(n)
        X, feas = X[perm], feas[perm]
        return X, self.evaluate(X), feas

    def warm_start(self, seed):
        X, f, _ = self.warm_start_labeled(seed)
        return X, f


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TIER1 = ([f"{g}_{r}_int" for g in GEOMETRY_KEYS for r in RATIOS]
         + [f"{g}_f020_bnd" for g in GEOMETRY_KEYS])
TIER2 = [f"{g}_{r}_bnd" for g in GEOMETRY_KEYS for r in ("f050", "f005")]
ALL_CONFIGS = TIER1 + TIER2
# H4 (plan 2.2): ratio 0.2, interior, warm-start contamination 10/30 % (4 geometries)
# and warm-start size 50 (G2, G3)
H4_VARIANTS = ([(f"{g}_f020_int", {"contamination": c}) for g in GEOMETRY_KEYS
                for c in (0.1, 0.3)]
               + [(f"{g}_f020_int", {"n_warm": 50}) for g in ("g2", "g3")])


def parse_name(name):
    gkey, rkey, loc = name.split("_")[:3]
    return gkey, rkey, loc


def build_suite(tier="1", params=None, **kw):
    """tier in {'1','2','all','h4'} or an explicit list of base names; kw are
    passed to ConstrainedSuiteProblem (coverage, n_warm, contamination)."""
    params = params or load_params()
    probs = {}
    if tier == "h4":
        for base, opts in H4_VARIANTS:
            p = ConstrainedSuiteProblem(*parse_name(base), params=params, **{**kw, **opts})
            probs[p.name] = p
        return probs
    names = {"1": TIER1, "2": TIER2, "all": ALL_CONFIGS}[tier] if isinstance(tier, str) else list(tier)
    for base in names:
        p = ConstrainedSuiteProblem(*parse_name(base), params=params, **kw)
        probs[p.name] = p
    return probs


# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------
def verify_suite(params=None, seed=VERIFY_SEED, n=VERIFY_N, warm_seed=43, write=True):
    import pandas as pd
    params = params or load_params()
    X = _mc_sample(seed, n)
    rows = []
    for name in ALL_CONFIGS:
        p = ConstrainedSuiteProblem(*parse_name(name), params=params)
        rec = params["configs"][name]
        frac = float(p.feasible(X).mean())
        dist, _ = boundary_distance(p.geom)
        X0, f0, fl = p.warm_start_labeled(warm_seed)
        margin_ok = (dist >= INTERIOR_MARGIN_MIN if p.loc == "int"
                     else abs(dist - BOUNDARY_MARGIN) <= BOUNDARY_MARGIN_TOL)
        row = {"name": name, "tier": 1 if name in TIER1 else 2,
               "geometry": p.gkey, "ratio_target": p.ratio_target, "location": p.loc,
               "theta": rec["theta"], "shift": rec["shift"],
               "ratio_calib": rec["ratio_calib"], "ratio_verify": frac,
               "ratio_ok": abs(frac - p.ratio_target) <= RATIO_TOL,
               "c0_margin": dist, "margin_ok": bool(margin_ok),
               "g_at_c0": rec["g_at_c0"], "g_std_uniform": rec["g_std_uniform"],
               "f_star": p.f_star, "f_c0": rec["f_c0"],
               "warm_all_feasible": bool(p.feasible(X0).all()),
               "warm_f0_min": float(f0.min()), "warm_f0_median": float(np.median(f0)),
               "warm_min_dist_c0": float(np.linalg.norm(X0 - p.c0, axis=1).min()),
               "warm_frac_in_2s0": float((np.linalg.norm(X0 - p.c0, axis=1) < 2 * p.s0).mean()),
               "g3_gap": p.geom.gap if p.gkey == "g3" else np.nan,
               "g3_gap_ok": (p.geom.gap >= G3_GAP_MIN) if p.gkey == "g3" else True}
        rows.append(row)
    df = pd.DataFrame(rows)
    if write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "suite_verification.csv")
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return df


# ---------------------------------------------------------------------------
# 2-D sections through c0 (sanity plot now; Fig 2(a) later)
# ---------------------------------------------------------------------------
def section_plane(prob):
    """Orthonormal plane (origin c0, basis e1, e2) that shows the geometry best.
    G3: e1 along the line from c0 to ball B's centre, e2 perpendicular inside
    the subspace (both balls visible).  G4: e1 = one free axis, e2 = sheet
    normal x_j.  G1/G2: x0, x1 (G1 boundary: x0 + far-corner direction v)."""
    E = np.eye(DIM)
    if prob.gkey == "g3":
        S = prob.geom.S
        e1 = np.zeros(DIM)
        e1[S] = -prob.geom.u                      # c0 -> b
        e2 = np.zeros(DIM)
        e2[S[0]] = 1.0
        e2 -= e2 @ e1 * e1
        return (e1, e2 / np.linalg.norm(e2),
                "distance from $c_0$ toward ball B", "perpendicular (in subspace)")
    if prob.gkey == "g4":
        j = prob.geom.j
        k = 0 if j != 0 else 1
        return E[k], E[j], f"$x_{k + 1}$", f"$x_{j + 1}$"   # 1-indexed in figures
    return E[0], E[1], "$x_1$", "$x_2$"


def section_grid(prob, n=300, half=1.0):
    """Feasibility mask (n x n) and objective values on the plane through c0
    with coordinates s1, s2 in [-half, half]; points outside the box are
    marked -1 in the mask."""
    e1, e2, l1, l2 = section_plane(prob)
    t = np.linspace(-half, half, n)
    A, B = np.meshgrid(t, t)
    P = prob.c0[None, :] + A.ravel()[:, None] * e1[None, :] + B.ravel()[:, None] * e2[None, :]
    inside = np.all((P >= 0.0) & (P <= 1.0), axis=1)
    mask = prob.feasible(P).astype(int)
    mask[~inside] = -1
    F = prob.evaluate(P).reshape(n, n)
    return (e1, e2, l1, l2), t, mask.reshape(n, n), F


GEOMETRY_TITLES = {"g1": "G1  convex", "g2": "G2  non-convex",
                   "g3": "G3  disconnected", "g4": "G4  thin band"}
LOCATION_TITLES = {"int": "interior optimum", "bnd": "boundary optimum"}


def plot_sections(outdir, ratio="f020", warm_seed=43, dpi=300, stem=None):
    """Manuscript-style 2 x 4 panel (rows: interior / boundary, columns: G1..G4):
    plane section through c0, feasible set in green, objective contours in
    grey, warm start projected, c0 as a star.  Fig 2(a) prototype."""
    import figstyle as FS
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 9.8))
    letters = "abcdefgh"
    for row, loc in enumerate(LOCATIONS):
        for col, gkey in enumerate(GEOMETRY_KEYS):
            p = ConstrainedSuiteProblem(gkey, ratio, loc)
            (e1, e2, l1, l2), t, mask, F = section_grid(p)
            X0, _ = p.warm_start(warm_seed)
            # coordinate-aligned planes are drawn in absolute box coordinates;
            # the rotated G3 plane in offsets from c0
            o1 = float(p.c0 @ e1) if np.count_nonzero(e1) == 1 else 0.0
            o2 = float(p.c0 @ e2) if np.count_nonzero(e2) == 1 else 0.0
            t1, t2 = t + o1, t + o2
            s1, s2 = (X0 - p.c0) @ e1 + o1, (X0 - p.c0) @ e2 + o2   # warm start projected
            ax = axes[row, col]
            FS.draw_feasible_mask(ax, t1, t2, mask)
            Fm = np.ma.masked_where(mask < 0, F)
            ax.contour(t1, t2, Fm, levels=np.linspace(-1.0, -0.1, 7), colors="0.55",
                       linewidths=0.7, alpha=0.8, zorder=3)
            ax.scatter(s1, s2, s=9, c=FS.WARM_COLOR, alpha=0.55, lw=0, zorder=4)
            ax.scatter([o1], [o2], **FS.C0_MARKER)
            # axis limits = box extent within this plane
            ins = np.argwhere(mask >= 0)
            lo1, hi1 = t1[ins[:, 1].min()], t1[ins[:, 1].max()]
            lo2, hi2 = t2[ins[:, 0].min()], t2[ins[:, 0].max()]
            pad = 0.0 if (o1 or o2) else 0.04
            ax.set_xlim(lo1 - pad, hi1 + pad)
            ax.set_ylim(lo2 - pad, hi2 + pad)
            ax.set_aspect("equal")
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            if row == 0:
                ax.set_title(GEOMETRY_TITLES[gkey])
            ax.text(0.03, 0.97, f"{LOCATION_TITLES[loc]}\nmargin = {p.config['c0_margin']:.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7",
                              alpha=0.9), zorder=8)
            FS.panel_label(ax, letters[row * 4 + col], dx=-0.14, dy=1.02)
    fig.suptitle(f"Constraint suite: plane sections through $c_0$ "
                 f"(feasible ratio {RATIOS[ratio]:.2f}; other coordinates fixed at $c_0$)",
                 fontsize=16, fontweight="bold", y=1.0)
    fig.legend(handles=FS.legend_handles_constraint()[:3]
               + [plt.Line2D([], [], color="0.55", lw=0.9, label="objective $f$ contours")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), frameon=True,
               edgecolor="0.7", fancybox=True, borderpad=0.7, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return FS.savefig(fig, outdir, stem or f"FigS_suite_sections_{ratio}", dpi=dpi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate", action="store_true",
                    help="(re)compute constraint_suite_params.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing params file")
    ap.add_argument("--verify", action="store_true",
                    help="independent MC verification of every configuration")
    ap.add_argument("--list", action="store_true", help="print tier membership")
    ap.add_argument("--plot", action="store_true",
                    help="manuscript-style plane sections through c0 -> figures/FigS_suite_sections_<ratio>.png/.pdf")
    ap.add_argument("--plot-ratio", default="f020", choices=list(RATIOS))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if args.calibrate:
        if os.path.exists(PARAMS_PATH) and not args.force:
            raise SystemExit(f"{PARAMS_PATH} exists; use --force to overwrite")
        t0 = time.time()
        params = calibrate_all()
        with open(PARAMS_PATH, "w") as fh:
            json.dump(params, fh, indent=1)
        print(f"wrote {PARAMS_PATH} ({time.time() - t0:.0f}s)")
    if args.list:
        print("TIER1:", TIER1)
        print("TIER2:", TIER2)
        print("H4   :", [f"{b} {o}" for b, o in H4_VARIANTS])
    if args.verify:
        import pandas as pd
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 40)
        df = verify_suite()
        cols = ["name", "theta", "shift", "ratio_target", "ratio_verify", "ratio_ok",
                "c0_margin", "margin_ok", "f_star", "warm_f0_min", "warm_min_dist_c0",
                "warm_frac_in_2s0", "g3_gap"]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nratio_ok {df.ratio_ok.sum()}/{len(df)}  margin_ok {df.margin_ok.sum()}/{len(df)}"
              f"  warm_all_feasible {df.warm_all_feasible.sum()}/{len(df)}"
              f"  g3_gap_ok {df.g3_gap_ok.sum()}/{len(df)}")
    if args.plot:
        plot_sections(args.figdir, ratio=args.plot_ratio, dpi=args.dpi)
    if not (args.calibrate or args.verify or args.list or args.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
