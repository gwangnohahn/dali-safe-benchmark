"""cTuRBO-known: TuRBO with the TRUE constraint g(x) (upper reference, paper 2).

Thin entry point; the implementation lives in run_scbo.run_tr_constrained
(mode="known") so that SCBO and cTuRBO-known share every line of the trust-
region loop and differ only in candidate selection.  Candidates outside the
feasible set are removed before Thompson sampling, so the method never
proposes an infeasible point; V = 0 by construction and its feasible regret is
the best a constraint-aware local method achieves on this landscape.

Usage: python run_cturbo_known.py --problem g3_f005_bnd --seed 43 [--outdir DIR]
"""

import argparse
import time

import numpy as np

from suite_common import write_run, COMMON
from run_scbo import run_cturbo_known_syn

METHODS = {"cTuRBOknown": run_cturbo_known_syn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="g2_f020_int")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    import problems_constraint_suite as CS
    import problems_synthetic as SP
    reg = {**SP.build_problems(), **CS.build_suite("all")}
    prob = reg[args.problem]
    t0 = time.perf_counter()
    rows, proposals = run_cturbo_known_syn(prob, args.seed, args.trials)
    wall = time.perf_counter() - t0
    V = sum(r["n_infeasible"] for r in rows)
    print(f"cTuRBOknown {prob.name} seed {args.seed}: best_f {rows[-1]['best_f']:.4f} "
          f"(f* {getattr(prob, 'f_star', float('nan')):.4f}), violations {V}, "
          f"cand feasible frac (mean) {np.nanmean([r['cand_feasible_frac'] for r in rows]):.3f}, "
          f"{wall:.1f}s")
    if args.outdir:
        write_run(args.outdir, "cTuRBOknown", prob, args.seed, rows, proposals)


if __name__ == "__main__":
    main()
