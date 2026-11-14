"""Unified runner for the constraint suite (paper 2, W1-4).

Runs the 8 (+1) methods on the 24 suite configurations and the H4 variants
with the paper-1 protocol, writing paper-1-schema CSVs (+ extra columns) with
completed-run skipping and a job-split option for parallel processes.

Methods (CSV names)            implementation                       candidate log
  DALIv2      DALI            run_synthetic_benchmark.run_dali_v2_syn   gen (diffusion, pre-clip)
  DALIv2LE    DALI+LE b=2     run_dali_le.run_dali_le_syn(2, True)      gen + le
  LEonly      LE-only b=5     run_dali_le.run_dali_le_syn(5, False)     le
  TuRBO       TuRBO-1         run_synthetic_benchmark.run_turbo_syn     -
  GA          XGB-surrogate GA run_synthetic_benchmark.run_ga_syn       -
  BO          skopt gp_hedge  run_synthetic_benchmark.run_bo_syn        -
  Random      uniform box     run_synthetic_benchmark.run_random_syn    (proposals = pool)
  SCBO        constraint GP   run_scbo.run_scbo_syn                     gen (TR pool)
  cTuRBOknown true g(x)       run_scbo.run_cturbo_known_syn             gen (TR pool)
Paper-1 files are NOT modified: the runner swaps their EvalTracker for
SuiteTracker at import time (module-global lookup) and installs the
candidate hooks (suite_common.install_candidate_hooks).

Blocks (results/constraint_suite/<block>/):
  tier1      TIER1 (16 configs) x 9 methods x 20 seeds  (DEFAULT_SEEDS + NEW_SEEDS)
  tier2      TIER2 (8)         x 9 x 10 (DEFAULT_SEEDS)
  h4_contam  {g}_f020_int_c10/_c30 (8) x 9 x 10
  h4_warm    g2/g3_f020_int_w50 (2)   x 9 x 10
  dim2       cbranin2 (paper-1 2-D problem) x 9 x 10
  nav        EXPLORATORY: ncamel2/nhart3/nhart6 (+_hole) x 6 methods x 10 seeds = 360
  gate       EXPLORATORY: kNN gate (DALIv2LEgate, LEonlygate) on 7 boundary/interior configs x 10 seeds = 140
  realdata   real_{sfs,nomc,ccpp}_q99 (feasibility_oracle_real) x 9 x 10
  realdata_sens  real_*_q95 / _q999 (threshold sensitivity, SI) x 9 x 10
  h3_ablation  {g}_f020_int x4 + g2_f020_bnd, methods VerifRandom + DALInoclip, 10 seeds (Fig 5(d))
  h3_uniform   {g}_f020_int_uniform x4 (warm start without local points; EXPLORATORY) x 6 methods x 10
  h3_uniform_bnd  {g}_f020_bnd_uniform x4 (uniform record AND optimum at the boundary; EXPLORATORY) x 6 x 10
  custom     --problems/--methods/--seeds given explicitly
(9 methods = the 8 compared + cTuRBOknown reference.)

Outputs per run: <method>_<problem>_seed<seed>.csv
  method, problem, seed, iteration, best_value, best_f, n_infeasible,
  cycle_time_sec, cand_n, cand_feasible_frac, cand_le_n, cand_le_feasible_frac
and <...>_proposals.csv: iteration, x0..x5, f_true, feasible, g.
Existing run CSVs are skipped (safe to re-launch / resume).

Usage:
  python run_constraint_suite.py --block tier1 --list                  # count runs
  python run_constraint_suite.py --block tier1 --jobs 16 --job-id 3    # one of 16 workers
  python run_constraint_suite.py --block custom --problems g2_f020_int --methods SCBO,DALIv2 \
         --seeds 43 --trials 2 --outdir scratch/smoke
Environment: OMP/OPENBLAS/MKL/TORCH_NUM_THREADS=4, CUDA_VISIBLE_DEVICES="" (server CLAUDE.md).
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import suite_common as SC
from suite_common import SuiteTracker, write_run, run_exists, DEFAULT_SEEDS, NEW_SEEDS, COMMON

SC.apply_runtime_patches()      # SuiteTracker swap, candidate hooks, paper-1 margins (cbranin2/cgm6 g),
                                # clipped BO scaler (boundary rows of real datasets)

import run_synthetic_benchmark as RB
import run_dali_le as LE
import run_scbo as SCBO
import run_ablations as AB
import run_dali_gate as GT
import problems_synthetic as SP
import problems_constraint_suite as CS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite")

METHODS = {
    "DALIv2": RB.run_dali_v2_syn,
    "DALIv2LE": lambda p, s, t: LE.run_dali_le_syn(p, s, t, b_local=2, use_diffusion=True),
    "LEonly": lambda p, s, t: LE.run_dali_le_syn(p, s, t, b_local=5, use_diffusion=False),
    "TuRBO": RB.run_turbo_syn,
    "GA": RB.run_ga_syn,
    "BO": RB.run_bo_syn,
    "Random": RB.run_random_syn,
    "SCBO": SCBO.run_scbo_syn,
    "cTuRBOknown": SCBO.run_cturbo_known_syn,
    # H3 ablations (plan section 1 / Fig 5(d)): diffusion replaced by uniform
    # candidates (verifier only), and DALI without clipping to the box
    "VerifRandom": AB.run_verif_random_syn,
    "DALInoclip": AB.run_dali_noclip_syn,
    # EXPLORATORY mitigation (not pre-registered): kNN in-distribution gate on
    # both candidate pools (run_dali_gate.py)
    "DALIv2LEgate": GT.run_dali_le_gate_syn,
    "LEonlygate": GT.run_le_only_gate_syn,
}
METHOD_ORDER = ["DALIv2", "DALIv2LE", "LEonly", "TuRBO", "GA", "BO", "Random",
                "SCBO", "cTuRBOknown"]
ALL_SEEDS = DEFAULT_SEEDS + NEW_SEEDS


REAL_MAIN = ["real_sfs_q99", "real_nomc_q99", "real_ccpp_q99"]
REAL_SENS = [f"real_{d}_{q}" for d in ("sfs", "nomc", "ccpp") for q in ("q95", "q999")]


def build_registry(problems=()):
    reg = dict(SP.build_problems())          # paper-1 problems (cbranin2 for dim2)
    reg.update(CS.build_suite("all"))
    reg.update(CS.build_suite("h4"))
    if any(p.endswith("_uniform") for p in problems):
        reg.update(CS.build_suite([p[:-len("_uniform")] for p in problems if p.endswith("_uniform")],
                                  coverage="uniform"))
    if any(p.startswith("n") and p.split("_")[0] in ("ncamel2", "nhart3", "nhart6") for p in problems):
        import problems_navigation as NAV      # EXPLORATORY navigation block (not pre-registered)
        reg.update(NAV.build_nav_problems())
    if any(p.startswith("real_") for p in problems):
        import feasibility_oracle_real as FR   # loads the xlsx files + fits the XGB oracles
        qk = sorted({p.split("_")[-1] for p in problems if p.startswith("real_")})
        reg.update(FR.build_real_problems(q_keys=tuple(qk)))
    return reg


BLOCKS = {
    "tier1": dict(problems=CS.TIER1, seeds=ALL_SEEDS),
    "tier2": dict(problems=CS.TIER2, seeds=DEFAULT_SEEDS),
    "h4_contam": dict(problems=[f"{g}_f020_int_c{c}" for g in CS.GEOMETRY_KEYS for c in (10, 30)],
                      seeds=DEFAULT_SEEDS),
    "h4_warm": dict(problems=[f"{g}_f020_int_w50" for g in ("g2", "g3")], seeds=DEFAULT_SEEDS),
    "dim2": dict(problems=["cbranin2"], seeds=DEFAULT_SEEDS),
    "realdata": dict(problems=REAL_MAIN, seeds=DEFAULT_SEEDS),
    "realdata_sens": dict(problems=REAL_SENS, seeds=DEFAULT_SEEDS),
    # H3 ablations (pre-registered, Fig 5(d)): interior f020 x 4 geometries + the
    # failing boundary case; methods VerifRandom, DALInoclip
    "h3_ablation": dict(problems=[f"{g}_f020_int" for g in CS.GEOMETRY_KEYS] + ["g2_f020_bnd"],
                        seeds=DEFAULT_SEEDS, methods=["VerifRandom", "DALInoclip"]),
    # EXPLORATORY (not pre-registered): warm start WITHOUT the 15 % local points
    # (coverage="uniform") -- does diffusion navigation matter when the record
    # does not already contain the optimum's basin?
    "h3_uniform": dict(problems=[f"{g}_f020_int_uniform" for g in CS.GEOMETRY_KEYS],
                       seeds=DEFAULT_SEEDS,
                       methods=["DALIv2LE", "LEonly", "DALIv2", "TuRBO", "SCBO", "cTuRBOknown"]),
    # EXPLORATORY (not pre-registered): the fourth cell of the applicability map
    # (Fig 1(b)) -- record spread uniformly AND optimum at the hidden boundary.
    # 4 geometries x 6 methods x 10 seeds = 240 runs
    "h3_uniform_bnd": dict(problems=[f"{g}_f020_bnd_uniform" for g in CS.GEOMETRY_KEYS],
                           seeds=DEFAULT_SEEDS,
                           methods=["DALIv2LE", "LEonly", "DALIv2", "TuRBO", "SCBO", "cTuRBOknown"]),
    # EXPLORATORY (not pre-registered): navigation value of the diffusion generator.
    # Constrained multimodal 2/3/6-D problems with a record spread over the
    # feasible set (paper-1 cbranin2 regime) and a "_hole" variant whose record
    # never visited the optimum's neighbourhood.  See problems_navigation.py.
    # EXPLORATORY (not pre-registered): does a kNN in-distribution gate repair the
    # G2 x boundary failure without hurting the interior?  7 configs x 2 methods x 10 seeds = 140 runs
    "gate": dict(problems=["g2_f020_bnd", "g2_f050_bnd", "g2_f005_bnd", "g1_f020_bnd", "g3_f020_bnd", "g4_f020_bnd",
                           "g2_f020_int"],
                 seeds=DEFAULT_SEEDS, methods=["DALIv2LEgate", "LEonlygate"]),
    "nav": dict(problems=["ncamel2", "nhart3", "nhart6", "ncamel2_hole", "nhart3_hole", "nhart6_hole"],
                seeds=DEFAULT_SEEDS,
                methods=["DALIv2LE", "LEonly", "DALIv2", "TuRBO", "SCBO", "cTuRBOknown"]),
}


def run_list(problems, methods, seeds):
    """Ordered (problem, seed, method) triples: problem-major, then seed, then
    method, so that partial progress stays balanced across configurations."""
    return [(p, s, m) for p in problems for s in seeds for m in methods]


def run_one(reg, method, pname, seed, trials, outdir, log_fh=None, fail_path=None):
    prob = reg[pname]
    tag = f"{method}_{pname}_seed{seed}"
    if run_exists(outdir, method, pname, seed):
        print(f"[skip] {tag} (exists)", flush=True)
        return "skip"
    print(f"[run ] {tag} ...", flush=True)
    t0 = time.perf_counter()
    try:
        rows, proposals = METHODS[method](prob, seed, trials)
    except Exception:
        msg = f"[FAIL] {tag}\n{traceback.format_exc()}"
        print(msg, flush=True)
        if fail_path:   # tracebacks go to a separate file so the JSONL log stays parseable
            with open(fail_path, "a") as ff:
                ff.write(msg + "\n")
        if log_fh:
            log_fh.write(json.dumps({"tag": tag, "status": "fail"}) + "\n")
            log_fh.flush()
        return "fail"
    wall = time.perf_counter() - t0
    write_run(outdir, method, prob, seed, rows, proposals, wall=wall)
    if log_fh:
        log_fh.write(json.dumps({"tag": tag, "wall_sec": round(wall, 1),
                                 "best_f": rows[-1]["best_f"],
                                 "V": int(sum(r["n_infeasible"] for r in rows)),
                                 "cand_feas": float(np.nanmean([r["cand_feasible_frac"] for r in rows]))
                                 if not all(np.isnan(r["cand_feasible_frac"]) for r in rows) else None})
                     + "\n")
        log_fh.flush()
    return "done"


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", default="custom", choices=list(BLOCKS) + ["custom"])
    ap.add_argument("--problems", default="", help="comma list (custom block, or override)")
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    ap.add_argument("--seeds", default="", help="comma list (override block seeds)")
    ap.add_argument("--trials", type=int, default=COMMON["trials"])
    ap.add_argument("--outdir", default=None,
                    help="default results/constraint_suite/<block> (custom: results/constraint_suite/custom)")
    ap.add_argument("--jobs", type=int, default=1, help="split the run list over N workers")
    ap.add_argument("--job-id", type=int, default=0, help="this worker's index 0..N-1")
    ap.add_argument("--list", action="store_true", help="print the run list summary and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the runs this worker would do")
    args = ap.parse_args()

    if args.block == "custom":
        problems = [p for p in args.problems.split(",") if p]
        seeds = [int(s) for s in args.seeds.split(",") if s] or DEFAULT_SEEDS
    else:
        blk = BLOCKS[args.block]
        problems = [p for p in args.problems.split(",") if p] or list(blk["problems"])
        seeds = [int(s) for s in args.seeds.split(",") if s] or list(blk["seeds"])
    methods = [m for m in args.methods.split(",") if m]
    if args.block != "custom" and "methods" in BLOCKS[args.block] and args.methods == ",".join(METHOD_ORDER):
        methods = list(BLOCKS[args.block]["methods"])   # block-specific default method list
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")
    outdir = args.outdir or os.path.join(RESULTS_ROOT, args.block)

    runs = run_list(problems, methods, seeds)
    mine = runs[args.job_id::args.jobs]
    n_done = sum(run_exists(outdir, m, p, s) for p, s, m in runs)
    print(f"block={args.block} problems={len(problems)} methods={len(methods)} seeds={len(seeds)} "
          f"-> {len(runs)} runs ({n_done} already done); worker {args.job_id}/{args.jobs} "
          f"takes {len(mine)}; outdir={outdir}", flush=True)
    if args.list:
        return
    if args.dry_run:
        for p, s, m in mine:
            print(("done " if run_exists(outdir, m, p, s) else "todo ") + f"{m}_{p}_seed{s}")
        return

    reg = build_registry(problems)
    missing = [p for p in problems if p not in reg]
    if missing:
        raise SystemExit(f"unknown problems: {missing}")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"_runlog_job{args.job_id}.jsonl")
    fail_path = os.path.join(outdir, f"_failures_job{args.job_id}.log")
    with open(log_path, "a") as log_fh:
        counts = {"done": 0, "skip": 0, "fail": 0}
        for p, s, m in mine:
            counts[run_one(reg, m, p, s, args.trials, outdir, log_fh, fail_path)] += 1
    print(f"worker {args.job_id}: {counts}", flush=True)


if __name__ == "__main__":
    main()
