# dali-safe-benchmark

Benchmark, code and results for

> G.-N. Ahn, B. K. Kaang, J. Woo, J. Kim, S. Jang, J. S. Myung, G. S. Na,
> "Benchmarking generative experiment proposal under unknown constraints for autonomous materials and process optimization" (submitted, 2026).

The benchmark asks how often an experiment-proposal algorithm proposes infeasible conditions when the constraint is
unknown to it. Every algorithm starts from the same record of 200 successful experiments, proposes 100 more
(20 cycles × 5), and is scored by the number of violations *V*, the feasible regret *r* and the time to first
violation *T*₁. Nine algorithms are compared:

| Class | Methods (name in the result files) |
|---|---|
| record-based | DALI+LE (`DALIv2LE`), DALI (`DALIv2`), LE-only (`LEonly`) |
| constraint-learning | SCBO (`SCBO`) |
| constraint-unaware | TuRBO (`TuRBO`), GA (`GA`), BO (`BO`), Random (`Random`) |
| reference (true constraint given) | cTuRBO(known) (`cTuRBOknown`) |

## Repository structure

```
code/
  run_constraint_suite.py        unified runner (all blocks; job splitting; completed runs are skipped)
  suite_common.py                tracker, CSV schema, seeds
  problems_constraint_suite.py   synthetic constraint suite (G1–G4 × 50/20/5 % × interior/boundary) and variants
  problems_navigation.py         exploratory navigation problems
  feasibility_oracle_real.py     data-density feasibility oracle for the real datasets
  run_scbo.py, run_cturbo_known.py   SCBO and the known-constraint reference
  run_ablations.py               "verifier + uniform" and "DALI (no clip)" controls
  run_dali_gate.py               exploratory kNN gate
  analyze_constraint_suite.py    metrics, tests and summaries (pre-specified analysis plan)
  make_fig*.py, make_si_*.py, figstyle*.py   figure panels and supplementary figures/tables
  constraint_suite_params.json, navigation_params.json, realdata_params.json   frozen parameters
  ml/, util/, problems_synthetic.py, run_synthetic_benchmark.py, run_dali_le.py, run_dali_v2.py,
  run_new_baselines.py, run_so_experiments.py, run_v2_final.py, run_warmstart_sweep.py
                                 unmodified files of the DALI repository (see NOTICE)
  dataset/                       CCPP data file and checksums (see DATA.md)
results/constraint_suite/<block>/  summary files of every block (see DATA.md)
raw/*.zip                        per-cycle and per-proposal CSVs (synthetic blocks and CCPP; unzip at the repository root)
docs/                            pre-specified analysis plan and execution notes
DATA.md  REPRODUCE.md  LICENSE  NOTICE
```

## Installation

Python 3.10 or later.

```
pip install -r code/requirements.txt
```

All runs of the paper were executed on CPU (4 threads per process); no GPU is needed.

## Quick start

```
cd code
python problems_constraint_suite.py --verify                 # re-derive the frozen suite parameters
python run_constraint_suite.py --block tier1 --list          # count the runs of the main test block
python run_constraint_suite.py --block custom --problems g2_f020_int --methods SCBO,DALIv2LE --seeds 43
python analyze_constraint_suite.py --results ../results/constraint_suite/tier1
```

`REPRODUCE.md` lists the command, seeds and expected output of every block and the figure each block feeds.

## Data

See `DATA.md`. The synthetic blocks need no external data. Of the three real datasets, CCPP is included; SFS and
NOMC must be obtained from their original sources.

## Relation to the DALI repository

DALI, LE and the constraint-unaware baselines are the implementations of
https://github.com/gwangnohahn/dali-inverse-design. The runner of this repository imports those files unchanged and
replaces only their tracker at run time, so that every proposal, its feasibility and its margin are logged.

## Citation

```
@article{ahn2026benchmark,
  title   = {Benchmarking generative experiment proposal under unknown constraints for autonomous materials and process optimization},
  author  = {Ahn, Gwang-Noh and Kaang, Byung Kwon and Woo, Junhee and Kim, Jongwoon and Jang, Seunghun and Myung, Jin Suk and Na, Gyoung S.},
  year    = {2026},
  note    = {submitted}
}
```

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`).
