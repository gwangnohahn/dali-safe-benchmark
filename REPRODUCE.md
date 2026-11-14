# Reproducing the paper

All commands are run from `code/`. A run is identified by (method, problem, seed); the runner skips runs whose
CSV already exists, so an interrupted block can be restarted with the same command.

## 0. Quick check (about 10 minutes on two CPU cores)

```
python problems_constraint_suite.py --verify
python run_constraint_suite.py --block custom --problems g2_f020_int --seeds 43 --outdir ../scratch/check
python analyze_constraint_suite.py --results ../scratch/check
```

The first command should end with `ratio_ok 24/24  margin_ok 23/24  warm_all_feasible 24/24  g3_gap_ok 24/24`
(the one margin below target is the G4 5 % configuration described in the paper). The run of the nine methods on
G2 (20 %, interior), seed 43, should give the values of the same run in
`results/constraint_suite/tier1/summary_metrics_per_run.csv`:

| Method | Violations | Final feasible best *f* |
|---|---|---|
| DALIv2LE | 0 | −0.999118 |
| DALIv2 | 0 | −0.895688 |
| LEonly | 0 | −0.997836 |
| SCBO | 0 | −0.99938 |
| TuRBO | 0 | −0.99537 |
| GA | 0 | −0.989277 |
| BO | 94 | −0.999625 |
| Random | 81 | −0.895688 |
| cTuRBOknown | 0 | −0.99963 |

In a clean environment (Python 3.13, torch 2.15, botorch 0.18, gpytorch 1.15, xgboost 3.4, pymoo 0.6.2,
scikit-optimize 0.10.2, numpy 2.5; Linux, CPU only; checked on 2026-11-14) the violation counts of all nine methods
and the final values of DALIv2LE, DALIv2, LEonly, GA, BO and Random were identical to the stored ones; the three
methods built on BoTorch Gaussian processes (SCBO, TuRBO, cTuRBOknown) differed in the sixth decimal of *f*.

## 1. Check the frozen parameters

```
python problems_constraint_suite.py --verify
python feasibility_oracle_real.py --verify        # needs the three dataset files (DATA.md)
```

## 2. Run the blocks

```
python run_constraint_suite.py --block <block> --list                    # run count
python run_constraint_suite.py --block <block> --jobs 16 --job-id 0      # worker 0 of 16; repeat for 1 ... 15
```

Each worker was run with 4 CPU threads (`OMP_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=""`).

| Block | Content | Configurations × methods × seeds | Runs | Reported in |
|---|---|---|---|---|
| `tier1` | main test: 4 geometries × (interior 50 / 20 / 5 %, boundary 20 %) | 16 × 9 × 20 | 2,880 | Figs. 2–5, 8 |
| `tier2` | boundary optimum, feasible fraction 50 / 5 % | 8 × 9 × 10 | 720 | Fig. 6d |
| `h4_contam` | record contamination 10 / 30 % | 8 × 9 × 10 | 720 | Fig. 6b |
| `h4_warm` | record of 50 points (G2, G3) | 2 × 9 × 10 | 180 | Fig. 6c |
| `h3_uniform` | uniform record, interior 20 % (exploratory) | 4 × 6 × 10 | 240 | Figs. 6e,f, 8 |
| `h3_uniform_bnd` | uniform record, boundary 20 % (exploratory) | 4 × 6 × 10 | 240 | Fig. 8 |
| `h3_ablation` | verifier + uniform, DALI (no clip) | 5 × 2 × 10 | 100 | Fig. 5d |
| `gate` | kNN gate (exploratory) | 7 × 2 × 10 | 140 | Supplementary Fig. 7 |
| `realdata` | SFS, NOMC, CCPP (99th-percentile oracle) | 3 × 9 × 10 | 270 | Fig. 7 |
| `realdata_sens` | oracle threshold 95th / 99.9th percentile | 6 × 9 × 10 | 540 | Supplementary Fig. 6 |
| `nav` | navigation problems with and without a hole in the record (exploratory) | 6 × 6 × 10 | 360 | Supplementary Table 4 |
| `dim2` | 2-D constrained Branin | 1 × 9 × 10 | 90 | Supplementary Fig. 8 |
| | | total | 6,480 | |

Seeds: `43, 77, 777, 34, 1004, 0, 1, 2, 3, 7` (all blocks) and, for `tier1` only, also
`11, 13, 17, 19, 23, 29, 31, 37, 41, 47`. The seed fixes both the warm start and the random numbers of the algorithm.

## 3. Analyse

To analyse the stored runs instead of new ones, first unzip the archives of that block in `raw/` at the repository root (the main block is in four parts, `tier1_g1.zip` … `tier1_g4.zip`).

```
python analyze_constraint_suite.py --results ../results/constraint_suite/<block>
```

writes `summary.json`, `summary_metrics_per_run.csv` and `summary_table_S1_stats.csv` into the block folder
(metrics, Mann–Whitney tests with Holm correction, Cliff's δ, Kaplan–Meier and log-rank tests, bootstrap confidence
intervals). The files shipped in `results/` are the ones behind the numbers of the paper.

## 4. Figures and tables

The `make_fig*` scripts compute the plotted values of Figs. 2–8 from the summary files and lay out the panels
(`make_fig*_data.py`, `make_fig*_v2.py` and the slide-layout scripts `make_fig*_pptx.js`); `make_si_figs_v2.py`,
`make_si_extra.py` and `make_si_tables.py` produce the supplementary figures and tables. The published figures are
these panels after manual adjustment of labels and layout in a slide editor; no plotted value was edited.

## Expected values (main test block, mean over 12 interior configurations × 20 seeds)

| Method | Violations per 100 experiments |
|---|---|
| DALI+LE | 0.10 |
| DALI | 0.16 |
| LE-only | 0.13 |
| SCBO | 1.8 |
| TuRBO / GA / BO / Random | 8.0 / 9.1 / 78.9 / 75.0 |
| cTuRBO(known) | 0 |
