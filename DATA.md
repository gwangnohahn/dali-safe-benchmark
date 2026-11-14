# Data

## Synthetic problems

The constraint suite (24 configurations: four feasible-region geometries × three feasible fractions × interior /
near-boundary optimum), its variants and the navigation problems are analytic and need no external data. Their
parameters were computed once and are frozen in

- `code/constraint_suite_params.json` (size parameter, shift and verified feasible fraction of each configuration),
- `code/navigation_params.json` (constraint and reference optimum of the navigation problems),
- `code/realdata_params.json` (reference sets, thresholds and empirical optimum of the real-data oracle).

`python problems_constraint_suite.py --verify` and `python feasibility_oracle_real.py --verify` re-derive the
frozen values.

## Real datasets

The three datasets are those used in Ahn et al., *Digital Discovery* (2026), with the same preprocessing.

| Dataset | Rows | Condition variables | Source | Redistributed here |
|---|---|---|---|---|
| SFS (steel fatigue) | 437 | C, Si, Mn, Ni, Cr | A. Agrawal et al., *Integr. Mater. Manuf. Innov.* 3, 90–108 (2014), doi:10.1186/2193-9772-3-8; underlying data: NIMS MatNavi | no (NIMS MatNavi terms restrict redistribution) |
| NOMC (non-oxidative methane conversion) | 575 | temperature and flow rate of two reactors | S. W. Lee et al., *Chem. Eng. J.* 481, 148286 (2024), doi:10.1016/j.cej.2023.148286 | no (source article grants no redistribution licence) |
| CCPP (combined-cycle power plant) | 9,568 | ambient temperature, vacuum, ambient pressure, relative humidity | P. Tüfekci, *Int. J. Electr. Power Energy Syst.* 60, 126–140 (2014), doi:10.1016/j.ijepes.2014.02.027; UCI Machine Learning Repository | yes (CC BY 4.0) |

File names, layout and SHA-256 checksums are in `code/dataset/README.md`.

## Feasibility on real data

Real data carry no true constraint. The benchmark encodes the rule "a condition outside the recorded operating
region is a failure" as a data-density oracle (mean distance to the five nearest data rows in standardized
coordinates, threshold = 99th percentile of the data's own leave-one-out distances; `code/feasibility_oracle_real.py`).
This oracle regards the neighbourhood of the data as feasible and is therefore structurally favourable to methods
that propose near the record; the paper states this limitation.

## Result files

`results/constraint_suite/<block>/` holds, for every block of the paper,

- `summary.json`, `summary_metrics_per_run.csv`, `summary_table_S1_stats.csv` (written by `analyze_constraint_suite.py`;
  one row per run in `summary_metrics_per_run.csv`),
- `_runlog_job*.jsonl` (execution log of each worker).

`raw/<block>.zip` holds the raw files of 5,940 of the 6,480 runs: all runs of the ten synthetic blocks (5,670) and the
CCPP runs of the two real-data blocks (270; `realdata_ccpp.zip`, `realdata_sens_ccpp.zip`). The main test block is split by geometry into `tier1_g1.zip` … `tier1_g4.zip`. Unzip an archive at the
repository root to restore `results/constraint_suite/<block>/`:

- `<method>_<problem>_seed<seed>.csv` — one row per cycle: best value so far, number of infeasible proposals in
  the cycle, cycle time, size and feasible fraction of the candidate pools;
- `<method>_<problem>_seed<seed>_proposals.csv` — one row per proposal: proposed condition (`x0`, `x1`, …),
  objective value returned to the algorithm (`f_true`), feasibility and constraint margin (`g`).

The summary files can be regenerated from the raw files with `analyze_constraint_suite.py`; the violation counts
and final best values of the raw files agree with `summary_metrics_per_run.csv` for every run.

The raw files of the SFS and NOMC runs (540 runs) are not included, because they are derived from datasets that may
not be redistributed; their per-run metrics are in `summary_metrics_per_run.csv`, and the raw files can be
regenerated with the commands of `REPRODUCE.md` once the two datasets have been obtained from their sources.
