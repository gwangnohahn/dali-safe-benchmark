# Pre-specified analysis plan (v3, final) — English translation of Sections 1–3

Written 2026-09-09, before the runs. The Korean original (`analysis_plan_v3_final_ko.md`) is authoritative; this
file is a translation of its Sections 1–3, the parts that were fixed before execution (hypotheses, benchmark
design, metrics and tests). Sections 0 and 4–9 of the original (schedule, compute estimate, writing plan and
other project-management notes) are not part of the analysis plan and are not reproduced. "Ahn et al. (2026a)"
is the DALI paper (*Digital Discovery*, 2026). Analyses added after execution are labelled "exploratory" in the
paper, and departures from this plan are listed in Supplementary Note 7 of the paper.

---

## 1. Claim and hypotheses

**Problem setting.** Nobody knows the constraint function. The record of past experiments (the warm start) consists
almost only of feasible points (failed experiments are rarely recorded). An infeasible proposal is a failed
experiment that only consumes budget.

**Claim.** A conditional diffusion generator trained on the record learns the feasible region implicitly, so that
even without knowing the constraint its proposals stay inside the feasible region **from the first experiment**,
and this property is kept when local refinement (LE) is attached, so that the loss of performance is small.

**Position in the literature.** Unknown/crash-constraint BO (Gelbart 2014; Bachoc 2020), safe BO (SafeOpt, Sui 2015;
Berkenkamp), constraint-learning BO (SCBO, Eriksson & Poloczek 2021), diffusion-based optimization (Diffusion-BBO,
Chen 2026). Existing methods learn the constraint by experiencing violations, or move conservatively under strong
assumptions. DALI reads the safe region directly from the record.

**Hypotheses fixed in advance**

| Hypothesis | Content | Rejection condition |
|---|---|---|
| H1 compliance | The cumulative number of violations V of DALI+LE is significantly lower than that of the constraint-unaware baselines (TuRBO, GA, BO, RS) | Fewer than 12 of the 16 Tier 1 configurations with a significant win over all four baselines |
| H1b early safety | The number of evaluations to the first violation, T₁, is longer than for every baseline including SCBO | Pooled survival curves: log-rank test not significant (vs SCBO) |
| H2 cost of safety | The feasible regret of DALI+LE is close to that of the strongest baseline, and the regret per avoided violation is lower than for SCBO | The final feasible best of DALI+LE is not significantly better than the best warm-start value |
| H3 mechanism | Compliance comes from in-distribution generation. When the diffusion generator is removed (LE-only, verifier + random) compliance disappears, and clipping and the verifier do not explain compliance | V of LE-only ≈ V of DALI+LE (not significant in 3 or more of the 4 representative configurations) |
| H4 limits | Compliance or performance degrades with a boundary optimum, a contaminated warm start and a disconnected set; the degree is quantified | (not a rejection test) |
| H5 real data | Under a data-derived hidden feasibility, the ordering of H1 and H2 holds on three process datasets | H1 fails on 2 or more of the 3 |

H4 is the section in which we show the failing conditions ourselves, keeping the framing of Ahn et al. (2026a).

---

## 2. Benchmark design

### 2.1 Synthetic constraint suite (main experiment)

Base landscape: the 6-D Gaussian mixture of track B of Ahn et al. (2026a) (c₀ fixed, seven decoys, s₀ = 0.08) is
reused, with a constraint g(x) ≥ 0 imposed. The hidden constraint, the death penalty
(f_pen = max(f_warm) + 0.1·range) and the feasible-only warm start (200 points by rejection sampling) are the same
as in track C of Ahn et al. (2026a).

Factor 1 — constraint geometry (four types)

| Symbol | Geometry | Definition | Purpose |
|---|---|---|---|
| G1 | convex | ball ‖x − a‖ ≤ ρ | the easiest case |
| G2 | non-convex | (1/6)Σ sin(2πxᵢ) ≥ τ (cgm6 of Ahn et al. (2026a)) | continuity with Ahn et al. (2026a) |
| G3 | disconnected | union of two balls, c₀ in one of them only | the warm start covers one piece only |
| G4 | thin band | \|h(x)\| ≤ ε (h quadratic) | narrow operating window, both sides of the boundary are dangerous |

Factor 2 — feasible fraction (three levels): 0.5 / 0.2 / 0.05. ρ, τ, ε are adjusted and checked by Monte Carlo
(2 × 10⁵ points) to within ±0.02.

Factor 3 — location of the optimum (two levels): interior (margin of c₀ ≥ 0.1·range) / boundary (margin ≈ 0.01).
c₀ is fixed and the constraint is translated.

- **Tier 1 (required, 16 configurations):** 4 geometries × 3 fractions × interior (12) + 4 geometries × fraction 0.2 × boundary (4).
- **Tier 2 (optional, 8 configurations):** boundary × fraction 0.5 / 0.05.

### 2.2 Limit conditions (H4)

Fraction 0.2, interior, four geometries:

- warm-start contamination 0 / 10 / 30 % (replacement by infeasible points). The default has no labels (everything
  is used for training). The condition with labels is a second priority (training on feasible points only vs
  conditioning on a flag).
- warm-start size 200 / 50 (G2 and G3 only).

### 2.3 Dimension check

2-D (cbranin2 of Ahn et al. (2026a), unchanged) and 10-D (Levy 10-D + G1, G2), as spot checks.

### 2.4 Real process datasets (H5)

SFS, NOMC, CCPP. Hidden feasibility oracle: after standardizing the whole dataset, a condition is infeasible when
its kNN₅ distance exceeds the 99th percentile of held-out data (this encodes "outside the recorded operating
region is a failure"). Sensitivity to the threshold (95 / 99 / 99.9 %) goes to the Supplementary Information. The
warm start is drawn from the data and is therefore feasible automatically. The baselines of Ahn et al. (2026a)
were run without a penalty, so **all are rerun**.

### 2.5 Methods compared (eight)

| Method | Constraint information | Role |
|---|---|---|
| DALI (configuration of Ahn et al. (2026a)) | none | proposed method |
| DALI+LE (b_local = 2) | none | proposed method (with refinement) |
| LE-only | none | control without the diffusion generator |
| TuRBO / GA / BO (gp_hedge) / RS | none (penalty values only) | constraint-unaware baselines |
| SCBO | observes the constraint margin m(x) after each evaluation and fits a constraint GP | constraint-learning baseline. DALI does not see m(x), so the comparison favours SCBO |
| cTuRBO-known | knows the constraint function completely | upper-bound reference |

Implementation: SCBO wraps the official BoTorch implementation. The LE family uses `run_dali_le.py` unchanged.
Every method logs candidate coordinates, the feasibility flag and the margin by default.

---

## 3. Metrics and statistics (fixed in advance)

### 3.1 Metrics

| Metric | Definition | Hypothesis |
|---|---|---|
| V | cumulative number of infeasible proposals in 100 evaluations (primary metric) | H1 |
| T₁ | number of evaluations to the first violation (no violation in 100 = right-censored) | H1b |
| f_best^feas, r | best objective value among feasible proposals; feasible simple regret = f_best^feas − f* | H2 |
| safety cost | (r_method − r_bestbaseline) / (V_bestbaseline − V_method) | H2 |
| margin distribution | m(x) of the proposals | H3, H4 |
| d_kNN | kNN₅ distance from a proposal to the warm start (standardized) | H3 |
| candidate-stage feasible fraction | fraction of feasible candidates among the 200 candidates right after generation (before filtering and clipping) | H3 |

### 3.2 Tests

- Seeds: the 10 of Ahn et al. (2026a) (43, 77, 777, 34, 1004, 0, 1, 2, 3, 7) + 10 new ones (11, 13, 17, 19, 23, 29,
  31, 37, 41, 47). Tier 1 uses 20, the rest 10.
- Per configuration: two-sided Mann–Whitney U test of DALI+LE against each baseline, Holm correction within the
  configuration, Cliff's δ reported alongside, bootstrap 95 % confidence interval of the mean.
- T₁: Kaplan–Meier + log-rank (per configuration and pooled).
- Grid level (H1): number of significant wins among the 16 configurations + sign test. A regression summary
  (V ~ method + geometry + fraction) is auxiliary.
- H3: DALI+LE vs LE-only vs verifier + random, by the same procedure.
- All results (p, δ, confidence intervals) are released in full as Table S1 of the Supplementary Information.

### 3.3 Chain of evidence (order of figures = order of the argument)

1. Figs 2 and 3 — constraint-unaware methods violate often and the DALI family rarely (H1). The performance gap is small and the cost of safety is low (H2).
2. Fig 4 — SCBO becomes safe after experiencing violations, whereas DALI is safe from the start (H1b).
3. Fig 5 — removing the diffusion generator removes compliance, and the feasible fraction of candidates is already high right after generation (H3).
4. Fig 6 — how far the property degrades with a boundary optimum, contamination and a disconnected set (H4) → conditions of applicability.
5. Fig 7 — reproduction on real data (H5).
