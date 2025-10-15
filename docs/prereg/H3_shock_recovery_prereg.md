# Preregistration — H3: Constructive Shocks (<0.5) Improve Outcomes
**Version:** v2  
**Claim:** Shocks with severity < 0.5 improve post-shock survival/efficiency vs. baseline; ≥0.8 degrade. Regrowth effect explicitly ablated.

## 1) Hypothesis
H3a: Post-shock outcome (t=+8..+24) higher than pre-shock baseline for severity<0.5.  
H3b: Outcome lower for severity≥0.8.  
H3c: Regrowth parameter has a measurable effect (or is documented invariant with CIs).

## 2) Outcomes & Window
- Survival fraction / efficiency index; event-study window [-12, +24].

## 3) Data & Design
- Pre-registered bins: {<0.5, ≈0.5, ≥0.8}; seeds fixed; population size fixed.

## 4) Confirmatory Analysis
- Event-study; DiD-style estimation with matched controls.
- Report CIs; include regrowth ablation grid {0.0, 0.05, 0.1, 0.2, 0.4}.

## 5) Diagnostics & Robustness
- Placebo shocks; shuffle timestamps (should fail).

## 6) Decision Criteria
- Directionally consistent CIs vs. baseline; placebo fails.

## 7–8) Deviations & Reproducibility
As in H1; publish artifact tree, seeds, and plots.

