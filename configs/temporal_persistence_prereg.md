---
title: temporal_persistence_prereg.md
date: 2025-10-16
version: draft
checksum: 9b1a5bfbcfc1
---

# Temporal Persistence - Preregistration Notes

Study goal
---
Measure the temporal persistence of a binary survival-like process F(t) under parameter sweeps.

Primary outcome measures
---
- tau_c: characteristic decay time estimated from exponential fit to F(t)
- t_arrow_var: variance of first-passage times across deterministic seeds

Planned analyses
---
- Fit exponential and stretched exponential to F(t) and report tau_c and R^2.
- Test linear relationship: tau_c ~ epsilon/eta.
- Test linear relationship: t_arrow_var ~ (1 - lambda_est)/epsilon (use lambda_init as proxy).
- Quadratic fit: tau_c ~ a*rho^2 + b*rho + c, report vertex and 95% bootstrap CI.

Hypotheses
---
- "tau_c ∝ epsilon / eta"
- "t_arrow_var ∝ (1 - lambda_est) / epsilon"
- "tau_c peaks near rho≈0.08"

Acceptance criteria
---
- "R2 >= 0.85 for exp fit on >=70% runs"
- "Spearman tau_c vs epsilon/eta >= 0.6 (p<0.01)"
- "Quadratic tau_c(rho) interior maximum in [0.05,0.12]"

Determinism and seeds
---
Deterministic seeds are used and recorded per-run. Bootstrap CIs use the `integrity.bootstrap_n` value (default 800).


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
