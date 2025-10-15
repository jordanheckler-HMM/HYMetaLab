# Preregistration — H1: Fear × CCI Moderation (Protective Effect)
**Version:** v2  
**Claim (directional):** In city–week panels, higher CCI reduces the marginal effect of fear on aggression (negative Fear×CCI coefficient), controlling for seasonality and city fixed effects.

## 1) Hypothesis
H1: β(Fear×CCI) < 0 in a confirmatory GLM panel with City and Week fixed effects.

## 2) Outcomes
- Primary: Weekly aggression rate per 100k (city–week panel).
- Secondary: Event-level aggression counts normalized by population.

## 3) Data
- Crime, events, search trends (fear terms), inequality proxy/CCI proxy.
- Weekly frequency, YYYY–YYYY window. Version raw files; record SHA256.

## 4) Confirmatory Model
AggressionRate_it ~ β0 + β1Fear_it + β2CCI_it + β3*(Fear_it×CCI_it) + City_i + Week_t + ε_it
- Test: H0: β3 ≥ 0 vs H1: β3 < 0 (one-sided).

## 5) Diagnostics & Robustness
- Placebo pre-trend checks; seasonality controls; seed-locked splits.
- Baselines: ARIMA/Prophet, regularized GLM, gradient boosting.

## 6) Decision Criteria
- Sign: β3 < 0 with 95% CI not crossing 0 and predictive gain > baseline.

## 7) Deviations Policy
Any deviations logged in an Exploratory Addendum with timestamps.

## 8) Reproducibility
- Docker tag + seeds; `scripts/repro_v2.sh` entrypoint.
- Expected outputs: model table, CI plot, out-of-sample metrics.

