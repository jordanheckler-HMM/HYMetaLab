# Preregistration — H2: Inequality Predicts Collapse; Moderation by CCI
**Version:** v2  
**Claim (threshold):** Collapse probability increases with Gini; CCI raises the threshold. Time-scale alignment is mandatory.

## 1) Hypothesis
H2a: γ1 > 0 in `Pr(Collapse_it=1) = logit^{-1}(γ0 + γ1*Gini_it + γ2*CCI_it + γ3*(Gini×CCI)_it + City_i + Year_t)`.  
H2b (threshold): Odds of collapse rise sharply above Gini ≈ 0.30; CCI shifts this boundary upward.

## 2) Outcomes
- Primary: Collapse flag (city–period), defined ex-ante (>90th percentile for ≥3 consecutive periods).
- Secondary: Time-to-next-collapse.

## 3) Data & Time-Scale
- Annual Gini must be **resampled** to weekly (FFILL) *or* outcomes aggregated to annual before modeling. Document choice.

## 4) Confirmatory Model & Threshold Test
- Logistic with spline or segmented regression around 0.30.
- Interaction term γ3 to test moderation.

## 5) Diagnostics & Robustness
- Out-of-sample AUC vs. baselines; sensitivity to resampling method.

## 6) Decision Criteria
- γ1 > 0 (p<0.05); threshold test significant; interaction γ3 < 0 (protective).

## 7) Deviations Policy & 8) Reproducibility
As in H1; align time scales or fail fast.

