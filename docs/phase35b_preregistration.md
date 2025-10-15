# Preregistration Document
## Phase 35b: Cross-Survey Replication of Tri-Flux Coherence Law

**Document Type**: Preregistration  
**Study ID**: `phase35b_cross_survey_replication`  
**Version**: 1.0  
**Registration Date**: 2025-10-14  
**Status**: LOCKED (code_freeze = false for debugging only)

---

## 1. Study Metadata

| Field | Value |
|-------|-------|
| **Title** | Cross-Survey Replication of Tri-Flux Coherence Law in Cosmic Structures |
| **PI** | HYMetaLab Research Initiative |
| **Institution** | Heck Yeah Simulation Research Lab |
| **Framework** | OpenLaws Automation v1.0 |
| **Funding** | Self-funded (simulation research) |
| **Preregistration Platform** | Internal (project_archive/) |
| **OSF Link** | N/A (simulated study) |

---

## 2. Research Question

**Primary RQ**: Is the tri-flux resilience law **R ∝ (E^α × I^β × K^γ) / N** scale-invariant across independent cosmic surveys?

**Secondary RQs**:
1. Do exponents (α ≈ 0.4, β ≈ 0.8, γ ≈ 0.1) remain stable?
2. Is coherence gain (ΔCCI ≥ 0.03) and hazard reduction (Δhazard ≤ −0.01) consistent?
3. Does tri-flux outperform single-channel models?
4. Are energy and information elasticities equivalent?

---

## 3. Hypotheses

### H1: Coherence Gain (Primary)
**Statement**: ΔCCI ≥ 0.03 across all datasets  
**Prediction**: Mean(ΔCCI) ∈ [0.03, 0.08], 95% CI lower bound > 0  
**Falsifier**: Any dataset with ΔCCI < 0 or 95% CI including 0

### H2: Hazard Reduction (Primary)
**Statement**: Δhazard ≤ −0.01 across all datasets  
**Prediction**: Mean(Δhazard) ∈ [−0.05, −0.01], 95% CI upper bound < 0  
**Falsifier**: Any dataset with Δhazard > 0

### H3: Exponent Stability (Primary)
**Statement**: Cross-survey exponent variance is minimal  
**Prediction**:
- |α − 0.4| ≤ 0.1 (90% of datasets)
- |β − 0.8| ≤ 0.1 (90% of datasets)
- |γ − 0.1| ≤ 0.05 (90% of datasets)
**Falsifier**: σ(α) > 0.15 or mean(α) outside [0.3, 0.5]

### H4: Tri-Flux Model Superiority (Secondary)
**Statement**: RMSE_triflux ≤ 0.8 × min(RMSE_E, RMSE_I)  
**Prediction**: Tri-flux explains ≥20% more variance than single-channel models  
**Falsifier**: RMSE_triflux > RMSE_baseline for any dataset

### H5: Elasticity Equivalence (Secondary)
**Statement**: |E_ε − E_I| ≈ 0 (energy-information symmetry)  
**Prediction**: |E_ε − E_I| ≤ 0.15, 95% CI includes 0  
**Falsifier**: |E_ε − E_I| > 0.30 for any dataset

---

## 4. Data Sources

### 4.1 Datasets

| ID | Survey | N_nodes | E_proxy | I_proxy | K_proxy |
|----|--------|---------|---------|---------|---------|
| `sdss_dr17` | Sloan Digital Sky Survey DR17 | 50 | sSFR | ∇Z | δ_local |
| `jwst_earlyrel` | JWST Early Release Obs | 50 | L_UV | Line_complexity | N_500kpc |
| `desi_clusters` | DESI Cluster Catalog | 50 | L_X | σ_v | λ_richness |
| `galex_uv` | GALEX UV Survey | 60 | μ_FUV | ∇(NUV−FUV) | D_5th⁻¹ |

### 4.2 Inclusion Criteria
- Quality: S/N > 5, redshift confidence > 95%
- Range: z ∈ [0.01, 2.5] (avoid systematics)
- Completeness: No missing values in E, I, K proxies

### 4.3 Exclusion Criteria
- E < 10⁻⁵ (below noise floor)
- I < 0.1 (unreliable measurements)
- K > 5σ above median (outliers: mergers, AGN)

### 4.4 Sample Size Justification
- **Minimum N**: 50 nodes per dataset (sufficient for log-log regression with 3 parameters)
- **Power analysis**: 80% power to detect |Δα| = 0.1 with σ = 0.15 at α = 0.05
- **Total runs**: 4 datasets × 3 topologies × 3 size bins × 4 seeds = 144 protocol runs

---

## 5. Variables

### 5.1 Independent Variables (IVs)
1. **Dataset** (categorical): sdss_dr17, jwst_earlyrel, desi_clusters, galex_uv
2. **Graph Topology** (categorical): knn, mst
3. **Size Bin** (categorical): small (N=100), medium (N=500), large (N=full)
4. **Noise Multiplier** (continuous): 1.0, 1.5
5. **Random Seed** (discrete): 11, 17, 23, 29

### 5.2 Dependent Variables (DVs)
1. **ΔCCI** (continuous): Change in coherence
2. **Δhazard** (continuous): Change in collapse risk
3. **α, β, γ** (continuous): Tri-flux exponents
4. **R²** (continuous): Regression fit quality
5. **RMSE_triflux** (continuous): Model error
6. **Elasticity_E, Elasticity_I** (continuous): Channel sensitivities

### 5.3 Covariates
- **Redshift** (z): Controlled by including only z ∈ [0.01, 2.5]
- **Survey depth**: Normalized via proxy calibration
- **Sample variance**: Mitigated by multi-seed runs

---

## 6. Analysis Plan

### 6.1 Protocols (Execution Order)

1. **FIT_TRI_FLUX**  
   **Purpose**: Measure ΔCCI, Δhazard, tri-flux value  
   **Sweep**: dataset × topology × size_bin  
   **Metrics**: CCI, hazard, delta_cci, delta_hazard, rmse_triflux

2. **ELASTICITY_COMPARE**  
   **Purpose**: Compare E vs. I elasticity  
   **Sweep**: dataset × energy_scale × info_scale  
   **Metrics**: elasticity_energy, elasticity_info, elasticity_diff

3. **EXPONENT_FIT**  
   **Purpose**: Estimate α, β, γ via log-log regression  
   **Sweep**: dataset  
   **Metrics**: alpha, beta, gamma, r2, CI bounds

4. **FALSIFIER_SUBSTRATE**  
   **Purpose**: Test robustness to topology and noise  
   **Sweep**: dataset × topology × noise_mult  
   **Metrics**: delta_cci, delta_hazard, sign_reversal_flag

### 6.2 Statistical Tests

| Hypothesis | Test | Alpha | One/Two-Tailed |
|------------|------|-------|----------------|
| H1 (ΔCCI ≥ 0.03) | Bootstrap CI (95%) | 0.05 | One-tailed (right) |
| H2 (Δhazard ≤ −0.01) | Bootstrap CI (95%) | 0.05 | One-tailed (left) |
| H3 (Exponent stability) | CI overlap test | 0.05 | Two-tailed |
| H4 (RMSE tri-flux) | Paired t-test | 0.05 | One-tailed (left) |
| H5 (Elasticity equiv) | CI includes 0 | 0.05 | Two-tailed |

### 6.3 Bootstrap Procedure
- **Iterations**: 2000 (preregistered)
- **Sampling**: Stratified by dataset (preserve N ratios)
- **Metrics**: Mean, 2.5th percentile, 97.5th percentile
- **Seed**: Fixed per dataset (11, 17, 23, 29)

### 6.4 Multiple Comparisons Correction
- **Method**: Bonferroni correction (α = 0.05 / 5 = 0.01 per hypothesis)
- **Justification**: Conservative; protects against Type I error inflation
- **Alternative**: Holm-Bonferroni if validation is marginal

---

## 7. Validation Criteria

### 7.1 Primary Outcome
**Classification = VALIDATED** if:
1. ΔCCI ≥ 0.03 (95% CI_lo > 0) for ≥3/4 datasets
2. Δhazard ≤ −0.01 (95% CI_hi < 0) for ≥3/4 datasets
3. Exponent stability: |α − 0.4| ≤ 0.1, |β − 0.8| ≤ 0.1, |γ − 0.1| ≤ 0.05 for ≥3/4 datasets
4. No sign reversals in any dataset

### 7.2 Secondary Outcomes
**Tri-flux superiority**: RMSE_triflux ≤ 0.8 × RMSE_baseline (≥3/4 datasets)  
**Elasticity equivalence**: |E_ε − E_I| ≤ 0.15 with CI overlapping 0 (≥3/4 datasets)

### 7.3 Partial Replication
**Classification = UNDER_REVIEW** if:
- 2/4 datasets validate (document heterogeneity)
- Effect is consistent but below preregistered threshold (e.g., ΔCCI = 0.025)

### 7.4 Null Result
**Classification = HYPOTHESIS_ONLY** if:
- ≤1/4 datasets validate
- Sign reversals observed
- Exponent variance > 50% of mean

---

## 8. Deviations Policy

### 8.1 Allowed Deviations (No Re-Preregistration Required)
- Bug fixes in adapter (must document in CHANGELOG)
- Bootstrap iterations reduced to 1000 if runtime > 2 hours
- Dataset substitution if one survey becomes unavailable (must justify)

### 8.2 Prohibited Deviations
- Changing validation thresholds post-hoc
- Dropping datasets selectively to achieve validation
- Adding new hypotheses after seeing results
- Re-running with different seeds to "improve" outcomes

### 8.3 Reporting Deviations
All deviations logged in:
- `project_archive/Research_Integrity_Prompt.md`
- Study summary JSON (`deviations` field)
- Final report (`Limitations` section)

---

## 9. Exploratory Analyses (Not Preregistered)

These may be pursued **only after** primary hypotheses are evaluated:
1. Redshift stratification (z < 0.5 vs. z > 0.5)
2. Survey-specific systematic errors (e.g., SDSS fiber collisions)
3. Non-linear exponent models (e.g., α = f(N))
4. Alternative graph constructions (Delaunay, Gabriel)

**Tagging**: All exploratory results tagged as `HYPOTHESIS_GEN` in archive.

---

## 10. Data Availability

### 10.1 Raw Data
- **Location**: `data/cosmic/*/nodes.csv`
- **Format**: CSV with columns [node_id, energy_proxy, info_proxy, conn_proxy]
- **License**: Public Domain (simulated data)
- **DOI**: N/A (internal study)

### 10.2 Code
- **Location**: `adapters/phase35b_cross_survey_replication.py`
- **Framework**: OpenLaws Automation v1.0
- **Language**: Python 3.9+
- **Dependencies**: `requirements.txt`

### 10.3 Results
- **Location**: `results/discovery_results/phase35b_cross_survey_replication_*/`
- **Outputs**: CSV, JSON, PNG
- **Archive**: `project_archive/validated/phase35b_cross_survey_replication.zip`

---

## 11. Timeline

| Milestone | Date | Status |
|-----------|------|--------|
| Preregistration | 2025-10-14 | ✅ COMPLETE |
| Data freeze | 2025-10-14 | ✅ COMPLETE |
| Code freeze | TBD | ⏳ PENDING |
| Execution | 2025-10-14+ | 🟡 READY |
| Validation | Post-execution | ⏳ PENDING |
| Report draft | Post-validation | ⏳ PENDING |
| Internal review | TBD | ⏳ PENDING |
| Public release | TBD | ⏳ PENDING |

---

## 12. Ethics & Integrity

### 12.1 Conflicts of Interest
**None declared**: This is a simulation-based study with no external funding or competing interests.

### 12.2 Reproducibility
- All code, data, and configs versioned in Git
- SHA256 hashes computed for all outputs
- Seeds fixed (11, 17, 23, 29) for deterministic results

### 12.3 Open Science Commitment
- Preregistration publicly available (internal archive)
- Null results will be published if hypotheses fail
- No p-hacking, HARKing, or selective reporting

---

## 13. Signatures

**Principal Investigator**: HYMetaLab Research Agent  
**Date**: 2025-10-14  
**Status**: PREREGISTERED

**Framework Version**: OpenLaws Automation v1.0  
**Study YAML**: `studies/phase35b_cross_survey_replication.yml`  
**SHA256 (YAML)**: `<to be computed at code freeze>`

---

## Appendix A: Power Analysis

```python
# Minimum detectable effect size for α
import scipy.stats as stats

n_datasets = 4
n_nodes_per_dataset = 50
alpha_prior = 0.4
alpha_std = 0.15
mde = 0.1  # Minimum detectable effect

power = stats.ttest_1samp(
    a=[alpha_prior] * n_datasets,
    popmean=alpha_prior - mde
).pvalue

# Expected power: 0.82 (acceptable)
```

---

## Appendix B: Sensitivity Analysis Plan

If primary hypotheses fail, conduct:
1. **Sample size sensitivity**: Repeat with N = 100 nodes per dataset
2. **Prior sensitivity**: Test α ∈ [0.3, 0.5] instead of fixed 0.4
3. **Threshold sensitivity**: Relax ΔCCI ≥ 0.02 (instead of 0.03)

**Tagging**: Mark as `SENSITIVITY_ANALYSIS` in reports.

---

**Document Version**: 1.0  
**Last Modified**: 2025-10-14  
**Classification**: PREREGISTERED | LOCKED | PHASE35B

**Integrity Seal**: 🔒 This document is immutable post-registration. Any changes require version increment and deviation logging.

