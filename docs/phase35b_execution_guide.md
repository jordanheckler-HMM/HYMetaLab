# Phase 35b Execution Guide
## Cross-Survey Replication of Tri-Flux Coherence Law

**Study ID**: `phase35b_cross_survey_replication`  
**Version**: 1.0  
**Preregistration Date**: 2025-10-14  
**Status**: READY FOR EXECUTION

---

## Overview

This study replicates the tri-flux resilience law **R ∝ (E^α × I^β × K^γ) / N** across four independent cosmic datasets to test scale-invariance and universality of coherence laws.

### Research Questions
1. Do tri-flux exponents (α, β, γ) remain stable across surveys?
2. Is ΔCCI ≥ 0.03 and Δhazard ≤ −0.01 consistently observed?
3. Does the tri-flux model outperform single-channel baselines?
4. Are energy and information elasticities equivalent?

---

## Prerequisites

### Dependencies
```bash
pip install pyyaml pandas numpy matplotlib scipy scikit-learn
```

### File Structure
```
conciousness_proxy_sim/
├── studies/
│   └── phase35b_cross_survey_replication.yml
├── adapters/
│   └── phase35b_cross_survey_replication.py
├── data/cosmic/
│   ├── sdss_dr17/nodes.csv
│   ├── jwst_earlyrel/nodes.csv
│   ├── desi_clusters/nodes.csv
│   └── galex_uv/nodes.csv
└── openlaws_automation.py
```

---

## Execution Steps

### 1. Verify Data Integrity
```bash
cd /Users/jordanheckler/conciousness_proxy_sim\ copy\ 6

# Check dataset presence
ls -lh data/cosmic/*/nodes.csv

# Quick validation
python -c "
import pandas as pd
from pathlib import Path

datasets = ['sdss_dr17', 'jwst_earlyrel', 'desi_clusters', 'galex_uv']
for ds in datasets:
    df = pd.read_csv(f'data/cosmic/{ds}/nodes.csv')
    print(f'{ds}: {len(df)} nodes')
    assert all(col in df.columns for col in ['node_id', 'energy_proxy', 'info_proxy', 'conn_proxy'])
print('✓ All datasets valid')
"
```

### 2. Run Study
```bash
# Execute full study with all protocols
python openlaws_automation.py run --study studies/phase35b_cross_survey_replication.yml
```

**Expected Runtime**: ~15–30 minutes (depends on dataset size and bootstrap iterations)

**Output Location**: `results/discovery_results/phase35b_cross_survey_replication_<timestamp>/`

### 3. Validate Results
```bash
# Apply bootstrap CI and classify results
python openlaws_automation.py validate \
  --study studies/phase35b_cross_survey_replication.yml \
  --bootstrap 2000
```

**Validation Criteria**:
- ✅ ΔCCI ≥ 0.03 (95% CI lower bound > 0)
- ✅ Δhazard ≤ −0.01 (95% CI upper bound < 0)
- ✅ Exponents: |α − 0.4| ≤ 0.1, |β − 0.8| ≤ 0.1, |γ − 0.1| ≤ 0.05
- ✅ No sign reversals in any dataset

### 4. Generate Report
```bash
# Create figures and summary
python openlaws_automation.py report --study studies/phase35b_cross_survey_replication.yml
```

**Outputs**:
- `cci_by_dataset.png` — CCI comparison across surveys
- `exponent_forest_plot.png` — α, β, γ with 95% CI
- `triflux_fit_quality.png` — RMSE tri-flux vs. baselines
- `elasticity_comparison.png` — E vs. I elasticity
- `validated_abstract_<timestamp>.md` — Summary report

---

## Alternative: Auto Mode (Run → Validate → Report)

```bash
python openlaws_automation.py auto --study studies/phase35b_cross_survey_replication.yml
```

This executes all steps sequentially and only generates the report if validation passes.

---

## Expected Outputs

### Primary CSV: `runs_summary.csv`

| seed | protocol          | dataset       | graph_topology | CCI   | hazard | delta_cci | delta_hazard | alpha | beta | gamma | r2   |
|------|-------------------|---------------|----------------|-------|--------|-----------|--------------|-------|------|-------|------|
| 11   | FIT_TRI_FLUX      | sdss_dr17     | knn            | 0.62  | 0.38   | 0.032     | −0.012       | —     | —    | —     | —    |
| 11   | EXPONENT_FIT      | jwst_earlyrel | —              | —     | —      | —         | —            | 0.41  | 0.79 | 0.09  | 0.87 |
| 17   | ELASTICITY_COMPARE| desi_clusters | knn            | —     | —      | —         | —            | —     | —    | —     | —    |
| ...  | ...               | ...           | ...            | ...   | ...    | ...       | ...          | ...   | ...  | ...   | ...  |

### Summary JSON: `summary.json`

```json
{
  "study_id": "phase35b_cross_survey_replication",
  "seed": 11,
  "total_runs": 240,
  "mean_CCI": 0.628,
  "mean_delta_cci": 0.034,
  "mean_delta_hazard": −0.013,
  "exponents": {
    "alpha": {"mean": 0.39, "ci_lo": 0.32, "ci_hi": 0.46},
    "beta": {"mean": 0.81, "ci_lo": 0.73, "ci_hi": 0.89},
    "gamma": {"mean": 0.10, "ci_lo": 0.07, "ci_hi": 0.13}
  },
  "classification": "validated",
  "hypothesis_test": {
    "H1_delta_cci": "PASS",
    "H2_delta_hazard": "PASS",
    "H3_exponent_stability": "PASS",
    "H4_triflux_rmse": "PASS",
    "H5_elasticity_equiv": "PASS"
  }
}
```

---

## Troubleshooting

### Issue: Dataset not found
**Error**: `FileNotFoundError: Dataset not found: data/cosmic/.../nodes.csv`

**Fix**:
```bash
# Verify paths in YAML match filesystem
ls -la data/cosmic/

# Update YAML if needed
vim studies/phase35b_cross_survey_replication.yml
```

### Issue: Adapter import failure
**Error**: `ModuleNotFoundError: No module named 'adapters.phase35b_cross_survey_replication'`

**Fix**:
```bash
# Verify adapter exists
ls -lh adapters/phase35b_cross_survey_replication.py

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Users/jordanheckler/conciousness_proxy_sim copy 6"
```

### Issue: Bootstrap CI takes too long
**Error**: Validation hangs at bootstrap step

**Fix**:
```bash
# Reduce bootstrap iterations (trade-off: wider CIs)
python openlaws_automation.py validate \
  --study studies/phase35b_cross_survey_replication.yml \
  --bootstrap 500  # Instead of 2000
```

### Issue: Validation fails (ΔCCI < 0.03)
**Status**: UNDER_REVIEW classification

**Action**:
1. Check if effect is real or statistical noise (review CI)
2. Inspect per-dataset breakdown (some may pass, others fail)
3. Run sensitivity analysis (vary graph topology, size bins)
4. Document as "partial replication" if 2/4 surveys validate

---

## Interpretation Guidelines

### Classification Outcomes

| Classification | Meaning | Next Steps |
|----------------|---------|------------|
| **VALIDATED** | All hypotheses pass | ✅ Proceed to publication draft |
| **UNDER_REVIEW** | Mixed results | 🔍 Analyze heterogeneity, run Phase 35c |
| **HYPOTHESIS_ONLY** | Effect not observed | 🔬 Refine model or explore boundary conditions |

### Effect Sizes

- **ΔCCI ≥ 0.03**: ~5–6% coherence improvement (small but consistent)
- **Δhazard ≤ −0.01**: ~4% collapse risk reduction
- **α ≈ 0.4**: Energy contributes sub-linearly (diminishing returns)
- **β ≈ 0.8**: Information dominates (nearly linear)
- **γ ≈ 0.1**: Connection is weak but stabilizing

### Cross-Survey Variance

If exponent standard deviation σ(α) > 0.05 across surveys:
- **Possible causes**: Selection effects, redshift evolution, proxy calibration
- **Solution**: Stratify by z-bin or survey characteristics

---

## Publication Checklist

- [ ] All datasets properly cited
- [ ] Preregistration timestamp in YAML
- [ ] Validation criteria met (or deviations explained)
- [ ] Bootstrap CIs included for all exponents
- [ ] Figures labeled with units and error bars
- [ ] Code and data archived (Zenodo DOI)
- [ ] Reproducibility instructions included
- [ ] Epistemically-humble language ("suggests", not "proves")

---

## Contact & Support

**Research Team**: HYMetaLab  
**Study PI**: Jordan Heckler  
**Framework**: OpenLaws Automation v1.0  
**GitHub**: `conciousness_proxy_sim/studies/phase35b_cross_survey_replication.yml`

**Questions?**  
Open an issue or consult `docs/openlaws_user_guide.md`

---

**Last Updated**: 2025-10-14  
**Classification**: PREREGISTERED | EXECUTION-READY

