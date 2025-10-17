---
title: README.md
date: 2025-10-16
version: draft
checksum: 39d29e527fd6
---

# Universal Resilience Experiment - Patch v3

This experiment tests the **Universal Resilience (UR) hypothesis**: *Resilience may increases with constructive stress and coherence, and may decreases with inequality.*

## Patch v3 Changes

This patch introduces major improvements to may increase outcome variance and robustly fit the UR law:

### 🎯 Heterogeneity
- **Per-Agent Multipliers:** Resilience, regeneration, and mortality vary across agents
- **Controlled Variance:** Configurable heterogeneity parameters for reproducible variance
- **Lognormal Resilience:** Natural variation in agent robustness

### ⚡ Time-Scaled Shocks
- **Gradual Shocks:** Replace single-tick penalties with duration × scope × severity
- **Exponential Taper:** Shock effects decay over time with configurable half-life
- **Targeted Agents:** Only selected agents experience shock effects

### 📊 Dynamic Collapse
- **Recovery-Based Definition:** Collapse = failure to recover to baseline within window
- **Baseline Windows:** Pre-shock stability measurement
- **Consecutive Thresholds:** Sustained recovery requirements

### 🧠 UR Robust Fitting
- **Grid Search + Regression:** Coarse grid initialization + log-OLS refinement
- **Log Smoothing:** Safe handling of near-zero values
- **Ridge Regularization:** Prevents numerical instability

### 🔧 Extended Configuration
- **450 Steps:** Longer runs for recovery observation
- **Early Shocks:** 30% timing for adequate recovery window
- **Quality Diagnostics:** Variance tracking and warnings

## How to run

# quick variance sanity
python -m experiments.universal_resilience.run --quick

# full grid
python -m experiments.universal_resilience.run

# if UR is skipped for low variance, widen variance per the knobs in the report banner, then rerun quick → full.

## Experimental Design

### Parameter Grid
- **Shock Severities:** [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]
- **Shock Durations:** [20, 40, 80] steps
- **Shock Scopes:** [0.4, 0.7, 1.0] fraction of agents
- **Target Gini Values:** [0.10, 0.20, 0.30, 0.40]  
- **Coherence Levels:** [low, med, high]
- **Population Sizes:** [100] (configurable)
- **Replicates per Cell:** 6 (configurable)
- **Simulation Steps:** 450 (configurable)
- **Shock Timing:** 30% of steps (step 135 for 450-step runs)

### Coherence Mapping
- **Low:** noise=0.20, social_coupling=0.30, coherence_value=0.40
- **Medium:** noise=0.10, social_coupling=0.60, coherence_value=0.70  
- **High:** noise=0.05, social_coupling=0.80, coherence_value=0.90

### Total Experimental Cells
- **Full Experiment:** 6 × 3 × 3 × 4 × 3 × 1 = 648 cells × 6 replicates = 3,888 runs
- **Quick Test:** 3 × 2 × 2 × 2 × 2 × 1 = 48 cells × 3 replicates = 144 runs

## Implementation Details

### Inequality Initialization
- Uses lognormal distribution to achieve target Gini coefficients
- Validates actual Gini within ±0.01 tolerance of target
- Retries up to 50 iterations if target not achieved

### Heterogeneity Implementation
- **Resilience Multipliers:** Lognormal distribution (σ=0.15)
- **Regeneration Jitter:** ±50% uniform variation
- **Mortality Jitter:** ±50% uniform variation
- **Per-Agent Storage:** Multipliers stored in agent dictionaries

### Time-Scaled Shock System
- **Duration-Based:** Shocks last 20-80 steps with exponential taper
- **Scope-Based:** Only selected agents experience shock effects
- **Triangular Envelope:** Ramp-up and ramp-down within shock window
- **Exponential Decay:** Post-shock effects decay with half-life=35 steps

### Dynamic Collapse Detection
- **Baseline Window:** 50 steps before shock for stability measurement
- **Recovery Window:** 120 steps after shock to attempt recovery
- **Recovery Threshold:** Must return to ≥70% of baseline alive fraction
- **Consecutive Requirement:** 12 consecutive steps meeting threshold
- **Collapse Definition:** Failure to recover within recovery window

### Coherence Implementation  
- Maps coherence levels to agent noise and social coupling parameters
- Uses coherence_value as CCI proxy when true CCI unavailable
- Affects agent behavior and resource consumption patterns
- Resource loss proportional to shock severity

### Metrics Computed
- **Resilience:** final_alive_fraction, area_under_survival_curve, min_alive_fraction
- **Recovery:** recovery_time (steps to 70% of baseline), recovered_flag, collapsed_flag
- **Inequality:** measured_gini (validated against target)
- **Coherence:** coherence_value_mean, cci_pre_shock_mean, cci_post_shock_mean
- **Shock Effects:** constructiveness, ur_score (original and learned)
- **System Health:** variance_alive_fraction, deaths_this_step

## Statistical Analysis

### Models Tested
1. **Single-factor baselines:**
   - Resilience ~ constructiveness (original peak=0.5)
   - Resilience ~ constructiveness (learned peak p*)
   - Resilience ~ coherence  
   - Resilience ~ 1/gini

2. **Universal Resilience:**
   - Resilience ~ UR_score (original formula)
   - Resilience ~ UR_score (learned exponents a,b,c)

3. **Full interaction:**
   - Resilience ~ constructiveness × coherence + gini

### Success Criteria
- UR Score R² > single-factor R² values
- Learned UR formula outperforms original formula
- Significant positive coefficient for UR_score
- Coherence shows protective effect
- Constructiveness peak p* learned from data
- Dynamic collapse detection captures recovery failures
- Gini shows negative effect on resilience

## Output Structure

Results are saved in timestamped directories under `discovery_results/universal_resilience/`:

```
UR_YYYYMMDD_HHMMSS/
├── raw_runs/               # Individual run logs (optional)
├── metrics/               
│   ├── cell_results.csv    # All individual run results
│   ├── cell_aggregates.csv # Aggregated results per cell
│   └── model_fits.csv      # Statistical model results
├── figures/               
│   ├── resilience_vs_UR_score.png
│   ├── heatmap_shock_gini_by_coherence_[low|med|high].png
│   ├── recovery_time_vs_severity.png
│   ├── collapse_rate_by_gini.png
│   ├── cci_pre_post_by_coherence.png
│   ├── model_comparison_bar.png
│   └── variance_panels.png
├── config/                
│   ├── config.yaml        # Copy of configuration used
│   ├── ur_params.json     # Learned parameters (p*, a, b, c)
│   └── diagnostics.csv    # Quality diagnostics and warnings
├── REPORT.md              # Comprehensive analysis report
└── run_manifest.json      # Execution metadata
```

## Configuration

Edit `config.yaml` to customize the experiment:

### Key Parameters
- **`experiment.steps`:** Simulation length (default: 450)
- **`experiment.shock_step_ratio`:** When to apply shock (default: 0.30)
- **`experiment.replicates`:** Runs per cell (default: 6)
- **`heterogeneity.enable`:** Enable per-agent variation (default: true)
- **`heterogeneity.resilience_sigma`:** Resilience variation (default: 0.15)
- **`heterogeneity.regen_jitter`:** Regeneration jitter (default: 0.50)
- **`heterogeneity.mort_base_jitter`:** Mortality jitter (default: 0.50)
- **`shock.taper_half_life`:** Shock decay rate (default: 35)
- **`collapse.baseline_window`:** Pre-shock stability window (default: 50)
- **`collapse.recovery_window`:** Post-shock recovery window (default: 120)
- **`ur_learning.learn_constructiveness_peak`:** Learn optimal peak (default: true)
- **`ur_learning.learn_exponents`:** Learn UR exponents (default: true)

```yaml
experiment:
  steps: 450
  shock_step_ratio: 0.30
  populations: [100]
  severities: [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]
  durations: [20, 40, 80]
  scopes: [0.4, 0.7, 1.0]
  ginis: [0.10, 0.20, 0.30, 0.40]
  coherence_levels: [low, med, high]
  replicates: 6
  seeds_base: 424242

coherence_map:
  low:   { noise: 0.20, social_coupling: 0.30, coherence_value: 0.40 }
  med:   { noise: 0.10, social_coupling: 0.60, coherence_value: 0.70 }
  high:  { noise: 0.05, social_coupling: 0.80, coherence_value: 0.90 }

heterogeneity:
  enable: true
  resilience_sigma: 0.15
  regen_jitter: 0.50
  mort_base_jitter: 0.50

shock:
  taper_half_life: 35
  severity_to_damage: 1.00
  duration_weight: 1.00
  scope_weight: 1.00

collapse:
  baseline_window: 50
  recovery_window: 120
  recovery_threshold: 0.70
  consecutive_ok_steps: 12
  gini_threshold: 0.25

ur_learning:
  learn_constructiveness_peak: true
  peak_grid: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  learn_exponents: true
  exponent_grid:
    a: [0.5, 1.0, 1.5, 2.0]
    b: [0.5, 1.0, 1.5, 2.0]
    c: [0.5, 1.0, 1.5, 2.0]
  log_smoothing_eps: 1e-3
  ridge_lambda: 1e-4

gini_tolerance: 0.01
max_gini_iterations: 50
```

## Dependencies

- Python 3.7+
- numpy
- pandas  
- matplotlib
- scikit-learn
- scipy
- pyyaml

## Integration

This experiment integrates with existing simulation modules:
- **Shock Resilience:** Uses existing shock classification logic
- **Goal Externalities:** Leverages collapse detection thresholds
- **Calibration Experiment:** Uses CCI computation when available
- **Survival Experiment:** Applies survival curve analysis

If these modules are unavailable, fallback implementations are used.

## Limitations

- **Gini Implementation:** Uses lognormal approximation; actual Gini may deviate from target
- **Coherence Proxy:** Uses noise/coupling as CCI proxy; true CCI requires prediction vs outcome data
- **Shock Model:** Simplified shock application; real-world shocks may be more complex
- **Survival Dynamics:** Basic resource-based survival; may not capture all resilience mechanisms
- **Collapse Definition:** Binary flag based on alive fraction < 0.3 and Gini > 0.3
- **Sample Size:** Results depend on number of replicates per cell
- **Causality:** Statistical associations do not imply causal relationships

## Testing

Run the quick test to verify functionality:

```bash
python -m experiments.universal_resilience.run --quick
```

This should produce:
- `discovery_results/universal_resilience/UR_*/REPORT.md`
- `discovery_results/universal_resilience/UR_*/metrics/cell_aggregates.csv`  
- `discovery_results/universal_resilience/UR_*/figures/resilience_vs_UR_score.png`

## Expected Results

If the Universal Resilience hypothesis is correct, we should observe:

1. **UR Score Performance:** R² > 0.5 for UR_score predicting resilience
2. **Constructiveness Effect:** Peak resilience at severity ≈ 0.5
3. **Coherence Protection:** Higher coherence may reduces collapse risk
4. **Inequality Harm:** Higher Gini may increases collapse probability
5. **Model Comparison:** UR_score outperforms single-factor baselines

The experiment provides comprehensive statistical analysis and visualization to test these predictions.


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
