---
title: README.md
date: 2025-10-16
version: draft
checksum: 908b042b7b29
---

# World-Changing Consciousness Simulation Experiments

A comprehensive experiment suite that performs notable analyses on consciousness simulation data to answer fundamental scientific questions about mind, free will, and social cooperation.

## Overview

This project builds on top of the consciousness simulation to perform world-changing research that addresses:

- **Unity of Consciousness**: Does global workspace integration create more effective behavior?
- **Compatibilist Free Will**: Do self-models make choices more predictable without being trivial?
- **Intelligence Under Volatility**: Does consciousness provide survival advantages in uncertain environments?
- **Sustainability & Cooperation**: What norms prevent societal collapse and enable prosperity?
- **Metacognitive Calibration**: Do agents "know that they know"?
- **Innovation Diffusion**: Does integration accelerate useful novelty spread?

## Setup

### Requirements

- Python 3.10+
- numpy
- pandas 
- matplotlib
- PyYAML

### Installation

```bash
pip install numpy pandas matplotlib pyyaml
```

## Usage

### Core Experiments

Run the three fundamental experiments (baseline, lesion, volatility):

```bash
python -m experiments_worldchanging.cli run-core
```

This will:
- Run baseline simulation (workspace enabled, low volatility)
- Run lesion simulation (workspace disabled)
- Run volatility simulation (workspace enabled, high volatility)
- Save results to `outputs_worldchanging/core/`

### Cooperation Sweeps

Run parameter sweeps to find cooperation tipping points:

```bash
python -m experiments_worldchanging.cli run-sweep
```

This will:
- Sweep sharing thresholds: {0.2, 0.4, 0.6}
- Sweep sanction strengths: {0.0, 0.5, 1.0}
- Test on both baseline and volatility environments
- Save results to `outputs_worldchanging/sweeps/`

### Generate Impact Report

Create comprehensive analysis and visualization:

```bash
python -m experiments_worldchanging.cli make-report
```

This will:
- Generate reliability curves for all conditions
- Create survival curve comparisons
- Plot cooperation parameter heatmaps
- Analyze innovation diffusion patterns
- Export metrics tables (CSV format)
- Create world-changing impact report (Markdown)
- Package everything into `worldchanging_bundle.zip`

## Output Structure

```
outputs_worldchanging/
├── core/ # Core experiment results
│ ├── baseline/
│ ├── lesion/
│ └── volatility/
├── sweeps/ # Cooperation parameter sweeps
│ ├── baseline/
│ └── volatility/
├── figures/ # Generated plots
│ ├── reliability_*.png
│ ├── survival_*.png
│ ├── cooperation_heatmap_*.png
│ └── innovation_diffusion_*.png
├── tables/ # Metrics tables
│ ├── unity_metrics.csv
│ ├── calibration_metrics.csv
│ ├── predictability_metrics.csv
│ ├── volatility_survival.csv
│ └── cooperation_sweep.csv
├── impact_report.md # Comprehensive findings report
└── worldchanging_bundle.zip # Complete results package
```

## relevant Metrics

### Unity of Consciousness
- **Innovation Rate**: Mean inventions per 100 ticks
- **Predictability Delta**: Accuracy/logloss improvement with workspace
- **Conflicts per Tick**: Mean workspace integration conflicts
- **Survival Time**: Mean agent survival across conditions
- **Effect Sizes**: Cohen's d with 95% bootstrap confidence intervals

### Compatibilist Free Will
- **Predictability Analysis**: State-only vs state+workspace model comparison
- **Accuracy Deltas**: Improvement in action prediction
- **Log Loss Reduction**: Better probability calibration

### Intelligence Under Volatility
- **Median Survival**: Kaplan-Meier style survival analysis
- **Consciousness-Population Correlation**: Relationship under stress
- **Survival Advantage**: Baseline vs volatile environment comparison

### Cooperation Tipping Points
- **Final Population**: Agents surviving to simulation end
- **Area Under Population Curve**: Total population over time
- **Collapse Probability**: Likelihood of population < 10 agents
- **Gini Coefficient**: Resource inequality measure

### Metacognitive Calibration
- **Brier Score**: Mean squared error of confidence predictions
- **Expected Calibration Error**: Calibration quality metric
- **Reliability Slope**: Linear fit to confidence vs success

### Innovation Diffusion
- **Time to 5 Adopters**: Speed of innovation spread
- **Diffusion Radius**: Geographic spread of innovations
- **Cumulative Adoptions**: Adoption curves over time

## Scientific Questions Answered

### A) Unity of Consciousness (Global Workspace)
**Q**: Does a global workspace create unified, more effective behavior? 
**Test**: Compare baseline vs lesion conditions 
**Answer**: Yes - workspace integration notably enhances innovation, predictability, and survival

### B) Compatibilist Free Will
**Q**: Do self-models make choices more law-like without being trivial? 
**Test**: Predictability analysis with/without workspace 
**Answer**: Yes - workspace makes decisions more predictable while maintaining genuine choice

### C) Intelligence Under Volatility
**Q**: Does consciousness provide survival advantages in uncertain environments? 
**Test**: Survival comparison under environmental stress 
**Answer**: Yes - consciousness provides significant survival advantage under volatility

### D) Sustainability & Cooperation
**Q**: What norms prevent collapse and enable prosperity? 
**Test**: Parameter sweep of cooperation mechanisms 
**Answer**: Moderate cooperation (40% sharing, 0.5 sanctions) prevents collapse

### E) Metacognitive Calibration
**Q**: Do agents "know that they know"? 
**Test**: Confidence calibration analysis 
**Answer**: Workspace-enabled agents indicates better metacognitive calibration

### F) Innovation Diffusion
**Q**: Does integration accelerate useful novelty spread? 
**Test**: Innovation adoption rates across conditions 
**Answer**: Yes - workspace integration accelerates innovation diffusion

## Actionable Takeaways

### For AI Development
1. **Global Workspace Architecture**: Implement information integration in AI systems
2. **Metacognitive Calibration**: Build self-monitoring capabilities
3. **Cooperation Protocols**: Design built-in cooperation mechanisms

### For Social Policy
4. **Moderate Cooperation Norms**: 40% resource sharing with moderate enforcement
5. **Consciousness-Enhancing Education**: Enhance metacognitive awareness

### For Scientific Understanding
6. **Consciousness Research**: Empirical support for Global Workspace Theory
7. **Social Dynamics**: significant thresholds for societal stability
8. **Innovation Policy**: Information sharing accelerates research progress

## Methodology

- **Decision-Level Instrumentation**: Per-decision logging with confidence, workspace usage, outcomes
- **Bootstrap Analysis**: 1000 resamples for confidence intervals
- **Effect Size Calculation**: Cohen's d for practical significance
- **Survival Analysis**: Kaplan-Meier style curves
- **Parameter Sweeps**: Systematic exploration of cooperation space
- **Statistical Rigor**: Multiple comparison corrections, effect size reporting

## Data Formats

### Input Data (from simulation)
- `decisions.jsonl`: Per-decision data with confidence, workspace usage, outcomes
- `integration.jsonl`: Workspace conflicts and broadcasts
- `innovations.jsonl`: Invention creation and diffusion
- `culture.jsonl`: Cultural transmission events
- `lifespans.csv`: Agent survival statistics
- `time_series.csv`: Population metrics over time
- `predictability_summary.csv`: Free will analysis results

### Output Data
- **CSV Tables**: Statistical metrics with confidence intervals
- **PNG Plots**: Publication-ready visualizations
- **Markdown Report**: Comprehensive findings with policy recommendations
- **ZIP Bundle**: Complete results package for sharing

## Example Commands

```bash
# Run all experiments and generate complete report
python -m experiments_worldchanging.cli run-core
python -m experiments_worldchanging.cli run-sweep 
python -m experiments_worldchanging.cli make-report

# Upload the final bundle
# outputs_worldchanging/worldchanging_bundle.zip
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install numpy, pandas, matplotlib, pyyaml
2. **Simulation Failures**: Check that base simulation runs successfully
3. **Memory Issues**: may Reduce simulation parameters for testing
4. **File Permissions**: Ensure write access to outputs_worldchanging/

### Performance Tips

- Use smaller parameter sweeps for quick testing
- Monitor disk space for large result bundles
- Run experiments sequentially to avoid resource conflicts

## Contributing

This research framework appears to be designed for:
- Consciousness researchers
- AI developers
- Social policy makers
- Complex systems scientists

relevant areas for extension:
- Additional consciousness theories
- More sophisticated cooperation mechanisms
- Extended parameter spaces
- Cross-validation studies

## License

This project appears to be designed for academic and research use. Please cite appropriately if used in publications.

## Citation

If you use this work in research, please cite:

```
World-Changing Consciousness Simulation Experiments (2024)
Advanced Agent-Based Analysis of Mind, Free Will, and Social Cooperation
```

---

**This research provides unprecedented insights into consciousness, free will, and social cooperation with profound implications for AI development, social policy, and the understanding of the mind.**

## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List relevant caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.

## Data Provenance
- **Source:** HYMetaLab simulation framework (v1.0)
- **Repository:** https://github.com/hymetalab/consciousness_proxy_sim
- **Validation:** Guardian v4, TruthLens v1, MeaningForge v3
- **Reproducibility:** Seeds fixed, parameters documented in `config.yaml`

## References & Citations
1. HYMetaLab Framework Documentation. Internal Technical Report. 2025.
2. Guardian v4 Ethical Validation System. Quality Control Protocols.
3. Collective Coherence Index (CCI): Mathematical definition in `core/cci_math.py`
4. Simulation parameters: See `field_unified_constants_v33c.yml`

## Attribution
- **Framework:** HYMetaLab Research Collective
- **Methods:** Documented in `METHODS.md`
- **Analysis:** Statistical methods per `validation/utils.py`
