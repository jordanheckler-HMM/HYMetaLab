---
title: README.md
date: 2025-10-16
version: draft
}35c5426d2a
---

# Real-World Validation Module

This module validates simulation predictions against real-world data by automatically fetching open datasets, mapping them to simulation constructs, and generating comprehensive analysis reports.

## Quick Start

```bash
# Run a single scenario
python -m real_world_validation.cli run --scenario 2008_us_market

# Run all scenarios
python -m real_world_validation.cli run --all

# Run with fresh data (ignore cache)
python -m real_world_validation.cli run --scenario covid_italy --fresh

# Run with custom date range
python -m real_world_validation.cli run --scenario gini_gdp_global_2000_2024 --start 2000-01-01 --end 2024-12-31
```

## Available Scenarios

### 1. 2008 US Market Crisis
- **Type:** Market analysis
- **Data:** S&P 500 market data from Stooq
- **Analysis:** Drawdown shocks, recovery patterns, volatility-based collapse risk
- **Command:** `python -m real_world_validation.cli run --scenario 2008_us_market`

### 2. COVID-19 Italy
- **Type:** Epidemic analysis  
- **Data:** OWID COVID-19 data, World Bank Gini/GDP
- **Analysis:** Case spike shocks, recovery patterns, inequality-based collapse risk
- **Command:** `python -m real_world_validation.cli run --scenario covid_italy`

### 3. Global Inequality & GDP (2000-2024)
- **Type:** Macroeconomic analysis
- **Data:** World Bank Gini and GDP growth for 6 countries
- **Analysis:** GDP drawdown shocks, recovery patterns, Gini-based collapse risk
- **Command:** `python -m real_world_validation.cli run --scenario gini_gdp_global_2000_2024`

## Output Structure

Each scenario generates results in `./discovery_results/real_world/<scenario_id>/`:

```
discovery_results/real_world/<scenario_id>/
├── data_clean/
│   └── harmonized_data.csv          # Cleaned, harmonized time series
├── metrics/
│   ├── shocks.csv                   # Shock events with classifications
│   ├── survival.csv                 # Recovery periods and metrics
│   ├── collapse.csv                 # Collapse risk analysis
│   └── cci.csv                      # CCI data (if available)
├── figures/
│   ├── risk_over_time.png           # Risk timeline with thresholds
│   ├── shock_timeline.png           # Shock events by classification
│   ├── survival_curve.png           # Recovery curve with power-law fit
│   └── data_overview.png            # Multi-series overview
├── REPORT.md                        # Comprehensive analysis report
└── run_manifest.json               # Execution metadata and file hashes
```

## Key Features

### Data Sources
- **OWID COVID-19:** Global pandemic data with cases, deaths, stringency
- **World Bank:** Gini coefficients and GDP growth by country
- **Stooq:** Daily market data with automatic fallback (SPX → SPY)

### Simulation Constructs
- **Shocks:** Classified as constructive (<0.5), transition (~0.5), or destructive (>0.5)
- **Survival:** Recovery patterns fitted with power-law curves (reports α parameter)
- **Collapse:** Risk based on inequality thresholds (flags Gini ≥ 0.3)
- **CCI:** Consciousness Calibration Index (when prediction vs outcome data available)

### Analysis Pipeline
1. **Fetch:** Download data with local caching (24-hour TTL)
2. **ETL:** Clean, harmonize, and validate time series
3. **Map:** Convert to simulation constructs (shocks, survival, collapse, CCI)
4. **Bridge:** Connect to existing simulation modules
5. **Visualize:** Generate matplotlib figures
6. **Report:** Create Markdown reports with findings and reproducibility steps

## Testing

Run smoke tests to verify functionality:

```bash
python tests_real_world_validation/test_smoke.py
```

Tests verify:
- Scenario configuration loading
- End-to-end pipeline execution
- Shock classification counts
- Collapse threshold analysis
- Survival recovery metrics
- Output file generation (≥3 figures, REPORT.md, manifest)

## Dependencies

The module requires:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `pyyaml` - Configuration parsing
- `requests` - HTTP data fetching

## Configuration

Scenarios are defined in `real_world_validation/scenarios.yaml`. Each scenario specifies:
- Data sources and parameters
- Analysis window (start/end dates)
- Metrics configuration (shock methods, thresholds, etc.)

## Limitations

- **Data Quality:** Results depend on source data completeness
- **Proxy Measures:** Some constructs use proxies when direct data unavailable
- **CCI Requirements:** Requires prediction vs outcome data (often not available)
- **Causality:** Shows correlations, not necessarily causal relationships
- **API Limits:** External APIs may have rate limits or authentication requirements

## Integration

The module integrates with existing simulation modules:
- `shock_resilience.py` - Shock classification
- `goal_externalities.py` - Collapse risk calculation  
- `survival_experiment.py` - Recovery analysis
- `calibration_experiment.py` - CCI computation

If these modules are unavailable, fallback implementations are used.

## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
