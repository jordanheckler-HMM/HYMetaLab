---
title: report.md
date: 2025-10-16
version: draft
y7b3c97bbbf
---

# Patterns of Consciousness: Recurrence & Causality Tests

Author: Heck Yeah Lab

## Config

```json
{
  "paths": {
    "input_glob": "data/*.csv",
    "outputs_dir": "outputs"
  },
  "preprocess": {
    "zscore": true,
    "resample_to": null
  },
  "recurrence": {
    "tau_strategy": "percentile",
    "tau_percentile": 10,
    "tau_fixed": 0.8,
    "norm": "euclidean"
  },
  "rqa": {
    "min_diag_line": 2,
    "min_vert_line": 2
  },
  "dtw": {
    "mode": "pca1",
    "window": null
  },
  "surrogates": {
    "n": 100,
    "kinds": [
      "phase",
      "shuffle"
    ]
  },
  "granger": {
    "max_lag": 12,
    "ic": "aic",
    "fdr_alpha": 0.05
  },
  "plotting": {
    "dpi": 220
  },
  "report": {
    "title": "Patterns of Consciousness: Recurrence & Causality Tests",
    "author": "Heck Yeah Lab",
    "include_te": true
  }
}
```

## Summaries

### recurrence_global_tau

```
1.1002468497299625
```
### rqa_global

```
{'RR': np.float64(0.1), 'DET': np.float64(0.6751388888888888), 'LAM': np.float64(0.7512583333333334), 'Lmax': 6000, 'ENTR': np.float64(1.4387124622031382), 'TT': np.float64(3.7950006735367414)}
```
### rqa_epochs

```
{0: {'RR': np.float64(0.1), 'DET': np.float64(0.6751388888888888), 'LAM': np.float64(0.7512583333333334), 'Lmax': 6000, 'ENTR': np.float64(1.4387124622031382), 'TT': np.float64(3.7950006735367414)}}
```
### dtw_matrix_shape

```
(1, 1)
```
### surrogates_sample

```
[('phase', {'RR': np.float64(0.1), 'DET': np.float64(0.6797588888888889), 'LAM': np.float64(0.7816677777777777), 'Lmax': 6000, 'ENTR': np.float64(1.494326771471911), 'TT': np.float64(4.003067022114993)}), ('phase', {'RR': np.float64(0.1), 'DET': np.float64(0.6712333333333333), 'LAM': np.float64(0.7755663888888888), 'Lmax': 6000, 'ENTR': np.float64(1.439148969994507), 'TT': np.float64(3.8692765440858548)}), ('phase', {'RR': np.float64(0.1), 'DET': np.float64(0.6389894444444445), 'LAM': np.float64(0.7540086111111111), 'Lmax': 6000, 'ENTR': np.float64(1.3232503394441137), 'TT': np.float64(3.6111715686013657)}), ('phase', {'RR': np.float64(0.1), 'DET': np.float64(0.6752533333333334), 'LAM': np.float64(0.7802144444444444), 'Lmax': 6000, 'ENTR': np.float64(1.4732435759043037), 'TT': np.float64(3.9578731996837955)}), ('phase', {'RR': np.float64(0.1), 'DET': np.float64(0.6806588888888889), 'LAM': np.float64(0.7838177777777777), 'Lmax': 6000, 'ENTR': np.float64(1.470642250643621), 'TT': np.float64(3.9417209365998667)})]
```

## Figures

![](figures/recurrence_plot_GLOBAL.png)
![](figures/recurrence_plot_EPOCH_0.png)
![](figures/dtw_heatmap.png)
![](figures/granger_heatmap.png)

## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
