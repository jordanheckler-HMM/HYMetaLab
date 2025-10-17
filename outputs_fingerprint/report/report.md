---
title: report.md
date: 2025-10-16
version: draft
checksum: e71b2078bd12
---

# Archetype Recurrence: Quick Fingerprint Report

- Seeds: 3 | Epochs: 5 | Agents: 240 | k: 4

## Key Metric

- **ARI\*** (mean off-diagonal epoch similarity): **0.998**

Higher ARI* means stronger recurrence of archetype centroids across epochs.

## What to look for

- The **centroid_scatter.png** should show clusters that re-appear across different epochs/seeds (overlapping or near-identical points).

- The **epoch_similarity_heatmap.png** should show brighter blocks for recurrent epochs (e.g., 0~2, 1~3).

## Files

- metrics/recurrence_summary.csv
- metrics/centroids.csv
- metrics/embeddings_sample.csv
- figures/centroid_scatter.png
- figures/epoch_similarity_heatmap.png


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.

**Generated:** 2025-10-17
**Framework:** HYMetaLab
**Validation:** Guardian v4


## Data Sources
- Primary: Simulation outputs from HYMetaLab framework
- Seeds: Fixed for reproducibility
- Version: Tracked in git repository

## References
1. HYMetaLab Framework Documentation (internal)
2. Guardian v4 Validation System
3. Reproducibility standards per SOP v1.1

**Citation Format:** Author (Year). Title. Framework/Journal.
