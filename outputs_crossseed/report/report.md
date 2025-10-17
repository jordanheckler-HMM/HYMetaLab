---
title: report.md
date: 2025-10-16
version: draft
H38f26203ca
---

# Cross-seed Recurrence & Causality — Quick Report

- Seeds: 3 | Epochs: 5 | Agents: 240 | k: 4

## Cross-seed Recurrence

- **ARI_seed** (mean epoch-wise centroid similarity across seeds): **0.996**

- See `figures/interseed_similarity_heatmap.png` (bright diagonal = same archetypes across seeds).

## Causality

- `metrics/granger_results.csv` with p-values for CCI→{Meaning, Coherence, Rc} (seed 0).

- `metrics/te_results.csv` with fast discrete TE proxy (bits). Higher TE suggests directional influence.

## Figures

- `centroid_scatter_by_seed.png` — repeating fingerprints across seeds.

- `granger_heatmap.png`, `te_bar.png` — causality visuals.

## Key Findings

- Cross-seed archetypal stability: 99.6% similarity

- No significant Granger causality detected at p<0.05

- Strongest information flow: CCI → Rc (0.367 bits)


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
