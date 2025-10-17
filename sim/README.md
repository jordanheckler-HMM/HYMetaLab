---
title: README.md
date: 2025-10-16
version: draft
t48803d4054
---

Sim package
===========

Utilities for writing/validating simulation outputs and a tiny deterministic smoke-run.

Quick start
-----------

Run the smoke-run (no external data required):

```bash
python3 smoke_run.py
```

This creates `outputs/smoke_<timestamp>/` with `run_manifest.json` and `decisions.jsonl` and validates the JSONL file against the schema in `schema/`.

Analysis
--------
Use `examples/quick_analysis.py` to plot average reported confidence from a run:

```bash
python3 examples/quick_analysis.py outputs/smoke_YYYYMMDD_HHMMSS
```


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.

## Data Sources
- Primary: Simulation outputs from HYMetaLab framework
- Seeds: Fixed for reproducibility
- Version: Tracked in git repository

## References
1. HYMetaLab Framework Documentation (internal)
2. Guardian v4 Validation System
3. Reproducibility standards per SOP v1.1

**Citation Format:** Author (Year). Title. Framework/Journal.
