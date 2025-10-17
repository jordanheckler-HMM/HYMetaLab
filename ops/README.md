---
title: README.md
date: 2025-10-16
version: draft
Ufc85787fb5
---

Heck Yeah Ops Kit

Utilities:
- `watchdog.py` - run functions in a subprocess with a timeout
- `validator.py` - simple hazmat validation and quarantine for bad run outputs
- `maintenance.py` - CLI for health, hazmat sweep and vacuuming exports
- `post_run_hazmat.py` - convenience script to run hazmat after a study

Usage examples:
  python ops/maintenance.py health
  python ops/maintenance.py hazmat <study_id>
  python ops/maintenance.py vacuum

Add `export PYTHONPATH="$PWD:$PYTHONPATH"` to your shell or source the repo `.env`.


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
