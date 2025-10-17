---
title: README.md
date: 2025-10-16
version: draft
checksum: 4986bc489b2c
---

Adapters README
================

The `safety_cci_adapter.py` includes safety and provenance features to avoid accidental fabrication and may improve reproducibility.

Usage highlights
- allow_mocks: boolean flag. The adapter will refuse to run built-in mock simulations unless this is True. Set via `cfg['allow_mocks']=True`, `run_study(..., allow_mocks=True)` or env var `ALLOW_MOCKS=1`.
- sim_entry: optional string `module:func` to call a real simulator entrypoint. e.g. `sim_entry: 'myproj.sim:run_sim'` in the config.

Provenance
- The adapter writes `data/provenance.json` containing `used_mocks`, a SHA256 of the config file (if provided), git commit (if available), python executable, platform, and package versions. CI will fail if any provenance indicates `used_mocks: true`.


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
