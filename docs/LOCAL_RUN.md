# Local Run Guide (Single-laptop prototype)

This document explains how to run the local prototype that simulates the core engines and
produces deterministic outputs under a global seed.

Prerequisites
- Python 3.10+ (venv recommended)
- From project root

Quick start

1. Ensure the global seed is set in `configs/global_seed.yml` (SEED: 42)
2. Run the full local pipeline:

```sh
make run_all_local
```

This will execute all engine version scripts and write JSON outputs under `outputs/{engine}/`.

Targets
- `make tl_v1` ... `tl_v5` : run TruthLens versions 1..5
- `make mf_v1` ... `mf_v5` : run MeaningForge versions
- `make oc_v1` ... `oc_v5` : run OriginChain versions
- `make al_v1` ... `al_v5` : run Aletheia versions
- `make run_all_local` : runs all of the above in sequence

Determinism checklist
- Confirm `configs/global_seed.yml` contains `SEED: 42`
- Confirm outputs include `seed` field with value 42

Output locations
- `outputs/truthlens/truthlens_v{n}.json`
- `outputs/meaningforge/meaningforge_v{n}.json`
- `outputs/originchain/originchain_v{n}.json`
- `outputs/guardian/guardian_v{n}.json`
- `outputs/aletheia/aletheia_v{n}.json`

Notes
- The run scripts are intentionally simple and deterministic for a prototype. They are
  placeholders for connecting real modules or model runners.

Screenshots checklist (manual)
- [ ] Outputs folder exists and contains per-engine JSONs
- [ ] Each JSON has a `seed` field equal to 42
- [ ] Make runs complete within 10 minutes on laptop

