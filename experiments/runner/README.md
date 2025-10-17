---
title: README.md
date: 2025-10-16
version: draft
checksum: 48e280986743
---

This folder contains a lightweight TypeScript gateway for planning and dispatching experiment payloads to running Python sim instances.

Quick commands (after running `npm install` in repo root):

- Plan runs:    npx ts-node experiments/runner/plan.ts
- Discover:     npx ts-node experiments/runner/discover.ts
- Dispatch:     npx ts-node experiments/runner/dispatch.ts
- Collect:      npx ts-node experiments/runner/collect.ts
- Summarize:    npx ts-node experiments/runner/summarize.ts
- Bundle:       npx ts-node experiments/runner/bundle.ts

Notes:
- The gateway expects each sim instance to expose a small HTTP API on localhost ports (default range 5201-5209):
  - GET /healthz -> returns JSON with { simInstanceId, metricsPort, dataDir, dbPath, instanceFingerprint, rngSeed }
  - POST /enqueue-run -> accepts payload JSON and returns 202 when accepted
  - GET /run-status/:runId -> returns run result JSON when complete

- A simple Python shim is provided to run next to each sim instance if needed: `experiments/runner/python_shim.py`.


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
