---
title: STACK_HEALTH_REPORT.md
date: 2025-10-16
version: draft
checksum: 5f9bd2aca5f3
---

# STACK_HEALTH_REPORT

generated: 2025-10-15T10:50:36.733114Z

- LaunchAgent: 4488	1	com.hymetalab.metadashboard
- Dashboard reachable: True (HTTP: 200)
- dashboard_data.json mtime (UTC): 2025-10-15T06:04:06.584425Z
- Guardian Auditor: OK (score: 70)
- Auto-Theorist dry-run outputs present: True

Details:
- metadashboard process: COMMAND  PID          USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
Python  4495 jordanheckler    6u  IPv4 0xcb7316da6576a13a      0t0  TCP 127.0.0.1:5055 (LISTEN)
Python  4499 jordanheckler    6u  IPv4 0xcb7316da6576a13a      0t0  TCP 127.0.0.1:5055 (LISTEN)
Python  4499 jordanheckler    8u  IPv4 0xcb7316da6576a13a      0t0  TCP 127.0.0.1:5055 (LISTEN)
- guardian summary: # Guardian Auditor Report v2.0

**Generated:** 2025-10-15 01:05:01  
**Overall Alignment Score:** 70/100 🟡  
**Status:** GOOD

---

## Component Scores

| Component | Score | Status | Details |
|-----------|-------|--------|---------|
| **Epistemic Humility** | 10/50 | 🔴 | 10 violations |
| **Preregistration** | 50/50 | 🟢 | 20 studies |
| **Reproducibility Index** | 66/50 | 🟢 | 66.7% reproducible 



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
