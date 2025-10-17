---
title: LabManager_Addendum_Data_Scoping.md
date: 2025-10-16
version: draft
checksum: c6a1bda33b57
---

## Data Classification & Scoping (QC-critical)

### Study Data Source (required)
Set `data_source` in every study's YAML/summary to one of:
- SIMULATION_ONLY
- EMPIRICAL_PARTIAL
- EMPIRICAL_FULL

**Claim Scoping Rules**
- SIMULATION_ONLY
  - Label: "simulation-validated" (rarely "empirically validated")
  - Abstract must include: "within simulation context"
  - Use: "suggests / indicates / supports the hypothesis"
  - Add: "hypothesis requiring empirical testing"
- EMPIRICAL_PARTIAL
  - Distinguish apparently which findings derive from real vs synthetic data
  - Scope claims to the portion supported by empirical data
- EMPIRICAL_FULL
  - May use "empirically tested/validated" (still avoid over-generalization)
  - Bound generalization to systems/parameters actually tested

### Circular Validation Prevention
Add to each study's metadata:
- `depends_on`: [list of study_ids] (optional)
- `independent_validation_basis`: one of
  - NEW_DATA_SOURCE | NEW_RESEARCH_GROUP | PREREGISTERED_REPLICATION
Rules:
- Studies using the **same synthetic dataset** cannot "validate" each other
- Independent validation requires at least one `independent_validation_basis`
- Lab Manager must flag cross-study p-hacking patterns

### Threshold Documentation
For each threshold used (e.g., ΔCCI ≥ 0.03, Δhazard ≤ −0.01, OpenLawsScore ≥ 0.75):
- `thresholds:` block MUST include:
  - `rationale`: text (pilot study, prior literature, policy)
  - `source`: citation/URL/DOI or repo file path
  - `version`: semver (e.g., 1.0.0)
- Threshold changes require:
  - `change_log:` entry (who/when/why) and version bump

### Epistemic Humility — Expanded Language Policy
**Avoid (auto-flag):**
- Strong: "suggests", "confirms", "definitive", "conclusive"
- Universal: "universal law", "applies to all systems"
- Empirical (misuse): "empirical validation" (for simulation output)
- Hype: "breakthrough", "notable", "novel", "paradigm shift"
- Comparisons to established laws (thermodynamics/Newton/Einstein)
**Prefer:**
- "suggests", "indicates", "consistent with", "supports the hypothesis"
- "simulation-validated", "within simulation context"
- "preliminary findings", "exploratory results"
- Explicit boundary statements (what was tested; what was not)

### Classification Matrix (scope-aware)
- VALIDATED: preregistered thresholds met + proper scope (per data_source)
- PARTIAL: one metric met or scope deficiency corrected in revision
- UNDER_REVIEW: insufficient data or pending CI/bootstrap
- FAILED: thresholds not met (valuable as hypothesis-generating)



## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
