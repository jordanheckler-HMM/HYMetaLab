---
title: METHODS_QC_PACKET.md
date: 2025-10-16
version: draft
checksum: 58fa51c08be0
---

# HYMetaLab Methods & QC Packet

## Research Integrity Standards

- **Guardian Alignment**: ≥ 70/100 (current: 70)
  - Language safety: 97%
  - Reproducibility: 66%
  - Preregistration: 50%

- **OpenLaws Protocol**:
  - Preregistered parameters in YAML
  - Deterministic seeds: [11, 17, 23, 29]
  - Bootstrap CI thresholds: 95%, n=1200
  - Classification: VALIDATED / PARTIAL / UNDER_REVIEW

- **Statistical Rigor**:
  - Bootstrap CIs added to all summaries
  - Preregistered thresholds: ΔCCI ≥ 0.03, Δhazard ≤ -0.01
  - Conservative classification using CI bounds

## Validated Studies

See META_ANALYSIS.json for cross-study synthesis.

## Data Availability

- Raw results: `discovery_results/*/`
- Study specifications: `studies/*.yml`
- QC reports: `qc/`
- Templates: `templates/`
- Ethics: `ETHICS.md`
- Citations: `CITATIONS.bib`

## Reproduction

All studies can be reproduced using:
```bash
python openlaws_automation.py run --study studies/[study_name].yml
python openlaws_automation.py validate --bootstrap 1000
python openlaws_automation.py report
```

## Contact

HYMetaLab Research Integrity Team


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.


## Data Provenance
- **Source:** HYMetaLab simulation framework (v1.0)
- **Repository:** https://github.com/hymetalab/consciousness_proxy_sim
- **Validation:** Guardian v4, TruthLens v1, MeaningForge v3
- **Reproducibility:** Seeds fixed, parameters documented in `config.yaml`

## References & Citations
1. HYMetaLab Framework Documentation. Internal Technical Report. 2025.
2. Guardian v4 Ethical Validation System. Quality Control Protocols.
3. Collective Coherence Index (CCI): Mathematical definition in `core/cci_math.py`
4. Simulation parameters: See `field_unified_constants_v33c.yml`

## Attribution
- **Framework:** HYMetaLab Research Collective
- **Methods:** Documented in `METHODS.md`
- **Analysis:** Statistical methods per `validation/utils.py`
