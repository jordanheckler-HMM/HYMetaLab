#!/usr/bin/env python3
"""
Generate an OSF registration blurb summarizing the v2 prereg + replication plan.
"""
from __future__ import annotations

import os
from datetime import date

BLURB = """# OSF Registration Blurb — Integrity Addendum v2
**Date:** {today}

## Title
Validation of Consciousness/Resilience Laws: H1 Fear×CCI, H2 Gini→Collapse(+CCI), H3 Constructive Shock Thresholds

## Summary
We preregister three hypotheses related to aggression/resilience in city-level panels. Analysis is fully containerized with deterministic seeds. Time-scale alignment is enforced for inequality features. All confirmatory models, baselines, and diagnostics are declared a priori.

## Hypotheses
- **H1:** CCI reduces the marginal effect of fear on aggression (β(Fear×CCI)<0).
- **H2:** Inequality predicts collapse with a threshold near Gini≈0.30; CCI moderates by raising the threshold.
- **H3:** Shocks with severity<0.5 improve post-shock outcomes; ≥0.8 degrade; regrowth parameter effect is explicitly ablated.

## Outcomes & Data
- Primary outcomes: weekly aggression rate per 100k (H1), collapse flag/time-to-collapse (H2), survival/efficiency indices (H3).
- Data: public crime/events/trends; inequality proxy; documented versioning and SHA256 for raw files.

## Analysis Plan
- Confirmatory equations and interactions specified in prereg files.
- Baselines (ARIMA/Prophet/regularized GLM/GBM) and event-study/DiD where relevant.
- Placebo/pre-trend checks; ablations; sensitivity to resampling.

## Reproducibility
- Docker image + `scripts/repro_v2.sh`; deterministic seeds.
- Expected artifacts per hypothesis (tables, plots, CIs, metrics).
- External replication invitation and QC checklist included.

## Deviations
All deviations will be logged in an Exploratory Addendum with timestamps.

## Links
- Repo: (add link)
- Preprints / Docs: (add link)
- Contact: (add email)
"""


def main():
    """Generate OSF registration blurb."""
    out = "docs/prereg/OSF_registration_blurb.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(BLURB.format(today=date.today().isoformat()))
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()
