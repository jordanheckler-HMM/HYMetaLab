# HYMetaLab Next-Gen Stack — Quickstart

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-nextgen.txt
```

## Run the pipeline

### 1) Generate hypotheses + prereg study
```bash
python tools/auto_theorist.py
```

### 2) Build cross-domain coherence map
```bash
python tools/coherence_mapper.py
open tools/cross_domain_map.html  # or xdg-open on Linux
```

### 3) Guardian alignment audit
```bash
python qc/guardian_auditor.py
cat qc/alignment_report.json
```

### 4) MetaDashboard (static build + optional local server)
```bash
python tools/metadashboard.py
# then: http://localhost:5055
```

## OpenLaws flow (example)
```bash
python openlaws_automation.py run --lab openlight
python openlaws_automation.py validate --lab openlight
python openlaws_automation.py report --lab openlight
```

## Archive validated studies
```bash
STAMP=$(date +%Y%m%d_%H%M%S)
zip -r "results/archive/validated_${STAMP}.zip" discovery_results
shasum -a 256 "results/archive/validated_${STAMP}.zip" > "results/archive/validated_${STAMP}.sha256"
```

