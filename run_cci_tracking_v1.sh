#!/usr/bin/env bash
set -euo pipefail

# Simple runner for the CCI tracking study
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

STUDY_YML="openlaws_studies/cci_tracking_v1_0/study.yml"

if [ -f "openlaws_automation.py" ]; then
  echo ">> Running via OpenLaws automation ..."
  python openlaws_automation.py run --study cci_tracking_v1_0
  python openlaws_automation.py validate --study cci_tracking_v1_0
  python openlaws_automation.py report --study cci_tracking_v1_0
else
  echo ">> OpenLaws automation not found — using fallback adapter run + quick_report."
  python - <<'PY'
import yaml, json, os
from openlaws_studies.adapters.cci_tracking_adapter import run_adapter, Config
with open('openlaws_studies/cci_tracking_v1_0/study.yml') as f:
    cfg = yaml.safe_load(f)
out_dir = cfg['exports']['data_dir']
conf = Config(
    seeds=cfg['constants']['seeds'],
    bootstrap_n=cfg['constants']['bootstrap_n'],
    confidence=cfg['constants']['confidence'],
    out_dir=out_dir
)
summary = run_adapter(conf)
print('Adapter summary:', summary)
PY
  python quick_report.py
fi

echo "Done. Data: discovery_results/cci_tracking/"
