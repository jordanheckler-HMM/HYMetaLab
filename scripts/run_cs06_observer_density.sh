#!/usr/bin/env bash
set -euo pipefail

# CS06 Observer Density Replication runner
# Usage: ./scripts/run_cs06_observer_density.sh [RUN_ID]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-cs06_observer_density_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="discovery_results/$RUN_ID"

echo "Starting CS06 replication run -> $OUT_DIR"

# Environment setup hints
export PYTHONHASHSEED=0
export OLAW_LOG_LEVEL=${OLAW_LOG_LEVEL:-INFO}

# Try to activate a venv if present
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
  echo "Activating venv at ./venv"
  # shellcheck disable=SC1091
  source venv/bin/activate
elif command -v conda >/dev/null 2>&1 && conda info --envs | grep -q "openlaws"; then
  echo "Activating conda env 'openlaws'"
  conda activate openlaws || true
else
  echo "No local venv found and no conda 'openlaws' env detected. Proceeding with current Python environment."
fi

mkdir -p "$OUT_DIR"

LOGFILE="$OUT_DIR/run.log"

echo "--- 1. Execute the simulation sweep ---" | tee -a "$LOGFILE"
python openlaws_automation.py run \
  --study studies/cs06_observer_density.yml \
  --out "$OUT_DIR/runs" \
  --log "$LOGFILE"

echo "--- 2. Summarize results ---" | tee -a "$LOGFILE"
python openlaws_automation.py summarize \
  --study cs06_observer_density_replication \
  --out "$OUT_DIR/summary.csv"

echo "--- 3. Validate with bootstrap + CI separation test ---" | tee -a "$LOGFILE"
python openlaws_automation.py validate \
  --study cs06_observer_density_replication \
  --validator validators/cs06.py \
  --summary "$OUT_DIR/summary.csv" \
  --out "$OUT_DIR/validation_checks.json"

echo "--- 4. Auto-generate report + plot ---" | tee -a "$LOGFILE"
python - <<'PY'
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, json, datetime, pathlib, sys
p = pathlib.Path('discovery_results') / "$RUN_ID"
summary_path = p / 'summary.csv'
if not summary_path.exists():
    print(f"Summary file not found: {summary_path}", file=sys.stderr)
    sys.exit(1)
df = pd.read_csv(summary_path)
fig, ax = plt.subplots(figsize=(6,4))
for s, g in df.groupby('seed'):
    ax.plot(g.observer_density, g.coherence_final, marker='o', label=f'seed {s}')
ax.set_xlabel('Observer density (ρ)')
ax.set_ylabel('Final coherence')
ax.set_title('CS06 Replication — Coherence vs. ρ')
ax.legend()
plt.tight_layout()
out_png = p / 'coherence_vs_rho.png'
plt.savefig(out_png)

res_path = p / 'validation_checks.json'
if res_path.exists():
    res = json.load(open(res_path))
else:
    res = { 'count_ci_separated': 0, 'seeds': [], 'validated_fraction': 0.0, 'passed': False, 'thresholds': {}, 'acceptance_criteria': {} }

report = f"""# CS06 Observer-Density Replication
Date: {datetime.datetime.now():%Y-%m-%d %H:%M}
Validated seeds: {res.get('count_ci_separated', 0)}/{len(res.get('seeds', []))}
Validated fraction: {res.get('validated_fraction', 0.0):.2f}
Pass: {res.get('passed', False)}
Thresholds: {res.get('thresholds', {})}
Acceptance: {res.get('acceptance_criteria', {})}
"""
open(p / 'report.md','w').write(report)
print(report)
PY

# --- 5. Archive checksum for reproducibility ---
echo "Archiving checksums to $OUT_DIR/CHECKSUMS.sha256"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUT_DIR"/* > "$OUT_DIR/CHECKSUMS.sha256"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUT_DIR"/* > "$OUT_DIR/CHECKSUMS.sha256"
else
  echo "WARNING: no sha256sum or shasum available; skipping checksums" | tee -a "$LOGFILE"
fi

echo "✅ CS06 complete. Results stored in $OUT_DIR"