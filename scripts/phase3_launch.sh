#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
PREREG="open_data/preregister.yml"
STD_DIR="open_data/standardized"
MAP_OUT="open_data/mapping.yml"
VAL_DIR="open_data/validation"
LOG="logs/lab_activity_log.md"

# Thresholds
TL_TARGET=0.90        # TruthLens score
MF_TARGET=0.90        # MeaningForge score
GUARDIAN_TARGET=90    # Guardian v4 score (0–100)

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "$(dirname "$LOG")" "$VAL_DIR"

echo "== Phase 3 — Mapping & Validation | $STAMP =="

# 1) TruthLens validation
echo "Running TruthLens…"
python3 truthlens_wrapper.py --check "$PREREG" \
  --json-out "$VAL_DIR/truthlens_report.json" \
  --summary-out "$VAL_DIR/truthlens_summary.md"

# 2) MeaningForge semantic validation
echo "Running MeaningForge…"
python3 meaningforge_wrapper.py --semantic "$PREREG" \
  --json-out "$VAL_DIR/meaningforge_report.json" \
  --summary-out "$VAL_DIR/meaningforge_summary.md"

# 3) Build mapping.yml
echo "Building mapping.yml…"
python3 tools/mapping_builder.py \
  --dataset "$STD_DIR" \
  --output "$MAP_OUT"

# 4) Guardian v4 validation of mapping.yml
echo "Guardian v4 validating mapping.yml…"
python3 qc/guardian_v4/guardian_v4.py --validate --file "$MAP_OUT" --report
GUARD_SUMMARY="qc/guardian_v4/guardian_summary_v4.md"
GUARD_REPORT="qc/guardian_v4/guardian_report_v4.json"

# ----- Gate checks (Python one-liner; no jq required) -----
echo "Evaluating gates…"
python3 - "$TL_TARGET" "$MF_TARGET" "$GUARDIAN_TARGET" <<'PY'
import json,sys,re
from pathlib import Path
# args
tl_t=float(sys.argv[1]); mf_t=float(sys.argv[2]); g_t=float(sys.argv[3])

vp=Path("open_data/validation")

def load_score(path, key_candidates=("score","overall_score","truthlens_score","meaningforge_score")):
    with open(path) as f:
        j=json.load(f)
    for k in key_candidates:
        if k in j and isinstance(j[k], (int,float)):
            return float(j[k])
    # fallback: search nested
    def walk(x):
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v,(int,float)) and k.lower().endswith("score"): return float(v)
                y=walk(v)
                if y is not None: return y
        elif isinstance(x, list):
            for e in x:
                y=walk(e)
                if y is not None: return y
        return None
    s=walk(j)
    if s is None: raise SystemExit(f"Could not find score in {path}")
    return s

tl=load_score(vp/"truthlens_report.json")
mf=load_score(vp/"meaningforge_report.json")

# Guardian score from v4 report JSON if present, else parse summary md
g_score=None
gj=Path("qc/guardian_v4/guardian_report_v4.json")
if gj.exists():
    with open(gj) as f:
        jr=json.load(f)
    g_score = float(jr.get("score") or jr.get("guardian_score") or 0)
else:
    # fallback: parse from summary md line like: "Guardian Alignment Score: 92.3/100"
    import re
    gs=Path("qc/guardian_v4/guardian_summary_v4.md")
    txt=gs.read_text() if gs.exists() else ""
    m=re.search(r"Guardian Alignment Score:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100", txt)
    g_score=float(m.group(1)) if m else 0.0

ok = (tl>=tl_t) and (mf>=mf_t) and (g_score>=g_t)
print(f"TruthLens={tl:.3f} (target {tl_t})  MeaningForge={mf:.3f} (target {mf_t})  Guardian={g_score:.1f} (target {g_t})")
open("open_data/validation/phase3_gate.txt","w").write(
    f"TL={tl:.3f}/{tl_t} MF={mf:.3f}/{mf_t} GUARDIAN={g_score:.1f}/{g_t} PASS={ok}\n"
)
sys.exit(0 if ok else 2)
PY

GATE=$?
cat "$VAL_DIR/phase3_gate.txt" || true

# 5) Append log
TL_LINE=$(sed -n '1p' "$VAL_DIR/phase3_gate.txt" 2>/dev/null || echo "")
{
  echo "### Phase 3 — Mapping & Validation  |  $STAMP"
  echo "- TruthLens:   ${TL_LINE%% MF=*}"
  echo "- MeaningForge & Guardian: $(echo "$TL_LINE" | sed 's/^.*MF=/MF=/')"
  echo "- mapping.yml: $MAP_OUT"
  echo "- Reports: $VAL_DIR/truthlens_summary.md, $VAL_DIR/meaningforge_summary.md, qc/guardian_v4/guardian_summary_v4.md"
  echo "- Status: $([ $GATE -eq 0 ] && echo "✅ PASS" || echo "❌ REVIEW")"
  echo
} >> "$LOG"

# 6) Exit messaging
if [ $GATE -eq 0 ]; then
  echo "✅ Phase 3 PASS. You may proceed to Phase 4:"
  echo "python3 originchain.py --merge guardian truthlens meaningforge --out origin_output.json"
else
  echo "❌ Phase 3 gate failed. See summaries in $VAL_DIR and qc/guardian_v4/. Fix & re-run."
fi
exit $GATE

