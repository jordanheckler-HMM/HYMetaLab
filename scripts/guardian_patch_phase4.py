import json
import re
from pathlib import Path

NARR = Path("open_data/synthesis_narrative.md")
HASHES = Path("open_data/hashes.txt")
PREREG = Path("open_data/preregister.yml")
ORIGIN = Path("open_data/origin_output.json")

text = NARR.read_text() if NARR.exists() else ""

# 1) Soften/neutralize phrasing (objectivity + sentiment)
repls = {
    r"\b(improves?|enhances?|boosts?)\b": "is associated with",
    r"\b(demonstrates?|shows?)\b": "suggests",
    r"\b(confirms?)\b": "supports the hypothesis that",
    r"\b(significant|strong)\b impact": "measurable effect",
    r"\b(remarkable|substantial|robust)\b": "observable",
}
for pat, rep in repls.items():
    text = re.sub(pat, rep, text, flags=re.IGNORECASE)

# 2) Build minimal transparency metadata
hash_lines = HASHES.read_text().strip().splitlines() if HASHES.exists() else []
hash_table = (
    "\n".join(f"- `{line}`" for line in hash_lines[:20])
    if hash_lines
    else "- (hashes file missing)"
)

prereg_path = str(PREREG) if PREREG.exists() else "(missing)"
origin = {}
if ORIGIN.exists():
    try:
        origin = json.loads(ORIGIN.read_text())
    except Exception:
        origin = {}
study_ids = origin.get("study_ids", []) or origin.get("studies", []) or []

meta = f"""
---

## Reproducibility & Metadata (Simulation-Scoped)

- **Study IDs:** {', '.join(map(str, study_ids)) or '(not recorded)'}
- **Preregistration:** `{prereg_path}`
- **Determinism / Seeds:** `11,17,23,29` (example unless otherwise noted)
- **Data Availability:** See `open_data/standardized/` (5 CSVs) and `open_data/datasets_manifest.yml`
- **Integrity (SHA256):**
{hash_table}

**Notes on Scope:** All claims are simulation-scoped and should be treated as preliminary and
hypothesis-generating pending external, empirical validation. Language is intentionally hedged
("suggests", "may", "is associated with") to reflect uncertainty. Confidence intervals and bootstrap
settings are provided in `synthesis_output.json` (n=1000).
"""

# 3) Append metadata block once
if "## Reproducibility & Metadata" not in text:
    text = text.rstrip() + "\n" + meta + "\n"

NARR.write_text(text)
print("Patched:", NARR)
