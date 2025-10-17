#!/usr/bin/env python3
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- Config -------------------------------------------------------------------
SOURCE_DIR = "./discovery_results/"
ARCHIVE_DIR = "./project_archive/"
CATEGORIES = ["validated", "under_review", "hypothesis_only"]
REPORT_FILE = "Research_Integrity_Prompt.md"
DATESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M")

# Move a copy of each summary into its category folder? (keeps originals intact)
COPY_SUMMARIES = True

Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)
for cat in CATEGORIES:
    Path(f"{ARCHIVE_DIR}/{cat}").mkdir(parents=True, exist_ok=True)


# --- Header matching helpers --------------------------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


# Map of canonical metric -> list of regex patterns that count as a match
HEADER_MAP = {
    "CCI": [
        r"(^|_)cci($|_)",
        r"(final|mean|avg).*cci",
        r"cci_?(mean|final|avg)",
    ],
    "survival_rate": [
        r"(^|_)survival(_|$)",
        r"(^|_)survival_rate(_|$)",
        r"(avg|mean).*survival",
    ],
    "collapse_risk": [
        r"(^|_)collapse(_|$)",
        r"(^|_)collapse_risk(_|$)",
        r"collapse.*(risk|prob)",
        r"(hazard_)?gini",
    ],
    "hazard": [
        r"(^|_)hazard(_|$)",
        r"(stability_)?hazard(_mean)?",
    ],
    # Integrity signals (optional but help raise “validated”)
    "cci_ci": [
        r"cci.*(ci|se|stderr|std_err|conf)",
        r"(ci|se).*cci",
    ],
    "survival_ci": [
        r"survival.*(ci|se|stderr|std_err|conf)",
        r"(ci|se).*survival",
    ],
    "preregistered": [
        r"(^|_)prereg(ister(ed)?|_flag|_status)?$",
        r"(^|_)locked_constants$",
    ],
}


def find_column(df: pd.DataFrame, key: str):
    """Return the best-matching column name (or None) for a canonical key."""
    patterns = [re.compile(p) for p in HEADER_MAP.get(key, [])]
    normalized = {_normalize(c): c for c in df.columns}
    # try exact normalized hits first
    for n, c in normalized.items():
        for pat in patterns:
            if pat.search(n):
                return c
    # fallback: try raw names (for odd cases)
    for c in df.columns:
        for pat in patterns:
            if pat.search(c.lower()):
                return c
    return None


def safe_mean(series: pd.Series):
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(s.mean())
    return None


def has_confidence(df: pd.DataFrame, key_main: str, key_ci: str) -> bool:
    cm = find_column(df, key_main)
    cc = find_column(df, key_ci)
    if not cm or not cc:
        return False
    v = pd.to_numeric(df[cm], errors="coerce")
    ci = pd.to_numeric(df[cc], errors="coerce")
    return v.notna().any() and ci.notna().any()


def prereg_status(df: pd.DataFrame) -> bool:
    col = find_column(df, "preregistered")
    if not col:
        return False
    vals = df[col].astype(str).str.lower()
    return vals.isin(["true", "1", "yes", "y", "locked", "preregistered"]).any()


# --- Scan & classify ----------------------------------------------------------
summary_files = list(Path(SOURCE_DIR).rglob("runs_summary.csv"))
records = []

for f in summary_files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        records.append({"file": str(f), "error": str(e), "category": "hypothesis_only"})
        continue

    # Resolve columns robustly
    cci_col = find_column(df, "CCI")
    surv_col = find_column(df, "survival_rate")
    coll_col = find_column(df, "collapse_risk")
    haz_col = find_column(df, "hazard")

    mean_CCI = safe_mean(df[cci_col]) if cci_col else None
    mean_survival = safe_mean(df[surv_col]) if surv_col else None
    mean_collapse = safe_mean(df[coll_col]) if coll_col else None
    mean_hazard = safe_mean(df[haz_col]) if haz_col else None

    # Integrity signals
    cci_has_ci = has_confidence(df, "CCI", "cci_ci")
    surv_has_ci = has_confidence(df, "survival_rate", "survival_ci")
    prereg = prereg_status(df)

    # Classification rules (conservative)
    # VALIDATED needs: prereg + CI/SE evidence + high means
    # UNDER_REVIEW: decent means but missing prereg OR CI
    # HYPOTHESIS_ONLY: everything else (or no usable metrics)
    category = "hypothesis_only"
    if (mean_CCI is not None) or (mean_survival is not None):
        if (
            (mean_CCI is not None and mean_CCI >= 0.70)
            and (mean_survival is not None and mean_survival >= 0.80)
            and (cci_has_ci or surv_has_ci)
            and prereg
        ):
            category = "validated"
        elif (mean_CCI is not None and mean_CCI >= 0.50) or (
            mean_survival is not None and mean_survival >= 0.65
        ):
            category = "under_review"

    rec = {
        "file": str(f),
        "category": category,
        "mean_CCI": round(mean_CCI, 4) if mean_CCI is not None else None,
        "mean_survival": round(mean_survival, 4) if mean_survival is not None else None,
        "mean_collapse": round(mean_collapse, 4) if mean_collapse is not None else None,
        "mean_hazard": round(mean_hazard, 4) if mean_hazard is not None else None,
        "cci_ci_present": bool(cci_has_ci),
        "survival_ci_present": bool(surv_has_ci),
        "preregistered_flag": bool(prereg),
        "detected_columns": {
            "CCI": cci_col,
            "survival_rate": surv_col,
            "collapse_risk": coll_col,
            "hazard": haz_col,
        },
    }
    records.append(rec)

    # Copy the summary into category folder for quick access
    if COPY_SUMMARIES:
        dest_dir = Path(ARCHIVE_DIR) / rec["category"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        # keep relative structure for readability
        rel = Path(f).relative_to(SOURCE_DIR)
        dest_path = dest_dir / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(f, dest_path)
        except Exception:
            pass

# --- Exports ------------------------------------------------------------------
summary_json = Path(ARCHIVE_DIR) / f"project_summary_{DATESTAMP}.json"
summary_csv = Path(ARCHIVE_DIR) / f"project_summary_{DATESTAMP}.csv"

with open(summary_json, "w") as js:
    json.dump(records, js, indent=2)

pd.DataFrame(records).to_csv(summary_csv, index=False)

validated = [r for r in records if r["category"] == "validated"]
under_review = [r for r in records if r["category"] == "under_review"]
hypothesis = [r for r in records if r["category"] == "hypothesis_only"]


def list_examples(items, n=5):
    lines = []
    for r in items[:n]:
        lines.append(
            f"- {r['file']}  |  CCI={r['mean_CCI']}  survival={r['mean_survival']}  prereg={r['preregistered_flag']}  CI(cci/surv)={r['cci_ci_present']}/{r['survival_ci_present']}"
        )
    if len(items) > n:
        lines.append(f"...(+{len(items)-n} more)")
    return "\n".join(lines) if items else "_none_"


prompt_md = f"""# Research Integrity & Organization Prompt
Date: {DATESTAMP}

## Purpose
Maintain scientific discipline in the Simulation Project.  
This file tells the Research Copilot what is **validated**, **under review**, and **hypothesis only**.

---

## ✅ VALIDATED FINDINGS
({len(validated)} datasets)
- Criteria: preregistered; CI/SE present (at least one of CCI or survival); stable high means (CCI ≥ 0.70, survival ≥ 0.80).
{list_examples(validated)}

---

## 🧩 UNDER REVIEW
({len(under_review)} datasets)
- Criteria: promising means (CCI ≥ 0.50 or survival ≥ 0.65) but missing prereg and/or CI/SE.
{list_examples(under_review)}

---

## 💭 HYPOTHESIS-ONLY
({len(hypothesis)} datasets)
- Criteria: exploratory ideas, incomplete metrics, or no integrity signals detected.
{list_examples(hypothesis)}

---

## Guidelines for Future Runs
1) Preregister hypotheses and constants; include a `preregistered` flag in summaries.
2) Export numeric columns with recognizable names or add CI/SE columns:
   - CCI, survival_rate, collapse_risk, hazard
   - cci_ci or cci_se; survival_ci or survival_se
3) Archive *all* runs (including null/failed) for transparency.
4) Distinguish correlation from causation; include effect sizes and CIs in reports.
5) Re-run this organizer after each experiment day to refresh categories.

*Generated by SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py*
"""

with open(Path(ARCHIVE_DIR) / REPORT_FILE, "w") as f:
    f.write(prompt_md)

print(f"[✓] Archive updated → {ARCHIVE_DIR}")
print(f"[✓] Summary JSON → {summary_json}")
print(f"[✓] Summary CSV  → {summary_csv}")
print(f"[✓] Integrity prompt → {Path(ARCHIVE_DIR) / REPORT_FILE}")
