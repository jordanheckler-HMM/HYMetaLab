#!/usr/bin/env python3
import json
import re
import shutil
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. pip install pandas", file=sys.stderr)
    sys.exit(1)

ROOT = Path(".").resolve()

# 1) Locate Phase 13 CSV
candidate_csvs = []
# canonical path from your integrity sheet
candidate_csvs.append(
    ROOT / "discovery_results" / "phase13" / "data" / "runs_summary.csv"
)
# also pick up trust_survival_v13* variants if present
for p in (ROOT / "discovery_results").glob("trust_survival_v13*/data/runs_summary.csv"):
    candidate_csvs.append(p)

phase13_csv = next((p for p in candidate_csvs if p.exists()), None)
if not phase13_csv:
    print(
        "ERROR: Could not find Phase 13 runs_summary.csv in expected locations.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[OK] Found Phase 13 CSV: {phase13_csv}")

# 2) Load and compute means (guarding NaNs & alt column names)
df = pd.read_csv(phase13_csv)


def pick_col(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None


col_cci = pick_col(["CCI", "cci", "mean_CCI"])
col_surv = pick_col(["survival_rate", "survival", "mean_survival"])
col_haz = pick_col(["hazard", "mean_hazard"])
col_col = pick_col(["collapse_risk", "collapse", "mean_collapse"])


def safe_mean(series):
    try:
        m = pd.to_numeric(series, errors="coerce").mean()
        return None if pd.isna(m) else float(m)
    except Exception:
        return None


mean_CCI = safe_mean(df[col_cci]) if col_cci else None
mean_survival = safe_mean(df[col_surv]) if col_surv else None
mean_hazard = safe_mean(df[col_haz]) if col_haz else None
mean_collapse = safe_mean(df[col_col]) if col_col else None

print(
    f"[OK] Means → CCI={mean_CCI}  survival={mean_survival}  hazard={mean_hazard}  collapse={mean_collapse}"
)

# 3) Detect prereg/CI flags from report if available
report_dir = phase13_csv.parents[1] / "report"
prereg = False
ci_cci = False
ci_surv = False

if report_dir.exists():
    # heuristic: look for validation_summary.json or md
    js = list(report_dir.glob("**/*validation*.json"))
    if js:
        try:
            jd = json.loads(js[0].read_text())
            prereg = bool(jd.get("preregistered", jd.get("preregistered", False)))
            ci_cci = bool(jd.get("ci", {}).get("CCI", False))
            ci_surv = bool(jd.get("ci", {}).get("survival_rate", False))
        except Exception:
            pass
    # fallback: scan markdown for simple flags
    mds = list(report_dir.glob("**/*summary*.md")) + list(report_dir.glob("**/*.md"))
    for mdp in mds:
        try:
            txt = mdp.read_text()
            if re.search(r"prereg(istered)?\s*[:=]\s*(true|yes)", txt, re.I):
                prereg = True
            if re.search(r"CI[^a-zA-Z]*(CCI)\s*(True|Pass)", txt, re.I):
                ci_cci = True
            if re.search(r"CI[^a-zA-Z]*(survival)", txt, re.I):
                ci_surv = True
        except Exception:
            pass

print(f"[OK] Flags → prereg={prereg}  CI(CCI)={ci_cci}  CI(Survival)={ci_surv}")

# 4) Patch latest project_summary_*.json (the one you uploaded most recently)
summary_files = sorted((ROOT).glob("project_summary_*.json"))
if not summary_files:
    print("WARN: No project_summary_*.json found to patch; skipping JSON update.")
else:
    target_json = summary_files[-1]
    data = json.loads(target_json.read_text())
    # back up
    shutil.copy2(target_json, target_json.with_suffix(".json.bak"))
    # find the phase13 entry
    patched = False
    for item in data if isinstance(data, list) else data.get("rows", []):
        # support both list and {rows:[...]} formats
        rec = item if isinstance(data, list) else item
        file_path = str(rec.get("file", ""))
        if (
            "discovery_results/phase13/data/runs_summary.csv" in file_path
            or "trust_survival_v13" in file_path
        ):
            if mean_CCI is not None:
                rec["mean_CCI"] = round(mean_CCI, 6)
            if mean_survival is not None:
                rec["mean_survival"] = round(mean_survival, 6)
            if mean_hazard is not None:
                rec["mean_hazard"] = round(mean_hazard, 6)
            if mean_collapse is not None:
                rec["mean_collapse"] = round(mean_collapse, 6)
            rec["preregistered_flag"] = bool(prereg)
            rec["cci_ci_present"] = bool(ci_cci)
            rec["survival_ci_present"] = bool(ci_surv)
            rec["category"] = rec.get("category", "under_review")
            patched = True
    if not patched:
        print("WARN: Did not find a Phase 13 row in the JSON; appending a new one.")
        new_row = {
            "file": str(phase13_csv),
            "mean_CCI": mean_CCI,
            "mean_survival": mean_survival,
            "mean_hazard": mean_hazard,
            "mean_collapse": mean_collapse,
            "preregistered_flag": prereg,
            "cci_ci_present": ci_cci,
            "survival_ci_present": ci_surv,
            "category": "under_review",
        }
        if isinstance(data, list):
            data.append(new_row)
        else:
            data.setdefault("rows", []).append(new_row)

    target_json.write_text(json.dumps(data, indent=2))
    print(f"[OK] Patched JSON: {target_json}")

# 5) Update Research_Integrity_Prompt.md line for Phase 13
rip = ROOT / "Research_Integrity_Prompt.md"
if not rip.exists():
    print(
        "WARN: Research_Integrity_Prompt.md not found; skipping integrity text patch."
    )
else:
    bak = rip.with_suffix(".md.bak")
    shutil.copy2(rip, bak)
    text = rip.read_text()

    # pattern for the phase13 line
    line_re = re.compile(
        r"^-[\s].*discovery_results/phase13/data/runs_summary\.csv.*$", re.M
    )

    def make_line():
        cci = "nan" if mean_CCI is None else f"{mean_CCI:.3f}"
        surv = "nan" if mean_survival is None else f"{mean_survival:.3f}"
        prereg_s = "True" if prereg else "False"
        ci_s = f"{'True' if (ci_cci and ci_surv) else 'False'}/{'True' if ci_surv else 'False'}"
        return f"- /Users/jordanheckler/conciousness_proxy_sim copy 6/discovery_results/phase13/data/runs_summary.csv  |  CCI={cci}  survival={surv}  prereg={prereg_s}  CI({ci_cci}/{ci_surv})"

    if line_re.search(text):
        text = line_re.sub(make_line(), text)
    else:
        # insert under "**Under Review" section if possible
        ur_re = re.compile(r"(\*\*Under Review.*?\n)", re.S)
        m = ur_re.search(text)
        if m:
            insert_at = m.end()
            text = text[:insert_at] + make_line() + "\n" + text[insert_at:]
        else:
            text += "\n" + make_line() + "\n"

    rip.write_text(text)
    print(f"[OK] Patched integrity file: {rip} (backup: {bak})")

print("\nDONE: Phase 13 metrics re-indexed and integrity entries repaired.\n")
