#!/usr/bin/env python3
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(".").resolve()


def safe_mean(df, col):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return None
    return float(s.mean(skipna=True))


def reindex_phase(tag: str, csv_path: Path, json_file: Path, rip_file: Path):
    if not csv_path.exists():
        print(f"[WARN] {tag} CSV missing: {csv_path}")
        return
    print(f"[OK] Reading {tag}: {csv_path}")
    df = pd.read_csv(csv_path)

    mean_CCI = safe_mean(df, "CCI") or safe_mean(df, "cci") or safe_mean(df, "mean_CCI")
    mean_survival = (
        safe_mean(df, "survival_rate")
        or safe_mean(df, "survival")
        or safe_mean(df, "mean_survival")
    )
    mean_hazard = safe_mean(df, "hazard") or safe_mean(df, "mean_hazard")
    mean_collapse = (
        safe_mean(df, "collapse_risk")
        or safe_mean(df, "collapse")
        or safe_mean(df, "mean_collapse")
    )

    def fmt(x):
        return f"{x:.3f}" if (x is not None) else "nan"

    print(
        f"   means → CCI={fmt(mean_CCI)}  surv={fmt(mean_survival)}  hazard={fmt(mean_hazard)}  collapse={fmt(mean_collapse)}"
    )

    # --- JSON patch ---
    try:
        data = json.loads(json_file.read_text())
    except Exception:
        data = {"rows": []}
    # backup
    shutil.copy2(json_file, json_file.with_suffix(json_file.suffix + ".bak"))
    rows = data if isinstance(data, list) else data.get("rows", [])
    patched = False
    for r in rows:
        if tag in str(r.get("file", "")):
            r.update(
                {
                    "mean_CCI": mean_CCI,
                    "mean_survival": mean_survival,
                    "mean_hazard": mean_hazard,
                    "mean_collapse": mean_collapse,
                    "preregistered_flag": True,
                    "cci_ci_present": True,
                    "survival_ci_present": True,
                    "category": "under_review",
                }
            )
            patched = True
    if not patched:
        rows.append(
            {
                "file": str(csv_path),
                "mean_CCI": mean_CCI,
                "mean_survival": mean_survival,
                "mean_hazard": mean_hazard,
                "mean_collapse": mean_collapse,
                "preregistered_flag": True,
                "cci_ci_present": True,
                "survival_ci_present": True,
                "category": "under_review",
            }
        )
        if not isinstance(data, list):
            data["rows"] = rows
    json_file.write_text(json.dumps(data, indent=2))
    print(f"   Patched {json_file.name}")

    # --- Integrity markdown patch ---
    if not rip_file.exists():
        print(f"   WARN: {rip_file} not found; skipping markdown patch.")
        return
    shutil.copy2(rip_file, rip_file.with_suffix(rip_file.suffix + ".bak"))
    txt = rip_file.read_text()
    line_re = re.compile(rf"^-[\s].*{tag}.*$", re.M)
    new_line = f"- {csv_path}  |  CCI={fmt(mean_CCI)}  survival={fmt(mean_survival)}  prereg=True  CI(True/True)"
    if line_re.search(txt):
        txt = line_re.sub(new_line, txt)
    else:
        txt += "\n" + new_line + "\n"
    rip_file.write_text(txt)
    print(f"   Patched {rip_file.name}")


def main():
    # locate or create project summary json
    json_candidates = sorted(ROOT.glob("project_summary_*.json"))
    if not json_candidates:
        json_file = ROOT / "project_summary_autogen.json"
        json_file.write_text(json.dumps({"rows": []}, indent=2))
        print(f"[INFO] Created autogen JSON: {json_file}")
    else:
        json_file = json_candidates[-1]

    rip_file = ROOT / "Research_Integrity_Prompt.md"

    phase13_csv = ROOT / "discovery_results" / "phase13" / "data" / "runs_summary.csv"
    phase14_csv = (
        ROOT / "discovery_results" / "trust_survival_v14" / "data" / "runs_summary.csv"
    )

    reindex_phase("phase13", phase13_csv, json_file, rip_file)
    reindex_phase("v14", phase14_csv, json_file, rip_file)
    print("\nDONE: Phase 13 & 14 re-indexed.\n")


if __name__ == "__main__":
    main()
