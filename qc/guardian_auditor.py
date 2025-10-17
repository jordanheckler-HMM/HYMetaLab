#!/usr/bin/env python3
"""
HYMetaLab Guardian Auditor v2.0
Expanded metrics: Reproducibility Index, Language Safety, Alignment Score
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Import qc_linter for language safety analysis
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
try:
    from qc_linter import BANNED, banned_hits
except:
    BANNED = [
        r"\bprove[sd]?\b",
        r"\bconfirm(?:s|ed)?\b",
        r"\bdefinitive\b",
        r"\buniversal law\b",
        r"\bbreakthrough\b",
        r"\bparadigm shift\b",
    ]

    def banned_hits(text):
        hits = []
        for pat in BANNED:
            for m in re.finditer(pat, text, flags=re.I):
                hits.append((m.group(0), m.start()))
        return hits


def read_text_files(root: Path) -> list[Path]:
    """Read all documentation files"""
    exts = {".md", ".txt", ".yml", ".yaml", ".json"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def humility_score(text: str) -> tuple[int, list[str]]:
    """Compute epistemic humility score with violations list"""
    penalties = 0
    violations = []

    for pat in BANNED:
        matches = re.finditer(pat, text, flags=re.I)
        for m in matches:
            penalties += 10
            violations.append(f"{m.group(0)} at position {m.start()}")
            if len(violations) >= 10:  # Cap violations list
                break

    score = max(0, 40 - penalties)

    # Bonus for proper disclaimers
    if "simulation" in text.lower() and "scope" in text.lower():
        score += 10

    return min(score, 50), violations


def compute_language_safety_score(root: Path) -> tuple[int, dict]:
    """
    Compute Language Safety Score from qc_linter analysis
    Score: 0-100 based on banned phrase density
    """
    texts = read_text_files(root)
    total_chars = 0
    total_violations = 0
    file_violations = {}

    for p in texts:
        try:
            text = p.read_text(errors="ignore")
            total_chars += len(text)
            hits = banned_hits(text)
            if hits:
                total_violations += len(hits)
                file_violations[str(p.relative_to(root))] = len(hits)
        except:
            pass

    if total_chars == 0:
        return 50, {"note": "No text files found"}

    # Density = violations per 10k characters
    density = (total_violations / total_chars) * 10000

    # Score: 100 at 0 density, 0 at density ≥ 5
    score = max(0, min(100, 100 - (density * 20)))

    details = {
        "total_violations": total_violations,
        "total_characters": total_chars,
        "density_per_10k": round(density, 2),
        "worst_files": dict(
            sorted(file_violations.items(), key=lambda x: x[1], reverse=True)[:5]
        ),
    }

    return int(score), details


def compute_reproducibility_index(root: Path) -> tuple[int, dict]:
    """
    Compute Reproducibility Index: % of reruns within 95% CI
    Scans discovery_results for multi-seed runs
    """
    discovery = root / "discovery_results"
    if not discovery.exists():
        return 50, {"note": "No discovery_results directory"}

    # Group studies by base name (ignore seed/timestamp suffixes)
    studies = defaultdict(list)

    for summary_path in discovery.glob("*/summary.json"):
        try:
            data = json.loads(summary_path.read_text())
            study_dir = summary_path.parent.name

            # Extract base study name (remove timestamps, seeds)
            base_name = re.sub(r"_\d{8}_\d{6}$", "", study_dir)
            base_name = re.sub(r"_seed\d+$", "", base_name)

            # Extract key metrics
            delta_cci = data.get("delta_cci") or data.get("hypothesis_test", {}).get(
                "mean_CCI_gain"
            )
            delta_hazard = data.get("delta_hazard")

            if delta_cci is not None:
                studies[base_name].append(
                    {
                        "delta_cci": float(delta_cci),
                        "delta_hazard": float(delta_hazard) if delta_hazard else None,
                    }
                )
        except:
            pass

    # Analyze reproducibility
    reproducible_count = 0
    total_groups = 0
    group_details = {}

    for study_name, runs in studies.items():
        if len(runs) < 2:
            continue

        total_groups += 1
        cci_values = [r["delta_cci"] for r in runs]

        # Compute 95% CI
        mean_cci = np.mean(cci_values)
        std_cci = np.std(cci_values, ddof=1) if len(cci_values) > 1 else 0
        ci_95 = 1.96 * std_cci / np.sqrt(len(cci_values))

        # Check if all values within CI
        within_ci = all(abs(v - mean_cci) <= ci_95 for v in cci_values)

        if within_ci:
            reproducible_count += 1

        group_details[study_name] = {
            "n_runs": len(runs),
            "mean_cci": round(mean_cci, 4),
            "std": round(std_cci, 4),
            "ci_95": round(ci_95, 4),
            "reproducible": within_ci,
        }

    if total_groups == 0:
        score = 50  # Neutral score if no multi-run studies
        percentage = 0
    else:
        percentage = (reproducible_count / total_groups) * 100
        score = int(percentage)  # Direct mapping: 100% reproducible = 100 score

    details = {
        "reproducible_groups": reproducible_count,
        "total_groups": total_groups,
        "percentage": round(percentage, 1),
        "sample_details": dict(list(group_details.items())[:3]),
    }

    return score, details


def prereg_and_manifest_score(root: Path) -> tuple[int, dict]:
    """Enhanced preregistration and manifest scoring"""
    yaml_files = list((root / "studies").glob("*.yml"))
    man_files = list(root.glob("discovery_results/*/run_manifest.json"))

    yaml_score = 25 if yaml_files else 0
    manifest_score = 25 if man_files else 0

    details = {
        "preregistered_studies": len(yaml_files),
        "run_manifests": len(man_files),
        "yaml_files": [str(p.relative_to(root)) for p in yaml_files[:5]],
    }

    return yaml_score + manifest_score, details


def traffic_light_status(score: int) -> tuple[str, str]:
    """
    Return traffic light indicator
    Returns: (status, emoji)
    """
    if score >= 75:
        return "EXCELLENT", "🟢"
    elif score >= 50:
        return "GOOD", "🟡"
    elif score >= 25:
        return "NEEDS_WORK", "🟠"
    else:
        return "CRITICAL", "🔴"


def generate_markdown_report(report: dict) -> str:
    """Generate human-readable markdown report"""
    md = f"""# Guardian Auditor Report v2.0

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Overall Alignment Score:** {report['alignment_score']}/100 {traffic_light_status(report['alignment_score'])[1]}  
**Status:** {traffic_light_status(report['alignment_score'])[0]}

---

## Component Scores

| Component | Score | Status | Details |
|-----------|-------|--------|---------|
| **Epistemic Humility** | {report['components']['humility']}/50 | {traffic_light_status(report['components']['humility']*2)[1]} | {len(report['details']['humility_violations'])} violations |
| **Preregistration** | {report['components']['prereg_manifest']}/50 | {traffic_light_status(report['components']['prereg_manifest']*2)[1]} | {report['details']['prereg']['preregistered_studies']} studies |
| **Reproducibility Index** | {report['components']['reproducibility']}/50 | {traffic_light_status(report['components']['reproducibility']*2)[1]} | {report['details']['reproducibility']['percentage']}% reproducible |
| **Language Safety** | {report['components']['language_safety']}/50 | {traffic_light_status(report['components']['language_safety']*2)[1]} | {report['details']['language_safety']['density_per_10k']} violations/10k chars |

---

## Reproducibility Analysis

**Reproducible Study Groups:** {report['details']['reproducibility']['reproducible_groups']} / {report['details']['reproducibility']['total_groups']}

Sample Studies:
"""

    for study_name, details in (
        report["details"]["reproducibility"].get("sample_details", {}).items()
    ):
        status_emoji = "✅" if details["reproducible"] else "⚠️"
        md += f"\n- {status_emoji} **{study_name}**: {details['n_runs']} runs, mean ΔCCI = {details['mean_cci']} ± {details['ci_95']}"

    md += f"""

---

## Language Safety

**Violation Density:** {report['details']['language_safety']['density_per_10k']} per 10k characters  
**Total Violations:** {report['details']['language_safety']['total_violations']}

Worst Files:
"""

    for file, count in (
        report["details"]["language_safety"].get("worst_files", {}).items()
    ):
        md += f"\n- {file}: {count} violations"

    md += """

---

## Recommendations

"""

    score = report["alignment_score"]
    if score >= 75:
        md += "✅ **Excellent standing.** Continue current practices.\n"
    elif score >= 50:
        md += "🟡 **Good progress.** Address specific weaknesses identified above.\n"
    else:
        md += "🔴 **Immediate action required.** Review and implement recommendations below:\n"

    if report["components"]["humility"] < 40:
        md += "\n- **Humility:** Review banned phrase list and update documentation\n"
    if report["components"]["reproducibility"] < 40:
        md += (
            "\n- **Reproducibility:** Implement multi-seed validation for all studies\n"
        )
    if report["components"]["language_safety"] < 40:
        md += "\n- **Language Safety:** Run qc_linter.py and address flagged files\n"

    md += """

---

**Policy:** OpenLaws Protocol v1.0 - Epistemic Humility + Reproducibility  
**Next Audit:** Recommended within 30 days
"""

    return md


from collections import defaultdict


def main():
    root = Path(".").resolve()

    print("🔍 Guardian Auditor v2.0 - Comprehensive QC Analysis")
    print("=" * 60)

    # Component 1: Epistemic Humility
    print("📝 Analyzing epistemic humility...")
    texts = read_text_files(root)
    total_text = " ".join([p.read_text(errors="ignore") for p in texts])
    humility, violations = humility_score(total_text)

    # Component 2: Preregistration
    print("📋 Checking preregistration status...")
    prereg, prereg_details = prereg_and_manifest_score(root)

    # Component 3: Reproducibility Index (NEW)
    print("🔄 Computing reproducibility index...")
    repro_score, repro_details = compute_reproducibility_index(root)

    # Component 4: Language Safety (NEW)
    print("🛡️ Analyzing language safety...")
    lang_safety, lang_details = compute_language_safety_score(root)

    # Compute weighted alignment score
    # Weights: Humility 25%, Prereg 25%, Reproducibility 25%, Language Safety 25%
    alignment_score = int(
        humility * 0.5  # Humility max 50 → 25% of total
        + prereg * 0.5  # Prereg max 50 → 25% of total
        + repro_score * 0.25  # Reproducibility max 100 → 25% of total
        + lang_safety * 0.25  # Language Safety max 100 → 25% of total
    )

    report = {
        "version": "2.0",
        "generated": datetime.now().isoformat(),
        "alignment_score": alignment_score,
        "status": traffic_light_status(alignment_score)[0],
        "status_emoji": traffic_light_status(alignment_score)[1],
        "components": {
            "humility": humility,
            "prereg_manifest": prereg,
            "reproducibility": repro_score,
            "language_safety": lang_safety,
        },
        "details": {
            "humility_violations": violations[:10],  # Cap at 10
            "prereg": prereg_details,
            "reproducibility": repro_details,
            "language_safety": lang_details,
        },
        "policy": "OpenLaws Protocol v1.0 - Enhanced with Reproducibility & Language Safety",
        "recommendations": [],
    }

    # Generate recommendations
    if humility < 40:
        report["recommendations"].append(
            "Improve epistemic humility: address banned phrases"
        )
    if repro_score < 50:
        report["recommendations"].append(
            "Increase reproducibility: implement multi-seed validation"
        )
    if lang_safety < 50:
        report["recommendations"].append(
            "Improve language safety: run qc_linter and fix violations"
        )
    if prereg < 40:
        report["recommendations"].append("Add preregistered studies and run manifests")

    # Export JSON
    json_path = Path("qc/guardian_report.json")
    json_path.write_text(json.dumps(report, indent=2))

    # Export Markdown
    md_path = Path("qc/guardian_summary.md")
    md_content = generate_markdown_report(report)
    md_path.write_text(md_content)

    # Console output
    print("\n" + "=" * 60)
    print("✅ Guardian Audit Complete")
    print(f"📊 Overall Score: {alignment_score}/100 {report['status_emoji']}")
    print(f"📁 JSON Report: {json_path}")
    print(f"📄 Markdown Summary: {md_path}")
    print("=" * 60)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
