#!/usr/bin/env python3
"""
HYMetaLab / Heck Yeah Simulation Research Lab — Agent Charter

This module provides a small validator/enricher that implements the lab's
operational standards and integrity checks for OpenLaws studies. It's
lightweight and intentionally conservative: it emits warnings and will
auto-fill non-destructive defaults (deterministic seeds, prereg flags,
exports) to improve reproducibility.

Usage:
  from openlaws_agent_charter import validate_and_enforce, CHARTER_TEXT
  validate_and_enforce(path_to_study_yaml)
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

CHARTER_TEXT = """
HYMetaLab / Heck Yeah Simulation Research Lab — Agent Charter

Role:
You are a Research Agent within the Heck Yeah Simulation Research Initiative (HYMetaLab).
Mission: uphold scientific integrity, reproducibility, and alignment while contributing
to OpenLaws experiments.

Key operational rules enforced by this helper:
- Deterministic seeds must be present (recommended: [11,17,23,29]).
- Preregistration flag or prereg text should be present in the YAML.
- Exports must include CSV/JSON/PNG and output directories under discovery_results/.
- Standard metrics (CCI, survival_rate, hazard, collapse_risk) should be requested.
- Integrity artifacts (summary.json, sha256) are suggested.

This file contains a non-intrusive validator that will emit warnings and
optionally write back minor defaults when safe.
"""


@dataclass
class CharterReport:
    modified: bool
    messages: dict[str, str]


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_and_enforce(study_path: Path, write_back: bool = True) -> CharterReport:
    """Load a study YAML, validate against HYMetaLab charter, and optionally
    write-back minimally safe defaults. Returns a short report object.

    Non-destructive defaults added:
    - constants.seeds -> [11,17,23,29] if missing
    - exports.outdir -> discovery_results/<study_id> if missing
    - preregistered_flag -> true if prereg text exists
    - exports.require_columns includes core metrics
    """
    p = Path(study_path)
    if not p.exists():
        return CharterReport(False, {"error": f"{study_path} not found"})

    orig = p.read_text(encoding="utf-8")
    try:
        cfg = yaml.safe_load(orig) or {}
    except Exception as e:
        return CharterReport(False, {"error": f"YAML parse error: {e}"})

    msgs = {}
    modified = False

    study_id = cfg.get("study_id") or Path(p).stem

    # Ensure constants.seeds
    const = cfg.setdefault("constants", {})
    if "seeds" not in const or not const.get("seeds"):
        const["seeds"] = [11, 17, 23, 29]
        msgs["seeds"] = "Inserted deterministic seeds [11,17,23,29]"
        modified = True

    # Ensure exports.outdir
    exports = cfg.setdefault("exports", {})
    outdir = exports.get("outdir")
    if not outdir:
        exports["outdir"] = f"discovery_results/{study_id}/"
        msgs["outdir"] = f"Set exports.outdir to discovery_results/{study_id}/"
        modified = True

    # Ensure required columns list includes core metrics
    req = exports.get("require_columns") or []
    core = ["CCI", "survival_rate", "hazard", "collapse_risk"]
    added = []
    for m in core:
        if m not in req:
            req.append(m)
            added.append(m)
    if added:
        exports["require_columns"] = req
        msgs["require_columns"] = f"Added required metrics: {added}"
        modified = True

    # Preregistration detection: look for prereg text
    prereg = cfg.get("hypothesis") or cfg.get("prereg") or {}
    prereg_flag = cfg.get("preregistered_flag")
    if not prereg_flag:
        if isinstance(prereg, dict) and prereg.get("text"):
            cfg["preregistered_flag"] = True
            msgs["prereg"] = "Set preregistered_flag=True (hypothesis.text present)"
            modified = True

    # Integrity suggestions: ensure there's a place to write a summary
    arch = cfg.setdefault("validation", {})
    if "integrity_artifacts" not in arch:
        arch["integrity_artifacts"] = ["summary.json", "sha256.txt"]
        msgs["integrity"] = (
            "Suggested integrity artifacts added to validation.integrity_artifacts"
        )
        modified = True

    # If write_back, write YAML back with minimal changes
    if modified and write_back:
        try:
            p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            msgs["written"] = f"Wrote back changes to {study_path}"
        except Exception as e:
            msgs["write_error"] = f"Failed to write back: {e}"

    # Compute a short integrity token for reference
    token = _sha256_of_text(yaml.safe_dump(cfg, sort_keys=True))
    msgs["integrity_token"] = token[:10]

    return CharterReport(modified, msgs)


def print_charter_summary(report: CharterReport):
    print("[HYMetaLab Charter] Validation Summary:")
    for k, v in report.messages.items():
        print(f" - {k}: {v}")
    if report.modified:
        print("[HYMetaLab Charter] Study YAML modified with safe defaults.")
