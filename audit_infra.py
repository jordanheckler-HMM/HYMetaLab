#!/usr/bin/env python3
"""Infrastructure audit script

Scans the repository for Python scripts and common backup files, attempts to
compile each .py file (syntax check), and writes a QC report JSON plus a
human-readable `infrastructure_audit.md` under `outputs/infra/`.
"""
from __future__ import annotations

import json
import os
import py_compile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs" / "infra"
IGNORED_DIRS = {"outputs", "venv", ".venv", ".git", "__pycache__"}


def is_backup(fname: str) -> bool:
    lowers = fname.lower()
    return any(
        lowers.endswith(ext) for ext in (".bak", ".old", ".backup", "~")
    ) or fname.endswith(".swp")


def scan_files():
    py_files = []
    backups = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # skip ignored dirs
        parts = set(Path(dirpath).parts)
        if parts & IGNORED_DIRS:
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            if fn.endswith(".py"):
                py_files.append(str(p.relative_to(ROOT)))
            elif is_backup(fn):
                backups.append(str(p.relative_to(ROOT)))
    return py_files, backups


def qc_compile(py_files: list[str]):
    results = {}
    for rel in py_files:
        p = ROOT / rel
        try:
            py_compile.compile(str(p), doraise=True)
            results[rel] = {"status": "pass"}
        except Exception as e:
            results[rel] = {"status": "fail", "error": str(e)}
    return results


def write_reports(py_files, backups, qc_results):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "py_count": len(py_files),
        "backup_count": len(backups),
        "qc": qc_results,
    }
    qc_path = OUTDIR / "qc_report.json"
    qc_path.write_text(json.dumps(report, indent=2))

    # human readable audit
    md_lines = [
        "# Infrastructure Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Python files: {report['py_count']}",
        f"Backup-like files: {report['backup_count']}",
        "",
        "## QC Summary",
        "",
    ]
    passes = sum(1 for v in qc_results.values() if v.get("status") == "pass")
    fails = len(qc_results) - passes
    md_lines.append(f"Passing modules: {passes}")
    md_lines.append(f"Failing modules: {fails}")
    md_lines.append("")
    if fails > 0:
        md_lines.append("## Failing modules details")
        for k, v in qc_results.items():
            if v.get("status") == "fail":
                md_lines.append(f"- {k}: {v.get('error')}")

    md_lines.append("")
    md_lines.append("## Inventory sample")
    for p in py_files[:50]:
        md_lines.append(f"- {p}")
    md_path = OUTDIR / "infrastructure_audit.md"
    md_path.write_text("\n".join(md_lines))
    return qc_path, md_path


def main():
    py_files, backups = scan_files()
    qc_results = qc_compile(py_files)
    qc_path, md_path = write_reports(py_files, backups, qc_results)
    print(f"Wrote QC report {qc_path} and audit {md_path}")


if __name__ == "__main__":
    main()
