#!/usr/bin/env python3
"""Lab Techs runner: execute an experiment, run validators, and record artifacts

Usage: python lab_techs_runner.py --script <path> --outdir <outdir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent


def run_script(script: Path, outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    # Run as a subprocess to avoid importing user code
    # Use the same Python interpreter that's running this script
    cmd = [sys.executable, str(script), "--outdir", str(outdir)]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def gather_artifacts(outdir: Path) -> List[Path]:
    if not outdir.exists():
        return []
    return [p for p in outdir.rglob("*") if p.is_file()]


def write_checksums(paths: List[Path], checksums_csv: Path):
    now = datetime.utcnow().isoformat() + "Z"
    lines = []
    for p in paths:
        s = sha256_of_file(p)
        lines.append(f"{p},{s},{now}")
    checksums_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "filepath,sha256,timestamp"
    if not checksums_csv.exists():
        checksums_csv.write_text(header + "\n" + "\n".join(lines) + "\n")
    else:
        with checksums_csv.open("a") as f:
            for line in lines:
                f.write(line + "\n")


def run_validators(outdir: Path) -> Dict[str, dict]:
    # Import local validators package
    validators_dir = ROOT / "validators"
    results = {}
    if not validators_dir.exists():
        print("No validators found; skipping validation")
        return results

    # Import by path
    import importlib.util

    for mod in ["guardian", "truthlens", "meaningforge"]:
        path = validators_dir / f"{mod}.py"
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(mod, str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore
        try:
            res = m.run(str(outdir))
            results[mod] = res
        except Exception as e:
            results[mod] = {"error": str(e)}

    # Write combined guardian output for downstream tools
    out_json = outdir / "validators_combined.json"
    out_json.write_text(json.dumps(results, indent=2))
    return results


def append_task_log(logpath: Path, entry: str):
    timestamp = datetime.utcnow().isoformat() + "Z"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open("a") as f:
        f.write(f"[{timestamp}] {entry}\n")


def write_run_report(outdir: Path, validators: Dict[str, dict], report_path: Path):
    lines = ["# Run Report", "", f"Run directory: {outdir}", ""]
    lines.append("## Validator summaries")
    lines.append("")
    for k, v in validators.items():
        lines.append(f"### {k}")
        lines.append("```json")
        lines.append(json.dumps(v, indent=2))
        lines.append("```")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Lab Techs runner")
    parser.add_argument("--script", required=True, help="Experiment script to run")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--log", default="Task_Log.md", help="Task log path")
    parser.add_argument(
        "--checksums", default="Checksums.csv", help="Checksums registry"
    )
    args = parser.parse_args()

    script = Path(args.script)
    outdir = Path(args.outdir)
    logpath = Path(args.log)
    checksums = Path(args.checksums)

    append_task_log(logpath, f"START run script={script} outdir={outdir}")
    rc = run_script(script, outdir)
    if rc != 0:
        append_task_log(logpath, f"ERROR script returned {rc}")
        print(f"Script failed with code {rc}")
        return rc

    artifacts = gather_artifacts(outdir)
    write_checksums(artifacts, checksums)
    validators = run_validators(outdir)
    report_path = outdir / "Run_Report.md"
    write_run_report(outdir, validators, report_path)
    append_task_log(
        logpath, f"END run script={script} outdir={outdir} artifacts={len(artifacts)}"
    )
    print(f"Run complete. Artifacts: {len(artifacts)}. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
