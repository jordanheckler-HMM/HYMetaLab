# qc/guardian_v4/api.py
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time


class GuardianError(Exception):
    pass


def _repo_root() -> pathlib.Path:
    # This file is qc/guardian_v4/api.py → repo_root = 3 levels up
    return pathlib.Path(__file__).resolve().parents[2]


def _guardian_paths(repo_root: pathlib.Path):
    report = repo_root / "qc/guardian_v4/guardian_report_v4.json"
    summary = repo_root / "qc/guardian_v4/guardian_summary_v4.md"
    main_py = repo_root / "qc/guardian_v4/guardian_v4.py"
    return main_py, report, summary


def evaluate_text(text: str, timeout_s: int = 20) -> dict:
    """
    Run Guardian v4 CLI reliably from repo root, return parsed metrics.
    """
    repo_root = _repo_root()
    main_py, report_path, summary_md = _guardian_paths(repo_root)

    if not main_py.exists():
        raise GuardianError(f"Missing Guardian entrypoint: {main_py}")

    # Write a temp markdown file for the CLI to validate
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(text if text.strip() else "Placeholder text for Guardian self-test.")
        temp_path = f.name

    # Build command (must run from repo root so relative qc/ paths resolve)
    cmd = [sys.executable, str(main_py), "--validate", "--file", temp_path, "--report"]

    try:
        run = subprocess.run(
            cmd,
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise GuardianError(f"Guardian CLI timed out after {timeout_s}s")
    except subprocess.CalledProcessError as e:
        # Surface useful stderr to Streamlit
        err = (e.stderr or "")[-1200:]
        out = (e.stdout or "")[-400:]
        raise GuardianError(f"Guardian CLI failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}")
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except:
            pass

    # Wait for the report to appear
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if report_path.exists() and report_path.stat().st_size > 0:
            break
        time.sleep(0.2)
    if not report_path.exists():
        raise GuardianError(f"guardian_report_v4.json not found at {report_path}")

    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        raise GuardianError(f"Failed to parse report JSON: {e}")

    summary_text = summary_md.read_text() if summary_md.exists() else ""

    # Normalize keys (tolerate minor schema diffs)
    metrics = data.get("metrics", {})
    risk_assessment = data.get("risk_assessment", {})

    score = (
        data.get("guardian_alignment_score")
        or data.get("score")
        or data.get("alignment_score")
        or 0.0
    )
    obj = metrics.get("objectivity_score", 0.0)
    trn = metrics.get("transparency_index_v2") or metrics.get("transparency", 0.0)
    ls = metrics.get("language_safety_score", 0.0)
    sent = metrics.get("sentiment_neutrality", 0.0)
    risk = risk_assessment.get("risk_level", "unknown")

    return {
        "summary_text": summary_text,
        "raw": data,
        "score": score,
        "objectivity": obj,
        "transparency": trn,
        "language_safety": ls,
        "sentiment": sent,
        "risk_level": risk,
    }
