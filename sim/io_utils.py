"""IO utilities: manifest writer, JSONL writer, and schema validator"""

import json
import os
import platform
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import Any

try:
    from jsonschema import ValidationError, validate  # type: ignore

    _HAVE_JSONSCHEMA = True
except Exception:
    validate = None
    ValidationError = Exception
    _HAVE_JSONSCHEMA = False

ROOT = (
    os.path.dirname(os.path.dirname(__file__))
    if __file__.endswith("io_utils.py")
    else "."
)


def write_run_manifest(export_dir: str, config: dict[str, Any], seed: int) -> str:
    os.makedirs(export_dir, exist_ok=True)
    manifest = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "config": config,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    path = os.path.join(export_dir, "run_manifest.json")
    # Try to capture git commit and pip freeze for provenance; ignore failures
    try:
        import subprocess

        git_rev = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        manifest["git_commit"] = git_rev
    except Exception:
        manifest["git_commit"] = None
    try:
        import subprocess

        pip_freeze = (
            subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
            )
            .decode()
            .splitlines()
        )
        manifest["pip_freeze"] = pip_freeze
    except Exception:
        manifest["pip_freeze"] = None

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def write_jsonl(path: str, records: Iterable[dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def validate_jsonl_against_schema(jsonl_path: str, schema_path: str) -> dict[str, Any]:
    results = {"total": 0, "valid": 0, "invalid": 0, "errors": []}
    # If jsonschema is available, use it. Otherwise run a lightweight fallback
    with open(schema_path) as sf:
        schema = json.load(sf)
    required = schema.get("required", [])
    props = schema.get("properties", {})

    with open(jsonl_path) as jf:
        for line_no, line in enumerate(jf, start=1):
            line = line.strip()
            if not line:
                continue
            results["total"] += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                results["invalid"] += 1
                results["errors"].append(
                    {"line": line_no, "error": f"json decode error: {e}"}
                )
                continue

            if _HAVE_JSONSCHEMA:
                try:
                    validate(instance=doc, schema=schema)
                    results["valid"] += 1
                except ValidationError as e:
                    results["invalid"] += 1
                    results["errors"].append({"line": line_no, "error": str(e)})
            else:
                # Lightweight fallback: check required fields and some basic types/ranges
                missing = [f for f in required if f not in doc]
                if missing:
                    results["invalid"] += 1
                    results["errors"].append(
                        {"line": line_no, "error": f"missing required: {missing}"}
                    )
                    continue

                # basic type checks for a few well-known fields
                ok = True
                if "tick" in doc and not isinstance(doc["tick"], int):
                    ok = False
                    results["errors"].append({"line": line_no, "error": "tick not int"})
                if "reported_conf" in doc:
                    v = doc["reported_conf"]
                    if not (isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0):
                        ok = False
                        results["errors"].append(
                            {"line": line_no, "error": "reported_conf not in [0,1]"}
                        )
                if ok:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
    return results
    return results
