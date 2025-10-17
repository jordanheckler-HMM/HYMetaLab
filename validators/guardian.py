#!/usr/bin/env python3
"""Simple Guardian validator wrapper

Exposes run(outdir) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path


def run(outdir: str | Path) -> dict:
    outdir = Path(outdir)
    # Look for a file named guardian_input.json or any output.json
    candidate = outdir / "guardian_input.json"
    result = {"version": "v1", "primary_score": 0.9}
    if candidate.exists():
        data = json.loads(candidate.read_text())
        # simple heuristic
        result["primary_score"] = 1.0 - float(data.get("harm_score", 0.0))
        result.update(data)
    else:
        # fallback: compute a naive safety score from any json file
        for p in outdir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                if "harm_score" in data or "safety_score" in data:
                    result.update(data)
                    break
            except Exception:
                continue

    # Write guardian output
    out = outdir / "guardian_output.json"
    out.write_text(json.dumps(result, indent=2))
    return result
