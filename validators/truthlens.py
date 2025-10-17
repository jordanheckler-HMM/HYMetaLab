#!/usr/bin/env python3
"""Simple TruthLens validator wrapper

Exposes run(outdir) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path


def run(outdir: str | Path) -> dict:
    outdir = Path(outdir)
    result = {"version": "v1", "truth_score": 0.8}
    # If there's a file named truth_input.json, read and process
    candidate = outdir / "truth_input.json"
    if candidate.exists():
        data = json.loads(candidate.read_text())
        result["truth_score"] = float(data.get("truth_score", result["truth_score"]))
        result.update(data)

    out = outdir / "truthlens_output.json"
    out.write_text(json.dumps(result, indent=2))
    return result
