#!/usr/bin/env python3
"""Simple MeaningForge validator wrapper

Exposes run(outdir) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path


def run(outdir: str | Path) -> dict:
    outdir = Path(outdir)
    result = {"version": "v1", "meaning_quotient": 0.75}
    candidate = outdir / "meaning_input.json"
    if candidate.exists():
        data = json.loads(candidate.read_text())
        result["meaning_quotient"] = float(
            data.get("meaning_quotient", result["meaning_quotient"])
        )
        result.update(data)

    out = outdir / "meaningforge_output.json"
    out.write_text(json.dumps(result, indent=2))
    return result
