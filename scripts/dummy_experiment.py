#!/usr/bin/env python3
"""Dummy experiment that writes a few JSON outputs and a result.csv

Accepts --outdir
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    guardian = {"version": "v1", "harm_score": 0.12, "safety_score": 0.88}
    truth = {"version": "v1", "truth_score": 0.91}
    meaning = {"version": "v1", "meaning_quotient": 0.83}

    (outdir / "guardian_input.json").write_text(json.dumps(guardian))
    (outdir / "truth_input.json").write_text(json.dumps(truth))
    (outdir / "meaning_input.json").write_text(json.dumps(meaning))

    # results csv
    csv = outdir / "results.csv"
    csv.write_text("metric,value\naccuracy,0.99\nloss,0.01\n")

    # a small text summary
    summary = outdir / "summary.txt"
    summary.write_text(f"Dummy experiment run at {datetime.utcnow().isoformat()}Z\n")

    print(f"Wrote dummy outputs to {outdir}")


if __name__ == "__main__":
    main()
