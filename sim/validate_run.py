"""Validate a run folder: find JSONL files and validate them against schema/ files."""

import os
import sys

from sim.io_utils import validate_jsonl_against_schema

SCHEMA_MAP = {
    "decisions.jsonl": "schema/decisions.schema.json",
    "integration.jsonl": "schema/integration.schema.json",
    "innovations.jsonl": "schema/innovations.schema.json",
    "culture.jsonl": "schema/culture.schema.json",
}


def validate_run(run_dir: str) -> dict[str, dict]:
    results = {}
    for filename, schema in SCHEMA_MAP.items():
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            results[filename] = validate_jsonl_against_schema(path, schema)
        else:
            results[filename] = {"found": False}
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sim/validate_run.py <run_dir>")
        sys.exit(2)
    run_dir = sys.argv[1]
    res = validate_run(run_dir)
    import json

    print(json.dumps(res, indent=2))
