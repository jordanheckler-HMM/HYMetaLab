from pathlib import Path
import json

BASE = Path("dev/logs/guardian_reports")


def push_report(trace_id: str, report: dict) -> str:
    BASE.mkdir(parents=True, exist_ok=True)
    fp = BASE / f"{trace_id}.json"
    fp.write_text(json.dumps(report, indent=2))
    # Simulate a Drive URI; Phase II can swap to real Drive API
    return f"drive://guardian/reports/{trace_id}.json"
