#!/usr/bin/env python3
"""
Zenodo Link Guard — Automatic Status Poller
Checks if Zenodo DOI is live and auto-updates dashboard_data.json
No API token required; uses simple DOI resolver check.

Usage:
  python3 tools/zenodo_link_guard.py

Or schedule via cron:
  */15 * * * * cd /path/to/repo && python3 tools/zenodo_link_guard.py >> logs/zenodo_guard.log 2>&1
"""

import json
import pathlib
import sys
import urllib.request
from datetime import datetime

dash = pathlib.Path("dashboard_data.json")
data = json.loads(dash.read_text())

# prefer version doi if present, otherwise concept
doi = data.get("zenodo_version_doi") or data.get("zenodo_concept_doi")
if not doi:
    print("No DOI found in dashboard_data.json")
    sys.exit(0)

# naive check: does the resolver return something?
url = f"https://doi.org/{doi.split('doi.org/')[-1].replace('https://','')}"
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        ok = 200 <= r.status < 400
except Exception as e:
    ok = False
    print(f"DOI check failed: {e}")

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

if ok and data.get("zenodo_status") != "published":
    data["zenodo_status"] = "published"
    # if we were holding only a concept DOI but it resolves, treat as live
    if not data.get("zenodo_version_doi"):
        data["zenodo_version_doi"] = data["zenodo_concept_doi"]
    dash.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[{timestamp}] ✅ Zenodo status updated to published.")

    # Log to integrity file
    log_entry = f"\n- {datetime.utcnow().strftime('%Y-%m-%d')}: Zenodo DOI published and verified live (auto-detected).\n"
    log_path = pathlib.Path("docs/integrity/SHA256_log.md")
    if log_path.exists():
        with open(log_path, "a") as f:
            f.write(log_entry)
        print(f"[{timestamp}] 📝 Integrity log updated.")
else:
    status = data.get("zenodo_status", "unknown")
    print(f"[{timestamp}] Zenodo status unchanged (ok={ok}, current_status={status}).")
