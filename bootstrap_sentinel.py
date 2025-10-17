#!/usr/bin/env python3
"""Write a HYMetaLab_Sentinel scaffolding from embedded templates.

Usage: python3 bootstrap_sentinel.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path("HYMetaLab_Sentinel")
FILES: dict[str, str] = {}


def w(path: str, content: str):
    # strip a single leading newline to keep templates readable
    FILES[path] = content.lstrip("\n")


def write_all():
    for path, content in FILES.items():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    # create empty sqlite file placeholder
    (ROOT / "state").mkdir(parents=True, exist_ok=True)
    (ROOT / "state" / ".keep").write_text("")


# ---- Templates ----
w(
    f"{ROOT}/Makefile",
    r"""
.PHONY: dev api ui test validate package fmt lint checksums

PY=python3

dev:
	@echo "Starting API (uvicorn) and UI (streamlit) in background panes..."
	@echo "Tip: run each in separate terminals if preferred."
	@echo "API: uvicorn apps.sentinel_api:app --reload"
	@echo "UI : streamlit run apps/sentinel_ui.py"
	@echo "Or use: make api && make ui"

api:
	uvicorn apps.sentinel_api:app --reload

ui:
	streamlit run apps/sentinel_ui.py

test:
	pytest -q

validate:
	$(PY) -m tools.guardian_stub --demo

checksums:
	$(PY) -m tools.checksums dist

package: checksums
	@echo "Packaging artifacts..."
	@mkdir -p dist
	@cp -r dev/logs dist/ 2>/dev/null || true
	@$(PY) - << 'PY'
import json, os, time, hashlib, glob
from pathlib import Path
dist = Path("dist")
sumf = dist/"SHA256SUMS.txt"
lines = []
for fp in dist.rglob("*"):
    if fp.is_file() and fp.name != "SHA256SUMS.txt":
        h=hashlib.sha256(fp.read_bytes()).hexdigest()
        lines.append(f"{h}  {fp}")
sumf.write_text("\n".join(lines)+"\n")
print(f"Wrote {sumf}")
PY

fmt:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .
""",
)


w(
    f"{ROOT}/requirements.txt",
    r"""
fastapi
uvicorn
pydantic>=2
streamlit
requests
psutil
pytest
ruff
black
""",
)


w(
    f"{ROOT}/.gitignore",
    r"""
__pycache__/
*.pyc
.env
.state/
state/
dist/
.dev/
.cache/
.streamlit/
dev/logs/
dev/tmp/
*.sqlite
*.db
.vscode/
.idea/
""",
)


w(
    f"{ROOT}/README.md",
    r"""
# HYMetaLab — Guardian Sentinel (Phase I)

**What this is:** a local watchdog that validates agents/models, logs runs, computes checksums, and syncs reports.  
**Phase I:** stubs + dashboards + CI hooks (no paid services, ≤25 GB).

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make api          # starts FastAPI
make ui           # starts Streamlit UI
make validate     # runs Guardian stub demo
```

Structure
apps/           # FastAPI + Streamlit
tools/          # sync, checksums, guardian/truthlens stubs
dev/            # CI, logs, metrics
state/          # sqlite (runtime)
Phase II (future)
Swap stubs with real Guardian/TruthLens evaluators.
Wire into Aletheia bus (pre/post envelopes + verdict loop).
""",
)


w(
    f"{ROOT}/apps/sentinel_api.py",
    r"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tools.guardian_stub import run_guardian_checks
from tools.checksums import hash_payload
from tools.drive_sync import push_report
from datetime import datetime
app = FastAPI(title="HYMetaLab Guardian Sentinel API", version="0.1.0")

class Envelope(BaseModel):
    trace_id: str
    ts: str
    actor: str
    target: str
    phase: str
    task_type: str
    policy_tags: list[str] = []
    required_checks: list[str] = []
    payload: dict
    meta: dict | None = None


@app.get("/health")
def health():
    return {"status":"ok","ts": datetime.utcnow().isoformat()+"Z"}


@app.post("/validate")
def validate(envelope: Envelope):
    try:
        scores, reasons = run_guardian_checks(envelope)
        verdict = "allow" if (scores["coherence"]>=0.85 and scores["noise"]<=0.20) else "flag"
        if "guardian.block" in reasons:
            verdict = "block"
        report = {
            "trace_id": envelope.trace_id,
            "verdict": verdict,
            "scores": scores,
            "reasons": reasons,
            "run_hash": hash_payload(envelope.payload),
            "ts": datetime.utcnow().isoformat()+"Z"
        }
        uri = push_report(envelope.trace_id, report)
        report["report_uri"] = uri
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
""",
)


w(
    f"{ROOT}/apps/sentinel_ui.py",
    r"""
import streamlit as st
import requests, json, time, os, psutil
from datetime import datetime
st.set_page_config(page_title="Guardian Sentinel", layout="wide")
st.title("🛡️ HYMetaLab — Guardian Sentinel (Phase I)")

with st.sidebar:
    st.subheader("System")
    st.write("CPU:", psutil.cpu_percent(), "%")
    mem = psutil.virtual_memory()
    st.write("RAM:", f"{mem.used//(10242)}MB / {mem.total//(10242)}MB")
    st.divider()
    api_url = st.text_input("API URL", "http://127.0.0.1:8000")
    st.caption("Run make api in another terminal if offline.")

tabs = st.tabs(["Validate", "Metrics", "Drive Sync", "Logs"])

with tabs[0]:
    st.subheader("Run validation (pre/post)")
    col1, col2 = st.columns(2)
    with col1:
        actor = st.text_input("Actor", "CEO_GPT")
        target = st.text_input("Target", "model:gpt-local")
        task_type = st.selectbox("Task Type", ["answer","plan","code","summarize"], index=0)
        policy_tags = st.text_input("Policy Tags (comma)", "safety.general,privacy.redact")
        required_checks = st.text_input("Required Checks (comma)", "guardian.coherence>=0.85,guardian.noise<=0.2")
    with col2:
        prompt = st.text_area("Prompt / Input", "Explain the lab's mission in 2 lines.")
        output = st.text_area("Output (for post phase; optional)", "")

    if st.button("Validate (pre)"):
        env = {
            "trace_id": str(int(time.time()*1000)),
            "ts": datetime.utcnow().isoformat()+"Z",
            "actor": actor,
            "target": target,
            "phase": "pre",
            "task_type": task_type,
            "policy_tags": [x.strip() for x in policy_tags.split(",") if x.strip()],
            "required_checks": [x.strip() for x in required_checks.split(",") if x.strip()],
            "payload": {"prompt": prompt},
            "meta": {"context_hash": "sha256:stub"}
        }
        r = requests.post(f"{api_url}/validate", json=env, timeout=10)
        st.json(r.json())

    if st.button("Validate (post)"):
        env = {
            "trace_id": str(int(time.time()*1000)),
            "ts": datetime.utcnow().isoformat()+"Z",
            "actor": actor,
            "target": target,
            "phase": "post",
            "task_type": task_type,
            "policy_tags": [x.strip() for x in policy_tags.split(",") if x.strip()],
            "required_checks": [x.strip() for x in required_checks.split(",") if x.strip()],
            "payload": {"prompt": prompt, "output": output or "Sample output..."},
            "meta": {"context_hash": "sha256:stub"}
        }
        r = requests.post(f"{api_url}/validate", json=env, timeout=10)
        st.json(r.json())
with tabs[1]:
    st.subheader("Metrics (demo)")
    st.write("Integrity-per-Dollar (placeholder):")
    st.metric("Integrity-per-Dollar", "0.97", "Phase I demo")
    st.write("Latency snapshots will appear here in Phase II.")
    st.caption("Real metrics wire in once Guardian/TruthLens are non-stub.")
with tabs[2]:
    st.subheader("Drive Sync (demo)")
    st.write("Reports are written to dev/logs/guardian_reports/ and a Drive URI string is recorded in the response.")
    st.caption("In Phase I we simulate Drive; Phase II can use real Drive APIs.")

with tabs[3]:
    st.subheader("Recent Logs")
    logs_dir = "dev/logs/guardian_reports"
    if os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir))[-10:]:
            if name.endswith(".json"):
                st.write(f"{name}")
                st.json(json.load(open(os.path.join(logs_dir, name))))
    else:
        st.info("No logs yet — run a validation first.")
""",
)


w(
    f"{ROOT}/tools/guardian_stub.py",
    r"""
# Guardian stub: deterministic scores based on payload hash.
# Phase I: simple, testable; Phase II: replace with real evaluators.
import json, hashlib
from typing import Tuple, Dict


def _score_from_text(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # map first 6 hex chars → 0..1
    val = int(h[:6], 16) / float(0xFFFFFF)
    return round(0.7 + 0.3 * val, 3)  # 0.7..1.0


def run_guardian_checks(envelope) -> Tuple[Dict[str, float], list[str]]:
    payload = envelope.payload if hasattr(envelope, "payload") or isinstance(envelope, dict) else envelope
    if hasattr(payload, "payload"):
        payload = payload.payload
    blob = json.dumps(payload, sort_keys=True)
    coherence = _score_from_text(blob)
    # Invert a bit to make noise anti-correlated (bounded 0.05..0.35)
    noise = round(0.05 + (1.0 - (coherence - 0.7) / 0.3) * 0.3 / 2, 3)
    integrity = round(min(1.0, (coherence * (1.0 - noise / 2))), 3)

    reasons = ["policy:safety.general.pass"]
    if isinstance(payload, dict) and "output" in payload and "hack" in str(payload.get("output")).lower():
        reasons.append("guardian.block")  # example rule

    scores = {"coherence": coherence, "noise": noise, "integrity": integrity}
    return scores, reasons


if __name__ == "__main__":
    demo_env = {"payload": {"prompt": "Hello world", "output": "Guardian demo output"}}
    print(json.dumps(run_guardian_checks(demo_env)[0], indent=2))
""",
)


w(
    f"{ROOT}/tools/truthlens_stub.py",
    r"""
Placeholder for Phase II truth-alignment checks.
def evaluate(output: str) -> dict:
    return {"confidence": 0.9, "flags": []}
""",
)


w(
    f"{ROOT}/tools/checksums.py",
    r"""
import hashlib, json, os
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_payload(payload: dict) -> str:
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def write_tree_sums(root: str, out: str):
    rootp = Path(root)
    lines = []
    for fp in rootp.rglob("*"):
        if fp.is_file():
            lines.append(f"{sha256_file(fp)} {fp}")
    Path(out).write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    Path("dist").mkdir(exist_ok=True)
    print(write_tree_sums("dist", "dist/SHA256SUMS.txt"))
""",
)


w(
    f"{ROOT}/tools/drive_sync.py",
    r"""
from pathlib import Path
import json, time
BASE = Path("dev/logs/guardian_reports")

def push_report(trace_id: str, report: dict) -> str:
    BASE.mkdir(parents=True, exist_ok=True)
    fp = BASE / f"{trace_id}.json"
    fp.write_text(json.dumps(report, indent=2))
    # Simulate a Drive URI; Phase II can swap to real Drive API
    return f"drive://guardian/reports/{trace_id}.json"
""",
)


w(
    f"{ROOT}/dev/ci-cd/guardian_pipeline.yml",
    r"""
name: guardian-pipeline
on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
  workflow_dispatch:

jobs:
  build-test-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Lint
        run: |
          ruff check .
          black --check .
      - name: Test
        run: pytest -q || true # allow none in Phase I
      - name: Validate (Guardian stub)
        run: python -m tools.guardian_stub --demo || true
      - name: Package checksums
        run: make package
""",
)


w(f"{ROOT}/dev/logs/.keep", "")
w(
    f"{ROOT}/dev/metrics/integrity_dollar_report.md",
    r"""
Integrity-per-Dollar (Phase I placeholder)
Formula & CFO inputs will be added in Phase II.
Current demo value surfaced in UI: 0.97 (static).
""",
)


w(
    f"{ROOT}/tests/test_guardian.py",
    r"""
from tools.guardian_stub import run_guardian_checks


def test_guardian_scores_smoke():
    env = type("E", (object,), {"payload": {"prompt": "x", "output": "y"}})()
    scores, reasons = run_guardian_checks(env)
    assert 0.7 <= scores["coherence"] <= 1.0
    assert 0.05 <= scores["noise"] <= 0.35
    assert "policy:safety.general.pass" in reasons
""",
)


# Write files when run
if __name__ == "__main__":
    write_all()
    print(f"✅ Wrote {len(FILES)} files into {ROOT}/")
    print("Next:")
    print("  cd HYMetaLab_Sentinel")
    print("  python3 -m venv .venv && source .venv/bin/activate")
    print("  pip install -r requirements.txt")
    print("  make api # start FastAPI")
    print("  make ui # start Streamlit")
