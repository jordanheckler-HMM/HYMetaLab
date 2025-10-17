# tools/guardian_client.py
from __future__ import annotations
import requests
import uuid
from datetime import datetime

# Local stub fallback
try:
    from tools.guardian_stub import run_guardian_checks as _local_guardian
except Exception as _e:
    _local_guardian = None

DEFAULT_API = "http://127.0.0.1:8000"


class GuardianVerdict(Exception):
    """Raised when guardian returns block (hard fail)."""


def validate(
    payload: dict,
    *,
    actor="LAB_TECH",
    target="pipeline:runner",
    phase="pre",
    api_base: str = DEFAULT_API,
) -> dict:
    """
    Try FastAPI /validate first. If API not available, fall back to local stub.
    Returns a dict with keys: verdict, scores, reasons, run_hash, report_uri
    """
    env = {
        "trace_id": str(uuid.uuid4()),
        "ts": datetime.utcnow().isoformat() + "Z",
        "actor": actor,
        "target": target,
        "phase": phase,
        "task_type": "pipeline",
        "policy_tags": ["safety.general"],
        "required_checks": ["guardian.coherence>=0.85", "guardian.noise<=0.2"],
        "payload": payload,
        "meta": {"context_hash": "sha256:stub"},
    }

    # 1) Try API
    try:
        r = requests.post(f"{api_base}/validate", json=env, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        # 2) Fallback to local stub
        if _local_guardian is None:
            raise
        scores, reasons = _local_guardian(type("E", (object,), {"payload": payload})())
        verdict = (
            "allow"
            if (scores["coherence"] >= 0.85 and scores["noise"] <= 0.20)
            else "flag"
        )
        if "guardian.block" in reasons:
            verdict = "block"
        return {
            "trace_id": env["trace_id"],
            "verdict": verdict,
            "scores": scores,
            "reasons": reasons,
            "run_hash": "sha256:local-stub",
            "report_uri": "drive://guardian/reports/local_stub.json",
        }
