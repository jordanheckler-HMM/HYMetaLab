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
    return {"status": "ok", "ts": datetime.utcnow().isoformat() + "Z"}


@app.post("/validate")
def validate(envelope: Envelope):
    try:
        scores, reasons = run_guardian_checks(envelope)
        verdict = (
            "allow"
            if (scores["coherence"] >= 0.85 and scores["noise"] <= 0.20)
            else "flag"
        )
        if "guardian.block" in reasons:
            verdict = "block"
        report = {
            "trace_id": envelope.trace_id,
            "verdict": verdict,
            "scores": scores,
            "reasons": reasons,
            "run_hash": hash_payload(envelope.payload),
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        uri = push_report(envelope.trace_id, report)
        report["report_uri"] = uri
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
