# Guardian stub: deterministic scores based on payload hash.
# Phase I: simple, testable; Phase II: replace with real evaluators.
import json
import hashlib
from typing import Tuple, Dict


def _score_from_text(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # map first 6 hex chars → 0..1
    val = int(h[:6], 16) / float(0xFFFFFF)
    return round(0.7 + 0.3 * val, 3)  # 0.7..1.0


def run_guardian_checks(envelope) -> Tuple[Dict[str, float], list[str]]:
    payload = (
        envelope.payload
        if hasattr(envelope, "payload") or isinstance(envelope, dict)
        else envelope
    )
    if hasattr(payload, "payload"):
        payload = payload.payload
    blob = json.dumps(payload, sort_keys=True)
    coherence = _score_from_text(blob)
    # Invert a bit to make noise anti-correlated (bounded 0.05..0.35)
    noise = round(0.05 + (1.0 - (coherence - 0.7) / 0.3) * 0.3 / 2, 3)
    integrity = round(min(1.0, (coherence * (1.0 - noise / 2))), 3)

    reasons = ["policy:safety.general.pass"]
    if (
        isinstance(payload, dict)
        and "output" in payload
        and "hack" in str(payload.get("output")).lower()
    ):
        reasons.append("guardian.block")  # example rule

    scores = {"coherence": coherence, "noise": noise, "integrity": integrity}
    return scores, reasons


if __name__ == "__main__":
    demo_env = {"payload": {"prompt": "Hello world", "output": "Guardian demo output"}}
    print(json.dumps(run_guardian_checks(demo_env)[0], indent=2))
