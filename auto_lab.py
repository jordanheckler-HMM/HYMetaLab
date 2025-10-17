#!/usr/bin/env python3
"""
HYMetaLab AutoLab Lite
- Generates simple hypotheses
- Pre-checks with Guardian
- Executes runner with the chosen hypothesis
- Updates a tiny knowledge file with success/fail score
No external services. Safe to stop anytime (Ctrl+C).
"""

from __future__ import annotations
import json
import time
import subprocess
import sys
import math
import random
from pathlib import Path
from datetime import datetime
from tools.guardian_client import validate

RNG = random.Random(42)
STATE_DIR = Path("autolab")
STATE_DIR.mkdir(exist_ok=True)
KB = STATE_DIR / "knowledge.json"
ALERT = Path("dev/logs/Integrity_Alert.md")

PASS_MIN = 0.85
NOISE_MAX = 0.20

SEEDS = [
    "baseline: demo-run",
    "η sweep 0.10–0.20, n=40",
    "trust+meaning ↑, sensitivity fixed, n=30",
    "noise stress test at η=0.18, n=50",
    "coherence calibration: small batch n=10",
]


def load_kb():
    if KB.exists():
        try:
            return json.loads(KB.read_text())
        except Exception:
            pass
    return {"runs": [], "hypotheses": {}}


def save_kb(kb):
    KB.write_text(json.dumps(kb, indent=2))


def propose(kb, k=3):
    """Pick a few candidate hypotheses: mutate top ones + add a seed."""
    hypos = list(kb["hypotheses"].items())
    hypos.sort(key=lambda kv: kv[1].get("score", 0), reverse=True)
    cands = [SEEDS[RNG.randrange(len(SEEDS))]]
    for name, meta in hypos[:2]:
        # tiny mutation
        cands.append(name + f" | tweak{RNG.randrange(100)}")
    # pad up to k with random seeds
    while len(cands) < k:
        cands.append(SEEDS[RNG.randrange(len(SEEDS))])
    return cands[:k]


def utility(kb, name):
    """Simple UCB-like utility: score + exploration bonus."""
    h = kb["hypotheses"].get(name, {"score": 0.5, "n": 0})
    total = 1 + sum(v.get("n", 0) for v in kb["hypotheses"].values())
    bonus = math.sqrt(math.log(total + 1) / (h["n"] + 1))
    return h.get("score", 0.5) + 0.2 * bonus


def choose(kb, candidates):
    return max(candidates, key=lambda c: utility(kb, c))


def guardian_precheck(hypo_text):
    env = {"prompt": hypo_text, "params": {"pilot": True}}
    verdict = validate(env, actor="AUTOLAB", target="pipeline:runner", phase="pre")
    s = verdict.get("scores", {})
    ok = (
        (s.get("coherence", 0) >= PASS_MIN)
        and (s.get("noise", 1) <= NOISE_MAX)
        and verdict.get("verdict") != "block"
    )
    return ok, verdict


def run_once(hypo_text):
    """Call the existing runner, passing the hypothesis text."""
    start = time.time()
    ALERT.unlink(missing_ok=True)
    p = subprocess.run(
        [sys.executable, "lab_techs_runner.py", "--hypothesis", hypo_text],
        capture_output=True,
        text=True,
    )
    stdout = p.stdout
    failed = ALERT.exists() and ALERT.stat().st_mtime >= start
    success = ("✅ Guardian PASS" in stdout) and not failed
    return success, stdout, failed


def update_kb(kb, hypo_text, success):
    h = kb["hypotheses"].setdefault(hypo_text, {"score": 0.5, "n": 0})
    # simple update rule: move score toward 1.0 on success, toward 0.0 on fail
    target = 1.0 if success else 0.0
    h["score"] = round(0.7 * h["score"] + 0.3 * target, 3)
    h["n"] += 1
    kb["runs"].append(
        {
            "ts": datetime.utcnow().isoformat() + "Z",
            "hypothesis": hypo_text,
            "success": success,
        }
    )


def main(cycles=1):
    kb = load_kb()
    for i in range(cycles):
        cands = propose(kb)
        choice = choose(kb, cands)

        ok, pre = guardian_precheck(choice)
        if not ok:
            update_kb(kb, choice, success=False)
            save_kb(kb)
            print(f"❌ Skipping (pre-check failed): {choice}")
            continue

        success, out, failed = run_once(choice)
        update_kb(kb, choice, success=success)
        save_kb(kb)
        print(("✅" if success else "❌"), "Run complete:", choice)


if __name__ == "__main__":
    # Run one cycle by default; you can pass a number e.g., python auto_lab.py 5
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cycles)
