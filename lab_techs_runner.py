#!/usr/bin/env python3
"""
HYMetaLab lab tech runner — Guardian-gated.
- Calls Guardian pre → runs task → Guardian post.
- If PASS < 0.85 or noise > 0.2 → writes dev/logs/Integrity_Alert.md
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from tools.guardian_client import validate

ALERT_MD = Path("dev/logs/Integrity_Alert.md")
ALERT_MD.parent.mkdir(parents=True, exist_ok=True)

PASS_MIN = 0.85
NOISE_MAX = 0.20


def _write_alert(reason: str, pre: dict | None, post: dict | None):
    lines = [
        "# 🔴 Integrity Alert",
        "",
        f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Reason: {reason}",
        "",
        "## Guardian Pre",
        "```json",
        json.dumps(pre or {}, indent=2),
        "```",
        "",
        "## Guardian Post",
        "```json",
        json.dumps(post or {}, indent=2),
        "```",
        "",
        "> Runner halted due to failing integrity threshold.",
        "",
    ]
    ALERT_MD.write_text("\n".join(lines))


def _passes(verdict: dict) -> bool:
    s = verdict.get("scores", {})
    return (
        (s.get("coherence", 0) >= PASS_MIN)
        and (s.get("noise", 1) <= NOISE_MAX)
        and verdict.get("verdict") != "block"
    )


def run_task() -> dict:
    """
    Your actual experiment/workflow goes here.
    Keep it simple for now: just echo inputs/outputs.
    """
    # Example "task result"
    return {"result": "dummy_ok", "metrics": {"latency_ms": 12}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", type=str, default="")
    args = parser.parse_args()

    hypo_text = args.hypothesis.strip()
    if not hypo_text:
        hypo_text = "baseline: demo-run"

    # 1) Guardian PRE check (on planned inputs)
    pre_payload = {"prompt": hypo_text, "params": {"k": 1}}
    pre = validate(pre_payload, actor="LAB_TECH", target="pipeline:runner", phase="pre")

    if not _passes(pre):
        _write_alert("Pre-check below threshold", pre, None)
        print("❌ Blocked by Guardian (pre). See dev/logs/Integrity_Alert.md")
        return

    # 2) Run the task
    out = run_task()

    # 3) Guardian POST check (on produced outputs)
    post_payload = {"prompt": hypo_text, "output": out}
    post = validate(
        post_payload, actor="LAB_TECH", target="pipeline:runner", phase="post"
    )

    if not _passes(post):
        _write_alert("Post-check below threshold", pre, post)
        print("❌ Blocked by Guardian (post). See dev/logs/Integrity_Alert.md")
        return

    # 4) Success path
    print("✅ Guardian PASS. Result:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
