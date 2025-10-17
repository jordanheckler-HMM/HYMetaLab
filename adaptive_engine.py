#!/usr/bin/env python3
"""Aletheia v5 Adaptive - small sandbox feedback loop for correction

Runs up to 3 cycles of corrective adjustments to component values to move the
system toward a target equilibrium. Writes `outputs/aletheia/feedback_matrix.json`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_eq_map() -> dict:
    p = ROOT / "outputs" / "aletheia" / "equilibrium_map.json"
    if not p.exists():
        raise SystemExit("equilibrium_map.json not found; run field_balancer.py first")
    return json.loads(p.read_text())


def apply_correction(components: dict, target_eq: float, weights: dict) -> dict:
    # Simple proportional correction: increase components that most help eq
    # Compute gradient-like influence (approx): weight * (1 - value)
    corr = {}
    total_influence = sum(
        weights.get(k, 1.0) * (1.0 - v) for k, v in components.items()
    )
    if total_influence == 0:
        return {k: 0.0 for k in components}
    for k, v in components.items():
        influence = weights.get(k, 1.0) * (1.0 - v)
        # correction proportion scaled to needed delta
        corr[k] = round((influence / total_influence) * 0.05, 4)  # small step
    return corr


def equilibrium_point(components: dict, weights: dict) -> float:
    total_w = sum(weights.values())
    weighted = sum(components[k] * weights.get(k, 1.0) for k in components)
    return weighted / total_w if total_w > 0 else 0.0


def main():
    data = read_eq_map()
    components = data.get("components", {})
    weights = data.get(
        "weights",
        {
            "guardian_alignment": 3.0,
            "truthlens_Ti": 2.5,
            "meaningforge_MQ": 2.0,
            "originchain_EQ": 2.5,
        },
    )
    target = data.get("equilibrium", 0.82)
    feedback = {"generated_at": datetime.utcnow().isoformat() + "Z", "cycles": []}

    max_cycles = 3
    for cycle in range(1, max_cycles + 1):
        current_eq = equilibrium_point(components, weights)
        if current_eq >= target:
            feedback["result"] = {
                "status": "converged",
                "cycles_used": cycle - 1,
                "equilibrium": round(current_eq, 4),
            }
            break
        corr = apply_correction(components, target, weights)
        # apply corrections
        for k, delta in corr.items():
            components[k] = round(min(1.0, components[k] + delta), 4)
        new_eq = equilibrium_point(components, weights)
        feedback["cycles"].append(
            {"cycle": cycle, "corrections": corr, "equilibrium": round(new_eq, 4)}
        )
        if new_eq >= target:
            feedback["result"] = {
                "status": "converged",
                "cycles_used": cycle,
                "equilibrium": round(new_eq, 4),
            }
            break
    else:
        feedback["result"] = {
            "status": "max_cycles_reached",
            "cycles_used": max_cycles,
            "equilibrium": round(equilibrium_point(components, weights), 4),
        }

    outp = ROOT / "outputs" / "aletheia" / "feedback_matrix.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(feedback, indent=2))
    print(f"Wrote {outp} with result: {feedback['result']}")


if __name__ == "__main__":
    main()
