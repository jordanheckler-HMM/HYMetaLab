#!/usr/bin/env python3
"""Aletheia v3 Balancer - compute equilibrium map and stability under perturbations

This script either reads live sensor values from `outputs/aletheia/sensor_network.json`
or uses fallback simulated component values, computes an equilibrium point (weighted
by the coherence equation weights), then applies small random perturbations to each
component to estimate stability. It writes `outputs/aletheia/equilibrium_map.json`.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

DEFAULT_COMPONENTS = {
    "guardian_alignment": 0.88,
    "truthlens_Ti": 0.82,
    "meaningforge_MQ": 0.75,
    "originchain_EQ": 0.9,
}


def load_sensor_values(path: Path) -> dict[str, float]:
    if not path.exists():
        return DEFAULT_COMPONENTS.copy()
    data = json.loads(path.read_text())
    # Map sensor names to the component keys
    mapping = {
        "Guardian": "guardian_alignment",
        "TruthLens": "truthlens_Ti",
        "MeaningForge": "meaningforge_MQ",
        "OriginChain": "originchain_EQ",
    }
    comps = {}
    for s in data.get("sensors", []):
        key = mapping.get(s.get("engine"))
        if key:
            comps[key] = s.get("value", DEFAULT_COMPONENTS[key])
    # Fill missing with defaults
    for k, v in DEFAULT_COMPONENTS.items():
        comps.setdefault(k, v)
    return comps


def load_weights(cfg_path: Path) -> dict[str, float]:
    if not cfg_path.exists():
        return {
            "guardian_alignment": 3.0,
            "truthlens_Ti": 2.5,
            "meaningforge_MQ": 2.0,
            "originchain_EQ": 2.5,
        }
    cfg = json.loads(cfg_path.read_text()) if cfg_path.suffix == ".json" else None
    # Try YAML if present (avoid adding yaml dependency)
    try:
        import yaml

        cfgy = yaml.safe_load(cfg_path.read_text())
        return cfgy.get("weights", {})
    except Exception:
        return {
            "guardian_alignment": 3.0,
            "truthlens_Ti": 2.5,
            "meaningforge_MQ": 2.0,
            "originchain_EQ": 2.5,
        }


def equilibrium_point(components: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.values())
    weighted = sum(components[k] * weights.get(k, 1.0) for k in components)
    return weighted / total_w if total_w > 0 else 0.0


def stability_test(
    base_components: dict[str, float],
    weights: dict[str, float],
    trials: int = 100,
    eps: float = 0.02,
    seed: int = 23,
) -> dict:
    rng = random.Random(seed)
    base_eq = equilibrium_point(base_components, weights)
    stable_count = 0
    details = []
    for i in range(trials):
        perturbed = {
            k: max(0.0, min(1.0, v + rng.uniform(-eps, eps)))
            for k, v in base_components.items()
        }
        eq = equilibrium_point(perturbed, weights)
        # define stability: if eq remains within 0.05 (5 percentage points) of base_eq
        delta = abs(eq - base_eq)
        ok = delta <= 0.05
        if ok:
            stable_count += 1
        details.append(
            {"trial": i, "eq": round(eq, 4), "delta": round(delta, 4), "ok": ok}
        )
    stability_pct = round(100.0 * stable_count / trials, 2)
    return {
        "base_eq": round(base_eq, 4),
        "stability_pct": stability_pct,
        "trials": trials,
        "details_sample": details[:10],
    }


def main():
    root = Path(__file__).resolve().parent
    sensor_path = root / "outputs" / "aletheia" / "sensor_network.json"
    cfg_path = root / "coherence_equation.yml"

    components = load_sensor_values(sensor_path)
    weights = load_weights(cfg_path)
    eq = equilibrium_point(components, weights)
    stability = stability_test(components, weights, trials=200, eps=0.02, seed=29)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": components,
        "weights": weights,
        "equilibrium": round(eq, 4),
        "stability": stability,
    }

    out_path = root / "outputs" / "aletheia" / "equilibrium_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"Wrote {out_path} equilibrium={payload['equilibrium']} stability_pct={stability['stability_pct']}"
    )


if __name__ == "__main__":
    main()
