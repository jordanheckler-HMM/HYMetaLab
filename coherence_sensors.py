#!/usr/bin/env python3
"""Aletheia v2 Sensors - poll four coherence engines and emit sensor network JSON

This script simulates polling four local engine endpoints (Guardian, TruthLens,
MeaningForge, OriginChain). It records latency per engine and stores a
`sensor_network.json` payload under `outputs/aletheia/`.

Designed for local bus latency < 1s; simulation uses deterministic seeds.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

ENGINES = [
    "Guardian",
    "TruthLens",
    "MeaningForge",
    "OriginChain",
]


def simulated_local_call(engine: str, seed: int) -> dict:
    """Simulate a local bus call to an engine and return latency and status."""
    rng = random.Random(hash(engine) ^ seed)
    # Simulate latency between 0.01s and 0.3s for local bus
    latency = round(rng.uniform(0.01, 0.3), 4)
    # Simulate a health score and a component value
    health = round(rng.uniform(0.8, 1.0), 3)
    value = round(rng.uniform(0.6, 0.95), 3)
    # Sleep to emulate call (but keep quick)
    time.sleep(min(latency, 0.05))
    return {"engine": engine, "latency_s": latency, "health": health, "value": value}


def poll_engines(seed: int = 17) -> dict:
    start = time.time()
    results = []
    for e in ENGINES:
        t0 = time.time()
        res = simulated_local_call(e, seed)
        t1 = time.time()
        # compute observed latency (wall-clock)
        res["observed_latency_s"] = round(t1 - t0, 4)
        results.append(res)
    total_time = round(time.time() - start, 4)
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "total_time_s": total_time,
        "sensors": results,
    }


def emit(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    seed = 17
    payload = poll_engines(seed)
    out = emit(Path("outputs/aletheia/sensor_network.json"), payload)
    print(f"Wrote {out} (total_time_s={payload['total_time_s']})")
    for s in payload["sensors"]:
        print(
            f"  {s['engine']}: latency_s={s['latency_s']} observed={s['observed_latency_s']} health={s['health']} value={s['value']}"
        )


if __name__ == "__main__":
    main()
