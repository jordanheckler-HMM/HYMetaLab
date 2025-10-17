#!/usr/bin/env python3
"""Data pipeline to merge Guardian/TruthLens/MeaningForge/OriginChain stores

Reads `outputs/aletheia/sensor_network.json`, `outputs/aletheia/total_coherence.json`,
and `outputs/aletheia/equilibrium_map.json`, merges component values into a
single `merged_store.json`, and computes a consistency percentage.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "aletheia"


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_schema():
    cfg = yaml.safe_load((ROOT / "unified_schema.yml").read_text())
    return cfg


def merge():
    sensor = load_json(OUT / "sensor_network.json")
    total = load_json(OUT / "total_coherence.json")
    eq = load_json(OUT / "equilibrium_map.json")
    schema = load_schema()
    tol = float(schema.get("tolerance", 0.05))

    merged = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "components": {},
        "sources": {},
        "consistency": {},
    }

    # gather values
    # sensor values map engine name -> component key
    mapping = {
        "Guardian": "guardian_alignment",
        "TruthLens": "truthlens_Ti",
        "MeaningForge": "meaningforge_MQ",
        "OriginChain": "originchain_EQ",
    }
    if sensor:
        for s in sensor.get("sensors", []):
            key = mapping.get(s.get("engine"))
            if key:
                merged["components"].setdefault(key, {})["sensor"] = s.get("value")
    if total:
        for k, v in total.get("components", {}).items():
            merged["components"].setdefault(k, {})["total_coherence"] = v
    if eq:
        for k, v in eq.get("components", {}).items():
            merged["components"].setdefault(k, {})["equilibrium_map"] = v

    # Compute consistency: for each component, check pairwise within tolerance
    comps = merged["components"]
    consistent_count = 0
    keys = list(schema.get("fields", {}).keys())
    for k in keys:
        vals = []
        for src in ("sensor", "total_coherence", "equilibrium_map"):
            if comps.get(k) and comps[k].get(src) is not None:
                vals.append(comps[k][src])
        merged["sources"][k] = vals
        ok = True
        if len(vals) == 0:
            ok = False
        else:
            base = vals[0]
            for v in vals[1:]:
                if abs(v - base) > tol:
                    ok = False
                    break
        merged["consistency"][k] = {"values": vals, "consistent": ok}
        if ok:
            consistent_count += 1

    consistency_pct = round(100.0 * consistent_count / len(keys), 2) if keys else 0.0
    merged["consistency_summary"] = {
        "consistent_components": consistent_count,
        "total_components": len(keys),
        "consistency_pct": consistency_pct,
    }

    outp = OUT / "merged_store.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(merged, indent=2))
    print(f"Wrote {outp} consistency={consistency_pct}%")
    # Median-based reconciliation: compute median across available source values
    median_reconciled = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "components": {},
        "consistency_summary": {},
    }
    import statistics

    consistent_after = 0
    for k in keys:
        vals = merged["sources"].get(k, [])
        if vals:
            med = statistics.median(vals)
            median_reconciled["components"][k] = {
                "median": round(med, 4),
                "sources": vals,
            }
            # check if all values within tolerance of median
            ok = all(abs(v - med) <= tol for v in vals)
            median_reconciled["components"][k]["consistent"] = ok
            if ok:
                consistent_after += 1
        else:
            median_reconciled["components"][k] = {
                "median": None,
                "sources": [],
                "consistent": False,
            }

    pct_after = round(100.0 * consistent_after / len(keys), 2) if keys else 0.0
    median_reconciled["consistency_summary"] = {
        "consistent_components": consistent_after,
        "total_components": len(keys),
        "consistency_pct": pct_after,
    }
    outp2 = OUT / "merged_store_median.json"
    outp2.write_text(json.dumps(median_reconciled, indent=2))
    print(f"Wrote median reconciled store {outp2} consistency_after={pct_after}%")
    return median_reconciled


if __name__ == "__main__":
    merge()
