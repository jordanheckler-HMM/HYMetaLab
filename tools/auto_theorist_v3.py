#!/usr/bin/env python3
"""
Auto-Theorist v3.0 — Incremental Trainer
Scans discovery_results/*/summary.json and produces autotheorist_queue.json
ranked by (ΔCCI high, Δhazard low).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = list(ROOT.glob("discovery_results/*/summary.json"))
OUT = ROOT / "autotheorist_queue.json"


def safe_get(d, *keys):
    """Safely get nested dict value"""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, 0)
        else:
            return 0
    return float(d) if d else 0.0


def feat(r):
    """Extract features from various possible locations"""
    f = r.get("features", r)
    params = r.get("params", {})

    features = [
        safe_get(f, "epsilon")
        or safe_get(params, "eps")
        or safe_get(params, "epsilon"),
        safe_get(f, "rho") or safe_get(params, "rho"),
        safe_get(f, "meaning_delta") or safe_get(params, "meaning_delta"),
        safe_get(f, "trust_delta") or safe_get(params, "trust_delta"),
    ]
    return features


def targets(r):
    """Extract target metrics from various possible locations"""
    # Try multiple field names and nested locations
    dcci = (
        safe_get(r, "delta_cci")
        or safe_get(r, "ΔCCI")
        or safe_get(r, "hypothesis_test", "mean_CCI_gain")
        or 0.0
    )

    dhaz = (
        safe_get(r, "delta_hazard")
        or safe_get(r, "Δhazard")
        or safe_get(r, "hypothesis_test", "mean_hazard_delta")
        or 0.0
    )

    return dcci, dhaz


rows = []
for s in SUMS:
    try:
        r = json.loads(Path(s).read_text())
    except:
        continue
    dcci, dhaz = targets(r)
    rows.append(
        {
            "path": str(s),
            "features": feat(r),
            "score": 1.5 * dcci - 1.0 * max(0, dhaz) + abs(min(0, dhaz)),
        }
    )

rows.sort(key=lambda x: x["score"], reverse=True)
OUT.write_text(json.dumps(rows, indent=2))
print(f"Wrote {len(rows)} records → {OUT}")
