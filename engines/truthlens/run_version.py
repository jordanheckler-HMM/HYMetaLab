#!/usr/bin/env python3
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import yaml

p = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load((p / "configs" / "global_seed.yml").read_text())
SEED = int(cfg.get("SEED", 42))

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=int, default=1)
args = parser.parse_args()

rng = random.Random(SEED + args.version)
value = round(rng.uniform(0.6, 0.95), 4)
out = p / "outputs" / "truthlens" / f"truthlens_v{args.version}.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "engine": "TruthLens",
    "version": args.version,
    "value": value,
    "seed": SEED,
    "timestamp": datetime.utcnow().isoformat() + "Z",
}
with open(out, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote {out}")
