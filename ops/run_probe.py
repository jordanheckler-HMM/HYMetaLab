import importlib
import json
import os
import sys
import time
from pathlib import Path

import yaml


def pick_study():
    studies_dir = Path("studies")
    if not studies_dir.exists():
        print("ERR: studies/ not found", file=sys.stderr)
        sys.exit(2)
    ymls = sorted(studies_dir.glob("*.yml"))
    if not ymls:
        print("ERR: no studies/*.yml found", file=sys.stderr)
        sys.exit(2)
    # prefer trust_hope if present, else first
    for p in ymls:
        if "trust_hope" in p.stem:
            return p
    return ymls[0]


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def first(lst, default=None):
    try:
        return list(lst)[0]
    except Exception:
        return default


if __name__ == "__main__":
    study_path = os.environ.get("STUDY_YML")
    if not study_path:
        study_path = str(pick_study())
    cfg = load_cfg(study_path)
    adapter_path = cfg.get("adapter", "adapters/sim_adapter_safe.py")
    mod_name = adapter_path.replace("/", ".").replace(".py", "")
    try:
        adapter = importlib.import_module(mod_name)
    except Exception as e:
        print(f"ERR: cannot import adapter {mod_name}: {e}", file=sys.stderr)
        sys.exit(2)
    if not hasattr(adapter, "run_sim"):
        print("ERR: adapter has no run_sim(**kwargs)", file=sys.stderr)
        sys.exit(2)
    sweep = cfg.get("sweep", {})
    params = {
        "epsilon": first(sweep.get("epsilon", [0.001]), 0.001),
        "seed": first(sweep.get("seeds", [11]), 11),
        "shock": first(sweep.get("shock", [0.5]), 0.5),
    }
    # optional agents field
    agents = sweep.get("agents")
    if agents:
        params["agents"] = first(agents, 100)
    print("PROBE_PARAMS", json.dumps(params))
    t0 = time.time()
    try:
        res = adapter.run_sim(**params)
    except TypeError as e:
        # if signature differs, try minimal call
        try:
            res = adapter.run_sim(params)  # dict-style
        except Exception as e2:
            print(f"ERR: run_sim call failed: {e} | {e2}", file=sys.stderr)
            sys.exit(2)
    except Exception as e:
        print(f"ERR: run_sim error: {e}", file=sys.stderr)
        sys.exit(2)
    dt = time.time() - t0
    print("PROBE_OK in %.2fs" % dt)
    # print a tiny digest if dict-like
    try:
        if isinstance(res, dict):
            keys = list(res.keys())[:5]
            print("RESULT_KEYS", keys)
    except Exception:
        pass
