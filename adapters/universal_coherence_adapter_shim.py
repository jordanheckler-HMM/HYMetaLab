from __future__ import annotations

import shutil
from pathlib import Path

import adapters.universal_coherence_adapter as real


def _find_latest_phase_dir(prefix: str = "phase31b_uclv") -> Path | None:
    base = Path("discovery_results")
    if not base.exists():
        return None
    # match both exact name and timestamped variants
    candidates = sorted(base.glob(f"{prefix}*"))
    if not candidates:
        return None
    # prefer the most recent candidate that actually contains expected files
    possible = [
        "runs_agents.csv",
        "runs_civ.csv",
        "summary_agents.json",
        "summary_civ.json",
    ]
    for cand in reversed(candidates):
        for name in possible:
            if (cand / name).exists():
                return cand
    # fallback to last candidate even if empty
    return candidates[-1]


def _wrap_run(fn, cfg, out_dir, seed):
    """Call the real adapter, then locate the latest phase31b_uclv* directory and copy expected artifacts
    into the automation out_dir so openlaws_automation can find them.
    """
    # Translate normalized cfg into adapter-style config expected by real adapters
    adapter_cfg = {}
    # seeds: accept direct 'seeds' or nested seed_policy
    if isinstance(cfg, dict):
        seeds = (
            cfg.get("seeds")
            or (cfg.get("seed_policy") or {}).get("seeds")
            or (cfg.get("common") or {}).get("seeds")
        )
        adapter_cfg["seeds"] = seeds or [101, 102, 103]
        adapter_cfg["epochs"] = (
            cfg.get("epochs") or (cfg.get("common") or {}).get("epochs") or 1000
        )
        # suite/tests: adapter expects 'suite' as list of tests
        suite = cfg.get("suite") or cfg.get("tests") or cfg.get("sweep") or []
        adapter_cfg["suite"] = suite
        # pass-through other common keys
        adapter_cfg["common"] = cfg.get("common", {})
    else:
        adapter_cfg = cfg

    # call the real adapter entrypoint which will write into discovery_results
    res = fn(adapter_cfg)

    # prefer using the adapter's configured OUT dir if available
    src_dir = None
    try:
        src_dir = real.OUT
    except Exception:
        src_dir = _find_latest_phase_dir("phase31b_uclv")

    if src_dir is None or not Path(src_dir).exists():
        raise RuntimeError(
            "No phase31b_uclv discovery_results directory found after adapter run"
        )

    dest = Path(out_dir) / "data"
    dest.mkdir(parents=True, exist_ok=True)

    # prefer runs_agents.csv then runs_civ.csv
    runs_file = None
    for candidate in ["runs_agents.csv", "runs_civ.csv"]:
        cpath = Path(src_dir) / candidate
        if cpath.exists():
            shutil.copy(cpath, dest / cpath.name)
            runs_file = str(dest / cpath.name)
            break

    # copy summaries if present
    for candidate in ["summary_agents.json", "summary_civ.json"]:
        cpath = Path(src_dir) / candidate
        if cpath.exists():
            shutil.copy(cpath, dest / cpath.name)

    if not runs_file:
        raise RuntimeError(f"Adapter completed but no runs CSV found in {src_dir}")

    return {"runs_summary_csv": runs_file, "trajectories_long_csv": None}


def run_agents(cfg, out_dir, seed):
    return _wrap_run(real.run_agents, cfg, out_dir, seed)


def run_civ(cfg, out_dir, seed):
    return _wrap_run(real.run_civ, cfg, out_dir, seed)
