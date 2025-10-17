from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from importlib import metadata as importlib_metadata
except Exception:
    try:
        import importlib_metadata
    except Exception:
        importlib_metadata = None


def _expand_grid(grid_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product utility for the parameter grid."""
    from itertools import product

    agent_count = grid_cfg["agent_count"]
    noise = grid_cfg["noise"]
    shock = grid_cfg["shock_severity"]
    goals = grid_cfg["goal_diversity"]
    openness = grid_cfg["openness"]
    runs = []
    for a, n, s, g, op in product(agent_count, noise, shock, goals, openness):
        cfg = {
            "agents": a,
            "noise": n,
            "shock_severity": s,
            "goal_diversity": g,
            "mechanism": op.get("mechanism"),
            "epsilon": op.get("epsilon", 0.0),
            "period": op.get("period", None),
        }
        cfg["label"] = f"{op['label']}_A{a}_N{n}_S{s}_G{g}"
        runs.append(cfg)
    return runs


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_study(*args, **kwargs) -> dict[str, Any]:
    """Flexible entrypoint supporting two call styles:

    1) run_study(study_cfg, config_file)  — legacy: second arg is a YAML file path
    2) run_study(cfg, out_dir, seed)       — OpenLaws automation style

    The adapter will detect which form is used and behave accordingly.
    """
    t0 = time.time()

    # Determine calling style
    cfg = None
    out_dir = None
    seed = 0

    # Style 1: (study_cfg, config_file)
    if (
        len(args) == 2
        and isinstance(args[1], str)
        and args[1].endswith((".yml", ".yaml"))
    ):
        study_cfg = args[0]
        config_file = args[1]
        import yaml

        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        # use outputs/out_dir from the config YAML
        out_dir = Path(
            cfg.get("outputs", {}).get("out_dir", "discovery_results/safety_cci_manual")
        )
        seeds = cfg.get("seeds", study_cfg.get("runner", {}).get("seeds", 1))
        epochs_cap = cfg.get(
            "epochs_cap", study_cfg.get("runner", {}).get("epochs_cap", 3000)
        )

    # Style 2: (cfg, out_dir, seed) or keyword args
    else:
        # prefer kwargs if provided
        if "config" in kwargs and isinstance(kwargs["config"], dict):
            cfg = kwargs["config"]
        elif len(args) >= 1 and isinstance(args[0], dict):
            cfg = args[0]

        if "out_dir" in kwargs:
            out_dir = Path(kwargs["out_dir"])
        elif len(args) >= 2:
            out_dir = Path(args[1])

        if "seed" in kwargs:
            seed = int(kwargs["seed"])
        elif len(args) >= 3:
            seed = int(args[2])

        # fallback defaults
        if cfg is None:
            cfg = {}
        seeds = cfg.get("seed_policy", {}).get("seeds", cfg.get("seeds", 1))
        epochs_cap = cfg.get(
            "epochs_cap", cfg.get("runner", {}).get("epochs_cap", 3000)
        )

    # Ensure out_dir exists and data/ subdir
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    # Safety: require explicit opt-in to run with built-in mocks
    allow_mocks = False
    # precedence: kwargs -> cfg -> env
    if "allow_mocks" in kwargs:
        allow_mocks = bool(kwargs.get("allow_mocks"))
    elif isinstance(cfg, dict) and cfg.get("allow_mocks") is not None:
        allow_mocks = bool(cfg.get("allow_mocks"))
    else:
        allow_mocks = bool(os.environ.get("ALLOW_MOCKS", "0") in ("1", "true", "True"))

    # Optional: user can point to a real sim entrypoint like 'myproj.sim:run_sim'
    sim_entry = None
    do_real_sim = False
    if isinstance(cfg, dict) and cfg.get("sim_entry"):
        sim_entry = cfg.get("sim_entry")
    if "sim_entry" in kwargs and kwargs.get("sim_entry"):
        sim_entry = kwargs.get("sim_entry")
    if sim_entry:
        # try to import callable
        try:
            mod_name, fn_name = sim_entry.split(":")
            mod = __import__(mod_name, fromlist=[fn_name])
            sim_fn = getattr(mod, fn_name)
            do_real_sim = callable(sim_fn)
        except Exception:
            do_real_sim = False

    # Grid must come from cfg['grid'] or cfg.get('sweep')
    grid = cfg.get("grid") or cfg.get("sweep") or {}
    if not grid:
        raise RuntimeError("No grid/sweep found in config")

    runs = _expand_grid(grid)

    # Placeholder accumulators for CSVs/JSON (replace with your project utilities)
    runs_summary = []
    trajectories = []
    openness_events = []
    shock_events = []
    conditions_lookup = []

    # Iterate runs
    for run_ix, rcfg in enumerate(runs):
        # if seeds is a list of seed ids, iterate accordingly; otherwise scalar number
        seed_vals = seeds if isinstance(seeds, list) else list(range(int(seeds)))
        for seed in seed_vals:
            run_id = f"{run_ix:04d}_seed{seed}"
            # === call into your sim ===
            # If a real sim entrypoint was provided we will call it. Otherwise we fall back to
            # the built-in mock implementation but only if allow_mocks is True.
            if do_real_sim:
                try:
                    # try common call patterns: (seed=..., config=...)
                    import inspect

                    sig = inspect.signature(sim_fn)
                    call_kwargs = {}
                    if "seed" in sig.parameters:
                        call_kwargs["seed"] = seed
                    if "config" in sig.parameters or "cfg" in sig.parameters:
                        call_kwargs["config"] = rcfg
                    # pass others if accepted
                    sim_out = sim_fn(**call_kwargs)
                    used_mocks = False
                except Exception as e:
                    raise RuntimeError(
                        f"sim_entry {sim_entry} exists but failed to run: {e}"
                    )
            else:
                if not allow_mocks:
                    raise RuntimeError(
                        "No real sim_entry provided and running with built-in mocks is disabled. "
                        "Set cfg['allow_mocks']=True or pass allow_mocks=True to proceed with mock runs."
                    )
                # --- MOCK structure (replace with real sim_out fields) ---
                sim_out = {
                    "run_id": run_id,
                    "label": rcfg["label"],
                    "seed": seed,
                    "epochs_cap": epochs_cap,
                    "stability": {
                        "CCI_mean": 0.62,
                        "hazard_mean": 0.18,
                        "CCI_slope": 0.0007,
                    },
                    "peak_CCI": 0.83,
                    "final_CCI": 0.66,
                    "hazard_peak": 0.35,
                    "early_stopped": False,
                    "collapse_flag": False,
                    "lifespan_epochs": epochs_cap,
                    "reservoirs_ok": True,
                    "time_sec": 0.0,
                    "traj_rows": [],
                    "openness_rows": [],
                    "shock_rows": [],
                }
                used_mocks = True
            # === end mock ===

            # Aggregate CSV rows
            runs_summary.append(
                {
                    "run_id": run_id,
                    "label": rcfg["label"],
                    "seed": seed,
                    "epsilon": rcfg.get("epsilon") or 0.0,
                    "mechanism": rcfg.get("mechanism"),
                    "epochs_cap": epochs_cap,
                    "lifespan_epochs": sim_out["lifespan_epochs"],
                    "collapse_flag": sim_out["collapse_flag"],
                    "peak_CCI": sim_out["peak_CCI"],
                    "final_CCI": sim_out["final_CCI"],
                    "stability_CCI_mean": sim_out["stability"]["CCI_mean"],
                    "stability_hazard_mean": sim_out["stability"]["hazard_mean"],
                    "stability_CCI_slope": sim_out["stability"]["CCI_slope"],
                    "hazard_peak": sim_out["hazard_peak"],
                    "reservoirs_ok": sim_out["reservoirs_ok"],
                    "time_sec": sim_out["time_sec"],
                }
            )
            trajectories.extend(sim_out["traj_rows"])
            openness_events.extend(sim_out["openness_rows"])
            shock_events.extend(sim_out["shock_rows"])
            conditions_lookup.append(
                {"run_id": run_id, "params_json": json.dumps(rcfg)}
            )

    # write artifacts
    data_dir = out_dir / "data"
    # runs_summary.csv
    if runs_summary:
        keys = list(runs_summary[0].keys())
        with open(data_dir / "runs_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(runs_summary)

    # trajectories_long.csv
    # Always write trajectories CSV (may be empty) so callers can read it
    traj_path = data_dir / "trajectories_long.csv"
    if trajectories:
        keys = list(trajectories[0].keys())
    else:
        # default minimal columns to keep downstream readers happy
        keys = ["run_id", "seed", "epoch", "CCI"]
    with open(traj_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        if trajectories:
            w.writerows(trajectories)

    # openness_events.csv
    if openness_events:
        keys = list(openness_events[0].keys())
        with open(data_dir / "openness_events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(openness_events)

    # shock_events.csv
    if shock_events:
        keys = list(shock_events[0].keys())
        with open(data_dir / "shock_events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(shock_events)

    # conditions_lookup.csv
    if conditions_lookup:
        keys = list(conditions_lookup[0].keys())
        with open(data_dir / "conditions_lookup.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(conditions_lookup)

    # summary json
    summary = {"n_runs": len(runs_summary)}
    with open(data_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # provenance: record whether mocks were used, config hashes, git commit, environment, packages
    prov = {
        "used_mocks": bool(locals().get("used_mocks", False)),
        "timestamp": time.time(),
        "python_executable": sys.executable,
        "platform": sys.platform,
    }
    # hash config file if given in style1
    try:
        if "config_file" in locals() and config_file and Path(config_file).exists():
            prov["config_file"] = str(config_file)
            prov["config_file_sha256"] = _hash_file(Path(config_file))
    except Exception:
        pass
    # git commit if available
    try:
        prov["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent)
            )
            .decode()
            .strip()
        )
    except Exception:
        prov["git_commit"] = None
    # package versions
    prov["packages"] = {}
    if importlib_metadata is not None:
        for pkg in ("numpy", "pandas"):
            try:
                prov["packages"][pkg] = importlib_metadata.version(pkg)
            except Exception:
                prov["packages"][pkg] = None
    else:
        prov["packages"] = None
    # record pip freeze if available
    try:
        prov["pip_freeze"] = (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
            .decode()
            .splitlines()
        )
    except Exception:
        prov["pip_freeze"] = None
    with open(data_dir / "provenance.json", "w") as pf:
        json.dump(prov, pf, indent=2)

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "safety_cci_results.md").write_text(
        "# Safety CCI Results\n\n(Autogenerated; fill via analyze/report steps)\n"
    )

    print(f"[safety_cci] Completed {len(runs_summary)} runs in {time.time()-t0:.1f}s")

    return {
        "runs_summary_csv": (
            str(data_dir / "runs_summary.csv") if runs_summary else None
        ),
        "trajectories_long_csv": str(traj_path),
        "summary_json": str(data_dir / "summary.json"),
        "report_dir": str(report_dir),
    }
