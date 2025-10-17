import importlib
from typing import Any

import pandas as pd

try:
    _simcore = importlib.import_module("simcore.runner")
    sweep_and_collect = getattr(_simcore, "sweep_and_collect", None)
except Exception:
    sweep_and_collect = None

from studies.trust_survival import adapter_trust as fallback_adapter
from studies.trust_survival import adapter_trust_adaptive_reinjection as adapter_rej


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def run(config: dict[str, Any], seed: int = None):
    cfg = dict(config)
    if seed is not None:
        cfg.setdefault("prereg", {}).setdefault("constants", {})["seeds"] = [seed]

    prereg = cfg.get("prereg", {}).get("constants", {})
    run_conf = {}
    run_conf["label"] = config.get("label", f"REINJECT_{seed}")
    run_conf["mode"] = config.get("mode", prereg.get("mode", ["ARC_CAP_REINJECT"])[0])
    run_conf["epochs_cap"] = config.get(
        "epochs", cfg.get("design", {}).get("epochs_cap", prereg.get("epochs", 950))
    )
    run_conf["epochs"] = run_conf["epochs_cap"]
    run_conf["agents"] = config.get(
        "agents", cfg.get("design", {}).get("agents", prereg.get("agents", 240))
    )
    run_conf["seeds"] = (
        [seed]
        if seed is not None
        else cfg.get("design", {}).get("seeds", prereg.get("seeds", []))
    )
    run_conf["clusters"] = config.get(
        "clusters", cfg.get("design", {}).get("clusters", prereg.get("clusters", 12))
    )
    run_conf["goal_inequality"] = config.get(
        "goal_inequality", prereg.get("goal_inequality", [0.20])[0]
    )
    run_conf["shock_severity"] = config.get(
        "shock_severity", prereg.get("shock_severity", [0.5])[0]
    )
    run_conf["traits"] = config.get(
        "traits",
        {
            "tau0": config.get("trust_tau0", prereg.get("trust_tau0", [0.6])[0]),
            "lr_trust": 0.05,
            "momentum": 0.10,
        },
    )
    run_conf["state"] = config.get("state", {"openness_eps": 0.50, "reservoir": 0.0})
    run_conf["log_thin"] = config.get("log_thin", 1)

    tau = run_conf["traits"].get("tau0", 0.6)
    base_coord = cfg.get("controls", {}).get("base_coord", 0.5)
    run_conf.setdefault(
        "coordination_strength", _clip(base_coord + 0.6 * (tau - 0.5), 0.30, 0.90)
    )
    base_noise = cfg.get("controls", {}).get("base_noise", 0.05)
    run_conf.setdefault(
        "defection_noise", max(0.0, base_noise * (1.0 - 0.8 * (tau - 0.5)))
    )

    try:
        adapter_rej._attach_base_trust(
            run_conf, cfg.get("controls", {}).get("base_noise", base_noise)
        )
    except Exception:
        pass

    if run_conf["mode"] in ["ARC_CASCADE_CL", "ARC_CAP_REINJECT"]:
        cfg.setdefault("resonance", {})
        cfg.setdefault("arc_controller", {})
        cfg.setdefault("cascade", {})
        cfg.setdefault("containment", {})
        cfg.setdefault("reinjection_adaptive", {})
        try:
            adapter_rej._attach_arc_and_containment(run_conf, cfg)
            if run_conf["mode"] == "ARC_CAP_REINJECT":
                topo = prereg.get("supernet_topology", ["smallworld"])[0]
                edges = adapter_rej._build_supernet(topo, run_conf.get("clusters", 12))
                adapter_rej._attach_adaptive_reinjection(run_conf, cfg, edges)
        except Exception:
            pass

    grid = [run_conf]

    if sweep_and_collect is not None:
        runs_summary, trajectories_long = sweep_and_collect(grid)
    else:
        runs_summary, trajectories_long = fallback_adapter.sweep_and_collect(grid)

    try:
        if isinstance(trajectories_long, pd.DataFrame):
            traj = trajectories_long
        else:
            if isinstance(trajectories_long, list) and len(trajectories_long) > 0:
                if all(hasattr(t, "columns") for t in trajectories_long):
                    traj = pd.concat(trajectories_long, ignore_index=True)
                else:
                    traj = pd.DataFrame(trajectories_long)
            else:
                traj = pd.DataFrame()
        if not traj.empty:
            if seed is not None and "seed" in traj.columns:
                traj = traj[traj["seed"] == int(seed)].copy()
            for col in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
                if col not in traj.columns:
                    traj[col] = pd.NA
            traj = traj.sort_values("epoch")
            return traj.reset_index(drop=True)
    except Exception:
        pass

    try:
        if hasattr(runs_summary, "empty") and not runs_summary.empty:
            row = runs_summary.iloc[0]
            E = int(run_conf.get("epochs", run_conf.get("epochs_cap", 400)))
            epochs = list(range(E))
            df = pd.DataFrame(
                {
                    "epoch": epochs,
                    "CCI": [row.get("CCI", float("nan"))] * E,
                    "survival_rate": [row.get("survival_rate", float("nan"))] * E,
                    "hazard": [row.get("hazard", float("nan"))] * E,
                    "collapse_risk": [row.get("collapse_risk", float("nan"))] * E,
                }
            )
            return df
    except Exception:
        pass

    return pd.DataFrame(
        columns=["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]
    )
