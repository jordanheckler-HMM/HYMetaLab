from typing import Any

import pandas as pd

try:
    from simcore.runner import sweep_and_collect
except Exception:
    sweep_and_collect = None

from studies.trust_survival import adapter_trust as fallback_adapter
from studies.trust_survival import adapter_trust_wave_propagation as adapter_wp


def run(config: dict[str, Any], seed: int = None):
    cfg = dict(config)
    if seed is not None:
        cfg.setdefault("prereg", {}).setdefault("constants", {})["seeds"] = [seed]

    prereg = cfg.get("prereg", {}).get("constants", {})
    run_conf = {}
    run_conf["label"] = config.get("label", f"WAVE_{seed}")
    run_conf["mode"] = config.get("mode", prereg.get("mode", "WAVE_PROP"))
    run_conf["network_topology"] = config.get(
        "network_topology", prereg.get("network_topology", ["ring"])[0]
    )
    run_conf["coupling_kappa"] = config.get(
        "coupling_kappa", prereg.get("coupling_kappa", [0.25])[0]
    )
    run_conf["phase_lag_phi"] = config.get(
        "phase_lag_phi", prereg.get("phase_lag_phi", [0.0])[0]
    )
    run_conf["exchange_prob"] = config.get(
        "exchange_prob", prereg.get("exchange_prob", [0.05])[0]
    )
    run_conf["eps0"] = config.get(
        "eps0", config.get("state", {}).get("openness_eps", 0.5)
    )
    run_conf["epochs_cap"] = config.get(
        "epochs", cfg.get("design", {}).get("epochs_cap", prereg.get("epochs", 1000))
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
        "clusters", cfg.get("design", {}).get("clusters", prereg.get("clusters", 10))
    )
    run_conf["goal_inequality"] = config.get(
        "goal_inequality", prereg.get("goal_inequality", [0.5])[0]
    )
    run_conf["shock_severity"] = config.get(
        "shock_severity", prereg.get("shock_severity", [0.5])[0]
    )
    run_conf["traits"] = config.get(
        "traits",
        {
            "tau0": config.get("trust_tau0", prereg.get("trust_tau0", [0.5])[0]),
            "lr_trust": 0.05,
            "momentum": 0.10,
        },
    )
    run_conf["state"] = config.get(
        "state", {"openness_eps_by_cluster": [run_conf["eps0"]] * run_conf["clusters"]}
    )
    run_conf["log_thin"] = config.get("log_thin", 1)

    # Attach wave hooks defaults
    cfg.setdefault("wave_dynamics", {})
    cfg["wave_dynamics"].setdefault("seed_clusters", [0, 1])
    cfg["wave_dynamics"].setdefault("propagation_window", 40)
    cfg["wave_dynamics"].setdefault("damping_factor", 0.96)
    cfg["wave_dynamics"].setdefault("reinforcement_gain", 1.10)
    cfg["wave_dynamics"].setdefault("noise_floor", 0.02)
    cfg["wave_dynamics"].setdefault("measure_period", 10)
    cfg["wave_dynamics"].setdefault("guard_hazard_uplift", 0.05)

    adapter_wp._attach_wave_hooks(run_conf, cfg)

    grid = [run_conf]
    if sweep_and_collect is not None:
        runs_summary, trajectories_long = sweep_and_collect(grid)
    else:
        runs_summary, trajectories_long = fallback_adapter.sweep_and_collect(grid)

    # Return epoch-level DataFrame
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

    # fallback synthesize
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
