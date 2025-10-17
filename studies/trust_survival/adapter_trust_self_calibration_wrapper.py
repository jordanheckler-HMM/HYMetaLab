from typing import Any

import pandas as pd

try:
    from simcore.runner import sweep_and_collect
except Exception:
    sweep_and_collect = None

from studies.trust_survival import adapter_trust as fallback_adapter
from studies.trust_survival import adapter_trust_self_calibration as adapter_sc


def run(config: dict[str, Any], seed: int = None):
    cfg = dict(config)
    if seed is not None:
        cfg.setdefault("prereg", {}).setdefault("constants", {})["seeds"] = [seed]

    prereg = cfg.get("prereg", {}).get("constants", {})
    run_conf = {}
    run_conf["label"] = config.get("label", f"SC_{seed}")
    run_conf["mode"] = config.get("mode", prereg.get("mode", "OPT"))
    run_conf["epochs_cap"] = config.get(
        "epochs", cfg.get("design", {}).get("epochs_cap", prereg.get("epochs", 800))
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
        "clusters", cfg.get("design", {}).get("clusters", prereg.get("clusters", 8))
    )
    run_conf["network_topology"] = config.get(
        "network_topology", prereg.get("network_topology", ["smallworld"])[0]
    )
    run_conf["coupling_kappa"] = config.get(
        "coupling_kappa", prereg.get("coupling_kappa", [0.25])[0]
    )
    run_conf["phase_lag_phi"] = config.get(
        "phase_lag_phi", prereg.get("phase_lag_phi", [0.5])[0]
    )
    run_conf["exchange_prob"] = config.get(
        "exchange_prob", prereg.get("exchange_prob", [0.12])[0]
    )
    run_conf["goal_inequality"] = config.get(
        "goal_inequality", prereg.get("goal_inequality", [0.35])[0]
    )
    run_conf["shock_severity"] = config.get(
        "shock_severity", prereg.get("shock_severity", [0.5])[0]
    )
    run_conf["traits"] = config.get(
        "traits",
        {
            "tau0": config.get("trust_tau0", prereg.get("trust_tau0", [0.7])[0]),
            "lr_trust": 0.05,
            "momentum": 0.10,
        },
    )
    run_conf["log_thin"] = config.get("log_thin", 1)

    # default controller block if missing
    cfg.setdefault("controller", {})
    cfg["controller"].setdefault("window", 60)
    cfg["controller"].setdefault(
        "target", {"CCI": 0.72, "hazard_max": 0.47, "survival": 0.80}
    )
    cfg["controller"].setdefault(
        "gains", {"k_hazard": -0.4, "k_survival": 0.5, "k_integral": 0.12}
    )
    cfg["controller"].setdefault(
        "momentum_tune", {"base": 0.08, "k_coherence": 0.3, "k_vol": -0.25}
    )
    cfg["controller"].setdefault(
        "bounds", {"lr_min": 0.005, "lr_max": 0.15, "m_min": 0.0, "m_max": 0.3}
    )
    cfg["controller"].setdefault(
        "overshoot_guard",
        {"hazard_spike": 0.05, "clamp_lr_factor": 0.5, "cooldown": 40},
    )

    # attach calibrate hooks
    adapter_sc._attach_baseline_trust(
        run_conf, cfg.get("controls", {}).get("base_noise", 0.05)
    )
    if run_conf["mode"] == "SELF_CAL":
        adapter_sc._attach_self_calibration(run_conf, cfg)

    grid = [run_conf]
    if sweep_and_collect is not None:
        runs_summary, trajectories_long = sweep_and_collect(grid)
    else:
        runs_summary, trajectories_long = fallback_adapter.sweep_and_collect(grid)

    # return epoch DataFrame for seed
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
