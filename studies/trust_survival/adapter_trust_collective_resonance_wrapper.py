from typing import Any

import pandas as pd

try:
    from simcore.runner import sweep_and_collect
except Exception:
    sweep_and_collect = None

from studies.trust_survival import adapter_trust as fallback_adapter
from studies.trust_survival import adapter_trust_collective_resonance as adapter_cr


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def run(config: dict[str, Any], seed: int = None):
    cfg = dict(config)
    if seed is not None:
        cfg.setdefault("prereg", {}).setdefault("constants", {})["seeds"] = [seed]

    prereg = cfg.get("prereg", {}).get("constants", {})
    run_conf = {}
    run_conf["label"] = config.get("label", f"CRN_{seed}")
    run_conf["mode"] = config.get("mode", prereg.get("mode", "META"))
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

    # compute heuristic coordination_strength/defection_noise
    tau = run_conf["traits"].get("tau0", 0.5)
    base_coord = cfg.get("controls", {}).get("base_coord", 0.5)
    run_conf.setdefault(
        "coordination_strength", _clip(base_coord + 0.6 * (tau - 0.5), 0.3, 0.9)
    )
    base_noise = cfg.get("controls", {}).get("base_noise", 0.05)
    run_conf.setdefault(
        "defection_noise", max(0.0, base_noise * (1.0 - 0.8 * (tau - 0.5)))
    )

    # ensure network/resonance/phase_lock blocks exist
    cfg.setdefault("resonance", {})
    cfg.setdefault("phase_lock", {})
    cfg.setdefault("network", {})
    # defaults
    cfg["resonance"].setdefault("amplitude", 0.18)
    cfg["resonance"].setdefault("period_T", 120)
    cfg["resonance"].setdefault("eps_min", 0.05)
    cfg["resonance"].setdefault("eps_max", 0.85)
    cfg["phase_lock"].setdefault("cci_lock_threshold", 0.68)
    cfg["phase_lock"].setdefault("hazard_unlock_threshold", 0.52)
    cfg["phase_lock"].setdefault("ema_window", 40)
    cfg["phase_lock"].setdefault("desync_divergence", 0.75)
    cfg["phase_lock"].setdefault("lr_floor", 0.01)
    cfg["phase_lock"].setdefault("lr_ceiling", 0.15)
    cfg["phase_lock"].setdefault("sync_gain", 1.35)
    cfg["network"].setdefault("rewire_p", 0.2)
    cfg["network"].setdefault("broadcast_period", 10)
    cfg["network"].setdefault("guard_hazard_uplift", 0.05)

    # Build topology for edges
    n_clusters = run_conf["clusters"]
    edges = adapter_cr._build_topology(
        run_conf["network_topology"], n_clusters, cfg["network"].get("rewire_p", 0.2)
    )

    # Attach hooks
    adapter_cr._attach_trust_learning(
        run_conf, cfg.get("controls", {}).get("base_noise", base_noise)
    )
    if run_conf["mode"] == "CRN":
        adapter_cr._attach_resonance_and_network(
            run_conf,
            cfg,
            edges,
            run_conf["coupling_kappa"],
            run_conf["phase_lag_phi"],
            run_conf["exchange_prob"],
        )

    grid = [run_conf]
    if sweep_and_collect is not None:
        runs_summary, trajectories_long = sweep_and_collect(grid)
    else:
        runs_summary, trajectories_long = fallback_adapter.sweep_and_collect(grid)

    # return epoch-level DataFrame for seed
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
