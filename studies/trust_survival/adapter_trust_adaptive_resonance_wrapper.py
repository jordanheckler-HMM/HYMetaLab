import importlib
from typing import Any

import pandas as pd

try:
    _simcore = importlib.import_module("simcore.runner")
    sweep_and_collect = getattr(_simcore, "sweep_and_collect", None)
except Exception:
    sweep_and_collect = None

from studies.trust_survival import adapter_trust as fallback_adapter
from studies.trust_survival import adapter_trust_adaptive_resonance as adapter_arc


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def run(config: dict[str, Any], seed: int = None):
    cfg = dict(config)
    if seed is not None:
        cfg.setdefault("prereg", {}).setdefault("constants", {})["seeds"] = [seed]

    prereg = cfg.get("prereg", {}).get("constants", {})
    run_conf = {}
    run_conf["label"] = config.get("label", f"ARC_{seed}")
    run_conf["mode"] = config.get("mode", prereg.get("mode", ["ARC"])[0])
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
        "goal_inequality", prereg.get("goal_inequality", [0.20])[0]
    )
    run_conf["shock_severity"] = config.get(
        "shock_severity", prereg.get("shock_severity", [0.5])[0]
    )
    # traits
    run_conf["traits"] = config.get(
        "traits",
        {
            "tau0": config.get("trust_tau0", prereg.get("trust_tau0", [0.6])[0]),
            "lr_trust": 0.05,
            "momentum": 0.10,
        },
    )
    run_conf["state"] = config.get("state", {"openness_eps": 0.50})
    run_conf["log_thin"] = config.get("log_thin", 1)

    # compute coordination_strength and defection_noise heuristically if missing
    tau = run_conf["traits"].get("tau0", 0.6)
    base_coord = cfg.get("controls", {}).get("base_coord", 0.5)
    run_conf.setdefault(
        "coordination_strength", _clip(base_coord + 0.6 * (tau - 0.5), 0.30, 0.90)
    )
    base_noise = cfg.get("controls", {}).get("base_noise", 0.05)
    run_conf.setdefault(
        "defection_noise", max(0.0, base_noise * (1.0 - 0.8 * (tau - 0.5)))
    )

    # Attach base trust hooks
    try:
        adapter_arc._attach_base_trust(
            run_conf, cfg.get("controls", {}).get("base_noise", base_noise)
        )
    except Exception:
        pass

    # If ARC mode, ensure resonance/phase_lock/arc_controller blocks exist and attach hooks
    if run_conf["mode"] == "ARC":
        cfg.setdefault("resonance", {})
        cfg.setdefault("phase_lock", {})
        cfg.setdefault("arc_controller", {})
        # sensible defaults mirroring prereg
        cfg["resonance"].setdefault("amplitude", 0.12)
        cfg["resonance"].setdefault("period_T", 100)
        cfg["resonance"].setdefault("phase_phi", 0.0)
        cfg["resonance"].setdefault("eps_min", 0.05)
        cfg["resonance"].setdefault("eps_max", 0.80)
        cfg["resonance"].setdefault(
            "pulse", {"every": 100, "eps_delta": 0.06, "lr_boost": 1.10}
        )

        cfg["phase_lock"].setdefault("cci_lock_threshold", 0.70)
        cfg["phase_lock"].setdefault("hazard_unlock_threshold", 0.51)
        cfg["phase_lock"].setdefault("ema_window", 40)
        cfg["phase_lock"].setdefault("desync_divergence", 0.85)
        cfg["phase_lock"].setdefault("lr_floor", 0.006)
        cfg["phase_lock"].setdefault("lr_ceiling", 0.16)

        cfg["arc_controller"].setdefault("window", 60)
        cfg["arc_controller"].setdefault(
            "targets", {"CCI": 0.72, "survival": 0.80, "hazard_soft": 0.47}
        )
        cfg["arc_controller"].setdefault(
            "gains", {"k_survival": 0.60, "k_hazard": -0.25, "k_integral": 0.10}
        )
        cfg["arc_controller"].setdefault(
            "momentum_tune",
            {
                "base": 0.10,
                "sync_gain": 1.25,
                "desync_div": 0.85,
                "k_coherence": 0.25,
                "k_vol": -0.20,
            },
        )
        cfg["arc_controller"].setdefault(
            "bounds", {"lr_min": 0.006, "lr_max": 0.16, "m_min": 0.0, "m_max": 0.35}
        )
        cfg["arc_controller"].setdefault(
            "soft_guards", {"hazard_uplift_max": 0.05, "ramp_gamma": 0.96}
        )

        try:
            adapter_arc._attach_resonance(run_conf, cfg)
            adapter_arc._attach_arc(run_conf, cfg)
        except Exception:
            pass

    grid = [run_conf]

    # run the sweep using simcore runner when available else fallback
    if sweep_and_collect is not None:
        runs_summary, trajectories_long = sweep_and_collect(grid)
    else:
        runs_summary, trajectories_long = fallback_adapter.sweep_and_collect(grid)

    # Prefer to return epoch-level pandas DataFrame for the requested seed
    try:
        if isinstance(trajectories_long, (pd.DataFrame,)):
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

    # fallback: synthesize epoch series from runs_summary if available
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
