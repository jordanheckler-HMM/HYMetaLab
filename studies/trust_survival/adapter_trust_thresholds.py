import importlib
import random
from typing import Any

try:
    simcore = importlib.import_module("simcore.runner")
    sweep_and_collect = simcore.sweep_and_collect
    bootstrap_summary = simcore.bootstrap_summary
    plot_standard = simcore.plot_standard
    save_exports = simcore.save_exports
except Exception:
    sweep_and_collect = None
    bootstrap_summary = None
    plot_standard = None
    save_exports = None


def _clip(x, a, b):
    return max(a, min(b, x))


def _pairs_sorted(vals):
    up = sorted(vals)
    down = list(reversed(up))
    return up, down


def _detect_tipping(track, surv_thr=0.80, haz_thr=0.47, persist=150):
    surv_series = track.get("survival_rate_series", [])
    haz_series = track.get("hazard_series", [])
    if len(surv_series) < persist or len(haz_series) < persist:
        return False, False
    surv_bad = sum(1 for x in surv_series[-persist:] if x < surv_thr) >= int(
        0.9 * persist
    )
    haz_bad = sum(1 for x in haz_series[-persist:] if x > haz_thr) >= int(0.9 * persist)
    return surv_bad, haz_bad


def _threshold_slope(y_prev, y_next, x_prev, x_next):
    if x_next == x_prev:
        return float("inf")
    return (y_next - y_prev) / (x_next - x_prev)


def _attach_frozen_controller(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    run_cfg.setdefault("hooks", {})
    run_cfg.setdefault("state", {"openness_eps": 0.50, "reservoir": 0.0})
    arc = cfg.get("arc_controller", {})
    res = cfg.get("resonance", {})
    cont = cfg.get("containment", {})
    reinj = cfg.get("reinjection_adaptive", {})
    taf = cfg.get("taf", {})
    run_cfg["hooks"]["on_epoch_env"] = {
        "name": "env_window_update",
        "args": {"window": arc.get("window", 60), "track_var": True},
    }
    run_cfg["hooks"]["on_epoch_eps"] = {
        "name": "openness_eps_resonant",
        "args": {
            "eps_key": "state.openness_eps",
            "eps0_key": "eps0",
            "A": res.get("amplitude", 0.12),
            "T": res.get("period_T", 100),
            "phi": res.get("phase_phi", 0.0),
            "lo": res.get("eps_min", 0.05),
            "hi": res.get("eps_max", 0.80),
        },
    }
    run_cfg["hooks"]["on_epoch_pulse"] = {
        "name": "resonant_pulse",
        "args": {
            "every": res.get("pulse", {}).get("every", 100),
            "eps_delta": res.get("pulse", {}).get("eps_delta", 0.06),
            "lr_boost": res.get("pulse", {}).get("lr_boost", 1.10),
            "eps_key": "state.openness_eps",
        },
    }
    run_cfg["hooks"]["on_epoch_phase_lock"] = {
        "name": "phase_locked_learning",
        "args": {
            "cci_lock_threshold": 0.70,
            "hazard_unlock_threshold": 0.51,
            "ema_window": 40,
            "sync_gain": 1.0,
            "desync_divergence": 1.0,
            "lr_floor": 0.0,
            "lr_ceiling": 1.0,
            "lr_key": "traits.lr_trust",
            "momentum_key": "traits.momentum",
        },
    }
    run_cfg["hooks"]["on_epoch_arc"] = {
        "name": "adaptive_resonance_control",
        "args": {
            "targets": arc.get("targets", {}),
            "gains": arc.get("gains", {}),
            "momentum_tune": arc.get("momentum_tune", {}),
            "bounds": arc.get("bounds", {}),
            "soft_guards": arc.get("soft_guards", {}),
            "keys": {
                "lr": "traits.lr_trust",
                "m": "traits.momentum",
                "cci": "env.CCI_mean",
                "survival": "env.survival_rate_mean",
                "hazard": "env.hazard_mean",
                "hazard_var": "env.hazard_var",
                "lock_flag": "env.lock_flag",
            },
        },
    }
    run_cfg["hooks"]["on_epoch_contain"] = {
        "name": "containment_absorb",
        "args": {
            "trigger": cont.get("trigger", {}),
            "absorption": cont.get("absorption", {}),
            "safety": {
                "cap_reservoir": cont.get("reservoir", {}).get("cap", 1.8),
                "freeze_pulses": True,
            },
            "keys": {
                "eps": "state.openness_eps",
                "reservoir": "state.reservoir",
                "hazard": "env.hazard_mean",
                "hazard_var": "env.hazard_var",
            },
        },
    }
    run_cfg["hooks"]["on_epoch_reinject"] = {
        "name": "containment_reinject_adaptive",
        "args": {
            "min_delay": reinj.get("min_delay", 25),
            "ema_smoothing": reinj.get("ema_smoothing", 0.8),
            "gate": reinj.get("gate", {}),
            "neighborhood": {
                "edges": [],
                "kappa_bias": reinj.get("neighborhood", {}).get("kappa_bias", 0.6),
                "deficit_floor": reinj.get("neighborhood", {}).get(
                    "deficit_floor", 0.05
                ),
            },
            "schedule": reinj.get("schedule", {}),
            "effects": reinj.get("effects", {}),
            "keys": {
                "reservoir": "state.reservoir",
                "eps": "state.openness_eps",
                "lr": "traits.lr_trust",
                "cci_by_cluster": "env.cci_by_cluster",
                "survival_by_cluster": "env.survival_by_cluster",
                "hazard": "env.hazard_mean",
            },
        },
    }
    run_cfg["hooks"]["on_epoch_taf"] = {
        "name": "targeted_amplification_feedback",
        "args": {
            "momentum_window": taf.get("momentum_window", 30),
            "smoothing": taf.get("smoothing", 0.85),
            "gain": taf.get("gain", {}),
            "limiter": taf.get("limiter", {}),
            "routing": {
                "edges": [],
                "top_k": taf.get("routing", {}).get("top_k", 3),
                "spillover": taf.get("routing", {}).get("spillover", 0.15),
            },
            "keys": {
                "reservoir": "state.reservoir",
                "eps": "state.openness_eps",
                "lr": "traits.lr_trust",
                "hazard": "env.hazard_mean",
                "inter_sync": "env.inter_sync_index",
                "cci_by_cluster": "env.cci_by_cluster",
                "survival_by_cluster": "env.survival_by_cluster",
            },
        },
    }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    dd = cfg["design"]["factors"]
    agents = cfg["design"]["agents"]
    clusters = cfg["design"]["clusters"]
    seeds = cfg["design"]["seeds"]
    for domain in dd["domain"]:
        for A in agents:
            for C in clusters:
                for H in dd["hierarchy"]:
                    for k in dd["inter_kappa"]:
                        for delay in dd["inter_delay"]:
                            for sf in dd["seed_fraction"]:
                                for cadence in dd["shock_cadence"]:
                                    for sev in dd["shock_severity"]:
                                        for ineq in dd["goal_inequality"]:
                                            rcfg = {
                                                "label": f"{domain}_{A}a_{C}c_{H}_k{k:.2f}_d{delay}_sf{sf:.2f}_{cadence}_sev{sev:.2f}_ineq{ineq:.2f}",
                                                "domain": domain,
                                                "epochs_cap": cfg["design"][
                                                    "epochs_cap"
                                                ],
                                                "agents": A,
                                                "seeds": seeds,
                                                "clusters": C,
                                                "hierarchy": H,
                                                "inter_kappa": k,
                                                "inter_delay": delay,
                                                "seed_fraction": sf,
                                                "shock_cadence": cadence,
                                                "shock_severity": sev,
                                                "goal_inequality": ineq,
                                                "traits": {
                                                    "tau0": 0.70,
                                                    "lr_trust": 0.05,
                                                    "momentum": 0.10,
                                                },
                                                "state": {
                                                    "openness_eps": 0.50,
                                                    "reservoir": 0.0,
                                                },
                                                "log_thin": 1,
                                            }
                                            _attach_frozen_controller(rcfg, cfg)
                                            grid.append(rcfg)
    return grid


def _local_sweep_and_collect(grid: list[dict[str, Any]]):
    """Fallback synthetic runner that returns runs_summary (list of dicts) and trajectories (dict of DataFrames)."""
    import numpy as np
    import pandas as pd

    runs = []
    traj = {}
    for rcfg in grid:
        epochs = rcfg.get("epochs_cap", 600)
        cadence = rcfg.get("shock_cadence", "baseline")
        sev = rcfg.get("shock_severity", 0.5)
        ineq = rcfg.get("goal_inequality", 0.35)
        agents = rcfg.get("agents", 480)
        clusters = rcfg.get("clusters", 16)
        for seed in rcfg.get("seeds", [0]):
            random.seed(seed + int(sev * 1000))
            base_surv = 0.92 if agents <= 480 else 0.88
            base_cci = 0.86 if agents <= 480 else 0.82
            cadence_pen = {"baseline": 0.0, "x2": 0.03, "x4": 0.06}[cadence]
            sev_pen = max(0.0, (sev - 0.5)) * 0.25
            ineq_pen = (ineq - 0.35) * 0.15
            delay_pen = max(0.0, (rcfg.get("inter_delay", 10) - 20) / 100.0)
            noise = random.gauss(0, 0.02)
            final_surv = _clip(
                base_surv - cadence_pen - sev_pen - ineq_pen - delay_pen + noise,
                0.0,
                1.0,
            )
            final_cci = _clip(
                base_cci
                - cadence_pen
                - 0.8 * sev_pen
                - 0.6 * ineq_pen
                + random.gauss(0, 0.02),
                0.0,
                1.0,
            )
            hazard = _clip(
                0.03
                + 0.5 * (sev - 0.5)
                + cadence_pen
                + 0.2 * ineq_pen
                + random.gauss(0, 0.01),
                0.0,
                1.0,
            )
            collapse = 1 if final_surv < 0.50 else 0
            # produce epoch trajectories that gently move toward final values
            epochs_v = list(range(epochs))
            cci_v = np.clip(
                np.linspace(0.95, final_cci, epochs)
                + np.random.normal(0, 0.02, epochs),
                0,
                1,
            )
            surv_v = np.clip(
                np.linspace(1.0, final_surv, epochs)
                + np.random.normal(0, 0.02, epochs),
                0,
                1,
            )
            haz_v = np.clip(
                np.linspace(0.0, hazard, epochs) + np.random.normal(0, 0.008, epochs),
                0,
                1,
            )
            df = pd.DataFrame(
                {
                    "epoch": epochs_v,
                    "CCI": cci_v,
                    "survival_rate": surv_v,
                    "hazard": haz_v,
                    "collapse_risk": [float(collapse)] * epochs,
                }
            )
            key = f"{rcfg.get('label')}_{seed}"
            traj[key] = df
            runs.append(
                {
                    "label": rcfg.get("label"),
                    "domain": rcfg.get("domain"),
                    "seed": seed,
                    "CCI": float(final_cci),
                    "survival_rate": float(final_surv),
                    "hazard": float(hazard),
                    "collapse_risk": float(collapse),
                    "series": {
                        "CCI_series": cci_v.tolist(),
                        "survival_rate_series": surv_v.tolist(),
                        "hazard_series": haz_v.tolist(),
                    },
                }
            )
    return runs, traj


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    grid = build_grid(study_cfg)
    if sweep_and_collect is not None:
        runs_summary, trajectories = sweep_and_collect(grid)
    else:
        runs_summary, trajectories = _local_sweep_and_collect(grid)

    # detect tipping persistence for each run
    for r in runs_summary:
        series = r.get("series", {})
        surv_fail, haz_fail = _detect_tipping(
            series, surv_thr=0.80, haz_thr=0.47, persist=150
        )
        r["survival_fail_persist"] = int(surv_fail)
        r["hazard_fail_persist"] = int(haz_fail)

    # attempt bootstrap summary if available
    if bootstrap_summary is not None:
        summary = bootstrap_summary(
            runs_summary,
            factors=[
                "domain",
                "agents",
                "clusters",
                "hierarchy",
                "inter_kappa",
                "inter_delay",
                "seed_fraction",
                "shock_cadence",
                "shock_severity",
                "goal_inequality",
            ],
            metrics=["survival_rate", "hazard", "collapse_risk", "CCI"],
        )
    else:
        summary = {"n_runs": len(runs_summary)}

    if plot_standard is not None:
        figs = plot_standard(
            runs_summary,
            trajectories,
            outdir=study_cfg.get("exports", {}).get("figs_dir", "discovery_results"),
            extra=["threshold_maps", "hysteresis_loops", "elasticity_curves"],
        )
    else:
        figs = []

    if save_exports is not None:
        save_exports(
            study_cfg,
            runs_summary,
            trajectories,
            summary,
            figures=figs,
            study_name=study_cfg.get("name", "phase25_thresholds"),
        )

    return {"runs_summary": runs_summary, "summary": summary}


def run(config: dict[str, Any], seed: int):
    """Per-seed wrapper compatible with openlaws_automation.step_run.
    Builds a minimal rcfg from config and returns an epoch-level DataFrame.
    """
    rcfg = {
        "label": config.get("label", config.get("mode", "phase25")),
        "domain": config.get("domain", "cultural"),
        "epochs_cap": int(config.get("epochs", config.get("epochs_cap", 1100))),
        "agents": int(config.get("agents", 480)),
        "seeds": [seed],
        "clusters": int(config.get("clusters", 24)),
        "hierarchy": config.get("hierarchy", "flat"),
        "inter_kappa": config.get("inter_kappa", 0.12),
        "inter_delay": config.get("inter_delay", 10),
        "seed_fraction": config.get("seed_fraction", 0.25),
        "shock_cadence": config.get("shock_cadence", "baseline"),
        "shock_severity": config.get("shock_severity", 0.50),
        "goal_inequality": config.get("goal_inequality", 0.35),
        "traits": config.get(
            "traits", {"tau0": 0.70, "lr_trust": 0.05, "momentum": 0.10}
        ),
        "state": config.get("state", {"openness_eps": 0.50, "reservoir": 0.0}),
        "log_thin": 1,
    }
    _attach_frozen_controller(rcfg, config)
    if sweep_and_collect is not None:
        runs_summary, trajectories = sweep_and_collect([rcfg])
    else:
        runs_summary, trajectories = _local_sweep_and_collect([rcfg])
    key = f"{rcfg.get('label')}_{seed}"
    if isinstance(trajectories, dict) and key in trajectories:
        df = trajectories[key]
    else:
        # fallback to first trajectory
        df = list(trajectories.values())[0]
    for col in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]]
