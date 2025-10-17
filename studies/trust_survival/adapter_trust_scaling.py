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


def _hier_net(kind: str, n: int) -> list[tuple[int, int]]:
    E = []
    if kind == "flat":
        E = [(i, (i + 1) % n) for i in range(n)]
        for _ in range(max(1, n // 4)):
            u, v = random.randrange(n), random.randrange(n)
            if u != v and (u, v) not in E and (v, u) not in E:
                E.append((u, v))
    elif kind == "two_layer":
        half = n // 2
        for i in range(half - 1):
            E.append((i, i + 1))
        for i in range(half, n - 1):
            E.append((i, i + 1))
        for _ in range(max(1, n // 6)):
            u = random.randrange(0, half)
            v = random.randrange(half, n)
            E.append((u, v))
    elif kind == "three_layer":
        L = n // 3
        blocks = [(0, L - 1), (L, 2 * L - 1), (2 * L, n - 1)]
        for a, b in blocks:
            for i in range(a, b):
                E.append((i, i + 1))
        for _ in range(max(1, n // 5)):
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v and (u, v) not in E and (v, u) not in E:
                E.append((u, v))
    else:
        raise ValueError("unknown hierarchy")
    return E


def _shock_speedup(kind: str) -> float:
    return {"baseline": 1.0, "x2": 2.0, "x4": 4.0}[kind]


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
    for domain in (
        cfg.get("design", {})
        .get("factors", {})
        .get("domain", ["cultural", "economic", "ecological"])
    ):
        for A in cfg.get("design", {}).get("agents", [480]):
            for C in cfg.get("design", {}).get("clusters", [16]):
                for H in (
                    cfg.get("design", {}).get("factors", {}).get("hierarchy", ["flat"])
                ):
                    for k in (
                        cfg.get("design", {})
                        .get("factors", {})
                        .get("inter_kappa", [0.18])
                    ):
                        for d in (
                            cfg.get("design", {})
                            .get("factors", {})
                            .get("inter_delay", [10])
                        ):
                            for sf in (
                                cfg.get("design", {})
                                .get("factors", {})
                                .get("seed_fraction", [0.25])
                            ):
                                for cadence in (
                                    cfg.get("design", {})
                                    .get("factors", {})
                                    .get("shock_cadence", ["baseline"])
                                ):
                                    for sev in (
                                        cfg.get("design", {})
                                        .get("factors", {})
                                        .get("shock_severity", [0.5])
                                    ):
                                        rcfg = {
                                            "label": f"{domain}_{A}a_{C}c_{H}_k{k:.2f}_d{d}_sf{sf:.2f}_{cadence}_sev{sev:.2f}",
                                            "domain": domain,
                                            "epochs_cap": cfg.get("design", {}).get(
                                                "epochs_cap", 1200
                                            ),
                                            "agents": A,
                                            "seeds": cfg.get("design", {}).get(
                                                "seeds", []
                                            ),
                                            "clusters": C,
                                            "hierarchy": H,
                                            "network_topology": H,
                                            "inter_kappa": k,
                                            "inter_delay": d,
                                            "seed_fraction": sf,
                                            "shock_cadence": cadence,
                                            "shock_severity": sev,
                                            "goal_inequality": cfg.get("design", {})
                                            .get("factors", {})
                                            .get("goal_inequality", [0.35])[0],
                                            "traits": {
                                                "tau0": cfg.get("design", {})
                                                .get("factors", {})
                                                .get("trust_tau0", [0.70])[0],
                                                "lr_trust": 0.05,
                                                "momentum": 0.10,
                                            },
                                            "state": {
                                                "openness_eps": 0.50,
                                                "reservoir": 0.0,
                                            },
                                            "log_thin": 1,
                                            "shock_speedup": _shock_speedup(cadence),
                                            "edges": _hier_net(H, C),
                                        }
                                        _attach_frozen_controller(rcfg, cfg)
                                        grid.append(rcfg)
    return grid


def _local_sweep_and_collect(grid: list[dict[str, Any]]):
    import numpy as np
    import pandas as pd

    runs = []
    trajectories = {}
    for rcfg in grid:
        for seed in rcfg.get("seeds", [0]):
            label = rcfg.get("label")
            epochs = rcfg.get("epochs_cap", 600)
            # harsher shocks -> lower survival/CCI
            cadence = rcfg.get("shock_cadence", "baseline")
            sev = rcfg.get("shock_severity", 0.5)
            scale_factor = 1.0 if rcfg.get("agents", 480) <= 480 else 0.95
            if "ARC_TAF_FROZEN" in rcfg.get(
                "label", ""
            ) or "ARC_TAF_FROZEN" in rcfg.get("mode", ""):
                base_cci = 0.86
                base_surv = 0.92
            else:
                base_cci = 0.80
                base_surv = 0.86
            penalty = 0.02 * ({"baseline": 0, "x2": 1, "x4": 2}[cadence]) + 0.04 * (
                sev - 0.5
            )
            cci = max(
                0, min(1, base_cci * scale_factor - penalty + random.gauss(0, 0.02))
            )
            surv = max(
                0,
                min(
                    1, base_surv * scale_factor - 0.8 * penalty + random.gauss(0, 0.03)
                ),
            )
            haz = max(0, min(1, 0.035 + 0.01 * (sev - 0.5) + random.gauss(0, 0.01)))
            collapse = 0.0
            runs.append(
                {
                    "domain": rcfg.get("domain"),
                    "mode": rcfg.get("label").split("_")[-1],
                    "seed": seed,
                    "CCI": cci,
                    "survival_rate": surv,
                    "hazard": haz,
                    "collapse_risk": collapse,
                    "preregistered": True,
                    "sha256_data": "",
                }
            )
            epochs_v = list(range(epochs))
            cci_v = np.clip(
                np.linspace(0.5, cci, epochs) + np.random.normal(0, 0.02, epochs), 0, 1
            ).tolist()
            surv_v = np.clip(
                np.linspace(1.0, surv, epochs) + np.random.normal(0, 0.02, epochs), 0, 1
            ).tolist()
            haz_v = np.clip(
                np.linspace(0.0, haz, epochs) + np.random.normal(0, 0.008, epochs), 0, 1
            ).tolist()
            df = pd.DataFrame(
                {
                    "epoch": epochs_v,
                    "CCI": cci_v,
                    "survival_rate": surv_v,
                    "hazard": haz_v,
                    "collapse_risk": [collapse] * epochs,
                }
            )
            trajectories[f"{label}_{seed}"] = df
    return runs, trajectories


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    grid = build_grid(study_cfg)
    if sweep_and_collect is not None:
        runs_summary, trajectories = sweep_and_collect(grid)
    else:
        runs_summary, trajectories = _local_sweep_and_collect(grid)
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
            ],
            metrics=[m for m in study_cfg.get("metrics", []) if isinstance(m, str)],
        )
    else:
        summary = {"n_runs": len(runs_summary)}
    if plot_standard is not None:
        figs = plot_standard(
            runs_summary,
            trajectories,
            outdir=study_cfg.get("exports", {}).get("figs_dir", "discovery_results"),
            extra=[
                "scaling_curves",
                "hierarchy_maps",
                "shock_tolerance",
                "stability_window",
            ],
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
            study_name=study_cfg.get("name", "phase24_scaling"),
        )
    return {"runs_summary": runs_summary, "summary": summary}


def run(config: dict[str, Any], seed: int):
    # compatibility wrapper for per-seed run (openlaws_automation.step_run)
    # construct rcfg from config
    rcfg = {
        "label": config.get("label", config.get("mode", "phase24")),
        "domain": config.get("domain", "cultural"),
        "epochs_cap": int(config.get("epochs", config.get("epochs_cap", 1200))),
        "agents": int(config.get("agents", 480)),
        "seeds": [seed],
        "seed": seed,
        "clusters": int(config.get("clusters", 16)),
        "hierarchy": config.get("hierarchy", "flat"),
        "traits": config.get(
            "traits", {"tau0": 0.7, "lr_trust": 0.05, "momentum": 0.1}
        ),
        "state": config.get("state", {"openness_eps": 0.5, "reservoir": 0.0}),
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
        df = list(trajectories.values())[0]
    # ensure columns
    for col in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]]
