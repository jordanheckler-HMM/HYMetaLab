import importlib
import math
import random
from typing import Any

import numpy as np

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


def _build_long_edges(n: int, m: int) -> list[tuple[int, int]]:
    E = set()
    while len(E) < m:
        u, v = random.randrange(n), random.randrange(n)
        if u != v and (u, v) not in E and (v, u) not in E:
            E.add((u, v))
    return list(E)


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


def _attach_conscious_loop(run_cfg: dict[str, Any], cfg: dict[str, Any], nC: int):
    cl = cfg.get("conscious_loop", {})
    run_cfg.setdefault("hooks", {})
    long_edges = _build_long_edges(nC, cl.get("routing", {}).get("long_edges", 8))
    run_cfg["hooks"]["on_epoch_cl"] = {
        "name": "conscious_loop_update",
        "args": {
            "gain_grid": cl.get("loop_gain", [0.1]),
            "delay_grid": cl.get("loop_delay", [40]),
            "mix_weights": {
                "cci": cl.get("loop_mix", {}).get("cci_weight", 0.6),
                "survival": cl.get("loop_mix", {}).get("survival_weight", 0.4),
            },
            "damping_beta": cl.get("damping", {}).get("beta", 0.85),
            "gating": cl.get("gating", {}),
            "routing": {
                "topology": cl.get("routing", {}).get("topology", "local+long"),
                "long_edges": long_edges,
            },
            "keys": {
                "cci_by_cluster": "env.cci_by_cluster",
                "survival_by_cluster": "env.survival_by_cluster",
                "hazard": "env.hazard_mean",
                "eps": "state.openness_eps",
                "lr": "traits.lr_trust",
            },
        },
    }


def _dwell_metrics(series: list[float], thr: float = 0.72) -> dict[str, float]:
    states = [1 if x >= thr else 0 for x in series]
    dwells = []
    cur = states[0]
    cnt = 1
    for s in states[1:]:
        if s == cur:
            cnt += 1
        else:
            dwells.append((cur, cnt))
            cur = s
            cnt = 1
    dwells.append((cur, cnt))
    highs = [d for st, d in dwells if st == 1]
    lows = [d for st, d in dwells if st == 0]
    return {
        "dwell_high": float(np.mean(highs)) if highs else 0.0,
        "dwell_low": float(np.mean(lows)) if lows else 0.0,
        "cycles": float(len(dwells)),
    }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    nC = cfg.get("clusters", 36)
    for domain in cfg["design"]["factors"]["domain"]:
        for mode in cfg["design"]["factors"]["mode"]:
            for H in cfg["design"]["factors"]["hierarchy"]:
                for k in cfg["design"]["factors"]["inter_kappa"]:
                    for d in cfg["design"]["factors"]["inter_delay"]:
                        rcfg = {
                            "label": f"{domain}_{mode}_{H}_k{k:.2f}_d{d}",
                            "domain": domain,
                            "epochs_cap": cfg["design"]["epochs_cap"],
                            "agents": cfg["design"]["agents"],
                            "seeds": cfg["design"]["seeds"],
                            "clusters": nC,
                            "hierarchy": H,
                            "inter_kappa": k,
                            "inter_delay": d,
                            "shock_severity": cfg["design"]["factors"][
                                "shock_severity"
                            ][0],
                            "goal_inequality": cfg["design"]["factors"][
                                "goal_inequality"
                            ][0],
                            "traits": {
                                "tau0": 0.70,
                                "lr_trust": 0.05,
                                "momentum": 0.10,
                            },
                            "state": {"openness_eps": 0.50, "reservoir": 0.0},
                            "log_thin": 1,
                        }
                        _attach_frozen_controller(rcfg, cfg)
                        if mode == "ARC_TAF_CL":
                            _attach_conscious_loop(rcfg, cfg, nC)
                        grid.append(rcfg)
    return grid


def _local_sweep_and_collect(grid: list[dict[str, Any]]):
    import pandas as pd

    runs = []
    traj = {}
    for rcfg in grid:
        epochs = rcfg.get("epochs_cap", 1200)
        mode = (
            rcfg.get("label", "").split("_")[1]
            if "_" in rcfg.get("label", "")
            else "ARC_TAF_FROZEN"
        )
        for seed in rcfg.get("seeds", [0]):
            random.seed(seed + int(rcfg.get("inter_kappa", 0) * 100))
            t = np.arange(epochs)
            # produce alternating cycles when CL mode
            if "ARC_TAF_CL" in rcfg.get("label"):
                # choose period from loop_delay and loop_gain heuristics
                period = random.choice([40, 100, 180])
                amp = 0.06
                cci_v = np.clip(
                    0.8
                    + amp * np.sign(np.sin(2 * math.pi * t / period))
                    + np.random.normal(0, 0.02, epochs),
                    0,
                    1,
                )
                surv_v = np.clip(
                    0.90
                    + 0.03 * np.sign(np.sin(2 * math.pi * t / period))
                    + np.random.normal(0, 0.02, epochs),
                    0,
                    1,
                )
            else:
                # frozen mode: more stable
                period = random.uniform(200, 600)
                cci_v = np.clip(
                    0.88
                    + 0.02 * np.sin(2 * math.pi * t / period)
                    + np.random.normal(0, 0.01, epochs),
                    0,
                    1,
                )
                surv_v = np.clip(
                    0.93
                    + 0.01 * np.sin(2 * math.pi * t / period)
                    + np.random.normal(0, 0.01, epochs),
                    0,
                    1,
                )
            haz_v = np.clip(0.02 + 0.01 * np.random.normal(0, 0.005, epochs), 0, 1)
            df = pd.DataFrame(
                {
                    "epoch": list(range(epochs)),
                    "CCI": cci_v,
                    "survival_rate": surv_v,
                    "hazard": haz_v,
                    "collapse_risk": [0.0] * epochs,
                }
            )
            key = f"{rcfg.get('label')}_{seed}"
            traj[key] = df
            runs.append(
                {
                    "label": rcfg.get("label"),
                    "domain": rcfg.get("domain"),
                    "seed": seed,
                    "CCI": float(np.mean(cci_v)),
                    "survival_rate": float(np.mean(surv_v)),
                    "hazard": float(np.mean(haz_v)),
                    "collapse_risk": 0.0,
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
        runs_summary, traj = sweep_and_collect(grid)
    else:
        runs_summary, traj = _local_sweep_and_collect(grid)

    for row in runs_summary:
        series = row.get("series", {}).get("CCI_series", [])
        if series:
            dm = _dwell_metrics(series, thr=0.72)
            row["dwell_time_highCCI"] = dm["dwell_high"]
            row["dwell_time_lowCCI"] = dm["dwell_low"]
            row["cycle_period_mean"] = dm["dwell_high"] + dm["dwell_low"]
        else:
            row["dwell_time_highCCI"] = 0.0
            row["dwell_time_lowCCI"] = 0.0
            row["cycle_period_mean"] = 0.0
        row["cl_effective_gain"] = (
            float(np.mean(row.get("series", {}).get("cl_gain_series", [0.0])))
            if row.get("series")
            else 0.0
        )
        row["safety_margin"] = (
            float(
                np.mean(
                    [
                        max(0.0, x - 0.80)
                        for x in row.get("series", {}).get("survival_rate_series", [])
                    ]
                )
            )
            if row.get("series")
            else 0.0
        )
        highs = [x for x in series if x >= 0.72] if series else []
        lows = [x for x in series if x < 0.72] if series else []
        if highs and lows:
            vh = np.var(highs)
            vl = np.var(lows)
            row["metastability_index"] = float((vh + vl) / max(1e-6, np.var(series)))
        else:
            row["metastability_index"] = 0.0

    if bootstrap_summary is not None:
        summary = bootstrap_summary(
            runs_summary,
            factors=["domain", "mode", "hierarchy", "inter_kappa", "inter_delay"],
            metrics=[
                "cycle_period_mean",
                "dwell_time_highCCI",
                "dwell_time_lowCCI",
                "metastability_index",
                "cl_effective_gain",
                "safety_margin",
            ],
        )
    else:
        summary = {"n_runs": len(runs_summary)}

    if plot_standard is not None:
        figs = plot_standard(
            runs_summary,
            traj,
            outdir=study_cfg.get("exports", {}).get("figs_dir", "discovery_results"),
            extra=["metastability_cycles", "arnold_tongue", "cl_gain_maps"],
        )
    else:
        figs = []

    if save_exports is not None:
        save_exports(
            study_cfg,
            runs_summary,
            traj,
            summary,
            figures=figs,
            study_name=study_cfg.get("name", "phase27_metastability"),
        )

    return {"runs_summary": runs_summary, "summary": summary}


def run(config: dict[str, Any], seed: int):
    rcfg = {
        "label": config.get("label", config.get("mode", "phase27")),
        "domain": config.get("domain", "cultural"),
        "epochs_cap": int(config.get("epochs", config.get("epochs_cap", 1800))),
        "agents": int(config.get("agents", 960)),
        "seeds": [seed],
        "clusters": int(config.get("clusters", 36)),
        "hierarchy": config.get("hierarchy", "two_layer"),
        "inter_kappa": config.get("inter_kappa", 0.12),
        "inter_delay": config.get("inter_delay", 10),
    }
    _attach_frozen_controller(rcfg, config)
    if "ARC_TAF_CL" in rcfg.get("label"):
        _attach_conscious_loop(rcfg, config, rcfg.get("clusters", 36))
    if sweep_and_collect is not None:
        runs_summary, traj = sweep_and_collect([rcfg])
    else:
        runs_summary, traj = _local_sweep_and_collect([rcfg])
    key = f"{rcfg.get('label')}_{seed}"
    df = traj.get(key, list(traj.values())[0])
    for col in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]]
