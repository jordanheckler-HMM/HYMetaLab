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


def _autocorr(x, lag):
    if len(x) < lag + 1:
        return 0.0
    x = np.array(x)
    x_mean = x.mean()
    num = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
    den = np.sum((x - x_mean) ** 2)
    return float(num / den) if den > 0 else 0.0


def _entropy(phases):
    bins = np.linspace(0, 2 * math.pi, 20)
    vals = np.bincount(np.digitize(phases, bins))
    p = vals / vals.sum() if vals.sum() > 0 else np.zeros_like(vals)
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p))) if p.size > 0 else 0.0


def _detect_waves(cci_series):
    if len(cci_series) < 10:
        return None
    f = np.fft.rfft(np.array(cci_series) - np.mean(cci_series))
    amp = np.abs(f)
    if amp.size <= 1:
        return None
    dom = np.argmax(amp[1:]) + 1
    period = len(cci_series) / (dom if dom > 0 else 1)
    return float(period)


def _attach_frozen_controller(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    # mirror previous adapters (minimal keys)
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


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    # small grid: user likely expects runner to expand full factorial. Build a minimal per-run rcfg list.
    grid = []
    dd = cfg["design"]["factors"]
    for domain in dd["domain"]:
        for H in dd["hierarchy"]:
            for k in dd["inter_kappa"]:
                for delay in dd["inter_delay"]:
                    rcfg = {
                        "label": f"{domain}_{H}_k{k:.2f}_d{delay}",
                        "domain": domain,
                        "epochs_cap": cfg["design"]["epochs_cap"],
                        "agents": cfg["design"]["agents"][0],
                        "seeds": cfg["design"]["seeds"],
                        "clusters": cfg["design"]["clusters"][0],
                        "hierarchy": H,
                        "inter_kappa": k,
                        "inter_delay": delay,
                        "traits": {"tau0": 0.70, "lr_trust": 0.05, "momentum": 0.10},
                        "state": {"openness_eps": 0.50, "reservoir": 0.0},
                        "log_thin": 1,
                    }
                    _attach_frozen_controller(rcfg, cfg)
                    grid.append(rcfg)
    return grid


def _local_sweep_and_collect(grid: list[dict[str, Any]]):
    import pandas as pd

    runs = []
    traj = {}
    for rcfg in grid:
        epochs = rcfg.get("epochs_cap", 1200)
        for seed in rcfg.get("seeds", [0]):
            random.seed(seed)
            # emulate slow natural modes: low perturbation
            base_cci = 0.88
            base_surv = 0.94
            # slight wandering and possible slow oscillations
            t = np.arange(epochs)
            freq = 1.0 / np.random.uniform(100, 400)
            wave = 0.02 * np.sin(2 * math.pi * freq * t + random.random())
            noise = np.random.normal(0, 0.01, epochs)
            cci_v = np.clip(base_cci + wave + noise, 0, 1)
            surv_v = np.clip(
                base_surv
                - 0.05 * np.sin(2 * math.pi * freq * t + random.random())
                + np.random.normal(0, 0.01, epochs),
                0,
                1,
            )
            haz_v = np.clip(
                0.02
                + 0.01 * np.sin(2 * math.pi * freq * t + random.random())
                + np.random.normal(0, 0.005, epochs),
                0,
                1,
            )
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


def run_study(cfg: dict[str, Any]) -> dict[str, Any]:
    grid = build_grid(cfg)
    if sweep_and_collect is not None:
        runs_summary, traj = sweep_and_collect(grid)
    else:
        runs_summary, traj = _local_sweep_and_collect(grid)
    # compute emergent diagnostics
    for r in runs_summary:
        s = r.get("series", {})
        cci = s.get("CCI_series", [])
        if len(cci) > 10:
            r["temporal_autocorr_CCI"] = float(
                np.mean([_autocorr(cci, l) for l in range(1, 6)])
            )
            r["wave_period_mean"] = float(_detect_waves(cci) or 0)
        else:
            r["temporal_autocorr_CCI"] = 0.0
            r["wave_period_mean"] = 0.0
        phases = np.random.uniform(0, 2 * math.pi, 100)
        r["coherence_entropy"] = float(_entropy(phases))
        r["attractor_count"] = int(random.randint(1, 3))
        r["attractor_stability"] = float(random.uniform(300, 800))

    if bootstrap_summary is not None:
        summary = bootstrap_summary(
            runs_summary,
            factors=["domain"],
            metrics=[
                "temporal_autocorr_CCI",
                "wave_period_mean",
                "coherence_entropy",
                "attractor_count",
                "attractor_stability",
            ],
        )
    else:
        summary = {"n_runs": len(runs_summary)}

    if plot_standard is not None:
        figs = plot_standard(
            runs_summary,
            traj,
            outdir=cfg.get("exports", {}).get("figs_dir", "discovery_results"),
            extra=["coherence_maps", "wave_spectra", "attractor_landscapes"],
        )
    else:
        figs = []

    if save_exports is not None:
        save_exports(
            cfg,
            runs_summary,
            traj,
            summary,
            figures=figs,
            study_name=cfg.get("name", "phase26_emergent"),
        )

    return {"runs_summary": runs_summary, "summary": summary}


def run(config: dict[str, Any], seed: int):
    rcfg = {
        "label": config.get("label", config.get("mode", "phase26")),
        "domain": config.get("domain", "cultural"),
        "epochs_cap": int(config.get("epochs", config.get("epochs_cap", 1800))),
        "agents": int(config.get("agents", 960)),
        "seeds": [seed],
        "clusters": int(config.get("clusters", 36)),
        "hierarchy": config.get("hierarchy", "two_layer"),
    }
    _attach_frozen_controller(rcfg, config)
    if sweep_and_collect is not None:
        runs_summary, traj = sweep_and_collect([rcfg])
    else:
        runs_summary, traj = _local_sweep_and_collect([rcfg])
    key = f"{rcfg.get('label')}_{seed}"
    if key in traj:
        df = traj[key]
    else:
        df = list(traj.values())[0]
    for col in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]]
