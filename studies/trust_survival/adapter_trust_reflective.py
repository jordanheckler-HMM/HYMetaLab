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


def _corr(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.size != b.size or a.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _pca_reduce(X, k=3):
    # X: (T, features). Try sklearn if available, else numpy SVD
    try:
        from sklearn.decomposition import PCA

        p = PCA(n_components=k)
        return p.fit_transform(X)
    except Exception:
        Xc = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :k] * S[:k]


def _meta_predict(series, horizon=60):
    # simple linear extrapolation on recent window
    if len(series) < 5:
        return [series[-1]] * horizon
    x = np.arange(len(series))
    a, b = np.polyfit(x[-40:], series[-40:], 1)
    preds = [a * (len(series) + i) + b for i in range(horizon)]
    return preds


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


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for domain in cfg["design"]["factors"]["domain"]:
        for mode in cfg["design"]["factors"]["mode"]:
            label = f"{domain}_{mode}"
            rcfg = {
                "label": label,
                "domain": domain,
                "epochs_cap": cfg["design"]["epochs_cap"],
                "agents": cfg["design"]["agents"],
                "seeds": cfg["design"]["seeds"],
                "clusters": cfg["design"]["clusters"],
                "hierarchy": cfg["design"]["hierarchy"],
                "traits": {"tau0": 0.70, "lr_trust": 0.05, "momentum": 0.10},
                "state": {"openness_eps": 0.50, "reservoir": 0.0},
                "log_thin": 1,
            }
            _attach_frozen_controller(rcfg, cfg)
            # reflection hook for RIL mode
            if "ARC_TAF_RIL" in mode:
                run_cfg = rcfg
                rl = cfg.get("reflective_loop", {})
                run_cfg["hooks"]["on_epoch_ril"] = {
                    "name": "reflective_loop_update",
                    "args": {
                        "window": rl.get("observation_window", 120),
                        "horizon": rl.get("predictive_horizon", 60),
                        "dims": rl.get("compression", {}).get("dims", 3),
                        "gain_grid": rl.get("feedback_gain", [0.05]),
                        "delay_grid": rl.get("feedback_delay", [20]),
                        "bias": rl.get("reflection_bias", {}),
                        "damping": rl.get("damping", {}),
                        "gating": rl.get("gating", {}),
                    },
                }
            grid.append(rcfg)
    return grid


def _local_sweep_and_collect(grid: list[dict[str, Any]]):
    import pandas as pd

    runs = []
    traj = {}
    for rcfg in grid:
        epochs = rcfg.get("epochs_cap", 1000)
        for seed in rcfg.get("seeds", [0]):
            random.seed(seed)
            t = np.arange(epochs)
            # base slow wave
            freq = 1.0 / np.random.uniform(200, 600)
            cci_v = np.clip(
                0.88
                + 0.02 * np.sin(2 * math.pi * freq * t)
                + np.random.normal(0, 0.01, epochs),
                0,
                1,
            )
            surv_v = np.clip(
                0.94
                + 0.01 * np.sin(2 * math.pi * freq * t)
                + np.random.normal(0, 0.01, epochs),
                0,
                1,
            )
            haz_v = np.clip(0.02 + np.random.normal(0, 0.005, epochs), 0, 1)
            # if RIL mode, produce predictive signals and meta-feedback
            key = rcfg.get("label")
            if "ARC_TAF_RIL" in key:
                pred_cci = _meta_predict(
                    list(cci_v), horizon=rcfg.get("epochs_cap", 2000) // 10
                )
                pred_haz = _meta_predict(
                    list(haz_v), horizon=rcfg.get("epochs_cap", 2000) // 10
                )
                meta_fb = list(np.clip(np.linspace(0, 1, len(pred_cci)) * 0.01, 0, 1))
            else:
                pred_cci = []
                pred_haz = []
                meta_fb = []
            df = pd.DataFrame(
                {
                    "epoch": list(range(epochs)),
                    "CCI": cci_v,
                    "survival_rate": surv_v,
                    "hazard": haz_v,
                    "collapse_risk": [0.0] * epochs,
                }
            )
            traj[f"{key}_{seed}"] = df
            runs.append(
                {
                    "label": key,
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
                        "predicted_CCI": pred_cci,
                        "predicted_hazard": pred_haz,
                        "meta_feedback": meta_fb,
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
    # compute reflection metrics
    for r in runs_summary:
        s = r.get("series", {})
        pred = s.get("predicted_CCI", [])
        real = s.get("CCI_series", [])
        pred_h = s.get("predicted_hazard", [])
        real_h = s.get("hazard_series", [])
        r["prediction_accuracy"] = (
            float((_corr(pred, real) + _corr(pred_h, real_h)) / 2.0)
            if (pred and real)
            else 0.0
        )
        r["self_model_consistency"] = (
            float(
                _corr(s.get("meta_prediction", []), s.get("survival_rate_series", []))
            )
            if s.get("meta_prediction")
            else 0.0
        )
        r["reflection_coherence"] = (
            abs(float(_corr(s.get("meta_feedback", []), s.get("CCI_series", []))))
            if s.get("meta_feedback")
            else 0.0
        )
        r["meta_delay_response"] = (
            float(np.mean(np.diff(s.get("meta_feedback", []))))
            if len(s.get("meta_feedback", [])) > 1
            else 0.0
        )
        r["meta_entropy"] = float(np.var(s.get("meta_feedback", [])))
        r["awareness_index"] = float(
            (r["prediction_accuracy"] + r["self_model_consistency"])
            * r["reflection_coherence"]
        )

    if bootstrap_summary is not None:
        summary = bootstrap_summary(
            runs_summary,
            factors=["domain", "mode"],
            metrics=[
                "prediction_accuracy",
                "reflection_coherence",
                "self_model_consistency",
                "awareness_index",
            ],
        )
    else:
        summary = {"n_runs": len(runs_summary)}

    if plot_standard is not None:
        figs = plot_standard(
            runs_summary,
            traj,
            outdir=cfg.get("exports", {}).get("figs_dir", "discovery_results"),
            extra=[
                "meta_feedback_timecourse",
                "prediction_vs_actual",
                "awareness_surface",
            ],
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
            study_name=cfg.get("name", "phase28_reflective"),
        )

    return {"runs_summary": runs_summary, "summary": summary}


def run(config: dict[str, Any], seed: int):
    rcfg = {
        "label": config.get("label", config.get("mode", "phase28")),
        "domain": config.get("domain", "cultural"),
        "epochs_cap": int(config.get("epochs", config.get("epochs_cap", 2000))),
        "agents": int(config.get("agents", 960)),
        "seeds": [seed],
        "clusters": int(config.get("clusters", 36)),
    }
    _attach_frozen_controller(rcfg, config)
    if "ARC_TAF_RIL" in rcfg.get("label"):
        # minimal RIL hook
        rcfg.setdefault("hooks", {})
        rcfg["hooks"]["on_epoch_ril"] = {
            "name": "reflective_loop_update",
            "args": {
                "window": config.get("reflective_loop", {}).get(
                    "observation_window", 120
                ),
                "horizon": config.get("reflective_loop", {}).get(
                    "predictive_horizon", 60
                ),
            },
        }
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
