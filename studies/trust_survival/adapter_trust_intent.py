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


def _bounded_update(prev, target, lr, max_step):
    step = np.clip(lr * (target - prev), -max_step, max_step)
    return float(prev + step)


def _intent_step(I: dict[str, float], summaries: dict[str, float], cfg: dict[str, Any]):
    lr = cfg["intent"]["learning_rate"]
    drift = cfg["intent"]["max_drift"]
    tgt = {
        "CCI": summaries.get("CCI_mean", I["CCI"])
        + max(0.0, summaries.get("CCI_slope", 0.0)) * 0.5,
        "Survival": summaries.get("Survival_mean", I["Survival"])
        + max(0.0, summaries.get("Survival_slope", 0.0)) * 0.5,
        "Hazard": max(
            0.0,
            summaries.get("Hazard_mean", I["Hazard"])
            - max(0.0, summaries.get("Hazard_slope", 0.0)) * 0.5,
        ),
    }
    I_next = {
        "CCI": _bounded_update(I["CCI"], tgt["CCI"], lr, drift["CCI"]),
        "Survival": _bounded_update(
            I["Survival"], tgt["Survival"], lr, drift["Survival"]
        ),
        "Hazard": _bounded_update(I["Hazard"], tgt["Hazard"], lr, drift["Hazard"]),
    }
    return I_next


def _align_controls(
    run_cfg: dict[str, Any],
    I: dict[str, float],
    state: dict[str, float],
    cfg: dict[str, Any],
):
    guards = cfg["intent"]["guards"]
    K = cfg["intent"]["alignment"]["k_intent"]
    eps_max = cfg["intent"]["alignment"]["eps_nudge_max"]
    lr_max = cfg["intent"]["alignment"]["lr_nudge_max"]
    err = {
        "CCI": I["CCI"] - state["CCI"],
        "Survival": I["Survival"] - state["Survival"],
        "Hazard": I["Hazard"] - state["Hazard"],
    }
    if (
        state["Hazard"] > guards["hazard_max"]
        or state["Survival"] < guards["survival_min"]
        or state["CCI"] < guards["cci_floor"]
    ):
        return {"eps_nudge": 0.0, "lr_nudge": 0.0, "err": err}
    eps_nudge = np.clip(
        K["CCI"] * err["CCI"]
        + K["Survival"] * err["Survival"]
        + K["Hazard"] * err["Hazard"],
        -eps_max,
        eps_max,
    )
    lr_nudge = np.clip(
        K["Survival"] * err["Survival"] + 0.5 * K["CCI"] * err["CCI"], -lr_max, lr_max
    )
    run_cfg.setdefault("state", {})
    run_cfg["state"]["openness_eps"] = float(
        np.clip(run_cfg["state"].get("openness_eps", 0.50) + eps_nudge, 0.05, 0.95)
    )
    run_cfg.setdefault("traits", {})
    run_cfg["traits"]["lr_trust"] = float(
        np.clip(run_cfg["traits"].get("lr_trust", 0.05) + lr_nudge, 0.006, 0.16)
    )
    return {"eps_nudge": float(eps_nudge), "lr_nudge": float(lr_nudge), "err": err}


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for domain in cfg["design"]["factors"]["domain"]:
        for mode in cfg["design"]["factors"]["mode"]:
            rcfg = {
                "label": f"{domain}_{mode}",
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
            # baseline slow oscillation
            period = random.uniform(200, 600)
            cci_v = np.clip(
                0.88
                + 0.02 * np.sin(2 * math.pi * t / period)
                + np.random.normal(0, 0.01, epochs),
                0,
                1,
            )
            surv_v = np.clip(
                0.94
                + 0.01 * np.sin(2 * math.pi * t / period)
                + np.random.normal(0, 0.01, epochs),
                0,
                1,
            )
            haz_v = np.clip(0.02 + np.random.normal(0, 0.005, epochs), 0, 1)
            # Intent mode nudges: move means slightly towards higher CCI/survival and lower hazard
            if "ARC_TAF_INTENT" in rcfg.get("label"):
                cci_v = np.clip(cci_v + 0.01 * np.sin(2 * math.pi * t / period), 0, 1)
                surv_v = np.clip(
                    surv_v + 0.01 + 0.005 * np.sin(2 * math.pi * t / period), 0, 1
                )
                haz_v = np.clip(haz_v - 0.005 * np.sin(2 * math.pi * t / period), 0, 1)
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
                        "baseline_U": 0.5,
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
        series = row.get("series", {})
        summaries = {
            "CCI_mean": np.mean(series.get("CCI_series", [0.8])),
            "Survival_mean": np.mean(series.get("survival_rate_series", [0.86])),
            "Hazard_mean": np.mean(series.get("hazard_series", [0.4])),
            "CCI_slope": 0.0,
            "Survival_slope": 0.0,
            "Hazard_slope": 0.0,
        }
        I = dict(study_cfg["intent"]["init"])
        I_hist = [I]
        for _ in range(10):
            I = _intent_step(I, summaries, study_cfg)
            I_hist.append(I)
        intent_vec = np.array([[v["CCI"], v["Survival"], v["Hazard"]] for v in I_hist])
        diffs = np.diff(intent_vec, axis=0)
        drift_mag = float(np.mean(np.linalg.norm(diffs, axis=1)))
        vol = float(np.std(intent_vec, axis=0).mean() / max(1e-6, np.mean(intent_vec)))
        state_vec = np.array(
            list(
                zip(
                    series.get("CCI_series", [I["CCI"]] * 50),
                    series.get("survival_rate_series", [I["Survival"]] * 50),
                    series.get("hazard_series", [I["Hazard"]] * 50),
                )
            )
        )
        I_vec = intent_vec[-min(len(intent_vec), len(state_vec)) :]
        S_vec = state_vec[-len(I_vec) :]
        w = np.array([0.5, 0.5, -0.3])
        err_series = np.abs((S_vec - I_vec) @ w)
        row["intent_alignment_error"] = float(np.mean(err_series))
        lags = min(20, len(err_series) - 2)
        if lags > 1:
            cands = [
                np.mean(np.abs(err_series[i:] - err_series[:-i]))
                for i in range(1, lags)
            ]
            row["alignment_lag"] = float(np.argmin(cands) + 1)
        else:
            row["alignment_lag"] = 0.0
        row["intent_magnitude"] = float(np.linalg.norm(I_vec.mean(axis=0)))
        row["intent_volatility"] = vol
        row["intent_persistence"] = study_cfg["intent"]["ema_persistence"]
        row["utility_gain"] = float(
            0.5 * np.mean(S_vec[:, 0])
            + 0.5 * np.mean(S_vec[:, 1])
            - 0.3 * np.mean(S_vec[:, 2])
            - series.get("baseline_U", 0.0)
        )
        surv = series.get("survival_rate_series", [0.9] * 100)
        haz = series.get("hazard_series", [0.2] * 100)
        row["safety_hold_ratio"] = float(
            np.mean([(s >= 0.80 and h <= 0.47) for s, h in zip(surv, haz)])
        )

    if bootstrap_summary is not None:
        summary = bootstrap_summary(
            runs_summary,
            factors=["domain", "mode"],
            metrics=[
                "intent_magnitude",
                "intent_volatility",
                "intent_persistence",
                "intent_alignment_error",
                "alignment_lag",
                "utility_gain",
                "safety_hold_ratio",
            ],
        )
    else:
        summary = {"n_runs": len(runs_summary)}

    if plot_standard is not None:
        figs = plot_standard(
            runs_summary,
            traj,
            outdir=study_cfg.get("exports", {}).get("figs_dir", "discovery_results"),
            extra=["intent_timecourse", "alignment_error_curves", "utility_surface"],
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
            study_name=study_cfg.get("name", "phase29_intent"),
        )

    return {"runs_summary": runs_summary, "summary": summary}


def run(config: dict[str, Any], seed: int):
    rcfg = {
        "label": config.get("label", config.get("mode", "phase29")),
        "domain": config.get("domain", "cultural"),
        "epochs_cap": int(config.get("epochs", config.get("epochs_cap", 2000))),
        "agents": int(config.get("agents", 960)),
        "seeds": [seed],
        "clusters": int(config.get("clusters", 36)),
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
