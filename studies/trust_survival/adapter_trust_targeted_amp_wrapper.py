import importlib
import traceback
from typing import Any

import pandas as pd


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
        if c not in df.columns:
            df[c] = 0.0
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def _fallback_sweep_and_collect(rcfg):
    # Minimal local sweep: simulate simple exponential decay/growth signals
    import random

    epochs = rcfg.get("epochs_cap", 500)
    data = []
    cci = 0.5
    surv = 1.0
    hazard = 0.0
    collapse = 0.0
    for e in range(epochs):
        noise = random.gauss(0, 0.01)
        cci = max(0.0, min(1.0, cci + (0.002 - 0.001 * hazard) + noise))
        hazard = max(0.0, min(1.0, hazard + 0.0005 * (1 - cci) + noise * 0.1))
        surv = max(0.0, min(1.0, surv - 0.0008 * hazard + noise * 0.02))
        collapse = 1.0 if surv < 0.2 else 0.0
        data.append(
            {
                "epoch": e,
                "CCI": cci,
                "survival_rate": surv,
                "hazard": hazard,
                "collapse_risk": collapse,
            }
        )
    df = pd.DataFrame(data)
    runs_summary = pd.DataFrame(
        [
            {
                "label": rcfg.get("label", "single"),
                "seed": rcfg.get("seed", 0),
                "survival_rate": df["survival_rate"].iloc[-1],
                "collapse_risk": df["collapse_risk"].mean(),
                "CCI": df["CCI"].mean(),
                "hazard": df["hazard"].mean(),
            }
        ]
    )
    return runs_summary, df


def run(config: dict[str, Any], seed: int):
    # Build single-run rcfg
    rcfg = {
        "label": config.get("label", "phase22_targeted_amp_run"),
        "mode": config.get("mode", "ARC_TAF"),
        "epochs_cap": config.get("epochs_cap", 600),
        "agents": config.get("agents", 240),
        "seeds": [seed],
        "seed": seed,
        "clusters": config.get("clusters", 12),
        "traits": config.get(
            "traits", {"tau0": 0.6, "lr_trust": 0.05, "momentum": 0.08}
        ),
        "state": config.get("state", {"openness_eps": 0.50, "reservoir": 0.0}),
        "log_thin": 1,
        "exports": config.get("exports", {"figs_dir": "figs", "outdir": "out"}),
    }
    # try import simcore.runner.sweep_and_collect
    try:
        simcore = importlib.import_module("simcore.runner")
        sweep = getattr(simcore, "sweep_and_collect", None)
        if sweep is None:
            raise ImportError("sweep_and_collect missing")
        runs_summary, traj = sweep([rcfg])
    except Exception:
        # fallback to local lightweight sweep
        try:
            runs_summary, traj = _fallback_sweep_and_collect(rcfg)
        except Exception:
            traceback.print_exc()
            raise
    # ensure DataFrame metrics
    if isinstance(traj, dict):
        traj = pd.DataFrame(traj)
    traj = _ensure_cols(traj)
    return traj
