"""Wrapper adapter exposing run(config, seed) for openlaws_automation.
This wrapper builds a single-run grid from the automation-provided config,
computes coordination_strength/defection_noise when missing, calls
sweep_and_collect from simcore.runner if available, otherwise falls back to
the project's `studies/trust_survival/adapter_trust` helpers.
It returns epoch-level trajectories filtered to the requested seed so the
automation can compute stability-window metrics.
"""

import importlib
from typing import Any


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _map_static(trust_tau, base_coord, base_noise):
    coord = _clip(base_coord + 0.6 * (trust_tau - 0.5), 0.20, 0.90)
    defe = max(0.0, base_noise * (1.0 - 0.9 * (coord - 0.5)))
    return coord, defe


def _ensure_fields(run_cfg, study_cfg):
    # Compute coordination_strength and defection_noise if missing
    if run_cfg.get("coordination_strength") is None:
        coord, defe = _map_static(
            run_cfg.get(
                "trust_tau0", study_cfg.get("controls", {}).get("base_coord", 0.5)
            ),
            study_cfg.get("controls", {}).get("base_coord", 0.5),
            study_cfg.get("controls", {}).get("base_noise", 0.1),
        )
        run_cfg.setdefault("coordination_strength", coord)
        run_cfg.setdefault("defection_noise", defe)
    return run_cfg


def _call_sweep_and_collect(grid):
    # Prefer simcore.runner.sweep_and_collect if installed
    try:
        from simcore.runner import sweep_and_collect

        return sweep_and_collect(grid)
    except Exception:
        # Fallback to local adapter_trust.sweep_and_collect
        mod = importlib.import_module("studies.trust_survival.adapter_trust")
        return mod.sweep_and_collect(grid)


def run(config: dict[str, Any], seed: int):
    # config is a single-run expanded constants dict from the automation
    # Build a minimal grid with this single config
    cfg = dict(config)  # copy
    # some automation pass keys under different names; normalize
    study_cfg = (
        cfg.get("_study_cfg", {}) if isinstance(cfg.get("_study_cfg", {}), dict) else {}
    )

    run_cfg = {
        "label": cfg.get("label", f"run_seed{seed}"),
        "mode": cfg.get("mode", cfg.get("MODE", "STATIC")),
        "trust_tau0": cfg.get("trust_tau0", cfg.get("trust_tau", 0.5)),
        "epochs_cap": cfg.get("epochs", cfg.get("epochs_cap", 400)),
        "agents": cfg.get("agents", study_cfg.get("design", {}).get("agents", 120)),
        "seeds": [seed],
        "goal_inequality": cfg.get("goal_inequality"),
        "shock_severity": cfg.get("shock_severity"),
        "lr_trust": cfg.get("lr_trust"),
        "momentum": cfg.get("momentum"),
    }
    # ensure numeric fields needed by fallback simulator
    run_cfg = _ensure_fields(run_cfg, study_cfg or cfg)

    grid = [run_cfg]
    runs_summary, trajectories_long = _call_sweep_and_collect(grid)

    # filter trajectories to seed (they may be present as a dataframe-like list/dict)
    try:
        # assume trajectories_long is a pandas DataFrame
        traj = trajectories_long
        traj_seed = traj[traj["seed"] == seed]
        # ensure required columns exist; automation will call ensure_cols
        return traj_seed
    except Exception:
        # If trajectories_long isn't a DataFrame, try to return runs_summary-derived epoch frame
        try:
            # runs_summary may contain aggregated stability-window metrics; synthesize minimal epoch frame
            import pandas as pd

            row = (
                runs_summary[0]
                if isinstance(runs_summary, (list, tuple))
                else runs_summary.iloc[0]
            )
            epoch_frame = pd.DataFrame(
                [
                    {
                        "epoch": 0,
                        "CCI": row.get("CCI", None),
                        "survival_rate": row.get("survival_rate", None),
                        "hazard": row.get("hazard", None),
                        "collapse_risk": row.get("collapse_risk", None),
                        "seed": seed,
                    }
                ]
            )
            return epoch_frame
        except Exception:
            raise RuntimeError(
                "adapter_trust_meta_wrapper: could not extract trajectories from sweep output"
            )
