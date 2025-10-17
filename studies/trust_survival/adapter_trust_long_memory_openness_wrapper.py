"""Wrapper exposing run(config, seed) for Phase 11 automation.
Builds a single-run grid and calls sweep_and_collect (preferring simcore.runner,
falling back to local helpers). Returns epoch-level trajectories filtered by seed.
"""

import importlib
from typing import Any


def _call_sweep_and_collect(grid):
    try:
        from simcore.runner import sweep_and_collect

        return sweep_and_collect(grid)
    except Exception:
        mod = importlib.import_module("studies.trust_survival.adapter_trust")
        return mod.sweep_and_collect(grid)


def run(config: dict[str, Any], seed: int):
    cfg = dict(config)
    study_cfg = (
        cfg.get("_study_cfg", {}) if isinstance(cfg.get("_study_cfg", {}), dict) else {}
    )
    run_cfg = {
        "label": cfg.get("label", f"run_seed{seed}"),
        "mode": cfg.get("mode", cfg.get("MODE", "META")),
        "eps0": cfg.get("eps0", cfg.get("eps0", 0.1)),
        "trust_tau0": cfg.get("trust_tau0", cfg.get("trust_tau", 0.5)),
        "epochs_cap": cfg.get("epochs", cfg.get("epochs_cap", 700)),
        "agents": cfg.get("agents", study_cfg.get("design", {}).get("agents", 180)),
        "seeds": [seed],
        "goal_inequality": cfg.get("goal_inequality"),
        "shock_severity": cfg.get("shock_severity"),
    }
    grid = [run_cfg]
    runs_summary, trajectories_long = _call_sweep_and_collect(grid)
    try:
        traj = trajectories_long
        traj_seed = traj[traj["seed"] == seed]
        return traj_seed
    except Exception:
        try:
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
                "adapter_trust_long_memory_openness_wrapper: could not extract trajectories from sweep output"
            )
