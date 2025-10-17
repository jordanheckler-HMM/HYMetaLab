from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Try to import the project sim runner if available
try:
    sim_mod = importlib.import_module("sim")
    has_sim_run = hasattr(sim_mod, "run_sim")
except Exception:
    sim_mod = None
    has_sim_run = False


# Fallback light adapters (replace with your project imports):
def compute_hazard(S_prev, S_cur):
    return max(0.0, np.log(max(S_prev, 1e-9)) - np.log(max(S_cur, 1e-9)))


def stability_window_stats(series: np.ndarray, window: int = 200):
    win = series[-window:] if len(series) >= window else series
    slope = np.polyfit(np.arange(len(win)), win, 1)[0] if len(win) > 5 else 0.0
    return float(np.mean(win)), float(slope)


def simulate_child(
    parent_cci: float,
    epsilon: float,
    lambda_star: float,
    agents: int,
    horizon: int,
    rng: np.random.Generator,
):
    """
    Minimal stand-in for your child-universe runner:
    - Initialize child CCI near a fraction of parent_cci with ε, λ★ modifiers.
    - Evolve survival, hazard, and CCI with small stochasticity.
    Replace with your project’s gravity/CCI/shock pipeline for real runs.
    """
    # initialization
    cci = np.empty(horizon)
    hazard = np.empty(horizon)
    survival = np.empty(horizon)
    cci[0] = max(
        0.05,
        parent_cci
        * (0.55 + 0.25 * epsilon / 0.008 + 0.20 * (lambda_star - 0.85) / 0.10),
    )
    survival[0] = 0.85 + 0.05 * (parent_cci - 0.7) - 0.03 * (0.004 - epsilon)
    hazard[0] = max(
        0.0,
        0.25
        - 0.10 * (parent_cci - 0.7)
        - 0.08 * (epsilon / 0.008)
        - 0.05 * (lambda_star - 0.85) / 0.10,
    )

    # evolution
    for t in range(1, horizon):
        drift = (
            0.002 * (parent_cci - 0.7)
            + 0.0015 * (epsilon / 0.008)
            + 0.001 * (lambda_star - 0.85) / 0.10
        )
        noise = rng.normal(0, 0.004)
        cci[t] = np.clip(cci[t - 1] + drift + noise, 0.0, 1.0)

        # survival rises with cci; mild decay without openness
        survival[t] = np.clip(
            survival[t - 1]
            + 0.003 * (cci[t] - 0.5)
            + 0.001 * (epsilon / 0.008)
            + rng.normal(0, 0.002),
            0.01,
            0.999,
        )
        hazard[t] = compute_hazard(survival[t - 1], survival[t])

    return {
        "cci": cci,
        "hazard": hazard,
        "survival": survival,
    }


def run_study(config: dict[str, Any], out_dir: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sweep = config["sweep"]
    thresholds = config["thresholds"]["creation_success"]
    base = Path(out_dir)
    (base / "data").mkdir(parents=True, exist_ok=True)

    rows_summary = []
    traj_rows = []

    for parent_cci in sweep["parent_cci"]:
        for epsilon in sweep["epsilon"]:
            for lambda_star in sweep["lambda_star"]:
                for agents in sweep.get("agents", [100]):
                    horizon = int(sweep.get("horizon", [3000])[0])
                    # Prefer to call project sim runner when available
                    sim = None
                    if has_sim_run:
                        try:
                            # attempt to build a minimal YAML-style config the sim.run_sim expects
                            sim_cfg = {
                                "seed": int(seed),
                                "ticks": int(horizon),
                                "n_agents": int(agents),
                                # store extra knobs so downstream sim code can read them if supported
                                "parent_cci": float(parent_cci),
                                "epsilon": float(epsilon),
                                "lambda_star": float(lambda_star),
                            }
                            # write temp config to disk
                            tmp_cfg_path = (
                                base
                                / f"tmp_config_s{seed}_p{parent_cci:.2f}_e{epsilon:.4f}.yaml"
                            )
                            import yaml

                            yaml.safe_dump(sim_cfg, open(tmp_cfg_path, "w"))
                            out_dir_sim = sim_mod.run_sim(str(tmp_cfg_path))
                            # try to read expected outputs from sim run if they exist
                            # assume sim_mod's Logger wrote a time-series CSV under the output dir
                            ts_path = Path(out_dir_sim) / "time_series.csv"
                            if ts_path.exists():
                                df = pd.read_csv(ts_path)
                                # try to derive cci/hazard/survival from df if possible
                                sim = {
                                    "cci": (
                                        df["avg_consciousness"].to_numpy()
                                        if "avg_consciousness" in df.columns
                                        else np.full(horizon, parent_cci * 0.6)
                                    ),
                                    "hazard": np.zeros(horizon),
                                    "survival": np.clip(
                                        np.full(horizon, 0.9), 0.0, 1.0
                                    ),
                                }
                        except Exception as e:
                            print(
                                f"[adapter] sim.run_sim call failed, falling back to surrogate: {e}"
                            )
                            sim = None

                    if sim is None:
                        sim = simulate_child(
                            parent_cci, epsilon, lambda_star, agents, horizon, rng
                        )

                    cci_stab_mean, cci_stab_slope = stability_window_stats(sim["cci"])
                    haz_stab_mean, _ = stability_window_stats(sim["hazard"])
                    surv_stab_mean, _ = stability_window_stats(sim["survival"])

                    creation_success = int(
                        (cci_stab_mean >= thresholds["cci_min"])
                        and (haz_stab_mean <= thresholds["hazard_max"])
                    )
                    coherence_inheritance_ratio = cci_stab_mean / max(parent_cci, 1e-6)
                    # Placeholder eta estimate (use your entropy/eta metric if available):
                    eta_parent = 0.20 - 0.10 * (parent_cci - 0.7)
                    eta_child = max(0.0, 0.18 - 0.12 * (cci_stab_mean - 0.6))
                    entropy_transfer_efficiency = (eta_parent - eta_child) / max(
                        epsilon, 1e-6
                    )

                    run_id = f"pcci{parent_cci:.2f}_e{epsilon:.4f}_lam{lambda_star:.2f}_a{agents}_s{seed}"

                    # long trajectories (thin logging every 10)
                    for t in range(0, len(sim["cci"]), 10):
                        traj_rows.append(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "epoch": t,
                                "CCI": float(sim["cci"][t]),
                                "hazard": float(sim["hazard"][t]),
                                "survival_rate": float(sim["survival"][t]),
                            }
                        )

                    rows_summary.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "parent_cci": parent_cci,
                            "epsilon": epsilon,
                            "lambda_star": lambda_star,
                            "agents": agents,
                            "creation_success": creation_success,
                            "coherence_inheritance_ratio": float(
                                coherence_inheritance_ratio
                            ),
                            "entropy_transfer_efficiency": float(
                                entropy_transfer_efficiency
                            ),
                            "cci_stability_mean": float(cci_stab_mean),
                            "cci_stability_slope": float(cci_stab_slope),
                            "hazard_stability_mean": float(haz_stab_mean),
                            "survival_rate_final": float(sim["survival"][-1]),
                            "time_sec": np.nan,
                        }
                    )

    # write CSVs
    import csv

    if len(rows_summary) == 0:
        raise RuntimeError("No runs generated; check sweep configuration")

    with open(base / "data" / "runs_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_summary[0].keys()))
        w.writeheader()
        w.writerows(rows_summary)

    with open(base / "data" / "trajectories_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader()
        w.writerows(traj_rows)

    # minimal JSON summary; OpenLaws validate/report can extend
    summary = {
        "study_id": config.get("study_id", "phase23_creator_emergence"),
        "n_runs": len(rows_summary),
        "thresholds": thresholds,
        "notes": "Adapter uses a light surrogate; replace with project sim core for production.",
    }
    with open(base / "data" / "summary_bootstrap.json", "w") as f:
        json.dump(summary, f, indent=2)

    # return paths for OpenLaws pipeline
    return {
        "runs_summary_csv": str(base / "data" / "runs_summary.csv"),
        "trajectories_long_csv": str(base / "data" / "trajectories_long.csv"),
        "summary_json": str(base / "data" / "summary_bootstrap.json"),
    }
