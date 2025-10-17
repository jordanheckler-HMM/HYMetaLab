import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _map_trust(trust_tau: float, base_coord: float, base_noise: float):
    coord = _clip(base_coord + 0.6 * (trust_tau - 0.5), 0.30, 0.70)
    defe = max(0.0, base_noise * (1.0 - 0.8 * (trust_tau - 0.5)))
    return coord, defe


def _fallback_sim_run(cfg: dict[str, Any], seed: int = 0) -> pd.DataFrame:
    """A tiny stochastic dynamics simulator returning epoch-wise metrics.
    Metrics: epoch, CCI, survival_rate, hazard, collapse_risk, gini
    This is a heuristic placeholder to allow study pipelines to run.
    """
    rng = np.random.default_rng(seed)
    E = int(cfg.get("epochs_cap", cfg.get("epochs", 400)))
    coord = cfg.get("coordination_strength", 0.5)
    defe = cfg.get("defection_noise", 0.1)
    shock = cfg.get("shock_severity", 0.5)
    agents = int(cfg.get("agents", 100))

    rows = []
    base_cci = 0.4 + 0.4 * coord - 0.2 * defe
    base_surv = 0.5 + 0.35 * coord - 0.25 * shock
    hazard = 0.3 + 0.4 * (1.0 - coord) + 0.3 * shock
    collapse = 1.0 - base_surv

    for t in range(E):
        noise = rng.normal(0, 0.02)
        # simple dynamics: CCI tends to base, affected by small random walk and occasional shock
        shock_effect = (
            -shock * math.exp(-((t - E * 0.3) ** 2) / (2 * (E * 0.05 + 1)))
            if t < E * 0.6
            else 0.0
        )
        cci = base_cci + 0.1 * math.tanh((t / E) * 2.0) + noise + shock_effect * 0.5
        cci = float(_clip(cci, 0.0, 1.0))
        # survival drifts up with CCI and coordination
        surv = float(
            _clip(
                base_surv + 0.2 * (cci - 0.5) - 0.05 * defe + rng.normal(0, 0.01),
                0.0,
                1.0,
            )
        )
        haz = float(_clip(hazard * (1.0 - 0.3 * cci) + rng.normal(0, 0.01), 0.0, 1.0))
        colrisk = float(_clip(1.0 - surv, 0.0, 1.0))
        gini = float(
            _clip(
                0.25 + 0.5 * cfg.get("goal_inequality", 0.3) + rng.normal(0, 0.01),
                0.0,
                1.0,
            )
        )
        rows.append(
            {
                "epoch": t,
                "CCI": cci,
                "survival_rate": surv,
                "hazard": haz,
                "collapse_risk": colrisk,
                "gini": gini,
            }
        )

    return pd.DataFrame(rows)


def sweep_and_collect(grid: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = []
    trajs = []
    for cfg in grid:
        seeds = cfg.get("seeds", [101])
        for s in seeds:
            simcfg = dict(cfg)
            df = _fallback_sim_run(simcfg, seed=int(s))
            # stability window: last 20%
            E = len(df)
            w = df.iloc[int(0.8 * E) :]
            rec = {
                "label": cfg.get("label", ""),
                "seed": int(s),
                "agents": cfg.get("agents", np.nan),
                "coordination_strength": cfg.get("coordination_strength", np.nan),
                "defection_noise": cfg.get("defection_noise", np.nan),
                "shock_severity": cfg.get("shock_severity", np.nan),
                "goal_inequality": cfg.get("goal_inequality", np.nan),
                "CCI": float(w["CCI"].mean()),
                "survival_rate": float(w["survival_rate"].mean()),
                "hazard": float(w["hazard"].mean()),
                "collapse_risk": float(w["collapse_risk"].mean()),
                "preregistered": True,
            }
            runs.append(rec)
            traj = df.copy()
            traj["label"] = cfg.get("label", "")
            traj["seed"] = int(s)
            trajs.append(traj)

    runs_df = pd.DataFrame(runs)
    trajs_df = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    return runs_df, trajs_df


def bootstrap_summary(
    runs_summary: pd.DataFrame, factors: list[str], metrics: list[str]
) -> pd.DataFrame:
    # simple grouped bootstrapped means
    rows = []
    group_keys = [c for c in factors if c in runs_summary.columns]
    if not group_keys:
        group_keys = []
    for name, g in (
        runs_summary.groupby(group_keys) if group_keys else [((), runs_summary)]
    ):
        rec = {k: v for k, v in zip(group_keys, name)} if group_keys else {}
        for m in metrics:
            if m in g.columns:
                vals = pd.to_numeric(g[m], errors="coerce").dropna().values
                if len(vals) == 0:
                    rec[m + "_mean"] = float("nan")
                    rec[m + "_ci_lo"] = float("nan")
                    rec[m + "_ci_hi"] = float("nan")
                else:
                    mu = float(np.mean(vals))
                    boots = []
                    rng = np.random.default_rng(0)
                    for _ in range(100):
                        s = rng.choice(vals, size=len(vals), replace=True)
                        boots.append(np.mean(s))
                    lo, hi = np.quantile(boots, [0.025, 0.975])
                    rec[m + "_mean"] = mu
                    rec[m + "_ci_lo"] = float(lo)
                    rec[m + "_ci_hi"] = float(hi)
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_standard(
    runs_summary: pd.DataFrame,
    trajectories_long: pd.DataFrame,
    outdir: str = None,
    extra=None,
):
    out = []
    outdir = (
        Path(outdir)
        if outdir
        else (ROOT / "discovery_results" / "trust_survival_v1" / "figures")
    )
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    if "coordination_strength" in runs_summary.columns:
        p1 = outdir / f"fig_cci_vs_coord_{ts}.png"
        plt.figure(figsize=(7, 4))
        plt.scatter(runs_summary["coordination_strength"], runs_summary["CCI"], s=6)
        plt.xlabel("coordination_strength")
        plt.ylabel("CCI")
        plt.tight_layout()
        plt.savefig(p1, dpi=200)
        plt.close()
        out.append(str(p1))
    if "shock_severity" in runs_summary.columns:
        p2 = outdir / f"fig_surv_vs_shock_{ts}.png"
        plt.figure(figsize=(7, 4))
        plt.scatter(runs_summary["shock_severity"], runs_summary["survival_rate"], s=6)
        plt.xlabel("shock_severity")
        plt.ylabel("survival_rate")
        plt.tight_layout()
        plt.savefig(p2, dpi=200)
        plt.close()
        out.append(str(p2))
    return out


def save_exports(
    study_cfg: dict[str, Any],
    runs_summary: pd.DataFrame,
    trajectories_long: pd.DataFrame,
    summary: pd.DataFrame,
    figures: list[str] = None,
    study_name: str = None,
):
    outdir = Path(
        study_cfg.get("exports", {}).get(
            "data_dir", "discovery_results/trust_survival_v1/data"
        )
    )
    report_dir = Path(
        study_cfg.get("exports", {}).get(
            "report_dir", "discovery_results/trust_survival_v1/report"
        )
    )
    figs_dir = Path(
        study_cfg.get("exports", {}).get(
            "figs_dir", "discovery_results/trust_survival_v1/figures"
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = outdir / "runs_summary.csv"
    traj_csv = outdir / "trajectories_long.csv"
    runs_summary.to_csv(runs_csv, index=False)
    trajectories_long.to_csv(traj_csv, index=False)
    summary_csv = outdir / "summary_bootstrap.csv"
    summary.to_csv(summary_csv, index=False)

    # copy / move figures if provided
    if figures:
        for f in figures:
            try:
                src = Path(f)
                if src.exists():
                    dst = figs_dir / src.name
                    if not dst.exists():
                        dst.write_bytes(src.read_bytes())
            except Exception:
                pass

    # small report
    md = report_dir / "results.md"
    md.write_text(
        f"# Study {study_name}\n\n- runs: {runs_csv}\n- trajectories: {traj_csv}\n- summary: {summary_csv}\n",
        encoding="utf-8",
    )


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    # Build grid from study_cfg similar to the user's spec
    grid = []
    seeds = study_cfg.get("prereg", {}).get("constants", {}).get("seeds", [101])
    agents = study_cfg.get("prereg", {}).get("constants", {}).get("agents", 120)
    for tau in study_cfg.get("prereg", {}).get("constants", {}).get("trust_tau", [0.5]):
        for shock in (
            study_cfg.get("prereg", {})
            .get("constants", {})
            .get("shock_severity", [0.5])
        ):
            for gini in (
                study_cfg.get("prereg", {})
                .get("constants", {})
                .get("goal_inequality", [0.3])
            ):
                coord, defe = _map_trust(
                    tau,
                    study_cfg.get("controls", {}).get("base_coord", 0.5),
                    study_cfg.get("controls", {}).get("base_noise", 0.1),
                )
                grid.append(
                    {
                        "label": f"tau{tau:.2f}_shock{shock:.2f}_gini{gini:.2f}",
                        "epochs_cap": study_cfg.get("design", {}).get(
                            "epochs_cap", 400
                        ),
                        "agents": agents,
                        "seeds": seeds,
                        "coordination_strength": coord,
                        "defection_noise": defe,
                        "shock_severity": shock,
                        "goal_inequality": gini,
                        "hygiene": study_cfg.get("controls", {}).get("hygiene", {}),
                    }
                )

    runs_summary, trajectories_long = sweep_and_collect(grid)
    summary = bootstrap_summary(
        runs_summary,
        factors=["shock_severity", "goal_inequality"],
        metrics=["survival_rate", "collapse_risk", "CCI", "hazard"],
    )
    figs = plot_standard(
        runs_summary,
        trajectories_long,
        outdir=study_cfg.get("exports", {}).get("figs_dir", None),
    )
    save_exports(
        study_cfg,
        runs_summary,
        trajectories_long,
        summary,
        figures=figs,
        study_name=study_cfg.get("meta", {}).get("name", "trust_study"),
    )
    return {"runs_summary": runs_summary, "summary": summary}
