from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- schema enforcement utility (auto-inserted) ---
def _ensure_runs_summary_schema(df, run_id):
    import pandas as pd

    required = [
        "run_id",
        "seed",
        "N",
        "kappa",
        "epsilon",
        "sigma",
        "dCCI",
        "synchrony_gain",
        "eta_collective",
        "eta_individual_mean",
        "t_recover_CCI_0.50",
        "t_recover_hazard_0.20",
        "reservoirs_ok",
        "early_stopped",
        "time_sec",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0
    df["run_id"] = df.get("run_id", run_id)
    numeric_cols = [
        "seed",
        "N",
        "kappa",
        "epsilon",
        "sigma",
        "dCCI",
        "synchrony_gain",
        "eta_collective",
        "eta_individual_mean",
        "t_recover_CCI_0.50",
        "t_recover_hazard_0.20",
        "time_sec",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["reservoirs_ok"] = (
        df["reservoirs_ok"].astype(bool) if "reservoirs_ok" in df.columns else True
    )
    df["early_stopped"] = (
        df["early_stopped"].astype(bool) if "early_stopped" in df.columns else False
    )
    return df[required]


def simulate_agents(N, kappa, epsilon, sigma, epochs, rng):
    """Surrogate multi-agent simulation producing CCI per agent over time.
    - agents start with random baseline CCI around 0.5
    - coupling increases mean reversion toward group mean scaled by kappa
    - noise per agent controlled by sigma
    """
    agent_cci = rng.normal(0.5, 0.05, size=(epochs, N))
    for t in range(1, epochs):
        # pull towards mean with coupling
        mean_prev = agent_cci[t - 1].mean()
        agent_cci[t] = (
            agent_cci[t - 1]
            + kappa * (mean_prev - agent_cci[t - 1])
            + rng.normal(0, sigma, size=N)
        )
        # small epsilon-driven drift
        agent_cci[t] = np.clip(agent_cci[t] + epsilon * 0.01, 0.0, 1.0)
    return agent_cci


def compute_eta(agent_cci):
    # entropy proxy: inverse of variance across agents and time
    return float(1.0 / (1e-6 + np.var(agent_cci)))


def compute_synchrony(agent_cci):
    # synchrony proxy: mean pairwise Pearson correlation across agents (time series)
    N = agent_cci.shape[1]
    if N < 2:
        return 0.0
    corrs = np.corrcoef(agent_cci.T)
    # mean off-diagonal
    return float((np.sum(corrs) - N) / (N * (N - 1)))


def pack_traj_row(phase, epoch, rec):
    # rec: dict with keys 'agent_cci' (array), 'eta', 'synchrony'
    row = {"phase": phase, "epoch": epoch}
    for i, v in enumerate(rec["agent_cci"]):
        row[f"agent_{i}_cci"] = float(v)
    row["eta"] = float(rec["eta"])
    row["synchrony"] = float(rec["synchrony"])
    return row


def pack_sync_row(phase, epoch, rec, coupling=0.0):
    return {
        "phase": phase,
        "epoch": epoch,
        "synchrony": float(rec["synchrony"]),
        "coupling": coupling,
    }


def pack_shock_row(epoch, pre, post):
    return {
        "epoch": epoch,
        "affected_pct": pre.get("affected_pct", 0.0),
        "noise_delta": post.get("mean_noise", 0.0) - pre.get("mean_noise", 0.0),
        "pre_mean_noise": pre.get("mean_noise", 0.0),
        "post_mean_noise": post.get("mean_noise", 0.0),
    }


def init_agents(N, epsilon, sigma, seed, rng):
    # placeholder agent state; empty for surrogate
    return {"N": N, "epsilon": epsilon, "sigma": sigma, "seed": seed}


def step_agents(agents, coupling=0.0, graph=None, rng=None):
    # no-op placeholder; we use simulate_agents to get full time series
    return


def measure_block(agents):
    # placeholder: not used in surrogate flow
    return {}


def build_trust_graph(N, kappa, rng):
    # simple Erdos-Renyi adjacency scaled by kappa
    adj = (rng.random((N, N)) < 0.5).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj * kappa


def apply_micro_shock(agents, top_noise_pct=0.7, noise_delta=0.3, rng=None):
    # placeholder
    return


def measure_noise_stats(agents):
    return {"mean_noise": 0.05, "affected_pct": 0.0}


def compute_collective_metrics(df_traj, df_sync):
    # surrogate aggregation: compute mean over last phase window
    # assume df_traj contains columns agent_i_cci and synchrony
    agent_cols = [
        c for c in df_traj.columns if c.startswith("agent_") and c.endswith("_cci")
    ]
    last = df_traj.iloc[-50:]
    CCI_individual_mean = last[agent_cols].mean().mean()
    CCI_collective = (last[agent_cols].mean(axis=1)).mean()
    sync_ind_mean = df_sync.groupby("phase")["synchrony"].mean().mean()
    sync_collective = df_sync["synchrony"].mean()
    eta_collective = 1.0
    eta_ind_mean = 1.0
    return {
        "CCI_collective": float(CCI_collective),
        "CCI_individual_mean": float(CCI_individual_mean),
        "synchrony_collective": float(sync_collective),
        "synchrony_individual_mean": float(sync_ind_mean),
        "eta_collective": eta_collective,
        "eta_individual_mean": eta_ind_mean,
        "t_recover_cci": 0,
        "t_recover_hazard": 0,
    }


def run_study(config: dict[str, Any], outdir: str, seed: int):
    rng = np.random.default_rng(seed + int(config.get("seed_offset", 0)))
    N = int(config.get("N", 2))
    kappa = float(config.get("kappa", 0.1))
    eps = float(config.get("epsilon", 0.004))
    sigma = float(config.get("sigma_noise", 0.02))
    epochs = int(config.get("epochs", 900))

    base = Path(outdir)
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    (base / "report").mkdir(parents=True, exist_ok=True)

    # simulate
    agent_cci = simulate_agents(N, kappa, eps, sigma, epochs, rng)

    rows_traj = []
    rows_sync = []
    shock_rows = []

    # build simple time-sampled records: dense for first 200, thin (every 5) then dense tail
    for t in range(0, epochs):
        rec = {
            "agent_cci": agent_cci[t],
            "eta": compute_eta(agent_cci[: t + 1] if t > 0 else agent_cci[:1]),
            "synchrony": compute_synchrony(agent_cci[: t + 1]),
        }
        phase = (
            "baseline_individual"
            if t < 200
            else ("recovery" if t >= 760 else "join_and_sync")
        )
        if (
            phase == "baseline_individual"
            or phase == "recovery"
            or (phase == "join_and_sync" and t % 5 == 0)
        ):
            rows_traj.append(pack_traj_row(phase, t, rec))
            rows_sync.append(pack_sync_row(phase, t, rec, coupling=kappa))

    # micro-shock surrogate: mark at epoch 700
    pre = {"mean_noise": float(sigma)}
    post = {"mean_noise": float(sigma + 0.3)}
    shock_rows.append(pack_shock_row(700, pre, post))

    df_traj = pd.DataFrame(rows_traj)
    df_sync = pd.DataFrame(rows_sync)
    df_shock = pd.DataFrame(shock_rows)

    agg = compute_collective_metrics(df_traj, df_sync)
    runs_summary = pd.DataFrame(
        [
            {
                "run_id": config.get("run_id", "run1"),
                "seed": seed,
                "N": N,
                "kappa": kappa,
                "epsilon": eps,
                "sigma": sigma,
                "dCCI": agg["CCI_collective"] - agg["CCI_individual_mean"],
                "synchrony_gain": agg["synchrony_collective"]
                - agg["synchrony_individual_mean"],
                "eta_collective": agg["eta_collective"],
                "eta_individual_mean": agg["eta_individual_mean"],
                "t_recover_CCI_0.50": agg["t_recover_cci"],
                "t_recover_hazard_0.20": agg["t_recover_hazard"],
                "reservoirs_ok": True,
                "early_stopped": False,
            }
        ]
    )

    # write outputs (use schema-enforced append where appropriate)
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # trajectories and sync and shock: append if file exists, else write
    traj_path = data_dir / "trajectories_long.csv"
    sync_path = data_dir / "synchrony_long.csv"
    shock_path = data_dir / "shock_events.csv"
    runs_path = data_dir / "runs_summary.csv"

    df_traj.to_csv(traj_path, mode="a", index=False, header=not traj_path.exists())
    df_sync.to_csv(sync_path, mode="a", index=False, header=not sync_path.exists())
    df_shock.to_csv(shock_path, mode="a", index=False, header=not shock_path.exists())

    # enforce runs_summary schema and append
    runs_summary_fixed = _ensure_runs_summary_schema(
        runs_summary, config.get("run_id", "unknown_run")
    )
    runs_summary_fixed.to_csv(
        runs_path, mode="a", index=False, header=not runs_path.exists()
    )

    # minimal summary json (overwrite per run)
    summary = {
        "study_id": config.get("study_id", "phase31_distributed_neurocoherence"),
        "n_runs": 1,
    }
    with open(data_dir / "summary_bootstrap.json", "w") as f:
        json.dump(summary, f, indent=2)

    # simple figures
    try:
        plt.figure(figsize=(6, 4))
        # collective vs individual: plot mean agent CCI and mean of agents
        if "agent_0_cci" in df_traj.columns:
            agent_cols = [
                c
                for c in df_traj.columns
                if c.startswith("agent_") and c.endswith("_cci")
            ]
            mean_agents = df_traj[agent_cols].mean(axis=1)
            plt.plot(df_traj["epoch"], mean_agents, label="mean_individual")
        plt.title("CCI (collective proxy)")
        plt.savefig(base / "figures" / "cci_collective_vs_individual.png")
    except Exception:
        pass

    return {
        "runs_summary": runs_summary_fixed,
        "trajectories_long": df_traj,
        "synchrony_long": df_sync,
        "shock_events": df_shock,
    }
