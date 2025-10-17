from dataclasses import dataclass
from typing import Any

import numpy as np

rng = np.random.default_rng


@dataclass
class Phase33Config:
    agents: int
    steps: int
    dt: float
    eps: float
    noise_base: float
    governance_on_step: int
    perturb_step: int
    perturb_strength: float
    seeds: list


def _cci_from_components(cal, coh, em, noise):
    noise = np.clip(noise, 1e-3, 1.0)
    cal = np.clip(cal, 0.0, 1.0)
    coh = np.clip(coh, 0.0, 1.0)
    em = np.clip(em, 0.0, 1.0)
    raw = (cal * coh * em) / noise
    return raw / (1.0 + raw)


def _governance_decay(noise, t, on_step):
    if t < on_step:
        return noise
    factor = 0.8 + 0.2 * np.exp(-(t - on_step) / 40.0)
    return np.maximum(1e-3, noise * factor)


def _consensus_elasticity(cxci_pre, cxci_min_post, cxci_recovered):
    drop = cxci_pre - cxci_min_post
    rec = cxci_recovered - cxci_min_post
    if drop <= 1e-9:
        return 1.0
    return np.clip(rec / drop, 0.0, 1.5)


def run_phase33_meta(params: dict[str, Any], out_dir: str = None, seed: int = None):
    cfg = Phase33Config(
        agents=params["run"]["agents"],
        steps=params["run"]["steps"],
        dt=params["run"]["dt"],
        eps=params["run"]["eps"],
        noise_base=params["run"]["noise_base"],
        governance_on_step=params["run"]["governance_on_step"],
        perturb_step=params["run"]["perturb_step"],
        perturb_strength=params["run"]["perturb_strength"],
        seeds=params["preregistration"]["constants"]["seeds"],
    )

    ts = []
    per_seed_metrics = []

    def _run_seed(s):
        r = rng(s)
        cal = np.clip(r.normal(0.90, 0.03, cfg.agents), 0.75, 1.0)
        coh = np.clip(r.normal(0.85, 0.04, cfg.agents), 0.65, 1.0)
        em = np.clip(r.normal(0.80, 0.05, cfg.agents), 0.55, 1.0)
        noise = np.clip(r.normal(cfg.noise_base, 0.01, cfg.agents), 0.02, 0.30)

        cci_prev_mean = None
        epsilon_c_accum = 0.0
        epsilon_c_count = 0

        cxci_series = []
        eta_series = []
        local_ts = []

        for t in range(cfg.steps):
            cal = np.clip(cal + cfg.eps * 0.04 + r.normal(0, 0.002, cfg.agents), 0, 1)
            coh = np.clip(coh + cfg.eps * 0.05 + r.normal(0, 0.002, cfg.agents), 0, 1)
            em = np.clip(em + cfg.eps * 0.03 + r.normal(0, 0.002, cfg.agents), 0, 1)

            noise = np.clip(noise + r.normal(0, 0.001, cfg.agents), 1e-3, 1.0)
            noise = _governance_decay(noise, t, cfg.governance_on_step)

            if t == cfg.perturb_step:
                coh = np.clip(coh * (1.0 - cfg.perturb_strength), 0, 1)
                em = np.clip(em * (1.0 - cfg.perturb_strength), 0, 1)

            cci = _cci_from_components(cal, coh, em, noise)
            cci_mean = float(np.mean(cci))

            if t > 3:
                if cci_prev_mean is not None:
                    dcci_dt = (cci_mean - cci_prev_mean) / cfg.dt
                    epsilon_c_accum += max(0.0, dcci_dt)
                    epsilon_c_count += 1
                cci_prev_mean = cci_mean
            else:
                cci_prev_mean = cci_mean

            eta_context = float(np.var(cci))
            cxci = float(cci_mean / (eta_context + 1e-6))

            cxci_series.append(cxci)
            eta_series.append(eta_context)

            local_ts.append(
                {
                    "seed": s,
                    "t": t,
                    "CCI_mean": cci_mean,
                    "CxCI": cxci,
                    "eta_context": eta_context,
                }
            )

        pre = (
            np.mean(cxci_series[max(0, cfg.perturb_step - 5) : cfg.perturb_step])
            if cfg.perturb_step > 5
            else cxci_series[cfg.perturb_step - 1]
        )
        post_window = cxci_series[cfg.perturb_step : cfg.perturb_step + 5]
        cxci_min_post = (
            float(np.min(post_window))
            if len(post_window)
            else cxci_series[cfg.perturb_step]
        )
        recovered = np.mean(cxci_series[-5:])
        E_cons = _consensus_elasticity(pre, cxci_min_post, recovered)

        epsilon_c = epsilon_c_accum / max(1, epsilon_c_count)

        metrics = {
            "seed": s,
            "CCI_mean": float(np.mean([row["CCI_mean"] for row in local_ts])),
            "CxCI": float(np.mean(cxci_series)),
            "epsilon_c": float(epsilon_c),
            "eta_context": float(np.mean(eta_series)),
            "E_cons": float(E_cons),
            "CxCI_minus_meanCCI": float(np.mean(cxci_series))
            - float(np.mean([row["CCI_mean"] for row in local_ts])),
        }
        return metrics, local_ts

    # If requested to run a single seed and write to out_dir, do so and return file paths
    if seed is not None and out_dir is not None:
        metrics, local_ts = _run_seed(seed)
        import csv
        import json
        from pathlib import Path

        outp = Path(out_dir)
        data_dir = outp / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # write results json
        with open(data_dir / f"results_seed_{seed}.json", "w") as jf:
            json.dump({"summary": metrics}, jf, indent=2)

        # write timeseries CSV
        ts_file = data_dir / f"timeseries_seed_{seed}.csv"
        with open(ts_file, "w", newline="") as cf:
            writer = csv.DictWriter(
                cf, fieldnames=["seed", "t", "CCI_mean", "CxCI", "eta_context"]
            )
            writer.writeheader()
            for row in local_ts:
                writer.writerow(row)

        # write per-seed runs_summary CSV (single row)
        runs_file = data_dir / f"runs_summary_{seed}.csv"
        with open(runs_file, "w", newline="") as rf:
            w = csv.DictWriter(rf, fieldnames=list(metrics.keys()))
            w.writeheader()
            w.writerow(metrics)

        return {
            "runs_summary_csv": str(runs_file),
            "trajectories_long_csv": str(ts_file),
        }

    # Otherwise run all preregistered seeds and return aggregated summary/artifacts
    for s in cfg.seeds:
        metrics, local_ts = _run_seed(s)
        per_seed_metrics.append(metrics)
        ts.extend(local_ts)

    CCI_mean = float(np.mean([m["CCI_mean"] for m in per_seed_metrics]))
    CxCI = float(np.mean([m["CxCI"] for m in per_seed_metrics]))
    epsilon_c = float(np.mean([m["epsilon_c"] for m in per_seed_metrics]))
    eta_ctx = float(np.mean([m["eta_context"] for m in per_seed_metrics]))
    E_cons = float(np.mean([m["E_cons"] for m in per_seed_metrics]))

    summary = {
        "CCI_mean": CCI_mean,
        "CxCI": CxCI,
        "epsilon_c": epsilon_c,
        "eta_context": eta_ctx,
        "E_cons": E_cons,
        "CxCI_minus_meanCCI": CxCI - CCI_mean,
    }

    artifacts = {"timeseries": ts, "per_seed_metrics": per_seed_metrics}
    return {"summary": summary, "artifacts": artifacts}
