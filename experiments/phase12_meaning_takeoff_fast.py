#!/usr/bin/env python3
"""
Fast-mode Phase 12: Meaning Take-Off Calibration — single-seed, shorter epochs
This is a quicker run to produce a valid baseline and takeoff summary within ~90s if possible.
"""
import datetime as dt
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("./discovery_results/phase12")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# reuse Agent and run_condition from original script by importing if available
# lightweight duplicate definitions (avoid import issues in fast runner)


class Agent:
    def __init__(self, aid, noise_base=0.08):
        self.aid = aid
        self.noise = noise_base + np.random.uniform(-0.01, 0.01)
        self.coherence = 0.6 + np.random.uniform(-0.02, 0.02)
        self.coordination = np.random.uniform(0.3, 0.6)
        self.calibration = np.random.uniform(0.35, 0.65)
        self.energy = np.random.uniform(0.5, 1.0)
        self.info = np.random.uniform(0.5, 1.0)

    def cci(self):
        cf = max(0, self.coherence - 0.4) / 0.6
        cof = max(0, self.coordination - 0.2) / 0.8
        calf = max(0, self.calibration - 0.3) / 0.7
        np_pen = max(0, 1.0 - self.noise / 0.3)
        return float(np.clip(0.4 * cf + 0.3 * cof + 0.2 * calf + 0.1 * np_pen, 0, 1))

    def step(self, openness, kE, kI, dt=0.02):
        cci = self.cci()
        effE = kE * (0.8 + 0.4 * openness)
        effI = kI * (0.6 + 0.6 * openness)
        self.energy = np.clip(self.energy + dt * effE * (1.0 - self.energy), 0.1, 1.0)
        self.info = np.clip(self.info + dt * effI * (0.9 - self.info), 0.2, 1.0)
        energy_boost = 0.12 * (self.energy - 0.5)
        info_boost = 0.10 * (self.info - 0.6)
        coh_target = 0.60 + 0.18 * openness
        rate = 1.8 + energy_boost + info_boost
        self.coherence += dt * rate * (coh_target - self.coherence)
        self.coordination += dt * 1.2 * (
            0.45 + 0.15 * self.info - self.coordination
        ) + np.random.normal(0, 0.01)
        self.calibration += dt * (0.6 * self.energy) * (
            0.55 - self.calibration
        ) + np.random.normal(0, 0.01)
        resfac = 0.6 * self.energy + 0.4 * self.info
        noise_target = 0.12 - 0.08 * cci * resfac
        self.noise += dt * 1.2 * (noise_target - self.noise)
        self.coherence = np.clip(self.coherence, 0.2, 0.95)
        self.coordination = np.clip(self.coordination, 0.1, 0.9)
        self.calibration = np.clip(self.calibration, 0.2, 0.85)
        self.noise = np.clip(self.noise, 0.01, 0.6)


def run_condition(
    n_agents,
    epochs,
    openness,
    kE,
    kI,
    seed,
    noise_base=0.08,
    null_shuffle=False,
    thin_out=15,
):
    np.random.seed(seed)
    agents = [Agent(i, noise_base=noise_base) for i in range(n_agents)]
    if null_shuffle:
        kI_seq = np.full(epochs, kI)
        np.random.shuffle(kI_seq)
    else:
        kI_seq = np.full(epochs, kI)
    epoch_records = []
    shock_events = []
    dense_window = set(range(580, 621))
    shock_start, shock_end = 600, 604
    for epoch in range(int(epochs)):
        kI_epoch = kI_seq[epoch]
        for ag in agents:
            ag.step(openness, kE, kI_epoch)
        if kI_epoch > 0:
            n_links = int(0.02 * n_agents * kI_epoch)
            for _ in range(max(1, n_links)):
                i, j = np.random.randint(0, n_agents, 2)
                if i == j:
                    continue
                mstr = 0.08 * min(agents[i].info, agents[j].info)
                agents[i].calibration = np.clip(
                    agents[i].calibration + 0.5 * mstr, 0.2, 0.85
                )
                agents[j].calibration = np.clip(
                    agents[j].calibration + 0.5 * mstr, 0.2, 0.85
                )
        if shock_start <= epoch <= shock_end:
            noises = sorted(
                [(idx, ag.noise) for idx, ag in enumerate(agents)],
                key=lambda x: x[1],
                reverse=True,
            )
            topn = int(0.7 * n_agents)
            pre_mean = np.mean([ag.noise for ag in agents])
            for idx, _ in noises[:topn]:
                agents[idx].noise = np.clip(agents[idx].noise * (1.0 + 0.3), 0.01, 0.8)
                agents[idx].energy *= 0.85
                agents[idx].info *= 0.9
            post_mean = np.mean([ag.noise for ag in agents])
            shock_events.append(
                {
                    "epoch": epoch,
                    "affected_pct": topn / n_agents * 100,
                    "noise_delta": post_mean - pre_mean,
                    "pre_noise_mean": pre_mean,
                    "post_noise_mean": post_mean,
                }
            )
        mean_hazard = np.mean([ag.noise for ag in agents])
        if mean_hazard > 0.35:
            noises = sorted(
                [(i, ag.noise) for i, ag in enumerate(agents)],
                key=lambda x: x[1],
                reverse=True,
            )
            ntrim = int(0.3 * n_agents)
            for idx, _ in noises[:ntrim]:
                agents[idx].noise *= 0.8
        if len(epoch_records) >= 50:
            recent = [r["system_cci"] for r in epoch_records[-50:]]
            slope = (recent[-1] - recent[0]) / 49
            if slope < 0.001:
                for ag in agents:
                    ag.coordination = min(0.70, ag.coordination + 0.05)
        ccis = np.array([ag.cci() for ag in agents])
        noises = np.array([ag.noise for ag in agents])
        cohs = np.array([ag.coherence for ag in agents])
        mean_cci = float(ccis.mean())
        mean_hazard = float(noises.mean())
        survival = float((ccis > 0.3).mean())
        if epoch in dense_window or epoch % thin_out == 0:
            epoch_records.append(
                {
                    "epoch": epoch,
                    "system_cci": mean_cci,
                    "system_hazard": mean_hazard,
                    "system_coherence": float(cohs.mean()),
                    "survival_rate": survival,
                    "openness": openness,
                    "kE": kE,
                    "kI": kI_epoch,
                    "null_shuffle": bool(null_shuffle),
                }
            )
        if epoch > 300 and mean_hazard > 0.45 and survival < 0.40 and mean_cci < 0.45:
            break
    df = pd.DataFrame(epoch_records)
    shock_df = pd.DataFrame(shock_events)
    if len(df) > 0:
        final_win = (
            df[df.epoch >= (df.epoch.max() - 199)] if df.epoch.max() >= 199 else df
        )
        stability_cci_mean = float(final_win.system_cci.mean())
        stability_hazard_mean = float(final_win.system_hazard.mean())
        if len(final_win) > 1:
            x = final_win.epoch.values
            y = final_win.system_cci.values
            slope = np.polyfit(x - x[0], y, 1)[0]
        else:
            slope = 0.0
        auh = 0.0
        sw = df[(df.epoch >= 590) & (df.epoch <= 610)]
        if len(sw) > 1:
            auh = float(np.trapz(sw.system_hazard.values, sw.epoch.values))
    else:
        stability_cci_mean = 0.0
        stability_hazard_mean = 1.0
        slope = 0.0
        auh = 0.0
    return {
        "trajectories": df,
        "shock_events": shock_df,
        "metrics": {
            "stability_cci_mean": stability_cci_mean,
            "stability_hazard_mean": stability_hazard_mean,
            "stability_cci_slope": slope,
            "auh_590_610": auh,
        },
        "parameters": {
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "kE": kE,
            "kI": kI,
            "null_shuffle": null_shuffle,
            "seed": seed,
        },
    }


def run_fast():
    print("⚡ Phase12 (FAST) starting — seeds=1, epochs=800")
    n_agents = 150
    seeds = [1]
    epsilons = [0.005, 0.006, 0.007]
    kE_fixed = 1.0
    kIs = [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7]
    epochs_cap = 800
    noise_base = 0.08
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 15
    # include explicit baseline run (kE=1.0,kI=1.0)
    for eps in epsilons:
        for seed in seeds:
            cid += 1
            res = run_condition(
                n_agents,
                epochs_cap,
                eps,
                1.0,
                1.0,
                seed,
                noise_base=noise_base,
                null_shuffle=False,
                thin_out=thin_out,
            )
            res["condition_id"] = cid
            all_results.append(res)
            for kI in kIs:
                cid += 1
                res = run_condition(
                    n_agents,
                    epochs_cap,
                    eps,
                    kE_fixed,
                    kI,
                    seed,
                    noise_base=noise_base,
                    null_shuffle=False,
                    thin_out=thin_out,
                )
                res["condition_id"] = cid
                all_results.append(res)
    runtime = time.time() - start
    print(f"✅ Fast Phase12 complete: {runtime:.2f}s")
    return all_results, runtime


def analyze_export(all_results, runtime):
    # same analysis as original but simpler
    trajs = []
    shocks = []
    runs = []
    conds = []
    for r in all_results:
        df = r["trajectories"].copy()
        df["condition_id"] = r["condition_id"]
        trajs.append(df)
        if len(r["shock_events"]) > 0:
            s = r["shock_events"].copy()
            s["condition_id"] = r["condition_id"]
            shocks.append(s)
        params = r["parameters"]
        metrics = r["metrics"]
        runs.append(
            {
                "condition_id": int(r["condition_id"]),
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in params.items()
                },
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in metrics.items()
                },
            }
        )
        conds.append(
            {
                "condition_id": int(r["condition_id"]),
                "openness": float(params["openness"]),
                "kE": float(params["kE"]),
                "kI": float(params["kI"]),
                "null_shuffle": bool(params["null_shuffle"]),
            }
        )
    traj_df = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    shock_df = pd.concat(shocks, ignore_index=True) if shocks else pd.DataFrame()
    runs_df = pd.DataFrame(runs)
    conds_df = pd.DataFrame(conds)
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA_DIR / "trajectories_long.csv", index=False)
    if not shock_df.empty:
        shock_df.to_csv(DATA_DIR / "shock_events.csv", index=False)
    conds_df.to_csv(DATA_DIR / "conditions_lookup.csv", index=False)
    per_eps = []
    for eps in sorted(runs_df.openness.unique()):
        sub = runs_df[runs_df.openness == eps]

        def cci_at(kE, kI):
            sel = sub[(sub.kE == kE) & (sub.kI == kI)]
            return float(sel.stability_cci_mean.mean()) if len(sel) > 0 else np.nan

        baseline = cci_at(1.0, 1.0)
        cci_e_hi = cci_at(1.2, 1.0)
        cci_e_lo = cci_at(0.8, 1.0)
        delta_E = (
            cci_e_hi - cci_e_lo
            if not (np.isnan(cci_e_hi) or np.isnan(cci_e_lo))
            else np.nan
        )
        records = []
        for kI in sorted([k for k in sub.kI.unique() if k != 1.0]):
            cci_k = cci_at(1.0, kI)
            delta_I = (
                cci_k - baseline
                if not np.isnan(cci_k) and not np.isnan(baseline)
                else np.nan
            )
            k_prev = round(kI - 0.05, 3)
            k_next = round(kI + 0.05, 3)
            cci_prev = cci_at(1.0, k_prev)
            cci_next = cci_at(1.0, k_next)
            curvature_I = (
                cci_next - 2 * cci_k + cci_prev
                if not any(np.isnan(x) for x in [cci_prev, cci_k, cci_next])
                else np.nan
            )
            records.append(
                {"kI": kI, "cci": cci_k, "delta_I": delta_I, "curvature_I": curvature_I}
            )
        takeoffs = []
        kI_star = None
        for rec in records:
            takeoff = (
                not np.isnan(rec["curvature_I"]) and rec["curvature_I"] > 0
            ) and (
                not np.isnan(rec["delta_I"])
                and not np.isnan(delta_E)
                and rec["delta_I"] > delta_E
            )
            takeoffs.append({**rec, "takeoff_flag": bool(takeoff)})
            if takeoff and kI_star is None:
                kI_star = rec["kI"]
        per_eps.append(
            {
                "openness": float(eps),
                "baseline_cci": float(baseline) if not np.isnan(baseline) else None,
                "delta_E": float(delta_E) if not np.isnan(delta_E) else None,
                "takeoffs": takeoffs,
                "kI_star": kI_star,
            }
        )
    summary = {
        "experiment": "Phase12_FAST",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "per_epsilon": per_eps,
    }
    with open(DATA_DIR / "phase12_takeoff_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    all_results, runtime = run_fast()
    summary = analyze_export(all_results, runtime)
    print("\nFAST Phase12 complete. kI* per eps:")
    for p in summary["per_epsilon"]:
        print(
            f"eps={p['openness']:.3f}: kI*={p['kI_star']} (baseline_CCI={p['baseline_cci']})"
        )
    print("Outputs in discovery_results/phase12/")
