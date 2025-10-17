#!/usr/bin/env python3
"""
Phase 13: Collective Intelligence Scaling — FAST BATTERY
"""
import datetime as dt
import hashlib
import json
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("./discovery_results/phase13")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
BUNDLE_DIR = ROOT / "bundle"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR, BUNDLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class Agent:
    def __init__(self, aid, noise_base=0.08, gini=0.30, kappa=0.5):
        self.aid = aid
        # initial heterogeneity via gini proxy: spread info/energy
        spread = gini - 0.15
        self.noise = noise_base + np.random.uniform(-0.01, 0.01)
        self.coherence = 0.6 + np.random.uniform(-0.02, 0.02)
        self.coordination = np.clip(kappa + np.random.normal(0, 0.05), 0.1, 0.9)
        self.calibration = np.random.uniform(0.35, 0.65)
        self.energy = np.clip(np.random.uniform(0.5 - spread, 1.0 + spread), 0.1, 1.0)
        self.info = np.clip(np.random.uniform(0.5 - spread, 1.0 + spread), 0.2, 1.0)

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


def run_sim(
    n_agents, epochs, openness, kE, kI, seed, kappa, gini, noise_base=0.08, thin_out=10
):
    np.random.seed(seed)
    agents = [
        Agent(i, noise_base=noise_base, gini=gini, kappa=kappa) for i in range(n_agents)
    ]
    epoch_records = []
    shock_events = []
    dense_window = set(range(880, 921))
    shock_start, shock_end = 900, 904
    for epoch in range(int(epochs)):
        # each agent steps
        for ag in agents:
            ag.step(openness, kE, kI)
        # messaging pairs scaled by kI
        if kI > 0:
            n_links = int(0.015 * n_agents * kI)
            for _ in range(max(1, n_links)):
                i, j = np.random.randint(0, n_agents, 2)
                if i == j:
                    continue
                mstr = 0.06 * min(agents[i].info, agents[j].info)
                agents[i].calibration = np.clip(
                    agents[i].calibration + 0.5 * mstr, 0.2, 0.85
                )
                agents[j].calibration = np.clip(
                    agents[j].calibration + 0.5 * mstr, 0.2, 0.85
                )
        # shock window
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
                }
            )
        # hygiene
        mean_hazard = np.mean([ag.noise for ag in agents])
        if mean_hazard > 0.35:
            noises = sorted(
                [(i, ag.noise) for i, ag in enumerate(agents)],
                key=lambda x: x[1],
                reverse=True,
            )
            ntrim = int(0.25 * n_agents)
            for idx, _ in noises[:ntrim]:
                agents[idx].noise *= 0.8
        # capture
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
                    "kI": kI,
                    "n_agents": n_agents,
                    "kappa": kappa,
                    "gini": gini,
                }
            )
        if epoch > 300 and mean_hazard > 0.45 and survival < 0.40 and mean_cci < 0.45:
            break
    df = pd.DataFrame(epoch_records)
    shock_df = pd.DataFrame(shock_events)
    # final metrics over last 200 epochs
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
        # shock resilience
        sw = df[(df.epoch >= 890) & (df.epoch <= 910)]
        auh = (
            float(np.trapz(sw.system_hazard.values, sw.epoch.values))
            if len(sw) > 1
            else np.nan
        )
        # CCI recovery t50
        pre_idx = df[df.epoch < 900]
        pre_cci = (
            float(pre_idx.system_cci.tail(10).mean()) if len(pre_idx) > 0 else np.nan
        )
        shock_min = (
            float(df[(df.epoch >= 900) & (df.epoch <= 910)].system_cci.min())
            if len(df[(df.epoch >= 900) & (df.epoch <= 910)]) > 0
            else np.nan
        )
        t50 = np.nan
        if not np.isnan(pre_cci) and not np.isnan(shock_min):
            target = shock_min + 0.5 * (pre_cci - shock_min)
            post = df[df.epoch > 910]
            if len(post) > 0:
                reach = post[post.system_cci >= target]
                if len(reach) > 0:
                    t50 = int(reach.epoch.iloc[0] - 904)
        # hazard_decay_t20: time after shock until hazard drops 20% from shock peak
        hazard_peak = (
            float(df[(df.epoch >= 900) & (df.epoch <= 910)].system_hazard.max())
            if len(df[(df.epoch >= 900) & (df.epoch <= 910)]) > 0
            else np.nan
        )
        hdecay = np.nan
        if not np.isnan(hazard_peak):
            target_h = hazard_peak * 0.8
            post = df[df.epoch > 910]
            if len(post) > 0:
                reach = post[post.system_hazard <= target_h]
                if len(reach) > 0:
                    hdecay = int(reach.epoch.iloc[0] - 904)
    else:
        stability_cci_mean = 0.0
        stability_hazard_mean = 1.0
        slope = 0.0
        auh = np.nan
        t50 = np.nan
        hdecay = np.nan
    return {
        "trajectories": df,
        "shock_events": shock_df,
        "metrics": {
            "stability_cci_mean": stability_cci_mean,
            "stability_hazard_mean": stability_hazard_mean,
            "stability_cci_slope": slope,
            "auh_890_910": auh,
            "CCI_recover_t50": t50,
            "hazard_decay_t20": hdecay,
        },
        "parameters": {
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "kE": kE,
            "kI": kI,
            "kappa": kappa,
            "gini": gini,
            "seed": seed,
        },
    }


def run_grid():
    print("⚡ Phase 13: Collective Intelligence Scaling — FAST BATTERY")
    Ns = [100, 200, 400, 800]
    seeds = [1]
    epochs_cap = 2000
    eps = 0.006
    kappas = [0.40, 0.50, 0.60]
    ginis = [0.15, 0.30]
    kE = 1.0
    kIs = [1.0, 1.2]
    noise_base = 0.08
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 10
    for kappa in kappas:
        for gini in ginis:
            for kI in kIs:
                for N in Ns[:]:
                    for seed in list(seeds):
                        cid += 1
                        # baseline individual (short run with kI=0) to compute individual CCI baseline
                        baseline = run_sim(
                            N,
                            40,
                            eps,
                            kE,
                            0.0,
                            seed,
                            kappa,
                            gini,
                            noise_base=noise_base,
                            thin_out=thin_out,
                        )
                        res = run_sim(
                            N,
                            epochs_cap,
                            eps,
                            kE,
                            kI,
                            seed,
                            kappa,
                            gini,
                            noise_base=noise_base,
                            thin_out=thin_out,
                        )
                        res["baseline_individual_cci"] = float(
                            baseline["metrics"]["stability_cci_mean"]
                        )
                        res["condition_id"] = cid
                        all_results.append(res)
                        # auto-cuts based on time
                        elapsed = time.time() - start
                        if elapsed > 60:
                            print(
                                "   ⚠️ runtime >60s: dropping N=800 for remaining runs"
                            )
                            Ns = [n for n in Ns if n != 800]
                        if elapsed > 90:
                            print("   ⚠️ runtime >90s: reducing epochs->1600")
                            epochs_cap = 1600
                        if elapsed > 110:
                            print(
                                "   ⚠️ runtime >110s: thinning logs outside dense window"
                            )
                            thin_out = 15
    runtime = time.time() - start
    print(f"✅ Phase 13 complete: {runtime:.2f}s (auto-cuts may have applied)")
    return all_results, runtime


def analyze_and_export(all_results, runtime):
    print("📊 Analyzing Phase 13 results...")
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
        baseline = r.get("baseline_individual_cci", np.nan)
        runs.append(
            {
                "condition_id": int(r["condition_id"]),
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in params.items()
                },
                "baseline_individual_cci": (
                    float(baseline) if not np.isnan(baseline) else None
                ),
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in metrics.items()
                },
            }
        )
        conds.append(
            {
                "condition_id": int(r["condition_id"]),
                "n_agents": int(params["n_agents"]),
                "openness": float(params["openness"]),
                "kE": float(params["kE"]),
                "kI": float(params["kI"]),
                "kappa": float(params["kappa"]),
                "gini": float(params["gini"]),
            }
        )
    traj_df = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    shock_df = pd.concat(shocks, ignore_index=True) if shocks else pd.DataFrame()
    runs_df = pd.DataFrame(runs)
    conds_df = pd.DataFrame(conds)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA_DIR / "trajectories_long.csv", index=False)
    if not shock_df.empty:
        shock_df.to_csv(DATA_DIR / "shock_events.csv", index=False)
    conds_df.to_csv(DATA_DIR / "conditions_lookup.csv", index=False)

    # compute collective_gain and scaling exponent gamma per (kappa,gini,kI)
    summary_list = []
    for (kappa, gini, kI), group in runs_df.groupby(["kappa", "gini", "kI"]):
        # for scaling fit collect stability_cci_mean per N
        Ns = group.n_agents.values
        CCIs = group.stability_cci_mean.values
        baseline_ind = group.baseline_individual_cci.values
        collective_gain = np.array(CCIs) / np.array(baseline_ind)
        gamma = np.nan
        r2 = np.nan
        if len(Ns) >= 2 and np.all(CCIs > 0):
            X = np.column_stack([np.log(Ns), np.ones(len(Ns))])
            y = np.log(CCIs)
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            gamma = float(coef[0])
            intercept = float(coef[1])
            ypred = X.dot(coef)
            ss_res = float(((y - ypred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # amplification flag rely on mean collective_gain and gamma
        mean_cgain = (
            float(np.nanmean(collective_gain)) if len(collective_gain) > 0 else np.nan
        )
        amplification_flag = (not np.isnan(mean_cgain) and mean_cgain > 1.0) and (
            not np.isnan(gamma) and gamma > 0
        )
        summary_list.append(
            {
                "kappa": float(kappa),
                "gini": float(gini),
                "kI": float(kI),
                "gamma": gamma,
                "r2": r2,
                "mean_collective_gain": mean_cgain,
                "amplification_flag": bool(amplification_flag),
            }
        )

    # write summary JSON
    summary = {
        "experiment": "Phase13 Scaling",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "scaling_summary": summary_list,
    }
    with open(DATA_DIR / "phase13_scaling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures
    try:
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (n, kI), g in traj_df.groupby(["n_agents", "kI"]):
                ax.plot(g.epoch, g.system_cci, label=f"N={int(n)},kI={kI}", alpha=0.7)
            ax.axvspan(900, 904, alpha=0.1, color="red")
            ax.set_title("CCI trajectories (Phase13)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("CCI")
            ax.legend(fontsize="small", ncol=2)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "cci_long.png")
            plt.close()
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (n, kI), g in traj_df.groupby(["n_agents", "kI"]):
                ax.plot(
                    g.epoch, g.system_hazard, label=f"N={int(n)},kI={kI}", alpha=0.7
                )
            ax.set_title("Hazard trajectories (Phase13)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Hazard")
            ax.legend(fontsize="small", ncol=2)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "hazard_long.png")
            plt.close()
        # collective_gain vs N for each (kappa,gini,kI)
        fig, axes = plt.subplots(
            len(kappas), len(ginis), figsize=(4 * len(ginis), 3 * len(kappas)), dpi=140
        )
        axes = np.array(axes).reshape(len(kappas), len(ginis))
        for i, kappa in enumerate(sorted(runs_df.kappa.unique())):
            for j, gini in enumerate(sorted(runs_df.gini.unique())):
                ax = axes[i, j]
                for kI in sorted(runs_df.kI.unique()):
                    grp = runs_df[
                        (runs_df.kappa == kappa)
                        & (runs_df.gini == gini)
                        & (runs_df.kI == kI)
                    ]
                    if len(grp) > 0:
                        Ns = grp.n_agents.values
                        cci = grp.stability_cci_mean.values
                        base = grp.baseline_individual_cci.values
                        gain = cci / base
                        ax.plot(Ns, gain, "-o", label=f"kI={kI}")
                ax.set_title(f"kappa={kappa}, gini={gini}")
                ax.set_xlabel("N")
                ax.set_ylabel("collective_gain")
                ax.grid(True)
                ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "collective_amplification.png")
        plt.close()
        # scaling exponent heatmap
        dfsum = pd.DataFrame(summary_list)
        # pivot gamma by (kappa,gini) for each kI and plot small grid per kI
        for kI in sorted(dfsum.kI.unique()):
            sub = dfsum[dfsum.kI == kI]
            grid = sub.pivot(index="kappa", columns="gini", values="gamma")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
            im = ax.imshow(
                grid.values, cmap="RdYlBu", interpolation="nearest", aspect="auto"
            )
            ax.set_xticks(range(grid.shape[1]))
            ax.set_xticklabels([f"{c:.2f}" for c in grid.columns])
            ax.set_yticks(range(grid.shape[0]))
            ax.set_yticklabels([f"{r:.2f}" for r in grid.index])
            ax.set_title(f"gamma heatmap (kI={kI})")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"scaling_exponent_grid_kI_{kI:.2f}.png")
            plt.close()
    except Exception:
        pass

    # markdown report
    lines = []
    lines.append(f"# Phase 13 Collective Scaling — {TIMESTAMP}\n")
    lines.append(f"Runtime: {runtime:.2f}s\n")
    for s in summary_list:
        lines.append(
            f'- kappa={s["kappa"]}, gini={s["gini"]}, kI={s["kI"]} => gamma={s["gamma"]}, mean_gain={s["mean_collective_gain"]}, amplification={s["amplification_flag"]}\n'
        )
    with open(REPORT_DIR / "phase13_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle
    bundle_path = BUNDLE_DIR / f"phase13_{TIMESTAMP}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in [DATA_DIR, FIG_DIR, REPORT_DIR]:
            for fp in d.rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(ROOT.parent))
        sha_lines = []
        for d in [DATA_DIR, FIG_DIR, REPORT_DIR]:
            for fp in d.rglob("*"):
                if fp.is_file():
                    with open(fp, "rb") as fh:
                        sha_lines.append(
                            f"{hashlib.sha256(fh.read()).hexdigest()}  {fp.relative_to(ROOT)}"
                        )
        zf.writestr("SHA256SUMS.txt", "\n".join(sha_lines))

    return summary


def main():
    start = time.time()
    all_results, runtime = run_grid()
    summary = analyze_and_export(all_results, runtime)
    print("\n=== Phase 13 Complete (FAST takeaways) ===")
    for s in summary["scaling_summary"]:
        print(
            f"kappa={s['kappa']},gini={s['gini']},kI={s['kI']}: gamma={s['gamma']}, mean_gain={s['mean_collective_gain']}, amplifying={s['amplification_flag']}"
        )
    print(f"Results saved to {ROOT}")


if __name__ == "__main__":
    main()
