#!/usr/bin/env python3
"""
Phase 15: Entropy–Coherence Coupling Test — FAST BATTERY
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

ROOT = Path("./discovery_results/phase15")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
BUNDLE_DIR = ROOT / "bundle"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR, BUNDLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Agent:
    def __init__(self, aid, noise_base=0.08):
        self.aid = aid
        self.noise = noise_base + np.random.uniform(-0.01, 0.01)
        self.coherence = 0.6 + np.random.uniform(-0.02, 0.02)
        self.coordination = np.random.uniform(0.3, 0.6)
        self.calibration = np.random.uniform(0.35, 0.65)
        self.energy = np.random.uniform(0.5, 1.0)
        self.info = np.random.uniform(0.5, 1.0)
        self.cci_hist = []

    def cci(self):
        cf = max(0, self.coherence - 0.4) / 0.6
        cof = max(0, self.coordination - 0.2) / 0.8
        calf = max(0, self.calibration - 0.3) / 0.7
        np_pen = max(0, 1.0 - self.noise / 0.3)
        return float(np.clip(0.4 * cf + 0.3 * cof + 0.2 * calf + 0.1 * np_pen, 0, 1))

    def step(self, openness, kE, kI, psi=1.0, dt=0.02):
        # psi scales random perturbations (entropy flux)
        cci = self.cci()
        effE = kE * (0.8 + 0.4 * openness)
        effI = kI * (0.6 + 0.6 * openness)
        # random perturbation proportional to psi
        self.energy = np.clip(
            self.energy
            + dt * effE * (1.0 - self.energy)
            + 0.01 * np.random.normal(0, psi),
            0.05,
            1.0,
        )
        self.info = np.clip(
            self.info + dt * effI * (0.9 - self.info) + 0.01 * np.random.normal(0, psi),
            0.05,
            1.0,
        )
        energy_boost = 0.12 * (self.energy - 0.5)
        info_boost = 0.10 * (self.info - 0.6)
        coh_target = 0.60 + 0.18 * openness
        rate = 1.8 + energy_boost + info_boost
        self.coherence += dt * rate * (
            coh_target - self.coherence
        ) + 0.005 * np.random.normal(0, psi)
        self.coordination += dt * 1.2 * (
            0.45 + 0.15 * self.info - self.coordination
        ) + 0.005 * np.random.normal(0, psi)
        self.calibration += dt * (0.6 * self.energy) * (
            0.55 - self.calibration
        ) + 0.005 * np.random.normal(0, psi)
        resfac = 0.6 * self.energy + 0.4 * self.info
        noise_target = 0.12 - 0.08 * cci * resfac
        self.noise += dt * 1.2 * (noise_target - self.noise) + 0.002 * np.random.normal(
            0, psi
        )
        self.coherence = np.clip(self.coherence, 0.1, 0.99)
        self.coordination = np.clip(self.coordination, 0.05, 0.95)
        self.calibration = np.clip(self.calibration, 0.05, 0.95)
        self.noise = np.clip(self.noise, 0.001, 0.9)
        self.cci_hist.append(self.cci())


def system_entropy(agents, n_bins=20):
    # Shannon entropy over 'info' distribution
    infos = np.array([ag.info for ag in agents])
    hist, edges = np.histogram(infos, bins=n_bins, range=(0.0, 1.0), density=True)
    # avoid zeros
    p = hist + 1e-12
    p = p / p.sum()
    S = -(p * np.log(p)).sum()
    return float(S)


def system_coherence(agents, window=20):
    # mean pairwise correlation across agents' recent cci histories
    mats = []
    for ag in agents:
        h = (
            np.array(ag.cci_hist[-window:])
            if len(ag.cci_hist) > 0
            else np.array([ag.cci()])
        )
        if len(h) < window:
            # pad with last value
            h = np.pad(h, (window - len(h), 0), "edge")
        mats.append(h)
    M = np.vstack(mats)  # shape (n_agents, window)
    # compute correlation matrix across agents
    try:
        C = np.corrcoef(M)
        # mean off-diagonal
        n = C.shape[0]
        mean_corr = (C.sum() - np.trace(C)) / (n * (n - 1))
    except Exception:
        mean_corr = 0.0
    return float(mean_corr)


def run_condition(
    n_agents, epochs, openness, kE, kI, psi, seed, noise_base=0.08, thin_out=10
):
    np.random.seed(seed)
    agents = [Agent(i, noise_base=noise_base) for i in range(n_agents)]
    epoch_records = []
    shock_events = []
    dense_window = set(range(880, 921))
    shock_start, shock_end = 900, 904
    for epoch in range(int(epochs)):
        for ag in agents:
            ag.step(openness, kE, kI, psi=psi)
        # messaging scaled by kI
        if kI > 0:
            n_links = int(0.02 * n_agents * kI)
            for _ in range(max(1, n_links)):
                i, j = np.random.randint(0, n_agents, 2)
                if i == j:
                    continue
                mstr = 0.06 * min(agents[i].info, agents[j].info)
                agents[i].calibration = np.clip(
                    agents[i].calibration + 0.5 * mstr, 0.05, 0.95
                )
                agents[j].calibration = np.clip(
                    agents[j].calibration + 0.5 * mstr, 0.05, 0.95
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
                agents[idx].noise = np.clip(agents[idx].noise * (1.0 + 0.3), 0.001, 0.9)
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
        if mean_hazard > 0.45:
            noises = sorted(
                [(i, ag.noise) for i, ag in enumerate(agents)],
                key=lambda x: x[1],
                reverse=True,
            )
            ntrim = int(0.25 * n_agents)
            for idx, _ in noises[:ntrim]:
                agents[idx].noise *= 0.8
        # metrics
        mean_cci = float(np.mean([ag.cci() for ag in agents]))
        mean_hazard = float(np.mean([ag.noise for ag in agents]))
        if epoch in dense_window or epoch % thin_out == 0:
            S = system_entropy(agents)
            C = system_coherence(agents, window=20)
            epoch_records.append(
                {
                    "epoch": epoch,
                    "system_cci": mean_cci,
                    "system_hazard": mean_hazard,
                    "entropy": S,
                    "coherence": C,
                    "openness": openness,
                    "kE": kE,
                    "kI": kI,
                    "psi": psi,
                }
            )
        # early stop
        if epoch > 300 and mean_hazard > 0.55 and mean_cci < 0.45:
            break
    df = pd.DataFrame(epoch_records)
    shock_df = pd.DataFrame(shock_events)
    # final metrics
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
        S_bar = (
            float(final_win.entropy.mean())
            if "entropy" in final_win.columns
            else np.nan
        )
        C_bar = (
            float(final_win.coherence.mean())
            if "coherence" in final_win.columns
            else np.nan
        )
        auh = 0.0
        sw = df[(df.epoch >= 890) & (df.epoch <= 910)]
        if len(sw) > 1:
            auh = float(np.trapz(sw.system_hazard.values, sw.epoch.values))
        # CCI recover t50
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
    else:
        stability_cci_mean = 0.0
        stability_hazard_mean = 1.0
        slope = 0.0
        S_bar = np.nan
        C_bar = np.nan
        auh = np.nan
        t50 = np.nan
    return {
        "trajectories": df,
        "shock_events": shock_df,
        "metrics": {
            "stability_cci_mean": stability_cci_mean,
            "stability_hazard_mean": stability_hazard_mean,
            "stability_cci_slope": slope,
            "entropy_mean": S_bar,
            "coherence_mean": C_bar,
            "auh_890_910": auh,
            "CCI_recover_t50": t50,
        },
        "parameters": {
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "kE": kE,
            "kI": kI,
            "psi": psi,
            "seed": seed,
        },
    }


def run_experiment():
    print("⚡ Phase 15: Entropy–Coherence Coupling — FAST BATTERY")
    n_agents = 200
    seeds = [1]
    epochs_cap = 2000
    epsilons = [0.005, 0.006, 0.008]
    kE = 1.0
    kIs = [1.0, 1.2]
    psis = [0.8, 1.0, 1.2]
    noise_base = 0.08
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 10
    for eps in list(epsilons):
        for kI in kIs:
            for psi in psis:
                for seed in seeds:
                    cid += 1
                    res = run_condition(
                        n_agents,
                        epochs_cap,
                        eps,
                        kE,
                        kI,
                        psi,
                        seed,
                        noise_base=noise_base,
                        thin_out=thin_out,
                    )
                    res["condition_id"] = cid
                    all_results.append(res)
                    elapsed = time.time() - start
                    if elapsed > 45 and 0.008 in epsilons:
                        print(
                            "   ⚠️ runtime >45s: dropping epsilon=0.008 for remaining runs"
                        )
                        epsilons = [e for e in epsilons if e != 0.008]
                    if elapsed > 75:
                        print("   ⚠️ runtime >75s: reducing epochs->1500")
                        epochs_cap = 1500
                    if elapsed > 95:
                        print("   ⚠️ runtime >95s: thinning logs outside dense window")
                        thin_out = 15
    runtime = time.time() - start
    print(f"✅ Phase 15 runs complete: {runtime:.2f}s (auto-cuts may have applied)")
    return all_results, runtime


def analyze_export(all_results, runtime):
    print("📊 Analyzing Phase 15 results...")
    trajs = []
    shocks = []
    runs = []
    conds = []
    entropy_rows = []
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
                "kI": float(params["kI"]),
                "psi": float(params["psi"]),
            }
        )
        # per-epoch entropy/coherence rows
        if not df.empty and "entropy" in df.columns:
            for _, row in df.iterrows():
                entropy_rows.append(
                    {
                        "condition_id": int(r["condition_id"]),
                        "epoch": int(row.epoch),
                        "entropy": float(row.entropy),
                        "coherence": float(row.coherence),
                        "openness": float(row.openness),
                        "kI": float(row.kI),
                        "psi": float(row.psi),
                    }
                )
    traj_df = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    shock_df = pd.concat(shocks, ignore_index=True) if shocks else pd.DataFrame()
    runs_df = pd.DataFrame(runs)
    conds_df = pd.DataFrame(conds)
    entropy_df = pd.DataFrame(entropy_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA_DIR / "trajectories_long.csv", index=False)
    if not shock_df.empty:
        shock_df.to_csv(DATA_DIR / "shock_events.csv", index=False)
    conds_df.to_csv(DATA_DIR / "conditions_lookup.csv", index=False)
    entropy_df.to_csv(DATA_DIR / "entropy_coherence.csv", index=False)

    # compute per (eps,kI) the S_bar and C_bar for each psi and estimate eta via linear fit
    summary_list = []
    for (eps, kI), group in runs_df.groupby(["openness", "kI"]):
        # collect S_bar and C_bar per psi
        rows = (
            group.groupby("psi")
            .agg({"entropy_mean": "mean", "coherence_mean": "mean"})
            .reset_index()
        )
        if len(rows) < 2:
            continue
        S = rows["entropy_mean"].values
        C = rows["coherence_mean"].values
        # fit linear C = a + b*S => eta = -b
        A = np.column_stack([S, np.ones(len(S))])
        try:
            b, a = np.linalg.lstsq(A, C, rcond=None)[0]
            eta = -b
            # compute stderr and 95% CI for slope b
            ypred = A.dot([b, a])
            resid = C - ypred
            dof = max(1, len(S) - 2)
            s2 = (resid**2).sum() / dof
            cov_bb = s2 * np.linalg.inv(A.T.dot(A))[0, 0]
            stderr_b = np.sqrt(cov_bb)
            # 95% CI approx
            ci95 = 1.96 * stderr_b
        except Exception:
            eta = np.nan
            ci95 = np.nan
        # compute deltas relative to psi=1.0
        base = rows[rows.psi == 1.0]
        if len(base) > 0:
            S0 = float(base["entropy_mean"].iloc[0])
            C0 = float(base["coherence_mean"].iloc[0])
        else:
            S0 = np.nan
            C0 = np.nan
        deltas = []
        for _, r in rows.iterrows():
            deltas.append(
                {
                    "psi": float(r.psi),
                    "S_bar": float(r.entropy_mean),
                    "C_bar": float(r.coherence_mean),
                    "deltaS": float(r.entropy_mean - S0) if not np.isnan(S0) else None,
                    "deltaC": (
                        float(r.coherence_mean - C0) if not np.isnan(C0) else None
                    ),
                }
            )
        # correlation r between S and C
        try:
            rcoeff = float(np.corrcoef(S, C)[0, 1])
        except Exception:
            rcoeff = np.nan
        # winner production-safe check: pick condition with psi==1.0
        winner_rows = group[group["psi"] == 1.0]
        prod_safe = False
        if len(winner_rows) > 0:
            row = winner_rows.iloc[0]
            prod_safe = (
                (row.stability_cci_mean >= 0.50)
                and (row.stability_hazard_mean <= 0.20)
                and (row.stability_cci_slope >= 0.0005)
            )
        summary_list.append(
            {
                "openness": float(eps),
                "kI": float(kI),
                "eta": eta,
                "eta_ci95": ci95,
                "r": rcoeff,
                "deltas": deltas,
                "production_safe_at_psi1": bool(prod_safe),
            }
        )

    summary = {
        "experiment": "Phase15 Entropy-Coherence",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "per_condition": summary_list,
    }
    with open(DATA_DIR / "phase15_entropy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)

    # Figures
    try:
        # entropy vs coherence scatter with fitted lines per (eps,kI)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=140)
        for (eps, kI), group in entropy_df.groupby(["openness", "kI"]):
            xs = group.entropy.values
            ys = group.coherence.values
            ax.scatter(xs, ys, label=f"e={eps},kI={kI}", alpha=0.6)
            # fit line
            if len(xs) > 1:
                coef = np.polyfit(xs, ys, 1)
                xgrid = np.linspace(min(xs), max(xs), 50)
                ax.plot(xgrid, np.polyval(coef, xgrid), alpha=0.6)
        ax.set_xlabel("Entropy S")
        ax.set_ylabel("Coherence C")
        ax.set_title("Entropy vs Coherence")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "entropy_vs_coherence.png")
        plt.close()
        # eta by epsilon
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        dfsum = pd.DataFrame(summary_list)
        if not dfsum.empty:
            for kI in dfsum.kI.unique():
                sub = dfsum[dfsum.kI == kI]
                ax.errorbar(
                    sub.openness,
                    sub.eta,
                    yerr=sub.eta_ci95,
                    marker="o",
                    label=f"kI={kI}",
                )
        ax.set_xlabel("epsilon")
        ax.set_ylabel("eta")
        ax.set_title("eta by epsilon")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "eta_by_epsilon.png")
        plt.close()
        # cci and hazard long
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (eps, kI, psi), g in traj_df.groupby(["openness", "kI", "psi"]):
                ax.plot(
                    g.epoch, g.system_cci, label=f"e={eps},kI={kI},psi={psi}", alpha=0.6
                )
            ax.axvspan(900, 904, alpha=0.1, color="red")
            ax.set_title("CCI (phase15)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("CCI")
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "cci_long.png")
            plt.close()
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (eps, kI, psi), g in traj_df.groupby(["openness", "kI", "psi"]):
                ax.plot(
                    g.epoch,
                    g.system_hazard,
                    label=f"e={eps},kI={kI},psi={psi}",
                    alpha=0.6,
                )
            ax.set_title("Hazard (phase15)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("hazard")
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "hazard_long.png")
            plt.close()
    except Exception:
        pass

    # report
    lines = [
        f"# Phase 15 Entropy–Coherence — {TIMESTAMP}\n",
        f"Runtime: {runtime:.2f}s\n",
    ]
    for s in summary_list:
        lines.append(
            f"- eps={s['openness']}, kI={s['kI']}: eta={s['eta']}, eta_ci95={s['eta_ci95']}, r={s['r']}, prod_safe_at_psi1={s['production_safe_at_psi1']}\n"
        )
    with open(REPORT_DIR / "phase15_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle
    bundle_path = BUNDLE_DIR / f"phase15_{TIMESTAMP}.zip"
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
    all_results, runtime = run_experiment()
    summary = analyze_export(all_results, runtime)
    print("\n=== Phase 15 Complete (fast takeaways) ===")
    for p in summary["per_condition"]:
        print(
            f"eps={p['openness']}, kI={p['kI']}: eta={p['eta']}, r={p['r']}, prod_safe={p['production_safe_at_psi1']}"
        )
    print(f"Results saved to {ROOT}")


if __name__ == "__main__":
    main()
