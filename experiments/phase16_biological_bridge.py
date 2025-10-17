#!/usr/bin/env python3
"""
Phase 16: Biological Bridge — FAST BATTERY

Maps metabolic and neural parameters to resilience index R and fits bio-scaling exponents.
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

ROOT = Path("./discovery_results/phase16")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
BUNDLE_DIR = ROOT / "bundle"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR, BUNDLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# FAST mode: reduce grid and epochs to meet ~2 minute runtime for quick experiments.
FAST_MODE = True


def eta_from_phase15(eps):
    # provided analytic form (can be negative per Phase 15 fit)
    return -5e-6 * (eps**-1.3)


# reuse a lightweight agent model similar to Phase15
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

    def step(self, openness, kE, kI, Ksyn=0.5, psi=1.0, dt=0.02):
        cci = self.cci()
        effE = kE * (0.8 + 0.4 * openness)
        effI = kI * (0.6 + 0.6 * openness)
        # metabolic effects scale energy update; synaptic coupling affects coordination
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
        ) + 0.006 * np.random.normal(0, psi)
        # synaptic coupling increases coordination dynamics
        self.coordination += dt * (1.2 + 0.8 * Ksyn) * (
            0.45 + 0.15 * self.info - self.coordination
        ) + 0.006 * np.random.normal(0, psi)
        self.calibration += dt * (0.6 * self.energy) * (
            0.55 - self.calibration
        ) + 0.005 * np.random.normal(0, psi)
        resfac = 0.6 * self.energy + 0.4 * self.info
        noise_target = 0.12 - 0.08 * cci * resfac
        self.noise += dt * 1.2 * (noise_target - self.noise) + 0.002 * np.random.normal(
            0, psi
        )
        self.coherence = np.clip(self.coherence, 0.05, 0.99)
        self.coordination = np.clip(self.coordination, 0.05, 0.99)
        self.calibration = np.clip(self.calibration, 0.05, 0.99)
        self.noise = np.clip(self.noise, 0.001, 0.9)
        self.cci_hist.append(self.cci())


def system_entropy(agents, n_bins=20):
    infos = np.array([ag.info for ag in agents])
    hist, _ = np.histogram(infos, bins=n_bins, range=(0.0, 1.0), density=True)
    p = hist + 1e-12
    p = p / p.sum()
    S = -(p * np.log(p)).sum()
    return float(S)


def system_coherence(agents, window=20):
    mats = []
    for ag in agents:
        h = (
            np.array(ag.cci_hist[-window:])
            if len(ag.cci_hist) > 0
            else np.array([ag.cci()])
        )
        if len(h) < window:
            h = np.pad(h, (window - len(h), 0), "edge")
        mats.append(h)
    M = np.vstack(mats)
    try:
        C = np.corrcoef(M)
        n = C.shape[0]
        mean_corr = (C.sum() - np.trace(C)) / (n * (n - 1))
    except Exception:
        mean_corr = 0.0
    return float(mean_corr)


def run_condition(
    bio_system,
    n_agents,
    epochs,
    openness,
    E_met,
    I_neu,
    Ksyn,
    psi,
    seed,
    noise_base=0.08,
    thin_out=10,
):
    np.random.seed(seed)
    agents = [Agent(i, noise_base=noise_base) for i in range(n_agents)]
    epoch_records = []
    shock_events = []
    dense_window = set(range(880, 921))
    shock_start, shock_end = 900, 904
    for epoch in range(int(epochs)):
        for ag in agents:
            ag.step(openness, E_met, I_neu, Ksyn=Ksyn, psi=psi)
        # simple messaging
        if I_neu > 0:
            n_links = int(0.02 * n_agents * I_neu)
            for _ in range(max(1, n_links)):
                i, j = np.random.randint(0, n_agents, 2)
                if i == j:
                    continue
                mstr = 0.05 * min(agents[i].info, agents[j].info)
                agents[i].calibration = np.clip(
                    agents[i].calibration + 0.5 * mstr, 0.05, 0.99
                )
                agents[j].calibration = np.clip(
                    agents[j].calibration + 0.5 * mstr, 0.05, 0.99
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
                    "E_met": E_met,
                    "I_neu": I_neu,
                    "Ksyn": Ksyn,
                    "psi": psi,
                    "bio_system": bio_system,
                }
            )
        # hygiene
        if mean_hazard > 0.55 and mean_cci < 0.45 and epoch > 300:
            break
    df = pd.DataFrame(epoch_records)
    # metrics on final window
    if len(df) > 0:
        final_win = (
            df[df.epoch >= (df.epoch.max() - 199)] if df.epoch.max() >= 199 else df
        )
        stability_cci_mean = float(final_win.system_cci.mean())
        stability_hazard_mean = float(final_win.system_hazard.mean())
        slope = (
            float(
                np.polyfit(
                    final_win.epoch.values - final_win.epoch.values[0],
                    final_win.system_cci.values,
                    1,
                )[0]
            )
            if len(final_win) > 1
            else 0.0
        )
        R_mean = float(
            (final_win.system_cci / (final_win.system_hazard + 1e-12)).mean()
        )
        auh = 0.0
        sw = df[(df.epoch >= 890) & (df.epoch <= 910)]
        if len(sw) > 1:
            auh = float(np.trapz(sw.system_hazard.values, sw.epoch.values))
    else:
        stability_cci_mean = 0.0
        stability_hazard_mean = 1.0
        slope = 0.0
        R_mean = 0.0
        auh = np.nan
    return {
        "trajectories": df,
        "shock_events": pd.DataFrame(shock_events),
        "metrics": {
            "stability_cci_mean": stability_cci_mean,
            "stability_hazard_mean": stability_hazard_mean,
            "stability_cci_slope": slope,
            "R_mean": R_mean,
            "auh_890_910": auh,
        },
        "parameters": {
            "bio_system": bio_system,
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "E_met": E_met,
            "I_neu": I_neu,
            "Ksyn": Ksyn,
            "psi": psi,
            "seed": seed,
        },
    }


def run_experiment():
    print("⚛️ Phase 16: Biological Bridge — FAST BATTERY")
    if FAST_MODE:
        bio_systems = ["proto", "plant", "animal"]
        n_agents = 200
        seeds = [1]
        epochs_cap = 1200
        epsilons = [0.005, 0.006]
        E_mets = [0.9, 1.1]
        I_neus = [0.9, 1.1]
        Ksyns = [0.4, 0.6]
    else:
        bio_systems = ["proto", "plant", "animal", "human_sim"]
        n_agents = 200
        seeds = [1]
        epochs_cap = 2000
        epsilons = [0.005, 0.006, 0.008]
        E_mets = [0.8, 1.0, 1.2]
        I_neus = [0.8, 1.0, 1.2]
        Ksyns = [0.3, 0.5, 0.7]
    noise_base = 0.08
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 10
    for bio in bio_systems:
        for eps in list(epsilons):
            eta = eta_from_phase15(eps)
            for E in E_mets:
                for I in I_neus:
                    for K in Ksyns:
                        for seed in seeds:
                            cid += 1
                            res = run_condition(
                                bio,
                                n_agents,
                                epochs_cap,
                                eps,
                                E,
                                I,
                                K,
                                psi=1.0,
                                seed=seed,
                                noise_base=noise_base,
                                thin_out=thin_out,
                            )
                            res["condition_id"] = cid
                            res["eta"] = eta
                            all_results.append(res)
                            elapsed = time.time() - start
                            if elapsed > 45 and 0.008 in epsilons:
                                print(
                                    "   ⚠️ runtime >45s: dropping epsilon=0.008 for remaining runs"
                                )
                                epsilons = [e for e in epsilons if e != 0.008]
                            if elapsed > 75:
                                print("   ⚠️ runtime >75s: reducing epochs->1600")
                                epochs_cap = 1600
                            if elapsed > 95:
                                print(
                                    "   ⚠️ runtime >95s: thinning logs outside dense window"
                                )
                                thin_out = 15
    runtime = time.time() - start
    print(f"✅ Phase 16 runs complete: {runtime:.2f}s (auto-cuts may have applied)")
    return all_results, runtime


def analyze_export(all_results, runtime):
    print("📊 Analyzing Phase 16 results...")
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
        p = r["parameters"]
        m = r["metrics"]
        runs.append(
            {
                "condition_id": int(r["condition_id"]),
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in p.items()
                },
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in m.items()
                },
                "eta": float(r.get("eta", np.nan)),
            }
        )
        conds.append(
            {
                "condition_id": int(r["condition_id"]),
                "bio_system": p["bio_system"],
                "openness": float(p["openness"]),
                "E_met": float(p["E_met"]),
                "I_neu": float(p["I_neu"]),
                "Ksyn": float(p["Ksyn"]),
            }
        )
    traj_df = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    shock_df = pd.concat(shocks, ignore_index=True) if shocks else pd.DataFrame()
    runs_df = pd.DataFrame(runs)
    conds_df = pd.DataFrame(conds)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA_DIR / "trajectories_long.csv", index=False)
    conds_df.to_csv(DATA_DIR / "conditions_lookup.csv", index=False)
    if not shock_df.empty:
        shock_df.to_csv(DATA_DIR / "shock_events.csv", index=False)

    # compute sensitivities ΔR_E and ΔR_I: finite difference across E_met and I_neu
    sens_rows = []
    for (bio, eps), g in runs_df.groupby(["bio_system", "openness"]):
        # pivot on E_met,I_neu with same Ksyn
        for K in g.Ksyn.unique():
            sub = g[g.Ksyn == K]
            if sub.empty:
                continue

            # delta R over E (compare E=1.2 vs E=0.8, keeping I=1.0 if available)
            def getR(E, I):
                sel = sub[(np.isclose(sub.E_met, E)) & (np.isclose(sub.I_neu, I))]
                return sel.R_mean.iloc[0] if len(sel) > 0 else np.nan

            R_e_hi = getR(1.2, 1.0)
            R_e_lo = getR(0.8, 1.0)
            R_i_hi = getR(1.0, 1.2)
            R_i_lo = getR(1.0, 0.8)
            dRE = (
                (R_e_hi - R_e_lo)
                if not np.isnan(R_e_hi) and not np.isnan(R_e_lo)
                else np.nan
            )
            dRI = (
                (R_i_hi - R_i_lo)
                if not np.isnan(R_i_hi) and not np.isnan(R_i_lo)
                else np.nan
            )
            sens_rows.append(
                {
                    "bio_system": bio,
                    "openness": float(eps),
                    "Ksyn": float(K),
                    "deltaR_E": dRE,
                    "deltaR_I": dRI,
                }
            )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(DATA_DIR / "sensitivity_deltaR.csv", index=False)

    # Fit bio-scaling exponent γ_bio: log(R) ~ a*log(E)+b*log(I)+d*log(K)
    summary_list = []
    for bio, group in runs_df.groupby("bio_system"):
        grp = group.dropna(subset=["R_mean"])
        if len(grp) < 6:
            continue
        X = np.column_stack(
            [
                np.log(grp.E_met.values + 1e-12),
                np.log(grp.I_neu.values + 1e-12),
                np.log(grp.Ksyn.values + 1e-12),
                np.ones(len(grp)),
            ]
        )
        y = np.log(grp.R_mean.values + 1e-12)
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta, delta, intercept = coeffs
            beta_over_alpha = float(beta / alpha) if alpha != 0 else np.nan
        except Exception:
            alpha = beta = delta = intercept = np.nan
            beta_over_alpha = np.nan
        # average eta across eps in this bio group
        eta_avg = float(group.eta.mean()) if "eta" in group.columns else np.nan
        # winner check: pick baseline rows with psi==1.0 (we used psi=1.0)
        baseline = group
        prod_safe = False
        if len(baseline) > 0:
            row = baseline.iloc[0]
            prod_safe = (
                (row.stability_cci_mean >= 0.50)
                and (row.stability_hazard_mean <= 0.20)
                and (row.stability_cci_slope >= 0.0005)
            )
        summary_list.append(
            {
                "bio_system": bio,
                "alpha": float(alpha),
                "beta": float(beta),
                "delta": float(delta),
                "beta_over_alpha": beta_over_alpha,
                "eta_avg": eta_avg,
                "production_safe": bool(prod_safe),
            }
        )

    summary = {
        "experiment": "Phase16 Biological Bridge",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "per_bio_system": summary_list,
    }
    with open(DATA_DIR / "phase16_biobridge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures
    try:
        # ΔR_E vs ΔR_I scatter
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=140)
        if not sens_df.empty:
            ax.scatter(sens_df.deltaR_E, sens_df.deltaR_I, c="C0")
            ax.axhline(0, color="k", alpha=0.4)
            ax.axvline(0, color="k", alpha=0.4)
        ax.set_xlabel("ΔR_E")
        ax.set_ylabel("ΔR_I")
        ax.set_title("ΔR_E vs ΔR_I")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "resilience_vs_energy_info.png")
        plt.close()
        # bio scaling exponents
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        dfsum = pd.DataFrame(summary_list)
        if not dfsum.empty:
            ax.bar(dfsum.bio_system, dfsum.beta_over_alpha)
            ax.set_ylabel("β/α")
            ax.set_title("β/α by bio_system")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "bio_scaling_exponent.png")
        plt.close()
        # eta vs resilience scatter
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        if not runs_df.empty:
            pivot = (
                runs_df.groupby("bio_system")
                .agg({"eta": "mean", "R_mean": "mean"})
                .reset_index()
            )
            ax.scatter(pivot.eta, pivot.R_mean)
            for i, row in pivot.iterrows():
                ax.annotate(row.bio_system, (row.eta, row.R_mean))
        ax.set_xlabel("eta")
        ax.set_ylabel("R_mean")
        ax.set_title("eta vs Resilience")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "eta_vs_resilience.png")
        plt.close()
    except Exception:
        pass

    # report
    lines = [
        f"# Phase 16 Biological Bridge — {TIMESTAMP}\n",
        f"Runtime: {runtime:.2f}s\n",
    ]
    for s in summary_list:
        lines.append(
            f"- {s['bio_system']}: alpha={s['alpha']}, beta={s['beta']}, beta/alpha={s['beta_over_alpha']}, eta_avg={s['eta_avg']}, prod_safe={s['production_safe']}\n"
        )
    with open(REPORT_DIR / "phase16_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle
    bundle_path = BUNDLE_DIR / f"phase16_{TIMESTAMP}.zip"
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
    print("\n=== Phase 16 Complete (fast takeaways) ===")
    for p in summary["per_bio_system"]:
        print(
            f"{p['bio_system']}: beta/alpha={p['beta_over_alpha']}, eta_avg={p['eta_avg']}, prod_safe={p['production_safe']}"
        )
    print(f"Results saved to {ROOT}")


if __name__ == "__main__":
    main()
