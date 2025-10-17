#!/usr/bin/env python3
"""Phase 17: Artificial Mind Scaling — FAST BATTERY"""
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

ROOT = Path("./discovery_results/phase17")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
BUNDLE_DIR = ROOT / "bundle"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR, BUNDLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

FAST_MODE = True


class Agent:
    def __init__(self, aid, noise_base=0.08):
        self.aid = aid
        self.noise = noise_base + np.random.uniform(-0.01, 0.01)
        self.coherence = 0.55 + np.random.uniform(-0.03, 0.03)
        self.coordination = np.random.uniform(0.3, 0.6)
        self.calibration = np.random.uniform(0.35, 0.65)
        self.energy = np.random.uniform(0.5, 1.0)
        self.info = np.random.uniform(0.5, 1.0)
        self.cci_hist = []

    def cci(self):
        cf = max(0, self.coherence - 0.35) / 0.65
        cof = max(0, self.coordination - 0.2) / 0.8
        calf = max(0, self.calibration - 0.3) / 0.7
        np_pen = max(0, 1.0 - self.noise / 0.3)
        return float(
            np.clip(0.45 * cf + 0.30 * cof + 0.15 * calf + 0.10 * np_pen, 0, 1)
        )

    def step(self, openness, kE, kI, K=0.5, psi=1.0, dt=0.02):
        cci = self.cci()
        effE = kE * (0.6 + 0.6 * openness)
        effI = kI * (0.5 + 0.7 * openness)
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
        energy_boost = 0.1 * (self.energy - 0.5)
        info_boost = 0.12 * (self.info - 0.6)
        coh_target = 0.58 + 0.2 * openness
        rate = 1.6 + energy_boost + info_boost
        self.coherence += dt * rate * (
            coh_target - self.coherence
        ) + 0.006 * np.random.normal(0, psi)
        self.coordination += dt * (1.0 + 0.8 * K) * (
            0.4 + 0.2 * self.info - self.coordination
        ) + 0.006 * np.random.normal(0, psi)
        self.calibration += dt * (0.5 * self.energy) * (
            0.5 - self.calibration
        ) + 0.004 * np.random.normal(0, psi)
        resfac = 0.6 * self.energy + 0.4 * self.info
        noise_target = 0.10 - 0.07 * cci * resfac
        self.noise += dt * 1.1 * (noise_target - self.noise) + 0.002 * np.random.normal(
            0, psi
        )
        self.coherence = np.clip(self.coherence, 0.02, 0.99)
        self.coordination = np.clip(self.coordination, 0.01, 0.99)
        self.calibration = np.clip(self.calibration, 0.01, 0.99)
        self.noise = np.clip(self.noise, 0.001, 0.95)
        self.cci_hist.append(self.cci())


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
    ai_system,
    n_agents,
    epochs,
    openness,
    kE,
    kI,
    K,
    psi,
    seed,
    noise_base=0.08,
    thin_out=10,
):
    np.random.seed(seed)
    agents = [Agent(i, noise_base=noise_base) for i in range(n_agents)]
    epoch_records = []
    shock_start, shock_end = 900, 904
    dense_window = set(range(880, 921))
    for epoch in range(int(epochs)):
        for ag in agents:
            ag.step(openness, kE, kI, K=K, psi=psi)
        # inter-agent message mix controlled by K
        n_links = max(1, int(0.02 * n_agents * K * kI))
        for _ in range(n_links):
            i, j = np.random.randint(0, n_agents, 2)
            if i == j:
                continue
            msg = 0.04 * min(agents[i].info, agents[j].info)
            agents[i].calibration = np.clip(
                agents[i].calibration + 0.4 * msg, 0.01, 0.99
            )
            agents[j].calibration = np.clip(
                agents[j].calibration + 0.4 * msg, 0.01, 0.99
            )
        if shock_start <= epoch <= shock_end:
            noises = sorted(
                [(idx, ag.noise) for idx, ag in enumerate(agents)],
                key=lambda x: x[1],
                reverse=True,
            )
            topn = int(0.7 * n_agents)
            for idx, _ in noises[:topn]:
                agents[idx].noise = np.clip(
                    agents[idx].noise * (1.0 + 0.3), 0.001, 0.95
                )
                agents[idx].energy *= 0.9
                agents[idx].info *= 0.9
        mean_cci = float(np.mean([ag.cci() for ag in agents]))
        mean_hazard = float(np.mean([ag.noise for ag in agents]))
        if epoch in dense_window or epoch % thin_out == 0:
            Csys = system_coherence(agents, window=20)
            epoch_records.append(
                {
                    "epoch": epoch,
                    "system_cci": mean_cci,
                    "system_hazard": mean_hazard,
                    "coherence": Csys,
                    "openness": openness,
                    "kE": kE,
                    "kI": kI,
                    "K": K,
                    "psi": psi,
                    "ai_system": ai_system,
                }
            )
        # hygiene/early stop
        if epoch > 300 and mean_hazard > 0.55 and mean_cci < 0.45:
            break
    df = pd.DataFrame(epoch_records)
    if len(df) > 0:
        final = df[df.epoch >= (df.epoch.max() - 199)] if df.epoch.max() >= 199 else df
        stability_cci_mean = float(final.system_cci.mean())
        stability_hazard_mean = float(final.system_hazard.mean())
        slope = (
            float(
                np.polyfit(
                    final.epoch.values - final.epoch.values[0],
                    final.system_cci.values,
                    1,
                )[0]
            )
            if len(final) > 1
            else 0.0
        )
        auh = 0.0
        sw = df[(df.epoch >= 890) & (df.epoch <= 910)]
        if len(sw) > 1:
            auh = float(np.trapz(sw.system_hazard.values, sw.epoch.values))
        R_mean = float((final.system_cci / (final.system_hazard + 1e-12)).mean())
    else:
        stability_cci_mean = 0.0
        stability_hazard_mean = 1.0
        slope = 0.0
        auh = np.nan
        R_mean = 0.0
    return {
        "trajectories": df,
        "metrics": {
            "stability_cci_mean": stability_cci_mean,
            "stability_hazard_mean": stability_hazard_mean,
            "stability_cci_slope": slope,
            "auh_890_910": auh,
            "R_mean": R_mean,
        },
        "parameters": {
            "ai_system": ai_system,
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "kE": kE,
            "kI": kI,
            "K": K,
            "psi": psi,
            "seed": seed,
        },
    }


def run_experiment():
    print("🤖 Phase 17: Artificial Mind Scaling — FAST BATTERY")
    if FAST_MODE:
        ai_systems = ["tiny", "base", "large"]
        n_agents = 160
        seeds = [1]
        epochs_cap = 1000
        epsilons = [0.006, 0.008]
        kEs = [0.8, 1.0, 1.2]
        kIs = [0.8, 1.0, 1.2, 1.35]
        Ks = [0.5]
    else:
        ai_systems = ["tiny", "base", "large", "ensemble"]
        n_agents = 160
        seeds = [1]
        epochs_cap = 1800
        epsilons = [0.006, 0.008]
        kEs = [0.8, 1.0, 1.2]
        kIs = [0.8, 1.0, 1.2, 1.35]
        Ks = [0.3, 0.5, 0.7]
    psi_list = [1.0]
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 10
    for ai in ai_systems:
        for eps in list(epsilons):
            for kE in kEs:
                for kI in kIs:
                    for K in Ks:
                        for seed in seeds:
                            cid += 1
                            res = run_condition(
                                ai,
                                n_agents,
                                epochs_cap,
                                eps,
                                kE,
                                kI,
                                K,
                                psi=1.0,
                                seed=seed,
                                noise_base=0.08,
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
                                print("   ⚠️ runtime >75s: reducing epochs->1400")
                                epochs_cap = 1400
                            if elapsed > 95:
                                print(
                                    "   ⚠️ runtime >95s: thinning logs outside dense window"
                                )
                                thin_out = 15
                            if elapsed > 110 and "ensemble" in ai_systems:
                                print(
                                    '   ⚠️ runtime >110s: dropping ai_system="ensemble" for remaining runs'
                                )
                                ai_systems = [a for a in ai_systems if a != "ensemble"]
    runtime = time.time() - start
    print(f"✅ Phase 17 runs complete: {runtime:.2f}s (auto-cuts may have applied)")
    return all_results, runtime


def analyze_export(all_results, runtime):
    print("📊 Analyzing Phase 17 results...")
    trajs = []
    runs = []
    conds = []
    for r in all_results:
        df = r["trajectories"].copy()
        df["condition_id"] = r.get("condition_id", None)
        trajs.append(df)
        p = r["parameters"]
        m = r["metrics"]
        runs.append(
            {
                "condition_id": int(r.get("condition_id", np.nan)),
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in p.items()
                },
                **{
                    k: (float(v) if isinstance(v, (np.number,)) else v)
                    for k, v in m.items()
                },
            }
        )
        conds.append(
            {
                "condition_id": int(r.get("condition_id", np.nan)),
                "ai_system": p["ai_system"],
                "openness": float(p["openness"]),
                "kE": float(p["kE"]),
                "kI": float(p["kI"]),
                "K": float(p["K"]),
            }
        )
    traj_df = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    runs_df = pd.DataFrame(runs)
    conds_df = pd.DataFrame(conds)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA_DIR / "trajectories_long.csv", index=False)
    conds_df.to_csv(DATA_DIR / "conditions_lookup.csv", index=False)

    # exponent fit per (ai_system, eps) using rows where K==0.5 (or nearest)
    summary_list = []
    for (ai, eps), group in runs_df.groupby(["ai_system", "openness"]):
        sub = group[group.K == 0.5] if "K" in group.columns else group
        sub = sub.dropna(subset=["R_mean"])
        if len(sub) < 6:
            continue
        X = np.column_stack(
            [
                np.log(sub.kE.values + 1e-12),
                np.log(sub.kI.values + 1e-12),
                np.log(sub.K.values + 1e-12),
                np.ones(len(sub)),
            ]
        )
        y = np.log(sub.R_mean.values + 1e-12)
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta, delta, intercept = coeffs
            beta_over_alpha = float(beta / alpha) if alpha != 0 else np.nan
        except Exception:
            alpha = beta = delta = intercept = np.nan
            beta_over_alpha = np.nan

        # curvature_I at kE=1.0,K=0.5: need CCI at kI=1.0,1.2,1.35
        def cci_at(kI_val):
            sel = group[
                (np.isclose(group.kE, 1.0))
                & (np.isclose(group.K, 0.5))
                & (np.isclose(group.kI, kI_val))
            ]
            return float(sel.R_mean.mean()) if len(sel) > 0 else np.nan

        c1 = cci_at(1.0)
        c2 = cci_at(1.2)
        c3 = cci_at(1.35)
        curvature_I = np.nan
        if not np.isnan(c1) and not np.isnan(c2) and not np.isnan(c3):
            curvature_I = c3 - 2 * c2 + c1
        # flags
        proto_flag = (
            (1.5 <= beta_over_alpha < 3) if not np.isnan(beta_over_alpha) else False
        )
        aware_flag = (
            (3 <= beta_over_alpha < 6) if not np.isnan(beta_over_alpha) else False
        )
        collective_flag = False
        if not np.isnan(beta_over_alpha) and beta_over_alpha >= 6:
            collective_flag = True
        if not collective_flag and (curvature_I > 0 and delta > 0):
            collective_flag = True
        summary_list.append(
            {
                "ai_system": ai,
                "openness": float(eps),
                "alpha": float(alpha),
                "beta": float(beta),
                "delta": float(delta),
                "beta_over_alpha": beta_over_alpha,
                "curvature_I": curvature_I,
                "proto_flag": bool(proto_flag),
                "aware_flag": bool(aware_flag),
                "collective_flag": bool(collective_flag),
            }
        )

    summary = {
        "experiment": "Phase17 AI Scaling",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "per_system": summary_list,
    }
    with open(DATA_DIR / "phase17_ai_scaling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures
    try:
        dfsum = pd.DataFrame(summary_list)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        if not dfsum.empty:
            ax.bar(dfsum.ai_system, dfsum.beta_over_alpha)
            # bio baselines lines (from Phase 16 fast run): proto~2.31, plant~8.95, animal~8.54
            ax.axhline(2.31, color="C1", linestyle="--", label="bio:proto")
            ax.axhline(8.95, color="C2", linestyle=":", label="bio:plant")
            ax.axhline(8.54, color="C3", linestyle="-.", label="bio:animal")
        ax.set_ylabel("β/α")
        ax.set_title("β/α by AI system")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "ai_beta_over_alpha.png")
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        if not dfsum.empty:
            ax.bar(dfsum.ai_system, dfsum.curvature_I)
        ax.set_title("Info curvature by AI system")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "ai_info_curvature.png")
        plt.close()
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (ai, kE, kI), g in traj_df.groupby(["ai_system", "kE", "kI"]):
                ax.plot(g.epoch, g.system_cci, label=f"{ai},kE={kE},kI={kI}", alpha=0.5)
            ax.axvspan(900, 904, alpha=0.1, color="red")
            ax.set_title("CCI (phase17)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("CCI")
            ax.legend(fontsize="x-small")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "cci_long.png")
            plt.close()
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (ai, kE, kI), g in traj_df.groupby(["ai_system", "kE", "kI"]):
                ax.plot(
                    g.epoch, g.system_hazard, label=f"{ai},kE={kE},kI={kI}", alpha=0.5
                )
            ax.set_title("Hazard (phase17)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("hazard")
            ax.legend(fontsize="x-small")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "hazard_long.png")
            plt.close()
    except Exception:
        pass

    # report
    lines = [f"# Phase 17 AI Scaling — {TIMESTAMP}\n", f"Runtime: {runtime:.2f}s\n"]
    for s in summary_list:
        lines.append(
            f"- {s['ai_system']}, eps={s['openness']}: beta/alpha={s['beta_over_alpha']}, curvature_I={s['curvature_I']}, collective_flag={s['collective_flag']}\n"
        )
    with open(REPORT_DIR / "phase17_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle
    bundle_path = BUNDLE_DIR / f"phase17_{TIMESTAMP}.zip"
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
    print("\n=== Phase 17 Complete (fast takeaways) ===")
    for p in summary["per_system"]:
        print(
            f"{p['ai_system']}: beta/alpha={p['beta_over_alpha']}, collective_flag={p['collective_flag']}"
        )
    print(f"Results saved to {ROOT}")


if __name__ == "__main__":
    main()
