#!/usr/bin/env python3
"""Phase 18: Universal Unification Test — FAST BATTERY"""
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

ROOT = Path("./discovery_results/phase18")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
BUNDLE_DIR = ROOT / "bundle"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR, BUNDLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

FAST_MODE = True


def eta_from_phase15(eps):
    return -5e-6 * (eps**-1.3)


class Patch:
    def __init__(self, pid, noise_base=0.08):
        self.pid = pid
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

    def step(self, openness, kE, kI, lam=0.8, psi=1.0, dt=0.02):
        effE = kE * (0.8 + 0.4 * openness)
        effI = kI * (0.6 + 0.6 * openness)
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
        coh_target = 0.60 + 0.12 * openness
        rate = 1.6 + energy_boost + info_boost
        self.coherence += dt * rate * (
            coh_target - self.coherence
        ) + 0.005 * np.random.normal(0, psi)
        self.coordination += dt * (1.1) * (
            0.45 + 0.15 * self.info - self.coordination
        ) + 0.004 * np.random.normal(0, psi)
        self.calibration += dt * (0.6 * self.energy) * (
            0.55 - self.calibration
        ) + 0.004 * np.random.normal(0, psi)
        # structure retention: lam reduces loss
        self.noise += dt * 1.2 * (0.12 - self.noise) * (
            1.0 - 0.2 * lam
        ) + 0.002 * np.random.normal(0, psi)
        self.coherence = np.clip(self.coherence, 0.05, 0.99)
        self.noise = np.clip(self.noise, 0.001, 0.95)
        self.cci_hist.append(self.cci())


def system_coherence(patches, window=20):
    mats = []
    for p in patches:
        h = (
            np.array(p.cci_hist[-window:])
            if len(p.cci_hist) > 0
            else np.array([p.cci()])
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
    cosmic_regime,
    n_agents,
    epochs,
    openness,
    kE,
    kI,
    lam,
    eta,
    seed,
    noise_base=0.08,
    thin_out=15,
):
    np.random.seed(seed)
    patches = [Patch(i, noise_base=noise_base) for i in range(n_agents)]
    epoch_records = []
    shock_events = []
    dense_window = set(range(1180, 1221))
    shock_start, shock_end = 1200, 1204
    for epoch in range(int(epochs)):
        for p in patches:
            p.step(openness, kE, kI, lam=lam, psi=abs(eta))
        if shock_start <= epoch <= shock_end:
            noises = sorted(
                [(idx, p.noise) for idx, p in enumerate(patches)],
                key=lambda x: x[1],
                reverse=True,
            )
            topn = int(0.7 * n_agents)
            pre_mean = np.mean([p.noise for p in patches])
            for idx, _ in noises[:topn]:
                patches[idx].noise = np.clip(
                    patches[idx].noise * (1.0 + 0.3), 0.001, 0.95
                )
            post_mean = np.mean([p.noise for p in patches])
            shock_events.append(
                {
                    "epoch": epoch,
                    "affected_pct": topn / n_agents * 100,
                    "noise_delta": post_mean - pre_mean,
                }
            )
        mean_cci = float(np.mean([p.cci() for p in patches]))
        mean_hazard = float(np.mean([p.noise for p in patches]))
        if epoch in dense_window or epoch % thin_out == 0:
            Csys = system_coherence(patches, window=20)
            epoch_records.append(
                {
                    "epoch": epoch,
                    "system_cci": mean_cci,
                    "system_hazard": mean_hazard,
                    "coherence": Csys,
                    "openness": openness,
                    "kE": kE,
                    "kI": kI,
                    "lam": lam,
                    "eta": eta,
                    "cosmic_regime": cosmic_regime,
                }
            )
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
        R_mean = float((final.system_cci / (final.system_hazard + 1e-12)).mean())
        sw = df[(df.epoch >= 1180) & (df.epoch <= 1220)]
        auh = (
            float(np.trapz(sw.system_hazard.values, sw.epoch.values))
            if len(sw) > 1
            else 0.0
        )
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
            "auh_1180_1220": auh,
        },
        "parameters": {
            "cosmic_regime": cosmic_regime,
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "kE": kE,
            "kI": kI,
            "lam": lam,
            "eta": eta,
            "seed": seed,
        },
    }


def run_experiment():
    print("🌌 Phase 18: Universal Unification Test — FAST BATTERY")
    if FAST_MODE:
        cosmic_regimes = ["present", "future"]
        n_agents = 180
        seeds = [1]
        epochs_cap = 1400
        epsilons = [0.003, 0.005, 0.007]
        kEs = [0.9, 1.0, 1.1]
        kIs = [0.9, 1.0, 1.1]
        lambdas = [0.7, 0.9]
    else:
        cosmic_regimes = ["early_universe", "present", "future"]
        n_agents = 180
        seeds = [1]
        epochs_cap = 2500
        epsilons = [0.003, 0.005, 0.007, 0.009]
        kEs = [0.8, 1.0, 1.2]
        kIs = [0.8, 1.0, 1.2]
        lambdas = [0.7, 0.9]
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 15
    for regime in cosmic_regimes:
        for eps in list(epsilons):
            eta = eta_from_phase15(eps)
            for kE in kEs:
                for kI in kIs:
                    for lam in lambdas:
                        for seed in seeds:
                            cid += 1
                            res = run_condition(
                                regime,
                                n_agents,
                                epochs_cap,
                                eps,
                                kE,
                                kI,
                                lam,
                                eta,
                                seed,
                                noise_base=0.08,
                                thin_out=thin_out,
                            )
                            res["condition_id"] = cid
                            all_results.append(res)
                            elapsed = time.time() - start
                            if elapsed > 90 and 0.009 in epsilons:
                                print(
                                    "   ⚠️ runtime >90s: dropping epsilon=0.009 for remaining runs"
                                )
                                epsilons = [e for e in epsilons if e != 0.009]
                            if elapsed > 120:
                                print("   ⚠️ runtime >120s: reducing epochs->1800")
                                epochs_cap = 1800
                            if elapsed > 150:
                                print(
                                    "   ⚠️ runtime >150s: thinning logs outside dense window"
                                )
                                thin_out = 20
    runtime = time.time() - start
    print(f"✅ Phase 18 runs complete: {runtime:.2f}s (auto-cuts may have applied)")
    return all_results, runtime


def analyze_export(all_results, runtime):
    print("📊 Analyzing Phase 18 results...")
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
            }
        )
        conds.append(
            {
                "condition_id": int(r["condition_id"]),
                "cosmic_regime": p["cosmic_regime"],
                "openness": float(p["openness"]),
                "kE": float(p["kE"]),
                "kI": float(p["kI"]),
                "lam": float(p["lam"]),
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

    # beta/alpha fit per cosmic_regime and eps
    summary_list = []
    for (regime, eps), group in runs_df.groupby(["cosmic_regime", "openness"]):
        grp = group.dropna(subset=["R_mean"])
        if len(grp) < 4:
            continue
        X = np.column_stack(
            [
                np.log(grp.kE.values + 1e-12),
                np.log(grp.kI.values + 1e-12),
                np.ones(len(grp)),
            ]
        )
        y = np.log(grp.R_mean.values + 1e-12)
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta, intercept = coeffs
            beta_over_alpha = float(beta / alpha) if alpha != 0 else np.nan
        except Exception:
            alpha = beta = intercept = np.nan
            beta_over_alpha = np.nan

        # lambda effect: ΔR_mean between lam=0.9 and lam=0.7
        def R_for(lval):
            sel = grp[np.isclose(grp.lam, lval)]
            return float(sel.R_mean.mean()) if len(sel) > 0 else np.nan

        R_hi = R_for(0.9)
        R_lo = R_for(0.7)
        lam_effect = R_hi - R_lo if not (np.isnan(R_hi) or np.isnan(R_lo)) else np.nan
        # eta effect across eps values: approximate by difference from base eps=0.005 if present
        summary_list.append(
            {
                "cosmic_regime": regime,
                "openness": float(eps),
                "alpha": float(alpha),
                "beta": float(beta),
                "beta_over_alpha": beta_over_alpha,
                "lambda_effect": lam_effect,
            }
        )

    summary = {
        "experiment": "Phase18 Universal Unification",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "per_condition": summary_list,
    }
    with open(DATA_DIR / "phase18_universal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures
    try:
        dfsum = pd.DataFrame(summary_list)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        if not dfsum.empty:
            for regime, sub in dfsum.groupby("cosmic_regime"):
                ax.plot(sub.openness, sub.beta_over_alpha, marker="o", label=regime)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("beta/alpha")
        ax.set_title("cosmic beta/alpha vs epsilon")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "cosmic_beta_over_alpha.png")
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        if not dfsum.empty:
            for regime, sub in dfsum.groupby("cosmic_regime"):
                ax.plot(sub.openness, sub.lambda_effect, marker="o", label=regime)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Delta R (lam 0.9 - 0.7)")
        ax.set_title("Resilience vs lambda")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "resilience_vs_lambda.png")
        plt.close()
    except Exception:
        pass

    # report
    lines = [
        f"# Phase 18 Universal Unification — {TIMESTAMP}\n",
        f"Runtime: {runtime:.2f}s\n",
    ]
    for s in summary_list:
        lines.append(
            f"- {s['cosmic_regime']}, eps={s['openness']}: beta/alpha={s['beta_over_alpha']}, lambda_effect={s['lambda_effect']}\n"
        )
    with open(REPORT_DIR / "phase18_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle
    bundle_path = BUNDLE_DIR / f"phase18_{TIMESTAMP}.zip"
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
    all_results, runtime = run_experiment()
    summary = analyze_export(all_results, runtime)
    print("\n=== Phase 18 Complete (fast takeaways) ===")
    for p in summary["per_condition"]:
        print(
            f"{p['cosmic_regime']}, eps={p['openness']}: beta/alpha={p['beta_over_alpha']}, lambda_effect={p['lambda_effect']}"
        )
    print(f"Results saved to {ROOT}")


if __name__ == "__main__":
    main()
