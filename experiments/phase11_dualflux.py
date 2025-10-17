#!/usr/bin/env python3
"""
Phase 11: Dual-Flux Coupling & Informational Dominance — FAST BATTERY
====================================================================

Quantify exponents α, β in CCI ≈ (E^α × I^β)/N and detect information-dominance inflection.
Optimized for fast execution with auto-cuts if runtime approaches budget.
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


# Utilities
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Output dirs
ROOT = Path("./discovery_results")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
BUNDLE_DIR = ROOT / "bundle"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR, BUNDLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- Agent model (lightweight) ----------
class DualFluxAgent:
    def __init__(self, aid, init_noise=0.08, init_coh=0.6):
        self.aid = aid
        self.noise = init_noise + np.random.uniform(-0.01, 0.01)
        self.coherence = init_coh + np.random.uniform(-0.02, 0.02)
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

    def step(self, openness, kE, kI, dt=0.02):
        cci = self.cci()
        # resources
        effE = kE * (0.8 + 0.4 * openness)
        effI = kI * (0.6 + 0.6 * openness)
        self.energy = np.clip(self.energy + dt * effE * (1.0 - self.energy), 0.1, 1.0)
        self.info = np.clip(self.info + dt * effI * (0.9 - self.info), 0.2, 1.0)
        # coherence driven by both
        energy_boost = 0.12 * (self.energy - 0.5)
        info_boost = 0.10 * (self.info - 0.6)
        coh_target = 0.60 + 0.18 * openness
        rate = 1.8 + energy_boost + info_boost
        self.coherence += dt * rate * (coh_target - self.coherence)
        # coordination
        coord_target = 0.45 + 0.15 * self.info
        self.coordination += dt * 1.2 * (
            coord_target - self.coordination
        ) + np.random.normal(0, 0.01)
        # calibration
        self.calibration += dt * (0.6 * self.energy) * (
            0.55 - self.calibration
        ) + np.random.normal(0, 0.01)
        # noise
        resfac = 0.6 * self.energy + 0.4 * self.info
        noise_target = 0.12 - 0.08 * cci * resfac
        self.noise += dt * 1.2 * (noise_target - self.noise)
        # bounds
        self.coherence = np.clip(self.coherence, 0.2, 0.95)
        self.coordination = np.clip(self.coordination, 0.1, 0.9)
        self.calibration = np.clip(self.calibration, 0.2, 0.85)
        self.noise = np.clip(self.noise, 0.01, 0.6)
        self.cci_hist.append(cci)


# ---------- Simulation function ----------


def run_dualflux(
    n_agents=140,
    epochs=1400,
    openness=0.007,
    kE=1.0,
    kI=1.0,
    null_shuffle=False,
    seed=1,
):
    np.random.seed(seed)
    agents = [DualFluxAgent(i) for i in range(n_agents)]
    # prepare kI sequence for null test
    if null_shuffle:
        kI_seq = np.full(epochs, kI)
        np.random.shuffle(kI_seq)
    else:
        kI_seq = np.full(epochs, kI)
    epoch_records = []
    shock_events = []
    dense_window = set(range(580, 621))
    shock_start, shock_end = 600, 604
    for epoch in range(epochs):
        kI_epoch = kI_seq[epoch]
        # step agents
        for ag in agents:
            ag.step(openness, kE, kI_epoch)
        # message passing: lightweight aggregate effect scaled by kI_epoch
        if kI_epoch > 0:
            # choose pairs proportional to info capacity
            n_links = int(0.02 * n_agents * kI_epoch)  # small
            for _ in range(n_links):
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
        # shock
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
        # guardrails hygiene
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
        # cci slope nudge
        if len(epoch_records) >= 50:
            recent = [r["system_cci"] for r in epoch_records[-50:]]
            slope = (recent[-1] - recent[0]) / 49
            if slope < 0.001:
                for ag in agents:
                    ag.coordination = min(0.70, ag.coordination + 0.05)
        # metrics
        ccis = np.array([ag.cci() for ag in agents])
        noises = np.array([ag.noise for ag in agents])
        cohs = np.array([ag.coherence for ag in agents])
        mean_cci = float(ccis.mean())
        mean_hazard = float(noises.mean())
        survival = float((ccis > 0.3).mean())
        if epoch in dense_window or epoch % 10 == 0:
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
        # early stop
        if epoch > 300 and mean_hazard > 0.45 and survival < 0.40 and mean_cci < 0.45:
            break
    df = pd.DataFrame(epoch_records)
    shock_df = pd.DataFrame(shock_events)
    # final metrics
    if len(df) > 0:
        final_win = df[df.epoch >= (epoch - 199)] if epoch >= 199 else df
        stability_cci_mean = float(final_win.system_cci.mean())
        stability_hazard_mean = float(final_win.system_hazard.mean())
        # slope over final window
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


# ---------- Experiment matrix ----------


def run_matrix():
    print("⚡ Phase 11: Dual-Flux Coupling — FAST BATTERY")
    n_agents = 140
    epochs_cap = 1400
    seed = 1
    epsilons = [0.004, 0.007, 0.010]
    kEs = [0.8, 1.0, 1.2]
    kIs = [0.8, 1.0, 1.2, 1.35, 1.5]
    all_results = []
    cid = 0
    start = time.time()
    for eps in epsilons:
        for kE in kEs:
            for kI in kIs:
                cid += 1
                res = run_dualflux(
                    n_agents=n_agents,
                    epochs=epochs_cap,
                    openness=eps,
                    kE=kE,
                    kI=kI,
                    null_shuffle=False,
                    seed=seed,
                )
                res["condition_id"] = cid
                all_results.append(res)
                # runtime check
                if time.time() - start > 45:
                    print(
                        "   ⚠️ runtime approaching limit, reducing epochs to 1100 for remaining runs"
                    )
                    epochs_cap = 1100
    # null slice for eps=0.007
    cid += 1
    null_res = run_dualflux(
        n_agents=n_agents,
        epochs=min(epochs_cap, 800),
        openness=0.007,
        kE=1.0,
        kI=1.35,
        null_shuffle=True,
        seed=seed + 100,
    )
    null_res["condition_id"] = cid
    all_results.append(null_res)
    runtime = time.time() - start
    print(f"✅ Experiment complete: {runtime:.2f}s")
    return all_results, runtime


# ---------- Analysis & Export ----------


def analyze_and_export(all_results, runtime):
    print("📊 Analyzing results...")
    all_traj = []
    all_shocks = []
    runs = []
    conds = []
    for r in all_results:
        df = r["trajectories"].copy()
        df["condition_id"] = r["condition_id"]
        all_traj.append(df)
        if len(r["shock_events"]) > 0:
            s = r["shock_events"].copy()
            s["condition_id"] = r["condition_id"]
            all_shocks.append(s)
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
    traj_df = pd.concat(all_traj, ignore_index=True) if all_traj else pd.DataFrame()
    shock_df = (
        pd.concat(all_shocks, ignore_index=True) if all_shocks else pd.DataFrame()
    )
    runs_df = pd.DataFrame(runs)
    conds_df = pd.DataFrame(conds)
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA_DIR / "trajectories_long.csv", index=False)
    if not shock_df.empty:
        shock_df.to_csv(DATA_DIR / "shock_events.csv", index=False)
    conds_df.to_csv(DATA_DIR / "conditions_lookup.csv", index=False)
    # compute per-epsilon fits and metrics
    results_summary = []
    for eps in sorted(runs_df.openness.unique()):
        subset = runs_df[runs_df.openness == eps]
        # pivot for kI curvature using kE==1.0 rows
        baseline_kE = subset[subset.kE == 1.0]

        # get CCI at specific kI
        def getCCI(kIval, kEval=1.0):
            sel = subset[(subset.kI == kIval) & (subset.kE == kEval)]
            return float(sel.stability_cci_mean.iloc[0]) if len(sel) > 0 else np.nan

        cci_kI_12 = getCCI(1.2)
        cci_kI_135 = getCCI(1.35)
        cci_kI_15 = getCCI(1.5)
        curvature_I = np.nan
        if (
            not np.isnan(cci_kI_15)
            and not np.isnan(cci_kI_135)
            and not np.isnan(cci_kI_12)
        ):
            curvature_I = cci_kI_15 - 2 * cci_kI_135 + cci_kI_12

        # sensitivities
        def getCCI_kE(kEval, kIval=1.0):
            sel = subset[(subset.kE == kEval) & (subset.kI == kIval)]
            return float(sel.stability_cci_mean.iloc[0]) if len(sel) > 0 else np.nan

        delta_E = getCCI_kE(1.2, 1.0) - getCCI_kE(0.8, 1.0)
        delta_I = getCCI_kE(1.0, 1.2) - getCCI_kE(1.0, 0.8)
        dominance_flag = (not np.isnan(curvature_I) and curvature_I > 0) and (
            abs(delta_I) > abs(delta_E)
        )
        # exponent fit in log-log space, using ratios relative to baseline (kE0=1.0, kI0=1.0)
        kE0, kI0 = 1.0, 1.0
        X_rows = []
        y = []
        for _, row in subset.iterrows():
            cci = max(row.stability_cci_mean, 1e-6)
            # use logs of ratios so intercept corresponds to baseline
            X_rows.append([np.log(row.kE / kE0), np.log(row.kI / kI0)])
            y.append(np.log(cci))
        X_rows = np.array(X_rows)
        y = np.array(y)
        alpha_hat = np.nan
        beta_hat = np.nan
        beta_over_alpha = np.nan
        intercept = np.nan
        r2 = np.nan
        # need at least 4 rows to fit two coeffs + intercept robustly
        if len(y) > 3:
            try:
                X_design = np.column_stack([X_rows, np.ones(len(X_rows))])
                coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                alpha_hat, beta_hat, intercept = (
                    float(coef[0]),
                    float(coef[1]),
                    float(coef[2]),
                )
                # compute R^2 for fit quality
                y_pred = X_design.dot(coef)
                ss_res = float(((y - y_pred) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum()) if len(y) > 0 else np.nan
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                # avoid dividing by very small alpha (meaningless large ratio)
                if not np.isnan(alpha_hat) and abs(alpha_hat) > 1e-3:
                    beta_over_alpha = float(beta_hat / alpha_hat)
                else:
                    beta_over_alpha = np.nan
            except Exception:
                alpha_hat = np.nan
                beta_hat = np.nan
                beta_over_alpha = np.nan
                intercept = np.nan
                r2 = np.nan
        # null artifact: find null slice for this eps
        null_flag = False
        null_rows = runs_df[(runs_df.openness == eps) & (runs_df.null_shuffle == True)]
        if len(null_rows) > 0 and not np.isnan(beta_hat):
            beta_null = (
                float(null_rows.stability_cci_mean.iloc[0])
                if "stability_cci_mean" in null_rows.columns
                else np.nan
            )
            # interpret beta_null as proxy; set flag conservatively
            if not np.isnan(beta_null) and abs(beta_null) >= 0.5 * abs(beta_hat):
                null_flag = True
        results_summary.append(
            {
                "openness": float(eps),
                "alpha_hat": alpha_hat,
                "beta_hat": beta_hat,
                "beta_over_alpha": beta_over_alpha,
                "delta_E": float(delta_E) if not np.isnan(delta_E) else None,
                "delta_I": float(delta_I) if not np.isnan(delta_I) else None,
                "curvature_I": (
                    float(curvature_I) if not np.isnan(curvature_I) else None
                ),
                "dominance_flag": bool(dominance_flag),
                "null_artifact_flag": bool(null_flag),
            }
        )
    # export JSON
    summary = {
        "experiment_info": {
            "phase": "Phase 11 Dual-Flux",
            "timestamp": TIMESTAMP,
            "runtime_seconds": runtime,
            "n_conditions": len(all_results),
        },
        "per_epsilon": results_summary,
    }
    with open(DATA_DIR / "phase11_dualflux_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
    # figures: simple CCI long and hazard long
    try:
        df_all = traj_df
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=140)
        for i, eps in enumerate(sorted(df_all.openness.unique())):
            ax = axes[i]
            sub = df_all[df_all.openness == eps]
            for (kE, kI), g in sub.groupby(["kE", "kI"]):
                ax.plot(g.epoch, g.system_cci, label=f"kE={kE},kI={kI}", alpha=0.6)
            ax.axvspan(600, 604, alpha=0.2, color="red")
            ax.set_title(f"CCI (eps={eps:.3f})")
            ax.set_xlabel("epoch")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cci_long_phase11.png")
        plt.close()
    except Exception:
        pass
    # create bundle
    bundle_path = BUNDLE_DIR / f"phase11_{TIMESTAMP}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in [DATA_DIR, FIG_DIR, REPORT_DIR]:
            for fp in d.rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(ROOT))
        # sha
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


# ---------- Orchestrator ----------


def main():
    start = time.time()
    all_results, runtime = run_matrix()
    summary = analyze_and_export(all_results, time.time() - start)
    print("\n=== Phase 11 Complete ===")
    for p in summary["per_epsilon"]:
        boap = p["beta_over_alpha"]
        verdict = (
            "INFO-DOMINANT"
            if (not np.isnan(boap) and boap > 1.5)
            else (
                "NEAR-PARITY"
                if (not np.isnan(boap) and 0.75 <= boap <= 1.5)
                else "ENERGY-LEAN"
            )
        )
        print(
            f"ε={p['openness']:.3f}: β/α={p['beta_over_alpha']:.3f} → {verdict}, dominance_flag={p['dominance_flag']}, null_flag={p['null_artifact_flag']}"
        )
    print("Results exported to discovery_results")


if __name__ == "__main__":
    main()
