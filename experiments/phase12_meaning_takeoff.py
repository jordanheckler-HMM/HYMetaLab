#!/usr/bin/env python3
"""
Phase 12: Meaning Take-Off Calibration — FAST BATTERY
Produce takeoff detection for kI* where information self-amplifies.
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

ROOT = Path("./discovery_results/phase12")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
BUNDLE_DIR = ROOT / "bundle"
LOG_DIR = ROOT / "logs"
for d in [DATA_DIR, FIG_DIR, REPORT_DIR, BUNDLE_DIR, LOG_DIR]:
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
    thin_out=10,
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


def run_experiment():
    print("⚡ Phase 12: Meaning Take-Off Calibration — FAST BATTERY")
    n_agents = 150
    seeds = [1, 2]
    epsilons = [0.005, 0.006, 0.007]
    kE_fixed = 1.0
    kIs = [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7]
    epochs_cap = 1400
    noise_base = 0.08
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 10
    baseline_kE_vals = [0.8, 1.2]
    for eps in epsilons[:]:
        for seed in list(seeds):
            for kE in baseline_kE_vals:
                cid += 1
                res = run_condition(
                    n_agents,
                    epochs_cap,
                    eps,
                    kE,
                    1.0,
                    seed,
                    noise_base=noise_base,
                    null_shuffle=False,
                    thin_out=thin_out,
                )
                res["condition_id"] = cid
                all_results.append(res)
                if time.time() - start > 45:
                    print("   ⚠️ runtime approaching limit, dropping seeds to 1")
                    seeds = [seeds[0]]
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
                if time.time() - start > 60 and len(seeds) > 1:
                    print(
                        "   ⚠️ runtime continuing high, reducing seeds->1 and epochs->1000"
                    )
                    seeds = [seeds[0]]
                    epochs_cap = 1000
                if time.time() - start > 75:
                    print(
                        "   ⚠️ runtime high, thinning outside window and may drop last eps"
                    )
                    thin_out = 15
                if time.time() - start > 85:
                    if 0.007 in epsilons:
                        print(
                            "   ⚠️ runtime critical, dropping epsilon=0.007 for remaining runs"
                        )
                        epsilons = [e for e in epsilons if e != 0.007]
    runtime = time.time() - start
    print(
        f"✅ Phase 12 runs complete: {runtime:.2f}s (note: auto-cuts may have applied)"
    )
    return all_results, runtime


def analyze_export(all_results, runtime):
    print("📊 Analyzing Phase 12 results...")
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
        kIs_sorted = sorted(sub.kI.unique())
        for kI in sorted([k for k in kIs_sorted if k != 1.0]):
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
        for rec in records:
            takeoff = (
                not np.isnan(rec["curvature_I"]) and rec["curvature_I"] > 0
            ) and (
                not np.isnan(rec["delta_I"])
                and not np.isnan(delta_E)
                and rec["delta_I"] > delta_E
            )
            takeoffs.append(
                {
                    "kI": rec["kI"],
                    "cci": rec["cci"],
                    "delta_I": rec["delta_I"],
                    "curvature_I": rec["curvature_I"],
                    "takeoff_flag": bool(takeoff),
                }
            )
        kI_star = None
        for t in takeoffs:
            if t["takeoff_flag"]:
                kI_star = float(t["kI"])
                break
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
        "experiment": "Phase 12 Meaning Take-Off",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "per_epsilon": per_eps,
    }
    with open(DATA_DIR / "phase12_takeoff_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
    try:
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (eps, kI), g in traj_df.groupby(["openness", "kI"]):
                ax.plot(g.epoch, g.system_cci, label=f"e={eps:.3f},kI={kI}", alpha=0.6)
            ax.axvspan(600, 604, alpha=0.1, color="red")
            ax.set_title("CCI trajectories (Phase 12)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("CCI")
            ax.legend(fontsize="small", ncol=2)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "cci_long.png")
            plt.close()
        if not traj_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
            for (eps, kI), g in traj_df.groupby(["openness", "kI"]):
                ax.plot(
                    g.epoch, g.system_hazard, label=f"e={eps:.3f},kI={kI}", alpha=0.6
                )
            ax.set_title("Hazard trajectories (Phase 12)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Hazard")
            ax.legend(fontsize="small", ncol=2)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "hazard_long.png")
            plt.close()
        fig, axes = plt.subplots(
            1, len(per_eps), figsize=(5 * len(per_eps), 4), dpi=140
        )
        if len(per_eps) == 1:
            axes = [axes]
        for ax, p in zip(axes, per_eps):
            kis = [t["kI"] for t in p["takeoffs"]]
            ccis = [t["cci"] for t in p["takeoffs"]]
            ax.plot(kis, ccis, "-o")
            if p["kI_star"] is not None:
                ax.axvline(p["kI_star"], color="red", linestyle="--")
                ax.text(
                    p["kI_star"], max(ccis), f' kI*={p["kI_star"]:.2f}', color="red"
                )
            ax.set_title(f'CCI vs kI (eps={p["openness"]:.3f})')
            ax.set_xlabel("kI")
            ax.set_ylabel("CCI")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "takeoff_curve.png")
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=140)
        rows = []
        cols = []
        for p in per_eps:
            cols.append(p["openness"])
            rows.append([t["curvature_I"] for t in p["takeoffs"]])
        if rows:
            arr = np.array(rows).T
            im = ax.imshow(arr, aspect="auto", cmap="RdYlBu", interpolation="nearest")
            ax.set_yticks(range(len(per_eps[0]["takeoffs"])))
            ax.set_yticklabels([str(t["kI"]) for t in per_eps[0]["takeoffs"]])
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([f"{c:.3f}" for c in cols])
            ax.set_title("curvature_I (rows=kI, cols=eps)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "curvature_map.png")
            plt.close()
    except Exception:
        pass
    report_lines = []
    report_lines.append(f"# Phase 12 Meaning Take-Off — {TIMESTAMP}\n")
    report_lines.append(f"Runtime: {runtime:.2f}s\n")
    for p in per_eps:
        report_lines.append(f'## eps={p["openness"]:.3f}\n')
        report_lines.append(f'- baseline_cci: {p["baseline_cci"]}\n')
        report_lines.append(f'- delta_E (kE 1.2 - 0.8): {p["delta_E"]}\n')
        report_lines.append(f'- kI*: {p["kI_star"]}\n')
        report_lines.append("\n")
    with open(REPORT_DIR / "phase12_results.md", "w") as f:
        f.write("\n".join(report_lines))
    bundle_path = BUNDLE_DIR / f"phase12_{TIMESTAMP}.zip"
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
    print("\n=== Phase 12 Complete (fast takeaways) ===")
    for p in summary["per_epsilon"]:
        print(
            f"eps={p['openness']:.3f}: kI*={p['kI_star']} (baseline CCI={p['baseline_cci']})"
        )
    print(f"Results saved under {ROOT}")


if __name__ == "__main__":
    main()
