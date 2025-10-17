#!/usr/bin/env python3
"""
Phase 14: Meta-Temporal Feedback — LONG BATTERY
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

ROOT = Path("./discovery_results/phase14")
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

    def state_vector(self):
        return np.array(
            [
                self.coherence,
                self.coordination,
                self.calibration,
                self.energy,
                self.info,
                self.noise,
            ]
        )

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
        self.coherence = np.clip(self.coherence, 0.2, 0.99)
        self.coordination = np.clip(self.coordination, 0.1, 0.9)
        self.calibration = np.clip(self.calibration, 0.2, 0.85)
        self.noise = np.clip(self.noise, 0.01, 0.8)


def run_epoch_run(
    n_agents,
    epochs,
    openness,
    kE,
    kI,
    seed,
    thin_out=20,
    archive_in=None,
    reinject_scale=0.0,
    noise_base=0.08,
):
    np.random.seed(seed)
    agents = [Agent(i, noise_base=noise_base) for i in range(n_agents)]
    # reinject archive (archive_in is list of state vectors)
    if archive_in is not None and len(archive_in) > 0 and reinject_scale > 0:
        # inject into top fraction of agents small boost
        n_inject = max(1, int(0.02 * n_agents))
        inds = np.argsort([ag.cci() for ag in agents])[:n_inject]
        for i, vec in enumerate(archive_in[:n_inject]):
            ag = agents[inds[i % len(inds)]]
            # apply scaled reinforcement to info and coherence
            ag.info = np.clip(ag.info + reinject_scale * vec[4], 0.2, 1.0)
            ag.coherence = np.clip(ag.coherence + reinject_scale * vec[0], 0.2, 0.99)
    epoch_records = []
    shock_events = []
    dense_window = set(range(1980, 2021))
    shock_start, shock_end = 2000, 2004
    for epoch in range(int(epochs)):
        for ag in agents:
            ag.step(openness, kE, kI)
        # message passing scaled by kI
        if kI > 0:
            n_links = int(0.02 * n_agents * kI)
            for _ in range(max(1, n_links)):
                i, j = np.random.randint(0, n_agents, 2)
                if i == j:
                    continue
                mstr = 0.07 * min(agents[i].info, agents[j].info)
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
                }
            )
        mean_hazard = np.mean([ag.noise for ag in agents])
        if mean_hazard > 0.45:
            noises = sorted(
                [(i, ag.noise) for i, ag in enumerate(agents)],
                key=lambda x: x[1],
                reverse=True,
            )
            ntrim = int(0.2 * n_agents)
            for idx, _ in noises[:ntrim]:
                agents[idx].noise *= 0.85
        ccis = np.array([ag.cci() for ag in agents])
        cohs = np.array([ag.coherence for ag in agents])
        infos = np.array([ag.info for ag in agents])
        mean_cci = float(ccis.mean())
        mean_hazard = float(np.mean([ag.noise for ag in agents]))
        mean_coh = float(cohs.mean())
        if epoch in dense_window or epoch % thin_out == 0:
            epoch_records.append(
                {
                    "epoch": epoch,
                    "system_cci": mean_cci,
                    "system_hazard": mean_hazard,
                    "system_coherence": mean_coh,
                    "mean_info": float(infos.mean()),
                }
            )
        if (
            epoch > 300
            and mean_hazard > 0.55
            and (ccis > 0.3).mean() < 0.30
            and mean_cci < 0.40
        ):
            break
    df = pd.DataFrame(epoch_records)
    shock_df = pd.DataFrame(shock_events)
    # archive: select top 2% by cci
    states = np.array([ag.state_vector() for ag in agents])
    ccis_all = np.array([ag.cci() for ag in agents])
    topn = max(1, int(0.02 * n_agents))
    top_idx = np.argsort(ccis_all)[-topn:][::-1]
    archive = [states[i] for i in top_idx]
    # final metrics over last 200 epochs
    if len(df) > 0:
        final_win = (
            df[df.epoch >= (df.epoch.max() - 199)] if df.epoch.max() >= 199 else df
        )
        stability_cci_mean = float(final_win.system_cci.mean())
        stability_hazard_mean = float(final_win.system_hazard.mean())
        stability_coherence_mean = float(final_win.system_coherence.mean())
        if len(final_win) > 1:
            x = final_win.epoch.values
            y = final_win.system_cci.values
            slope = np.polyfit(x - x[0], y, 1)[0]
        else:
            slope = 0.0
    else:
        stability_cci_mean = 0.0
        stability_hazard_mean = 1.0
        stability_coherence_mean = 0.0
        slope = 0.0
    return {
        "trajectories": df,
        "shock_events": shock_df,
        "archive": archive,
        "metrics": {
            "stability_cci_mean": stability_cci_mean,
            "stability_hazard_mean": stability_hazard_mean,
            "stability_cci_slope": slope,
            "stability_coherence_mean": stability_coherence_mean,
        },
        "parameters": {
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "kE": kE,
            "kI": kI,
            "seed": seed,
            "reinject_scale": reinject_scale,
        },
    }


def run_phase14():
    print("⚡ Phase 14: Meta-Temporal Feedback — LONG BATTERY")
    epsilons = [0.006, 0.008]
    n_agents = 200
    seeds = [1]
    kappa = 0.5
    gini = 0.20
    kE = 1.0
    kIs = [1.0, 1.2]
    lambdas = [0.5, 0.7, 0.9]
    runs_per_chain = 3
    epochs_each = 4000
    all_results = []
    cid = 0
    start = time.time()
    thin_out = 20
    auto_dropped_eps = False
    for eps in list(epsilons):
        for kI in kIs:
            for lam in lambdas:
                # 3 linked runs with archive reinjection
                archive = None
                chain_records = []
                for run_idx in range(runs_per_chain):
                    cid += 1
                    # allow auto-cuts
                    elapsed = time.time() - start
                    epochs = epochs_each
                    if elapsed > 120 and not auto_dropped_eps:
                        print(
                            "   ⚠️ runtime >120s: dropping eps=0.008 for remaining chains"
                        )
                        if 0.008 in epsilons:
                            epsilons = [e for e in epsilons if e != 0.008]
                            auto_dropped_eps = True
                    if elapsed > 150:
                        print("   ⚠️ runtime >150s: reducing epochs per run -> 3000")
                        epochs = 3000
                    if elapsed > 170:
                        print("   ⚠️ runtime >170s: thinning logs outside dense window")
                        thin_out = 30
                    res = run_epoch_run(
                        n_agents,
                        epochs,
                        eps,
                        kE,
                        kI,
                        seeds[0],
                        thin_out=thin_out,
                        archive_in=archive,
                        reinject_scale=lam,
                    )
                    res["condition_id"] = cid
                    res["chain_idx"] = run_idx
                    res["lambda"] = lam
                    res["kI"] = kI
                    res["openness"] = eps
                    all_results.append(res)
                    chain_records.append(res)
                    # prepare archive for next run: use returned archive, scale by lambda
                    archive = [s for s in res["archive"]]
                    # time check
                    if time.time() - start > 280:
                        print("   ⚠️ runtime approaching 5min cap, breaking chain")
                        break
                # end chain
    runtime = time.time() - start
    print(f"✅ Phase 14 complete: {runtime:.2f}s (auto-cuts may have applied)")
    return all_results, runtime


def analyze_export(all_results, runtime):
    print("📊 Analyzing Phase 14 results...")
    trajs = []
    shocks = []
    runs = []
    conds = []
    # group by chain (kI, lam) and within chain run_idx
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
                "chain_idx": r.get("chain_idx", 0),
                "lambda": r.get("lambda", None),
                "kI": r.get("kI", None),
                "openness": r.get("openness", None),
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
                "kI": float(r.get("kI", np.nan)),
                "lambda": float(r.get("lambda", np.nan)),
                "openness": float(r.get("openness", np.nan)),
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

    # compute t_arrow and memory overlap and feedback gains per chain
    summary_chains = []
    # group by (openness,kI,lambda) and order by chain_idx
    grouped = runs_df.groupby(["openness", "kI", "lambda"])
    for name, grp in grouped:
        eps, kI, lam = name
        grp_sorted = grp.sort_values("chain_idx")
        t_arrows = []
        coherences = []
        hazards = []
        cci_means = []
        info_vectors = []
        for _, row in grp_sorted.iterrows():
            # read the last trajectories for that condition
            cid = int(row.condition_id)
            traj = traj_df[traj_df.condition_id == cid]
            if len(traj) == 0:
                continue
            # t_arrow proxy: (coherence - f_c_crit) * dissipation; f_c_crit=0.5
            f_c_crit = 0.5
            coh_mean = float(traj.system_coherence.mean())
            dissipation = float(traj.system_hazard.mean())
            t_arrow = (coh_mean - f_c_crit) * dissipation
            t_arrows.append(t_arrow)
            coherences.append(coh_mean)
            hazards.append(dissipation)
            cci_means.append(float(traj.system_cci.mean()))
            # memory: use mean_info as proxy
            info_vectors.append(
                float(traj.mean_info.mean()) if "mean_info" in traj.columns else np.nan
            )
        # memory overlap: cosine similarity between consecutive epoch-run info vectors
        memory_overlap = []
        for i in range(1, len(info_vectors)):
            a = info_vectors[i - 1]
            b = info_vectors[i]
            if np.isnan(a) or np.isnan(b):
                memory_overlap.append(np.nan)
            else:
                memory_overlap.append(
                    float((a * b) / (np.linalg.norm([a]) * np.linalg.norm([b])))
                )
        # feedback gains: delta CCI between epochs
        deltas = []
        for i in range(1, len(cci_means)):
            deltas.append(cci_means[i] - cci_means[i - 1])
        t_arrow_mean = float(np.nanmean(t_arrows)) if len(t_arrows) > 0 else np.nan
        t_arrow_var = float(np.nanvar(t_arrows)) if len(t_arrows) > 0 else np.nan
        temporal_stability = float(np.nanvar(t_arrows)) if len(t_arrows) > 0 else np.nan
        feedback_gain_mean = float(np.nanmean(deltas)) if len(deltas) > 0 else np.nan
        # production-safe check on final epoch-run
        final_row = grp_sorted.iloc[-1]
        prod_safe = (
            (final_row.stability_cci_mean >= 0.50)
            and (final_row.stability_hazard_mean <= 0.20)
            and (final_row.stability_cci_slope >= 0.0005)
        )
        summary_chains.append(
            {
                "openness": float(eps),
                "kI": float(kI),
                "lambda": float(lam),
                "t_arrow_mean": t_arrow_mean,
                "t_arrow_var": t_arrow_var,
                "temporal_stability": temporal_stability,
                "memory_overlap_mean": (
                    float(np.nanmean(memory_overlap))
                    if len(memory_overlap) > 0
                    else None
                ),
                "feedback_gain_mean": feedback_gain_mean,
                "production_safe_final": bool(prod_safe),
            }
        )

    summary = {
        "experiment": "Phase14 Meta-Temporal",
        "timestamp": TIMESTAMP,
        "runtime_seconds": runtime,
        "chains": summary_chains,
    }
    with open(DATA_DIR / "phase14_feedback_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)

    # figures
    try:
        # temporal stability heatmap (t_arrow variance vs lambda & eps)
        dfsum = pd.DataFrame(summary_chains)
        if not dfsum.empty:
            pivot = dfsum.pivot_table(
                index="lambda", columns="openness", values="t_arrow_var"
            )
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
            im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
            ax.set_title("t_arrow variance (rows=lambda, cols=eps)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "temporal_stability.png")
            plt.close()
        # t_arrow vs coherence
        if not dfsum.empty:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
            ax.scatter(
                dfsum["t_arrow_mean"], dfsum["kI"], c=dfsum["lambda"], cmap="plasma"
            )
            ax.set_xlabel("t_arrow_mean")
            ax.set_ylabel("kI")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "arrow_vs_coherence.png")
            plt.close()
        # feedback gain by lambda
        if not dfsum.empty:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
            for lam in sorted(dfsum["lambda"].unique()):
                sub = dfsum[dfsum["lambda"] == lam]
                ax.plot(sub["kI"], sub["feedback_gain_mean"], "-o", label=f"λ={lam}")
            ax.set_xlabel("kI")
            ax.set_ylabel("feedback_gain")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIG_DIR / "epoch_to_epoch_gain.png")
            plt.close()
    except Exception:
        pass

    # report
    lines = [
        f"# Phase 14 Meta-Temporal Feedback — {TIMESTAMP}\n",
        f"Runtime: {runtime:.2f}s\n",
    ]
    for s in summary_chains:
        lines.append(
            f"- eps={s['openness']}, kI={s['kI']}, λ={s['lambda']}: t_arrow_mean={s['t_arrow_mean']}, t_arrow_var={s['t_arrow_var']}, feedback_gain={s['feedback_gain_mean']}, production_safe_final={s['production_safe_final']}\n"
        )
    with open(REPORT_DIR / "phase14_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle
    bundle_path = BUNDLE_DIR / f"phase14_{TIMESTAMP}.zip"
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
    all_results, runtime = run_phase14()
    summary = analyze_export(all_results, runtime)
    print("\n=== Phase 14 Complete (fast takeaways) ===")
    for s in summary["chains"]:
        print(
            f"eps={s['openness']}, kI={s['kI']}, λ={s['lambda']}: t_arrow_var={s['t_arrow_var']}, prod_safe={s['production_safe_final']}"
        )
    print(f"Results saved to {ROOT}")


if __name__ == "__main__":
    main()
