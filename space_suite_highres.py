#!/usr/bin/env python3
"""
High-resolution Space Fundamentals Test Suite (Q4–Q10)
Runs conditions at ~5000 epochs, dense logging, computes stability-window metrics, and exports to discovery_results/<stamp>/
"""
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me
from experiments import gravity_nbody

# Config
EPOCHS = 5000
SEEDS = [111, 222, 333]
LOG_DENSE = 500
STABILITY_WINDOW = 200

STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = Path("discovery_results") / STAMP
DATA = OUT / "data"
FIGS = OUT / "figures"
SUM = OUT / "summaries"
REP = OUT / "report"
BUNDLE = OUT / "bundle"
for d in (OUT, DATA, FIGS, SUM, REP, BUNDLE):
    d.mkdir(parents=True, exist_ok=True)

# Guardrails
GUARD = {
    "early_stop": {"risk": 0.45, "survival": 0.40, "cci": 0.45},
}


def should_log(epoch):
    if epoch <= LOG_DENSE:
        return True
    return (epoch % 10) == 0


def compute_field_coherence(pts):
    # simple proxy: inverse of normalized radial std
    if not pts:
        return 0.0
    rs = [math.sqrt(p.get("x", 0.0) ** 2 + p.get("y", 0.0) ** 2) for p in pts]
    std = float(np.std(rs))
    val = 1.0 / (1.0 + std)
    return float(max(0.0, min(1.0, val)))


def run_gravity_highres(params, seed):
    n = params.get("agents", 200)
    steps = EPOCHS
    dt = params.get("dt", 0.01)
    # call run_nbody with sample_stride=1 for dense logging and return_data
    out_root, pts, energy_records, traj_records, com_records = gravity_nbody.run_nbody(
        n=n, steps=steps, dt=dt, seed=seed, return_data=True, sample_stride=1
    )
    # collect per-epoch data from energy_records and traj_records
    # energy_records has entries per tick; traj_records collects per-particle snapshots
    epoch_idx = {r["tick"]: r for r in energy_records}
    # compute per-epoch aggregates: bound_fraction and field_coherence from final pts only (gravity_nbody computed bound_frac)
    # but we can compute field_coherence per tick by scanning traj_records
    per_epoch = {}
    for r in energy_records:
        t = r["tick"]
        per_epoch[t] = {
            "ke": r.get("ke"),
            "pe": r.get("pe"),
            "total": r.get("total"),
            "dt": r.get("dt"),
        }
    # build particle lists per tick
    ticks_pts = {}
    for pr in traj_records:
        t = pr["tick"]
        ticks_pts.setdefault(t, []).append(pr)
    # compute field_coherence per tick using ticks_pts
    epochs = sorted(list(per_epoch.keys()))
    eoi_like = []
    field_coh_series = []
    for t in epochs:
        pts_t = ticks_pts.get(t, [])
        fc = compute_field_coherence(pts_t)
        field_coh_series.append(fc)
    # read gravity_summary for bound_fraction and energy drift
    try:
        with open(Path(out_root) / "gravity_summary.json") as f:
            gsum = json.load(f)
    except Exception:
        gsum = {}
    bound_frac = gsum.get("bound_fraction", None)
    energy_drift = gsum.get("max_energy_drift", None)
    escape_count = gsum.get("escape_count", None)

    # compute hazard series as 1 - bound_fraction (constant) or using escape fraction if available
    hazard_series = [1.0 - (bound_frac if bound_frac is not None else 0.0)] * len(
        epochs
    )

    # assemble trajectories rows per logging cadence
    traj_rows = []
    for i, t in enumerate(epochs):
        if should_log(t):
            traj_rows.append(
                {
                    "epoch": t,
                    "ke": per_epoch[t]["ke"],
                    "pe": per_epoch[t]["pe"],
                    "total": per_epoch[t]["total"],
                    "field_coherence": field_coh_series[i],
                    "hazard": hazard_series[i],
                }
            )

    # compute stability window metrics over last STABILITY_WINDOW actual epochs
    last_indices = -STABILITY_WINDOW if len(epochs) >= STABILITY_WINDOW else None
    fc_window = field_coh_series[last_indices:]
    haz_window = hazard_series[last_indices:]
    stability_CCI_mean = float(np.mean(fc_window)) if fc_window else float("nan")
    stability_hazard_mean = float(np.mean(haz_window)) if haz_window else float("nan")
    # slope
    if fc_window and len(fc_window) >= 2:
        x = np.arange(len(fc_window))
        m, b = np.polyfit(x, fc_window, 1)
        stability_CCI_slope = float(m)
    else:
        stability_CCI_slope = float("nan")

    peak_CCI = float(np.max(field_coh_series)) if field_coh_series else float("nan")
    final_CCI = float(field_coh_series[-1]) if field_coh_series else float("nan")

    summary = {
        "mechanism": "gravity_nbody",
        "agents": n,
        "seed": seed,
        "stability_CCI_mean": stability_CCI_mean,
        "stability_hazard_mean": stability_hazard_mean,
        "stability_CCI_slope": stability_CCI_slope,
        "peak_CCI": peak_CCI,
        "final_CCI": final_CCI,
        "collapse_flag": bool(stability_hazard_mean > 0.45),
        "lifespan_epochs": None,
        "energy_drift": energy_drift,
        "bound_fraction": bound_frac,
        "field_coherence": float(
            np.mean(field_coh_series) if field_coh_series else 0.0
        ),
        "time_sec": 0.0,
    }

    return summary, traj_rows, out_root


def run_meaning_highres(params, seed):
    # Build agents and run step_update for EPOCHS steps with dense logging
    n = params.get("agents", 200)
    rng = np.random.RandomState(int(seed))
    agents = me.initialize_agents(
        n, params.get("goal_diversity", 3), params.get("noise", 0.08), rng
    )
    cci_series = []
    alive_series = []
    for t in range(EPOCHS):
        me.step_update(agents, params.get("shock", 0.0), "chronic", rng)
        cci = me.collective_cci(agents)
        alive_frac = sum(1 for a in agents if a["alive"]) / len(agents)
        if should_log(t):
            pass
        cci_series.append(cci)
        alive_series.append(alive_frac)

    last_indices = -STABILITY_WINDOW if len(cci_series) >= STABILITY_WINDOW else None
    fc_window = cci_series[last_indices:]
    haz_window = [1.0 - a for a in alive_series[last_indices:]]
    stability_CCI_mean = float(np.mean(fc_window)) if fc_window else float("nan")
    stability_hazard_mean = float(np.mean(haz_window)) if haz_window else float("nan")
    if fc_window and len(fc_window) >= 2:
        x = np.arange(len(fc_window))
        m, b = np.polyfit(x, fc_window, 1)
        stability_CCI_slope = float(m)
    else:
        stability_CCI_slope = float("nan")

    peak_CCI = float(np.max(cci_series)) if cci_series else float("nan")
    final_CCI = float(cci_series[-1]) if cci_series else float("nan")

    summary = {
        "mechanism": "meaning_experiment",
        "agents": n,
        "seed": seed,
        "stability_CCI_mean": stability_CCI_mean,
        "stability_hazard_mean": stability_hazard_mean,
        "stability_CCI_slope": stability_CCI_slope,
        "peak_CCI": peak_CCI,
        "final_CCI": final_CCI,
        "collapse_flag": bool(stability_hazard_mean > 0.45),
        "lifespan_epochs": None,
        "energy_drift": None,
        "bound_fraction": None,
        "field_coherence": float(np.mean(cci_series) if cci_series else 0.0),
        "time_sec": 0.0,
    }
    traj_rows = []
    for i, (cci, alive) in enumerate(zip(cci_series, alive_series)):
        if should_log(i):
            traj_rows.append({"epoch": i, "cci": cci, "alive_frac": alive})
    return summary, traj_rows, None


def build_highres_conditions():
    conds = []
    # Q4
    conds += [
        ("Q4_2D", {"mechanism": "gravity_nbody", "neighbor_degree": 4, "agents": 200}),
        ("Q4_3D", {"mechanism": "gravity_nbody", "neighbor_degree": 6, "agents": 200}),
        ("Q4_4D", {"mechanism": "gravity_nbody", "neighbor_degree": 8, "agents": 200}),
        ("Q4_5D", {"mechanism": "gravity_nbody", "neighbor_degree": 10, "agents": 200}),
    ]
    # Q5
    conds += [
        (
            "Q5_infinite",
            {"mechanism": "gravity_nbody", "bounds": "none", "agents": 190},
        ),
        ("Q5_torus", {"mechanism": "gravity_nbody", "wrap_edges": True, "agents": 190}),
        ("Q5_reflect", {"mechanism": "gravity_nbody", "reflect": True, "agents": 190}),
    ]
    # Q6
    conds += [
        ("Q6_flat", {"mechanism": "gravity_analysis", "curvature": 0.0, "agents": 200}),
        ("Q6_pos", {"mechanism": "gravity_analysis", "curvature": 0.01, "agents": 200}),
        (
            "Q6_neg",
            {"mechanism": "gravity_analysis", "curvature": -0.01, "agents": 200},
        ),
    ]
    # Q7
    conds += [
        (
            "Q7_softWrap",
            {"mechanism": "gravity_nbody", "wrap_edges": True, "agents": 200},
        ),
        (
            "Q7_softBoundary",
            {"mechanism": "gravity_nbody", "edge_damping": 0.05, "agents": 200},
        ),
        (
            "Q7_hardWall",
            {
                "mechanism": "gravity_nbody",
                "edge_type": "hard",
                "inelastic_loss": 0.02,
                "agents": 200,
            },
        ),
    ]
    # Q8
    conds += [
        (
            "Q8_none",
            {"mechanism": "meaning_experiment", "hidden_dim": 0, "agents": 200},
        ),
        (
            "Q8_latent2",
            {
                "mechanism": "meaning_experiment",
                "hidden_dim": 2,
                "latent_noise": 0.10,
                "coupling_latent": 0.15,
                "agents": 200,
            },
        ),
        (
            "Q8_latent4",
            {
                "mechanism": "meaning_experiment",
                "hidden_dim": 4,
                "latent_noise": 0.15,
                "coupling_latent": 0.10,
                "agents": 200,
            },
        ),
    ]
    # Q9
    conds += [
        (
            "Q9_absolute",
            {"mechanism": "gravity_nbody", "frame": "absolute", "agents": 190},
        ),
        ("Q9_relative", {"mechanism": "gravity_nbody", "frame": "com", "agents": 190}),
    ]
    # Q10
    conds += [
        (
            "Q10_closed",
            {"mechanism": "gravity_nbody", "expansion": False, "agents": 190},
        ),
        (
            "Q10_expanding",
            {
                "mechanism": "gravity_nbody",
                "expansion": True,
                "expansion_params": {"early_pct": 0.03, "every": 12},
                "agents": 190,
            },
        ),
    ]
    return conds


def main():
    conds = build_highres_conditions()
    all_runs = []
    all_traj = []
    run_id = 0
    start_time = time.time()
    for qlabel, params in conds:
        for seed in SEEDS:
            run_id += 1
            print(f"RUN {run_id}: {qlabel} seed={seed} -- starting")
            t0 = time.time()
            if params["mechanism"] == "gravity_nbody":
                summ, traj_rows, out_root = run_gravity_highres(params, seed)
            else:
                summ, traj_rows, out_root = run_meaning_highres(params, seed)
            t1 = time.time()
            summ["time_sec"] = t1 - t0
            summ["run_id"] = run_id
            summ["label"] = qlabel
            summ["seed"] = seed
            all_runs.append(summ)
            # ensure stability window entries exist; if NaN, force recompute from traj_rows
            if math.isnan(summ.get("stability_CCI_mean", 0.0)):
                # recompute
                cci_vals = [
                    r.get("cci") or r.get("field_coherence")
                    for r in traj_rows
                    if (
                        r.get("cci") is not None or r.get("field_coherence") is not None
                    )
                ]
                if len(cci_vals) >= STABILITY_WINDOW:
                    window = cci_vals[-STABILITY_WINDOW:]
                    summ["stability_CCI_mean"] = float(np.mean(window))
                    x = np.arange(len(window))
                    m, b = np.polyfit(x, window, 1)
                    summ["stability_CCI_slope"] = float(m)
            # write per-run traj rows with run metadata
            for r in traj_rows:
                r.update({"run_id": run_id, "label": qlabel, "seed": seed})
            all_traj.append(traj_rows)
            # quick save per-run summary to disk
            pd.DataFrame([summ]).to_csv(DATA / f"run_{run_id}_summary.csv", index=False)
            print(f'RUN {run_id}: completed in {summ["time_sec"]:.1f}s')

    # aggregate
    runs_df = pd.DataFrame(all_runs)
    runs_df.to_csv(DATA / "runs_summary.csv", index=False)
    # flatten trajectories
    traj_flat = [r for sub in all_traj for r in sub]
    pd.DataFrame(traj_flat).to_csv(DATA / "trajectories_long.csv", index=False)
    # conditions lookup
    conds_lookup = [
        {"run_id": r["run_id"], "label": r["label"], "seed": r["seed"]}
        for r in all_runs
    ]
    pd.DataFrame(conds_lookup).to_csv(DATA / "conditions_lookup.csv", index=False)

    # compute per-question winners
    winners = []
    for q in set([r["label"].split("_")[0] for r in all_runs]):
        sub = runs_df[runs_df["label"].str.startswith(q)]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["stability_CCI_mean"] = sub["stability_CCI_mean"].astype(float)
        sub["stability_hazard_mean"] = sub["stability_hazard_mean"].astype(float)
        winner = sub.sort_values(
            ["stability_hazard_mean", "stability_CCI_mean"], ascending=[True, False]
        ).iloc[0]
        winners.append(
            {
                "question": q,
                "winner_label": winner["label"],
                "stability_CCI_mean": float(winner["stability_CCI_mean"]),
                "stability_hazard_mean": float(winner["stability_hazard_mean"]),
            }
        )

    with open(SUM / "space_highres_summary.json", "w") as f:
        json.dump(
            {"stamp": STAMP, "n_runs": len(all_runs), "winners": winners}, f, indent=2
        )

    # figures: simple aggregate plots
    try:
        traj_df = pd.DataFrame(traj_flat)
        if "field_coherence" in traj_df.columns:
            grp = traj_df.groupby("epoch")["field_coherence"].mean()
            plt.figure()
            plt.plot(grp.index, grp.values)
            plt.title("Field coherence (mean)")
            plt.savefig(FIGS / "cci_long.png")
            plt.close()
        if "hazard" in traj_df.columns:
            grp2 = traj_df.groupby("epoch")["hazard"].mean()
            plt.figure()
            plt.plot(grp2.index, grp2.values)
            plt.title("hazard (mean)")
            plt.savefig(FIGS / "hazard_long.png")
            plt.close()
    except Exception:
        pass

    # write report
    lines = [
        f"# Space High-Resolution Suite — {STAMP}",
        "",
        f"Run count: {len(all_runs)}",
        "",
    ]
    lines.append("## Winners")
    for w in winners:
        lines.append(
            f"- {w['question']}: {w['winner_label']} (hazard={w['stability_hazard_mean']:.3f}, cci={w['stability_CCI_mean']:.3f})"
        )
    with open(REP / "space_highres_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle
    bundle = BUNDLE / f"space_highres_{STAMP}.zip"
    with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
        for p in list(OUT.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(OUT)))
    h = hashlib.sha256()
    with open(bundle, "rb") as bf:
        for chunk in iter(lambda: bf.read(1 << 20), b""):
            h.update(chunk)
    with open(BUNDLE / "SHA256SUMS.txt", "w") as sf:
        sf.write(f"{h.hexdigest()}  {bundle.name}\n")

    print("High-res suite complete. Outputs in", str(OUT))


if __name__ == "__main__":
    main()
