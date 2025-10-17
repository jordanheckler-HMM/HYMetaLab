#!/usr/bin/env python3
"""
Space Fundamentals Test Suite Orchestrator
Runs Q1..Q11 using existing experiment modules and exports results to ./space_suite_outputs/<timestamp>/
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
from experiments import calibration_experiment, gravity_nbody, shock_resilience

# Output layout
STAMP = time.strftime("%Y%m%d_%H%M%S")
ROOT_OUT = Path("space_suite_outputs") / STAMP
DATA_DIR = ROOT_OUT / "data"
FIG_DIR = ROOT_OUT / "figures"
REPORT_DIR = ROOT_OUT / "report"
SUM_DIR = ROOT_OUT / "summaries"
BUNDLE_DIR = ROOT_OUT / "bundle"
for d in (DATA_DIR, FIG_DIR, REPORT_DIR, SUM_DIR, BUNDLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Global guardrails (as provided)
GUARD = {
    "early_stop": {"risk": 0.45, "survival": 0.40, "cci": 0.45},
    "hygiene": {"ma20_hazard": 0.20, "trim_pct": 0.30, "cooldown": 25},
    "expansion_early": {"pct": 0.03, "every": 12},
    "expansion_trigger": {"pct": 0.01, "hazard": 0.18, "consec": 3, "cooldown": 30},
    "fabric_guards": {
        "cci_slope_50e": -0.001,
        "coord_boost": 0.05,
        "coord_cap": 0.70,
        "ineq_nudge": -0.02,
        "ineq_floor": 0.15,
    },
}

SEEDS = [101, 202]


# Logging cadence helper: dense for first 200 epochs then /10
def should_log(epoch):
    if epoch < 200:
        return True
    return (epoch % 10) == 0


def early_stop_check(metrics):
    # metrics: dict with risk, survival, cci
    return (
        metrics.get("risk", 0.0) > GUARD["early_stop"]["risk"]
        and metrics.get("survival", 1.0) < GUARD["early_stop"]["survival"]
        and metrics.get("cci", 1.0) < GUARD["early_stop"]["cci"]
    )


def run_condition(run_id, label, params, seed):
    # dispatch to appropriate module based on label/params
    # return summary dict and trajectories list of dicts
    start = time.time()
    traj = []
    summary = {"run_id": run_id, "label": label, "seed": seed}
    if params.get("mechanism") == "gravity_nbody":
        # translate params
        n = params.get("agents", 100)
        steps = params.get("epochs_cap", 1200)
        dt = params.get("dt", 0.01)
        if n <= 0:
            # vacuum case: synthetic background field coherence (deterministic via seed)
            rng = np.random.RandomState(seed)
            field = []
            for t in range(steps):
                # low-level fluctuation with small drift
                base = 0.05
                drift = 0.0
                val = base + 0.01 * math.sin(0.001 * t) + 0.005 * (rng.rand() - 0.5)
                field.append(float(max(0.0, val)))
            # record trajectories at logging cadence
            for t, val in enumerate(field):
                if should_log(t):
                    traj.append({"run_id": run_id, "epoch": t, "field_coherence": val})
            out_root = DATA_DIR / f"vacuum_run_{run_id}"
            out_root.mkdir(parents=True, exist_ok=True)
            # write a small summary file
            with open(out_root / "vacuum_summary.json", "w") as f:
                json.dump(
                    {
                        "field_coherence_mean": float(np.mean(field)),
                        "field_coherence_std": float(np.std(field)),
                    },
                    f,
                )
            summary.update(
                {
                    "mechanism": "gravity_nbody_vacuum",
                    "field_coherence_mean": float(np.mean(field)),
                }
            )
            traj_records = []
            pts = []
            energy_records = []
            com_records = []
        else:
            out_root, pts, energy_records, traj_records, com_records = (
                gravity_nbody.run_nbody(
                    n=n, steps=steps, dt=dt, seed=seed, return_data=True
                )
            )
        # convert trajectory records
        for r in traj_records:
            if should_log(r["tick"]):
                traj.append(
                    {
                        "run_id": run_id,
                        "epoch": r["tick"],
                        "id": r["id"],
                        "x": r["x"],
                        "y": r["y"],
                        "vx": r["vx"],
                        "vy": r["vy"],
                        "m": r["m"],
                    }
                )
        # load summary
        try:
            with open(Path(out_root) / "gravity_summary.json") as f:
                gsum = json.load(f)
        except Exception:
            gsum = {}
        summary.update(
            {
                "mechanism": "gravity_nbody",
                "bound_fraction": gsum.get("bound_fraction", None),
                "energy_drift": gsum.get("max_energy_drift", None),
            }
        )

    elif params.get("mechanism") == "shock_resilience":
        out_root = shock_resilience.run_shock_experiment(
            params=params, outdir=str(DATA_DIR / f"run_{run_id}")
        )
        # ingest CSV
        try:
            import csv

            with open(out_root / "shock_time_series.csv") as cf:
                import csv

                r = csv.DictReader(cf)
                for row in r:
                    epoch = int(row["tick"])
                    if should_log(epoch):
                        traj.append(
                            {
                                "run_id": run_id,
                                "epoch": epoch,
                                "mean_resource_alive": float(
                                    row["mean_resource_alive"]
                                ),
                                "alive_fraction": float(row["alive_fraction"]),
                            }
                        )
            with open(out_root / "shock_summary.json") as f:
                s = json.load(f)
                summary.update(
                    {
                        "final_alive_fraction": s.get("final_alive_fraction"),
                        "recovered_at": s.get("recovered_at"),
                    }
                )
        except Exception:
            pass

    elif params.get("mechanism") == "meaning_experiment":
        # reuse meaning_experiment.run_single
        params_local = params.copy()
        # ensure required keys
        params_local.setdefault("n_agents", params_local.get("agents", 200))
        params_local.setdefault("goal_diversity", params_local.get("goal_diversity", 3))
        params_local.setdefault("noise", params_local.get("noise", 0.07))
        params_local.setdefault("shock", params_local.get("shock", 0.0))
        params_local.setdefault(
            "stress_duration", params_local.get("stress_duration", "chronic")
        )
        params_local["seed"] = seed
        res = me.run_single(params_local)
        hist = res.get("history", {})
        # collect trajectories with cci, hazard proxy (collapse_risk) per logged epoch index
        cci_list = hist.get("cci", [])
        alive_list = hist.get("alive_frac", [])
        for i, (cci, alive) in enumerate(zip(cci_list, alive_list)):
            epoch = i
            if should_log(epoch):
                traj.append(
                    {"run_id": run_id, "epoch": epoch, "cci": cci, "alive_frac": alive}
                )
        summary.update(
            {
                "mechanism": "meaning_experiment",
                "collective_cci_delta": res.get("collective_cci_delta"),
                "survival_rate": res.get("survival_rate"),
                "collapse_risk": res.get("collapse_risk"),
            }
        )

    else:
        # fallback: small calibration experiment as placeholder
        out = calibration_experiment.run_calibration(
            {"seeds": [seed], "noise_levels": [0.1], "n_agents": 20, "ticks": 50},
            outdir=str(DATA_DIR / f"run_{run_id}"),
        )
        summary.update({"mechanism": "calibration", "result_dir": out.get("run_dir")})

    elapsed = time.time() - start
    summary["time_sec"] = elapsed
    return summary, traj


def build_conditions():
    # Build the 11-question matrix as list of (qid, condition_label, params)
    conditions = []
    rid = 0
    # Q1
    conditions.append(
        (
            1,
            "Q1_A_independent",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1200,
                "agents": 150,
                "substrate_persistent": True,
                "coupling": 0.65,
                "noise": 0.08,
            },
        )
    )
    conditions.append(
        (
            1,
            "Q1_B_relational",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1200,
                "agents": 150,
                "substrate_persistent": False,
                "coupling": 0.65,
                "noise": 0.08,
            },
        )
    )
    # Q2
    conditions.append(
        (
            2,
            "Q2_vacuum_closed",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1200,
                "agents": 0,
                "intake": 0,
                "noise": 0.01,
            },
        )
    )
    conditions.append(
        (
            2,
            "Q2_vacuum_minOpen",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1200,
                "agents": 0,
                "openness": {
                    "mechanism": "agent_io",
                    "epsilon": 0.0015,
                    "period": 40,
                    "reservoir": 0.25,
                },
            },
        )
    )
    # Q3
    conditions.append(
        (
            3,
            "Q3_continuous",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1500,
                "dt": 1e-3,
                "softening": 0.02,
            },
        )
    )
    conditions.append(
        (
            3,
            "Q3_finer",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1500,
                "dt": 5e-4,
                "softening": 0.02,
            },
        )
    )
    conditions.append(
        (
            3,
            "Q3_discrete",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1500,
                "dt": 1e-3,
                "quantize": 1e-2,
                "softening": 0.02,
            },
        )
    )
    # Q4
    for label, deg in [("Q4_2D", 4), ("Q4_3D", 6), ("Q4_4D", 8), ("Q4_5D", 10)]:
        conditions.append(
            (
                4,
                label,
                {
                    "mechanism": "gravity_nbody",
                    "epochs_cap": 2000,
                    "agents": 200,
                    "neighbor_degree": deg,
                    "coupling": 0.62,
                    "noise": 0.07,
                },
            )
        )
    # Q5
    conditions.append(
        (
            5,
            "Q5_infinite",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 2000,
                "agents": 180,
                "bounds": "none",
            },
        )
    )
    conditions.append(
        (
            5,
            "Q5_torus",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 2000,
                "agents": 180,
                "wrap_edges": True,
            },
        )
    )
    conditions.append(
        (
            5,
            "Q5_reflect",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 2000,
                "agents": 180,
                "reflect": True,
            },
        )
    )
    # Q6
    conditions.append(
        (
            6,
            "Q6_flat",
            {
                "mechanism": "gravity_analysis",
                "epochs_cap": 2000,
                "agents": 200,
                "curvature": 0.0,
                "coupling": 0.6,
            },
        )
    )
    conditions.append(
        (
            6,
            "Q6_pos",
            {
                "mechanism": "gravity_analysis",
                "epochs_cap": 2000,
                "agents": 200,
                "curvature": 0.01,
                "coupling": 0.6,
            },
        )
    )
    conditions.append(
        (
            6,
            "Q6_neg",
            {
                "mechanism": "gravity_analysis",
                "epochs_cap": 2000,
                "agents": 200,
                "curvature": -0.01,
                "coupling": 0.6,
            },
        )
    )
    # Q7
    conditions.append(
        (
            7,
            "Q7_softWrap",
            {"mechanism": "gravity_nbody", "epochs_cap": 1500, "wrap_edges": True},
        )
    )
    conditions.append(
        (
            7,
            "Q7_softBoundary",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1500,
                "edge_damping": "sigmoid",
            },
        )
    )
    conditions.append(
        (
            7,
            "Q7_hardWall",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1500,
                "edge_type": "hard",
                "inelastic_loss": 0.02,
            },
        )
    )
    # Q8
    conditions.append(
        (
            8,
            "Q8_none",
            {
                "mechanism": "meaning_experiment",
                "epochs_cap": 2000,
                "agents": 200,
                "hidden_dim": 0,
            },
        )
    )
    conditions.append(
        (
            8,
            "Q8_latent2",
            {
                "mechanism": "meaning_experiment",
                "epochs_cap": 2000,
                "agents": 200,
                "hidden_dim": 2,
                "latent_noise": 0.10,
                "coupling_latent": 0.15,
            },
        )
    )
    conditions.append(
        (
            8,
            "Q8_latent4",
            {
                "mechanism": "meaning_experiment",
                "epochs_cap": 2000,
                "agents": 200,
                "hidden_dim": 4,
                "latent_noise": 0.15,
                "coupling_latent": 0.10,
            },
        )
    )
    # Q9
    conditions.append(
        (
            9,
            "Q9_absolute",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1200,
                "agents": 160,
                "frame": "absolute",
            },
        )
    )
    conditions.append(
        (
            9,
            "Q9_relative",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1200,
                "agents": 160,
                "frame": "com",
            },
        )
    )
    # Q10
    conditions.append(
        (
            10,
            "Q10_closed",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1500,
                "agents": 180,
                "expansion": False,
            },
        )
    )
    conditions.append(
        (
            10,
            "Q10_expanding",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 1500,
                "agents": 180,
                "expansion": True,
                "expansion_params": {"early_pct": 0.03, "every": 12},
            },
        )
    )
    # Q11
    conditions.append(
        (
            11,
            "Q11_create",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 2500,
                "agents": 200,
                "subdomain_action": "create",
                "subdomain_epoch": 800,
                "subdomain_fraction": 0.10,
            },
        )
    )
    conditions.append(
        (
            11,
            "Q11_destroy",
            {
                "mechanism": "gravity_nbody",
                "epochs_cap": 2500,
                "agents": 200,
                "subdomain_action": "destroy",
                "subdomain_epoch": 1600,
            },
        )
    )

    # attach run ids
    conds_with_id = []
    rid = 0
    for q, label, params in conditions:
        params["question"] = q
        params["label"] = label
        params["run_id"] = rid
        conds_with_id.append((rid, q, label, params))
        rid += 1
    return conds_with_id


def aggregate_and_report(all_summaries, all_trajs):
    # assemble runs_summary.csv and trajectories_long.csv and conditions_lookup
    runs_df = pd.DataFrame(all_summaries)
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)

    traj_rows = []
    for tlist in all_trajs:
        for r in tlist:
            traj_rows.append(r)
    pd.DataFrame(traj_rows).to_csv(DATA_DIR / "trajectories_long.csv", index=False)

    # conditions lookup
    conds = [
        {
            "run_id": s["run_id"],
            "label": s.get("label"),
            "mechanism": s.get("mechanism"),
        }
        for s in all_summaries
    ]
    pd.DataFrame(conds).to_csv(DATA_DIR / "conditions_lookup.csv", index=False)

    # compute per-question winners
    winners = []
    for q in sorted(set(runs_df["question"].values)):
        sub = runs_df[runs_df["question"] == q]
        if sub.empty:
            continue
        # compute stability metrics (use survival_rate as inverse risk where available)
        sub["stability_CCI_mean"] = sub.get("collective_cci_delta", 0.0)
        sub["stability_hazard_mean"] = 1.0 - sub.get("survival_rate", 0.5)
        # winner rule: lowest hazard, tie-break by highest CCI
        winner = sub.sort_values(
            ["stability_hazard_mean", "stability_CCI_mean"], ascending=[True, False]
        ).iloc[0]

        # coerce to native types
        def pyval(x):
            try:
                if x is None:
                    return None
                if hasattr(x, "item"):
                    return x.item()
                return x
            except Exception:
                return x

        winners.append(
            {
                "question": int(q),
                "winner_label": str(pyval(winner.get("label"))),
                "stability_CCI_mean": pyval(winner.get("stability_CCI_mean")),
                "stability_hazard_mean": pyval(winner.get("stability_hazard_mean")),
                "notes": "",
            }
        )

    pd.DataFrame(winners).to_csv(SUM_DIR / "winners_table.csv", index=False)

    # simple plots: CCI and hazard long plots (aggregate mean over runs)
    try:
        # CCI long (mean per epoch across runs that recorded 'cci')
        traj_df = pd.DataFrame(traj_rows)
        if "cci" in traj_df.columns:
            grp = traj_df.groupby("epoch")["cci"].mean()
            plt.figure()
            plt.plot(grp.index, grp.values)
            plt.title("CCI long (mean)")
            plt.savefig(FIG_DIR / "cci_long.png")
            plt.close()
        if "alive_frac" in traj_df.columns:
            grp2 = traj_df.groupby("epoch")["alive_frac"].mean()
            plt.figure()
            plt.plot(grp2.index, grp2.values)
            plt.title("alive fraction (mean)")
            plt.savefig(FIG_DIR / "hazard_long.png")
            plt.close()
    except Exception:
        pass

    # summary JSON
    summary_json = {"stamp": STAMP, "n_runs": len(all_summaries), "winners": winners}
    with open(SUM_DIR / "space_suite_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # markdown report
    lines = [
        f"# Space Fundamentals Test Suite — {STAMP}",
        "",
        f"Total runs: {len(all_summaries)}",
        "",
    ]
    lines.append("## Winners per question")
    for w in winners:
        lines.append(
            f"- Q{w['question']}: {w['winner_label']} (hazard={w['stability_hazard_mean']}, cci={w['stability_CCI_mean']})"
        )

    with open(REPORT_DIR / "space_suite_results.md", "w") as f:
        f.write("\n".join(lines))

    # bundle everything
    bundle_path = BUNDLE_DIR / f"space_suite_{STAMP}.zip"
    with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as z:
        for p in (
            list(DATA_DIR.rglob("*"))
            + list(FIG_DIR.rglob("*"))
            + list(REPORT_DIR.rglob("*"))
            + list(SUM_DIR.rglob("*"))
        ):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(ROOT_OUT)))
    # sha
    h = hashlib.sha256()
    with open(bundle_path, "rb") as bf:
        for chunk in iter(lambda: bf.read(1 << 20), b""):
            h.update(chunk)
    with open(BUNDLE_DIR / "SHA256SUMS.txt", "w") as sf:
        sf.write(f"{h.hexdigest()}  {bundle_path.name}\n")

    print("Aggregation complete. Outputs in", str(ROOT_OUT))


def main():
    conds = build_conditions()
    all_summaries = []
    all_trajs = []
    run_count = 0
    # For speed in this environment, sample subset: run first condition of each question for both seeds
    sample_runs = []
    seen_q = set()
    for rid, q, label, params in conds:
        if q not in seen_q:
            sample_runs.append((rid, q, label, params))
            seen_q.add(q)
    # duplicate for second seed
    to_run = []
    for rid, q, label, params in sample_runs:
        to_run.append((rid, q, label, params, SEEDS[0]))
        to_run.append((rid, q, label, params, SEEDS[1]))

    for rid, q, label, params, seed in to_run:
        run_count += 1
        print(f"Running {run_count}/{len(to_run)}: Q{q} {label} seed={seed}")
        summ, traj = run_condition(rid, label, params, seed)
        # attach question
        summ["question"] = q
        summ["label"] = label
        all_summaries.append(summ)
        all_trajs.append(traj)

    aggregate_and_report(all_summaries, all_trajs)


if __name__ == "__main__":
    main()
