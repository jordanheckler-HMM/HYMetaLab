#!/usr/bin/env python3
# religion_followups.py
# Driver for the follow-up sweeps for the religion hope mechanism experiment.

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

from meaning_experiment import analyze_results, run_experiment_grid

ROOT = Path(".")
DISC = Path("./discovery_results")


# Helper to find outputs by label pattern
def find_runs(label_like):
    pats = sorted(DISC.glob(f"{label_like}*"))
    return pats


# Wrapper to run a sweep and return df
def run_sweep(label, **kwargs):
    print(f"Running sweep: {label}")
    df = run_experiment_grid(
        label=label,
        agents_list=kwargs.get("agents", [100, 200]),
        shocks=kwargs.get("shocks", [0.5]),
        stress_duration=kwargs.get("stress_duration", ["acute"]),
        goal_diversity=kwargs.get("goal_diversity", [2, 3, 4, 5]),
        noise_list=kwargs.get("noise", [0.05, 0.1]),
        replicates=kwargs.get("replicates", 6),
        export=True,
    )
    return df


# Small analysis helper to aggregate matching runs
def aggregate_and_analyze(
    where,
    compare,
    metrics,
    ci=False,
    seeds=None,
    report=None,
    extra_plots=None,
    bundle=False,
    bundle_name=None,
):
    # find matching runs on disk
    if where.get("label_like"):
        pat = f"{where['label_like']}*"
        matches = list(DISC.glob(pat))
    elif where.get("label_in"):
        matches = []
        for li in where["label_in"]:
            matches += list(DISC.glob(f"{li}*"))
    else:
        matches = []
    if not matches:
        print("No matching runs found for", where)
        return None
    # collect CSVs
    dfs = []
    for p in matches:
        csv = Path(p) / "results.csv"
        if csv.exists():
            dfs.append(pd.read_csv(csv))
    if not dfs:
        print("No result CSVs found in matches")
        return None
    all_df = pd.concat(dfs, ignore_index=True)
    # run analyze_results which writes per-run md and branch comparisons
    rep_path = report or (Path(matches[0]) / f"aggregated_{compare}.md")
    comp = analyze_results(all_df, out_report=rep_path)

    # optional bundle
    if bundle:
        bundle_root = Path(f"04_LATEST_RESULTS/{bundle_name}")
        bundle_root.mkdir(parents=True, exist_ok=True)
        # copy data and figures
        for p in matches:
            dst = bundle_root / p.name
            if dst.exists():
                continue
            # zip the run dir
            zfn = bundle_root / f"{p.name}.zip"
            with zipfile.ZipFile(zfn, "w") as z:
                for f in p.rglob("*"):
                    z.write(f, arcname=str(f.relative_to(p)))
        # create sha256s
        with open(bundle_root / "SHA256SUMS.txt", "w") as out:
            for z in bundle_root.glob("*.zip"):
                h = hashlib.sha256(z.read_bytes()).hexdigest()
                out.write(f"{h}  {z.name}\n")
        print("Bundle created at", bundle_root)
    return comp


if __name__ == "__main__":
    # Example: run HS mortality for seeds 101..106 - here we run only seed 101 as a quick smoke test
    SEEDS = [101]
    for seed in SEEDS:
        df = run_sweep(
            label=f"HS_Mortality_seed{seed}",
            agents=[100, 200],
            goal_diversity=[2, 3, 4, 5],
            shocks=[0.9, 1.0],
            stress_duration=["acute"],
            noise=[0.05, 0.1],
            replicates=3,
            mortality_model={
                "enabled": True,
                "fatality_base": 0.02,
                "k_shock": 0.25,
                "collapse_threshold": 0.45,
            },
        )
        print("HS mortality quick test done; results in discovery_results")

    # Example aggregate
    aggregate_and_analyze(
        where={"label_like": "HS_Mortality_seed"},
        compare="branch_selected",
        metrics=["survival_rate", "collapse_flag", "lifespan_epochs"],
        report="followups_highshock.md",
        bundle=False,
    )
