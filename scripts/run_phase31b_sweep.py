#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HARNESS = "01_CORE_SIMULATION/ultimate_simulation_optimized.py"
EPS = [0.2, 0.5, 0.8]
CCI = [0.3, 0.6, 0.9]
ETA = [0.5, 1.0, 2.0]
SEEDS = [123, 456, 789]
COLS = [
    "domain",
    "test",
    "seed",
    "epsilon",
    "cci",
    "eta",
    "inv_eta",
    "final_CCI",
    "stability_CCI_mean",
    "survival_rate",
    "stability_hazard_mean",
]


def run_one(domain, test, seed, eps, cci, eta, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        HARNESS,
        "--seed",
        str(seed),
        "--export-dir",
        str(out),
        "--modules",
        "consciousness,quantum,biology",
        "--epsilon",
        str(eps),
        "--cci",
        str(cci),
        "--eta",
        str(eta),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p = json.load(open(out / "results.json"))["phase31b"]
    return {
        "domain": domain,
        "test": test,
        "seed": seed,
        "epsilon": p["epsilon"],
        "cci": p["cci"],
        "eta": p["eta"],
        "inv_eta": p["inv_eta"],
        "final_CCI": p["final_CCI"],
        "stability_CCI_mean": p["stability_CCI_mean"],
        "survival_rate": p["survival_rate"],
        "stability_hazard_mean": p["stability_hazard_mean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--base-export", required=True)
    ap.add_argument("--run-sanitizer", action="store_true")
    a = ap.parse_args()
    base = Path(a.base_export)
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "runs_summary.csv"
    first = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if first:
            w.writeheader()
        for eps in EPS:
            for s in SEEDS:
                w.writerow(
                    run_one(
                        a.domain,
                        "TestA_vary_epsilon",
                        s,
                        eps,
                        0.6,
                        1.0,
                        base / f"A_eps{eps}_s{s}",
                    )
                )
        for c in CCI:
            for s in SEEDS:
                w.writerow(
                    run_one(
                        a.domain,
                        "TestB_vary_cci",
                        s,
                        0.5,
                        c,
                        1.0,
                        base / f"B_cci{c}_s{s}",
                    )
                )
        for e in ETA:
            for s in SEEDS:
                w.writerow(
                    run_one(
                        a.domain,
                        "TestC_vary_eta",
                        s,
                        0.5,
                        0.6,
                        e,
                        base / f"C_eta{e}_s{s}",
                    )
                )
    print("[OK] wrote", csv_path)
    if a.run_sanitizer:
        subprocess.run(
            [
                sys.executable,
                "scripts/sanitize_and_report.py",
                "--domain",
                a.domain,
                "--runs",
                str(csv_path),
                "--outdir",
                str(base),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
