#!/usr/bin/env python3
"""Create a simple report directory for a study.

Usage: python -m sim.report --study truth_dynamics --out reports/TD07_08_summary
"""
import argparse
import glob
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    # copy aggregated CSVs
    src_pattern = os.path.join("analysis", args.study, "*.csv")
    for f in glob.glob(src_pattern):
        shutil.copy(f, outdir)

    # copy discovery_result figures for meaning_phase5_4 (baseline/anneal)
    fig_patterns = [
        "discovery_results/meaning_phase5_4_*/bars_meaning_by_phase_v2.png",
        "discovery_results/meaning_phase5_4_*/curves_truth_div_val_meaning_v2.png",
    ]
    for pat in fig_patterns:
        for f in glob.glob(pat):
            shutil.copy(f, outdir)

    # basic README
    readme = os.path.join(outdir, "README.md")
    with open(readme, "w") as fh:
        fh.write(f"Report for study {args.study}\n\n")
        fh.write("Contents:\n")
        for entry in sorted(os.listdir(outdir)):
            fh.write(f" - {entry}\n")

    print("Report assembled at", outdir)


if __name__ == "__main__":
    main()
