#!/usr/bin/env python3
"""Simple analyzer to compare variant summary CSVs.

Usage: python -m sim.analyze --study truth_dynamics --compare baseline_v54,anneal_v1,anneal_v2,anneal_adaptive
It will search common output locations for files named like summary_*.csv and merge them.
"""
import argparse
import glob
import os

import pandas as pd


def find_summary(variant: str) -> list[str]:
    patterns = [
        f"discovery_results/**/summary_{variant}*.csv",
        f"outputs/**/summary_{variant}*.csv",
        f"discovery_results/**/{variant}*/summary*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


def load_first_summary(variant: str):
    files = find_summary(variant)
    if not files:
        return None, None
    # prefer a summary_{variant}.csv exact match
    for f in files:
        base = os.path.basename(f)
        if base.startswith(f"summary_{variant}"):
            return f, pd.read_csv(f)
    return files[0], pd.read_csv(files[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--compare", required=True, help="comma-separated variant names"
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.compare.split(",") if v.strip()]
    rows = []
    found = {}
    for v in variants:
        path, df = load_first_summary(v)
        if df is None:
            print(f"Warning: no summary found for variant '{v}'")
            continue
        found[v] = path
        # flatten if single-row
        if df.shape[0] == 1:
            row = df.iloc[0].to_dict()
            row["_variant"] = v
            rows.append(row)
        else:
            # include up to 3 rows if multi-row summary (add variant tag)
            for i in range(min(3, df.shape[0])):
                base = df.iloc[i].to_dict()
                row = {k: val for k, val in base.items()}
                row["_variant"] = v
                rows.append(row)

    if not rows:
        print("No data to aggregate")
        return

    agg = pd.DataFrame(rows)
    outdir = os.path.join("analysis", args.study)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'compare_{"_".join(variants)}.csv')
    agg.to_csv(outpath, index=False)
    print("Wrote aggregated comparison to", outpath)
    for v, p in found.items():
        print(f"{v}: {p}")


if __name__ == "__main__":
    main()
