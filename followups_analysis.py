#!/usr/bin/env python3
# followups_analysis.py
# Aggregates results for HS_Mortality, ShockRecovery, and Misinformation followups.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISC = Path("./discovery_results")
OUT = Path("./04_LATEST_RESULTS")
OUT.mkdir(exist_ok=True)


def load_matching(label_like):
    # scan all discovery result folders and select rows where results.csv.label contains label_like
    dfs = []
    for p in DISC.iterdir():
        f = p / "results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if (
            "label" in df.columns
            and df["label"].astype(str).str.contains(label_like).any()
        ):
            df["run_folder"] = p.name
            dfs.append(df[df["label"].astype(str).str.contains(label_like)])
    if not dfs:
        print("No matches for", label_like)
        return None
    return pd.concat(dfs, ignore_index=True)


def bootstrap_ci(series, n=1000, alpha=0.05):
    arr = np.array(series.dropna())
    if len(arr) == 0:
        return (np.nan, np.nan)
    boots = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n)
    ]
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return (lo, hi)


def analyze_hs_mortality(label_like="HS_Mortality"):
    df = load_matching(label_like)
    if df is None:
        return
    # compute per-branch summaries with CI across runs
    groups = df.groupby("branch_selected")
    rows = []
    for name, g in groups:
        survival_mean = g.survival_rate.mean()
        survival_ci = bootstrap_ci(g.survival_rate)
        collapse_mean = g.collapse_flag.mean() if "collapse_flag" in g.columns else 0.0
        lifespan_mean = (
            g.lifespan_epochs.mean() if "lifespan_epochs" in g.columns else np.nan
        )
        rows.append(
            {
                "branch": name,
                "survival_mean": survival_mean,
                "survival_ci_lo": survival_ci[0],
                "survival_ci_hi": survival_ci[1],
                "collapse_rate": collapse_mean,
                "mean_lifespan": lifespan_mean,
            }
        )
    out = pd.DataFrame(rows).sort_values("branch")
    out_path = OUT / f"{label_like}_summary.csv"
    out.to_csv(out_path, index=False)

    # simple plot: survival mean with CI
    plt.figure(figsize=(6, 4))
    x = out["branch"]
    y = out["survival_mean"]
    ylo = out["survival_mean"] - out["survival_ci_lo"]
    yhi = out["survival_ci_hi"] - out["survival_mean"]
    plt.bar(x, y, yerr=[ylo.fillna(0), yhi.fillna(0)], capsize=5)
    plt.title("Survival Rate by Branch (HS Mortality)")
    plt.ylabel("survival_rate")
    plt.tight_layout()
    plt.savefig(OUT / f"{label_like}_survival_by_branch.png")
    plt.close()

    # write report
    with open(OUT / f"{label_like}_report.md", "w") as f:
        f.write(f"# HS Mortality Summary\n\nFound {len(df)} rows across runs.\n\n")
        try:
            table_text = out.to_markdown()
        except Exception:
            try:
                table_text = out.to_string()
            except Exception:
                table_text = "<unable to render table>"
        f.write(table_text)
    print("HS mortality analysis written to", OUT)
    return out


def analyze_ab_compare(label_patterns, mapping=None, out_name="AB_compare"):
    # label_patterns: list of substrings to match in the 'label' column
    dfs = []
    for p in DISC.iterdir():
        f = p / "results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "label" not in df.columns:
            continue
        # keep rows matching any pattern
        mask = False
        for pat in label_patterns:
            mask = mask | df["label"].astype(str).str.contains(pat)
        if mask.any():
            sel = df[mask].copy()
            sel["run_folder"] = p.name
            dfs.append(sel)
    if not dfs:
        print("No matches for AB patterns", label_patterns)
        return None
    all_df = pd.concat(dfs, ignore_index=True)

    # derive enabled_flag from label convention (EN vs DIS)
    def derive_flag(label):
        s = str(label)
        if "_EN_" in s or s.endswith("_EN") or "_EN" in s:
            return "EN"
        if "_DIS_" in s or s.endswith("_DIS") or "_DIS" in s:
            return "DIS"
        # fallback: check for EN/DIS substrings anywhere
        if "EN_" in s or "EN" in s:
            return "EN"
        return "DIS"

    all_df["enabled_flag"] = all_df["label"].apply(derive_flag)

    # group and compute metrics & bootstrap CI
    groups = all_df.groupby("enabled_flag")
    rows = []
    for name, g in groups:
        row = {"enabled_flag": name}
        # include common metrics if present
        for col in [
            "survival_rate",
            "collapse_flag",
            "lifespan_epochs",
            "hope_index",
            "collective_cci_delta",
        ]:
            if col in g.columns:
                vals = g[col]
                row[f"{col}_mean"] = float(vals.mean())
                lo, hi = bootstrap_ci(vals)
                row[f"{col}_ci_lo"] = lo
                row[f"{col}_ci_hi"] = hi
            else:
                row[f"{col}_mean"] = None
                row[f"{col}_ci_lo"] = None
                row[f"{col}_ci_hi"] = None
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("enabled_flag")
    out_path = OUT / f"{out_name}_summary.csv"
    out.to_csv(out_path, index=False)

    # simple plots: survival rate comparison
    plt.figure(figsize=(6, 4))
    x = out["enabled_flag"]
    y = out["survival_rate_mean"]
    yerr_lo = out["survival_rate_mean"] - out["survival_rate_ci_lo"]
    yerr_hi = out["survival_rate_ci_hi"] - out["survival_rate_mean"]
    plt.bar(x, y, yerr=[yerr_lo.fillna(0), yerr_hi.fillna(0)], capsize=5)
    plt.title("Survival Rate: Enabled vs Disabled")
    plt.ylabel("survival_rate")
    plt.tight_layout()
    plt.savefig(OUT / f"{out_name}_survival_compare.png")
    plt.close()

    # write report
    with open(OUT / f"{out_name}_report.md", "w") as f:
        f.write(
            f"# A/B Enabled vs Disabled Summary\n\nFound {len(all_df)} rows across runs.\n\n"
        )
        try:
            table_text = out.to_markdown()
        except Exception:
            try:
                table_text = out.to_string()
            except Exception:
                table_text = "<unable to render table>"
        f.write(table_text)
    print("A/B analysis written to", OUT)
    return out


if __name__ == "__main__":
    analyze_hs_mortality()
