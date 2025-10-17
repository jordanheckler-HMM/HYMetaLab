#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt


def export_results(df, output_dir, plots=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # save dataframe
    df.to_csv(out / "results.csv", index=False)
    # simple plots
    if plots:
        for p in plots:
            if p == "cci_vs_time" and "cci_mean" in df.columns:
                plt.figure(figsize=(6, 4))
                for mode, sub in df.groupby("mode"):
                    plt.plot(sub["agents"], sub["cci_mean"], marker="o", label=mode)
                plt.xlabel("agents")
                plt.ylabel("cci_mean")
                plt.title("CCI mean by agents and mode")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out / "plot_cci_vs_time.png", dpi=200)
                plt.close()
            if p == "coherence_map" and "gravity_coherence" in df.columns:
                plt.figure(figsize=(6, 4))
                for mode, sub in df.groupby("mode"):
                    plt.plot(
                        sub["agents"], sub["gravity_coherence"], marker="x", label=mode
                    )
                plt.xlabel("agents")
                plt.ylabel("gravity_coherence")
                plt.title("Gravity coherence by agents and mode")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out / "plot_coherence_map.png", dpi=200)
                plt.close()
            if p == "collapse_vs_mode" and "collapse_risk" in df.columns:
                plt.figure(figsize=(6, 4))
                means = df.groupby("mode")["collapse_risk"].mean()
                means.plot(kind="bar")
                plt.ylabel("avg collapse_risk")
                plt.title("Collapse risk by time mode")
                plt.tight_layout()
                plt.savefig(out / "plot_collapse_vs_mode.png", dpi=200)
                plt.close()
    return {"csv": str(out / "results.csv")}
