import json
import os

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = "discovery_results/cci_tracking"
FIG_PATH = os.path.join("figures", "cci_trend.png")
RPT_PATH = os.path.join("reports", "CCI_Tracking_v1.0_report.md")

df = pd.read_csv(os.path.join(OUT_DIR, "runs_summary.csv"))
with open(os.path.join(OUT_DIR, "bootstrap_ci.json")) as f:
    summ = json.load(f)

plt.figure()
plt.plot(df["seed"], df["CCI_norm"], marker="o")
plt.title("CCI (normalized) by Seed")
plt.xlabel("Seed")
plt.ylabel("CCI_norm")
plt.grid(True, alpha=0.3)
plt.savefig(FIG_PATH, bbox_inches="tight")
plt.close()

with open(RPT_PATH, "w") as f:
    f.write("# CCI Tracking v1.0 — Report\n\n")
    f.write("**Summary:**\n\n")
    f.write(f"- CCI_mean: {summ['CCI_mean']:.4f}\n")
    f.write(f"- 95% CI: [{summ['CI_lower']:.4f}, {summ['CI_upper']:.4f}]\n")
    f.write(f"- sigma_run: {summ['sigma_run']:.4f}\n")
    f.write(f"- n_runs: {summ['n_runs']}\n\n")
    f.write("## Seed runs\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n\n## Figure\n\n![CCI Trend](../figures/cci_trend.png)\n")
print("Quick report written to", RPT_PATH, "and figure at", FIG_PATH)
