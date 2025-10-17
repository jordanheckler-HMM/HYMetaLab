import json
import os

import matplotlib.pyplot as plt


def plot_calibration(summary_json: str, out_png: str):
    with open(summary_json) as f:
        data = json.load(f)
    calib = data["calibration"]
    bins = [c["bin"] for c in calib if c["n"] > 0]
    avg_rep = [c["avg_reported"] for c in calib if c["n"] > 0]
    empirical = [c["empirical"] for c in calib if c["n"] > 0]
    ci_lower = [c.get("ci_lower") for c in calib if c["n"] > 0]
    ci_upper = [c.get("ci_upper") for c in calib if c["n"] > 0]

    plt.figure()
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(avg_rep, empirical, marker="o")
    # draw error bars if present
    if any(x is not None for x in ci_lower):
        lower = [e - l if l is not None else 0 for e, l in zip(empirical, ci_lower)]
        upper = [u - e if u is not None else 0 for e, u in zip(empirical, ci_upper)]
        yerr = [lower, upper]
        plt.errorbar(
            avg_rep, empirical, yerr=yerr, fmt="none", ecolor="gray", alpha=0.6
        )
    plt.xlabel("Reported probability")
    plt.ylabel("Empirical frequency")
    plt.title("Calibration curve")
    plt.grid(True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    print("Saved calibration plot to", out_png)
