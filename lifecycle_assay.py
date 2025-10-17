#!/usr/bin/env python3
"""
Lifecycle Assay — CCI from creation → collapse
- Spawns many independent lifecycles.
- Evolves each system epoch-by-epoch under mild/mid/hard shocks with drift.
- Logs CCI, Survival, Collapse at each epoch until collapse or max epochs.
- Exports: per-epoch CSV, averaged trajectory, and quick plots.

Outputs: ./discovery_results/lifecycle_<stamp>/
"""

import datetime
import json
import pathlib
import random
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------- Config tuned for short runs --------
SEED = 777
LIFECYCLES = 32  # how many independent runs
MAX_EPOCHS = 28  # epochs per lifecycle (cap)
POP = 28  # agents sampled per epoch
SHOCKS = [0.2, 0.5, 0.8]  # mild/mid/hard
# drift controls (how systems age): expansion slows decline; stagnation/contract accelerate it
DRIFT = dict(
    d_coord_base=-0.004,  # baseline coordination drift per epoch
    d_fair_base=-0.003,
    d_noise_base=+0.004,  # entropy creeps up
    d_energy_base=-0.002,
)
# creation burst (initial "bang")
INIT_BOOST = dict(energy=+0.10, coordination=+0.06, fairness=+0.04, noise=-0.02)

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = pathlib.Path("./discovery_results") / f"lifecycle_{RUN_STAMP}"
ROOT.mkdir(parents=True, exist_ok=True)

print("🔄 Starting Lifecycle Assay...")
print(f"📁 Results will be saved to: {ROOT}")
print("🧬 Tracking consciousness systems from birth to collapse...")

rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("⚙️  Configuration:")
print(f"   - Lifecycles: {LIFECYCLES}")
print(f"   - Max epochs per lifecycle: {MAX_EPOCHS}")
print(f"   - Population per epoch: {POP}")
print(f"   - Shock levels: {SHOCKS}")

# -------- Helpers --------
BOUNDS = dict(
    energy=(0, 1), coordination=(0.45, 0.9), fairness=(0.3, 0.9), noise=(0.0, 0.20)
)


def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))


def savefig(p):
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()


def cci_from_components(calibration, coherence, emergence, noise):
    raw = (
        max(0, calibration)
        * max(0, coherence)
        * max(0, emergence)
        / max(1e-6, noise + 0.10)
    )
    return raw / (1.0 + raw)


def assess(system: dict[str, float]) -> dict[str, float]:
    # Map system state → CCI components
    cal = clamp(
        0.74
        + 0.12 * (system["coordination"] - 0.6)
        - 0.20 * system["noise"]
        + 0.08 * (system["fairness"] - 0.5),
        0,
        1,
    )
    coh = clamp(
        0.68
        + 0.22 * (system["coordination"] - 0.6)
        - 0.10 * abs(system["noise"] - 0.05),
        0,
        1,
    )
    emg = clamp(
        0.60 + 0.18 * (system["energy"] - 0.6) + 0.08 * (system["fairness"] - 0.5), 0, 1
    )
    cci = cci_from_components(cal, coh, emg, system["noise"])
    # Survival across shocks (mean) + collapse risk proxy
    survs = []
    for s in SHOCKS:
        base = (
            0.58
            + 0.20 * (system["coordination"] - 0.6)
            + 0.15 * (system["energy"] - 0.6)
            - 0.25 * system["noise"]
        )
        survs.append(np.clip(base - 0.22 * (s - 0.2), 0, 1))
    survival = float(np.mean(survs))
    collapse = clamp(
        0.30
        + 0.35 * (0.6 - system["fairness"])
        - 0.18 * (system["coordination"] - 0.6)
        - 0.10 * (system["energy"] - 0.6),
        0,
        1,
    )
    return dict(
        cci=cci, survival=survival, collapse=collapse, cal=cal, coh=coh, emg=emg
    )


def new_system() -> dict[str, float]:
    # baseline
    sys = {
        "energy": 0.62 + 0.06 * rng.standard_normal(),
        "coordination": 0.62 + 0.05 * rng.standard_normal(),
        "fairness": 0.55 + 0.05 * rng.standard_normal(),
        "noise": 0.06 + 0.02 * rng.standard_normal(),
    }
    # initial "bang"
    for k, v in INIT_BOOST.items():
        sys[k] = clamp(sys[k] + v, *BOUNDS[k])
    return {k: clamp(v, *BOUNDS[k]) for k, v in sys.items()}


def age_system(sys: dict[str, float]) -> dict[str, float]:
    # baseline drift
    sys["coordination"] = clamp(
        sys["coordination"] + DRIFT["d_coord_base"] + 0.01 * rng.standard_normal(),
        *BOUNDS["coordination"],
    )
    sys["fairness"] = clamp(
        sys["fairness"] + DRIFT["d_fair_base"] + 0.01 * rng.standard_normal(),
        *BOUNDS["fairness"],
    )
    sys["noise"] = clamp(
        sys["noise"] + DRIFT["d_noise_base"] + 0.005 * rng.standard_normal(),
        *BOUNDS["noise"],
    )
    sys["energy"] = clamp(
        sys["energy"] + DRIFT["d_energy_base"] + 0.01 * rng.standard_normal(),
        *BOUNDS["energy"],
    )
    return sys


def run_lifecycle(lc_id: int) -> pd.DataFrame:
    sys = new_system()
    rows = []
    collapsed_epoch = None

    for epoch in range(MAX_EPOCHS):
        # "population" sampling adds a bit of variance to metrics each epoch
        ccis, survs, cols = [], [], []
        for _ in range(POP):
            m = assess(sys)
            ccis.append(m["cci"])
            survs.append(m["survival"])
            cols.append(m["collapse"])
        row = dict(
            lifecycle=lc_id,
            epoch=epoch,
            energy=sys["energy"],
            coordination=sys["coordination"],
            fairness=sys["fairness"],
            noise=sys["noise"],
            cci=np.mean(ccis),
            survival=np.mean(survs),
            collapse=np.mean(cols),
        )
        rows.append(row)

        # stop if collapsed (risk high & survival low)
        if row["collapse"] >= 0.40 and row["survival"] <= 0.45:
            collapsed_epoch = epoch
            break
        # age to next epoch
        sys = age_system(sys)

    df = pd.DataFrame(rows)
    df["collapsed_at"] = collapsed_epoch
    df["max_epochs_reached"] = collapsed_epoch is None
    return df


def main():
    print(f"\n🔄 Running {LIFECYCLES} independent consciousness lifecycles...")

    all_runs = []
    collapse_count = 0
    survival_count = 0

    for i in range(LIFECYCLES):
        if (i + 1) % 8 == 0:
            print(f"   Completed {i + 1}/{LIFECYCLES} lifecycles...")

        lifecycle_df = run_lifecycle(i)
        all_runs.append(lifecycle_df)

        # Track collapse statistics
        if lifecycle_df["collapsed_at"].iloc[0] is not None:
            collapse_count += 1
        else:
            survival_count += 1

    print(
        f"   → {collapse_count} systems collapsed, {survival_count} survived to max epochs"
    )

    df = pd.concat(all_runs, ignore_index=True)
    out = ROOT / "lifecycle_history.csv"
    df.to_csv(out, index=False)
    print(f"📊 Raw data saved: {len(df)} total epoch measurements")

    # Build the average trajectory (align by epoch index)
    print("📈 Computing average trajectory across all lifecycles...")
    g = (
        df.groupby("epoch")[["cci", "survival", "collapse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    g.columns = [
        "epoch",
        "cci_mean",
        "cci_std",
        "surv_mean",
        "surv_std",
        "col_mean",
        "col_std",
    ]
    g.to_csv(ROOT / "average_trajectory.csv", index=False)

    # Compute lifecycle statistics
    lifecycle_stats = (
        df.groupby("lifecycle")
        .agg(
            {
                "cci": ["max", "min", lambda x: x.iloc[-1]],  # peak, trough, final
                "epoch": "max",  # length
                "collapsed_at": "first",
            }
        )
        .reset_index()
    )

    # Flatten column names
    lifecycle_stats.columns = [
        "lifecycle",
        "peak_cci",
        "min_cci",
        "final_cci",
        "length_epochs",
        "collapsed_at",
    ]
    lifecycle_stats["length_epochs"] += 1  # convert max epoch index to length
    lifecycle_stats.to_csv(ROOT / "lifecycle_stats.csv", index=False)

    # Quick plots
    print("🎨 Generating visualizations...")
    try:
        # CCI trajectory with confidence bands
        plt.figure(figsize=(12, 8))

        # Main trajectory
        plt.subplot(2, 2, 1)
        plt.plot(
            g["epoch"],
            g["cci_mean"],
            marker="o",
            linewidth=2,
            label="Mean CCI",
            color="blue",
        )
        plt.fill_between(
            g["epoch"],
            g["cci_mean"] - g["cci_std"],
            g["cci_mean"] + g["cci_std"],
            alpha=0.3,
            color="blue",
            label="±1 SD",
        )
        plt.title("CCI Evolution Across Lifecycle")
        plt.xlabel("Epoch")
        plt.ylabel("Consciousness Calibration Index")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Survival & Collapse
        plt.subplot(2, 2, 2)
        plt.plot(
            g["epoch"],
            g["surv_mean"],
            marker="o",
            label="Survival",
            color="green",
            linewidth=2,
        )
        plt.plot(
            g["epoch"],
            g["col_mean"],
            marker="o",
            label="Collapse Risk",
            color="red",
            linewidth=2,
        )
        plt.fill_between(
            g["epoch"],
            g["surv_mean"] - g["surv_std"],
            g["surv_mean"] + g["surv_std"],
            alpha=0.2,
            color="green",
        )
        plt.fill_between(
            g["epoch"],
            g["col_mean"] - g["col_std"],
            g["col_mean"] + g["col_std"],
            alpha=0.2,
            color="red",
        )
        plt.title("Survival vs Collapse Risk")
        plt.xlabel("Epoch")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Phase portrait: CCI vs (1-Collapse)
        plt.subplot(2, 2, 3)
        # Color by epoch to show temporal evolution
        scatter = plt.scatter(
            df["cci"],
            1.0 - df["collapse"],
            c=df["epoch"],
            s=15,
            alpha=0.6,
            cmap="viridis",
        )
        plt.colorbar(scatter, label="Epoch")
        plt.xlabel("CCI")
        plt.ylabel("Stability (1 - Collapse Risk)")
        plt.title("Phase Space: CCI vs Stability")
        plt.grid(True, alpha=0.3)

        # Lifecycle length histogram
        plt.subplot(2, 2, 4)
        lengths = lifecycle_stats["length_epochs"]
        plt.hist(lengths, bins=10, alpha=0.7, color="purple", edgecolor="black")
        plt.axvline(
            lengths.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {lengths.mean():.1f} epochs",
        )
        plt.xlabel("Lifecycle Length (epochs)")
        plt.ylabel("Number of Systems")
        plt.title("Distribution of System Lifespans")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        savefig(ROOT / "lifecycle_analysis.png")

        # Individual trajectories plot
        plt.figure(figsize=(12, 6))

        # Sample a few individual trajectories for visualization
        sample_lifecycles = np.random.choice(
            df["lifecycle"].unique(),
            min(8, len(df["lifecycle"].unique())),
            replace=False,
        )

        plt.subplot(1, 2, 1)
        for lc in sample_lifecycles:
            lc_data = df[df["lifecycle"] == lc]
            alpha = 0.7 if lc_data["collapsed_at"].iloc[0] is not None else 0.4
            linestyle = "-" if lc_data["collapsed_at"].iloc[0] is not None else "--"
            plt.plot(
                lc_data["epoch"],
                lc_data["cci"],
                alpha=alpha,
                linestyle=linestyle,
                linewidth=1.5,
            )

        plt.plot(g["epoch"], g["cci_mean"], "k-", linewidth=3, label="Average")
        plt.title("Sample Individual CCI Trajectories")
        plt.xlabel("Epoch")
        plt.ylabel("CCI")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Peak vs Final CCI
        plt.subplot(1, 2, 2)
        collapsed_mask = lifecycle_stats["collapsed_at"].notna()

        plt.scatter(
            lifecycle_stats[collapsed_mask]["peak_cci"],
            lifecycle_stats[collapsed_mask]["final_cci"],
            c="red",
            alpha=0.7,
            s=60,
            label="Collapsed",
        )
        plt.scatter(
            lifecycle_stats[~collapsed_mask]["peak_cci"],
            lifecycle_stats[~collapsed_mask]["final_cci"],
            c="blue",
            alpha=0.7,
            s=60,
            label="Survived",
        )

        # Add diagonal line
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Peak = Final")
        plt.xlabel("Peak CCI")
        plt.ylabel("Final CCI")
        plt.title("Peak vs Final Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        savefig(ROOT / "individual_trajectories.png")

    except Exception as e:
        print(f"⚠️  Plot generation error: {e}")
        with open(ROOT / "plot_errors.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())

    # Summary statistics
    peak = float(lifecycle_stats["peak_cci"].mean())
    end_cci = float(lifecycle_stats["final_cci"].mean())
    avg_length = float(lifecycle_stats["length_epochs"].mean())
    collapse_rate = collapse_count / LIFECYCLES

    summary = dict(
        run_stamp=RUN_STAMP,
        lifecycles=LIFECYCLES,
        max_epochs=MAX_EPOCHS,
        avg_peak_cci=round(peak, 3),
        avg_final_cci=round(end_cci, 3),
        avg_length_epochs=round(avg_length, 1),
        collapse_rate=round(collapse_rate, 3),
        survival_rate=round(1 - collapse_rate, 3),
        cci_decline=round(peak - end_cci, 3),
        notes="Lifecycle pattern: creation burst → peak CCI → gradual decline due to entropy, coordination decay, and fairness erosion.",
    )

    with open(ROOT / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n📊 LIFECYCLE ANALYSIS COMPLETE:")
    print(f"   → Average peak CCI: {peak:.3f}")
    print(f"   → Average final CCI: {end_cci:.3f}")
    print(
        f"   → CCI decline: {peak - end_cci:.3f} ({((peak - end_cci)/peak)*100:.1f}%)"
    )
    print(f"   → Average lifespan: {avg_length:.1f} epochs")
    print(f"   → Collapse rate: {collapse_rate:.1%}")
    print(f"   → Survival rate: {(1-collapse_rate):.1%}")

    # Report
    with open(ROOT / "REPORT.md", "w") as f:
        f.write(f"# Lifecycle Assay — {RUN_STAMP}\n\n")
        f.write("## Overview\n\n")
        f.write(
            "This experiment tracked consciousness systems through their complete lifecycle from creation to collapse (or maximum epochs).\n\n"
        )

        f.write("## Key Findings\n\n")
        f.write(f"- **Systems tested**: {LIFECYCLES}\n")
        f.write(f"- **Average peak CCI**: {peak:.3f}\n")
        f.write(f"- **Average final CCI**: {end_cci:.3f}\n")
        f.write(
            f"- **CCI decline**: {peak - end_cci:.3f} ({((peak - end_cci)/peak)*100:.1f}% degradation)\n"
        )
        f.write(f"- **Average lifespan**: {avg_length:.1f} epochs\n")
        f.write(f"- **Collapse rate**: {collapse_rate:.1%}\n")
        f.write(f"- **Survival rate**: {(1-collapse_rate):.1%}\n\n")

        f.write("## Lifecycle Pattern\n\n")
        f.write(
            "1. **Creation Burst**: Initial boost in energy, coordination, fairness\n"
        )
        f.write("2. **Peak Performance**: CCI reaches maximum in early epochs\n")
        f.write(
            "3. **Gradual Decline**: Entropy increases, coordination/fairness decay\n"
        )
        f.write(
            "4. **Collapse or Stabilization**: System either fails or reaches equilibrium\n\n"
        )

        f.write("## Files Generated\n\n")
        f.write("- `lifecycle_history.csv`: Per-epoch metrics for all systems\n")
        f.write("- `average_trajectory.csv`: Mean±std trajectory across lifecycles\n")
        f.write("- `lifecycle_stats.csv`: Peak, final, length statistics per system\n")
        f.write("- `lifecycle_analysis.png`: Comprehensive 4-panel analysis\n")
        f.write(
            "- `individual_trajectories.png`: Sample individual paths + peak vs final\n"
        )
        f.write("- `SUMMARY.json`: Key statistical results\n")

    print("\n✅ LIFECYCLE ASSAY COMPLETE!")
    print(f"📁 Results saved to: {ROOT}")

    # Copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory...")
    shutil.copy(ROOT / "REPORT.md", "./LIFECYCLE_RESULTS.md")
    shutil.copy(ROOT / "SUMMARY.json", "./LIFECYCLE_SUMMARY.json")
    shutil.copy(ROOT / "lifecycle_history.csv", "./LIFECYCLE_DATA.csv")

    print("✅ Easy access files created:")
    print("   - LIFECYCLE_RESULTS.md")
    print("   - LIFECYCLE_SUMMARY.json")
    print("   - LIFECYCLE_DATA.csv")


if __name__ == "__main__":
    main()
