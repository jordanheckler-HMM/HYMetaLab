#!/usr/bin/env python3
"""
Lifecycle Assay (LONG) — Do systems eventually collapse?
- Tracks many lifecycles up to 120 epochs (or until collapse).
- Logs per-epoch CCI / Survival / Collapse.
- Exports the average trajectory, Kaplan–Meier survival curve, and hazard over time.
- Optional: periodic "expansion pulses" to test renewal vs inevitable decline.

Outputs -> ./discovery_results/lifecycle_long_<stamp>/
"""

import datetime
import json
import pathlib
import random
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Tunables (kept fast) ----------
SEED = 2026
LIFECYCLES = 48  # number of independent runs
MAX_EPOCHS = 120  # horizon per lifecycle
POP = 18  # agents sampled per epoch (keeps runtime low)
SHOCKS = [0.2, 0.5, 0.8]

# Aging drift (slightly stronger vs short assay to allow eventual collapse)
DRIFT = dict(
    d_coord_base=-0.005, d_fair_base=-0.004, d_noise_base=+0.005, d_energy_base=-0.003
)

# Initial "bang"
INIT_BOOST = dict(energy=+0.10, coordination=+0.06, fairness=+0.05, noise=-0.02)

# OPTIONAL: renewal pulses (simulate expansion/coordination campaigns)
# Set to None to disable; or e.g. every 20 epochs, apply small boosts.
EXPANSION_PULSE_EVERY = None  # e.g., 20
PULSE = dict(energy=+0.04, coordination=+0.03, fairness=+0.03, noise=-0.01)

# Collapse boundary (early stop)
COLLAPSE_RISK_CUTOFF = 0.45
SURVIVAL_CUTOFF = 0.40

# ---------- Setup ----------
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = pathlib.Path("./discovery_results") / f"lifecycle_long_{RUN_STAMP}"
ROOT.mkdir(parents=True, exist_ok=True)

print("🕰️ Starting LONG Lifecycle Assay...")
print(f"📁 Results will be saved to: {ROOT}")
print("⏳ Extended tracking: Do consciousness systems eventually collapse?")

rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("⚙️  Extended Configuration:")
print(f"   - Lifecycles: {LIFECYCLES}")
print(f"   - Max epochs per lifecycle: {MAX_EPOCHS} (4x longer than standard)")
print(f"   - Population per epoch: {POP}")
print(
    f"   - Collapse threshold: Collapse≥{COLLAPSE_RISK_CUTOFF} & Survival≤{SURVIVAL_CUTOFF}"
)
print("   - Stronger aging drift for collapse testing")
if EXPANSION_PULSE_EVERY:
    print(f"   - Renewal pulses every {EXPANSION_PULSE_EVERY} epochs")
else:
    print("   - No renewal pulses (pure aging)")

BOUNDS = dict(
    energy=(0, 1), coordination=(0.45, 0.9), fairness=(0.30, 0.90), noise=(0.00, 0.20)
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


def assess(state: dict[str, float]) -> dict[str, float]:
    cal = clamp(
        0.74
        + 0.12 * (state["coordination"] - 0.6)
        - 0.22 * state["noise"]
        + 0.08 * (state["fairness"] - 0.5),
        0,
        1,
    )
    coh = clamp(
        0.68 + 0.22 * (state["coordination"] - 0.6) - 0.10 * abs(state["noise"] - 0.05),
        0,
        1,
    )
    emg = clamp(
        0.60 + 0.18 * (state["energy"] - 0.6) + 0.08 * (state["fairness"] - 0.5), 0, 1
    )
    cci = cci_from_components(cal, coh, emg, state["noise"])

    survs = []
    for s in SHOCKS:
        base = (
            0.58
            + 0.20 * (state["coordination"] - 0.6)
            + 0.15 * (state["energy"] - 0.6)
            - 0.26 * state["noise"]
        )
        survs.append(np.clip(base - 0.22 * (s - 0.2), 0, 1))
    survival = float(np.mean(survs))
    collapse = clamp(
        0.30
        + 0.38 * (0.6 - state["fairness"])
        - 0.18 * (state["coordination"] - 0.6)
        - 0.10 * (state["energy"] - 0.6),
        0,
        1,
    )

    return dict(
        cci=cci, survival=survival, collapse=collapse, cal=cal, coh=coh, emg=emg
    )


def new_system() -> dict[str, float]:
    st = {
        "energy": 0.62 + 0.06 * rng.standard_normal(),
        "coordination": 0.62 + 0.05 * rng.standard_normal(),
        "fairness": 0.55 + 0.05 * rng.standard_normal(),
        "noise": 0.06 + 0.02 * rng.standard_normal(),
    }
    for k, v in INIT_BOOST.items():
        st[k] = clamp(st[k] + v, *BOUNDS[k])
    return {k: clamp(v, *BOUNDS[k]) for k, v in st.items()}


def age_system(st: dict[str, float]) -> dict[str, float]:
    st["coordination"] = clamp(
        st["coordination"] + DRIFT["d_coord_base"] + 0.01 * rng.standard_normal(),
        *BOUNDS["coordination"],
    )
    st["fairness"] = clamp(
        st["fairness"] + DRIFT["d_fair_base"] + 0.01 * rng.standard_normal(),
        *BOUNDS["fairness"],
    )
    st["noise"] = clamp(
        st["noise"] + DRIFT["d_noise_base"] + 0.005 * rng.standard_normal(),
        *BOUNDS["noise"],
    )
    st["energy"] = clamp(
        st["energy"] + DRIFT["d_energy_base"] + 0.01 * rng.standard_normal(),
        *BOUNDS["energy"],
    )
    return st


def pulse_if_due(st: dict[str, float], epoch: int) -> dict[str, float]:
    if EXPANSION_PULSE_EVERY and epoch > 0 and epoch % EXPANSION_PULSE_EVERY == 0:
        for k, v in PULSE.items():
            st[k] = clamp(st[k] + v, *BOUNDS[k])
    return st


def run_lifecycle(lc_id: int) -> pd.DataFrame:
    st = new_system()
    rows = []
    collapsed_epoch = None

    for epoch in range(MAX_EPOCHS):
        # assess via small "population" sample
        ccis, survs, cols = [], [], []
        for _ in range(POP):
            m = assess(st)
            ccis.append(m["cci"])
            survs.append(m["survival"])
            cols.append(m["collapse"])
        row = dict(
            lifecycle=lc_id,
            epoch=epoch,
            energy=st["energy"],
            coordination=st["coordination"],
            fairness=st["fairness"],
            noise=st["noise"],
            cci=float(np.mean(ccis)),
            survival=float(np.mean(survs)),
            collapse=float(np.mean(cols)),
        )
        rows.append(row)

        # stop on collapse
        if (
            row["collapse"] >= COLLAPSE_RISK_CUTOFF
            and row["survival"] <= SURVIVAL_CUTOFF
        ):
            collapsed_epoch = epoch
            break

        # age + optional pulse
        st = age_system(st)
        st = pulse_if_due(st, epoch + 1)

    df = pd.DataFrame(rows)
    df["collapsed_epoch"] = collapsed_epoch
    df["reached_max"] = collapsed_epoch is None
    return df


def km_from_events(
    lengths: np.ndarray, censored: np.ndarray, max_t: int
) -> pd.DataFrame:
    """
    Quick Kaplan–Meier estimator:
    lengths : time-to-event or time-to-censor (epochs)
    censored: 1 if censored (no collapse), 0 if event (collapsed)
    """
    # counts at each time
    n_at_risk = []
    n_events = []
    surv = []
    S = 1.0
    for t in range(max_t + 1):
        at_risk = np.sum(lengths >= t + 1)
        events = np.sum((lengths == t + 1) & (censored == 0))
        n_at_risk.append(at_risk)
        n_events.append(events)
        if at_risk > 0:
            S *= 1 - events / at_risk
        surv.append(S)
    return pd.DataFrame(
        {
            "epoch": np.arange(max_t + 1),
            "n_at_risk": n_at_risk,
            "events": n_events,
            "KM_survival": surv,
        }
    )


def main():
    print(
        f"\n🕰️ Running {LIFECYCLES} extended lifecycles (up to {MAX_EPOCHS} epochs each)..."
    )

    all_runs = []
    collapse_count = 0
    survival_count = 0

    for i in range(LIFECYCLES):
        if (i + 1) % 12 == 0:
            print(f"   Completed {i + 1}/{LIFECYCLES} extended lifecycles...")

        lifecycle_df = run_lifecycle(i)
        all_runs.append(lifecycle_df)

        # Track collapse statistics
        if lifecycle_df["collapsed_epoch"].iloc[0] is not None:
            collapse_count += 1
        else:
            survival_count += 1

    print(
        f"   → {collapse_count} systems collapsed, {survival_count} survived full duration"
    )

    df = pd.concat(all_runs, ignore_index=True)
    df.to_csv(ROOT / "lifecycle_history.csv", index=False)
    print(f"📊 Raw data saved: {len(df)} total epoch measurements")

    # Average trajectory
    print("📈 Computing extended trajectory analysis...")
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

    # Endpoints for survival analysis
    ends = (
        df.sort_values(["lifecycle", "epoch"])
        .groupby("lifecycle")
        .tail(1)
        .reset_index(drop=True)
    )
    lengths = ends["epoch"].values + 1  # epochs are 0-indexed
    events = (
        (
            (ends["collapse"] >= COLLAPSE_RISK_CUTOFF)
            & (ends["survival"] <= SURVIVAL_CUTOFF)
        )
        .astype(int)
        .values
    )
    censored = 1 - events

    print("📉 Computing Kaplan-Meier survival analysis...")
    km = km_from_events(lengths, censored, max_t=int(df["epoch"].max()))
    km.to_csv(ROOT / "survival_curve.csv", index=False)

    # Hazard (Nelson–Aalen-ish): h_t ≈ events / at_risk
    km["hazard"] = np.where(km["n_at_risk"] > 0, km["events"] / km["n_at_risk"], 0.0)
    km.to_csv(ROOT / "survival_curve_with_hazard.csv", index=False)

    # Compute statistics
    avg_peak = float(df.groupby("lifecycle")["cci"].max().mean())
    avg_final = float(ends["cci"].mean())
    collapse_rate = float(events.mean())  # 0..1
    avg_lifespan = float(lengths.mean())
    final_km_survival = km["KM_survival"].iloc[-1] if len(km) > 0 else 0.0
    max_hazard = km["hazard"].max() if len(km) > 0 else 0.0

    print("\n📊 EXTENDED LIFECYCLE ANALYSIS:")
    print(f"   → Average peak CCI: {avg_peak:.3f}")
    print(f"   → Average final CCI: {avg_final:.3f}")
    print(
        f"   → CCI decline: {avg_peak - avg_final:.3f} ({((avg_peak - avg_final)/avg_peak)*100:.1f}%)"
    )
    print(f"   → Average lifespan: {avg_lifespan:.1f} epochs")
    print(f"   → Collapse rate: {collapse_rate:.1%}")
    print(f"   → Final KM survival probability: {final_km_survival:.3f}")
    print(f"   → Maximum hazard rate: {max_hazard:.4f} per epoch")

    # Plots
    print("🎨 Generating extended lifecycle visualizations...")
    try:
        # Multi-panel extended analysis
        plt.figure(figsize=(15, 10))

        # CCI trajectory
        plt.subplot(2, 3, 1)
        plt.plot(g["epoch"], g["cci_mean"], marker="o", linewidth=2, color="blue")
        plt.fill_between(
            g["epoch"],
            g["cci_mean"] - g["cci_std"],
            g["cci_mean"] + g["cci_std"],
            alpha=0.3,
            color="blue",
        )
        plt.title(f"Extended CCI Evolution ({MAX_EPOCHS} epochs)")
        plt.xlabel("Epoch")
        plt.ylabel("CCI")
        plt.grid(True, alpha=0.3)

        # Survival vs Collapse
        plt.subplot(2, 3, 2)
        plt.plot(
            g["epoch"],
            g["surv_mean"],
            label="Survival",
            marker="o",
            color="green",
            linewidth=2,
        )
        plt.plot(
            g["epoch"],
            g["col_mean"],
            label="Collapse Risk",
            marker="o",
            color="red",
            linewidth=2,
        )
        plt.axhline(
            y=SURVIVAL_CUTOFF,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="Survival Threshold",
        )
        plt.axhline(
            y=COLLAPSE_RISK_CUTOFF,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Collapse Threshold",
        )
        plt.title("Survival vs Collapse Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Kaplan–Meier survival
        plt.subplot(2, 3, 3)
        plt.step(
            km["epoch"], km["KM_survival"], where="post", linewidth=2, color="purple"
        )
        plt.title("Kaplan–Meier Survival Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Survival Probability")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        # Hazard over time
        plt.subplot(2, 3, 4)
        plt.plot(km["epoch"], km["hazard"], marker="o", color="orange", linewidth=2)
        plt.title("Collapse Hazard Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Hazard (per epoch)")
        plt.grid(True, alpha=0.3)

        # System parameters evolution
        param_g = df.groupby("epoch")[
            ["energy", "coordination", "fairness", "noise"]
        ].mean()
        plt.subplot(2, 3, 5)
        plt.plot(param_g.index, param_g["energy"], label="Energy", linewidth=2)
        plt.plot(
            param_g.index, param_g["coordination"], label="Coordination", linewidth=2
        )
        plt.plot(param_g.index, param_g["fairness"], label="Fairness", linewidth=2)
        plt.plot(param_g.index, param_g["noise"], label="Noise", linewidth=2)
        plt.title("System Parameters Evolution")
        plt.xlabel("Epoch")
        plt.ylabel("Parameter Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Lifespan distribution
        plt.subplot(2, 3, 6)
        plt.hist(lengths, bins=15, alpha=0.7, color="brown", edgecolor="black")
        plt.axvline(
            lengths.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {lengths.mean():.1f} epochs",
        )
        if collapse_count > 0:
            collapsed_lengths = lengths[events == 1]
            if len(collapsed_lengths) > 0:
                plt.axvline(
                    collapsed_lengths.mean(),
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"Collapsed Mean: {collapsed_lengths.mean():.1f}",
                )
        plt.xlabel("Lifespan (epochs)")
        plt.ylabel("Count")
        plt.title("System Lifespan Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        savefig(ROOT / "extended_lifecycle_analysis.png")

        # Individual trajectory comparison (survivors vs collapsed)
        plt.figure(figsize=(12, 6))

        # Sample survivors and collapsed systems
        survivors = ends[events == 0]["lifecycle"].values
        collapsed_systems = ends[events == 1]["lifecycle"].values

        plt.subplot(1, 2, 1)
        # Plot sample survivors
        sample_survivors = (
            np.random.choice(survivors, min(5, len(survivors)), replace=False)
            if len(survivors) > 0
            else []
        )
        for lc in sample_survivors:
            lc_data = df[df["lifecycle"] == lc]
            plt.plot(
                lc_data["epoch"], lc_data["cci"], alpha=0.7, linewidth=2, color="blue"
            )
        plt.title(f"Survivor Trajectories (n={len(survivors)})")
        plt.xlabel("Epoch")
        plt.ylabel("CCI")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Plot sample collapsed systems
        sample_collapsed = (
            np.random.choice(
                collapsed_systems, min(5, len(collapsed_systems)), replace=False
            )
            if len(collapsed_systems) > 0
            else []
        )
        for lc in sample_collapsed:
            lc_data = df[df["lifecycle"] == lc]
            plt.plot(
                lc_data["epoch"], lc_data["cci"], alpha=0.7, linewidth=2, color="red"
            )
        plt.title(f"Collapsed Trajectories (n={len(collapsed_systems)})")
        plt.xlabel("Epoch")
        plt.ylabel("CCI")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        savefig(ROOT / "survivor_vs_collapsed.png")

    except Exception as e:
        print(f"⚠️  Plot generation error: {e}")
        with open(ROOT / "plot_errors.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())

    # Summary
    summary = dict(
        run_stamp=RUN_STAMP,
        lifecycles=LIFECYCLES,
        max_epochs=MAX_EPOCHS,
        pop=POP,
        avg_peak_cci=round(avg_peak, 3),
        avg_final_cci=round(avg_final, 3),
        cci_decline=round(avg_peak - avg_final, 3),
        avg_lifespan_epochs=round(avg_lifespan, 1),
        collapse_rate=round(collapse_rate, 3),
        survival_rate=round(1 - collapse_rate, 3),
        final_km_survival=round(final_km_survival, 3),
        max_hazard_rate=round(max_hazard, 4),
        collapse_threshold_collapse=COLLAPSE_RISK_CUTOFF,
        collapse_threshold_survival=SURVIVAL_CUTOFF,
        notes="Extended timeline reveals long-term consciousness evolution patterns. Stronger aging drift tests eventual collapse potential.",
    )
    with open(ROOT / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Detailed report
    with open(ROOT / "REPORT.md", "w") as f:
        f.write(f"# Lifecycle Assay (Extended) — {RUN_STAMP}\n\n")
        f.write("## Overview\n\n")
        f.write(
            f"Extended lifecycle analysis tracking {LIFECYCLES} consciousness systems over up to {MAX_EPOCHS} epochs "
        )
        f.write(
            "to investigate long-term stability and eventual collapse potential.\n\n"
        )

        f.write("## Key Findings\n\n")
        f.write(f"- **Systems tested**: {LIFECYCLES}\n")
        f.write(f"- **Maximum duration**: {MAX_EPOCHS} epochs\n")
        f.write(f"- **Average peak CCI**: {avg_peak:.3f}\n")
        f.write(f"- **Average final CCI**: {avg_final:.3f}\n")
        f.write(
            f"- **Total CCI decline**: {avg_peak - avg_final:.3f} ({((avg_peak - avg_final)/avg_peak)*100:.1f}%)\n"
        )
        f.write(f"- **Average lifespan**: {avg_lifespan:.1f} epochs\n")
        f.write(f"- **Collapse rate**: {collapse_rate:.1%}\n")
        f.write(f"- **Final survival probability**: {final_km_survival:.3f}\n")
        f.write(f"- **Peak hazard rate**: {max_hazard:.4f} per epoch\n\n")

        f.write("## Long-term Pattern\n\n")
        if collapse_rate > 0:
            f.write(
                "Extended observation reveals that consciousness systems can eventually collapse "
            )
            f.write(
                "when subjected to continuous aging processes without renewal mechanisms.\n\n"
            )
        else:
            f.write(
                "Remarkable resilience: No systems collapsed even over extended {MAX_EPOCHS}-epoch timeline. "
            )
            f.write(
                "Consciousness appears to stabilize at reduced but viable levels.\n\n"
            )

        f.write("## Files Generated\n\n")
        f.write(
            "- `lifecycle_history.csv`: Per-epoch metrics for all extended lifecycles\n"
        )
        f.write("- `average_trajectory.csv`: Mean±std trajectory across lifecycles\n")
        f.write("- `survival_curve.csv`: Kaplan-Meier survival analysis\n")
        f.write(
            "- `survival_curve_with_hazard.csv`: Survival curve with hazard rates\n"
        )
        f.write("- `extended_lifecycle_analysis.png`: 6-panel comprehensive analysis\n")
        f.write("- `survivor_vs_collapsed.png`: Trajectory comparison by outcome\n")
        f.write("- `SUMMARY.json`: Statistical summary\n\n")

        f.write("## Methodology\n\n")
        f.write(
            f"- **Collapse criteria**: Collapse risk ≥ {COLLAPSE_RISK_CUTOFF} AND Survival ≤ {SURVIVAL_CUTOFF}\n"
        )
        f.write(
            "- **Aging parameters**: Enhanced drift rates to test long-term stability\n"
        )
        f.write(
            "- **Survival analysis**: Kaplan-Meier estimator with hazard rate calculation\n"
        )
        if EXPANSION_PULSE_EVERY:
            f.write(f"- **Renewal pulses**: Every {EXPANSION_PULSE_EVERY} epochs\n")
        else:
            f.write("- **No renewal**: Pure aging without intervention\n")

    print("\n✅ EXTENDED LIFECYCLE ASSAY COMPLETE!")
    print(f"📁 Results saved to: {ROOT}")

    # Copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory...")
    try:
        shutil.copy(ROOT / "REPORT.md", "./LIFECYCLE_LONG_RESULTS.md")
        shutil.copy(ROOT / "SUMMARY.json", "./LIFECYCLE_LONG_SUMMARY.json")
        shutil.copy(ROOT / "lifecycle_history.csv", "./LIFECYCLE_LONG_DATA.csv")

        print("✅ Easy access files created:")
        print("   - LIFECYCLE_LONG_RESULTS.md")
        print("   - LIFECYCLE_LONG_SUMMARY.json")
        print("   - LIFECYCLE_LONG_DATA.csv")
    except Exception as e:
        print(f"⚠️  File copy error: {e}")


if __name__ == "__main__":
    main()
