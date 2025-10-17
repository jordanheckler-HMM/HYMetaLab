#!/usr/bin/env python3
"""
Baby-Universe Lab (creation + travel mechanisms) — time-boxed (≤ ~10 min)
Tracks:
  A) CREATION: spawn child universes with inheritance/mutation; assess child stability & parent back-reaction.
  B) TRAVEL: simulate four mechanisms (WH, DW, FVB, IO) with synthetic physics-aware proxies.

Outputs: ./discovery_results/baby_universe_<stamp>/
  - creation_history.csv / creation_summary.json
  - travel_history.csv   / travel_summary.json
  - bars_creation.png, bars_travel.png, pareto_travel.png
  - REPORT.md
"""

import datetime
import json
import pathlib
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 2026
TIME_BUDGET_MIN = 10
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = pathlib.Path("./discovery_results") / f"baby_universe_{RUN_STAMP}"
ROOT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("👶 Starting Baby-Universe Lab...")
print(f"📁 Results will be saved to: {ROOT}")
print("🔬 Investigating universe creation and interdimensional travel mechanisms...")


# ---------- Helpers ----------
def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))


def norm01(x):
    return float(np.clip(x, 0, 1))


def savefig(path):
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# Parent "state" (map your modules here if available)
BASE_PARENT = dict(
    energy=0.72,  # external energy/space available (expansion proxy)
    coordination=0.66,  # rules/institutions alignment
    fairness=0.60,  # resource distribution (low inequality)
    noise=0.06,  # randomness/entropy
)
BOUNDS = dict(
    energy=(0, 1), coordination=(0.4, 0.9), fairness=(0.3, 0.9), noise=(0, 0.2)
)

print(
    f"🌌 Parent universe baseline: Energy={BASE_PARENT['energy']:.2f}, Coord={BASE_PARENT['coordination']:.2f}, Fair={BASE_PARENT['fairness']:.2f}, Noise={BASE_PARENT['noise']:.2f}"
)


def cci_est(calibration, coherence, emergence, noise):
    raw = (
        max(0, calibration)
        * max(0, coherence)
        * max(0, emergence)
        / max(1e-6, noise + 0.10)
    )
    return raw / (1.0 + raw)


# ---------- A) CREATION ----------
def spawn_child(parent: dict[str, float], inherit=0.7, mut=0.08) -> dict[str, float]:
    child = {}
    for k in ["energy", "coordination", "fairness", "noise"]:
        base = inherit * parent[k] + (1 - inherit) * (0.5 + 0.3 * np.random.randn())
        if k == "noise":
            base = base + np.random.normal(0, mut * 0.5)
        else:
            base = base + np.random.normal(0, mut)
        child[k] = clamp(base, *BOUNDS[k])
    return child


def assess_universe(state: dict[str, float]) -> dict[str, float]:
    # Map to CCI components
    calibration = clamp(
        0.75 + 0.10 * (state["fairness"] - 0.5) - 0.10 * state["noise"], 0, 1
    )
    coherence = clamp(
        0.70 + 0.20 * (state["coordination"] - 0.6) - 0.10 * abs(state["noise"] - 0.05),
        0,
        1,
    )
    emergence = clamp(
        0.60 + 0.20 * (state["energy"] - 0.6) + 0.10 * (state["fairness"] - 0.5), 0, 1
    )
    cci = cci_est(calibration, coherence, emergence, state["noise"])
    # Survival & collapse heuristics
    survival = clamp(
        0.55
        + 0.20 * (state["coordination"] - 0.6)
        + 0.15 * (state["energy"] - 0.6)
        - 0.25 * state["noise"],
        0,
        1,
    )
    collapse = clamp(
        0.30
        + 0.35 * (0.6 - state["fairness"])
        - 0.20 * (state["coordination"] - 0.6)
        - 0.10 * (state["energy"] - 0.6),
        0,
        1,
    )
    return dict(cci=cci, survival=survival, collapse=collapse)


def creation_experiment(
    parent=BASE_PARENT, N=120, inherit=0.7, mut=0.08, cost_factor=0.08
):
    print("\n🧬 UNIVERSE CREATION EXPERIMENT")
    print(f"   Spawning {N} child universes...")
    print(f"   Inheritance factor: {inherit:.1%}")
    print(f"   Mutation rate: {mut:.1%}")
    print(f"   Parent cost factor: {cost_factor:.1%}")

    rows = []
    successful_children = 0

    for i in range(N):
        if (i + 1) % 30 == 0:
            print(f"     Created {i+1}/{N} child universes...")

        child = spawn_child(parent, inherit, mut)
        m = assess_universe(child)

        # Check success criteria
        if m["cci"] >= 0.75 and m["survival"] >= 0.55 and m["collapse"] <= 0.25:
            successful_children += 1

        # parent back-reaction cost (energy + fairness drain)
        back_reaction = norm01(cost_factor * (0.5 + 0.7 * np.random.rand()))
        parent_effect = dict(
            parent_energy=clamp(
                parent["energy"] - 0.5 * back_reaction, *BOUNDS["energy"]
            ),
            parent_fairness=clamp(
                parent["fairness"] - 0.4 * back_reaction, *BOUNDS["fairness"]
            ),
        )
        rows.append({**child, **m, **parent_effect, "back_reaction": back_reaction})

    df = pd.DataFrame(rows)
    # child "success" threshold (tweakable)
    success = df[
        (df["cci"] >= 0.75) & (df["survival"] >= 0.55) & (df["collapse"] <= 0.25)
    ]

    summary = dict(
        N=N,
        success_rate=float(len(success) / len(df)),
        n_successful=len(success),
        child_mean_cci=float(df["cci"].mean()),
        child_mean_survival=float(df["survival"].mean()),
        child_mean_collapse=float(df["collapse"].mean()),
        parent_energy_after=float(df["parent_energy"].mean()),
        parent_fairness_after=float(df["parent_fairness"].mean()),
        avg_back_reaction=float(df["back_reaction"].mean()),
    )

    print(f"   → Successful children: {len(success)}/{N} ({len(success)/N:.1%})")
    print(f"   → Child mean CCI: {df['cci'].mean():.3f}")
    print(f"   → Child mean survival: {df['survival'].mean():.3f}")
    print(f"   → Child mean collapse: {df['collapse'].mean():.3f}")
    print(
        f"   → Parent energy drain: {BASE_PARENT['energy']:.3f} → {df['parent_energy'].mean():.3f}"
    )
    print(
        f"   → Parent fairness drain: {BASE_PARENT['fairness']:.3f} → {df['parent_fairness'].mean():.3f}"
    )

    return df, summary


# ---------- B) TRAVEL MECHANISMS ----------
def travel_trial(mech: str, state: dict[str, float]) -> dict[str, float]:
    E, C, F, N = (
        state["energy"],
        state["coordination"],
        state["fairness"],
        state["noise"],
    )
    if mech == "WH":  # Wormhole stabilization
        energy_cost = 0.25 - 0.10 * E - 0.05 * C  # more energy/coord lowers cost
        throat_stability = clamp(0.45 + 0.25 * C + 0.10 * E - 0.30 * N, 0, 1)
        causality_risk = clamp(0.35 - 0.15 * C + 0.05 / N if N > 0 else 0.9, 0, 1)
        success = clamp(throat_stability - 0.5 * causality_risk, 0, 1)
    elif mech == "DW":  # Domain-wall crossing
        barrier = clamp(0.6 - 0.25 * E - 0.10 * C, 0.05, 1.0)
        tunneling = clamp(0.35 + 0.20 * E + 0.10 * C - 0.20 * N, 0, 1)
        success = clamp(tunneling - 0.5 * barrier, 0, 1)
        energy_cost = 0.20 + 0.10 * barrier
        causality_risk = clamp(0.10 + 0.15 * barrier, 0, 1)
    elif mech == "FVB":  # False-vacuum bubble
        nucleation = clamp(0.30 + 0.25 * E - 0.15 * N, 0, 1)
        pinch_off = clamp(
            0.40 + 0.20 * C - 0.10 * N, 0, 1
        )  # higher = safer child (no parent-eating)
        parent_eat = clamp(0.35 - 0.25 * C + 0.20 * N, 0, 1)
        success = clamp(nucleation * pinch_off - 0.6 * parent_eat, 0, 1)
        energy_cost = 0.30 + 0.20 * nucleation
        causality_risk = clamp(0.15 + 0.20 * parent_eat, 0, 1)
    elif mech == "IO":  # Information-only transfer
        fidelity = clamp(0.55 + 0.25 * C + 0.15 * F - 0.10 * N, 0, 1)
        success = fidelity
        energy_cost = 0.06 - 0.03 * E
        causality_risk = 0.02
    else:
        return dict(success=np.nan, energy_cost=np.nan, causality_risk=np.nan)
    return dict(
        success=success, energy_cost=norm01(energy_cost), causality_risk=causality_risk
    )


def travel_experiment(parent=BASE_PARENT, reps=200):
    print("\n🚀 INTERDIMENSIONAL TRAVEL EXPERIMENT")
    print(f"   Testing 4 travel mechanisms × {reps} trials each = {4*reps} total tests")
    print(
        "   Mechanisms: WH (Wormhole), DW (Domain Wall), FVB (False Vacuum), IO (Information Only)"
    )

    rows = []
    mechanisms = ["WH", "DW", "FVB", "IO"]
    mech_names = {
        "WH": "Wormhole",
        "DW": "Domain Wall",
        "FVB": "False Vacuum Bubble",
        "IO": "Information Transfer",
    }

    for mech in mechanisms:
        print(f"     Testing {mech_names[mech]} mechanism...")
        success_count = 0

        for rep in range(reps):
            # jitter parent within bounds to simulate conditions variation
            state = {
                "energy": clamp(
                    parent["energy"] + 0.05 * np.random.randn(), *BOUNDS["energy"]
                ),
                "coordination": clamp(
                    parent["coordination"] + 0.05 * np.random.randn(),
                    *BOUNDS["coordination"],
                ),
                "fairness": clamp(
                    parent["fairness"] + 0.05 * np.random.randn(), *BOUNDS["fairness"]
                ),
                "noise": clamp(
                    parent["noise"] + 0.02 * np.random.randn(), *BOUNDS["noise"]
                ),
            }
            m = travel_trial(mech, state)
            if m["success"] > 0.5:
                success_count += 1
            rows.append(dict(mechanism=mech, **state, **m))

        print(
            f"       → Success rate: {success_count}/{reps} ({success_count/reps:.1%})"
        )

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("mechanism")[["success", "energy_cost", "causality_risk"]]
        .mean()
        .reset_index()
    )

    # composite: higher is better (success – cost – risk, normalized)
    comp = []
    for _, r in agg.iterrows():
        composite = clamp(
            0.60 * r["success"] - 0.25 * r["energy_cost"] - 0.15 * r["causality_risk"],
            0,
            1,
        )
        comp.append(composite)
    agg["composite"] = comp
    agg_sorted = agg.sort_values("composite", ascending=False)

    print("\n   🏆 TRAVEL MECHANISM RANKINGS:")
    for i, (_, row) in enumerate(agg_sorted.iterrows()):
        mech_full = mech_names[row["mechanism"]]
        print(
            f"   {i+1}. {mech_full}: Composite={row['composite']:.3f} (Success={row['success']:.3f}, Cost={row['energy_cost']:.3f}, Risk={row['causality_risk']:.3f})"
        )

    return df, agg_sorted


# ---------- Runner ----------
def main():
    t0 = time.time()

    print(f"⏱️  Time budget: {TIME_BUDGET_MIN} minutes")

    # A) Creation
    print(f"\n{'='*60}")
    dfc, sumc = creation_experiment()
    dfc.to_csv(ROOT / "creation_history.csv", index=False)
    with open(ROOT / "creation_summary.json", "w") as f:
        json.dump(sumc, f, indent=2)

    # plot creation results
    plt.figure(figsize=(10, 6))
    metrics = ["CCI", "Survival", "1-Collapse"]
    values = [dfc["cci"].mean(), dfc["survival"].mean(), 1 - dfc["collapse"].mean()]

    bars = plt.bar(metrics, values)
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 0.7:
            bar.set_color("green")
        elif val > 0.5:
            bar.set_color("orange")
        else:
            bar.set_color("red")
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Score (0-1)")
    plt.title(
        f"Child Universe Performance (N={len(dfc)}, Success Rate={sumc['success_rate']:.1%})"
    )
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    savefig(ROOT / "bars_creation.png")

    # B) Travel
    print(f"\n{'='*60}")
    if (time.time() - t0) < (TIME_BUDGET_MIN * 60 - 60):  # leave 1 min buffer
        dft, aggt = travel_experiment()
        dft.to_csv(ROOT / "travel_history.csv", index=False)
        with open(ROOT / "travel_summary.json", "w") as f:
            json.dump(aggt.to_dict(orient="records"), f, indent=2)

        # plot travel mechanisms
        plt.figure(figsize=(12, 6))
        x = np.arange(len(aggt))
        width = 0.28

        bars1 = plt.bar(
            x - width,
            aggt["success"].values,
            width,
            label="Success",
            color="green",
            alpha=0.7,
        )
        bars2 = plt.bar(
            x,
            aggt["energy_cost"].values,
            width,
            label="Energy Cost",
            color="orange",
            alpha=0.7,
        )
        bars3 = plt.bar(
            x + width,
            aggt["causality_risk"].values,
            width,
            label="Causality Risk",
            color="red",
            alpha=0.7,
        )

        plt.xticks(x, [f"{row['mechanism']}" for _, row in aggt.iterrows()], rotation=0)
        plt.ylabel("Score (0-1)")
        plt.legend()
        plt.title("Travel Mechanisms Comparison")
        plt.grid(True, alpha=0.3)
        savefig(ROOT / "bars_travel.png")

        # Pareto (success vs safety/efficiency)
        plt.figure(figsize=(10, 6))
        safety_eff = 1.0 - (0.5 * aggt["energy_cost"] + 0.5 * aggt["causality_risk"])

        colors = ["blue", "green", "red", "purple"]
        for i, (_, row) in enumerate(aggt.iterrows()):
            plt.scatter(
                row["success"], safety_eff.iloc[i], s=100, c=colors[i], alpha=0.7
            )
            plt.annotate(
                row["mechanism"],
                (row["success"], safety_eff.iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
            )

        plt.xlabel("Success Rate →")
        plt.ylabel("Safety/Efficiency (1 - Cost/Risk) →")
        plt.title("Travel Mechanisms: Pareto Frontier")
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        savefig(ROOT / "pareto_travel.png")
    else:
        print("⏰ Skipping travel experiment due to time constraints")
        aggt = pd.DataFrame()

    # Generate comprehensive report
    elapsed = (time.time() - t0) / 60

    report_data = {
        "run_stamp": RUN_STAMP,
        "creation_results": sumc,
        "travel_results": aggt.to_dict(orient="records") if not aggt.empty else [],
        "runtime_minutes": elapsed,
    }

    with open(ROOT / "experiment_summary.json", "w") as f:
        json.dump(report_data, f, indent=2)

    # Detailed markdown report
    with open(ROOT / "REPORT.md", "w") as f:
        f.write(f"# Baby-Universe Lab — {RUN_STAMP}\n\n")
        f.write("## Overview\n\n")
        f.write(
            "This experiment investigated two fundamental aspects of universe engineering:\n"
        )
        f.write(
            "- **A) CREATION**: Spawning child universes with genetic inheritance and mutation\n"
        )
        f.write("- **B) TRAVEL**: Four interdimensional travel mechanisms\n\n")

        f.write("## Creation Results\n\n")
        f.write(f"- **Child universes spawned**: {sumc['N']}\n")
        f.write(
            f"- **Success rate**: {sumc['success_rate']:.1%} ({sumc['n_successful']} viable)\n"
        )
        f.write(f"- **Child mean CCI**: {sumc['child_mean_cci']:.3f}\n")
        f.write(f"- **Child mean survival**: {sumc['child_mean_survival']:.3f}\n")
        f.write(f"- **Child mean collapse**: {sumc['child_mean_collapse']:.3f}\n")
        f.write(
            f"- **Parent energy impact**: {BASE_PARENT['energy']:.3f} → {sumc['parent_energy_after']:.3f}\n"
        )
        f.write(
            f"- **Parent fairness impact**: {BASE_PARENT['fairness']:.3f} → {sumc['parent_fairness_after']:.3f}\n\n"
        )

        if not aggt.empty:
            f.write("## Travel Results\n\n")
            f.write("Travel mechanism rankings by composite performance:\n\n")
            for i, (_, row) in enumerate(aggt.iterrows()):
                mech_names = {
                    "WH": "Wormhole",
                    "DW": "Domain Wall",
                    "FVB": "False Vacuum Bubble",
                    "IO": "Information Transfer",
                }
                f.write(
                    f"{i+1}. **{mech_names[row['mechanism']]}**: {row['composite']:.3f}\n"
                )
                f.write(f"   - Success: {row['success']:.3f}\n")
                f.write(f"   - Energy Cost: {row['energy_cost']:.3f}\n")
                f.write(f"   - Causality Risk: {row['causality_risk']:.3f}\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `creation_history.csv`: Raw child universe data\n")
        f.write("- `creation_summary.json`: Creation experiment summary\n")
        f.write("- `travel_history.csv`: Raw travel trial data\n")
        f.write("- `travel_summary.json`: Travel mechanism rankings\n")
        f.write("- `bars_creation.png`: Child universe performance chart\n")
        f.write("- `bars_travel.png`: Travel mechanism comparison\n")
        f.write("- `pareto_travel.png`: Travel efficiency/safety plot\n")

    print("\n✅ BABY-UNIVERSE LAB COMPLETE!")
    print(f"📁 Results saved to: {ROOT}")
    print(f"⏱️  Runtime: {elapsed:.1f} minutes")

    if not aggt.empty and len(aggt) > 0:
        winner = aggt.iloc[0]
        mech_names = {
            "WH": "Wormhole",
            "DW": "Domain Wall",
            "FVB": "False Vacuum Bubble",
            "IO": "Information Transfer",
        }
        print(
            f"🏆 Best travel mechanism: {mech_names[winner['mechanism']]} (Score: {winner['composite']:.3f})"
        )

    # Copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory...")
    shutil.copy(ROOT / "REPORT.md", "./BABY_UNIVERSE_RESULTS.md")
    shutil.copy(ROOT / "experiment_summary.json", "./BABY_UNIVERSE_SUMMARY.json")
    if dfc is not None:
        shutil.copy(ROOT / "creation_history.csv", "./BABY_UNIVERSE_DATA.csv")

    print("✅ Easy access files created:")
    print("   - BABY_UNIVERSE_RESULTS.md")
    print("   - BABY_UNIVERSE_SUMMARY.json")
    print("   - BABY_UNIVERSE_DATA.csv")


if __name__ == "__main__":
    main()
