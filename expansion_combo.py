#!/usr/bin/env python3
"""
Expansion Combo Test (≤ ~10 min)
Goal: Identify the best combination(s) of levers for stable expansion.

Levers (same semantics as your last run):
  A) constructive_shocks
  B) coord_training
  C) costly_signal
  D) energy_influx
  E) ineq_control

We test:
- All PAIRS (10 total)
- The "Magic Trio" hypothesized best: energy_influx + coord_training + ineq_control
- Control (no levers)

Outputs -> ./discovery_results/expansion_combo_<stamp>/
  - samples.csv (per-epoch rows)
  - RANKINGS.json (sorted by composite score)
  - ATEs_vs_control.json (pairwise/trio average treatment effects)
  - bars_composite.png (top combos)
  - radar_top5.png (radar of top 5 on 4 metrics)
  - REPORT.md (quick narrative)
"""

import datetime
import itertools
import json
import pathlib
import random
import time
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Config (fast) ----------------
SEED = 2025
EPOCHS = 6
POP = 32
SHOCKS = [0.2, 0.5, 0.8]
TIME_BUDGET_MIN = 10
PER_CALL_TIMEOUT = 8

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = pathlib.Path("./discovery_results") / f"expansion_combo_{RUN_STAMP}"
ROOT.mkdir(parents=True, exist_ok=True)

print("🧪 Starting Expansion Combo Test...")
print(f"📁 Results will be saved to: {ROOT}")

LEVER_NAMES = [
    "constructive_shocks",
    "coord_training",
    "costly_signal",
    "energy_influx",
    "ineq_control",
]
PAIRS = list(itertools.combinations(LEVER_NAMES, 2))
TRIOS = [("energy_influx", "coord_training", "ineq_control")]  # "magic trio"

print("🔬 Testing combinations:")
print("   - Control (no levers)")
print(f"   - All pairs: {len(PAIRS)} combinations")
print("   - Magic trio: energy_influx + coord_training + ineq_control")
print(
    f"📊 Total: {1 + len(PAIRS) + len(TRIOS)} combinations × {EPOCHS} epochs × {POP} agents = {(1 + len(PAIRS) + len(TRIOS)) * EPOCHS * POP} evaluations"
)

# Bounds
BOUNDS = dict(
    coherence=(0.55, 0.90),
    noise=(0.00, 0.20),
    coupling=(0.55, 0.85),
    coordination=(0.45, 0.85),
    goal_diversity=(2, 5),
)


# ---------------- Adapters ----------------
def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


shock_resilience = _safe_import("shock_resilience")
calibration_experiment = _safe_import("calibration_experiment")
goal_externalities = _safe_import("goal_externalities")


def run_shock(agent: dict[str, Any], shock: float) -> dict[str, Any]:
    cfg = {**agent, "shock_severity": shock, "max_steps": 170}
    t0 = time.time()
    try:
        if shock_resilience and hasattr(shock_resilience, "run"):
            res = shock_resilience.run(cfg)
        elif shock_resilience and hasattr(shock_resilience, "run_experiment"):
            res = shock_resilience.run_experiment(cfg)
        else:
            rnd = np.random.rand()
            base = (
                0.58
                + 0.18 * (agent["coherence"] - 0.7)
                + 0.10 * (agent["coordination"] - 0.6)
                + 0.08 * (agent["coupling"] - 0.65)
            )
            surv = np.clip(
                base + 0.10 * rnd - 0.24 * (shock - 0.2) - 0.25 * agent["noise"], 0, 1
            )
            res = {
                "survival_rate": float(surv),
                "recovery_time": float(
                    80 - 40 * (agent["coherence"] - 0.7) + 25 * (shock - 0.2)
                ),
            }
        if time.time() - t0 > PER_CALL_TIMEOUT:
            res["timeout_flag"] = True
        return res
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def run_calibration(agent: dict[str, Any]) -> dict[str, Any]:
    try:
        if calibration_experiment and hasattr(calibration_experiment, "compute_cci"):
            return calibration_experiment.compute_cci({**agent, "max_steps": 140})
        elif calibration_experiment and hasattr(calibration_experiment, "run"):
            return calibration_experiment.run({**agent, "max_steps": 140})
        else:
            cal = 0.74 + 0.12 * (agent["coherence"] - 0.7) - 0.25 * agent["noise"]
            coh = agent["coherence"]
            emi = np.clip(
                0.58
                + 0.10 * (agent["goal_diversity"] - 3)
                + 0.12 * agent.get("_space", 0.0),
                0,
                1,
            )
            noi = agent["noise"]
            raw = max(0, cal) * max(0, coh) * max(0, emi) / (noi + 0.10)
            cci = raw / (1.0 + raw)
            return {
                "cci": float(np.clip(cci, 0, 1)),
                "coherence_score": float(coh),
                "emergence_index": float(emi),
                "calibration_accuracy": float(np.clip(cal, 0, 1)),
                "noise_level": float(noi),
            }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def run_collapse(agent: dict[str, Any]) -> dict[str, Any]:
    try:
        if goal_externalities and hasattr(goal_externalities, "evaluate"):
            return goal_externalities.evaluate({**agent, "max_steps": 110})
        else:
            gini = np.clip(
                0.28
                + 0.05 * (agent["goal_diversity"] - 3)
                - 0.12 * agent["coordination"],
                0,
                1,
            )
            collapse = np.clip(0.30 + 0.9 * (gini - 0.30), 0, 1)
            return {"collapse_risk": float(collapse), "gini": float(gini)}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# ---------------- Helpers ----------------
rng = np.random.default_rng(SEED)


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def new_agent(coord, coup, noise, space):
    return {
        "coherence": clamp(0.70 + 0.10 * (coord - 0.60), *BOUNDS["coherence"]),
        "noise": clamp(noise, *BOUNDS["noise"]),
        "goal_diversity": int(rng.integers(3, 5)),
        "coupling": clamp(coup, *BOUNDS["coupling"]),
        "coordination": clamp(coord, *BOUNDS["coordination"]),
        "_space": float(np.clip(space, 0, 1)),
    }


def apply_levers(levers: dict[str, int]) -> dict[str, float]:
    d = dict(
        d_coord=0.0, d_coup=0.0, d_noise=0.0, d_space=0.0, d_emerg=0.0, d_gini=-0.0
    )
    if levers.get("constructive_shocks"):
        d["d_coord"] += 0.008
        d["d_coup"] += 0.006
        d["d_noise"] -= 0.003
        d["d_emerg"] += 0.02
    if levers.get("coord_training"):
        d["d_coord"] += 0.012
        d["d_noise"] -= 0.004
    if levers.get("costly_signal"):
        d["d_coord"] += 0.006
        d["d_coup"] += 0.004
        d["d_noise"] -= 0.002
    if levers.get("energy_influx"):
        d["d_space"] += 0.03
        d["d_coup"] += 0.006
        d["d_emerg"] += 0.03
    if levers.get("ineq_control"):
        d["d_coord"] += 0.006
        d["d_noise"] -= 0.002
        d["d_gini"] -= 0.02
    return d


def epoch_roll(state: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
    state["coord"] = clamp(state["coord"] + deltas["d_coord"], *BOUNDS["coordination"])
    state["coup"] = clamp(state["coup"] + deltas["d_coup"], *BOUNDS["coupling"])
    state["noise"] = clamp(state["noise"] + deltas["d_noise"], *BOUNDS["noise"])
    state["space"] = float(np.clip(state["space"] + deltas["d_space"], 0, 1))
    state["emerg_boost"] = float(
        np.clip(state.get("emerg_boost", 0.0) + deltas["d_emerg"], 0, 1)
    )
    state["gini_adj"] = float(
        np.clip(state.get("gini_adj", 0.0) + deltas.get("d_gini", -0.0), -0.5, 0.0)
    )
    return state


def evaluate_epoch(state: dict[str, float]) -> dict[str, float]:
    pop = [
        new_agent(
            state["coord"] + 0.02 * rng.standard_normal(),
            state["coup"] + 0.02 * rng.standard_normal(),
            max(0.0, state["noise"] + 0.01 * rng.standard_normal()),
            state["space"] + 0.03 * rng.standard_normal(),
        )
        for _ in range(POP)
    ]
    for a in pop:
        a["_space"] = float(np.clip(a["_space"], 0, 1))

    survs, cci_vals, colls, emis = [], [], [], []
    for a in pop:
        svals = [run_shock(a, s)["survival_rate"] for s in SHOCKS]
        survs.append(np.mean(svals))
        cal = run_calibration(a)
        cci_vals.append(cal["cci"])
        emis.append(cal["emergence_index"])
        col = run_collapse(a)
        colls.append(
            float(np.clip(col["collapse_risk"] + state.get("gini_adj", 0.0), 0, 1))
        )

    expansion = (
        (state["space"] - 0.20)
        + (state["coord"] - 0.60)
        + (state["coup"] - 0.68)
        - (state["noise"] - 0.06)
        + (np.mean(emis) - 0.58)
    )
    exp_norm = float(np.clip(0.5 + 0.9 * expansion, 0, 1))

    return {
        "ExpansionIndex": exp_norm,
        "Survival": float(np.mean(survs)),
        "CCI": float(np.mean(cci_vals)),
        "Collapse": float(np.mean(colls)),
        "Emergence": float(np.mean(emis)),
    }


def run_combo(name: str, levers_on: list[str]) -> pd.DataFrame:
    # State starts neutral
    state = {
        "coord": 0.60,
        "coup": 0.68,
        "noise": 0.06,
        "space": 0.20,
        "emerg_boost": 0.0,
        "gini_adj": 0.0,
    }
    rows = []
    for ep in range(EPOCHS):
        deltas = apply_levers({k: 1 for k in levers_on})
        state = epoch_roll(state, deltas)
        m = evaluate_epoch(state)
        rows.append(
            {
                "combo": name,
                "epoch": ep,
                **{k: int(k in levers_on) for k in LEVER_NAMES},
                "coord": state["coord"],
                "coupling": state["coup"],
                "noise": state["noise"],
                "space": state["space"],
                **m,
            }
        )
    return pd.DataFrame(rows)


# ---------------- Main ----------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    # Build combos: control + pairs + trio(s)
    combos = [("CONTROL", [])]
    combos += [
        (f"PAIR:{a.split('_')[0][:4]}&{b.split('_')[0][:4]}", [a, b])
        for (a, b) in PAIRS
    ]
    combos += [
        (f"TRIO:{'-'.join([x.split('_')[0][:4] for x in t])}", list(t)) for t in TRIOS
    ]

    print("\n🔄 Running combination tests...")
    all_rows = []
    for i, (name, lever_list) in enumerate(combos):
        print(f"  📋 Testing {i+1}/{len(combos)}: {name}")
        if lever_list:
            print(f"     Levers active: {', '.join(lever_list)}")
        else:
            print("     Control - no levers")

        df = run_combo(name, lever_list)
        all_rows.append(df)

        # Show quick result
        final_expansion = df["ExpansionIndex"].iloc[-1]
        final_survival = df["Survival"].iloc[-1]
        final_collapse = df["Collapse"].iloc[-1]
        final_cci = df["CCI"].iloc[-1]
        print(
            f"     → Final: Exp={final_expansion:.3f}, Surv={final_survival:.3f}, CCI={final_cci:.3f}, Coll={final_collapse:.3f}"
        )

        if (time.time() - t0) > (TIME_BUDGET_MIN * 60):
            print(f"  ⏰ Time budget exceeded at combo {i+1}")
            break

    print("\n💾 Analyzing combination results...")
    data = pd.concat(all_rows, ignore_index=True)
    data.to_csv(ROOT / "samples.csv", index=False)

    # Aggregates
    agg = (
        data.groupby("combo")[["ExpansionIndex", "Survival", "CCI", "Collapse"]]
        .mean()
        .reset_index()
    )
    # Composite "StabilityScore": mean of (Expansion, Survival, CCI, 1-Collapse)
    agg["StabilityScore"] = (
        agg["ExpansionIndex"] + agg["Survival"] + agg["CCI"] + (1.0 - agg["Collapse"])
    ) / 4.0
    agg_sorted = agg.sort_values("StabilityScore", ascending=False)
    rankings = agg_sorted.to_dict(orient="records")
    with open(ROOT / "RANKINGS.json", "w") as f:
        json.dump(rankings, f, indent=2)

    print("\n🏆 TOP COMBINATION RANKINGS:")
    for i, row in enumerate(agg_sorted.head(5).iterrows()):
        r = row[1]
        print(
            f"{i+1}. {r['combo']}: Stability={r['StabilityScore']:.3f} (Exp={r['ExpansionIndex']:.3f}, Surv={r['Survival']:.3f}, CCI={r['CCI']:.3f}, Coll={r['Collapse']:.3f})"
        )

    # ATEs vs Control
    ctrl = data[data["combo"] == "CONTROL"]
    ates = {}
    print("\n📈 Computing Average Treatment Effects vs Control...")
    for name, _ in combos[1:]:
        d = data[data["combo"] == name]
        expansion_ate = float(
            d["ExpansionIndex"].mean() - ctrl["ExpansionIndex"].mean()
        )
        survival_ate = float(d["Survival"].mean() - ctrl["Survival"].mean())
        cci_ate = float(d["CCI"].mean() - ctrl["CCI"].mean())
        collapse_ate = float(d["Collapse"].mean() - ctrl["Collapse"].mean())

        ates[name] = {
            "ExpansionIndex_ATE": expansion_ate,
            "Survival_ATE": survival_ate,
            "CCI_ATE": cci_ate,
            "Collapse_ATE": collapse_ate,
        }

        # Calculate composite ATE
        composite_ate = (expansion_ate + survival_ate + cci_ate - collapse_ate) / 4.0
        print(
            f"  {name}: Composite ATE={composite_ate:+.3f} (Exp={expansion_ate:+.3f}, Surv={survival_ate:+.3f}, CCI={cci_ate:+.3f}, Coll={collapse_ate:+.3f})"
        )

    with open(ROOT / "ATEs_vs_control.json", "w") as f:
        json.dump(ates, f, indent=2)

    # Bars of top combos by composite
    print("📊 Generating visualizations...")
    try:
        top = agg_sorted.head(8)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(top)), top["StabilityScore"].values)
        # Color bars by performance
        for i, bar in enumerate(bars):
            score = top["StabilityScore"].iloc[i]
            if score > 0.75:
                bar.set_color("darkgreen")
            elif score > 0.70:
                bar.set_color("green")
            elif score > 0.65:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        plt.xticks(range(len(top)), top["combo"].values, rotation=30, ha="right")
        plt.ylabel("Composite Stability Score (0..1)")
        plt.title("Top Combos by Composite Stability")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROOT / "bars_composite.png", dpi=160, bbox_inches="tight")
        plt.close()

        # Radar plot for top 5
        def radar(df_row, labels=("ExpansionIndex", "Survival", "CCI", "Collapse")):
            # Collapse inverted for "higher is better"
            vals = [
                df_row["ExpansionIndex"],
                df_row["Survival"],
                df_row["CCI"],
                1.0 - df_row["Collapse"],
            ]
            return vals

        top5 = agg_sorted.head(5)
        labels = ["Expansion", "Survival", "CCI", "1-Collapse"]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(10, 8))
        colors = ["darkgreen", "green", "blue", "orange", "red"]
        for i, (_, r) in enumerate(top5.iterrows()):
            vals = radar(r)
            vals += vals[:1]
            plt.polar(
                angles,
                vals,
                marker="o",
                alpha=0.7,
                label=r["combo"],
                color=colors[i],
                linewidth=2,
            )
        plt.thetagrids(np.degrees(angles[:-1]), labels)
        plt.title("Top 5 Combos — Radar (higher is better)")
        plt.legend(bbox_to_anchor=(1.2, 1.05))
        plt.savefig(ROOT / "radar_top5.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠️  Plot generation error: {e}")
        with open(ROOT / "plot_errors.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())

    # Report
    winner = agg_sorted.iloc[0]
    report_data = {
        "run_stamp": RUN_STAMP,
        "winner": {
            "combo": winner["combo"],
            "stability_score": float(winner["StabilityScore"]),
            "expansion": float(winner["ExpansionIndex"]),
            "survival": float(winner["Survival"]),
            "cci": float(winner["CCI"]),
            "collapse": float(winner["Collapse"]),
        },
        "design": {
            "epochs": EPOCHS,
            "pop": POP,
            "shocks": SHOCKS,
            "total_combos": len(combos),
        },
        "notes": "Composite score = mean of (Expansion, Survival, CCI, 1-Collapse)",
    }

    with open(ROOT / "experiment_summary.json", "w") as f:
        json.dump(report_data, f, indent=2)

    md = []
    md.append(f"# Expansion Combo Test — {RUN_STAMP}\n")
    md.append("## Best Combination Analysis\n")
    md.append(
        "This experiment tested pairs and trios of interventions to find optimal combinations for stable expansion.\n"
    )
    md.append("### Winner\n")
    md.append(
        f"**{winner['combo']}** - Composite Stability Score: {winner['StabilityScore']:.3f}\n"
    )
    md.append(f"- Expansion Index: {winner['ExpansionIndex']:.3f}\n")
    md.append(f"- Survival Rate: {winner['Survival']:.3f}\n")
    md.append(f"- CCI: {winner['CCI']:.3f}\n")
    md.append(f"- Collapse Risk: {winner['Collapse']:.3f}\n")
    md.append("### Files Generated\n")
    md.append("- samples.csv: per-epoch rows for all combos\n")
    md.append(
        "- RANKINGS.json: combos sorted by composite (Expansion, Survival, CCI, 1-Collapse)\n"
    )
    md.append("- ATEs_vs_control.json: improvement vs control\n")
    md.append("- bars_composite.png / radar_top5.png\n")

    with open(ROOT / "REPORT.md", "w") as f:
        f.write("\n".join(md))

    print("\n✅ EXPANSION COMBO TEST COMPLETE!")
    print(f"📁 Results saved to: {ROOT}")
    print(
        f"🏆 Winner: {winner['combo']} (Stability Score: {winner['StabilityScore']:.3f})"
    )

    # Copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory...")
    shutil.copy(ROOT / "REPORT.md", "./EXPANSION_COMBO_RESULTS.md")
    shutil.copy(ROOT / "experiment_summary.json", "./EXPANSION_COMBO_SUMMARY.json")
    shutil.copy(ROOT / "samples.csv", "./EXPANSION_COMBO_DATA.csv")

    print("✅ Easy access files created:")
    print("   - EXPANSION_COMBO_RESULTS.md")
    print("   - EXPANSION_COMBO_SUMMARY.json")
    print("   - EXPANSION_COMBO_DATA.csv")


if __name__ == "__main__":
    main()
