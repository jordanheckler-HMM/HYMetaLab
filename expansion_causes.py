#!/usr/bin/env python3
"""
Expansion Causes Test (≤ ~10 min)
Goal: Identify which interventions CAUSE expansion:
  A) Constructive shocks
  B) Coordination training
  C) Costly signaling
  D) External energy influx (space)
  E) Inequality control (lower Gini)

Design:
- Randomized, short-epoch simulations with 5 binary levers (2^5 combos sampled).
- Outcome per epoch = ExpansionIndex = Δspace + Δcoord + Δcoup − Δnoise + Δemergence_gain (normalized).
- Also track CCI, Survival (avg across shocks), Collapse risk.
- Compute ATEs (difference vs control) and a simple linear effect model.

Outputs:
./discovery_results/expansion_causes_<stamp>/
  - samples.csv, ATEs.json, effects.json, REPORT.md
  - bars_ATE_expansion.png, bars_ATE_survival.png, bars_ATE_collapse.png
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

# ---------- Config (fast) ----------
SEED = 2025
EPOCHS_PER_RUN = 6  # short
POP = 32
SHOCKS = [0.2, 0.5, 0.8]
TIME_BUDGET_MIN = 10
PER_CALL_TIMEOUT = 8

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = pathlib.Path("./discovery_results") / f"expansion_causes_{RUN_STAMP}"
ROOT.mkdir(parents=True, exist_ok=True)

print("🧪 Starting Expansion Causes Test...")
print(f"📁 Results will be saved to: {ROOT}")

# Bounds
BOUNDS = dict(
    coherence=(0.55, 0.90),
    noise=(0.00, 0.20),
    coupling=(0.55, 0.85),
    coordination=(0.45, 0.85),
    goal_diversity=(2, 5),
)

# Levers (binary interventions)
LEVER_NAMES = [
    "constructive_shocks",
    "coord_training",
    "costly_signal",
    "energy_influx",
    "ineq_control",
]

print(f"🔬 Testing 5 intervention levers: {LEVER_NAMES}")
print(
    f"📊 Running 2^5 = 32 combinations × {EPOCHS_PER_RUN} epochs × {POP} agents = {32*EPOCHS_PER_RUN*POP} total evaluations"
)


# ---------- Adapters ----------
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


# ---------- Helpers ----------
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


def apply_levers(state: dict[str, float], levers: dict[str, int]) -> dict[str, float]:
    """
    Map levers to small drifts per epoch.
    """
    d = dict(
        d_coord=0.0, d_coup=0.0, d_noise=0.0, d_space=0.0, d_emerg=0.0, d_gini=-0.0
    )

    if levers["constructive_shocks"]:
        # gentle, constructive pulses increase learning & coherence a bit
        d["d_coord"] += 0.008
        d["d_coup"] += 0.006
        d["d_noise"] -= 0.003
        d["d_emerg"] += 0.02

    if levers["coord_training"]:
        # accountability rituals → higher coordination, lower noise
        d["d_coord"] += 0.012
        d["d_noise"] -= 0.004

    if levers["costly_signal"]:
        # filter free-riders: slightly higher coord/coherence, small noise drop
        d["d_coord"] += 0.006
        d["d_coup"] += 0.004
        d["d_noise"] -= 0.002

    if levers["energy_influx"]:
        # new resources/territory/tech: more space, some coupling
        d["d_space"] += 0.03
        d["d_coup"] += 0.006
        d["d_emerg"] += 0.03

    if levers["ineq_control"]:
        # reduce Gini => better coordination and lower collapse later
        d["d_coord"] += 0.006
        d["d_noise"] -= 0.002
        d["d_gini"] -= 0.02

    return d


def epoch_roll(state: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
    # update state
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
    # population around the state
    pop = [
        new_agent(
            state["coord"] + 0.02 * rng.standard_normal(),
            state["coup"] + 0.02 * rng.standard_normal(),
            max(0.0, state["noise"] + 0.01 * rng.standard_normal()),
            state["space"] + 0.03 * rng.standard_normal(),
        )
        for _ in range(POP)
    ]
    # emerg boost applied
    for a in pop:
        a["_space"] = float(np.clip(a["_space"], 0, 1))

    survs, cci_vals, colls, emis = [], [], [], []
    for a in pop:
        # shocks
        svals = [run_shock(a, s)["survival_rate"] for s in SHOCKS]
        survs.append(np.mean(svals))
        # CCI
        cal = run_calibration(a)
        cci_vals.append(cal["cci"])
        emis.append(cal["emergence_index"])
        # collapse
        col = run_collapse(a)
        # apply inequality adjustment (proxy): lower collapse if gini_adj more negative
        colls.append(
            float(np.clip(col["collapse_risk"] + state.get("gini_adj", 0.0), 0, 1))
        )

    # compute expansion index components (epoch deltas are approximated by current state minus a neutral baseline)
    # baseline ~ (coord=0.60, coup=0.68, noise=0.06, space=0.20, emergence~0.58)
    expansion = (
        (state["space"] - 0.20)
        + (state["coord"] - 0.60)
        + (state["coup"] - 0.68)
        - (state["noise"] - 0.06)
        + (np.mean(emis) - 0.58)
    )
    # normalize rough 0..1
    exp_norm = float(np.clip(0.5 + 0.9 * expansion, 0, 1))

    return {
        "ExpansionIndex": exp_norm,
        "Survival": float(np.mean(survs)),
        "CCI": float(np.mean(cci_vals)),
        "Collapse": float(np.mean(colls)),
        "Emergence": float(np.mean(emis)),
    }


def run_one_combo(levers_bits: tuple[int, ...]) -> pd.DataFrame:
    # initial neutral state
    state = {
        "coord": 0.60,
        "coup": 0.68,
        "noise": 0.06,
        "space": 0.20,
        "emerg_boost": 0.0,
        "gini_adj": 0.0,
    }
    levers = {name: int(bit) for name, bit in zip(LEVER_NAMES, levers_bits)}
    rows = []
    for ep in range(EPOCHS_PER_RUN):
        # apply lever drifts
        deltas = apply_levers(state, levers)
        state = epoch_roll(state, deltas)
        # measure
        m = evaluate_epoch(state)
        rows.append(
            {
                "epoch": ep,
                **levers,
                "coord": state["coord"],
                "coupling": state["coup"],
                "noise": state["noise"],
                "space": state["space"],
                **m,
            }
        )
    return pd.DataFrame(rows)


# ---------- Main ----------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    # Sample all 2^5 = 32 combos (fast), but break if time budget breached
    combos = list(itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [0, 1]))
    all_rows = []

    print("\n🔄 Running intervention combinations...")
    for i, bits in enumerate(combos):
        combo_str = "".join([f"{name}:{bit}" for name, bit in zip(LEVER_NAMES, bits)])
        print(f"  📋 Combo {i+1}/32: {combo_str}")

        df = run_one_combo(bits)
        all_rows.append(df)

        # Show quick result
        final_expansion = df["ExpansionIndex"].iloc[-1]
        final_survival = df["Survival"].iloc[-1]
        final_collapse = df["Collapse"].iloc[-1]
        print(
            f"     → Final: Expansion={final_expansion:.3f}, Survival={final_survival:.3f}, Collapse={final_collapse:.3f}"
        )

        if (time.time() - t0) > (TIME_BUDGET_MIN * 60):
            print(f"  ⏰ Time budget exceeded at combo {i+1}")
            break

    print("\n💾 Analyzing results...")
    data = pd.concat(all_rows, ignore_index=True)
    data.to_csv(ROOT / "samples.csv", index=False)

    # ATEs vs pure control (all zeros)
    ctrl = data[
        (data["constructive_shocks"] == 0)
        & (data["coord_training"] == 0)
        & (data["costly_signal"] == 0)
        & (data["energy_influx"] == 0)
        & (data["ineq_control"] == 0)
    ]
    ates = {}

    print("\n📈 Computing Average Treatment Effects (ATEs)...")
    for name in LEVER_NAMES:
        treated = data[data[name] == 1]
        expansion_ate = float(
            treated["ExpansionIndex"].mean() - ctrl["ExpansionIndex"].mean()
        )
        survival_ate = float(treated["Survival"].mean() - ctrl["Survival"].mean())
        collapse_ate = float(treated["Collapse"].mean() - ctrl["Collapse"].mean())

        ates[name] = {
            "ExpansionIndex_ATE": expansion_ate,
            "Survival_ATE": survival_ate,
            "Collapse_ATE": collapse_ate,
        }
        print(
            f"  {name}: Expansion ATE={expansion_ate:+.3f}, Survival ATE={survival_ate:+.3f}, Collapse ATE={collapse_ate:+.3f}"
        )

    with open(ROOT / "ATEs.json", "w") as f:
        json.dump(ates, f, indent=2)

    # Simple linear effect model (no interactions, fast OLS-by-formula)
    # y = beta0 + sum(beta_i * lever_i)
    print("\n🔢 Computing linear effect models...")
    X = data[LEVER_NAMES].astype(float)
    for target, fname in [
        ("ExpansionIndex", "effects_expansion.json"),
        ("Survival", "effects_survival.json"),
        ("Collapse", "effects_collapse.json"),
    ]:
        # closed-form OLS
        X1 = np.column_stack([np.ones(len(X))] + [X[c].values for c in LEVER_NAMES])
        y = data[target].values
        beta = np.linalg.pinv(X1.T @ X1) @ (X1.T @ y)
        eff = {"intercept": float(beta[0])}
        for i, n in enumerate(LEVER_NAMES, start=1):
            eff[n] = float(beta[i])
        with open(ROOT / fname, "w") as f:
            json.dump(eff, f, indent=2)

    # Quick ATE bars
    print("📊 Generating ATE comparison plots...")

    def barplot(metric_key, outpng, title, invert=False):
        labels = LEVER_NAMES
        vals = [ates[n][metric_key] for n in labels]
        if invert:
            vals = [-v for v in vals]  # so downward (collapse) improvements plot up
        plt.figure()
        bars = plt.bar(range(len(labels)), vals)
        # Color bars based on value (positive = good, negative = bad for most metrics)
        for i, bar in enumerate(bars):
            if vals[i] > 0:
                bar.set_color("green" if not invert else "red")
            else:
                bar.set_color("red" if not invert else "green")
        plt.xticks(range(len(labels)), labels, rotation=20)
        plt.ylabel("ATE vs Control")
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.savefig(ROOT / outpng, dpi=160, bbox_inches="tight")
        plt.close()

    barplot(
        "ExpansionIndex_ATE",
        "bars_ATE_expansion.png",
        "ATE: ExpansionIndex (higher is better)",
    )
    barplot("Survival_ATE", "bars_ATE_survival.png", "ATE: Survival (higher is better)")
    barplot(
        "Collapse_ATE",
        "bars_ATE_collapse.png",
        "ATE: Collapse Risk (lower is better, bars inverted)",
        invert=True,
    )

    # Report
    report = {
        "run_stamp": RUN_STAMP,
        "design": {
            "epochs_per_run": EPOCHS_PER_RUN,
            "pop": POP,
            "shocks": SHOCKS,
            "levers": LEVER_NAMES,
        },
        "notes": "ATEs are average differences vs control across all epochs and combos containing each lever.",
        "key_findings": {
            "strongest_expansion_lever": max(
                ates, key=lambda x: ates[x]["ExpansionIndex_ATE"]
            ),
            "strongest_survival_lever": max(
                ates, key=lambda x: ates[x]["Survival_ATE"]
            ),
            "strongest_collapse_reducer": min(
                ates, key=lambda x: ates[x]["Collapse_ATE"]
            ),
        },
    }
    with open(ROOT / "experiment_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    md = []
    md.append(f"# Expansion Causes Test — {RUN_STAMP}\n")
    md.append("## Intervention Effects Analysis\n")
    md.append(
        "This randomized experiment tested 5 interventions across 32 combinations to identify causal drivers of expansion.\n"
    )
    md.append("### Key Findings\n")
    md.append(
        f"- **Strongest Expansion Driver**: {report['key_findings']['strongest_expansion_lever']}\n"
    )
    md.append(
        f"- **Strongest Survival Driver**: {report['key_findings']['strongest_survival_lever']}\n"
    )
    md.append(
        f"- **Best Collapse Reducer**: {report['key_findings']['strongest_collapse_reducer']}\n"
    )
    md.append("### Files Generated\n")
    md.append("- samples.csv: raw experimental data\n")
    md.append("- ATEs.json: average treatment effects per intervention\n")
    md.append("- effects_*.json: linear effect sizes for Expansion/Survival/Collapse\n")
    md.append(
        "- bars_ATE_expansion.png / bars_ATE_survival.png / bars_ATE_collapse.png\n"
    )

    with open(ROOT / "REPORT.md", "w") as f:
        f.write("\n".join(md))

    print("\n✅ EXPANSION CAUSES TEST COMPLETE!")
    print(f"📁 Results saved to: {ROOT}")
    print(
        f"🏆 Strongest expansion driver: {report['key_findings']['strongest_expansion_lever']}"
    )
    print(
        f"🏆 Strongest survival driver: {report['key_findings']['strongest_survival_lever']}"
    )
    print(
        f"🏆 Best collapse reducer: {report['key_findings']['strongest_collapse_reducer']}"
    )

    # Copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory...")
    shutil.copy(ROOT / "REPORT.md", "./EXPANSION_CAUSES_RESULTS.md")
    shutil.copy(ROOT / "experiment_summary.json", "./EXPANSION_CAUSES_SUMMARY.json")
    shutil.copy(ROOT / "samples.csv", "./EXPANSION_CAUSES_DATA.csv")

    print("✅ Easy access files created:")
    print("   - EXPANSION_CAUSES_RESULTS.md")
    print("   - EXPANSION_CAUSES_SUMMARY.json")
    print("   - EXPANSION_CAUSES_DATA.csv")


if __name__ == "__main__":
    main()
