#!/usr/bin/env python3
"""
Expansion vs Stagnation vs Contraction — Stability Test (≤ ~10 min)
What it does
- Runs three regimes:
    A) EXPANSION: capacity/space/coordination trend upward (small, steady growth)
    B) STAGNATION: flat; no trend
    C) CONTRACTION: small negative trend (shrinking capacity)
- Each regime simulates short epochs with mild/mid/hard shocks and logs:
    survival_avg, cci_est, collapse_risk (per epoch + overall means)
- Uses your modules if available; otherwise fast synthetic fallbacks.
Outputs
- ./discovery_results/expansion_test_<stamp>/
    - *_history.csv, *_summary.json, REPORT.md
    - compare_bars.png (survival/cci/collapse across regimes)
"""

import datetime
import json
import pathlib
import random
import time
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- CONFIG (fast) ----------------
SEED = 4242
EPOCHS = 8  # short runs
POP = 36  # small pop per epoch
SHOCKS = [0.2, 0.5, 0.8]
TIME_BUDGET_MIN = 10
PER_CALL_TIMEOUT = 8

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = pathlib.Path("./discovery_results") / f"expansion_test_{RUN_STAMP}"
ROOT.mkdir(parents=True, exist_ok=True)

print("🚀 Starting Expansion vs Stagnation vs Contraction Stability Test...")
print(f"📁 Results will be saved to: {ROOT}")

# Expansion levers per regime (drift per epoch)
REGIMES = {
    "EXPANSION": {
        "d_coord": +0.015,
        "d_coupling": +0.010,
        "d_space": +0.02,
        "d_noise": -0.004,
    },
    "STAGNATION": {
        "d_coord": +0.000,
        "d_coupling": +0.000,
        "d_space": +0.00,
        "d_noise": +0.000,
    },
    "CONTRACTION": {
        "d_coord": -0.015,
        "d_coupling": -0.010,
        "d_space": -0.02,
        "d_noise": +0.004,
    },
}
# Parameter bounds
BOUNDS = dict(
    coherence=(0.55, 0.90),
    noise=(0.00, 0.20),
    coupling=(0.55, 0.85),
    coordination=(0.45, 0.85),
    goal_diversity=(2, 5),
)

print(
    f"📊 Running 3 regimes × {EPOCHS} epochs × {POP} agents × {len(SHOCKS)} shock levels = {3*EPOCHS*POP*len(SHOCKS)} total evaluations"
)


# -------------- Adapters ---------------
def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


shock_resilience = _safe_import("shock_resilience")
calibration_experiment = _safe_import("calibration_experiment")
goal_externalities = _safe_import("goal_externalities")
gravity_analysis = _safe_import(
    "gravity_analysis"
)  # optional, used for "space/bound" proxy


def run_shock(agent: dict[str, Any], shock: float) -> dict[str, Any]:
    cfg = {**agent, "shock_severity": shock, "max_steps": 180}
    t0 = time.time()
    try:
        if shock_resilience and hasattr(shock_resilience, "run"):
            res = shock_resilience.run(cfg)
        elif shock_resilience and hasattr(shock_resilience, "run_experiment"):
            res = shock_resilience.run_experiment(cfg)
        else:
            # Synthetic: survival benefits from coherence/coordination/coupling; harmed by noise and shock
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
            return calibration_experiment.compute_cci({**agent, "max_steps": 150})
        elif calibration_experiment and hasattr(calibration_experiment, "run"):
            return calibration_experiment.run({**agent, "max_steps": 150})
        else:
            cal = 0.74 + 0.12 * (agent["coherence"] - 0.7) - 0.25 * agent["noise"]
            coh = agent["coherence"]
            # Emergence boosted by goal_diversity and "space"
            emi = np.clip(
                0.58
                + 0.10 * (agent["goal_diversity"] - 3)
                + 0.18 * agent.get("_space", 0.0),
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
            return goal_externalities.evaluate({**agent, "max_steps": 120})
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


def new_agent(coord, coup, noise):
    return {
        "coherence": clamp(0.70 + 0.10 * (coord - 0.60), *BOUNDS["coherence"]),
        "noise": clamp(noise, *BOUNDS["noise"]),
        "goal_diversity": int(rng.integers(3, 5)),  # 3–4 mostly
        "coupling": clamp(coup, *BOUNDS["coupling"]),
        "coordination": clamp(coord, *BOUNDS["coordination"]),
        "_space": 0.0,  # an abstract "room to expand" proxy (0..1)
    }


def evolve_epoch(
    regime_name: str, epoch: int, coord: float, coup: float, noise: float, space: float
) -> dict[str, Any]:
    # Build a small population around the epoch-level state
    pop = [
        new_agent(
            coord + 0.02 * rng.standard_normal(),
            coup + 0.02 * rng.standard_normal(),
            max(0.0, noise + 0.01 * rng.standard_normal()),
        )
        for _ in range(POP)
    ]
    for a in pop:
        a["_space"] = np.clip(space + 0.05 * rng.standard_normal(), 0, 1)

    # Evaluate each agent across shocks; take means (fast)
    survs, cci_vals, colls = [], [], []
    for a in pop:
        svals = [run_shock(a, s)["survival_rate"] for s in SHOCKS]
        survs.append(np.mean(svals))
        cci_vals.append(run_calibration(a)["cci"])
        colls.append(run_collapse(a)["collapse_risk"])

    return {
        "regime": regime_name,
        "epoch": epoch,
        "coord": float(coord),
        "coupling": float(coup),
        "noise": float(noise),
        "space": float(space),
        "survival_avg": float(np.mean(survs)),
        "cci_avg": float(np.mean(cci_vals)),
        "collapse_risk_avg": float(np.mean(colls)),
    }


def run_regime(
    name: str, deltas: dict[str, float], out_dir: pathlib.Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"\n🔄 Running {name} regime...")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Start states (same across regimes)
    coord, coup, noise, space = 0.60, 0.68, 0.06, 0.20
    rows = []
    t0 = time.time()
    for ep in range(EPOCHS):
        print(
            f"  📅 Epoch {ep+1}/{EPOCHS} - coord:{coord:.3f}, coupling:{coup:.3f}, noise:{noise:.3f}, space:{space:.3f}"
        )
        epoch_result = evolve_epoch(name, ep, coord, coup, noise, space)
        rows.append(epoch_result)
        print(
            f"     → Survival:{epoch_result['survival_avg']:.3f}, CCI:{epoch_result['cci_avg']:.3f}, Collapse:{epoch_result['collapse_risk_avg']:.3f}"
        )

        # apply regime drift
        coord = clamp(coord + deltas["d_coord"], *BOUNDS["coordination"])
        coup = clamp(coup + deltas["d_coupling"], *BOUNDS["coupling"])
        noise = clamp(noise + deltas["d_noise"], *BOUNDS["noise"])
        space = float(np.clip(space + deltas["d_space"], 0, 1))
        if (time.time() - t0) > (TIME_BUDGET_MIN * 60):
            print(f"  ⏰ Time budget exceeded for {name}")
            break

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{name}_history.csv", index=False)

    summary = {
        "run_stamp": RUN_STAMP,
        "regime": name,
        "means": {
            "survival": round(float(df["survival_avg"].mean()), 3),
            "cci": round(float(df["cci_avg"].mean()), 3),
            "collapse": round(float(df["collapse_risk_avg"].mean()), 3),
        },
        "final_state": {
            "coord": float(df["coord"].iloc[-1]),
            "coupling": float(df["coupling"].iloc[-1]),
            "noise": float(df["noise"].iloc[-1]),
            "space": float(df["space"].iloc[-1]),
        },
    }
    with open(out_dir / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  ✅ {name} complete - Avg Survival:{summary['means']['survival']}, CCI:{summary['means']['cci']}, Collapse:{summary['means']['collapse']}"
    )
    return df, summary


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    exp_dir = ROOT / "EXPANSION"
    stg_dir = ROOT / "STAGNATION"
    con_dir = ROOT / "CONTRACTION"

    df_exp, sum_exp = run_regime("EXPANSION", REGIMES["EXPANSION"], exp_dir)
    df_stg, sum_stg = run_regime("STAGNATION", REGIMES["STAGNATION"], stg_dir)
    df_con, sum_con = run_regime("CONTRACTION", REGIMES["CONTRACTION"], con_dir)

    print("\n💾 Combining results...")
    all_df = pd.concat([df_exp, df_stg, df_con], ignore_index=True)
    all_df.to_csv(ROOT / "combined_history.csv", index=False)

    # Comparison plot
    try:
        print("📊 Generating comparison plot...")
        metrics = ["survival_avg", "cci_avg", "collapse_risk_avg"]
        labels = ["Survival (↑)", "CCI (↑)", "Collapse Risk (↓)"]
        reg_names = ["EXPANSION", "STAGNATION", "CONTRACTION"]
        means = []
        for r in reg_names:
            d = all_df[all_df["regime"] == r]
            means.append([float(d[m].mean()) for m in metrics])

        means = np.array(means)  # shape (3,3)
        x = np.arange(3)
        width = 0.28
        plt.figure()
        plt.bar(x - width, means[:, 0], width, label=labels[0])
        plt.bar(x, means[:, 1], width, label=labels[1])
        plt.bar(x + width, means[:, 2], width, label=labels[2])
        plt.xticks(x, reg_names)
        plt.ylabel("Score (0..1)")
        plt.title("Expansion vs Stagnation vs Contraction — Stability Metrics")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.savefig(ROOT / "compare_bars.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as e:
        with open(ROOT / "plot_errors.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())

    # Report
    combined = {
        "run_stamp": RUN_STAMP,
        "config": {"epochs": EPOCHS, "pop": POP, "shocks": SHOCKS, "regimes": REGIMES},
        "summary": {
            "EXPANSION": sum_exp,
            "STAGNATION": sum_stg,
            "CONTRACTION": sum_con,
        },
    }
    with open(ROOT / "SUMMARY.json", "w") as f:
        json.dump(combined, f, indent=2)

    md = []
    md.append(f"# Expansion Stability Test — {RUN_STAMP}\n")
    md.append(
        "We compare three drift regimes (expansion, flat, contraction) over short epochs.\n"
    )
    md.append("## Key Results\n")
    md.append(
        f"**EXPANSION**: Survival={sum_exp['means']['survival']}, CCI={sum_exp['means']['cci']}, Collapse={sum_exp['means']['collapse']}\n"
    )
    md.append(
        f"**STAGNATION**: Survival={sum_stg['means']['survival']}, CCI={sum_stg['means']['cci']}, Collapse={sum_stg['means']['collapse']}\n"
    )
    md.append(
        f"**CONTRACTION**: Survival={sum_con['means']['survival']}, CCI={sum_con['means']['cci']}, Collapse={sum_con['means']['collapse']}\n"
    )
    md.append(
        "## Key Artifacts\n- EXPANSION/EXPANSION_history.csv & _summary.json\n- STAGNATION/STAGNATION_history.csv & _summary.json\n- CONTRACTION/CONTRACTION_history.csv & _summary.json\n- combined_history.csv\n- compare_bars.png\n"
    )
    with open(ROOT / "REPORT.md", "w") as f:
        f.write("\n".join(md))

    print("\n✅ EXPANSION STABILITY TEST COMPLETE!")
    print(f"📁 Results saved to: {ROOT}")
    print(
        f"📊 EXPANSION: Survival={sum_exp['means']['survival']}, CCI={sum_exp['means']['cci']}, Collapse={sum_exp['means']['collapse']}"
    )
    print(
        f"📊 STAGNATION: Survival={sum_stg['means']['survival']}, CCI={sum_stg['means']['cci']}, Collapse={sum_stg['means']['collapse']}"
    )
    print(
        f"📊 CONTRACTION: Survival={sum_con['means']['survival']}, CCI={sum_con['means']['cci']}, Collapse={sum_con['means']['collapse']}"
    )

    # Copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory...")
    shutil.copy(ROOT / "REPORT.md", "./EXPANSION_TEST_RESULTS.md")
    shutil.copy(ROOT / "SUMMARY.json", "./EXPANSION_TEST_SUMMARY.json")
    shutil.copy(ROOT / "combined_history.csv", "./EXPANSION_TEST_DATA.csv")

    print("✅ Easy access files created:")
    print("   - EXPANSION_TEST_RESULTS.md")
    print("   - EXPANSION_TEST_SUMMARY.json")
    print("   - EXPANSION_TEST_DATA.csv")


if __name__ == "__main__":
    main()
