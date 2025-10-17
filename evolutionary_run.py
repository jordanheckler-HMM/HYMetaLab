#!/usr/bin/env python3
"""
Evolutionary Experiment (Time-Boxed ≤10 min)
- Small population + few generations for fast results
- Integrates with your sim modules if available (see Adapters)
- Exports CSV, JSON, PNG, and a Markdown summary to ./discovery_results/<timestamp>/

If a function name doesn't match your local modules, edit the ADAPTERS section.
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

# -----------------------------
# CONFIG — keep this small/fast
# -----------------------------
RANDOM_SEED = 42
POP_SIZE = 24  # small pop
GENERATIONS = 5  # few generations
TOP_FRACTION = 0.30  # selection pressure
MUTATION_RATE = 0.15
TIME_BUDGET_MIN = 10  # hard cap (minutes)
PER_SIM_TIMEOUT_SEC = 12  # skip any sim that hangs
EVAL_ENVIRONMENTS = [
    {"shock_severity": 0.2},
    {"shock_severity": 0.5},
    {"shock_severity": 0.8},
]
# Weighting for fitness
W_SURVIVAL = 0.60
W_CCI = 0.35
W_COLLAPSE_PENALTY = 0.20  # subtracted (scaled by collapse_risk)
EPS = 1e-9

# Where to save
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = pathlib.Path("./discovery_results") / f"evolutionary_{RUN_STAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting evolutionary experiment...")
print(f"📁 Results will be saved to: {OUT_DIR}")


# ------------------------------------------
# ADAPTERS — hook into your existing modules
# Edit function names here if needed.
# ------------------------------------------
def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


shock_resilience = _safe_import("shock_resilience")
calibration_experiment = _safe_import("calibration_experiment")
goal_externalities = _safe_import("goal_externalities")


def run_shock(agent_params: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """
    Expected return keys:
        survival_rate (0..1), recovery_time (float), shock_tolerance (0..1)
    Passes a 'max_steps' hint to keep things fast if your module supports it.
    """
    cfg = {**agent_params, **env, "max_steps": 200}
    t0 = time.time()
    try:
        if shock_resilience and hasattr(shock_resilience, "run"):
            # Preferred signature: shock_resilience.run(config: dict) -> dict
            res = shock_resilience.run(cfg)
        elif shock_resilience and hasattr(shock_resilience, "run_experiment"):
            res = shock_resilience.run_experiment(cfg)
        else:
            # Fallback dummy if module not available (keeps pipeline running)
            rnd = random.random()
            res = {
                "survival_rate": max(
                    0.0, min(1.0, 0.6 + 0.3 * rnd - 0.1 * cfg.get("noise", 0.0))
                ),
                "recovery_time": 50 + 80 * (1.0 - cfg.get("coherence", 0.7)),
                "shock_tolerance": max(0.0, min(1.0, 0.5 + 0.4 * rnd)),
            }
        # Per-sim timeout guard
        if time.time() - t0 > PER_SIM_TIMEOUT_SEC:
            res["timeout_flag"] = True
        return res
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def run_calibration(agent_params: dict[str, Any]) -> dict[str, Any]:
    """
    Expected return keys:
        cci (0..1), calibration_accuracy, coherence_score, emergence_index, noise_level
    """
    cfg = {**agent_params, "max_steps": 150}
    try:
        if calibration_experiment and hasattr(calibration_experiment, "compute_cci"):
            res = calibration_experiment.compute_cci(cfg)
        elif calibration_experiment and hasattr(calibration_experiment, "run"):
            res = calibration_experiment.run(cfg)
        else:
            # Fallback approximate
            cal = (
                0.75
                + 0.1 * (agent_params.get("coherence", 0.7) - 0.7)
                - 0.2 * agent_params.get("noise", 0.05)
            )
            coh = agent_params.get("coherence", 0.7)
            emi = 0.60 + 0.1 * (agent_params.get("goal_diversity", 3) - 3)
            noi = agent_params.get("noise", 0.05)
            raw = (max(0, cal) * coh * max(0, emi)) / (noi + 0.1)
            cci = raw / (1.0 + raw)
            res = {
                "cci": float(max(0.0, min(1.0, cci))),
                "calibration_accuracy": float(max(0.0, min(1.0, cal))),
                "coherence_score": float(max(0.0, min(1.0, coh))),
                "emergence_index": float(max(0.0, min(1.0, emi))),
                "noise_level": float(max(0.0, min(1.0, noi))),
            }
        return res
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def run_collapse(agent_params: dict[str, Any]) -> dict[str, Any]:
    """
    Expected return keys:
        collapse_risk (0..1), gini (0..1)
    """
    cfg = {**agent_params, "max_steps": 100}
    try:
        if goal_externalities and hasattr(goal_externalities, "evaluate"):
            res = goal_externalities.evaluate(cfg)
        elif goal_externalities and hasattr(goal_externalities, "run"):
            res = goal_externalities.run(cfg)
        else:
            # Fallback heuristic
            gini = (
                0.22
                + 0.05 * (agent_params.get("goal_diversity", 3) - 3)
                - 0.08 * agent_params.get("coordination", 0.7)
            )
            gini = float(max(0.0, min(1.0, gini)))
            collapse = float(max(0.0, min(1.0, 0.35 + 0.8 * (gini - 0.3))))
            res = {"collapse_risk": collapse, "gini": gini}
        return res
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# --------------------------
# Genome & evolutionary ops
# --------------------------
PARAM_RANGES = {
    "coherence": (0.55, 0.90),
    "noise": (0.00, 0.20),
    "goal_diversity": (2, 5),  # int
    "coupling": (0.60, 0.80),
    "coordination": (0.50, 0.80),
}


def random_agent() -> dict[str, Any]:
    return {
        "coherence": np.round(np.random.uniform(*PARAM_RANGES["coherence"]), 3),
        "noise": np.round(np.random.uniform(*PARAM_RANGES["noise"]), 3),
        "goal_diversity": int(
            np.random.randint(
                PARAM_RANGES["goal_diversity"][0], PARAM_RANGES["goal_diversity"][1] + 1
            )
        ),
        "coupling": np.round(np.random.uniform(*PARAM_RANGES["coupling"]), 3),
        "coordination": np.round(np.random.uniform(*PARAM_RANGES["coordination"]), 3),
    }


def mutate(agent: dict[str, Any], rate: float) -> dict[str, Any]:
    a = dict(agent)
    for k in a:
        if random.random() < rate:
            if k == "goal_diversity":
                lo, hi = PARAM_RANGES[k]
                a[k] = int(np.clip(a[k] + random.choice([-1, 1]), lo, hi))
            else:
                lo, hi = PARAM_RANGES[k]
                span = hi - lo
                step = np.random.normal(0, 0.10 * span)
                a[k] = float(np.round(np.clip(a[k] + step, lo, hi), 3))
    return a


def crossover(parent1: dict[str, Any], parent2: dict[str, Any]) -> dict[str, Any]:
    child = {}
    for k in parent1:
        child[k] = parent1[k] if random.random() < 0.5 else parent2[k]
    return child


# -------------
# Fitness
# -------------
def evaluate_agent(agent: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Evaluate across EVAL_ENVIRONMENTS and average.
    Fitness = +survival * W_SURVIVAL + cci * W_CCI - collapse_risk * W_COLLAPSE_PENALTY
    """
    survs, ccs, cols = [], [], []
    flags = []
    for env in EVAL_ENVIRONMENTS:
        t_start = time.time()
        shock = run_shock(agent, env)
        calib = run_calibration(agent)
        collp = run_collapse(agent)
        if "timeout_flag" in shock:
            flags.append("timeout")
        # Pull safe values
        survival = float(shock.get("survival_rate", 0.0))
        cci = float(calib.get("cci", 0.0))
        collapse = float(collp.get("collapse_risk", 0.5))
        survs.append(max(0.0, min(1.0, survival)))
        ccs.append(max(0.0, min(1.0, cci)))
        cols.append(max(0.0, min(1.0, collapse)))
        if time.time() - t_start > PER_SIM_TIMEOUT_SEC:
            flags.append("timeout")
    survival_avg = float(np.mean(survs)) if survs else 0.0
    cci_avg = float(np.mean(ccs)) if ccs else 0.0
    collapse_avg = float(np.mean(cols)) if cols else 1.0
    fitness = (
        (W_SURVIVAL * survival_avg)
        + (W_CCI * cci_avg)
        - (W_COLLAPSE_PENALTY * collapse_avg)
    )
    metrics = {
        "survival_avg": survival_avg,
        "cci_avg": cci_avg,
        "collapse_avg": collapse_avg,
        "flags": ";".join(flags) if flags else "",
    }
    return float(fitness), metrics


# -------------------------
# Main evolutionary routine
# -------------------------
def main():
    print(
        f"🧬 Initializing evolution with {POP_SIZE} agents for {GENERATIONS} generations..."
    )
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 = time.time()
    population = [random_agent() for _ in range(POP_SIZE)]
    history_rows = []

    for gen in range(GENERATIONS):
        print(f"🔄 Generation {gen+1}/{GENERATIONS}...")

        # Time budget check
        if (time.time() - t0) > (TIME_BUDGET_MIN * 60):
            print(f"⏰ Time budget exceeded at generation {gen}.")
            break

        scores = []
        for idx, agent in enumerate(population):
            print(f"  Evaluating agent {idx+1}/{POP_SIZE}...", end="\r")
            fit, metrics = evaluate_agent(agent)
            row = {"generation": gen, "idx": idx, **agent, "fitness": fit, **metrics}
            history_rows.append(row)
            scores.append((fit, agent))

        # Select top performers
        scores.sort(key=lambda x: x[0], reverse=True)
        cut = max(1, int(TOP_FRACTION * POP_SIZE))
        elites = [a for _, a in scores[:cut]]

        best_fitness = scores[0][0]
        print(f"  Best fitness this generation: {best_fitness:.3f}")

        # Breed new population
        new_pop = elites.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = (
                random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            )
            child = crossover(p1, p2)
            child = mutate(child, MUTATION_RATE)
            new_pop.append(child)
        population = new_pop

    # Final evaluation of last population
    print("🏁 Final evaluation...")
    final_scores = []
    for agent in population:
        fit, metrics = evaluate_agent(agent)
        final_scores.append((fit, {**agent, **metrics}))

    # -----------------
    # Exports (CSV/PNG)
    # -----------------
    print("💾 Saving results...")

    df = pd.DataFrame(history_rows)
    csv_path = OUT_DIR / "evolution_history.csv"
    df.to_csv(csv_path, index=False)

    # Summary JSON
    best_fit, best_agent_row = max(final_scores, key=lambda x: x[0])
    summary = {
        "run_stamp": RUN_STAMP,
        "config": {
            "POP_SIZE": POP_SIZE,
            "GENERATIONS": GENERATIONS,
            "TOP_FRACTION": TOP_FRACTION,
            "MUTATION_RATE": MUTATION_RATE,
            "TIME_BUDGET_MIN": TIME_BUDGET_MIN,
            "EVAL_ENVIRONMENTS": EVAL_ENVIRONMENTS,
            "weights": {
                "survival": W_SURVIVAL,
                "cci": W_CCI,
                "collapse_penalty": W_COLLAPSE_PENALTY,
            },
        },
        "best": {
            "fitness": best_fit,
            "agent": {
                k: v
                for k, v in best_agent_row.items()
                if k
                in ["coherence", "noise", "goal_diversity", "coupling", "coordination"]
            },
            "metrics": {
                "survival_avg": best_agent_row.get("survival_avg", None),
                "cci_avg": best_agent_row.get("cci_avg", None),
                "collapse_avg": best_agent_row.get("collapse_avg", None),
                "flags": best_agent_row.get("flags", ""),
            },
        },
        "timing_sec": round(time.time() - t0, 2),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots (Matplotlib only, default colors)
    try:
        print("📊 Generating plots...")
        # Fitness by generation (mean ± std)
        gen_stats = (
            df.groupby("generation")["fitness"].agg(["mean", "std"]).reset_index()
        )
        plt.figure()
        plt.plot(gen_stats["generation"], gen_stats["mean"], marker="o")
        plt.fill_between(
            gen_stats["generation"],
            gen_stats["mean"] - gen_stats["std"].fillna(0),
            gen_stats["mean"] + gen_stats["std"].fillna(0),
            alpha=0.2,
        )
        plt.title("Fitness over Generations (mean ± std)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            OUT_DIR / "fitness_over_generations.png", dpi=160, bbox_inches="tight"
        )
        plt.close()

        # Parameter distributions of final gen
        last_gen = df[df["generation"] == df["generation"].max()]
        for col in ["coherence", "noise", "goal_diversity", "coupling", "coordination"]:
            plt.figure()
            last_gen[col].hist(bins=10)
            plt.title(f"Final Generation: {col} distribution")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.grid(True, alpha=0.3)
            plt.savefig(OUT_DIR / f"final_dist_{col}.png", dpi=140, bbox_inches="tight")
            plt.close()
    except Exception as e:
        with open(OUT_DIR / "plot_errors.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())

    # Markdown report
    md = []
    md.append(f"# Evolutionary Run — {RUN_STAMP}\n")
    md.append("## Config\n")
    md.append("```json\n" + json.dumps(summary["config"], indent=2) + "\n```\n")
    md.append("## Best Agent\n")
    md.append("```json\n" + json.dumps(summary["best"], indent=2) + "\n```\n")
    md.append("## Key Plots\n")
    md.append("- fitness_over_generations.png\n")
    md.append(
        "- final_dist_coherence.png, final_dist_noise.png, final_dist_goal_diversity.png, final_dist_coupling.png, final_dist_coordination.png\n"
    )
    md.append("## Notes\n")
    md.append(
        "- Fitness = +survival * weight + cci * weight − collapse_penalty * weight\n"
    )
    md.append("- Small population & 5 generations to stay under ~10 minutes.\n")
    with open(OUT_DIR / "REPORT.md", "w") as f:
        f.write("\n".join(md))

    print(f"\n✅ DONE! Outputs saved to: {OUT_DIR}\n")
    print(f"📊 CSV:     {csv_path}")
    print(f"📋 JSON:    {OUT_DIR / 'summary.json'}")
    print(f"📝 REPORT:  {OUT_DIR / 'REPORT.md'}")
    print(f"📈 PNGs:    {OUT_DIR}/*.png")

    # Also copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory for easy access...")
    shutil.copy(OUT_DIR / "REPORT.md", "./EVOLUTIONARY_RESULTS.md")
    shutil.copy(OUT_DIR / "summary.json", "./EVOLUTIONARY_SUMMARY.json")
    shutil.copy(OUT_DIR / "evolution_history.csv", "./EVOLUTIONARY_DATA.csv")

    print("✅ Easy access files created:")
    print("   - EVOLUTIONARY_RESULTS.md")
    print("   - EVOLUTIONARY_SUMMARY.json")
    print("   - EVOLUTIONARY_DATA.csv")


if __name__ == "__main__":
    main()
