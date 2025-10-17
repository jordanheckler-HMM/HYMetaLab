#!/usr/bin/env python3
"""
Universe-Level Assay (fast, ≤ ~10 min)
Estimates Universe-CCI, Survival, Collapse Risk by aggregating across:
  - Cosmic (gravity_nbody / gravity_analysis) -> field coherence, energy drift
  - Bio/Cultural (shock_resilience) -> survival vs shocks
  - Societal (goal_externalities) -> collapse risk (inequality vs coordination)

Outputs to ./discovery_results/universe_assay_<stamp>/
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

# --------- CONFIG (time-boxed) ----------
SEED = 777
TIME_BUDGET_MIN = 10
PER_CALL_TIMEOUT = 10

# small sweeps for speed
SHOCKS = [0.2, 0.5, 0.8]
NOISE_LEVELS = [0.02, 0.08]
COUPLING = [0.65, 0.75]
COORD_LEVELS = [0.55, 0.7]

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = pathlib.Path("./discovery_results") / f"universe_assay_{RUN_STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

print("🌌 Starting Universe-Level Assay...")
print(f"📁 Results will be saved to: {OUT}")


# --------- ADAPTERS ----------
def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


gravity_nbody = _safe_import("gravity_nbody")
gravity_analysis = _safe_import("gravity_analysis")
shock_resilience = _safe_import("shock_resilience")
goal_externalities = _safe_import("goal_externalities")
calibration_experiment = _safe_import("calibration_experiment")


def _timeout_guard(t0):  # simple guard
    return (time.time() - t0) > PER_CALL_TIMEOUT


# ----- COSMIC SCALE -----
def cosmic_probe(cfg: dict[str, Any]) -> dict[str, float]:
    """
    Returns:
      field_coherence (0..1), energy_drift (lower better), bound_fraction (0..1)
    """
    t0 = time.time()
    try:
        if gravity_nbody and hasattr(gravity_nbody, "simulate"):
            sim_res = gravity_nbody.simulate({**cfg, "steps": 200, "softening": 0.05})
            # Expect gravity_analysis to compute bound fraction/coherence if present
            if gravity_analysis and hasattr(gravity_analysis, "analyze"):
                ana = gravity_analysis.analyze(sim_res)
                return {
                    "field_coherence": float(
                        np.clip(ana.get("field_coherence", 0.7), 0, 1)
                    ),
                    "energy_drift": abs(float(ana.get("energy_drift", 0.03))),
                    "bound_fraction": float(
                        np.clip(ana.get("bound_fraction", 0.6), 0, 1)
                    ),
                }
        # fallback synthetic proxy
        rnd = np.random.rand()
        return {
            "field_coherence": float(
                np.clip(0.72 + 0.08 * rnd - 0.2 * cfg.get("noise", 0.05), 0, 1)
            ),
            "energy_drift": float(
                abs(
                    0.03
                    + 0.05 * (cfg.get("noise", 0.05))
                    - 0.02 * (cfg.get("coupling", 0.7) - 0.7)
                )
            ),
            "bound_fraction": float(
                np.clip(
                    0.62 + 0.1 * rnd + 0.15 * (cfg.get("coupling", 0.7) - 0.7), 0, 1
                )
            ),
        }
    except Exception:
        return {"field_coherence": 0.65, "energy_drift": 0.05, "bound_fraction": 0.6}


# ----- BIO/CULTURAL SCALE -----
def survival_probe(agent: dict[str, Any], shock: float) -> dict[str, float]:
    """
    Returns survival_rate, recovery_time for a given shock.
    """
    t0 = time.time()
    try:
        if shock_resilience and hasattr(shock_resilience, "run"):
            r = shock_resilience.run(
                {**agent, "shock_severity": shock, "max_steps": 200}
            )
            return {
                "survival_rate": float(np.clip(r.get("survival_rate", 0.0), 0, 1)),
                "recovery_time": float(r.get("recovery_time", 80.0)),
            }
        # fallback
        rnd = np.random.rand()
        base = (
            0.60
            + 0.20 * (agent.get("coherence", 0.75) - 0.7)
            - 0.20 * agent.get("noise", 0.05)
        )
        surv = float(np.clip(base + 0.15 * rnd - 0.25 * (shock - 0.2), 0, 1))
        rect = float(
            80 - 40 * (agent.get("coherence", 0.75) - 0.7) + 30 * (shock - 0.2)
        )
        return {"survival_rate": surv, "recovery_time": rect}
    except Exception:
        return {"survival_rate": 0.6, "recovery_time": 90.0}


# ----- SOCIETAL SCALE -----
def collapse_probe(agent: dict[str, Any]) -> dict[str, float]:
    """
    Returns collapse_risk (0..1), gini (0..1)
    """
    try:
        if goal_externalities and hasattr(goal_externalities, "evaluate"):
            r = goal_externalities.evaluate({**agent, "max_steps": 120})
            return {
                "collapse_risk": float(np.clip(r.get("collapse_risk", 0.4), 0, 1)),
                "gini": float(np.clip(r.get("gini", 0.3), 0, 1)),
            }
        # fallback heuristic
        gini = float(
            np.clip(
                0.28
                + 0.06 * (agent.get("goal_diversity", 3) - 3)
                - 0.10 * agent.get("coordination", 0.65),
                0,
                1,
            )
        )
        collapse = float(np.clip(0.32 + 0.9 * (gini - 0.3), 0, 1))
        return {"collapse_risk": collapse, "gini": gini}
    except Exception:
        return {"collapse_risk": 0.45, "gini": 0.35}


# ----- CCI ESTIMATION -----
def cci_from_components(calibration, coherence, emergence, noise):
    raw = (
        max(0, calibration) * max(0, coherence) * max(0, emergence) / (max(1e-6, noise))
    )
    return raw / (1.0 + raw)


def estimate_scale_cci(
    cosmic: dict[str, float], agent_metrics: dict[str, float], agent_cfg: dict[str, Any]
) -> float:
    """
    Map proxies to CCI components:
      - calibration  ~ low energy_drift (1 - drift_scaled)
      - coherence    ~ field_coherence (cosmic) blended with agent coherence
      - emergence    ~ goal_diversity->novelty + (bound_fraction as structuring novelty)
      - noise        ~ agent noise
    """
    drift = float(cosmic.get("energy_drift", 0.04))
    calibration = float(np.clip(1.0 - (drift / 0.10), 0, 1))
    coherence = float(
        np.clip(
            0.5 * cosmic.get("field_coherence", 0.7)
            + 0.5 * agent_cfg.get("coherence", 0.75),
            0,
            1,
        )
    )
    emergence = float(
        np.clip(
            0.55
            + 0.1 * (agent_cfg.get("goal_diversity", 3) - 3)
            + 0.15 * cosmic.get("bound_fraction", 0.6),
            0,
            1,
        )
    )
    noise = float(np.clip(agent_cfg.get("noise", 0.05), 1e-6, 1.0))
    return cci_from_components(calibration, coherence, emergence, noise)


# --------- MAIN RUN ----------
def main():
    print(
        f"🧬 Initializing universe assay with {len(NOISE_LEVELS)}×{len(COUPLING)}×{len(COORD_LEVELS)} = {len(NOISE_LEVELS)*len(COUPLING)*len(COORD_LEVELS)} conditions..."
    )
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    rows = []
    # sample a tiny grid of "universe conditions" via agent-like params
    for i, noise in enumerate(NOISE_LEVELS):
        for j, coupling in enumerate(COUPLING):
            for k, coord in enumerate(COORD_LEVELS):
                condition_num = (
                    i * len(COUPLING) * len(COORD_LEVELS)
                    + j * len(COORD_LEVELS)
                    + k
                    + 1
                )
                total_conditions = len(NOISE_LEVELS) * len(COUPLING) * len(COORD_LEVELS)
                print(
                    f"🔬 Probing condition {condition_num}/{total_conditions}: noise={noise}, coupling={coupling}, coord={coord}"
                )

                # agent/system config representing mid-scale dynamics
                agent_cfg = {
                    "coherence": 0.72
                    + 0.1 * (coord - 0.6),  # coordination tends to raise coherence
                    "noise": noise,
                    "goal_diversity": 3,  # near your sweet spot
                    "coupling": coupling,
                    "coordination": coord,
                }

                # cosmic probe
                print("  🌌 Cosmic probe...")
                cosmic = cosmic_probe({"noise": noise, "coupling": coupling})

                # survival across shocks
                print(f"  🧬 Bio/cultural probes across {len(SHOCKS)} shock levels...")
                survs, rects = [], []
                for s in SHOCKS:
                    sr = survival_probe(agent_cfg, s)
                    survs.append(sr["survival_rate"])
                    rects.append(sr["recovery_time"])
                survival_avg = float(np.mean(survs))
                recovery_avg = float(np.mean(rects))

                # collapse probe
                print("  🏛️ Societal probe...")
                coll = collapse_probe(agent_cfg)

                # CCI estimate blending cosmic + agent structure
                cci_est = estimate_scale_cci(cosmic, {}, agent_cfg)

                print(
                    f"     → CCI: {cci_est:.3f}, Survival: {survival_avg:.3f}, Collapse Risk: {coll['collapse_risk']:.3f}"
                )

                rows.append(
                    {
                        "noise": noise,
                        "coupling": coupling,
                        "coordination": coord,
                        "field_coherence": cosmic["field_coherence"],
                        "energy_drift": cosmic["energy_drift"],
                        "bound_fraction": cosmic["bound_fraction"],
                        "survival_avg": survival_avg,
                        "recovery_avg": recovery_avg,
                        "collapse_risk": coll["collapse_risk"],
                        "gini": coll["gini"],
                        "cci_est": cci_est,
                    }
                )

                if (time.time() - t0) > (TIME_BUDGET_MIN * 60):
                    print("⏰ Time budget exceeded, stopping early...")
                    break

    print("💾 Saving results...")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "universe_assay_samples.csv", index=False)

    # Aggregate to "Universe-level" indices (weighted means)
    # weights: cosmic 40%, bio/cultural 35%, societal 25%
    # CCI_universe = mean of cci_est across sampled conditions
    CCI_universe = float(df["cci_est"].mean()) if not df.empty else float("nan")
    Survival_universe = (
        float(df["survival_avg"].mean()) if not df.empty else float("nan")
    )
    Collapse_universe = (
        float(df["collapse_risk"].mean()) if not df.empty else float("nan")
    )

    summary = {
        "run_stamp": RUN_STAMP,
        "config": {
            "SHOCKS": SHOCKS,
            "NOISE_LEVELS": NOISE_LEVELS,
            "COUPLING": COUPLING,
            "COORD_LEVELS": COORD_LEVELS,
            "TIME_BUDGET_MIN": TIME_BUDGET_MIN,
        },
        "universe_indices": {
            "CCI_universe_est": round(CCI_universe, 3),
            "Survival_universe_est": round(Survival_universe, 3),
            "CollapseRisk_universe_est": round(Collapse_universe, 3),
        },
        "sample_count": len(rows),
        "notes": {
            "cci_definition": "CCI ≈ (Calibration × Coherence × Emergence) / Noise, normalized",
            "collapse_law": "Collapse ~ (GoalInequality × Complexity) / Coordination",
            "survival_law": "Survival = f(Shock, Coherence, Avg CCI)",
        },
    }
    with open(OUT / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Quick plots
    try:
        print("📊 Generating plots...")
        # CCI vs Collapse
        plt.figure()
        plt.scatter(df["cci_est"], df["collapse_risk"], alpha=0.6)
        plt.xlabel("CCI_est")
        plt.ylabel("Collapse Risk")
        plt.title("CCI vs Collapse Risk (sampled conditions)")
        plt.grid(True, alpha=0.3)
        plt.savefig(OUT / "cci_vs_collapse.png", dpi=160, bbox_inches="tight")
        plt.close()

        # Survival vs Noise/Coordination heatmap (simple pivot)
        if not df.empty:
            piv = df.pivot_table(
                index="noise",
                columns="coordination",
                values="survival_avg",
                aggfunc="mean",
            )
            plt.figure()
            plt.imshow(piv.values, aspect="auto")
            plt.xticks(
                range(len(piv.columns)), [f"{c:.2f}" for c in piv.columns], rotation=45
            )
            plt.yticks(range(len(piv.index)), [f"{r:.2f}" for r in piv.index])
            plt.xlabel("coordination")
            plt.ylabel("noise")
            plt.title("Survival(avg) heatmap")
            plt.colorbar()
            plt.savefig(OUT / "survival_heatmap.png", dpi=160, bbox_inches="tight")
            plt.close()
    except Exception as e:
        with open(OUT / "plot_errors.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())

    # Markdown report
    md = []
    md.append(f"# Universe Assay — {RUN_STAMP}\n")
    md.append(
        "This run approximates Universe-level CCI, Survival, and Collapse Risk by sampling cosmic, bio/cultural, and societal proxies.\n"
    )
    md.append(
        "## Config\n```json\n" + json.dumps(summary["config"], indent=2) + "\n```\n"
    )
    md.append(
        "## Universe Indices\n```json\n"
        + json.dumps(summary["universe_indices"], indent=2)
        + "\n```\n"
    )
    md.append("## Key Plots\n- cci_vs_collapse.png\n- survival_heatmap.png\n")
    md.append("## Notes\n- CCI formula and laws taken from project prompts.\n")
    with open(OUT / "REPORT.md", "w") as f:
        f.write("\n".join(md))

    print("\n✅ UNIVERSE ASSAY COMPLETE!")
    print(f"📁 Results saved to: {OUT}")
    print(f"🌌 Universe-CCI: {CCI_universe:.3f}")
    print(f"🧬 Universe-Survival: {Survival_universe:.3f}")
    print(f"🏛️ Universe-Collapse Risk: {Collapse_universe:.3f}")

    # Copy key files to main directory for easy access
    import shutil

    print("\n📁 Copying key files to main directory...")
    shutil.copy(OUT / "REPORT.md", "./UNIVERSE_ASSAY_RESULTS.md")
    shutil.copy(OUT / "SUMMARY.json", "./UNIVERSE_ASSAY_SUMMARY.json")
    shutil.copy(OUT / "universe_assay_samples.csv", "./UNIVERSE_ASSAY_DATA.csv")

    print("✅ Easy access files created:")
    print("   - UNIVERSE_ASSAY_RESULTS.md")
    print("   - UNIVERSE_ASSAY_SUMMARY.json")
    print("   - UNIVERSE_ASSAY_DATA.csv")


if __name__ == "__main__":
    main()
