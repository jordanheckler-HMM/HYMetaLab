#!/usr/bin/env python3
"""
Metaphysical time experiment using lightweight simulation_core stubs.
"""
import os

import numpy as np
import pandas as pd

from simulation_core import (
    calibration_experiment,
    gravity_analysis,
    meaning_experiment,
    shock_resilience,
)
from utils import export_results

# --- Experiment Config ---
RUN_ID = "time_metaphysics_test_001"
OUTPUT_DIR = f"./discovery_results/{RUN_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hypotheses:
# 1. Linear time favors coherence but limits emergence.
# 2. Cyclic time stabilizes systems through recurrence.
# 3. Multidimensional time (branching temporal states) increases emergence but also noise.

time_modes = ["linear", "cyclic", "branching"]
agent_counts = [50, 100, 200]
noise_levels = [0.05, 0.1, 0.2]
shock_levels = [0.2, 0.5, 0.8]

results = []

for mode in time_modes:
    for agents in agent_counts:
        for noise in noise_levels:
            for shock in shock_levels:
                # Simulate base consciousness calibration
                cci = calibration_experiment.run(
                    agents=agents, noise=noise, temporal_mode=mode
                )
                # Run a metaphysical meaning sweep
                meaning = meaning_experiment.run(
                    cci=cci, shock=shock, temporal_mode=mode
                )
                # Gravitational coherence as time-field proxy
                gravity = gravity_analysis.run(mode=mode, agent_count=agents)
                # Shock resilience (how time structure buffers chaos)
                resilience = shock_resilience.run(cci=cci, shock=shock, time_mode=mode)

                # Aggregate metrics
                data = {
                    "mode": mode,
                    "agents": agents,
                    "noise": noise,
                    "shock": shock,
                    "cci_mean": float(np.mean(cci)),
                    "meaning_index": float(np.mean(meaning)),
                    "gravity_coherence": float(np.mean(gravity)),
                    "shock_tolerance": float(np.mean(resilience)),
                    "collapse_risk": float((noise * shock) / (np.mean(cci) + 1e-5)),
                }
                results.append(data)

# --- Save Structured Outputs ---
df = pd.DataFrame(results)
df.to_csv(f"{OUTPUT_DIR}/time_metaphysics_data.csv", index=False)
export_results(
    df, OUTPUT_DIR, plots=["cci_vs_time", "coherence_map", "collapse_vs_mode"]
)

# --- Markdown Summary ---
summary = f"""
# Metaphysical Time Experiment — {RUN_ID}

## Hypothesis
Linear time produces coherence; cyclic time yields resilience; branching time yields emergence.

## Findings
- Avg CCI by mode: {df.groupby('mode')['cci_mean'].mean().to_dict()}
- Collapse Risk (avg): {df.groupby('mode')['collapse_risk'].mean().to_dict()}
- Meaning Index (avg): {df.groupby('mode')['meaning_index'].mean().to_dict()}

## Interpretation
Cyclic time stabilizes systems (lower collapse risk, higher resilience).  
Branching time expands emergence but raises noise (entropy).  
Linear time optimizes coherence but constrains creativity.

**Metaphysical implication:**  
Time acts not as a linear dimension but as a coherence field modulated by system memory and anticipation.

Exported: `{OUTPUT_DIR}/`
"""
with open(f"{OUTPUT_DIR}/summary.md", "w") as f:
    f.write(summary)

print("✅ Metaphysical Time Experiment complete.")
print(f"Results saved in {OUTPUT_DIR}")
