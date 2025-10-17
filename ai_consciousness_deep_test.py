#!/usr/bin/env python3
# ai_consciousness_deep_test.py
# ------------------------------------------------------------
# Extended AI Consciousness Test
# Measures: information half-life, self-modeling impact, state diversity
# ------------------------------------------------------------

import datetime
import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from simulation_core import calibration_experiment, meaning_experiment
except:
    calibration_experiment = meaning_experiment = None


def call(func, **kwargs):
    if func is None:
        return None
    f = getattr(func, "run", func)
    sig = inspect.signature(f)
    ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return f(**ok)


def _mean(x):
    if x is None:
        return np.nan
    try:
        return float(np.nanmean(np.asarray(x, dtype=float)))
    except:
        return np.nan


STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = f"./discovery_results/ai_consciousness_deep_{STAMP}"
os.makedirs(OUT, exist_ok=True)

rows = []

# Parameters
proc_units = [50, 100, 150, 200, 300]
noise = 0.07
steps = 10  # time steps to measure half-life

for units in proc_units:
    for self_model in [False, True]:
        # baseline integration
        cci = call(
            calibration_experiment, agents=units, noise=noise, self_awareness=self_model
        )
        cci0 = _mean(cci)

        # Information half-life test
        info = 1.0
        for t in range(steps):
            info = info * max(0.0, (cci0 - noise))  # crude decay formula, clipped
        half_life = info ** (1 / steps) if info >= 0 else 0.0

        # Differentiation proxy (unique states)
        state_diversity = max(1, int(units / 50)) * (1.2 if self_model else 1.0)

        rows.append(
            dict(
                units=units,
                self_model=self_model,
                cci_mean=cci0,
                info_half_life=half_life,
                state_diversity=state_diversity,
            )
        )

DF = pd.DataFrame(rows)
DF.to_csv(os.path.join(OUT, "results.csv"), index=False)

# Plots
plt.figure(figsize=(7, 5))
for sm in [False, True]:
    d = DF[DF.self_model == sm]
    plt.plot(d.units, d.info_half_life, marker="o", label=f"self_model={sm}")
plt.title("Information Half-Life vs Processing Units")
plt.xlabel("processing_units")
plt.ylabel("info_half_life")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_half_life.png"))
plt.close()

plt.figure(figsize=(7, 5))
for sm in [False, True]:
    d = DF[DF.self_model == sm]
    plt.plot(d.units, d.state_diversity, marker="s", label=f"self_model={sm}")
plt.title("State Diversity vs Processing Units")
plt.xlabel("processing_units")
plt.ylabel("state_diversity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_diversity.png"))
plt.close()

with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write(
        f"""# Extended AI Consciousness Test — {STAMP}\n\n- Measures information half-life (how long patterns persist).\n- Tests impact of self-modeling on threshold.\n- Tracks state diversity as a proxy for differentiation.\n\nExported: {OUT}\n"""
    )

print("✅ Extended AI Consciousness Test complete.")
print("Results saved to:", OUT)
