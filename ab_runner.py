#!/usr/bin/env python3
# ab_runner.py — run small A/B smoke tests for Religion enabled vs disabled
import meaning_experiment as me

SEED = 201
AGENTS = [100, 200]
GOALS = [2, 3, 4, 5]
NOISE = [0.05, 0.1]

RELIGION_ENABLED = {"allowed_branches": ["Religion", "Education", "Fatalism"]}
RELIGION_DISABLED = {"allowed_branches": ["Education", "Fatalism"]}

# ENABLED quick smoke
me.EXTRA_RUN_KWARGS = {"branch_mask": RELIGION_ENABLED, "seed": SEED}
print("Running A ENABLED smoke")
me.run_experiment_grid(
    label=f"AB_HS_EN_seed{SEED}",
    agents_list=AGENTS,
    shocks=[0.9],
    stress_duration=["acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=3,
)

# DISABLED quick smoke
me.EXTRA_RUN_KWARGS = {"branch_mask": RELIGION_DISABLED, "seed": SEED}
print("Running A DISABLED smoke")
me.run_experiment_grid(
    label=f"AB_HS_DIS_seed{SEED}",
    agents_list=AGENTS,
    shocks=[0.9],
    stress_duration=["acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=3,
)

print(
    "AB smoke runs complete; outputs written under discovery_results (check latest folders)"
)
