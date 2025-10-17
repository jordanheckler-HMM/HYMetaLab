#!/usr/bin/env python3
# uni_runner.py — Cosmic-scale closed vs minimal-open smoke test
from importlib import reload

import meaning_experiment as me

reload(me)

SEED = 401
AGENTS = [200, 400]
GOALS = [3, 4, 5]
NOISE = [0.05, 0.1]
SHOCKS = [0.3, 0.5, 0.8]

CLOSED = {"mechanism": "closed", "epsilon": 0.0, "period": None}
OPEN_AgentIO = {"mechanism": "agent_io", "epsilon": 0.0025, "period": 40}
OPEN_Chemostat = {"mechanism": "chemostat", "epsilon": 0.0025, "period": 40}

# Closed
me.EXTRA_RUN_KWARGS = {"openness": CLOSED, "seed": SEED}
print("Running UNI_CLOSED smoke...")
me.run_experiment_grid(
    label=f"UNI_CLOSED_seed{SEED}",
    agents_list=AGENTS,
    shocks=SHOCKS,
    stress_duration=["chronic", "acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=3,
)

# Agent IO
me.EXTRA_RUN_KWARGS = {"openness": OPEN_AgentIO, "seed": SEED}
print("Running UNI_AGENTIO smoke...")
me.run_experiment_grid(
    label=f"UNI_AGENTIO_seed{SEED}",
    agents_list=AGENTS,
    shocks=SHOCKS,
    stress_duration=["chronic", "acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=3,
)

# Chemostat
me.EXTRA_RUN_KWARGS = {"openness": OPEN_Chemostat, "seed": SEED}
print("Running UNI_CHEMO smoke...")
me.run_experiment_grid(
    label=f"UNI_CHEMO_seed{SEED}",
    agents_list=AGENTS,
    shocks=SHOCKS,
    stress_duration=["chronic", "acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=3,
)

print("Universe smoke tests complete")
