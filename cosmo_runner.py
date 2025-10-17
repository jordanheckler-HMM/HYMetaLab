#!/usr/bin/env python3
# cosmo_runner.py — run a smoke test for cosmic openness + heat-death
from importlib import reload

import meaning_experiment as me

reload(me)

SEED = 451
AGENTS = [200, 400]
GOALS = [3, 4, 5]
NOISE = [0.05, 0.1]
SHOCKS = [0.3, 0.5, 0.8]

OPEN_AGENTIO = {"mechanism": "agent_io", "epsilon": 0.0025, "period": 40}
OPEN_CHEMO = {"mechanism": "chemostat", "epsilon": 0.0025, "period": 40}

HEAT_DEATH_SCHEDULE = [
    {"epoch": 0, "epsilon": 0.0030},
    {"epoch": 2000, "epsilon": 0.0020},
    {"epoch": 4000, "epsilon": 0.0010},
    {"epoch": 6000, "epsilon": 0.0005},
    {"epoch": 8000, "epsilon": 0.0002},
    {"epoch": 9500, "epsilon": 0.0000},
]

# Agent-I/O baseline
me.EXTRA_RUN_KWARGS = {"openness": OPEN_AGENTIO, "seed": SEED}
print("Running COSMO_AGENTIO smoke...")
me.run_experiment_grid(
    label=f"COSMO_AGENTIO_seed{SEED}",
    agents_list=AGENTS,
    shocks=SHOCKS,
    stress_duration=["chronic", "acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=2,
)

# Chemostat baseline
me.EXTRA_RUN_KWARGS = {"openness": OPEN_CHEMO, "seed": SEED}
print("Running COSMO_CHEMO smoke...")
me.run_experiment_grid(
    label=f"COSMO_CHEMO_seed{SEED}",
    agents_list=AGENTS,
    shocks=SHOCKS,
    stress_duration=["chronic", "acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=2,
)

# Heat-death AgentIO
me.EXTRA_RUN_KWARGS = {
    "openness": {"mechanism": "agent_io", "epsilon": 0.0030, "period": 40},
    "heat_death": {
        "enabled": True,
        "schedule": HEAT_DEATH_SCHEDULE,
        "log_fraction": ["Religion", "Education", "Fatalism"],
    },
    "logging": {"dense_window": [9300, 9700], "thin_stride": 10},
    "seed": SEED,
}
print("Running COSMO_HEATDEATH_AGENTIO smoke...")
me.run_experiment_grid(
    label=f"COSMO_HEATDEATH_AGENTIO_seed{SEED}",
    agents_list=AGENTS,
    shocks=[0.2, 0.4, 0.6],
    stress_duration=["chronic", "acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=2,
)

# Heat-death Chemo
me.EXTRA_RUN_KWARGS = {
    "openness": {"mechanism": "chemostat", "epsilon": 0.0030, "period": 40},
    "heat_death": {
        "enabled": True,
        "schedule": HEAT_DEATH_SCHEDULE,
        "log_fraction": ["Religion", "Education", "Fatalism"],
    },
    "logging": {"dense_window": [9300, 9700], "thin_stride": 10},
    "seed": SEED,
}
print("Running COSMO_HEATDEATH_CHEMO smoke...")
me.run_experiment_grid(
    label=f"COSMO_HEATDEATH_CHEMO_seed{SEED}",
    agents_list=AGENTS,
    shocks=[0.2, 0.4, 0.6],
    stress_duration=["chronic", "acute"],
    goal_diversity=GOALS,
    noise_list=NOISE,
    replicates=2,
)

print("Cosmo smoke tests complete")
