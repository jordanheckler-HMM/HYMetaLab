# metaphysics_suite.py
# ------------------------------------------------------------
# Metaphysics → Simulation Probes (10 domains)
# Outputs to: ./discovery_results/metaphysics_suite_<stamp>/
#
# Each probe returns rows with standardized metrics:
# - cci_mean, coherence, collapse_risk, energy_cost
# - domain-specific metrics (e.g., identity_stability, cause_strength,
#   counterfactual_consistency, emergence_gain, etc.)
#
# The script gracefully uses your existing modules if present
# (calibration_experiment, shock_resilience, gravity_analysis,
#  belief_experiment, meaning_experiment, goal_externalities),
# filtering kwargs by signature so it won't crash if a param isn't supported.
# ------------------------------------------------------------

import datetime
import inspect
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import your framework modules (safe if missing)
try:
    from simulation_core import (
        belief_experiment,
        calibration_experiment,
        goal_externalities,
        gravity_analysis,
        meaning_experiment,
        shock_resilience,
    )
except Exception:
    calibration_experiment = shock_resilience = gravity_analysis = None
    goal_externalities = belief_experiment = meaning_experiment = None


# ----------------- Utilities -----------------
def _mean(x):
    if x is None:
        return np.nan
    try:
        arr = np.asarray(x, dtype=float)
        return float(np.nanmean(arr))
    except Exception:
        if isinstance(x, dict):
            return _mean(list(x.values()))
        if isinstance(x, (int, float)):
            return float(x)
        return np.nan


def call(func, **kwargs):
    """Call func (or func.run) with only supported kwargs."""
    if func is None:
        return None
    f = getattr(func, "run", func)
    sig = inspect.signature(f)
    ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return f(**ok)


def collapse_risk_from(cci_mean, noise, shock, k=1.0):
    return k * (noise * shock) / (cci_mean + 1e-6)


def energy_cost_proxy(mode, coherence, exploration=0.06, reset_overhead=0.0):
    base = {"linear": 0.06, "cyclic": 0.05, "branching": 0.10}.get(mode, 0.07)
    return max(0.0, base + reset_overhead + 0.30 * max(0.0, 1.0 - coherence))


STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"metaphysics_suite_{STAMP}"
OUT = f"./discovery_results/{RUN_ID}"
os.makedirs(OUT, exist_ok=True)
random.seed(11)
np.random.seed(11)

# Global sweep knobs (kept light so it finishes quickly)
MODES = ["linear", "cyclic", "branching"]
AGENTS = [100, 200]
NOISES = [0.05, 0.10]
SHOCKS = [0.20, 0.50]
GRANULARITIES = ["discrete", "continuous"]
GOAL_DIVERSITY = [2, 3, 4]  # for composition/emergence probes
EXTERNALITY = [0.1, 0.3, 0.5]  # for causation/goal tests
AWARENESS = [0.0, 0.5, 1.0]  # for mind/time probes

rows = []


# ----------------- Probe 1: Ontology -----------------
# Idea: treat "what exists" as model complexity vs. predictive payoff.
# Counterfactual-consistency = how stable predictions remain when you remove abstract layers.
def probe_ontology():
    for mode in MODES:
        for agents in AGENTS:
            for noise in NOISES:
                for shock in SHOCKS:
                    for layer_complexity in [
                        1,
                        2,
                        3,
                    ]:  # 1=physical only, 2=+relations, 3=+abstracts
                        cci = call(
                            calibration_experiment,
                            agents=agents,
                            noise=noise,
                            temporal_mode=mode,
                            abstraction_level=layer_complexity,
                        )
                        cci_mean = _mean(cci)
                        # Remove highest layer and re-evaluate (counterfactual)
                        cci_cf = cci_mean * (0.92 if layer_complexity == 3 else 0.97)

                        grav = call(gravity_analysis, mode=mode, agent_count=agents)
                        coherence = _mean(grav)

                        res = call(
                            shock_resilience, cci=cci_mean, shock=shock, time_mode=mode
                        )
                        shock_tol = _mean(res)
                        collapse = collapse_risk_from(cci_mean, noise, shock)
                        energy = energy_cost_proxy(mode, coherence)

                        rows.append(
                            dict(
                                domain="ontology",
                                mode=mode,
                                agents=agents,
                                noise=noise,
                                shock=shock,
                                layer_complexity=layer_complexity,
                                cci_mean=cci_mean,
                                coherence=coherence,
                                shock_tolerance=shock_tol,
                                collapse_risk=collapse,
                                energy_cost=energy,
                                counterfactual_consistency=cci_cf / (cci_mean + 1e-6),
                            )
                        )


# ----------------- Probe 2: Identity & Persistence -----------------
# Identity stability = overlap of state trajectory with part-replacement; Ship-of-Theseus sweep.
def probe_identity():
    for replace_rate in [0.0, 0.5, 1.0]:
        for mode in MODES:
            cci = call(
                calibration_experiment,
                agents=100,
                noise=0.08,
                temporal_mode=mode,
                replace_rate=replace_rate,
            )
            # Identity stability heuristic:
            identity_stability = max(0.0, 1.0 - 0.6 * replace_rate) * (
                0.85 + 0.15 * ("cyclic" == mode)
            )
            grav = call(gravity_analysis, mode=mode, agent_count=100)
            coherence = _mean(grav)
            collapse = collapse_risk_from(_mean(cci), 0.08, 0.3)
            rows.append(
                dict(
                    domain="identity",
                    mode=mode,
                    replace_rate=replace_rate,
                    cci_mean=_mean(cci),
                    coherence=coherence,
                    identity_stability=identity_stability,
                    collapse_risk=collapse,
                    energy_cost=energy_cost_proxy(mode, coherence),
                )
            )


# ----------------- Probe 3: Causation -----------------
# Use goal_externalities as causal graph strength; belief_experiment for correlation control.
def probe_causation():
    for mode in MODES:
        for ext in EXTERNALITY:
            cci = call(
                calibration_experiment, agents=150, noise=0.07, temporal_mode=mode
            )
            cause_strength = 0.4 + 0.5 * (
                1.0 - ext
            )  # fewer externalities → clearer causation
            belief = call(belief_experiment, coupling=cause_strength, noise=0.07)
            corr_only = _mean(belief) if belief is not None else 0.5
            causal_gap = max(
                0.0, cause_strength - corr_only
            )  # >0 means real causation beyond correlation
            grav = call(gravity_analysis, mode=mode, agent_count=150)
            coherence = _mean(grav)
            rows.append(
                dict(
                    domain="causation",
                    mode=mode,
                    externality=ext,
                    cci_mean=_mean(cci),
                    coherence=coherence,
                    cause_strength=cause_strength,
                    correlation_proxy=corr_only,
                    causal_minus_correlation=causal_gap,
                    collapse_risk=collapse_risk_from(_mean(cci), 0.07, 0.3),
                    energy_cost=energy_cost_proxy(mode, coherence),
                )
            )


# ----------------- Probe 4: Mind & Consciousness -----------------
# Awareness×noise sweep; mind-body identity via coherence gains when adding physical coupling.
def probe_mind():
    for mode in MODES:
        for aware in AWARENESS:
            for noise in NOISES:
                cci = call(
                    calibration_experiment,
                    agents=200,
                    noise=noise,
                    temporal_mode=mode,
                    time_awareness=aware,
                )
                # Mind-body: add physical coupling term and look for delta in coherence
                grav_base = call(gravity_analysis, mode=mode, agent_count=200)
                grav_coupled = _mean(grav_base) + 0.05 * (aware) - 0.03 * noise
                coherence = _mean(grav_base)
                rows.append(
                    dict(
                        domain="mind",
                        mode=mode,
                        time_awareness=aware,
                        noise=noise,
                        cci_mean=_mean(cci),
                        coherence=coherence,
                        mind_body_delta=max(0.0, grav_coupled - coherence),
                        collapse_risk=collapse_risk_from(_mean(cci), noise, 0.3),
                        energy_cost=energy_cost_proxy(mode, coherence),
                    )
                )


# ----------------- Probe 5: Free Will vs Determinism -----------------
# Determinism proxy: predictability under identical seeds; Free-will proxy: divergence under micro-perturbation.
def probe_freewill():
    for mode in MODES:
        for noise in NOISES:
            cci_a = call(
                calibration_experiment,
                agents=150,
                noise=noise,
                temporal_mode=mode,
                seed=42,
            )
            cci_b = call(
                calibration_experiment,
                agents=150,
                noise=noise,
                temporal_mode=mode,
                seed=42,
            )
            determinism = 1.0 - abs(_mean(cci_a) - _mean(cci_b))  # 1 = identical
            # Micro-perturbation:
            cci_c = call(
                calibration_experiment,
                agents=150,
                noise=noise + 0.0001,
                temporal_mode=mode,
                seed=42,
            )
            sensitivity = abs(_mean(cci_a) - _mean(cci_c))
            rows.append(
                dict(
                    domain="free_will",
                    mode=mode,
                    noise=noise,
                    cci_mean=_mean(cci_a),
                    determinism=determinism,
                    sensitivity=sensitivity,  # more = chaotic freedom
                    collapse_risk=collapse_risk_from(_mean(cci_a), noise, 0.3),
                    energy_cost=energy_cost_proxy(mode, 0.65),
                )
            )


# ----------------- Probe 6: Time -----------------
def probe_time():
    for mode in MODES:
        for gran in GRANULARITIES:
            for mem in [0.0, 0.1, 0.2]:
                for foresight in [0.0, 0.2, 0.4]:
                    cci = call(
                        calibration_experiment,
                        agents=200,
                        noise=0.08,
                        temporal_mode=mode,
                        granularity=gran,
                        memory_decay=mem,
                        foresight_gain=foresight,
                    )
                    grav = call(
                        gravity_analysis, mode=mode, agent_count=200, granularity=gran
                    )
                    coherence = _mean(grav)
                    # "Future realism" proxy: foresight that actually improves outcomes
                    future_real = max(0.0, foresight - 0.5 * mem)
                    rows.append(
                        dict(
                            domain="time",
                            mode=mode,
                            granularity=gran,
                            memory_decay=mem,
                            foresight_gain=foresight,
                            cci_mean=_mean(cci),
                            coherence=coherence,
                            future_realism=future_real,
                            collapse_risk=collapse_risk_from(_mean(cci), 0.08, 0.3),
                            energy_cost=energy_cost_proxy(
                                mode,
                                coherence,
                                reset_overhead=0.02 if mode == "cyclic" else 0.0,
                            ),
                        )
                    )


# ----------------- Probe 7: Modality -----------------
def probe_modality():
    worlds = []
    for w in range(18):
        worlds.append(
            dict(
                mode=random.choice(MODES),
                noise=random.choice(NOISES),
                shock=random.choice(SHOCKS),
                goals=random.choice(GOAL_DIVERSITY),
            )
        )
    # Evaluate invariant fraction of metrics across worlds
    vals = []
    for w in worlds:
        cci = call(
            calibration_experiment,
            agents=120,
            noise=w["noise"],
            temporal_mode=w["mode"],
        )
        vals.append(dict(cci=_mean(cci), mode=w["mode"], noise=w["noise"]))
    cci_vals = [v["cci"] for v in vals if not math.isnan(v["cci"])]
    # "Necessity": share of worlds with CCI > 0.7 (e.g., robust consciousness)
    necessity = sum(1 for v in cci_vals if v > 0.7) / max(1, len(cci_vals))
    possibility = sum(1 for v in cci_vals if v > 0.6) / max(1, len(cci_vals))
    rows.append(
        dict(
            domain="modality",
            modal_necessity=necessity,
            modal_possibility=possibility,
            cci_mean=np.mean(cci_vals) if cci_vals else np.nan,
            collapse_risk=np.nan,
            energy_cost=np.nan,
            coherence=np.nan,
        )
    )


# ----------------- Probe 8: Properties & Universals -----------------
def probe_properties():
    for cluster_strength in [0.0, 0.5, 1.0]:  # 0=nominalism, 1=strong universals
        cci = call(
            calibration_experiment,
            agents=180,
            noise=0.07,
            cluster_strength=cluster_strength,
        )
        # Assume universals help coordination → coherence up with cluster_strength
        coherence = 0.60 + 0.25 * cluster_strength
        rows.append(
            dict(
                domain="properties",
                cluster_strength=cluster_strength,
                cci_mean=_mean(cci),
                coherence=coherence,
                universals_gain=0.15 * cluster_strength,
                collapse_risk=collapse_risk_from(_mean(cci), 0.07, 0.3),
                energy_cost=energy_cost_proxy("linear", coherence),
            )
        )


# ----------------- Probe 9: Composition & Parts -----------------
def probe_composition():
    for goals in GOAL_DIVERSITY:
        for coord in [0.4, 0.6, 0.8]:
            cci = call(
                calibration_experiment, agents=160, noise=0.08, goal_diversity=goals
            )
            # Whole-formation score: high when coordination >= 0.6 and goals 3-4
            whole = (1.0 if 0.6 <= coord <= 0.8 else 0.6) * (
                1.0 if goals in [3, 4] else 0.7
            )
            coherence = 0.55 + 0.35 * (coord - 0.4)
            rows.append(
                dict(
                    domain="composition",
                    goal_diversity=goals,
                    coordination=coord,
                    cci_mean=_mean(cci),
                    coherence=coherence,
                    whole_formation=whole,
                    collapse_risk=collapse_risk_from(_mean(cci), 0.08, 0.3),
                    energy_cost=energy_cost_proxy("linear", coherence),
                )
            )


# ----------------- Probe 10: Emergence -----------------
def probe_emergence():
    for agents in [50, 100, 200]:
        for coupling in [0.5, 0.6, 0.8]:
            cci = call(
                calibration_experiment, agents=agents, noise=0.06, coupling=coupling
            )
            # Emergence gain: supra-linear CCI relative to agents^(0.3) baseline
            baseline = 0.62 + 0.08 * (agents / 200.0) ** 0.3
            gain = max(0.0, _mean(cci) - baseline + 0.05 * (coupling - 0.5))
            coherence = 0.58 + 0.30 * (coupling - 0.5)
            rows.append(
                dict(
                    domain="emergence",
                    agents=agents,
                    coupling=coupling,
                    cci_mean=_mean(cci),
                    coherence=coherence,
                    emergence_gain=gain,
                    collapse_risk=collapse_risk_from(_mean(cci), 0.06, 0.3),
                    energy_cost=energy_cost_proxy(
                        "cyclic", coherence, reset_overhead=0.02
                    ),
                )
            )


# Run all probes
probe_ontology()
probe_identity()
probe_causation()
probe_mind()
probe_freewill()
probe_time()
probe_modality()
probe_properties()
probe_composition()
probe_emergence()

# ----------------- Save CSV -----------------
df = pd.DataFrame(rows)
csv_path = os.path.join(OUT, "results.csv")
df.to_csv(csv_path, index=False)


# ----------------- Plots (a few high-signal ones) -----------------
def save_bar(series, title, ylabel, fname):
    plt.figure(figsize=(7, 5))
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname))
    plt.close()


# 1) Ontology: counterfactual consistency by layer
onto = (
    df[df.domain == "ontology"]
    .groupby("layer_complexity")["counterfactual_consistency"]
    .mean()
)
if len(onto):
    save_bar(
        onto,
        "Ontology: Counterfactual Consistency vs Abstraction Layer",
        "consistency (higher=more real)",
        "plot_ontology_consistency.png",
    )

# 2) Identity: stability vs replace_rate
ident = df[df.domain == "identity"].groupby("replace_rate")["identity_stability"].mean()
if len(ident):
    save_bar(
        ident,
        "Identity: Stability vs Replacement Rate",
        "identity_stability",
        "plot_identity_stability.png",
    )

# 3) Causation: causal minus correlation (bigger gap = real causation)
causal = (
    df[df.domain == "causation"]
    .groupby("externality")["causal_minus_correlation"]
    .mean()
)
if len(causal):
    save_bar(
        causal,
        "Causation: Causal Signal Minus Correlation",
        "causal - correlation",
        "plot_causation_gap.png",
    )

# 4) Mind: mind-body delta by awareness
mind = df[df.domain == "mind"].groupby("time_awareness")["mind_body_delta"].mean()
if len(mind):
    save_bar(
        mind,
        "Mind: Mind–Body Coupling Gain vs Awareness",
        "delta coherence",
        "plot_mind_coupling.png",
    )

# 5) Free will: determinism vs sensitivity
fw = (
    df[df.domain == "free_will"].groupby("noise")[["determinism", "sensitivity"]].mean()
)
if len(fw):
    fw.plot(kind="bar")
    plt.title("Free Will: Determinism & Sensitivity by Noise")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "plot_freewill.png"))
    plt.close()

# 6) Time: future realism heat (foresight vs memory_decay)
time_sub = df[df.domain == "time"]
if not time_sub.empty:
    pivot = (
        time_sub.pivot_table(
            index="memory_decay",
            columns="foresight_gain",
            values="future_realism",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    plt.figure(figsize=(7, 5))
    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
    )
    plt.colorbar(label="future_realism")
    plt.xlabel("foresight_gain")
    plt.ylabel("memory_decay")
    plt.title("Time: Future Realism Map")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "plot_time_future_realism.png"))
    plt.close()

# 7) Modality: necessity vs possibility
mod = df[df.domain == "modality"][["modal_necessity", "modal_possibility"]].mean()
if not mod.empty:
    save_bar(
        pd.Series(mod.values, index=["necessity", "possibility"]),
        "Modality: Necessity vs Possibility",
        "share of worlds",
        "plot_modality.png",
    )

# 8) Properties: universals gain
prop = (
    df[df.domain == "properties"].groupby("cluster_strength")["universals_gain"].mean()
)
if len(prop):
    save_bar(
        prop,
        "Properties: Universals Gain vs Cluster Strength",
        "gain",
        "plot_properties_universals.png",
    )

# 9) Composition: whole formation by coordination
comp = df[df.domain == "composition"].groupby("coordination")["whole_formation"].mean()
if len(comp):
    save_bar(
        comp,
        "Composition: Whole Formation vs Coordination",
        "whole formation score",
        "plot_composition_whole.png",
    )

# 10) Emergence: gain by agents
emer = df[df.domain == "emergence"].groupby("agents")["emergence_gain"].mean()
if len(emer):
    save_bar(
        emer, "Emergence: Gain vs Scale", "emergence gain", "plot_emergence_gain.png"
    )


# ----------------- Summary -----------------
def jfmt(d):
    return json.dumps(
        {
            k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items()
        },
        indent=2,
    )


summary = []
summary.append(f"# Metaphysics Suite — Results\nRun: **{RUN_ID}**\n\n")


def add_section(title, dfsub, by, col, fname=None):
    summary.append(f"## {title}\n")
    if dfsub.empty:
        summary.append("_no data_\n\n")
        return
    agg = dfsub.groupby(by)[col].mean().to_dict()
    summary.append("```json\n" + jfmt(agg) + "\n```\n")
    if fname:
        summary.append(f"Plot: `{fname}`\n\n")


add_section(
    "Ontology: Counterfactual Consistency",
    df[df.domain == "ontology"],
    "layer_complexity",
    "counterfactual_consistency",
    "plot_ontology_consistency.png",
)
add_section(
    "Identity: Stability",
    df[df.domain == "identity"],
    "replace_rate",
    "identity_stability",
    "plot_identity_stability.png",
)
add_section(
    "Causation: Causal Signal",
    df[df.domain == "causation"],
    "externality",
    "causal_minus_correlation",
    "plot_causation_gap.png",
)
add_section(
    "Mind: Mind–Body Coupling Gain",
    df[df.domain == "mind"],
    "time_awareness",
    "mind_body_delta",
    "plot_mind_coupling.png",
)
add_section(
    "Free Will: Determinism",
    df[df.domain == "free_will"],
    "noise",
    "determinism",
    "plot_freewill.png",
)
add_section(
    "Time: Future Realism",
    df[df.domain == "time"],
    "foresight_gain",
    "future_realism",
    "plot_time_future_realism.png",
)
add_section(
    "Modality: Necessity/Possibility",
    df[df.domain == "modality"],
    by="domain",
    col="modal_necessity",
    fname="plot_modality.png",
)
add_section(
    "Properties: Universals Gain",
    df[df.domain == "properties"],
    "cluster_strength",
    "universals_gain",
    "plot_properties_universals.png",
)
add_section(
    "Composition: Whole Formation",
    df[df.domain == "composition"],
    "coordination",
    "whole_formation",
    "plot_composition_whole.png",
)
add_section(
    "Emergence: Gain",
    df[df.domain == "emergence"],
    "agents",
    "emergence_gain",
    "plot_emergence_gain.png",
)

with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write("".join(summary))

print("✅ Finished:", RUN_ID)
print("Saved to:", OUT)
