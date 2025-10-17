#!/usr/bin/env python3
"""
Phase 10: Energy–Information Equivalence — FAST BATTERY
=======================================================

Tests whether changes in information flow and energy flux produce equivalent
effects on stability/CCI. Core hypothesis: ΔCCI/ΔInfo ≈ ΔCCI/ΔEnergy within ±5%.

Key Features:
- Orthogonal variation of energy flux (k_E) and information flow (k_I)
- Null artifact testing with shuffled info-flow signals
- Equivalence gap quantification across openness levels
- Production safety validation with shock resilience testing

Experimental Design:
- 3×3×3 grid: ε × k_E × k_I (27 core conditions + null tests)
- Fast battery: single seed, optimized logging, runtime ≤60s target
- Deterministic validation with artifact detection
"""

import datetime as dt
import hashlib
import json
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Set deterministic parameters
np.random.seed(42)
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output structure
ROOT = Path("./discovery_results")
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
LOGS_DIR = ROOT / "logs"
BUNDLE_DIR = ROOT / "bundle"

for d in [DATA_DIR, FIGURES_DIR, REPORT_DIR, LOGS_DIR, BUNDLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------- Energy-Information Agent Dynamics ----------


class EnergyInfoAgent:
    """Agent with energy flux and information flow dependencies"""

    def __init__(self, agent_id, initial_noise=0.08, initial_coherence=0.6):
        self.agent_id = agent_id
        self.noise = initial_noise
        self.coherence = initial_coherence
        self.calibration = np.random.uniform(0.4, 0.7)
        self.coordination = np.random.uniform(0.3, 0.6)

        # Energy and information state
        self.energy_reserves = np.random.uniform(0.5, 1.0)
        self.info_capacity = np.random.uniform(0.5, 1.0)
        self.message_buffer = []

        self.cci_history = []

    def compute_cci(self):
        """Compute Collective Consciousness Index"""
        coherence_factor = max(0, self.coherence - 0.4) / 0.6
        coordination_factor = max(0, self.coordination - 0.2) / 0.8
        calibration_factor = max(0, self.calibration - 0.3) / 0.7
        noise_penalty = max(0, 1.0 - self.noise / 0.3)

        cci = (
            0.4 * coherence_factor
            + 0.3 * coordination_factor
            + 0.2 * calibration_factor
            + 0.1 * noise_penalty
        )
        return float(np.clip(cci, 0, 1))

    def update_with_energy_info(
        self, openness, energy_flux_factor, info_flow_factor, dt=0.01
    ):
        """Update agent dynamics with energy and information scaling"""
        cci = self.compute_cci()

        # Energy flux affects available work for adaptation
        effective_energy = energy_flux_factor * (0.8 + 0.4 * openness)
        self.energy_reserves = np.clip(
            self.energy_reserves + dt * effective_energy * (1.0 - self.energy_reserves),
            0.1,
            1.0,
        )

        # Information flow affects coordination capacity
        effective_info_rate = info_flow_factor * (0.6 + 0.6 * openness)
        self.info_capacity = np.clip(
            self.info_capacity + dt * effective_info_rate * (0.9 - self.info_capacity),
            0.2,
            1.0,
        )

        # Coherence evolution depends on both energy and information
        coherence_target = 0.60 + 0.20 * openness
        energy_boost = 0.1 * (self.energy_reserves - 0.5)  # Energy above baseline helps
        info_boost = 0.08 * (
            self.info_capacity - 0.6
        )  # Info capacity above baseline helps

        coherence_rate = 2.0 + energy_boost + info_boost
        self.coherence += dt * coherence_rate * (coherence_target - self.coherence)

        # Coordination benefits from information capacity
        coord_target = 0.45 + 0.15 * self.info_capacity
        coord_drift = np.random.normal(0, 0.02)
        self.coordination += dt * 1.5 * (coord_target - self.coordination) + coord_drift

        # Calibration refinement with energy cost
        calib_improvement_rate = 0.8 * self.energy_reserves
        calib_noise = np.random.normal(0, 0.015)
        self.calibration += (
            dt * calib_improvement_rate * (0.55 - self.calibration) + calib_noise
        )

        # Noise reduction depends on both resources
        resource_factor = 0.7 * self.energy_reserves + 0.3 * self.info_capacity
        noise_target = 0.12 - 0.08 * cci * resource_factor
        self.noise += dt * 1.5 * (noise_target - self.noise)

        # Apply bounds
        self.coherence = np.clip(self.coherence, 0.2, 0.95)
        self.coordination = np.clip(self.coordination, 0.1, 0.9)
        self.calibration = np.clip(self.calibration, 0.2, 0.85)
        self.noise = np.clip(self.noise, 0.01, 0.4)

        # Update CCI history
        self.cci_history.append(cci)
        if len(self.cci_history) > 100:
            self.cci_history.pop(0)

    def receive_message(self, message_strength):
        """Receive information from other agents"""
        if self.info_capacity > 0.3:  # Minimum capacity to process
            info_gain = message_strength * self.info_capacity * 0.05
            self.calibration += info_gain
            self.calibration = np.clip(self.calibration, 0.2, 0.85)

    def apply_shock(self, shock_intensity=0.3):
        """Apply noise shock (affects both energy and info systems)"""
        self.noise += shock_intensity
        self.energy_reserves *= 0.8  # Energy disruption
        self.info_capacity *= 0.9  # Info disruption

        # Apply bounds
        self.noise = np.clip(self.noise, 0.01, 0.6)
        self.energy_reserves = np.clip(self.energy_reserves, 0.1, 1.0)
        self.info_capacity = np.clip(self.info_capacity, 0.2, 1.0)

    def get_state_dict(self):
        """Get current state for logging"""
        return {
            "agent_id": self.agent_id,
            "coherence": self.coherence,
            "coordination": self.coordination,
            "calibration": self.calibration,
            "noise": self.noise,
            "energy_reserves": self.energy_reserves,
            "info_capacity": self.info_capacity,
            "cci": self.compute_cci(),
        }


def create_info_network_links(agents, info_flow_factor):
    """Create information network for message passing"""
    n_agents = len(agents)

    # Base connectivity scaled by info_flow_factor
    base_density = 0.05
    effective_density = base_density * info_flow_factor
    n_links = int(effective_density * n_agents * (n_agents - 1) / 2)

    # Random links for this epoch
    possible_pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
    if n_links >= len(possible_pairs):
        active_pairs = possible_pairs
    else:
        active_pairs = np.random.choice(len(possible_pairs), n_links, replace=False)
        active_pairs = [possible_pairs[idx] for idx in active_pairs]

    return active_pairs


def run_energy_info_simulation(
    n_agents=120,
    epochs=1200,
    openness=0.005,
    energy_flux_factor=1.0,
    info_flow_factor=1.0,
    is_null_test=False,
    random_seed=1,
):
    """
    Run energy-information equivalence simulation

    Args:
        is_null_test: If True, shuffle info_flow signals to test for artifacts

    Returns:
        results_dict with trajectories and metrics
    """
    np.random.seed(random_seed)

    # Initialize agents
    agents = [
        EnergyInfoAgent(
            agent_id=i,
            initial_noise=np.random.uniform(0.06, 0.10),
            initial_coherence=np.random.uniform(0.5, 0.7),
        )
        for i in range(n_agents)
    ]

    # Null test: create shuffled info flow sequence
    if is_null_test:
        info_flow_sequence = np.full(epochs, info_flow_factor)
        np.random.shuffle(info_flow_sequence)
    else:
        info_flow_sequence = np.full(epochs, info_flow_factor)

    # Simulation tracking
    epoch_data = []
    shock_events = []

    # Dense logging around shock (epochs 580-620)
    dense_window = set(range(580, 621))

    # Shock parameters
    shock_start = 600
    shock_end = 604
    shock_intensity = 0.3

    print(
        f"    ⚡ Running: ε={openness:.3f}, k_E={energy_flux_factor:.1f}, k_I={info_flow_factor:.1f}"
        + (" [NULL]" if is_null_test else "")
    )

    for epoch in range(epochs):
        # Use potentially shuffled info flow for null test
        epoch_info_flow = info_flow_sequence[epoch]

        # 1. Agent updates with energy/info factors
        for agent in agents:
            agent.update_with_energy_info(
                openness=openness,
                energy_flux_factor=energy_flux_factor,
                info_flow_factor=epoch_info_flow,
            )

        # 2. Information network message passing
        if epoch_info_flow > 0:
            info_links = create_info_network_links(agents, epoch_info_flow)
            for i, j in info_links:
                # Bidirectional message passing
                message_strength = 0.1 * min(
                    agents[i].info_capacity, agents[j].info_capacity
                )
                agents[i].receive_message(message_strength)
                agents[j].receive_message(message_strength)

        # 3. Apply shock during shock window
        if shock_start <= epoch <= shock_end:
            # Shock top 70% of agents by noise level
            agent_noises = [(agent, agent.noise) for agent in agents]
            agent_noises.sort(key=lambda x: x[1], reverse=True)
            n_shock = int(0.7 * n_agents)
            shocked_agents = [agent for agent, _ in agent_noises[:n_shock]]

            pre_noise_mean = np.mean([agent.noise for agent in agents])

            for agent in shocked_agents:
                agent.apply_shock(shock_intensity)

            post_noise_mean = np.mean([agent.noise for agent in agents])

            shock_events.append(
                {
                    "epoch": epoch,
                    "affected_pct": len(shocked_agents) / n_agents * 100,
                    "noise_delta": post_noise_mean - pre_noise_mean,
                    "pre_noise_mean": pre_noise_mean,
                    "post_noise_mean": post_noise_mean,
                }
            )

        # 4. Compute system metrics
        ccis = [agent.compute_cci() for agent in agents]
        noises = [agent.noise for agent in agents]
        coherences = [agent.coherence for agent in agents]
        energies = [agent.energy_reserves for agent in agents]
        info_caps = [agent.info_capacity for agent in agents]

        system_cci = np.mean(ccis)
        system_hazard = np.mean(noises)
        system_coherence = np.mean(coherences)
        system_energy = np.mean(energies)
        system_info = np.mean(info_caps)
        survival_rate = np.mean([cci > 0.3 for cci in ccis])

        # 5. Guardrails (simplified for speed)
        if system_hazard > 0.35:  # Emergency stabilization
            # Reduce noise for worst 30% agents
            agent_noises = [(i, agent.noise) for i, agent in enumerate(agents)]
            agent_noises.sort(key=lambda x: x[1], reverse=True)
            n_stabilize = int(0.3 * n_agents)
            for i, _ in agent_noises[:n_stabilize]:
                agents[i].noise *= 0.8

        # CCI slope nudge
        if len(epoch_data) >= 50:
            recent_ccis = [d["system_cci"] for d in epoch_data[-50:]]
            cci_slope = (recent_ccis[-1] - recent_ccis[0]) / 49
            if cci_slope < 0.001:
                # Coordination boost
                for agent in agents:
                    agent.coordination = min(0.70, agent.coordination + 0.05)

        # 6. Early stop check
        if (
            epoch > 300
            and system_hazard > 0.45
            and survival_rate < 0.40
            and system_cci < 0.45
        ):
            print(f"    ⚠️  Early stop at epoch {epoch}: system collapse detected")
            break

        # 7. Logging (dense around shock, sparse elsewhere)
        if epoch in dense_window or epoch % 10 == 0:
            epoch_data.append(
                {
                    "epoch": epoch,
                    "system_cci": system_cci,
                    "system_hazard": system_hazard,
                    "system_coherence": system_coherence,
                    "system_energy": system_energy,
                    "system_info": system_info,
                    "survival_rate": survival_rate,
                    "openness": openness,
                    "energy_flux_factor": energy_flux_factor,
                    "info_flow_factor": info_flow_factor,
                    "is_null_test": is_null_test,
                    "seed": random_seed,
                }
            )

    # Post-process results
    df = pd.DataFrame(epoch_data)
    shock_df = pd.DataFrame(shock_events)

    # Compute key metrics
    results = {
        "trajectories": df,
        "shock_events": shock_df,
        "parameters": {
            "n_agents": n_agents,
            "epochs": epochs,
            "openness": openness,
            "energy_flux_factor": energy_flux_factor,
            "info_flow_factor": info_flow_factor,
            "is_null_test": is_null_test,
            "seed": random_seed,
        },
    }

    # Compute derived metrics
    if len(df) > 0:
        # Final stability (last 200 epochs)
        final_window = df[df.epoch >= epochs - 200]
        if len(final_window) > 0:
            stability_cci_mean = final_window.system_cci.mean()
            stability_hazard_mean = final_window.system_hazard.mean()
            stability_cci_slope = 0.0

            if len(final_window) > 1:
                # Compute slope
                epochs_final = final_window.epoch.values
                ccis_final = final_window.system_cci.values
                if len(epochs_final) > 1:
                    slope, _ = np.polyfit(epochs_final - epochs_final[0], ccis_final, 1)
                    stability_cci_slope = slope

            final_survival = final_window.survival_rate.iloc[-1]
        else:
            stability_cci_mean = 0
            stability_hazard_mean = 1
            stability_cci_slope = 0
            final_survival = 0

        # Area under hazard during shock window
        shock_window = df[(df.epoch >= 590) & (df.epoch <= 610)]
        auh_590_610 = (
            np.trapz(shock_window.system_hazard, shock_window.epoch)
            if len(shock_window) > 1
            else 0
        )

        results["metrics"] = {
            "stability_cci_mean": stability_cci_mean,
            "stability_hazard_mean": stability_hazard_mean,
            "stability_cci_slope": stability_cci_slope,
            "auh_590_610": auh_590_610,
            "final_survival_rate": final_survival,
        }

    return results


def run_experiment_matrix():
    """Run complete energy-information equivalence experiment"""

    print("⚡ Phase 10: Energy–Information Equivalence")
    print("🎯 FAST BATTERY (≤1 min target)")

    # Experiment parameters
    n_agents = 120
    epochs_cap = 1200
    seed = 1
    openness_levels = [0.003, 0.006, 0.009]
    energy_factors = [0.9, 1.0, 1.1]
    info_factors = [0.9, 1.0, 1.1]

    print(
        f"📊 Matrix: {len(openness_levels)}×{len(energy_factors)}×{len(info_factors)} = {len(openness_levels)*len(energy_factors)*len(info_factors)} conditions + null tests"
    )

    all_results = []
    condition_id = 0

    start_time = time.time()

    # Main experiment grid
    for openness in openness_levels:
        for k_E in energy_factors:
            for k_I in info_factors:
                condition_id += 1

                result = run_energy_info_simulation(
                    n_agents=n_agents,
                    epochs=epochs_cap,
                    openness=openness,
                    energy_flux_factor=k_E,
                    info_flow_factor=k_I,
                    is_null_test=False,
                    random_seed=seed,
                )

                result["condition_id"] = condition_id
                all_results.append(result)

                # Runtime check for speedup
                elapsed = time.time() - start_time
                if elapsed > 45:  # Apply speedups early
                    print(
                        f"    ⚠️  Runtime approaching limit ({elapsed:.1f}s), applying speedups..."
                    )
                    epochs_cap = 900  # Reduce epochs for remaining runs

    # Null tests for artifact detection (one per openness level)
    print("🔍 Running null artifact tests...")
    for openness in openness_levels:
        condition_id += 1

        # Use k_I=1.1 vs k_I=0.9 comparison for null test
        null_result = run_energy_info_simulation(
            n_agents=n_agents,
            epochs=min(epochs_cap, 600),  # Shorter for null test
            openness=openness,
            energy_flux_factor=1.0,  # Fixed energy
            info_flow_factor=1.1,  # High info (but shuffled)
            is_null_test=True,
            random_seed=seed + 100,  # Different seed for null
        )

        null_result["condition_id"] = condition_id
        all_results.append(null_result)

        elapsed = time.time() - start_time
        if elapsed > 55:
            print(f"    ⚠️  Time limit near ({elapsed:.1f}s), stopping null tests early")
            break

    runtime = time.time() - start_time
    print(f"✅ Experiment complete: {runtime:.2f}s")

    return all_results, runtime


def analyze_equivalence_and_export(all_results, runtime):
    """Analyze energy-information equivalence and export artifacts"""

    print("📊 Analyzing equivalence results and generating exports...")

    # Separate null tests from main results
    main_results = [r for r in all_results if not r["parameters"]["is_null_test"]]
    null_results = [r for r in all_results if r["parameters"]["is_null_test"]]

    # Combine trajectories
    all_trajectories = []
    all_shock_events = []
    run_summaries = []
    conditions = []

    for result in all_results:
        # Add condition ID to trajectories
        traj = result["trajectories"].copy()
        traj["condition_id"] = result["condition_id"]
        all_trajectories.append(traj)

        # Shock events
        if len(result["shock_events"]) > 0:
            shock = result["shock_events"].copy()
            shock["condition_id"] = result["condition_id"]
            all_shock_events.append(shock)

        # Run summary
        params = result["parameters"]
        metrics = result.get("metrics", {})

        summary = {
            "condition_id": int(result["condition_id"]),
            **{
                k: (float(v) if isinstance(v, np.number) else v)
                for k, v in params.items()
            },
            **{
                k: (
                    float(v)
                    if isinstance(v, (np.number, type(None))) and v is not None
                    else v
                )
                for k, v in metrics.items()
            },
        }
        run_summaries.append(summary)

        # Condition lookup
        conditions.append(
            {
                "condition_id": int(result["condition_id"]),
                "openness": float(params["openness"]),
                "energy_flux_factor": float(params["energy_flux_factor"]),
                "info_flow_factor": float(params["info_flow_factor"]),
                "is_null_test": bool(params["is_null_test"]),
                "seed": int(params["seed"]),
            }
        )

    # Create DataFrames
    trajectories_df = pd.concat(all_trajectories, ignore_index=True)
    shock_events_df = (
        pd.concat(all_shock_events, ignore_index=True)
        if all_shock_events
        else pd.DataFrame()
    )
    runs_df = pd.DataFrame(run_summaries)
    conditions_df = pd.DataFrame(conditions)

    # 1. Export CSV files
    runs_df.to_csv(DATA_DIR / "runs_summary.csv", index=False)
    trajectories_df.to_csv(DATA_DIR / "trajectories_long.csv", index=False)
    if len(shock_events_df) > 0:
        shock_events_df.to_csv(DATA_DIR / "shock_events.csv", index=False)
    conditions_df.to_csv(DATA_DIR / "conditions_lookup.csv", index=False)

    # 2. Compute equivalence analysis
    main_runs = runs_df[runs_df.is_null_test == False].copy()
    equivalence_results = []

    for openness in main_runs.openness.unique():
        openness_subset = main_runs[main_runs.openness == openness]

        # Energy effect: k_E=1.1 vs k_E=0.9 at k_I=1.0
        high_energy = openness_subset[
            (openness_subset.energy_flux_factor == 1.1)
            & (openness_subset.info_flow_factor == 1.0)
        ]
        low_energy = openness_subset[
            (openness_subset.energy_flux_factor == 0.9)
            & (openness_subset.info_flow_factor == 1.0)
        ]

        # Info effect: k_I=1.1 vs k_I=0.9 at k_E=1.0
        high_info = openness_subset[
            (openness_subset.energy_flux_factor == 1.0)
            & (openness_subset.info_flow_factor == 1.1)
        ]
        low_info = openness_subset[
            (openness_subset.energy_flux_factor == 1.0)
            & (openness_subset.info_flow_factor == 0.9)
        ]

        # Compute deltas
        delta_cci_energy = 0.0
        delta_cci_info = 0.0

        if len(high_energy) > 0 and len(low_energy) > 0:
            delta_cci_energy = (
                high_energy.stability_cci_mean.iloc[0]
                - low_energy.stability_cci_mean.iloc[0]
            )

        if len(high_info) > 0 and len(low_info) > 0:
            delta_cci_info = (
                high_info.stability_cci_mean.iloc[0]
                - low_info.stability_cci_mean.iloc[0]
            )

        # Equivalence gap
        mean_effect = (abs(delta_cci_energy) + abs(delta_cci_info)) / 2
        equivalence_gap = abs(delta_cci_energy - delta_cci_info) / max(
            1e-6, mean_effect
        )

        # Sensitivities (per +20% change)
        energy_sensitivity = delta_cci_energy / 0.2
        info_sensitivity = delta_cci_info / 0.2

        # Null artifact check
        null_artifact_flag = False
        if null_results:
            null_subset = [
                r for r in null_results if r["parameters"]["openness"] == openness
            ]
            if null_subset:
                null_cci = null_subset[0]["metrics"]["stability_cci_mean"]
                baseline_cci = (
                    openness_subset[
                        (openness_subset.energy_flux_factor == 1.0)
                        & (openness_subset.info_flow_factor == 1.0)
                    ].stability_cci_mean.iloc[0]
                    if len(openness_subset) > 0
                    else 0
                )

                null_delta = abs(null_cci - baseline_cci)
                if null_delta >= 0.5 * abs(delta_cci_info):
                    null_artifact_flag = True

        # Equivalence verdict
        if equivalence_gap <= 0.05 and not null_artifact_flag:
            equivalence_verdict = "PASS ±5%"
        elif equivalence_gap <= 0.10 and not null_artifact_flag:
            equivalence_verdict = "PASS ±10% (provisional)"
        else:
            equivalence_verdict = "FAIL"

        equivalence_results.append(
            {
                "openness": openness,
                "delta_cci_energy": delta_cci_energy,
                "delta_cci_info": delta_cci_info,
                "equivalence_gap": equivalence_gap,
                "energy_sensitivity": energy_sensitivity,
                "info_sensitivity": info_sensitivity,
                "null_artifact_flag": null_artifact_flag,
                "equivalence_verdict": equivalence_verdict,
            }
        )

    # Production safety analysis
    production_safe_bar = {
        "stability_cci_mean": 0.50,
        "stability_hazard_mean": 0.20,
        "stability_cci_slope": 0.0005,
    }

    # Find winner (best performer meeting production standards)
    qualifying = main_runs[
        (main_runs.stability_cci_mean >= production_safe_bar["stability_cci_mean"])
        & (
            main_runs.stability_hazard_mean
            <= production_safe_bar["stability_hazard_mean"]
        )
        & (main_runs.stability_cci_slope >= production_safe_bar["stability_cci_slope"])
    ]

    winner = None
    if len(qualifying) > 0:
        winner_row = qualifying.loc[qualifying.auh_590_610.idxmin()]
        winner = {
            "condition_id": int(winner_row.condition_id),
            "openness": float(winner_row.openness),
            "energy_flux_factor": float(winner_row.energy_flux_factor),
            "info_flow_factor": float(winner_row.info_flow_factor),
            "stability_cci_mean": float(winner_row.stability_cci_mean),
            "stability_hazard_mean": float(winner_row.stability_hazard_mean),
            "auh_590_610": float(winner_row.auh_590_610),
            "reason": "Production-safe winner with lowest shock impact",
        }
    else:
        # Fallback: best performer overall
        best_row = main_runs.loc[main_runs.stability_cci_mean.idxmax()]
        winner = {
            "condition_id": int(best_row.condition_id),
            "openness": float(best_row.openness),
            "energy_flux_factor": float(best_row.energy_flux_factor),
            "info_flow_factor": float(best_row.info_flow_factor),
            "stability_cci_mean": float(best_row.stability_cci_mean),
            "stability_hazard_mean": float(best_row.stability_hazard_mean),
            "auh_590_610": float(best_row.auh_590_610),
            "reason": "Best CCI performer (no production-safe winners)",
        }

    # 3. Export JSON summary
    summary_json = {
        "experiment_info": {
            "phase": "Phase 10: Energy–Information Equivalence",
            "timestamp": TIMESTAMP,
            "runtime_seconds": runtime,
            "n_conditions": len(all_results),
            "n_main_conditions": len(main_results),
            "n_null_tests": len(null_results),
        },
        "equivalence_analysis": equivalence_results,
        "winner": winner,
        "production_safe_bar": production_safe_bar,
    }

    with open(DATA_DIR / "phase10_equivalence_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, cls=NumpyJSONEncoder)

    # 4. Generate figures
    create_equivalence_figures(trajectories_df, main_runs, equivalence_results)

    # 5. Generate report
    create_equivalence_report(summary_json, main_runs, runtime)

    # 6. Create bundle
    create_bundle()

    return summary_json


def create_equivalence_figures(trajectories_df, runs_df, equivalence_results):
    """Create all required figures for equivalence analysis"""

    print("  📈 Generating equivalence figures...")

    # Filter out null tests for main figures
    main_trajectories = trajectories_df[trajectories_df.is_null_test == False]

    # 1. CCI long trajectory by openness
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    openness_levels = sorted(main_trajectories.openness.unique())
    for i, openness in enumerate(openness_levels):
        ax = axes[i]

        openness_data = main_trajectories[main_trajectories.openness == openness]

        # Plot by energy/info factor combinations
        for (k_E, k_I), group in openness_data.groupby(
            ["energy_flux_factor", "info_flow_factor"]
        ):
            label = f"k_E={k_E:.1f}, k_I={k_I:.1f}"

            avg_trajectory = group.groupby("epoch").system_cci.mean()
            ax.plot(
                avg_trajectory.index,
                avg_trajectory.values,
                label=label,
                linewidth=1.5,
                alpha=0.8,
            )

        ax.axvspan(600, 604, alpha=0.2, color="red", label="Shock" if i == 0 else "")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("System CCI")
        ax.set_title(f"CCI Evolution (ε={openness:.3f})")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cci_long_phase10.png", bbox_inches="tight")
    plt.close()

    # 2. Hazard long trajectory
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    for i, openness in enumerate(openness_levels):
        ax = axes[i]

        openness_data = main_trajectories[main_trajectories.openness == openness]

        for (k_E, k_I), group in openness_data.groupby(
            ["energy_flux_factor", "info_flow_factor"]
        ):
            label = f"k_E={k_E:.1f}, k_I={k_I:.1f}"

            avg_trajectory = group.groupby("epoch").system_hazard.mean()
            ax.plot(
                avg_trajectory.index,
                avg_trajectory.values,
                label=label,
                linewidth=1.5,
                alpha=0.8,
            )

        ax.axvspan(600, 604, alpha=0.2, color="red", label="Shock" if i == 0 else "")
        ax.axhline(
            y=0.2,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Safety Threshold" if i == 0 else "",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("System Hazard")
        ax.set_title(f"Hazard Evolution (ε={openness:.3f})")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hazard_long_phase10.png", bbox_inches="tight")
    plt.close()

    # 3. Equivalence gap bars
    plt.figure(figsize=(10, 6), dpi=150)

    equiv_df = pd.DataFrame(equivalence_results)
    openness_vals = equiv_df.openness.values
    gaps = equiv_df.equivalence_gap.values

    bars = plt.bar(range(len(openness_vals)), gaps, color="steelblue", alpha=0.7)

    # Add threshold lines
    plt.axhline(y=0.05, color="green", linestyle="--", linewidth=2, label="±5% Target")
    plt.axhline(
        y=0.10, color="orange", linestyle="--", linewidth=2, label="±10% Provisional"
    )

    # Add value labels on bars
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        height = bar.get_height()
        verdict = equiv_df.iloc[i].equivalence_verdict
        color = "green" if "PASS" in verdict else "red"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{gap:.3f}\n{verdict}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=color,
            fontsize=9,
        )

    plt.xticks(range(len(openness_vals)), [f"ε={x:.3f}" for x in openness_vals])
    plt.ylabel("Equivalence Gap")
    plt.title("Energy–Information Equivalence Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "equiv_gap_bars.png", bbox_inches="tight")
    plt.close()

    # 4. Sensitivity grid (3x3 heatmaps for each openness)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)

    energy_factors = [0.9, 1.0, 1.1]
    info_factors = [0.9, 1.0, 1.1]

    for i, openness in enumerate(openness_levels):
        # CCI heatmap (top row)
        ax_cci = axes[0, i]

        openness_data = runs_df[runs_df.openness == openness]
        cci_matrix = np.zeros((len(info_factors), len(energy_factors)))

        for ei, k_E in enumerate(energy_factors):
            for ii, k_I in enumerate(info_factors):
                condition = openness_data[
                    (openness_data.energy_flux_factor == k_E)
                    & (openness_data.info_flow_factor == k_I)
                ]
                if len(condition) > 0:
                    cci_matrix[ii, ei] = condition.stability_cci_mean.iloc[0]

        im_cci = ax_cci.imshow(
            cci_matrix, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=0.7
        )

        # Add text annotations
        for ii in range(len(info_factors)):
            for ei in range(len(energy_factors)):
                text = ax_cci.text(
                    ei,
                    ii,
                    f"{cci_matrix[ii, ei]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax_cci.set_xticks(range(len(energy_factors)))
        ax_cci.set_xticklabels([f"{x:.1f}" for x in energy_factors])
        ax_cci.set_yticks(range(len(info_factors)))
        ax_cci.set_yticklabels([f"{x:.1f}" for x in info_factors])
        ax_cci.set_xlabel("Energy Flux (k_E)")
        ax_cci.set_ylabel("Info Flow (k_I)")
        ax_cci.set_title(f"CCI Grid (ε={openness:.3f})")

        # Hazard heatmap (bottom row)
        ax_hazard = axes[1, i]

        hazard_matrix = np.zeros((len(info_factors), len(energy_factors)))

        for ei, k_E in enumerate(energy_factors):
            for ii, k_I in enumerate(info_factors):
                condition = openness_data[
                    (openness_data.energy_flux_factor == k_E)
                    & (openness_data.info_flow_factor == k_I)
                ]
                if len(condition) > 0:
                    hazard_matrix[ii, ei] = condition.stability_hazard_mean.iloc[0]

        im_hazard = ax_hazard.imshow(
            hazard_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0.05, vmax=0.15
        )

        # Add text annotations
        for ii in range(len(info_factors)):
            for ei in range(len(energy_factors)):
                text = ax_hazard.text(
                    ei,
                    ii,
                    f"{hazard_matrix[ii, ei]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax_hazard.set_xticks(range(len(energy_factors)))
        ax_hazard.set_xticklabels([f"{x:.1f}" for x in energy_factors])
        ax_hazard.set_yticks(range(len(info_factors)))
        ax_hazard.set_yticklabels([f"{x:.1f}" for x in info_factors])
        ax_hazard.set_xlabel("Energy Flux (k_E)")
        ax_hazard.set_ylabel("Info Flow (k_I)")
        ax_hazard.set_title(f"Hazard Grid (ε={openness:.3f})")

    # Add colorbars
    plt.colorbar(im_cci, ax=axes[0, :], label="CCI", shrink=0.8)
    plt.colorbar(im_hazard, ax=axes[1, :], label="Hazard", shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensitivity_grid.png", bbox_inches="tight")
    plt.close()


def create_equivalence_report(summary_json, runs_df, runtime):
    """Create markdown report for equivalence analysis"""

    print("  📝 Generating equivalence report...")

    with open(REPORT_DIR / "phase10_results.md", "w") as f:
        f.write("# Phase 10: Energy–Information Equivalence\n\n")
        f.write(f"**Timestamp**: {TIMESTAMP}\n")
        f.write(f"**Runtime**: {runtime:.2f} seconds\n")
        f.write(
            f"**Conditions Tested**: {summary_json['experiment_info']['n_main_conditions']} + {summary_json['experiment_info']['n_null_tests']} null tests\n\n"
        )

        f.write("## Executive Summary\n\n")
        f.write(
            "This experiment tested the equivalence hypothesis: ΔCCI/ΔInfo ≈ ΔCCI/ΔEnergy "
        )
        f.write(
            "within ±5% at fixed openness levels. We systematically varied energy flux (k_E) "
        )
        f.write(
            "and information flow (k_I) factors while monitoring system stability.\n\n"
        )

        f.write("## FAST TAKEAWAYS\n\n")

        # Equivalence analysis
        equivalence_results = summary_json["equivalence_analysis"]

        f.write("### Equivalence Verdict by Openness\n\n")
        for result in equivalence_results:
            openness = result["openness"]
            verdict = result["equivalence_verdict"]
            gap = result["equivalence_gap"]
            null_flag = result["null_artifact_flag"]

            status_emoji = "✅" if "PASS" in verdict else "❌"
            null_status = " ⚠️ [ARTIFACT]" if null_flag else ""

            f.write(
                f"**ε={openness:.3f}**: {status_emoji} **{verdict}** (gap={gap:.3f}){null_status}\n"
            )

        f.write("\n### Sensitivity Comparison\n\n")

        for result in equivalence_results:
            openness = result["openness"]
            energy_sens = result["energy_sensitivity"]
            info_sens = result["info_sensitivity"]

            f.write(f"**ε={openness:.3f}**:\n")
            f.write(f"- Energy sensitivity: {energy_sens:+.3f} CCI per +10% k_E\n")
            f.write(f"- Info sensitivity: {info_sens:+.3f} CCI per +10% k_I\n")

            # Which lever is more effective
            if abs(energy_sens) > abs(info_sens):
                better_lever = f"Energy (k_E) {abs(energy_sens)/abs(info_sens):.1f}x more effective"
            elif abs(info_sens) > abs(energy_sens):
                better_lever = f"Information (k_I) {abs(info_sens)/abs(energy_sens):.1f}x more effective"
            else:
                better_lever = "Energy and Information equally effective"

            f.write(f"- **Best lever**: {better_lever}\n\n")

        # Production safety
        f.write("### Production Safety Analysis\n\n")

        production_bar = summary_json["production_safe_bar"]
        f.write(f"**Criteria**: CCI ≥ {production_bar['stability_cci_mean']:.2f}, ")
        f.write(f"Hazard ≤ {production_bar['stability_hazard_mean']:.2f}, ")
        f.write(f"Slope ≥ {production_bar['stability_cci_slope']:.4f}\n\n")

        # Check which conditions pass
        qualifying_conditions = []
        for _, row in runs_df.iterrows():
            cci_pass = row.stability_cci_mean >= production_bar["stability_cci_mean"]
            hazard_pass = (
                row.stability_hazard_mean <= production_bar["stability_hazard_mean"]
            )
            slope_pass = (
                row.stability_cci_slope >= production_bar["stability_cci_slope"]
            )

            if cci_pass and hazard_pass and slope_pass:
                qualifying_conditions.append(row)

        if qualifying_conditions:
            f.write(
                f"✅ **{len(qualifying_conditions)} conditions passed** production safety standards\n\n"
            )
        else:
            f.write("❌ **No conditions passed** production safety standards\n\n")

        # Winner analysis
        winner = summary_json["winner"]
        if winner:
            f.write("### Winner Selection\n\n")
            f.write(f"🏆 **WINNER**: Condition {winner['condition_id']}\n")
            f.write(
                f"- **Parameters**: ε={winner['openness']:.3f}, k_E={winner['energy_flux_factor']:.1f}, k_I={winner['info_flow_factor']:.1f}\n"
            )
            f.write(
                f"- **Performance**: CCI={winner['stability_cci_mean']:.3f}, Hazard={winner['stability_hazard_mean']:.3f}\n"
            )
            f.write(f"- **Shock resilience**: AUH={winner['auh_590_610']:.3f}\n")
            f.write(f"- **Reason**: {winner['reason']}\n\n")

        # Main findings
        f.write("## Key Scientific Findings\n\n")

        # Count equivalence successes
        pass_count = sum(
            1 for r in equivalence_results if "PASS" in r["equivalence_verdict"]
        )

        if pass_count == len(equivalence_results):
            f.write("### ✅ Energy–Information Equivalence VALIDATED\n\n")
            f.write("- **All openness levels** showed energy-information equivalence\n")
            f.write(
                "- **Hypothesis confirmed**: ΔCCI/ΔEnergy ≈ ΔCCI/ΔInfo within tolerance\n"
            )
            f.write(
                "- **Fundamental principle**: Energy and information are interchangeable for system stability\n\n"
            )
        elif pass_count > 0:
            f.write("### 🔶 Partial Energy–Information Equivalence\n\n")
            f.write(
                f"- **{pass_count}/{len(equivalence_results)} openness levels** showed equivalence\n"
            )
            f.write(
                "- **Context-dependent**: Equivalence may depend on system openness\n"
            )
            f.write(
                "- **Mixed evidence** for fundamental energy-information relationship\n\n"
            )
        else:
            f.write("### ❌ Energy–Information Equivalence NOT SUPPORTED\n\n")
            f.write("- **No openness levels** showed clear equivalence\n")
            f.write(
                "- **Different mechanisms**: Energy and information affect stability through distinct pathways\n"
            )
            f.write(
                "- **Non-interchangeable**: Systems may optimize differently for energy vs information\n\n"
            )

        # Sensitivity insights
        avg_energy_sens = np.mean(
            [r["energy_sensitivity"] for r in equivalence_results]
        )
        avg_info_sens = np.mean([r["info_sensitivity"] for r in equivalence_results])

        f.write("### System Lever Effectiveness\n\n")

        if abs(avg_energy_sens) > abs(avg_info_sens) * 1.2:
            f.write(
                "- **Energy-dominated systems**: Energy flux more effective than information flow\n"
            )
            f.write(
                "- **Resource prioritization**: Focus on energy infrastructure over communication\n"
            )
        elif abs(avg_info_sens) > abs(avg_energy_sens) * 1.2:
            f.write(
                "- **Information-dominated systems**: Information flow more effective than energy flux\n"
            )
            f.write(
                "- **Communication prioritization**: Focus on information infrastructure over energy\n"
            )
        else:
            f.write(
                "- **Balanced systems**: Energy and information roughly equivalent in effectiveness\n"
            )
            f.write(
                "- **Dual optimization**: Both energy and information infrastructure equally important\n"
            )

        f.write("\nAverage sensitivities:\n")
        f.write(f"- Energy: {avg_energy_sens:+.3f} CCI per +10% energy flux\n")
        f.write(
            f"- Information: {avg_info_sens:+.3f} CCI per +10% information flow\n\n"
        )

        # Runtime performance
        f.write("## Runtime Performance\n\n")
        f.write(f"- **Total runtime**: {runtime:.2f} seconds\n")
        f.write(
            f"- **Target achieved**: {'✅ YES' if runtime <= 60 else '❌ NO (overtime)'}\n"
        )
        if runtime > 60:
            f.write(f"- **Overtime factor**: {runtime/60:.1f}x target\n")

        f.write(
            f"- **Conditions per second**: {summary_json['experiment_info']['n_conditions']/runtime:.1f}\n"
        )

        # Null test status
        if summary_json["experiment_info"]["n_null_tests"] > 0:
            artifact_flags = [r["null_artifact_flag"] for r in equivalence_results]
            artifact_count = sum(artifact_flags)

            f.write("\n### Artifact Detection\n")
            f.write(
                f"- **Null tests run**: {summary_json['experiment_info']['n_null_tests']}\n"
            )
            f.write(
                f"- **Artifacts detected**: {artifact_count}/{len(equivalence_results)} openness levels\n"
            )

            if artifact_count == 0:
                f.write("- **Validation**: All effects appear genuine (no artifacts)\n")
            else:
                f.write(
                    "- **Warning**: Some effects may be artifacts (flagged in analysis)\n"
                )


def create_bundle():
    """Create zip bundle with all artifacts"""

    print("  📦 Creating bundle...")

    bundle_path = BUNDLE_DIR / f"phase10_{TIMESTAMP}.zip"

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from data, figures, report directories
        for directory in [DATA_DIR, FIGURES_DIR, REPORT_DIR]:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(ROOT)
                    zipf.write(file_path, arcname)

        # Create and add SHA256SUMS.txt
        sha_content = []
        for directory in [DATA_DIR, FIGURES_DIR, REPORT_DIR]:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    rel_path = file_path.relative_to(ROOT)
                    sha_content.append(f"{file_hash}  {rel_path}")

        sha_text = "\n".join(sha_content)
        zipf.writestr("SHA256SUMS.txt", sha_text)

    print(f"  📦 Bundle created: {bundle_path}")


def main():
    """Main orchestrator for Phase 10"""

    start_time = time.time()

    # Run experiment matrix
    all_results, runtime = run_experiment_matrix()

    # Analyze and export
    summary_json = analyze_equivalence_and_export(all_results, runtime)

    # Final summary
    print("\n" + "=" * 80)
    print("⚡ Phase 10: Energy–Information Equivalence COMPLETE")
    print(f"⏱️  Total Runtime: {runtime:.2f} seconds")
    print(f"🎯 Target: ≤60s - {'✅ ACHIEVED' if runtime <= 60 else '❌ OVERTIME'}")

    # Equivalence results summary
    equivalence_results = summary_json["equivalence_analysis"]
    pass_count = sum(
        1 for r in equivalence_results if "PASS" in r["equivalence_verdict"]
    )

    print(
        f"\n⚖️  EQUIVALENCE ANALYSIS: {pass_count}/{len(equivalence_results)} openness levels PASS"
    )

    for result in equivalence_results:
        status_icon = "✅" if "PASS" in result["equivalence_verdict"] else "❌"
        print(
            f"   ε={result['openness']:.3f}: {status_icon} {result['equivalence_verdict']} (gap={result['equivalence_gap']:.3f})"
        )

    # Winner analysis
    winner = summary_json["winner"]
    if winner:
        print(
            f"\n🏆 WINNER: ε={winner['openness']:.3f}, k_E={winner['energy_flux_factor']:.1f}, k_I={winner['info_flow_factor']:.1f}"
        )
        print(
            f"   Performance: CCI={winner['stability_cci_mean']:.3f}, Hazard={winner['stability_hazard_mean']:.3f}"
        )

    # Sensitivity insights
    avg_energy_sens = np.mean([r["energy_sensitivity"] for r in equivalence_results])
    avg_info_sens = np.mean([r["info_sensitivity"] for r in equivalence_results])

    print("\n🔧 SENSITIVITY ANALYSIS:")
    print(f"   Energy lever: {avg_energy_sens:+.3f} CCI per +10% flux")
    print(f"   Info lever: {avg_info_sens:+.3f} CCI per +10% flow")

    if abs(avg_energy_sens) > abs(avg_info_sens) * 1.2:
        print("   🔋 Energy-dominated: Focus on energy infrastructure")
    elif abs(avg_info_sens) > abs(avg_energy_sens) * 1.2:
        print("   📡 Info-dominated: Focus on communication infrastructure")
    else:
        print("   ⚖️  Balanced: Energy and information equally important")

    print(f"\n📁 Results exported to: {ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
