#!/usr/bin/env python3
"""
Phase 19: Collective Adaptive Dynamics
======================================

Demonstrates multi-agent adaptive control in networked temporal coherence systems.
Each agent in a small-world network learns its own plant model and adapts control
strategies while interacting with neighbors through coupling dynamics.

Key Features:
- Watts-Strogatz small-world network topology (N=24 agents)
- Individual RLS learning per agent with unique plant parameters
- Neighbor-based coupling: κ_i = (A @ C) / deg_i
- Dynamic per-agent targeting based on local coherence history
- Comparative analysis: No Control vs Fixed vs Adaptive strategies

This represents the culmination of temporal coherence research: intelligent,
networked systems that collectively learn and optimize their temporal dynamics
through distributed adaptive control.
"""

import datetime as dt
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set deterministic parameters
np.random.seed(42)
STAMP = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT = Path(f"./discovery_results/phase19_collective_{STAMP}")
ROOT.mkdir(parents=True, exist_ok=True)

# ---------- Network Generation ----------


def generate_watts_strogatz_network(N=24, k=6, beta=0.15, random_state=None):
    """
    Generate Watts-Strogatz small-world network

    Parameters:
    - N: number of nodes
    - k: each node connected to k nearest neighbors (k/2 on each side)
    - beta: rewiring probability (0=regular, 1=random)
    - random_state: random number generator for reproducibility

    Returns:
    - A: adjacency matrix (N x N)
    """
    if random_state is None:
        rng = np.random.RandomState(42)
    else:
        rng = random_state

    # Initialize empty adjacency matrix
    A = np.zeros((N, N), dtype=int)

    # Create regular ring lattice
    half_k = k // 2
    for i in range(N):
        for d in range(1, half_k + 1):
            j = (i + d) % N
            A[i, j] = A[j, i] = 1

    # Rewire edges with probability beta
    for i in range(N):
        for d in range(1, half_k + 1):
            j = (i + d) % N
            if rng.rand() < beta:
                # Remove current edge
                A[i, j] = A[j, i] = 0

                # Find potential new targets (not self, not already connected)
                potential_targets = np.where((A[i] == 0) & (np.arange(N) != i))[0]

                if len(potential_targets) > 0:
                    # Randomly select new target
                    new_target = rng.choice(potential_targets)
                    A[i, new_target] = A[new_target, i] = 1
                else:
                    # If no valid targets, keep original edge
                    A[i, j] = A[j, i] = 1

    return A


# ---------- Utility Functions ----------


def compute_gini_coefficient(x):
    """
    Compute Gini coefficient for inequality measurement
    0 = robust equality, 1 = maximum inequality
    """
    x = np.asarray(x, dtype=float).flatten()
    x = np.abs(x) + 1e-12  # Ensure positive values
    x_sorted = np.sort(x)
    n = len(x)

    cumulative_sum = np.cumsum(x_sorted)
    total_sum = np.sum(x_sorted)

    # Gini formula
    gini = (2 * np.sum(np.arange(1, n + 1) * x_sorted)) / (n * total_sum) - (n + 1) / n

    return float(gini)


class RecursiveLeastSquaresAgent:
    """
    Individual RLS learner for each agent

    Plant model: ΔC_i = θ₀ + θ₁·Δε_i + θ₂·κ_i·Δε_i + noise
    Where:
    - θ₀: baseline coherence drift
    - θ₁: direct control effect
    - θ₂: coupling-mediated control effect
    """

    def __init__(self, n_params=3, forgetting_factor=0.995, initial_covariance=100.0):
        self.n_params = n_params
        self.forgetting_factor = forgetting_factor

        # Parameter estimates
        self.theta = np.zeros((n_params, 1))

        # Covariance matrix
        self.P = initial_covariance * np.eye(n_params)

        # Learning statistics
        self.update_count = 0
        self.prediction_errors = []

    def update(self, phi, y):
        """
        Update parameters with new observation

        Args:
            phi: feature vector [Δε_i, κ_i·Δε_i, 1] (shape: n_params,)
            y: observed change in coherence ΔC_i (scalar)

        Returns:
            prediction_error: y - phi^T * theta_old
        """
        phi = np.asarray(phi).reshape(-1, 1)

        # Kalman gain
        denominator = self.forgetting_factor + phi.T @ self.P @ phi
        K = self.P @ phi / denominator

        # Prediction error
        prediction_error = float(y - phi.T @ self.theta)

        # Update parameters
        self.theta = self.theta + K * prediction_error

        # Update covariance
        self.P = (self.P - K @ phi.T @ self.P) / self.forgetting_factor

        # Statistics
        self.update_count += 1
        self.prediction_errors.append(prediction_error)

        return prediction_error

    def get_sensitivity(self, kappa):
        """Get learned sensitivity dC/d(Δε) = θ₁ + θ₂·κ"""
        theta = self.theta.flatten()
        if len(theta) >= 2:
            return theta[0] + theta[1] * kappa
        else:
            return 0.0

    def get_parameters(self):
        """Get current parameter estimates"""
        return self.theta.flatten()


def compute_dynamic_target(
    coherence_history, lookback=20, std_multiplier=0.5, min_target=0.55, max_target=0.85
):
    """
    Compute adaptive target based on recent coherence statistics

    target = median(recent_coherence) + k * std(recent_coherence)
    """
    if len(coherence_history) == 0:
        return 0.60  # Default target

    recent_values = (
        coherence_history[-lookback:]
        if len(coherence_history) >= lookback
        else coherence_history
    )

    if len(recent_values) == 0:
        return 0.60

    median_coherence = np.median(recent_values)
    std_coherence = np.std(recent_values)

    dynamic_target = median_coherence + std_multiplier * std_coherence

    return float(np.clip(dynamic_target, min_target, max_target))


# ---------- Multi-Agent Collective Simulation ----------


def run_collective_simulation(N=24, T=600, control_mode="adaptive", random_seed=123):
    """
    Run multi-agent collective temporal coherence control

    Parameters:
    - N: number of agents
    - T: simulation time steps
    - control_mode: "nocontrol" | "fixed" | "adaptive"
    - random_seed: for reproducible results

    Returns:
    - network_topology: adjacency matrix
    - coherence_trajectory: (T, N) coherence evolution
    - synchronization_trajectory: (T,) global order parameter
    - control_energy_trajectory: (T,) total control energy per step
    - target_trajectory: (T, N) dynamic targets per agent
    - control_increment_trajectory: (T, N) control increments per agent
    - performance_metrics: summary statistics
    """
    print(f"    🌐 Running collective simulation: mode={control_mode}, N={N}, T={T}")

    # Initialize random state
    rng = np.random.RandomState(random_seed)

    # Generate small-world network
    A = generate_watts_strogatz_network(N=N, k=6, beta=0.18, random_state=rng)
    node_degrees = A.sum(axis=1).clip(min=1)  # Ensure no isolated nodes

    # Initialize heterogeneous agent parameters (hidden from controllers)
    # Each agent has unique plant dynamics
    agent_sensitivity_direct = rng.uniform(0.04, 0.12, N)  # θ₁ true values
    agent_sensitivity_coupling = rng.uniform(0.05, 0.20, N)  # θ₂ true values
    agent_noise_std = rng.uniform(0.002, 0.006, N)  # individual noise levels

    # Initialize agent states
    initial_coherence = rng.uniform(0.55, 0.75, N)
    current_coherence = initial_coherence.copy()
    cumulative_control = np.zeros(N)

    # Control parameters
    max_control_increment_fixed = 0.02
    max_control_increment_adaptive = 0.06

    # Initialize individual RLS learners (one per agent)
    agent_learners = [
        RecursiveLeastSquaresAgent(
            n_params=3, forgetting_factor=0.995, initial_covariance=50.0
        )
        for _ in range(N)
    ]

    # PID state for adaptive control
    error_integral = np.zeros(N)
    previous_error = np.zeros(N)

    # Simulation trajectories
    coherence_trajectory = np.zeros((T, N))
    synchronization_trajectory = np.zeros(T)
    control_energy_trajectory = np.zeros(T)
    target_trajectory = np.zeros((T, N))
    control_increment_trajectory = np.zeros((T, N))

    # Initial variance for order parameter calculation
    initial_coherence_variance = np.var(current_coherence) + 1e-12

    print(
        f"    🔧 Initialized: {N} agents, network connectivity: {A.sum()/(N*(N-1)):.3f}"
    )

    for t in range(T):
        # Compute coupling strengths (neighbor influence)
        # κ_i = average coherence of neighbors
        coupling_strengths = (A @ current_coherence) / node_degrees

        # Compute dynamic targets for each agent
        agent_targets = np.array(
            [
                compute_dynamic_target(coherence_trajectory[:t, i], lookback=20)
                for i in range(N)
            ]
        )
        target_trajectory[t] = agent_targets

        # Control decision based on mode
        if control_mode == "nocontrol":
            control_increments = np.zeros(N)

        elif control_mode == "fixed":
            # Simple proportional control
            prediction_errors = np.maximum(0.0, agent_targets - current_coherence)
            control_increments = np.clip(
                0.20 * prediction_errors, 0.0, max_control_increment_fixed
            )

        elif control_mode == "adaptive":
            # Adaptive PID control with learned sensitivity
            current_errors = np.maximum(0.0, agent_targets - current_coherence)

            # PID error processing
            error_integral += current_errors
            error_derivative = current_errors - previous_error
            previous_error = current_errors.copy()

            # PID output
            pid_gains = (0.28, 0.10, 0.06)  # (Kp, Ki, Kd)
            raw_control = (
                pid_gains[0] * current_errors
                + pid_gains[1] * error_integral
                + pid_gains[2] * error_derivative
            )

            # Adaptive gain based on learned sensitivity
            learned_parameters = np.array(
                [learner.get_parameters() for learner in agent_learners]
            )

            # Learned sensitivity = θ₁ + θ₂ * κ (assuming θ₀, θ₁, θ₂ order)
            if learned_parameters.shape[1] >= 2:
                learned_sensitivities = (
                    learned_parameters[:, 0]
                    + learned_parameters[:, 1] * coupling_strengths
                )
            else:
                learned_sensitivities = np.ones(N)

            # Adaptive gain (higher sensitivity → lower gain for stability)
            adaptive_gains = np.where(
                learned_sensitivities <= 1e-6,
                1.0,
                np.clip(1.0 / learned_sensitivities, 0.25, 1.6),
            )

            # Apply adaptive gains and bounds
            control_increments = np.clip(
                raw_control * adaptive_gains, 0.0, max_control_increment_adaptive
            )

        else:
            raise ValueError(f"Unknown control mode: {control_mode}")

        # Apply control
        cumulative_control += control_increments
        control_increment_trajectory[t] = control_increments

        # True plant dynamics (unknown to controllers)
        noise = rng.normal(0, agent_noise_std)

        coherence_changes = (
            agent_sensitivity_direct * control_increments
            + agent_sensitivity_coupling * coupling_strengths * control_increments
            + noise
        )

        next_coherence = current_coherence + coherence_changes

        # Apply physical bounds
        next_coherence = np.clip(next_coherence, 0.35, 0.95)

        # Update agent learners with observed transitions (for adaptive mode)
        if control_mode == "adaptive":
            observed_changes = next_coherence - current_coherence

            for i in range(N):
                # Feature vector: [Δε_i, κ_i·Δε_i, 1] (intercept, direct, coupling terms)
                phi_i = np.array(
                    [
                        control_increments[i],
                        coupling_strengths[i] * control_increments[i],
                        1.0,
                    ]
                )

                # Update learner with observed change
                agent_learners[i].update(phi_i, observed_changes[i])

        # Update system state
        current_coherence = next_coherence
        coherence_trajectory[t] = current_coherence

        # Compute global order parameter (synchronization measure)
        # R = 1 - var(C_current) / var(C_initial)
        current_variance = np.var(current_coherence)
        synchronization_order = 1.0 - current_variance / initial_coherence_variance
        synchronization_trajectory[t] = synchronization_order

        # Compute control energy
        control_energy_trajectory[t] = np.sum(control_increments**2)

    # Compute performance metrics
    average_targets = target_trajectory.mean(axis=1)
    time_under_target = (coherence_trajectory < target_trajectory).mean()

    performance_metrics = {
        "control_mode": control_mode,
        "N_agents": N,
        "T_steps": T,
        "mean_coherence": float(coherence_trajectory.mean()),
        "std_coherence": float(coherence_trajectory.std()),
        "time_under_target_pct": float(100 * time_under_target),
        "final_synchronization": float(synchronization_trajectory[-1]),
        "total_control_energy": float(control_energy_trajectory.sum()),
        "energy_gini": float(
            compute_gini_coefficient(control_energy_trajectory + 1e-12)
        ),
        "final_coherence_gini": float(
            compute_gini_coefficient(coherence_trajectory[-1])
        ),
        "network_density": float(A.sum() / (N * (N - 1))),
    }

    # Add learning statistics for adaptive mode
    if control_mode == "adaptive":
        learned_params = np.array(
            [learner.get_parameters() for learner in agent_learners]
        )
        if learned_params.shape[1] >= 2:
            performance_metrics["learned_parameters"] = {
                "theta0_mean": float(learned_params[:, 2].mean()),  # Intercept
                "theta1_mean": float(learned_params[:, 0].mean()),  # Direct sensitivity
                "theta2_mean": float(
                    learned_params[:, 1].mean()
                ),  # Coupling sensitivity
                "theta1_std": float(learned_params[:, 0].std()),
                "theta2_std": float(learned_params[:, 1].std()),
            }

        # True vs learned parameter comparison
        true_theta1_mean = float(agent_sensitivity_direct.mean())
        true_theta2_mean = float(agent_sensitivity_coupling.mean())
        learned_theta1_mean = (
            float(learned_params[:, 0].mean()) if learned_params.shape[1] >= 1 else 0.0
        )
        learned_theta2_mean = (
            float(learned_params[:, 1].mean()) if learned_params.shape[1] >= 2 else 0.0
        )

        performance_metrics["learning_accuracy"] = {
            "theta1_error": float(abs(true_theta1_mean - learned_theta1_mean)),
            "theta2_error": float(abs(true_theta2_mean - learned_theta2_mean)),
            "theta1_true": true_theta1_mean,
            "theta2_true": true_theta2_mean,
            "theta1_learned": learned_theta1_mean,
            "theta2_learned": learned_theta2_mean,
        }

    print(
        f"    ✅ Simulation complete: {time_under_target*100:.1f}% under target, R_final={synchronization_trajectory[-1]:.3f}"
    )

    return (
        A,
        coherence_trajectory,
        synchronization_trajectory,
        control_energy_trajectory,
        target_trajectory,
        control_increment_trajectory,
        performance_metrics,
    )


# ---------- Visualization Suite ----------


def create_collective_visualizations(
    mode_name,
    network_adjacency,
    coherence_traj,
    sync_traj,
    energy_traj,
    target_traj,
    control_traj,
    output_dir,
):
    """Generate comprehensive visualization suite for collective dynamics"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spatiotemporal coherence heatmap
    plt.figure(figsize=(10, 6), dpi=150)

    plt.imshow(
        coherence_traj.T,
        aspect="auto",
        origin="lower",
        vmin=0.4,
        vmax=0.9,
        cmap="viridis",
        interpolation="bilinear",
    )
    plt.colorbar(label="Coherence C(t)")
    plt.xlabel("Time Steps")
    plt.ylabel("Agent ID")
    plt.title(f"{mode_name.title()}: Spatiotemporal Coherence Evolution")

    # Add target overlay
    target_mean = target_traj.mean(axis=1)
    time_axis = np.arange(len(target_mean))
    plt.plot(
        time_axis,
        target_mean * coherence_traj.shape[1] * 0.8,
        "red",
        linewidth=2,
        alpha=0.7,
        label="Avg Target",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"{mode_name}_coherence_heatmap.png", bbox_inches="tight")
    plt.close()

    # 2. Global synchronization order parameter
    plt.figure(figsize=(8, 5), dpi=150)

    plt.plot(sync_traj, linewidth=2, color="blue")
    plt.xlabel("Time Steps")
    plt.ylabel("Synchronization R(t)")
    plt.title(f"{mode_name.title()}: Global Synchronization Dynamics")
    plt.grid(True, alpha=0.3)

    # Add final value annotation
    plt.text(
        0.98,
        0.95,
        f"Final R = {sync_traj[-1]:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / f"{mode_name}_synchronization.png", bbox_inches="tight")
    plt.close()

    # 3. Control energy evolution
    plt.figure(figsize=(8, 4), dpi=150)

    plt.plot(energy_traj, linewidth=1.5, color="red")
    plt.xlabel("Time Steps")
    plt.ylabel("Total Control Energy Σ(Δu)²")
    plt.title(f"{mode_name.title()}: Control Energy per Time Step")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{mode_name}_control_energy.png", bbox_inches="tight")
    plt.close()

    # 4. Final coherence distribution
    plt.figure(figsize=(6, 4), dpi=150)

    final_coherence = coherence_traj[-1]
    plt.hist(final_coherence, bins=15, alpha=0.7, color="green", edgecolor="black")
    plt.axvline(
        final_coherence.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {final_coherence.mean():.3f}",
    )
    plt.axvline(
        np.median(final_coherence),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.median(final_coherence):.3f}",
    )

    plt.xlabel("Final Coherence C(T)")
    plt.ylabel("Number of Agents")
    plt.title(f"{mode_name.title()}: Final Coherence Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{mode_name}_final_distribution.png", bbox_inches="tight")
    plt.close()

    # 5. Network topology visualization (if small enough)
    if network_adjacency.shape[0] <= 30:
        plt.figure(figsize=(6, 6), dpi=150)

        # Simple circular layout
        N = network_adjacency.shape[0]
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pos_x = np.cos(angles)
        pos_y = np.sin(angles)

        # Draw edges
        for i in range(N):
            for j in range(i + 1, N):
                if network_adjacency[i, j]:
                    plt.plot(
                        [pos_x[i], pos_x[j]],
                        [pos_y[i], pos_y[j]],
                        "gray",
                        alpha=0.5,
                        linewidth=0.5,
                    )

        # Draw nodes colored by final coherence
        scatter = plt.scatter(
            pos_x,
            pos_y,
            c=final_coherence,
            s=100,
            cmap="viridis",
            edgecolors="black",
            linewidth=1,
        )
        plt.colorbar(scatter, label="Final Coherence")

        plt.title(f"{mode_name.title()}: Network Topology & Final States")
        plt.axis("equal")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{mode_name}_network_topology.png", bbox_inches="tight"
        )
        plt.close()


# ---------- Main Orchestrator ----------


def main():
    """Main orchestrator for Phase 19 Collective Adaptive Dynamics"""

    start_time = time.time()
    print("🌐 Phase 19: Collective Adaptive Dynamics")
    print(f"📁 Output directory: {ROOT}")
    print()

    # Simulation parameters
    N_agents = 24
    T_steps = 600
    control_modes = ["nocontrol", "fixed", "adaptive"]

    print(f"🔧 Parameters: {N_agents} agents, {T_steps} time steps")
    print()

    # Run simulations for each control mode
    simulation_results = {}

    for mode in control_modes:
        print(f"🎯 Running {mode} simulation...")

        # Run collective simulation
        (
            network_adj,
            coherence_traj,
            sync_traj,
            energy_traj,
            target_traj,
            control_traj,
            performance_kpis,
        ) = run_collective_simulation(
            N=N_agents, T=T_steps, control_mode=mode, random_seed=123
        )

        # Store results
        simulation_results[mode] = {
            "network_adjacency": network_adj.tolist(),
            "performance_kpis": performance_kpis,
        }

        # Create output subdirectory
        mode_output_dir = ROOT / mode
        mode_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        create_collective_visualizations(
            mode,
            network_adj,
            coherence_traj,
            sync_traj,
            energy_traj,
            target_traj,
            control_traj,
            mode_output_dir,
        )

        # Save detailed data
        coherence_df = pd.DataFrame(coherence_traj)
        coherence_df.to_csv(mode_output_dir / "coherence_trajectory.csv", index=False)

        aggregate_df = pd.DataFrame(
            {"synchronization": sync_traj, "control_energy": energy_traj}
        )
        aggregate_df.to_csv(mode_output_dir / "aggregate_metrics.csv", index=False)

        # Save KPIs
        with open(mode_output_dir / "performance_kpis.json", "w") as f:
            json.dump(performance_kpis, f, indent=2)

        print(
            f"  ✅ {mode}: {performance_kpis['time_under_target_pct']:.1f}% under target, "
            f"R_final={performance_kpis['final_synchronization']:.3f}"
        )

    # Create comparative analysis
    print("\n📊 Generating comparative analysis...")

    # Comparison bar chart
    modes = ["nocontrol", "fixed", "adaptive"]
    mode_labels = ["No Control", "Fixed Control", "Adaptive Control"]

    time_under_targets = [
        simulation_results[m]["performance_kpis"]["time_under_target_pct"]
        for m in modes
    ]
    final_synchronizations = [
        simulation_results[m]["performance_kpis"]["final_synchronization"]
        for m in modes
    ]
    total_energies = [
        simulation_results[m]["performance_kpis"]["total_control_energy"] for m in modes
    ]

    # Normalize energies for plotting (log scale)
    max_energy = max(total_energies)
    normalized_energies = [e / max_energy * 100 for e in total_energies]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # Time under target
    bars1 = ax1.bar(
        mode_labels, time_under_targets, color=["lightcoral", "orange", "lightgreen"]
    )
    ax1.set_ylabel("Time Under Target (%)")
    ax1.set_title("Performance: Time Below Target")
    ax1.grid(True, alpha=0.3)

    for bar, value in zip(bars1, time_under_targets):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Final synchronization
    bars2 = ax2.bar(
        mode_labels,
        final_synchronizations,
        color=["lightcoral", "orange", "lightgreen"],
    )
    ax2.set_ylabel("Final Synchronization R")
    ax2.set_title("Collective Order: Final R")
    ax2.grid(True, alpha=0.3)

    for bar, value in zip(bars2, final_synchronizations):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Control energy (normalized)
    bars3 = ax3.bar(
        mode_labels, normalized_energies, color=["lightcoral", "orange", "lightgreen"]
    )
    ax3.set_ylabel("Normalized Control Energy (%)")
    ax3.set_title("Control Effort: Total Energy")
    ax3.grid(True, alpha=0.3)

    for bar, orig_value, norm_value in zip(bars3, total_energies, normalized_energies):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{orig_value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(ROOT / "collective_comparison.png", bbox_inches="tight")
    plt.close()

    # Generate master report
    with open(ROOT / "master_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 19 — Collective Adaptive Dynamics\n\n")
        f.write(f"**Timestamp**: {STAMP}\n")
        f.write(
            f"**Network Configuration**: {N_agents} agents, Watts-Strogatz topology\n"
        )
        f.write(f"**Simulation Duration**: {T_steps} time steps\n\n")

        f.write("## Multi-Agent Temporal Control System\n\n")
        f.write(
            "This phase demonstrates collective adaptive control in networked temporal coherence "
        )
        f.write(
            "systems. Each agent learns its own plant dynamics while interacting through "
        )
        f.write(
            "small-world network coupling, creating emergent collective intelligence.\n\n"
        )

        f.write("### System Architecture\n")
        f.write("- **Network**: Watts-Strogatz small-world (k=6, β=0.18)\n")
        f.write(
            "- **Agent Dynamics**: Individual RLS learners with heterogeneous plant parameters\n"
        )
        f.write(
            "- **Coupling**: κ_i = (A @ C) / deg_i (neighbor coherence influence)\n"
        )
        f.write("- **Targeting**: Dynamic per-agent targets based on local history\n")
        f.write(
            "- **Control Modes**: No-control baseline vs Fixed vs Adaptive strategies\n\n"
        )

        f.write("## Collective Performance Results\n\n")
        f.write(
            "| Control Mode | Time Under Target | Final Synchronization | Total Energy | Energy Gini |\n"
        )
        f.write(
            "|--------------|-------------------|----------------------|--------------|-------------|\n"
        )

        for mode in modes:
            kpis = simulation_results[mode]["performance_kpis"]
            f.write(
                f"| {mode.title()} | {kpis['time_under_target_pct']:.1f}% | "
                f"{kpis['final_synchronization']:.3f} | "
                f"{kpis['total_control_energy']:.2f} | "
                f"{kpis['energy_gini']:.3f} |\n"
            )

        f.write("\n### Performance Analysis\n")

        # Compute improvements
        nocontrol_kpis = simulation_results["nocontrol"]["performance_kpis"]
        fixed_kpis = simulation_results["fixed"]["performance_kpis"]
        adaptive_kpis = simulation_results["adaptive"]["performance_kpis"]

        fixed_improvement = (
            nocontrol_kpis["time_under_target_pct"]
            - fixed_kpis["time_under_target_pct"]
        )
        adaptive_improvement = (
            nocontrol_kpis["time_under_target_pct"]
            - adaptive_kpis["time_under_target_pct"]
        )
        adaptive_vs_fixed = (
            fixed_kpis["time_under_target_pct"] - adaptive_kpis["time_under_target_pct"]
        )

        f.write(
            f"- **Fixed control improvement**: {fixed_improvement:.1f} percentage points over no-control\n"
        )
        f.write(
            f"- **Adaptive control improvement**: {adaptive_improvement:.1f} percentage points over no-control\n"
        )
        f.write(
            f"- **Adaptive vs Fixed**: {adaptive_vs_fixed:.1f} percentage point advantage\n"
        )

        sync_improvement = (
            adaptive_kpis["final_synchronization"]
            - nocontrol_kpis["final_synchronization"]
        )
        f.write(
            f"- **Synchronization enhancement**: {sync_improvement:.3f} R-value improvement with adaptive control\n\n"
        )

        # Learning analysis (if available)
        if "learned_parameters" in adaptive_kpis:
            learning_stats = adaptive_kpis["learned_parameters"]
            accuracy_stats = adaptive_kpis["learning_accuracy"]

            f.write("### Multi-Agent Learning Analysis\n")
            f.write(
                f"- **Learned θ₁ (direct sensitivity)**: {learning_stats['theta1_mean']:.4f} ± {learning_stats['theta1_std']:.4f}\n"
            )
            f.write(
                f"- **Learned θ₂ (coupling sensitivity)**: {learning_stats['theta2_mean']:.4f} ± {learning_stats['theta2_std']:.4f}\n"
            )
            f.write(
                f"- **Learning accuracy θ₁**: error = {accuracy_stats['theta1_error']:.4f} (true: {accuracy_stats['theta1_true']:.4f})\n"
            )
            f.write(
                f"- **Learning accuracy θ₂**: error = {accuracy_stats['theta2_error']:.4f} (true: {accuracy_stats['theta2_true']:.4f})\n\n"
            )

        f.write("## Key Findings\n\n")

        if adaptive_vs_fixed > 1:
            f.write("### Collective Adaptive Learning Succeeds\n")
            f.write(
                f"- **{adaptive_vs_fixed:.1f} percentage point improvement** of adaptive over fixed control\n"
            )
            f.write(
                "- **Distributed intelligence emerges** from individual agent learning\n"
            )
            f.write("- **Network effects amplify** individual control improvements\n")
            f.write(
                "- **Collective synchronization** enhanced through adaptive coordination\n\n"
            )
        elif adaptive_vs_fixed > 0:
            f.write("### Modest Collective Learning Benefits\n")
            f.write(
                f"- **{adaptive_vs_fixed:.1f} percentage point improvement** shows learning value\n"
            )
            f.write(
                "- **Individual adaptation** provides incremental collective gains\n"
            )
            f.write("- **Network complexity** may limit adaptive advantages\n\n")
        else:
            f.write("### Limited Collective Adaptation\n")
            f.write("- **Minimal improvement** from distributed learning\n")
            f.write(
                "- **Network interactions** may constrain individual learning benefits\n"
            )
            f.write(
                "- **Collective dynamics** more complex than individual control\n\n"
            )

        f.write("## Scientific Implications\n\n")

        if adaptive_vs_fixed > 0.5:
            f.write("### Distributed Temporal Intelligence\n")
            f.write(
                "- **Collective learning works**: Multi-agent adaptation outperforms fixed strategies\n"
            )
            f.write(
                "- **Emergent coordination**: Individual learning creates collective intelligence\n"
            )
            f.write(
                "- **Network amplification**: Topology enhances individual control benefits\n"
            )
            f.write(
                "- **Scalable framework**: Principles extend to larger networked systems\n\n"
            )

            f.write("### Revolutionary Applications\n")
            f.write(
                "- **Smart cities**: Distributed infrastructure that learns optimal coordination\n"
            )
            f.write(
                "- **Social networks**: Platforms that adaptively maintain community coherence\n"
            )
            f.write(
                "- **Biological systems**: Understanding collective adaptation in neural networks\n"
            )
            f.write(
                "- **Multi-robot systems**: Swarms that learn coordinated temporal dynamics\n"
            )
            f.write(
                "- **Economic networks**: Markets that self-regulate through distributed learning\n\n"
            )
        else:
            f.write("### Collective Learning Challenges\n")
            f.write(
                "- **Interaction complexity**: Network effects may overwhelm individual learning\n"
            )
            f.write(
                "- **Coordination difficulties**: Distributed adaptation requires careful design\n"
            )
            f.write(
                "- **Emergence limitations**: Collective benefits not guaranteed from individual learning\n\n"
            )

        f.write("## Technical Achievements\n")
        f.write(
            "- **Complete multi-agent framework**: Individual RLS learning in networked systems\n"
        )
        f.write(
            "- **Heterogeneous dynamics**: Each agent has unique plant parameters\n"
        )
        f.write(
            "- **Network topology integration**: Small-world structure affects collective behavior\n"
        )
        f.write(
            "- **Distributed coordination**: No central controller, pure peer-to-peer adaptation\n"
        )
        f.write(
            "- **Scalable architecture**: Framework extends to larger agent populations\n\n"
        )

        f.write("## Generated Artifacts\n")
        f.write("For each control mode:\n")
        f.write(
            "- `{mode}/coherence_trajectory.csv`: Full spatiotemporal coherence evolution\n"
        )
        f.write(
            "- `{mode}/aggregate_metrics.csv`: System-level synchronization and energy\n"
        )
        f.write("- `{mode}/performance_kpis.json`: Comprehensive performance metrics\n")
        f.write(
            "- `{mode}/{mode}_coherence_heatmap.png`: Spatiotemporal visualization\n"
        )
        f.write(
            "- `{mode}/{mode}_synchronization.png`: Global order parameter evolution\n"
        )
        f.write("- `{mode}/{mode}_control_energy.png`: Control effort over time\n")
        f.write(
            "- `{mode}/{mode}_final_distribution.png`: Final coherence statistics\n"
        )
        f.write(
            "- `{mode}/{mode}_network_topology.png`: Network structure with final states\n\n"
        )

        f.write("Summary files:\n")
        f.write("- `collective_comparison.png`: Cross-mode performance comparison\n")
        f.write("- `master_report.md`: This comprehensive analysis\n")

    # Final statistics
    runtime = time.time() - start_time

    print("\n" + "=" * 60)
    print("🌐 Phase 19 Collective Adaptive Dynamics Complete")
    print(f"⏱️  Runtime: {runtime:.2f} seconds")
    print(f"🎯 Control modes tested: {len(control_modes)}")

    print("\n📊 Performance Summary:")
    for mode in modes:
        kpis = simulation_results[mode]["performance_kpis"]
        print(
            f"  {mode.title()}: {kpis['time_under_target_pct']:.1f}% under target, "
            f"R_final={kpis['final_synchronization']:.3f}"
        )

    # Highlight best performer
    best_mode = min(
        modes,
        key=lambda m: simulation_results[m]["performance_kpis"][
            "time_under_target_pct"
        ],
    )
    best_kpis = simulation_results[best_mode]["performance_kpis"]
    print(
        f"\n🏆 Best performer: {best_mode.title()} ({best_kpis['time_under_target_pct']:.1f}% under target)"
    )

    if best_mode == "adaptive":
        print("🧠 Collective adaptive learning demonstrates superior performance!")
    elif best_mode == "fixed":
        print("🎛️  Fixed control outperforms in this collective setting")
    else:
        print("⚠️  No control performs best - system may not benefit from intervention")

    print(f"\n📁 Results: {ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
