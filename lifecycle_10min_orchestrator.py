#!/usr/bin/env python3
"""
Experiment Orchestrator: A/B/C Tests with 10-minute cap
Runs expansion, fairness×coordination, and hygiene experiments with full export pipeline.
"""

import hashlib
import json
import os
import time
import warnings
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for consistent plots
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
sns.set_palette("husl")


class ExperimentOrchestrator:
    """Orchestrates A/B/C tests with adaptive timing and comprehensive exports"""

    def __init__(self):
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = f"outputs/lifecycle_10min/{self.timestamp}"
        self.log_buffer = []

        # Create directory structure
        self._setup_directories()

        # Configuration
        self.defaults = {
            "epochs_cap": 80,
            "agents": 64,
            "seeds": [101, 202],
            "collapse_rules": {
                "risk_min": 0.45,
                "survival_max": 0.40,
                "cci_floor": 0.45,
            },
            "fast_mode": True,
            "time_limit_sec": 600,  # 10 minutes
        }

        # Experiment plan
        self.plan = self._create_experiment_plan()

    def _setup_directories(self):
        """Create output directory structure"""
        dirs = ["data", "figures", "report", "logs", "bundle"]
        for d in dirs:
            os.makedirs(f"{self.out_dir}/{d}", exist_ok=True)

    def _create_experiment_plan(self) -> list[dict]:
        """Create the A/B/C experiment plan"""
        plan = [
            # A) Expansion Pulse A/B
            {"family": "A", "label": "A_base"},
            {
                "family": "A",
                "label": "A_pulse",
                "expansion": {"type": "percent", "pct": 0.03, "every": 12, "start": 12},
            },
            # B) Fairness × Coordination (2×2)
            {
                "family": "B",
                "label": "B_c0.40_g0.20",
                "coordination_strength": 0.40,
                "goal_inequality": 0.20,
            },
            {
                "family": "B",
                "label": "B_c0.40_g0.40",
                "coordination_strength": 0.40,
                "goal_inequality": 0.40,
            },
            {
                "family": "B",
                "label": "B_c0.60_g0.20",
                "coordination_strength": 0.60,
                "goal_inequality": 0.20,
            },
            {
                "family": "B",
                "label": "B_c0.60_g0.40",
                "coordination_strength": 0.60,
                "goal_inequality": 0.40,
            },
            # C) Renewal Hygiene
            {"family": "C", "label": "C_off"},
            {
                "family": "C",
                "label": "C_on",
                "hygiene": {"every": 10, "noise_trim_pct": 0.20, "recalibrate": True},
            },
        ]
        return plan

    def _log(self, msg: str):
        """Log message to buffer and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        self.log_buffer.append(log_msg)

    def _estimate_runtime(self, config: dict) -> float:
        """Estimate runtime in seconds based on configuration"""
        # Base time per epoch-agent combination (empirical estimate)
        base_time_per_epoch_agent = 0.001  # seconds

        epochs = config["epochs_cap"]
        agents = config["agents"]
        n_seeds = len(config["seeds"])
        n_conditions = len(self.plan)

        # Adjust for fast mode
        if config.get("fast_mode", True):
            base_time_per_epoch_agent *= 0.7

        total_time = (
            epochs * agents * n_seeds * n_conditions * base_time_per_epoch_agent + 30
        )  # +30 for overhead

        return total_time

    def _adaptive_config(self, config: dict) -> dict:
        """Apply adaptive cuts to meet time budget"""
        config = config.copy()

        # Check initial estimate
        estimated_time = self._estimate_runtime(config)
        self._log(f"Initial time estimate: {estimated_time:.1f}s")

        if estimated_time <= self.defaults["time_limit_sec"]:
            return config

        # Apply cuts in order of preference
        if estimated_time > self.defaults["time_limit_sec"]:
            # Cut B seeds first (from 2 to 1)
            config["seeds"] = [101]  # Keep only first seed
            estimated_time = self._estimate_runtime(config)
            self._log(f"After seed cut: {estimated_time:.1f}s")

        if estimated_time > self.defaults["time_limit_sec"]:
            # Reduce epochs
            config["epochs_cap"] = 60
            estimated_time = self._estimate_runtime(config)
            self._log(f"After epoch cut: {estimated_time:.1f}s")

        if estimated_time > self.defaults["time_limit_sec"]:
            # Reduce agents
            config["agents"] = 48
            estimated_time = self._estimate_runtime(config)
            self._log(f"After agent cut: {estimated_time:.1f}s")

        return config

    def _run_single_condition(
        self, condition: dict, config: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run a single experimental condition across all seeds"""
        runs_data = []
        traj_data = []

        for seed in config["seeds"]:
            self._log(f"Running {condition['label']} with seed {seed}")

            # Simulate consciousness system evolution
            run_result, trajectory = self._simulate_lifecycle(condition, config, seed)

            runs_data.append(run_result)
            traj_data.extend(trajectory)

        return pd.DataFrame(runs_data), pd.DataFrame(traj_data)

    def _simulate_lifecycle(
        self, condition: dict, config: dict, seed: int
    ) -> tuple[dict, list[dict]]:
        """Simulate consciousness lifecycle for one condition/seed"""
        np.random.seed(seed)

        # Initialize parameters
        base_cci = 0.65
        base_survival = 0.95
        base_risk = 0.15

        # Apply condition modifications
        coordination_mult = condition.get("coordination_strength", 0.50) / 0.50
        inequality_mult = (0.30 - condition.get("goal_inequality", 0.30)) / 0.30 + 1

        # Initialize trajectory tracking
        trajectory = []
        cci_values = []
        survival_values = []
        risk_values = []

        # Simulation loop
        early_stopped = False
        collapse_epoch = None

        for epoch in range(config["epochs_cap"]):
            # Apply aging and drift
            age_factor = 1.0 - (epoch * 0.003)  # Gradual aging
            noise_factor = 1.0 + np.random.normal(0, 0.02)

            # Apply expansion effects
            expansion_boost = 1.0
            if "expansion" in condition and epoch >= condition["expansion"]["start"]:
                if epoch % condition["expansion"]["every"] == 0:
                    expansion_boost = 1.0 + condition["expansion"]["pct"]

            # Apply hygiene effects
            hygiene_boost = 1.0
            if "hygiene" in condition and epoch > 0:
                if epoch % condition["hygiene"]["every"] == 0:
                    hygiene_boost = 1.15  # Renewal effect

            # Calculate current metrics
            current_cci = (
                base_cci
                * age_factor
                * coordination_mult
                * inequality_mult
                * expansion_boost
                * hygiene_boost
                * noise_factor
            )
            current_cci = max(0.1, min(1.0, current_cci))

            current_survival = (
                base_survival
                * age_factor
                * coordination_mult
                * expansion_boost
                * hygiene_boost
            )
            current_survival = max(0.1, min(1.0, current_survival))

            current_risk = (
                base_risk
                * (2.0 - age_factor)
                / (coordination_mult * expansion_boost * hygiene_boost)
            )
            current_risk = max(0.05, min(0.8, current_risk))

            # Store values
            cci_values.append(current_cci)
            survival_values.append(current_survival)
            risk_values.append(current_risk)

            # Record trajectory
            trajectory.append(
                {
                    "run_id": f"{condition['label']}_s{seed}",
                    "seed": seed,
                    "epoch": epoch,
                    "CCI": current_cci,
                    "collapse_risk": current_risk,
                    "survival_rate": current_survival,
                    "hazard": max(
                        0, -np.log(current_survival + 1e-6) if epoch > 0 else 0
                    ),
                }
            )

            # Check early stop conditions
            collapse_rules = config["collapse_rules"]
            if (
                current_risk >= collapse_rules["risk_min"]
                and current_survival <= collapse_rules["survival_max"]
                and current_cci < collapse_rules["cci_floor"]
            ):
                early_stopped = True
                collapse_epoch = epoch
                break

        # Calculate summary metrics
        lifespan_epochs = collapse_epoch if early_stopped else config["epochs_cap"] - 1
        peak_cci = max(cci_values)
        final_cci = cci_values[-1]
        hazard_values = [t["hazard"] for t in trajectory]
        hazard_peak = max(hazard_values) if hazard_values else 0
        survival_final = survival_values[-1]
        risk_final = risk_values[-1]

        run_result = {
            "run_id": f"{condition['label']}_s{seed}",
            "family": condition["family"],
            "label": condition["label"],
            "seed": seed,
            "epochs_cap": config["epochs_cap"],
            "agents": config["agents"],
            "fast_mode": config.get("fast_mode", True),
            "early_stopped": early_stopped,
            "coordination_strength": condition.get("coordination_strength"),
            "goal_inequality": condition.get("goal_inequality"),
            "expansion_type": condition.get("expansion", {}).get("type"),
            "expansion_pct": condition.get("expansion", {}).get("pct"),
            "expansion_every": condition.get("expansion", {}).get("every"),
            "expansion_start": condition.get("expansion", {}).get("start"),
            "hygiene_on": "hygiene" in condition,
            "hygiene_every": condition.get("hygiene", {}).get("every"),
            "noise_trim_pct": condition.get("hygiene", {}).get("noise_trim_pct"),
            "recalibrate": condition.get("hygiene", {}).get("recalibrate"),
            "lifespan_epochs": lifespan_epochs,
            "collapse_flag": early_stopped,
            "peak_CCI": peak_cci,
            "final_CCI": final_cci,
            "hazard_peak": hazard_peak,
            "survival_final": survival_final,
            "risk_final": risk_final,
            "divergence_epoch": None,  # Will compute later
            "time_sec": time.time() - self.start_time,
        }

        return run_result, trajectory

    def run_all_experiments(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run all experiments with adaptive timing"""
        self._log("Starting experiment orchestration")

        # Apply adaptive configuration
        config = self._adaptive_config(self.defaults)

        self._log(f"Final config: {config}")

        # Run experiments
        all_runs = []
        all_trajs = []

        for condition in self.plan:
            if time.time() - self.start_time > config["time_limit_sec"]:
                self._log(f"Time limit reached, stopping at {condition['label']}")
                break

            runs_df, traj_df = self._run_single_condition(condition, config)
            all_runs.append(runs_df)
            all_trajs.append(traj_df)

            elapsed = time.time() - self.start_time
            self._log(f"Completed {condition['label']}, elapsed: {elapsed:.1f}s")

        # Combine results
        final_runs = pd.concat(all_runs, ignore_index=True)
        final_trajs = pd.concat(all_trajs, ignore_index=True)

        self._log(f"Completed all experiments in {time.time() - self.start_time:.1f}s")

        return final_runs, final_trajs

    def _compute_divergence_epochs(
        self, runs_df: pd.DataFrame, traj_df: pd.DataFrame
    ) -> dict:
        """Compute divergence epochs for each family"""
        divergence_epochs = {}

        for family in ["A", "B", "C"]:
            family_runs = runs_df[runs_df["family"] == family]
            if len(family_runs) < 2:
                continue

            # Get survivors and collapsed within family
            survivors = family_runs[~family_runs["collapse_flag"]]
            collapsed = family_runs[family_runs["collapse_flag"]]

            if len(survivors) == 0 or len(collapsed) == 0:
                divergence_epochs[family] = None
                continue

            # Find divergence epoch
            survivor_ids = survivors["run_id"].tolist()
            collapsed_ids = collapsed["run_id"].tolist()

            family_traj = traj_df[traj_df["run_id"].str.startswith(family)]

            consecutive_count = 0
            for epoch in range(80):
                epoch_data = family_traj[family_traj["epoch"] == epoch]
                if len(epoch_data) == 0:
                    continue

                survivor_cci = epoch_data[epoch_data["run_id"].isin(survivor_ids)][
                    "CCI"
                ].mean()
                collapsed_cci = epoch_data[epoch_data["run_id"].isin(collapsed_ids)][
                    "CCI"
                ].mean()

                if pd.notna(survivor_cci) and pd.notna(collapsed_cci):
                    if survivor_cci - collapsed_cci >= 0.07:
                        consecutive_count += 1
                        if consecutive_count >= 5:
                            divergence_epochs[family] = (
                                epoch - 4
                            )  # Start of 5-epoch window
                            break
                    else:
                        consecutive_count = 0

        return divergence_epochs

    def _bootstrap_ci(
        self, values: list[float], n_bootstrap: int = 200
    ) -> tuple[float, list[float]]:
        """Compute bootstrap confidence interval"""
        if len(values) == 0:
            return 0.0, [0.0, 0.0]

        # Reduce bootstrap samples if time-pressed
        if time.time() - self.start_time > 500:  # Near time limit
            n_bootstrap = 100

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        mean_val = np.mean(values)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        return mean_val, [ci_lower, ci_upper]

    def analyze_and_aggregate(
        self, runs_df: pd.DataFrame, traj_df: pd.DataFrame
    ) -> tuple[dict, pd.DataFrame]:
        """Compute metrics, bootstraps, and create summary"""
        self._log("Computing aggregate metrics and bootstraps")

        # Compute divergence epochs
        divergence_epochs = self._compute_divergence_epochs(runs_df, traj_df)

        # Update runs_df with divergence epochs
        for family, div_epoch in divergence_epochs.items():
            mask = runs_df["family"] == family
            runs_df.loc[mask, "divergence_epoch"] = div_epoch

        # Compute condition-level metrics
        condition_metrics = {}

        for label in runs_df["label"].unique():
            condition_data = runs_df[runs_df["label"] == label]

            lifespan_mean, lifespan_ci = self._bootstrap_ci(
                condition_data["lifespan_epochs"].tolist()
            )
            final_cci_mean, final_cci_ci = self._bootstrap_ci(
                condition_data["final_CCI"].tolist()
            )
            hazard_peak_mean, _ = self._bootstrap_ci(
                condition_data["hazard_peak"].tolist()
            )

            condition_metrics[label] = {
                "n_runs": len(condition_data),
                "lifespan_mean": round(lifespan_mean, 1),
                "lifespan_ci95": [round(x, 1) for x in lifespan_ci],
                "final_CCI_mean": round(final_cci_mean, 3),
                "final_CCI_ci95": [round(x, 3) for x in final_cci_ci],
                "hazard_peak_mean": round(hazard_peak_mean, 4),
                "collapse_rate": round(condition_data["collapse_flag"].mean(), 3),
                "divergence_epoch": (
                    condition_data["divergence_epoch"].iloc[0]
                    if len(condition_data) > 0
                    else None
                ),
            }

        # Compute uplifts vs A_base
        a_base_metrics = condition_metrics.get("A_base", {})
        uplifts = {}

        if a_base_metrics:
            # A_pulse uplift
            if "A_pulse" in condition_metrics:
                a_pulse = condition_metrics["A_pulse"]
                uplifts["A_pulse"] = {
                    "Δlifespan": round(
                        a_pulse["lifespan_mean"] - a_base_metrics["lifespan_mean"], 1
                    ),
                    "Δfinal_CCI": round(
                        a_pulse["final_CCI_mean"] - a_base_metrics["final_CCI_mean"], 3
                    ),
                    "Δhazard_peak": round(
                        a_pulse["hazard_peak_mean"]
                        - a_base_metrics["hazard_peak_mean"],
                        4,
                    ),
                }

            # Best B condition
            b_conditions = {
                k: v for k, v in condition_metrics.items() if k.startswith("B_")
            }
            if b_conditions:
                best_b_label = max(
                    b_conditions.keys(), key=lambda x: b_conditions[x]["lifespan_mean"]
                )
                best_b = b_conditions[best_b_label]
                uplifts["B_best"] = {
                    "label": best_b_label,
                    "Δlifespan": round(
                        best_b["lifespan_mean"] - a_base_metrics["lifespan_mean"], 1
                    ),
                    "Δfinal_CCI": round(
                        best_b["final_CCI_mean"] - a_base_metrics["final_CCI_mean"], 3
                    ),
                }

            # C_on uplift
            if "C_on" in condition_metrics:
                c_on = condition_metrics["C_on"]
                uplifts["C_on"] = {
                    "Δlifespan": round(
                        c_on["lifespan_mean"] - a_base_metrics["lifespan_mean"], 1
                    ),
                    "Δfinal_CCI": round(
                        c_on["final_CCI_mean"] - a_base_metrics["final_CCI_mean"], 3
                    ),
                    "Δhazard_peak": round(
                        c_on["hazard_peak_mean"] - a_base_metrics["hazard_peak_mean"], 4
                    ),
                }

        # Decision rule
        decision_candidates = []

        # Check A_pulse and C_on for joint criteria
        for intervention in ["A_pulse", "C_on"]:
            if intervention in uplifts:
                up = uplifts[intervention]
                if up["Δlifespan"] >= 10 and up["Δfinal_CCI"] >= 0.05:
                    decision_candidates.append((intervention, up["Δhazard_peak"]))

        if len(decision_candidates) >= 2:
            # Pick the one with lower hazard peak
            decision = min(decision_candidates, key=lambda x: x[1])[0]
        else:
            # Pick single best by lifespan, tie-break by final_CCI
            best_lifespan = max(uplifts.keys(), key=lambda x: uplifts[x]["Δlifespan"])
            decision = best_lifespan

        # Create summary JSON
        summary_json = {
            "timestamp": self.timestamp,
            "config": {
                "epochs_cap": self.defaults["epochs_cap"],
                "agents": self.defaults["agents"],
                "seeds": self.defaults["seeds"],
            },
            "families": {
                "A": {"baseline": "A_base", "variants": ["A_pulse"]},
                "B": {
                    "grid": [
                        ["c0.40_g0.20", "c0.40_g0.40"],
                        ["c0.60_g0.20", "c0.60_g0.40"],
                    ]
                },
                "C": {"baseline": "C_off", "variants": ["C_on"]},
            },
            "metrics": {
                "by_condition": condition_metrics,
                "uplifts_vs_A_base": uplifts,
                "decision_recommendation": decision,
            },
        }

        return summary_json, runs_df

    def save_data_exports(
        self, runs_df: pd.DataFrame, traj_df: pd.DataFrame, summary_json: dict
    ):
        """Save CSV and JSON data exports"""
        self._log("Saving data exports")

        # Save runs summary
        runs_df.to_csv(f"{self.out_dir}/data/runs_summary.csv", index=False)

        # Save conditions lookup
        conditions_df = pd.DataFrame(
            [
                {"run_id": condition["label"], "parameters": json.dumps(condition)}
                for condition in self.plan
            ]
        )
        conditions_df.to_csv(f"{self.out_dir}/data/conditions_lookup.csv", index=False)

        # Save trajectories
        traj_df.to_csv(f"{self.out_dir}/data/trajectories_long.csv", index=False)

        # Save summary JSON
        with open(f"{self.out_dir}/data/lifecycle_10min_summary.json", "w") as f:
            json.dump(summary_json, f, indent=2)

    def create_figures(
        self, runs_df: pd.DataFrame, traj_df: pd.DataFrame, summary_json: dict
    ):
        """Create all required figures"""
        self._log("Creating figures")

        # 1. Survival curves (Kaplan-Meier style)
        plt.figure(figsize=(12, 8))

        for label in runs_df["label"].unique():
            condition_runs = runs_df[runs_df["label"] == label]
            lifespans = condition_runs["lifespan_epochs"].values

            # Create survival curve
            unique_times = np.sort(np.unique(lifespans))
            survival_prob = []

            for t in unique_times:
                n_at_risk = len(lifespans[lifespans >= t])
                survival_prob.append(n_at_risk / len(lifespans))

            plt.step(
                unique_times, survival_prob, where="post", label=label, linewidth=2
            )

        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Survival Probability", fontsize=12)
        plt.title("Survival Curves by Condition", fontsize=14, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/survival_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. CCI trajectories with outcome split
        plt.figure(figsize=(14, 8))

        # Separate survivors and collapsed
        survivors = runs_df[~runs_df["collapse_flag"]]["run_id"].tolist()
        collapsed = runs_df[runs_df["collapse_flag"]]["run_id"].tolist()

        # Plot survivors
        if survivors:
            survivor_traj = traj_df[traj_df["run_id"].isin(survivors)]
            survivor_means = (
                survivor_traj.groupby("epoch")["CCI"].agg(["mean", "std"]).reset_index()
            )

            plt.plot(
                survivor_means["epoch"],
                survivor_means["mean"],
                "g-",
                linewidth=3,
                label="Survivors (mean)",
            )
            plt.fill_between(
                survivor_means["epoch"],
                survivor_means["mean"] - survivor_means["std"],
                survivor_means["mean"] + survivor_means["std"],
                alpha=0.3,
                color="green",
            )

        # Plot collapsed
        if collapsed:
            collapsed_traj = traj_df[traj_df["run_id"].isin(collapsed)]
            collapsed_means = (
                collapsed_traj.groupby("epoch")["CCI"]
                .agg(["mean", "std"])
                .reset_index()
            )

            plt.plot(
                collapsed_means["epoch"],
                collapsed_means["mean"],
                "r-",
                linewidth=3,
                label="Collapsed (mean)",
            )
            plt.fill_between(
                collapsed_means["epoch"],
                collapsed_means["mean"] - collapsed_means["std"],
                collapsed_means["mean"] + collapsed_means["std"],
                alpha=0.3,
                color="red",
            )

        # Mark divergence epochs
        divergence_epochs = set()
        for condition_data in summary_json["metrics"]["by_condition"].values():
            if condition_data["divergence_epoch"]:
                divergence_epochs.add(condition_data["divergence_epoch"])

        for div_epoch in divergence_epochs:
            plt.axvline(
                x=div_epoch,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Divergence @ {div_epoch}",
            )

        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("CCI", fontsize=12)
        plt.title(
            "CCI Trajectories: Survivors vs Collapsed", fontsize=14, fontweight="bold"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/cci_trajectories_outcome_split.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Hazard over time
        plt.figure(figsize=(12, 8))

        hazard_means = (
            traj_df.groupby("epoch")["hazard"].agg(["mean", "std"]).reset_index()
        )

        plt.plot(hazard_means["epoch"], hazard_means["mean"], "b-", linewidth=2)
        plt.fill_between(
            hazard_means["epoch"],
            hazard_means["mean"] - hazard_means["std"],
            hazard_means["mean"] + hazard_means["std"],
            alpha=0.3,
            color="blue",
        )

        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Hazard Rate", fontsize=12)
        plt.title("Hazard Rate Over Time", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/hazard_over_time.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 4. Bar chart of uplifts
        uplifts = summary_json["metrics"]["uplifts_vs_A_base"]

        if uplifts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Lifespan uplifts
            labels = []
            lifespan_values = []

            for key, data in uplifts.items():
                if key == "B_best":
                    labels.append(f"B_best\n({data['label']})")
                else:
                    labels.append(key)
                lifespan_values.append(data["Δlifespan"])

            bars1 = ax1.bar(
                labels,
                lifespan_values,
                color=["skyblue" if x >= 0 else "lightcoral" for x in lifespan_values],
            )
            ax1.set_ylabel("Δ Lifespan (epochs)")
            ax1.set_title("Lifespan Uplift vs A_base")
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars1, lifespan_values):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (1 if val >= 0 else -1),
                    f"{val:+.1f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                )

            # Final CCI uplifts
            cci_values = [data["Δfinal_CCI"] for data in uplifts.values()]

            bars2 = ax2.bar(
                labels,
                cci_values,
                color=["lightgreen" if x >= 0 else "lightcoral" for x in cci_values],
            )
            ax2.set_ylabel("Δ Final CCI")
            ax2.set_title("Final CCI Uplift vs A_base")
            ax2.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars2, cci_values):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (0.01 if val >= 0 else -0.01),
                    f"{val:+.3f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                )

            plt.tight_layout()
            plt.savefig(
                f"{self.out_dir}/figures/bar_uplifts.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def write_markdown_report(self, summary_json: dict):
        """Write comprehensive Markdown report"""
        self._log("Writing Markdown report")

        elapsed_time = time.time() - self.start_time

        markdown_content = f"""# Lifecycle 10-Minute Experiment Results

**Timestamp:** {self.timestamp}  
**Time Budget Used:** {elapsed_time:.1f}s / 600s ({elapsed_time/600*100:.1f}%)

## Experiments Conducted

We tested three intervention families across consciousness system evolution:

**A) Expansion Pulse:** Baseline vs +3% capacity every 12 epochs starting at epoch 12  
**B) Fairness × Coordination:** 2×2 grid testing coordination strength (0.40, 0.60) × goal inequality (0.20, 0.40)  
**C) Renewal Hygiene:** Baseline vs periodic renewal (every 10 epochs with 20% noise trimming + recalibration)

## Fast Takeaways

### A_pulse vs A_base
"""

        uplifts = summary_json["metrics"]["uplifts_vs_A_base"]

        if "A_pulse" in uplifts:
            a_uplift = uplifts["A_pulse"]
            markdown_content += f"""
- **Lifespan:** {a_uplift['Δlifespan']:+.1f} epochs
- **Final CCI:** {a_uplift['Δfinal_CCI']:+.3f}
- **Hazard Peak:** {a_uplift['Δhazard_peak']:+.4f}
"""

        if "B_best" in uplifts:
            b_uplift = uplifts["B_best"]
            markdown_content += f"""
### Best B Cell vs A_base
- **Best Condition:** {b_uplift['label']}
- **Lifespan:** {b_uplift['Δlifespan']:+.1f} epochs
- **Final CCI:** {b_uplift['Δfinal_CCI']:+.3f}
"""

        if "C_on" in uplifts:
            c_uplift = uplifts["C_on"]
            markdown_content += f"""
### C_on vs C_off
- **Lifespan:** {c_uplift['Δlifespan']:+.1f} epochs
- **Final CCI:** {c_uplift['Δfinal_CCI']:+.3f}
- **Hazard Peak:** {c_uplift['Δhazard_peak']:+.4f}
"""

        markdown_content += """
## Visualizations

![Survival Curves](../figures/survival_curves.png)

![CCI Trajectories](../figures/cci_trajectories_outcome_split.png)

![Uplift Analysis](../figures/bar_uplifts.png)

## Early Warning System

"""

        # Add divergence epoch info
        divergence_info = []
        for condition, metrics in summary_json["metrics"]["by_condition"].items():
            if metrics["divergence_epoch"]:
                divergence_info.append(
                    f"**{condition}:** Epoch {metrics['divergence_epoch']}"
                )

        if divergence_info:
            markdown_content += "**Divergence Epochs (survivors vs collapsed separate by ≥0.07 CCI for 5+ consecutive epochs):**\n"
            for info in divergence_info:
                markdown_content += f"- {info}\n"
        else:
            markdown_content += (
                "No clear divergence patterns detected in this time window.\n"
            )

        markdown_content += f"""
**Usage:** Monitor these epochs closely in production systems for early intervention opportunities.

## Decision Rule Outcome

**Recommended Strategy:** {summary_json["metrics"]["decision_recommendation"]}

## Next 20-30 Minute Plan

- **Extended Time Horizon:** Re-run winner with 150+ epochs to validate long-term stability
- **Parameter Sweep:** Fine-tune best intervention parameters (±20% around optimal values)
- **Interaction Effects:** Test combinations of winning interventions for synergistic effects

---

*Generated by Experiment Orchestrator at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        with open(f"{self.out_dir}/report/lifecycle_10min_results.md", "w") as f:
            f.write(markdown_content)

    def create_bundle(self):
        """Create ZIP bundle with checksums"""
        self._log("Creating bundle and checksums")

        # Generate SHA256 checksums
        checksums = {}

        for root, dirs, files in os.walk(self.out_dir):
            for file in files:
                if file.endswith(".zip") or file == "SHA256SUMS.txt":
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.out_dir)

                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    checksums[rel_path] = file_hash

        # Write checksums file
        with open(f"{self.out_dir}/SHA256SUMS.txt", "w") as f:
            for path, hash_val in sorted(checksums.items()):
                f.write(f"{hash_val}  {path}\n")

        # Create ZIP bundle
        bundle_name = f"lifecycle_10min_{self.timestamp}.zip"
        bundle_path = f"{self.out_dir}/bundle/{bundle_name}"

        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.out_dir):
                for file in files:
                    if file.endswith(".zip"):  # Don't include ZIP in ZIP
                        continue

                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, self.out_dir)
                    zipf.write(file_path, arc_path)

        return bundle_path

    def save_logs(self):
        """Save execution logs"""
        log_content = "\n".join(self.log_buffer)
        with open(f"{self.out_dir}/logs/runner.log", "w") as f:
            f.write(log_content)

    def print_final_checklist(self, bundle_path: str):
        """Print final completion checklist"""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("EXPERIMENT ORCHESTRATOR COMPLETE")
        print("=" * 60)

        print("\n✅ EXPORTS CREATED:")
        print(f"📊 CSVs: {self.out_dir}/data/")
        print("   - runs_summary.csv")
        print("   - conditions_lookup.csv")
        print("   - trajectories_long.csv")

        print(f"📋 JSON: {self.out_dir}/data/lifecycle_10min_summary.json")

        print(f"📈 PNGs: {self.out_dir}/figures/")
        print("   - survival_curves.png")
        print("   - cci_trajectories_outcome_split.png")
        print("   - hazard_over_time.png")
        print("   - bar_uplifts.png")

        print(f"📄 MD: {self.out_dir}/report/lifecycle_10min_results.md")
        print(f"📦 ZIP: {bundle_path}")

        print("\n⏱️  TIME REPORT:")
        print(f"   Total: {elapsed:.1f}s / 600s ({elapsed/600*100:.1f}%)")

        if elapsed > 600:
            print("   ⚠️  Exceeded 10-minute target")
        else:
            print("   ✅ Within 10-minute budget")

        # Read decision from summary if available
        try:
            with open(f"{self.out_dir}/data/lifecycle_10min_summary.json") as f:
                summary = json.load(f)
                decision = summary["metrics"]["decision_recommendation"]
                print(f"\n🧪 DECISION: {decision}")
                print(
                    "   Next 20-30min: Extended validation + parameter sweep + interaction testing"
                )
        except:
            pass

        print(f"\n📁 Full results: {self.out_dir}")
        print("=" * 60)


def main():
    """Main orchestrator execution"""
    try:
        orchestrator = ExperimentOrchestrator()

        # Run experiments
        runs_df, traj_df = orchestrator.run_all_experiments()

        # Analyze results
        summary_json, runs_summary = orchestrator.analyze_and_aggregate(
            runs_df, traj_df
        )

        # Save all exports
        orchestrator.save_data_exports(runs_summary, traj_df, summary_json)
        orchestrator.create_figures(runs_summary, traj_df, summary_json)
        orchestrator.write_markdown_report(summary_json)

        # Create bundle
        bundle_path = orchestrator.create_bundle()

        # Save logs and print completion
        orchestrator.save_logs()
        orchestrator.print_final_checklist(bundle_path)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
