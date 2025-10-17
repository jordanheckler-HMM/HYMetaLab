#!/usr/bin/env python3
"""
Long-Horizon Runner: Fairness × Coordination Focus with Phased Execution
Detailed Phase 1 (0-200), extended Phase 2 (to 1000+), with adaptive time budget management.
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

# Configuration (editable)
TIME_BUDGET_SEC = 600  # hard cap
EPOCHS_PHASE1 = 200  # detailed
EPOCHS_PHASE2 = 1000  # target; may extend to 2000 if time allows
LOG_EVERY_PHASE1 = 1
LOG_EVERY_PHASE2 = 10  # thin logging
AGENTS_PHASE1 = 64
AGENTS_PHASE2 = 48
SEEDS_GRID = 1  # 1 for sweeps
SEEDS_FINAL = 2  # 2 for finalists


class LongHorizonRunner:
    """Long-horizon consciousness evolution with phased execution"""

    def __init__(self):
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = f"outputs/long_horizon/{self.timestamp}"
        self.log_buffer = []

        # Create directory structure
        self._setup_directories()

        # Early-stop rule
        self.early_stop_rule = {
            "risk_min": 0.45,
            "survival_max": 0.40,
            "cci_floor": 0.45,
        }

        # Experiment plan
        self.plan = self._create_experiment_plan()

        # Results tracking
        self.phase1_results = {}
        self.phase2_results = {}
        self.finalist_results = {}

    def _setup_directories(self):
        """Create output directory structure"""
        dirs = ["data", "figures", "report", "logs", "bundle"]
        for d in dirs:
            os.makedirs(f"{self.out_dir}/{d}", exist_ok=True)

    def _create_experiment_plan(self) -> dict:
        """Create prioritized experiment plan"""
        plan = {
            # B: Fairness×Coordination fine grid (9 cells, priority order)
            "B_grid": [
                {
                    "family": "B",
                    "label": f"B_c{c:.2f}_g{g:.2f}",
                    "coordination_strength": c,
                    "goal_inequality": g,
                }
                for c in [0.50, 0.60, 0.70]
                for g in [0.15, 0.20, 0.25]
            ],
            # Controls (4 conditions)
            "controls": [
                {"family": "A", "label": "A_base"},
                {
                    "family": "A",
                    "label": "A_pulse",
                    "expansion": {
                        "type": "percent",
                        "pct": 0.03,
                        "every": 12,
                        "start": 12,
                    },
                },
                {"family": "C", "label": "C_off"},
                {
                    "family": "C",
                    "label": "C_on",
                    "hygiene": {
                        "every": 10,
                        "noise_trim_pct": 0.20,
                        "recalibrate": True,
                    },
                },
            ],
        }
        return plan

    def _log(self, msg: str):
        """Log message to buffer and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time
        log_msg = f"[{timestamp}] ({elapsed:.1f}s) {msg}"
        print(log_msg)
        self.log_buffer.append(log_msg)

    def _check_time_budget(self) -> tuple[bool, float]:
        """Check if we're within time budget"""
        elapsed = time.time() - self.start_time
        remaining = TIME_BUDGET_SEC - elapsed
        return remaining > 0, remaining

    def _simulate_consciousness_phase(
        self,
        condition: dict,
        seed: int,
        start_epoch: int,
        end_epoch: int,
        agents: int,
        log_every: int,
        initial_state: dict | None = None,
    ) -> tuple[dict, list[dict]]:
        """Simulate consciousness evolution for one phase"""
        np.random.seed(seed + start_epoch)  # Ensure reproducibility across phases

        # Initialize or continue from previous state
        if initial_state:
            current_cci = initial_state["cci"]
            current_survival = initial_state["survival"]
            current_risk = initial_state["risk"]
            base_age_factor = initial_state.get("age_factor", 1.0)
        else:
            current_cci = 0.65
            current_survival = 0.95
            current_risk = 0.15
            base_age_factor = 1.0

        # Apply condition modifications
        coordination_mult = condition.get("coordination_strength", 0.50) / 0.50
        inequality_mult = (0.30 - condition.get("goal_inequality", 0.30)) / 0.30 + 1.0

        # Initialize trajectory tracking
        trajectory = []
        cci_values = []
        survival_values = []
        risk_values = []

        # Phase simulation loop
        early_stopped = False
        collapse_epoch = None

        for epoch in range(start_epoch, end_epoch):
            # Apply aging and drift (cumulative over all epochs)
            age_factor = base_age_factor * (
                1.0 - (epoch * 0.002)
            )  # Slower aging for long horizons
            noise_factor = 1.0 + np.random.normal(
                0, 0.015
            )  # Reduced noise for stability

            # Apply expansion effects
            expansion_boost = 1.0
            if "expansion" in condition and epoch >= condition["expansion"]["start"]:
                if epoch % condition["expansion"]["every"] == 0:
                    expansion_boost = 1.0 + condition["expansion"]["pct"]

            # Apply hygiene effects
            hygiene_boost = 1.0
            if "hygiene" in condition and epoch > 0:
                if epoch % condition["hygiene"]["every"] == 0:
                    hygiene_boost = 1.12  # Moderate renewal effect

            # Calculate current metrics
            current_cci = (
                0.65
                * age_factor
                * coordination_mult
                * inequality_mult
                * expansion_boost
                * hygiene_boost
                * noise_factor
            )
            current_cci = max(0.1, min(1.0, current_cci))

            current_survival = (
                0.95
                * age_factor
                * coordination_mult
                * expansion_boost
                * hygiene_boost
                * (1.0 + np.random.normal(0, 0.01))
            )
            current_survival = max(0.1, min(1.0, current_survival))

            current_risk = (
                0.15
                * (2.0 - age_factor)
                / (coordination_mult * expansion_boost * hygiene_boost)
                * (1.0 + np.random.normal(0, 0.02))
            )
            current_risk = max(0.05, min(0.8, current_risk))

            # Store values
            cci_values.append(current_cci)
            survival_values.append(current_survival)
            risk_values.append(current_risk)

            # Record trajectory (respecting logging frequency)
            if epoch % log_every == 0 or epoch == end_epoch - 1:
                hazard = max(
                    0, -np.log(current_survival + 1e-6) if epoch > start_epoch else 0
                )
                trajectory.append(
                    {
                        "run_id": f"{condition['label']}_s{seed}",
                        "seed": seed,
                        "epoch": epoch,
                        "phase": 2 if epoch >= EPOCHS_PHASE1 else 1,
                        "CCI": current_cci,
                        "collapse_risk": current_risk,
                        "survival_rate": current_survival,
                        "hazard": hazard,
                    }
                )

            # Check early stop conditions
            if (
                current_risk >= self.early_stop_rule["risk_min"]
                and current_survival <= self.early_stop_rule["survival_max"]
                and current_cci < self.early_stop_rule["cci_floor"]
            ):
                early_stopped = True
                collapse_epoch = epoch
                break

        # Calculate phase summary metrics
        final_epoch = collapse_epoch if early_stopped else end_epoch - 1
        peak_cci = max(cci_values) if cci_values else 0
        final_cci = cci_values[-1] if cci_values else 0

        # Calculate hazard statistics
        hazard_values = [t["hazard"] for t in trajectory if t["hazard"] > 0]
        hazard_peak = max(hazard_values) if hazard_values else 0

        # Calculate CCI slope for last 50 epochs (promotion criterion)
        cci_slope_last50 = 0
        if len(cci_values) >= 50:
            recent_cci = cci_values[-50:]
            x = np.arange(len(recent_cci))
            cci_slope_last50 = np.polyfit(x, recent_cci, 1)[0]  # Linear slope

        phase_result = {
            "run_id": f"{condition['label']}_s{seed}",
            "family": condition["family"],
            "label": condition["label"],
            "seed": seed,
            "start_epoch": start_epoch,
            "end_epoch": final_epoch,
            "agents": agents,
            "early_stopped": early_stopped,
            "collapse_epoch": collapse_epoch,
            "peak_CCI": peak_cci,
            "final_CCI": final_cci,
            "hazard_peak": hazard_peak,
            "survival_final": survival_values[-1] if survival_values else 0,
            "risk_final": risk_values[-1] if risk_values else 0,
            "cci_slope_last50": cci_slope_last50,
            "phase_duration_sec": time.time() - self.start_time,
        }

        # Final state for potential continuation
        final_state = {
            "cci": current_cci,
            "survival": current_survival,
            "risk": current_risk,
            "age_factor": age_factor,
        }

        return phase_result, trajectory, final_state

    def run_phase1(self) -> dict:
        """Run Phase 1: Detailed logging for all conditions (0-200 epochs)"""
        self._log("=== PHASE 1: Detailed Analysis (0-200 epochs) ===")

        phase1_results = {}
        all_conditions = self.plan["B_grid"] + self.plan["controls"]

        for condition in all_conditions:
            within_budget, remaining = self._check_time_budget()
            if not within_budget:
                self._log(
                    f"Time budget exceeded, stopping Phase 1 at {condition['label']}"
                )
                break

            self._log(f"Phase 1: Running {condition['label']}")

            result, trajectory, final_state = self._simulate_consciousness_phase(
                condition, SEEDS_GRID, 0, EPOCHS_PHASE1, AGENTS_PHASE1, LOG_EVERY_PHASE1
            )

            phase1_results[condition["label"]] = {
                "result": result,
                "trajectory": trajectory,
                "final_state": final_state,
                "condition": condition,
            }

            elapsed = time.time() - self.start_time
            self._log(
                f"Completed Phase 1 for {condition['label']}: "
                f"final_CCI={result['final_CCI']:.3f}, "
                f"hazard_peak={result['hazard_peak']:.3f}, "
                f"slope={result['cci_slope_last50']:.5f}, "
                f"elapsed={elapsed:.1f}s"
            )

        self.phase1_results = phase1_results
        return phase1_results

    def _select_phase2_promotions(self, phase1_results: dict) -> list[str]:
        """Select conditions for Phase 2 promotion"""
        promotions = []

        # Adjusted promotion criteria: hazard_peak ≤ 0.40 and CCI_slope_last50 ≥ -0.004/epoch
        # Also promote top 4 performers regardless if no conditions meet strict criteria

        # First try strict criteria
        for label, data in phase1_results.items():
            result = data["result"]

            # Check promotion criteria
            hazard_ok = result["hazard_peak"] <= 0.40  # More lenient hazard threshold
            slope_ok = (
                result["cci_slope_last50"] >= -0.004
            )  # More lenient slope threshold

            if hazard_ok and slope_ok:
                promotions.append(label)
                self._log(
                    f"PROMOTED to Phase 2 (criteria): {label} "
                    f"(hazard={result['hazard_peak']:.3f}, "
                    f"slope={result['cci_slope_last50']:.5f})"
                )

        # If no promotions, select top 4 by final_CCI performance
        if not promotions:
            self._log(
                "No conditions met promotion criteria, selecting top 4 performers"
            )
            sorted_by_cci = sorted(
                phase1_results.items(),
                key=lambda x: x[1]["result"]["final_CCI"],
                reverse=True,
            )

            promotions = [item[0] for item in sorted_by_cci[:4]]

            for label in promotions:
                result = phase1_results[label]["result"]
                self._log(
                    f"PROMOTED to Phase 2 (top performer): {label} "
                    f"(final_CCI={result['final_CCI']:.3f}, "
                    f"hazard={result['hazard_peak']:.3f}, "
                    f"slope={result['cci_slope_last50']:.5f})"
                )

        # Log non-promoted conditions
        for label, data in phase1_results.items():
            if label not in promotions:
                result = data["result"]
                self._log(
                    f"Phase 1 only: {label} "
                    f"(hazard={result['hazard_peak']:.3f}, "
                    f"slope={result['cci_slope_last50']:.5f})"
                )

        return promotions

    def run_phase2(self, promotions: list[str]) -> dict:
        """Run Phase 2: Extended runs to 1000+ epochs"""
        self._log(f"=== PHASE 2: Extended Runs ({len(promotions)} conditions) ===")

        phase2_results = {}

        for label in promotions:
            within_budget, remaining = self._check_time_budget()
            if remaining < 0.2 * TIME_BUDGET_SEC:  # Stop promoting if <20% time left
                self._log(f"Stopping Phase 2 promotions at {label} (time budget)")
                break

            self._log(f"Phase 2: Extending {label} to {EPOCHS_PHASE2} epochs")

            phase1_data = self.phase1_results[label]

            result, trajectory, final_state = self._simulate_consciousness_phase(
                phase1_data["condition"],
                SEEDS_GRID,
                EPOCHS_PHASE1,
                EPOCHS_PHASE2,
                AGENTS_PHASE2,
                LOG_EVERY_PHASE2,
                initial_state=phase1_data["final_state"],
            )

            # Combine Phase 1 and Phase 2 trajectories
            combined_trajectory = phase1_data["trajectory"] + trajectory

            # Update result with combined metrics
            all_cci = [t["CCI"] for t in combined_trajectory]
            all_hazard = [t["hazard"] for t in combined_trajectory if t["hazard"] > 0]

            result.update(
                {
                    "total_epochs": result["end_epoch"],
                    "peak_CCI": max(all_cci),
                    "hazard_peak": max(all_hazard) if all_hazard else 0,
                }
            )

            phase2_results[label] = {
                "result": result,
                "trajectory": combined_trajectory,
                "final_state": final_state,
                "condition": phase1_data["condition"],
            }

            elapsed = time.time() - self.start_time
            self._log(
                f"Completed Phase 2 for {label}: "
                f"total_epochs={result['total_epochs']}, "
                f"final_CCI={result['final_CCI']:.3f}, "
                f"elapsed={elapsed:.1f}s"
            )

        self.phase2_results = phase2_results
        return phase2_results

    def _select_finalists(self, phase2_results: dict) -> list[str]:
        """Select top 2 finalists by final_CCI and low hazard_peak"""
        if len(phase2_results) < 2:
            return list(phase2_results.keys())

        # Sort by final_CCI (descending), then hazard_peak (ascending) as tiebreaker
        sorted_conditions = sorted(
            phase2_results.items(),
            key=lambda x: (-x[1]["result"]["final_CCI"], x[1]["result"]["hazard_peak"]),
        )

        finalists = [item[0] for item in sorted_conditions[:2]]

        self._log("=== FINALISTS SELECTED ===")
        for i, finalist in enumerate(finalists):
            result = phase2_results[finalist]["result"]
            self._log(
                f"#{i+1}: {finalist} "
                f"(final_CCI={result['final_CCI']:.3f}, "
                f"hazard={result['hazard_peak']:.3f})"
            )

        return finalists

    def run_finalists(self, finalists: list[str]) -> dict:
        """Run finalists with multiple seeds and potentially extended epochs"""
        self._log(
            f"=== FINALIST RUNS: {len(finalists)} conditions, {SEEDS_FINAL} seeds each ==="
        )

        finalist_results = {}

        for label in finalists:
            within_budget, remaining = self._check_time_budget()
            if remaining < 0.1 * TIME_BUDGET_SEC:  # Need at least 10% budget
                self._log(f"Skipping finalist {label} (insufficient time)")
                break

            condition = self.phase2_results[label]["condition"]
            finalist_runs = []

            # Determine target epochs based on remaining time
            target_epochs = EPOCHS_PHASE2
            if remaining > 0.4 * TIME_BUDGET_SEC:  # Plenty of time left
                target_epochs = 2000
                self._log(f"Extending {label} to 2000 epochs (sufficient time)")

            for seed in range(1, SEEDS_FINAL + 1):
                within_budget, remaining = self._check_time_budget()
                if not within_budget:
                    self._log(
                        f"Time budget exceeded during finalist {label} seed {seed}"
                    )
                    break

                self._log(f"Finalist: {label} seed {seed} → {target_epochs} epochs")

                # Run full trajectory
                result, trajectory, final_state = self._simulate_consciousness_phase(
                    condition, seed, 0, target_epochs, AGENTS_PHASE2, LOG_EVERY_PHASE2
                )

                finalist_runs.append(
                    {
                        "result": result,
                        "trajectory": trajectory,
                        "final_state": final_state,
                    }
                )

            finalist_results[label] = {"condition": condition, "runs": finalist_runs}

        self.finalist_results = finalist_results
        return finalist_results

    def _bootstrap_ci(
        self, values: list[float], n_bootstrap: int = 100
    ) -> tuple[float, list[float]]:
        """Compute bootstrap confidence interval (reduced samples for time)"""
        if len(values) == 0:
            return 0.0, [0.0, 0.0]

        # Further reduce if time-pressed
        if time.time() - self.start_time > 0.8 * TIME_BUDGET_SEC:
            n_bootstrap = 50

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        mean_val = np.mean(values)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        return mean_val, [ci_lower, ci_upper]

    def analyze_and_aggregate(self) -> dict:
        """Analyze results and create summary"""
        self._log("=== ANALYZING RESULTS ===")

        summary = {
            "timestamp": self.timestamp,
            "config": {
                "time_budget_sec": TIME_BUDGET_SEC,
                "epochs_phase1": EPOCHS_PHASE1,
                "epochs_phase2": EPOCHS_PHASE2,
                "agents_phase1": AGENTS_PHASE1,
                "agents_phase2": AGENTS_PHASE2,
                "seeds_grid": SEEDS_GRID,
                "seeds_final": SEEDS_FINAL,
            },
            "phase_summary": {
                "phase1_conditions": len(self.phase1_results),
                "phase2_promotions": len(self.phase2_results),
                "finalists": len(self.finalist_results),
            },
            "results": {},
        }

        # Aggregate finalist results
        for label, data in self.finalist_results.items():
            runs = data["runs"]

            final_cci_values = [r["result"]["final_CCI"] for r in runs]
            hazard_values = [r["result"]["hazard_peak"] for r in runs]
            lifespan_values = [r["result"]["end_epoch"] for r in runs]
            collapse_flags = [r["result"]["early_stopped"] for r in runs]

            final_cci_mean, final_cci_ci = self._bootstrap_ci(final_cci_values)
            hazard_mean, hazard_ci = self._bootstrap_ci(hazard_values)
            lifespan_mean, lifespan_ci = self._bootstrap_ci(lifespan_values)

            summary["results"][label] = {
                "n_runs": len(runs),
                "final_CCI_mean": round(final_cci_mean, 4),
                "final_CCI_ci95": [round(x, 4) for x in final_cci_ci],
                "hazard_peak_mean": round(hazard_mean, 4),
                "hazard_peak_ci95": [round(x, 4) for x in hazard_ci],
                "lifespan_mean": round(lifespan_mean, 1),
                "lifespan_ci95": [round(x, 1) for x in lifespan_ci],
                "collapse_rate": round(np.mean(collapse_flags), 3),
                "max_epochs_reached": max(lifespan_values),
                "coordination_strength": data["condition"].get("coordination_strength"),
                "goal_inequality": data["condition"].get("goal_inequality"),
            }

        # Decision rule: Winner selection
        if summary["results"]:
            # Find conditions that reached ≥1000 epochs
            long_horizon_conditions = {
                k: v
                for k, v in summary["results"].items()
                if v["max_epochs_reached"] >= 1000
            }

            if long_horizon_conditions:
                # Among long-horizon conditions, pick by final_CCI with hazard constraint
                safe_conditions = {
                    k: v
                    for k, v in long_horizon_conditions.items()
                    if v["hazard_peak_mean"] <= 0.20
                }

                if safe_conditions:
                    winner = max(
                        safe_conditions.keys(),
                        key=lambda x: safe_conditions[x]["final_CCI_mean"],
                    )
                    summary["decision"] = {
                        "winner": winner,
                        "criteria": "highest final_CCI with hazard_peak ≤ 0.20 at ≥1000 epochs",
                        "production_safe": True,
                    }
                else:
                    winner = max(
                        long_horizon_conditions.keys(),
                        key=lambda x: long_horizon_conditions[x]["final_CCI_mean"],
                    )
                    summary["decision"] = {
                        "winner": winner,
                        "criteria": "highest final_CCI at ≥1000 epochs (hazard risk noted)",
                        "production_safe": False,
                    }
            else:
                summary["decision"] = {
                    "winner": "None",
                    "criteria": "No conditions reached 1000+ epochs",
                    "production_safe": False,
                }

        return summary

    def save_data_exports(self, summary: dict):
        """Save all required data exports"""
        self._log("Saving data exports")

        # Collect all trajectories
        all_trajectories = []
        all_runs = []

        # Phase 1 only conditions
        for label, data in self.phase1_results.items():
            if label not in self.phase2_results and label not in self.finalist_results:
                all_trajectories.extend(data["trajectory"])
                all_runs.append(data["result"])

        # Phase 2 conditions
        for label, data in self.phase2_results.items():
            if label not in self.finalist_results:
                all_trajectories.extend(data["trajectory"])
                all_runs.append(data["result"])

        # Finalist conditions
        for label, data in self.finalist_results.items():
            for run_data in data["runs"]:
                all_trajectories.extend(run_data["trajectory"])
                all_runs.append(run_data["result"])

        # Save trajectories
        traj_df = pd.DataFrame(all_trajectories)
        traj_df.to_csv(f"{self.out_dir}/data/trajectories_long.csv", index=False)

        # Save runs summary
        runs_df = pd.DataFrame(all_runs)
        runs_df.to_csv(f"{self.out_dir}/data/runs_summary.csv", index=False)

        # Save conditions lookup
        all_conditions = self.plan["B_grid"] + self.plan["controls"]
        conditions_df = pd.DataFrame(
            [
                {"condition_label": c["label"], "parameters": json.dumps(c)}
                for c in all_conditions
            ]
        )
        conditions_df.to_csv(f"{self.out_dir}/data/conditions_lookup.csv", index=False)

        # Save summary JSON
        with open(f"{self.out_dir}/data/long_horizon_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return traj_df, runs_df

    def create_figures(
        self, traj_df: pd.DataFrame, runs_df: pd.DataFrame, summary: dict
    ):
        """Create all required figures"""
        self._log("Creating figures")

        # 1. CCI Long-term trajectories
        plt.figure(figsize=(15, 8))

        # Plot finalists with confidence intervals
        for label in self.finalist_results.keys():
            label_traj = traj_df[traj_df["run_id"].str.startswith(label)]
            if len(label_traj) > 0:
                epoch_stats = (
                    label_traj.groupby("epoch")["CCI"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )

                plt.plot(
                    epoch_stats["epoch"],
                    epoch_stats["mean"],
                    linewidth=2,
                    label=f"{label} (finalist)",
                    marker="o",
                    markersize=3,
                )

                # Add confidence bands if multiple runs
                if epoch_stats["count"].max() > 1:
                    stderr = epoch_stats["std"] / np.sqrt(epoch_stats["count"])
                    plt.fill_between(
                        epoch_stats["epoch"],
                        epoch_stats["mean"] - 1.96 * stderr,
                        epoch_stats["mean"] + 1.96 * stderr,
                        alpha=0.2,
                    )

        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("CCI", fontsize=12)
        plt.title(
            "Long-Horizon CCI Evolution (Finalists)", fontsize=14, fontweight="bold"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.axvline(
            x=EPOCHS_PHASE1, color="red", linestyle="--", alpha=0.7, label="Phase 1→2"
        )
        plt.axvline(
            x=1000, color="orange", linestyle="--", alpha=0.7, label="Target Horizon"
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/cci_long.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Hazard Long-term
        plt.figure(figsize=(15, 8))

        for label in self.finalist_results.keys():
            label_traj = traj_df[traj_df["run_id"].str.startswith(label)]
            if len(label_traj) > 0:
                hazard_stats = (
                    label_traj.groupby("epoch")["hazard"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                plt.plot(
                    hazard_stats["epoch"],
                    hazard_stats["mean"],
                    linewidth=2,
                    label=f"{label}",
                    marker="s",
                    markersize=2,
                )

        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Hazard Rate", fontsize=12)
        plt.title("Long-Horizon Hazard Evolution", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(
            y=0.20, color="red", linestyle="--", alpha=0.7, label="Safety Threshold"
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/hazard_long.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Phase Comparison (0-200 vs 200-1000+ overlays)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Phase 1 vs Phase 2 CCI comparison
        phase1_traj = traj_df[traj_df["phase"] == 1]
        phase2_traj = traj_df[traj_df["phase"] == 2]

        if len(phase1_traj) > 0:
            phase1_stats = phase1_traj.groupby("epoch")["CCI"].mean().reset_index()
            ax1.plot(
                phase1_stats["epoch"],
                phase1_stats["CCI"],
                "b-",
                linewidth=3,
                label="Phase 1 (0-200)",
                alpha=0.8,
            )

        if len(phase2_traj) > 0:
            phase2_stats = phase2_traj.groupby("epoch")["CCI"].mean().reset_index()
            # Normalize Phase 2 epochs to start from 0 for comparison
            normalized_epochs = phase2_stats["epoch"] - EPOCHS_PHASE1
            ax1.plot(
                normalized_epochs,
                phase2_stats["CCI"],
                "r-",
                linewidth=3,
                label="Phase 2 (200+)",
                alpha=0.8,
            )

        ax1.set_xlabel("Normalized Epochs")
        ax1.set_ylabel("Mean CCI")
        ax1.set_title("Phase Comparison: CCI Evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Final performance by condition type
        if summary["results"]:
            labels = list(summary["results"].keys())
            final_ccis = [
                summary["results"][label]["final_CCI_mean"] for label in labels
            ]
            colors = [
                "lightblue" if label.startswith("B_") else "lightcoral"
                for label in labels
            ]

            bars = ax2.bar(range(len(labels)), final_ccis, color=colors)
            ax2.set_xlabel("Conditions")
            ax2.set_ylabel("Final CCI")
            ax2.set_title("Final Performance Comparison")
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha="right")
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, final_ccis):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/phase_compare.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def write_markdown_report(self, summary: dict):
        """Write comprehensive Markdown report"""
        elapsed = time.time() - self.start_time

        markdown_content = f"""# Long-Horizon Experiment Results

**Timestamp:** {self.timestamp}  
**Time Budget Used:** {elapsed:.1f}s / {TIME_BUDGET_SEC}s ({elapsed/TIME_BUDGET_SEC*100:.1f}%)

## What We Ran

**Focus:** Fairness × Coordination fine grid with expansion/hygiene controls  
**Phasing:** Phase 1 (0-200 epochs detailed) → Phase 2 (to 1000+ epochs) → Finalists (multi-seed)

### Conditions Tested:
- **B Grid:** 9 combinations of coordination_strength × goal_inequality
- **Controls:** A_base, A_pulse (+3% expansion), C_off, C_on (hygiene)

### Progression:
- **Phase 1:** {summary['phase_summary']['phase1_conditions']} conditions tested (0-200 epochs)
- **Phase 2:** {summary['phase_summary']['phase2_promotions']} promoted (to 1000+ epochs)  
- **Finalists:** {summary['phase_summary']['finalists']} selected for multi-seed validation

## Key Wins
"""

        if "decision" in summary and summary["decision"]["winner"] != "None":
            winner = summary["decision"]["winner"]
            winner_data = summary["results"].get(winner, {})

            markdown_content += f"""
### 🏆 Winner: {winner}
- **Final CCI:** {winner_data.get('final_CCI_mean', 0):.4f} ± {winner_data.get('final_CCI_ci95', [0,0])[1] - winner_data.get('final_CCI_ci95', [0,0])[0]:.4f}
- **Hazard Peak:** {winner_data.get('hazard_peak_mean', 0):.4f}
- **Max Epochs:** {winner_data.get('max_epochs_reached', 0)}
- **Coordination:** {winner_data.get('coordination_strength', 'N/A')}
- **Inequality:** {winner_data.get('goal_inequality', 'N/A')}
- **Production Safe:** {'✅' if summary['decision']['production_safe'] else '⚠️'}
"""

        # Add results table
        if summary["results"]:
            markdown_content += """
### Finalist Performance Summary:

| Condition | Final CCI | Hazard Peak | Max Epochs | Coordination | Inequality |
|-----------|-----------|-------------|------------|--------------|------------|
"""
            for label, data in summary["results"].items():
                markdown_content += f"""| {label} | {data['final_CCI_mean']:.4f} | {data['hazard_peak_mean']:.4f} | {data['max_epochs_reached']} | {data.get('coordination_strength', 'N/A')} | {data.get('goal_inequality', 'N/A')} |
"""

        markdown_content += """
## Charts

![Long-Horizon CCI](../figures/cci_long.png)

![Hazard Evolution](../figures/hazard_long.png)

![Phase Comparison](../figures/phase_compare.png)

"""

        # Conditions that reached different horizons
        reached_1000 = []
        reached_2000 = []
        early_stops = []

        for label, data in summary["results"].items():
            max_epochs = data["max_epochs_reached"]
            if max_epochs >= 2000:
                reached_2000.append(label)
            elif max_epochs >= 1000:
                reached_1000.append(label)

            if data["collapse_rate"] > 0:
                early_stops.append(f"{label} ({data['collapse_rate']*100:.1f}%)")

        markdown_content += f"""
## Horizon Analysis

**Reached ≥1000 epochs:** {', '.join(reached_1000) if reached_1000 else 'None'}  
**Reached ≥2000 epochs:** {', '.join(reached_2000) if reached_2000 else 'None'}  
**Early stops:** {', '.join(early_stops) if early_stops else 'None'}

## Forward Plan

Based on these long-horizon results:
1. **Validate Winner:** Re-run {summary.get('decision', {}).get('winner', 'top performer')} with extended time budget (3000+ epochs)
2. **Parameter Refinement:** Fine-tune coordination/inequality around optimal values
3. **Robustness Testing:** Test winner against various perturbations and stress conditions

---

*Generated by Long-Horizon Runner at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        with open(f"{self.out_dir}/report/long_horizon_results.md", "w") as f:
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

        # Write checksums
        with open(f"{self.out_dir}/SHA256SUMS.txt", "w") as f:
            for path, hash_val in sorted(checksums.items()):
                f.write(f"{hash_val}  {path}\n")

        # Create bundle
        bundle_name = f"long_horizon_{self.timestamp}.zip"
        bundle_path = f"{self.out_dir}/bundle/{bundle_name}"

        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.out_dir):
                for file in files:
                    if file.endswith(".zip"):
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

    def print_final_checklist(self, bundle_path: str, summary: dict):
        """Print final completion checklist"""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("LONG-HORIZON RUNNER COMPLETE")
        print("=" * 70)

        print("\n✅ EXPORTS CREATED:")
        print(f"📊 CSV: {self.out_dir}/data/")
        print("   - trajectories_long.csv")
        print("   - runs_summary.csv")
        print("   - conditions_lookup.csv")

        print(f"📋 JSON: {self.out_dir}/data/long_horizon_summary.json")

        print(f"📈 PNG: {self.out_dir}/figures/")
        print("   - cci_long.png")
        print("   - hazard_long.png")
        print("   - phase_compare.png")

        print(f"📄 MD: {self.out_dir}/report/long_horizon_results.md")
        print(f"📦 ZIP: {bundle_path}")

        print("\n🔬 HORIZON ANALYSIS:")

        # Check what reached different epoch thresholds
        reached_1000 = []
        reached_2000 = []
        for label, data in summary["results"].items():
            max_epochs = data["max_epochs_reached"]
            if max_epochs >= 2000:
                reached_2000.append(label)
            elif max_epochs >= 1000:
                reached_1000.append(label)

        print(f"   ≥1000 epochs: {', '.join(reached_1000) if reached_1000 else 'None'}")
        print(f"   ≥2000 epochs: {', '.join(reached_2000) if reached_2000 else 'None'}")

        # Early stops
        early_stop_count = sum(
            1 for data in summary["results"].values() if data["collapse_rate"] > 0
        )
        print(f"   Early stops: {early_stop_count} conditions")

        print("\n⏱️  TIME BUDGET:")
        print(
            f"   Used: {elapsed:.1f}s / {TIME_BUDGET_SEC}s ({elapsed/TIME_BUDGET_SEC*100:.1f}%)"
        )

        # Time budget assessment
        if elapsed > TIME_BUDGET_SEC:
            print("   ⚠️  Exceeded budget")
        elif elapsed > 0.9 * TIME_BUDGET_SEC:
            print("   🟡 Near budget limit")
        else:
            print("   ✅ Within budget")

        # Downgrades applied
        phase1_count = len(self.phase1_results)
        phase2_count = len(self.phase2_results)
        finalist_count = len(self.finalist_results)

        print("\n📊 PHASE PROGRESSION:")
        print(f"   Phase 1: {phase1_count}/13 conditions")
        print(f"   Phase 2: {phase2_count} promotions")
        print(f"   Finalists: {finalist_count} conditions")

        if "decision" in summary:
            decision = summary["decision"]
            print(f"\n🏆 WINNER: {decision['winner']}")
            print(f"   Criteria: {decision['criteria']}")
            print(
                f"   Safe: {'Yes' if decision['production_safe'] else 'No (high hazard)'}"
            )

        print(f"\n📁 Full results: {self.out_dir}")
        print("=" * 70)


def main():
    """Main execution function"""
    try:
        runner = LongHorizonRunner()

        # Phase 1: Detailed analysis
        phase1_results = runner.run_phase1()

        # Select promotions for Phase 2
        promotions = runner._select_phase2_promotions(phase1_results)

        # Phase 2: Extended runs
        if promotions:
            phase2_results = runner.run_phase2(promotions)

            # Select and run finalists
            if phase2_results:
                finalists = runner._select_finalists(phase2_results)
                if finalists:
                    runner.run_finalists(finalists)

        # Analyze and export
        summary = runner.analyze_and_aggregate()
        traj_df, runs_df = runner.save_data_exports(summary)
        runner.create_figures(traj_df, runs_df, summary)
        runner.write_markdown_report(summary)

        # Create bundle and logs
        bundle_path = runner.create_bundle()
        runner.save_logs()

        # Final checklist
        runner.print_final_checklist(bundle_path, summary)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
