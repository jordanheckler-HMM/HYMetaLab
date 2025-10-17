#!/usr/bin/env python3
"""
Combined Long-Horizon Experiment Orchestrator
Focus: High-coordination, low-inequality band with expansion pulses and renewal hygiene
Evaluates solo and combined interventions with phased execution and adaptive time management.
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
TIME_BUDGET_SEC = 900  # hard cap (15 minutes)


class CombinedLongHorizonOrchestrator:
    """Orchestrates combined expansion + hygiene long-horizon experiments"""

    def __init__(self):
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = f"outputs/combined_long/{self.timestamp}"
        self.log_buffer = []

        # Create directory structure
        self._setup_directories()

        # Configuration
        self.config = {
            "epochs_phase1": 600,
            "epochs_phase2": 3000,
            "agents_phase1": 64,
            "agents_phase2": 48,
            "log_every_phase1": 1,
            "log_every_phase2": 10,
            "seeds_grid": 1,
            "seeds_final": 2,
            "collapse_rules": {
                "risk_min": 0.45,
                "survival_max": 0.40,
                "cci_floor": 0.45,
            },
            "fast_mode": True,
            "time_budget_sec": TIME_BUDGET_SEC,
        }

        # Create experiment plan
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

    def _create_experiment_plan(self) -> list[dict]:
        """Create comprehensive experiment plan"""
        # Fairness/coordination anchors
        f_anchors = [
            {"label": "F1", "coord": 0.60, "ineq": 0.15},
            {"label": "F2", "coord": 0.60, "ineq": 0.20},
        ]

        # Intervention strategies
        interventions = [
            {"label": "None"},
            {
                "label": "E_early",
                "expansion": {"mode": "early", "pct": 0.03, "every": 12, "start": 12},
            },
            {
                "label": "E_mid",
                "expansion": {"mode": "mid", "pct": 0.02, "every": 20, "start": 600},
            },
            {
                "label": "E_adapt",
                "expansion": {
                    "mode": "adaptive",
                    "pct": 0.015,
                    "hazard_thresh": 0.20,
                    "cooldown": 30,
                },
            },
            {
                "label": "H_periodic",
                "hygiene": {"every": 50, "noise_trim_pct": 0.10, "recalibrate": True},
            },
            {
                "label": "H_burst",
                "hygiene": {
                    "bursts": [600, 900, 1200],
                    "noise_trim_pct": 0.20,
                    "recalibrate": True,
                },
            },
            {
                "label": "EH_early_periodic",
                "expansion": {"mode": "early", "pct": 0.03, "every": 12, "start": 12},
                "hygiene": {"every": 50, "noise_trim_pct": 0.10, "recalibrate": True},
            },
            {
                "label": "EH_mid_burst",
                "expansion": {"mode": "mid", "pct": 0.02, "every": 20, "start": 600},
                "hygiene": {
                    "bursts": [600, 900, 1200],
                    "noise_trim_pct": 0.20,
                    "recalibrate": True,
                },
            },
            {
                "label": "EH_adapt_periodic",
                "expansion": {
                    "mode": "adaptive",
                    "pct": 0.015,
                    "hazard_thresh": 0.20,
                    "cooldown": 30,
                },
                "hygiene": {"every": 50, "noise_trim_pct": 0.10, "recalibrate": True},
            },
        ]

        # Generate full condition matrix
        plan = []
        for f in f_anchors:
            for iv in interventions:
                condition = {
                    "family": f["label"],
                    "intervention": iv["label"],
                    "label": f"{f['label']}_{iv['label']}",
                    "coordination_strength": f["coord"],
                    "goal_inequality": f["ineq"],
                }

                # Add intervention parameters
                if "expansion" in iv:
                    condition["expansion"] = iv["expansion"]
                if "hygiene" in iv:
                    condition["hygiene"] = iv["hygiene"]

                plan.append(condition)

        return plan

    def _log(self, msg: str):
        """Log message to buffer and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time
        log_msg = f"[{timestamp}] ({elapsed:.1f}s) {msg}"
        print(log_msg)
        self.log_buffer.append(log_msg)

    def _check_time_budget(self) -> tuple[bool, float]:
        """Check remaining time budget"""
        elapsed = time.time() - self.start_time
        remaining = TIME_BUDGET_SEC - elapsed
        return remaining > 0, remaining

    def _apply_adaptive_downgrades(self):
        """Apply adaptive downgrades if approaching time limit"""
        elapsed = time.time() - self.start_time
        remaining = TIME_BUDGET_SEC - elapsed

        if remaining < 0.3 * TIME_BUDGET_SEC:  # <30% budget remaining
            self._log("⚠️ Applying adaptive downgrades due to time pressure")

            # Reduce finalist seeds
            if self.config["seeds_final"] > 1:
                self.config["seeds_final"] = 1
                self._log("🔽 Finalist seeds: 2 → 1")

            # Increase Phase 2 logging interval
            if self.config["log_every_phase2"] == 10:
                self.config["log_every_phase2"] = 20
                self._log("🔽 Phase 2 logging: every 10 → every 20 epochs")

            # Reduce horizon
            if self.config["epochs_phase2"] > 2000:
                self.config["epochs_phase2"] = 2000
                self._log("🔽 Phase 2 horizon: 3000 → 2000 epochs")

            # Reduce agents
            if self.config["agents_phase2"] > 40:
                self.config["agents_phase2"] = 40
                self._log("🔽 Phase 2 agents: 48 → 40")

    def _simulate_consciousness_phase(
        self,
        condition: dict,
        seed: int,
        start_epoch: int,
        end_epoch: int,
        agents: int,
        log_every: int,
        initial_state: dict | None = None,
    ) -> tuple[dict, list[dict], dict]:
        """Simulate consciousness evolution with interventions"""
        np.random.seed(seed + start_epoch + hash(condition["label"]) % 1000)

        # Initialize or continue from previous state
        if initial_state:
            current_cci = initial_state["cci"]
            current_survival = initial_state["survival"]
            current_risk = initial_state["risk"]
            age_factor = initial_state.get("age_factor", 1.0)
            expansion_cooldown = initial_state.get("expansion_cooldown", 0)
        else:
            current_cci = 0.65
            current_survival = 0.95
            current_risk = 0.15
            age_factor = 1.0
            expansion_cooldown = 0

        # Apply condition base modifications
        coordination_mult = condition["coordination_strength"] / 0.50
        inequality_mult = (0.30 - condition["goal_inequality"]) / 0.30 + 1.0

        # Track interventions applied
        interventions_applied = []
        trajectory = []

        # Simulation loop
        early_stopped = False
        collapse_epoch = None

        for epoch in range(start_epoch, end_epoch):
            # Apply aging and drift
            age_factor *= 0.9995  # Very gradual aging for long horizons
            noise_factor = 1.0 + np.random.normal(0, 0.01)

            # Initialize intervention multipliers
            expansion_boost = 1.0
            hygiene_boost = 1.0
            intervention_flag = False

            # Apply expansion interventions
            if "expansion" in condition:
                exp_config = condition["expansion"]

                if exp_config["mode"] == "early":
                    if (
                        epoch >= exp_config["start"]
                        and epoch % exp_config["every"] == 0
                    ):
                        expansion_boost = 1.0 + exp_config["pct"]
                        intervention_flag = True
                        interventions_applied.append(f"expansion_pulse_{epoch}")

                elif exp_config["mode"] == "mid":
                    if (
                        epoch >= exp_config["start"]
                        and epoch % exp_config["every"] == 0
                    ):
                        expansion_boost = 1.0 + exp_config["pct"]
                        intervention_flag = True
                        interventions_applied.append(f"expansion_mid_{epoch}")

                elif exp_config["mode"] == "adaptive":
                    if expansion_cooldown > 0:
                        expansion_cooldown -= 1
                    elif current_risk > exp_config["hazard_thresh"]:
                        expansion_boost = 1.0 + exp_config["pct"]
                        expansion_cooldown = exp_config["cooldown"]
                        intervention_flag = True
                        interventions_applied.append(f"expansion_adaptive_{epoch}")

            # Apply hygiene interventions
            if "hygiene" in condition:
                hyg_config = condition["hygiene"]

                if "every" in hyg_config:  # Periodic hygiene
                    if epoch > 0 and epoch % hyg_config["every"] == 0:
                        hygiene_boost = 1.0 + (
                            hyg_config["noise_trim_pct"] * 0.8
                        )  # Moderate boost
                        intervention_flag = True
                        interventions_applied.append(f"hygiene_periodic_{epoch}")

                if "bursts" in hyg_config:  # Burst hygiene
                    if epoch in hyg_config["bursts"]:
                        hygiene_boost = 1.0 + (
                            hyg_config["noise_trim_pct"] * 1.2
                        )  # Stronger burst effect
                        intervention_flag = True
                        interventions_applied.append(f"hygiene_burst_{epoch}")

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
                * (1.0 + np.random.normal(0, 0.005))
            )
            current_survival = max(0.1, min(1.0, current_survival))

            current_risk = (
                0.15
                * (2.0 - age_factor)
                / (coordination_mult * expansion_boost * hygiene_boost)
                * (1.0 + np.random.normal(0, 0.01))
            )
            current_risk = max(0.05, min(0.8, current_risk))

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
                        "CCI": current_cci,
                        "collapse_risk": current_risk,
                        "survival_rate": current_survival,
                        "hazard": hazard,
                        "intervention_flag": intervention_flag,
                    }
                )

            # Check early stop conditions
            if (
                current_risk >= self.config["collapse_rules"]["risk_min"]
                and current_survival <= self.config["collapse_rules"]["survival_max"]
                and current_cci < self.config["collapse_rules"]["cci_floor"]
            ):
                early_stopped = True
                collapse_epoch = epoch
                break

        # Calculate summary metrics
        final_epoch = collapse_epoch if early_stopped else end_epoch - 1

        # Extract trajectory metrics
        traj_cci = [t["CCI"] for t in trajectory]
        traj_hazard = [t["hazard"] for t in trajectory if t["hazard"] > 0]

        peak_cci = max(traj_cci) if traj_cci else 0
        final_cci = traj_cci[-1] if traj_cci else 0
        hazard_peak = max(traj_hazard) if traj_hazard else 0

        # Calculate CCI slope for last 50 epochs (promotion criterion)
        cci_slope_last50 = 0
        if len(traj_cci) >= 50:
            recent_cci = traj_cci[-50:]
            x = np.arange(len(recent_cci))
            cci_slope_last50 = np.polyfit(x, recent_cci, 1)[0]

        # Calculate stability metrics for final window (last 100 epochs equivalent)
        stability_window_size = min(
            100, len(traj_cci) // 10
        )  # Adjust for logging frequency
        stability_hazard_last100 = (
            np.mean(traj_hazard[-stability_window_size:])
            if len(traj_hazard) >= stability_window_size
            else hazard_peak
        )
        stability_cci_last100 = (
            np.mean(traj_cci[-stability_window_size:])
            if len(traj_cci) >= stability_window_size
            else final_cci
        )

        phase_result = {
            "run_id": f"{condition['label']}_s{seed}",
            "family": condition["family"],
            "intervention": condition["intervention"],
            "label": condition["label"],
            "seed": seed,
            "epochs_cap": end_epoch,
            "agents": agents,
            "fast_mode": self.config["fast_mode"],
            "early_stopped": early_stopped,
            "coordination_strength": condition["coordination_strength"],
            "goal_inequality": condition["goal_inequality"],
            "expansion_mode": condition.get("expansion", {}).get("mode"),
            "expansion_params": json.dumps(condition.get("expansion", {})),
            "hygiene_mode": (
                "periodic"
                if condition.get("hygiene", {}).get("every")
                else ("burst" if condition.get("hygiene", {}).get("bursts") else None)
            ),
            "hygiene_params": json.dumps(condition.get("hygiene", {})),
            "lifespan_epochs": final_epoch,
            "collapse_flag": early_stopped,
            "peak_CCI": peak_cci,
            "final_CCI": final_cci,
            "hazard_peak": hazard_peak,
            "survival_final": current_survival,
            "risk_final": current_risk,
            "divergence_epoch": None,  # Will compute later
            "stability_hazard_last100": stability_hazard_last100,
            "stability_cci_last100": stability_cci_last100,
            "cci_slope_last50": cci_slope_last50,
            "interventions_applied": interventions_applied,
            "time_sec": time.time() - self.start_time,
        }

        # Final state for continuation
        final_state = {
            "cci": current_cci,
            "survival": current_survival,
            "risk": current_risk,
            "age_factor": age_factor,
            "expansion_cooldown": expansion_cooldown,
        }

        return phase_result, trajectory, final_state

    def run_phase1(self) -> dict:
        """Run Phase 1: Detailed analysis (0-600 epochs)"""
        self._log("=== PHASE 1: Detailed Analysis (0-600 epochs) ===")

        phase1_results = {}

        for i, condition in enumerate(self.plan):
            within_budget, remaining = self._check_time_budget()
            if not within_budget:
                self._log(f"Time budget exceeded, stopping Phase 1 at condition {i}")
                break

            if remaining < 0.4 * TIME_BUDGET_SEC:  # Apply downgrades if needed
                self._apply_adaptive_downgrades()

            self._log(f"Phase 1: Running {condition['label']}")

            result, trajectory, final_state = self._simulate_consciousness_phase(
                condition,
                self.config["seeds_grid"],
                0,
                self.config["epochs_phase1"],
                self.config["agents_phase1"],
                self.config["log_every_phase1"],
            )

            phase1_results[condition["label"]] = {
                "result": result,
                "trajectory": trajectory,
                "final_state": final_state,
                "condition": condition,
            }

            self._log(
                f"Completed Phase 1 for {condition['label']}: "
                f"final_CCI={result['final_CCI']:.3f}, "
                f"hazard_peak={result['hazard_peak']:.3f}, "
                f"slope={result['cci_slope_last50']:.5f}"
            )

        self.phase1_results = phase1_results
        return phase1_results

    def _select_phase2_promotions(self, phase1_results: dict) -> list[str]:
        """Select conditions for Phase 2 promotion"""
        promotions = []

        # Promotion criteria: hazard_peak ≤ 0.20 and CCI_slope_last50 ≥ -0.002/epoch
        for label, data in phase1_results.items():
            result = data["result"]

            hazard_ok = result["hazard_peak"] <= 0.20
            slope_ok = result["cci_slope_last50"] >= -0.002

            if hazard_ok and slope_ok:
                promotions.append(label)
                self._log(f"PROMOTED to Phase 2: {label}")

        # If no promotions, select top performers by final_CCI
        if not promotions:
            self._log("No conditions met strict criteria, selecting top 6 performers")
            sorted_by_cci = sorted(
                phase1_results.items(),
                key=lambda x: x[1]["result"]["final_CCI"],
                reverse=True,
            )
            promotions = [item[0] for item in sorted_by_cci[:6]]

        return promotions

    def run_phase2(self, promotions: list[str]) -> dict:
        """Run Phase 2: Extended runs to 3000 epochs"""
        self._log(
            f"=== PHASE 2: Extended Runs to {self.config['epochs_phase2']} epochs ==="
        )

        phase2_results = {}

        for label in promotions:
            within_budget, remaining = self._check_time_budget()
            if remaining < 0.2 * TIME_BUDGET_SEC:
                self._log(f"Stopping Phase 2 promotions at {label} (time budget)")
                break

            self._log(
                f"Phase 2: Extending {label} to {self.config['epochs_phase2']} epochs"
            )

            phase1_data = self.phase1_results[label]

            result, trajectory, final_state = self._simulate_consciousness_phase(
                phase1_data["condition"],
                self.config["seeds_grid"],
                self.config["epochs_phase1"],
                self.config["epochs_phase2"],
                self.config["agents_phase2"],
                self.config["log_every_phase2"],
                initial_state=phase1_data["final_state"],
            )

            # Combine Phase 1 and Phase 2 trajectories
            combined_trajectory = phase1_data["trajectory"] + trajectory

            # Update result with combined metrics
            all_cci = [t["CCI"] for t in combined_trajectory]
            all_hazard = [t["hazard"] for t in combined_trajectory if t["hazard"] > 0]

            result.update(
                {
                    "total_epochs": result["lifespan_epochs"],
                    "peak_CCI": max(all_cci) if all_cci else result["peak_CCI"],
                    "hazard_peak": (
                        max(all_hazard) if all_hazard else result["hazard_peak"]
                    ),
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
                f"epochs={result['total_epochs']}, "
                f"final_CCI={result['final_CCI']:.3f}, "
                f"stability_CCI={result['stability_cci_last100']:.3f}"
            )

        self.phase2_results = phase2_results
        return phase2_results

    def _select_finalists(self, phase2_results: dict) -> list[str]:
        """Select finalists for multi-seed validation"""
        if len(phase2_results) < 2:
            return list(phase2_results.keys())

        # Select top 2 by stability metrics
        sorted_conditions = sorted(
            phase2_results.items(),
            key=lambda x: (
                x[1]["result"]["stability_cci_last100"],
                -x[1]["result"]["stability_hazard_last100"],
            ),
            reverse=True,
        )

        finalists = [item[0] for item in sorted_conditions[:2]]

        self._log("=== FINALISTS SELECTED ===")
        for i, finalist in enumerate(finalists):
            result = phase2_results[finalist]["result"]
            self._log(
                f"#{i+1}: {finalist} "
                f"(stability_CCI={result['stability_cci_last100']:.3f}, "
                f"stability_hazard={result['stability_hazard_last100']:.3f})"
            )

        return finalists

    def run_finalists(self, finalists: list[str]) -> dict:
        """Run finalists with multiple seeds"""
        self._log(
            f"=== FINALIST RUNS: {len(finalists)} conditions, {self.config['seeds_final']} seeds each ==="
        )

        finalist_results = {}

        for label in finalists:
            within_budget, remaining = self._check_time_budget()
            if remaining < 0.1 * TIME_BUDGET_SEC:
                self._log(f"Skipping finalist {label} (insufficient time)")
                break

            condition = self.phase2_results[label]["condition"]
            finalist_runs = []

            for seed in range(1, self.config["seeds_final"] + 1):
                within_budget, remaining = self._check_time_budget()
                if not within_budget:
                    break

                self._log(f"Finalist: {label} seed {seed}")

                result, trajectory, final_state = self._simulate_consciousness_phase(
                    condition,
                    seed,
                    0,
                    self.config["epochs_phase2"],
                    self.config["agents_phase2"],
                    self.config["log_every_phase2"],
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
        """Compute bootstrap confidence interval"""
        if len(values) == 0:
            return 0.0, [0.0, 0.0]

        # Reduce if time-pressed
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
        """Analyze results and create comprehensive summary"""
        self._log("=== ANALYZING RESULTS ===")

        summary = {
            "timestamp": self.timestamp,
            "config": self.config,
            "phase_summary": {
                "phase1_conditions": len(self.phase1_results),
                "phase2_promotions": len(self.phase2_results),
                "finalists": len(self.finalist_results),
            },
            "results": {},
            "winners": {},
        }

        # Aggregate finalist results
        for label, data in self.finalist_results.items():
            runs = data["runs"]

            # Extract metrics
            final_cci_values = [r["result"]["final_CCI"] for r in runs]
            stability_cci_values = [r["result"]["stability_cci_last100"] for r in runs]
            stability_hazard_values = [
                r["result"]["stability_hazard_last100"] for r in runs
            ]
            hazard_values = [r["result"]["hazard_peak"] for r in runs]
            lifespan_values = [r["result"]["lifespan_epochs"] for r in runs]
            collapse_flags = [r["result"]["collapse_flag"] for r in runs]

            # Compute statistics
            final_cci_mean, final_cci_ci = self._bootstrap_ci(final_cci_values)
            stability_cci_mean, stability_cci_ci = self._bootstrap_ci(
                stability_cci_values
            )
            stability_hazard_mean, stability_hazard_ci = self._bootstrap_ci(
                stability_hazard_values
            )
            hazard_mean, hazard_ci = self._bootstrap_ci(hazard_values)
            lifespan_mean, lifespan_ci = self._bootstrap_ci(lifespan_values)

            summary["results"][label] = {
                "n_runs": len(runs),
                "final_CCI_mean": round(final_cci_mean, 4),
                "final_CCI_ci95": [round(x, 4) for x in final_cci_ci],
                "stability_CCI_mean": round(stability_cci_mean, 4),
                "stability_CCI_ci95": [round(x, 4) for x in stability_cci_ci],
                "stability_hazard_mean": round(stability_hazard_mean, 4),
                "stability_hazard_ci95": [round(x, 4) for x in stability_hazard_ci],
                "hazard_peak_mean": round(hazard_mean, 4),
                "lifespan_mean": round(lifespan_mean, 1),
                "collapse_rate": round(np.mean(collapse_flags), 3),
                "max_epochs_reached": max(lifespan_values),
                "family": data["condition"]["family"],
                "intervention": data["condition"]["intervention"],
                "coordination_strength": data["condition"]["coordination_strength"],
                "goal_inequality": data["condition"]["goal_inequality"],
            }

        # Winner selection using production-safe criteria
        production_safe_candidates = []

        for label, metrics in summary["results"].items():
            if (
                metrics["max_epochs_reached"] >= self.config["epochs_phase2"]
                and metrics["stability_hazard_mean"] <= 0.20
                and metrics["stability_CCI_mean"] >= 0.50
            ):
                production_safe_candidates.append((label, metrics))

        if production_safe_candidates:
            # Select best production-safe candidate
            winner_label, winner_metrics = min(
                production_safe_candidates,
                key=lambda x: (x[1]["hazard_peak_mean"], -x[1]["final_CCI_mean"]),
            )

            summary["winners"] = {
                "production_safe_winner": winner_label,
                "criteria": f"reached ≥{self.config['epochs_phase2']} epochs AND stability_hazard ≤ 0.20 AND stability_CCI ≥ 0.50",
                "is_production_safe": True,
                "limiting_metrics": None,
            }
        else:
            # No production-safe winner, select best by stability CCI
            if summary["results"]:
                best_label = max(
                    summary["results"].keys(),
                    key=lambda x: summary["results"][x]["stability_CCI_mean"],
                )
                best_metrics = summary["results"][best_label]

                # Identify limiting factors
                limiting_factors = []
                if best_metrics["max_epochs_reached"] < self.config["epochs_phase2"]:
                    limiting_factors.append(
                        f"lifespan ({best_metrics['max_epochs_reached']} < {self.config['epochs_phase2']})"
                    )
                if best_metrics["stability_hazard_mean"] > 0.20:
                    limiting_factors.append(
                        f"stability_hazard ({best_metrics['stability_hazard_mean']:.3f} > 0.20)"
                    )
                if best_metrics["stability_CCI_mean"] < 0.50:
                    limiting_factors.append(
                        f"stability_CCI ({best_metrics['stability_CCI_mean']:.3f} < 0.50)"
                    )

                summary["winners"] = {
                    "production_safe_winner": "None",
                    "best_available": best_label,
                    "criteria": "highest stability_CCI_mean (NOT PRODUCTION-SAFE)",
                    "is_production_safe": False,
                    "limiting_metrics": limiting_factors,
                }

        # Add recommendations
        summary["recommendations"] = self._generate_recommendations(summary)

        return summary

    def _generate_recommendations(self, summary: dict) -> list[str]:
        """Generate next step recommendations"""
        recommendations = []

        if not summary["winners"].get("is_production_safe", False):
            limiting = summary["winners"].get("limiting_metrics", [])

            if any("lifespan" in lim for lim in limiting):
                recommendations.append(
                    "Extend time horizon to 5000+ epochs to validate long-term stability"
                )

            if any("stability_hazard" in lim for lim in limiting):
                recommendations.append(
                    "Test stronger hygiene interventions (higher frequency or intensity)"
                )

            if any("stability_CCI" in lim for lim in limiting):
                recommendations.append(
                    "Combine multiple expansion modes (early + adaptive) for sustained performance"
                )
        else:
            recommendations.append(
                "Validate winner against stress scenarios (resource constraints, external shocks)"
            )
            recommendations.append(
                "Fine-tune intervention parameters (±10% around optimal values)"
            )

        if len(recommendations) < 3:
            recommendations.append(
                "Test hybrid fairness anchors (coordination 0.65, inequality 0.12)"
            )

        return recommendations[:3]

    def save_data_exports(self, summary: dict):
        """Save all required data exports"""
        self._log("Saving data exports")

        # Collect all trajectories and runs
        all_trajectories = []
        all_runs = []

        # Process all results
        for phase_results in [self.phase1_results, self.phase2_results]:
            for label, data in phase_results.items():
                if label not in self.finalist_results:  # Avoid duplicates
                    all_trajectories.extend(data.get("trajectory", []))
                    if "result" in data:
                        all_runs.append(data["result"])

        # Add finalist results
        for label, data in self.finalist_results.items():
            for run_data in data["runs"]:
                all_trajectories.extend(run_data["trajectory"])
                all_runs.append(run_data["result"])

        # Save trajectories
        if all_trajectories:
            traj_df = pd.DataFrame(all_trajectories)
            traj_df.to_csv(f"{self.out_dir}/data/trajectories_long.csv", index=False)

        # Save runs summary
        if all_runs:
            runs_df = pd.DataFrame(all_runs)
            runs_df.to_csv(f"{self.out_dir}/data/runs_summary.csv", index=False)

        # Save conditions lookup
        conditions_df = pd.DataFrame(
            [{"run_id": c["label"], "parameters": json.dumps(c)} for c in self.plan]
        )
        conditions_df.to_csv(f"{self.out_dir}/data/conditions_lookup.csv", index=False)

        # Save summary JSON
        with open(f"{self.out_dir}/data/combined_long_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return pd.DataFrame(all_trajectories) if all_trajectories else pd.DataFrame(), (
            pd.DataFrame(all_runs) if all_runs else pd.DataFrame()
        )

    def create_figures(
        self, traj_df: pd.DataFrame, runs_df: pd.DataFrame, summary: dict
    ):
        """Create all required figures"""
        self._log("Creating figures")

        if len(traj_df) == 0 or len(runs_df) == 0:
            self._log("⚠️ No data available for figure generation")
            return

        # 1. CCI Long-term evolution
        plt.figure(figsize=(16, 10))

        # Group by family and intervention
        families = ["F1", "F2"]
        colors = plt.cm.Set3(np.linspace(0, 1, 9))

        for i, family in enumerate(families):
            plt.subplot(2, 1, i + 1)

            family_conditions = [
                label
                for label in self.finalist_results.keys()
                if label.startswith(family)
            ]

            for j, label in enumerate(family_conditions):
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
                        label=label.replace(f"{family}_", ""),
                        color=colors[j],
                        marker="o",
                        markersize=2,
                    )

                    if epoch_stats["count"].max() > 1:
                        stderr = epoch_stats["std"] / np.sqrt(epoch_stats["count"])
                        plt.fill_between(
                            epoch_stats["epoch"],
                            epoch_stats["mean"] - stderr,
                            epoch_stats["mean"] + stderr,
                            alpha=0.2,
                            color=colors[j],
                        )

            plt.xlabel("Epochs")
            plt.ylabel("CCI")
            plt.title(
                f"{family} (coord=0.60, ineq={'0.15' if family=='F1' else '0.20'})"
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.axvline(
                x=600, color="red", linestyle="--", alpha=0.5, label="Phase 1→2"
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/cci_long.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Hazard evolution with intervention markers
        plt.figure(figsize=(15, 8))

        for label in self.finalist_results.keys():
            label_traj = traj_df[traj_df["run_id"].str.startswith(label)]
            if len(label_traj) > 0:
                plt.plot(
                    label_traj["epoch"],
                    label_traj["hazard"],
                    label=label,
                    linewidth=2,
                    alpha=0.7,
                )

                # Mark interventions
                interventions = label_traj[label_traj["intervention_flag"] == True]
                if len(interventions) > 0:
                    plt.scatter(
                        interventions["epoch"],
                        interventions["hazard"],
                        s=20,
                        alpha=0.8,
                        marker="x",
                    )

        plt.xlabel("Epochs")
        plt.ylabel("Hazard Rate")
        plt.title("Hazard Evolution with Intervention Markers")
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

        # 3. Stability window (last 500 epochs)
        plt.figure(figsize=(15, 6))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # CCI stability
        for label in self.finalist_results.keys():
            label_traj = traj_df[traj_df["run_id"].str.startswith(label)]
            if len(label_traj) > 0:
                late_traj = label_traj[
                    label_traj["epoch"] >= max(0, label_traj["epoch"].max() - 500)
                ]
                ax1.plot(late_traj["epoch"], late_traj["CCI"], label=label, linewidth=2)

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("CCI")
        ax1.set_title("CCI Stability (Last 500 Epochs)")
        ax1.axhline(
            y=0.50,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="Production Threshold",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Hazard stability
        for label in self.finalist_results.keys():
            label_traj = traj_df[traj_df["run_id"].str.startswith(label)]
            if len(label_traj) > 0:
                late_traj = label_traj[
                    label_traj["epoch"] >= max(0, label_traj["epoch"].max() - 500)
                ]
                ax2.plot(
                    late_traj["epoch"], late_traj["hazard"], label=label, linewidth=2
                )

        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Hazard Rate")
        ax2.set_title("Hazard Stability (Last 500 Epochs)")
        ax2.axhline(
            y=0.20, color="red", linestyle="--", alpha=0.7, label="Safety Threshold"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.out_dir}/figures/stability_window.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 4. Uplift grid (interventions vs no-intervention baselines)
        plt.figure(figsize=(12, 8))

        if summary["results"]:
            # Get baseline performance for each family
            baselines = {}
            for family in ["F1", "F2"]:
                baseline_key = f"{family}_None"
                if baseline_key in summary["results"]:
                    baselines[family] = summary["results"][baseline_key]

            # Calculate uplifts
            families = []
            interventions = []
            delta_final_cci = []
            delta_hazard_peak = []

            for label, metrics in summary["results"].items():
                if "_None" not in label:  # Skip baselines
                    family = metrics["family"]
                    if family in baselines:
                        baseline = baselines[family]

                        families.append(family)
                        interventions.append(metrics["intervention"])
                        delta_final_cci.append(
                            metrics["final_CCI_mean"] - baseline["final_CCI_mean"]
                        )
                        delta_hazard_peak.append(
                            metrics["hazard_peak_mean"] - baseline["hazard_peak_mean"]
                        )

            if delta_final_cci:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # Final CCI uplifts
                x_pos = np.arange(len(interventions))
                colors = ["skyblue" if f == "F1" else "lightcoral" for f in families]

                bars1 = ax1.bar(x_pos, delta_final_cci, color=colors)
                ax1.set_ylabel("Δ Final CCI vs Baseline")
                ax1.set_title("Final CCI Uplift by Intervention")
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(interventions, rotation=45, ha="right")
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)

                # Add value labels
                for bar, val, fam in zip(bars1, delta_final_cci, families):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + (0.01 if val >= 0 else -0.01),
                        f"{val:+.3f}\n({fam})",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                    )

                # Hazard peak uplifts
                bars2 = ax2.bar(x_pos, delta_hazard_peak, color=colors)
                ax2.set_ylabel("Δ Hazard Peak vs Baseline")
                ax2.set_title("Hazard Peak Change by Intervention")
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(interventions, rotation=45, ha="right")
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

                # Add value labels
                for bar, val, fam in zip(bars2, delta_hazard_peak, families):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + (0.01 if val >= 0 else -0.01),
                        f"{val:+.3f}\n({fam})",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                    )

                plt.tight_layout()

        plt.savefig(
            f"{self.out_dir}/figures/uplift_grid.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def write_markdown_report(self, summary: dict):
        """Write comprehensive Markdown report"""
        elapsed = time.time() - self.start_time

        markdown_content = f"""# Combined Long-Horizon Experiment Results

**Timestamp:** {self.timestamp}  
**Time Budget Used:** {elapsed:.1f}s / {TIME_BUDGET_SEC}s ({elapsed/TIME_BUDGET_SEC*100:.1f}%)

## What We Ran

**Focus:** High-coordination, low-inequality band (F1: 0.60/0.15, F2: 0.60/0.20)  
**Interventions:** Expansion pulses (early/mid/adaptive) + Hygiene (periodic/burst) + Combined strategies  
**Phasing:** Phase 1 (0-600 epochs) → Phase 2 (to {self.config['epochs_phase2']} epochs) → Finalists (multi-seed)

### Quick Matrix:
| Family | No-intervention | E-early | E-mid | E-adaptive | H-periodic | H-burst | EH-early+periodic | EH-mid+burst | EH-adaptive+periodic |
|--------|----------------|---------|--------|------------|------------|---------|-------------------|--------------|----------------------|
| F1 (0.60/0.15) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| F2 (0.60/0.20) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Progression:
- **Phase 1:** {summary['phase_summary']['phase1_conditions']} conditions (0-600 epochs)
- **Phase 2:** {summary['phase_summary']['phase2_promotions']} promoted (to {self.config['epochs_phase2']} epochs)
- **Finalists:** {summary['phase_summary']['finalists']} selected for multi-seed validation

## Fast Takeaways
"""

        # Analyze intervention effectiveness
        if summary["results"]:
            # Find best solo interventions by type
            expansion_results = {
                k: v
                for k, v in summary["results"].items()
                if v["intervention"].startswith("E_")
            }
            hygiene_results = {
                k: v
                for k, v in summary["results"].items()
                if v["intervention"].startswith("H_")
            }
            combined_results = {
                k: v
                for k, v in summary["results"].items()
                if v["intervention"].startswith("EH_")
            }

            if expansion_results:
                best_expansion = max(
                    expansion_results.keys(),
                    key=lambda x: expansion_results[x]["stability_CCI_mean"],
                )
                markdown_content += f"\n• **Best Solo Expansion:** {best_expansion} (stability_CCI: {expansion_results[best_expansion]['stability_CCI_mean']:.3f})"

            if hygiene_results:
                best_hygiene = max(
                    hygiene_results.keys(),
                    key=lambda x: hygiene_results[x]["stability_CCI_mean"],
                )
                markdown_content += f"\n• **Best Solo Hygiene:** {best_hygiene} (stability_CCI: {hygiene_results[best_hygiene]['stability_CCI_mean']:.3f})"

            if combined_results:
                best_combined = max(
                    combined_results.keys(),
                    key=lambda x: combined_results[x]["stability_CCI_mean"],
                )
                markdown_content += f"\n• **Best Combined:** {best_combined} (stability_CCI: {combined_results[best_combined]['stability_CCI_mean']:.3f})"

            # Check for failure modes
            collapsed_conditions = [
                k for k, v in summary["results"].items() if v["collapse_rate"] > 0.5
            ]
            if collapsed_conditions:
                markdown_content += f"\n• **High Collapse Rate:** {', '.join(collapsed_conditions)} (>50% failure rate)"

        markdown_content += """

## Charts

![Long-Horizon CCI Evolution](../figures/cci_long.png)

![Hazard Evolution with Interventions](../figures/hazard_long.png)

![Stability Window Analysis](../figures/stability_window.png)

![Intervention Uplift Analysis](../figures/uplift_grid.png)

## Winner Analysis
"""

        winners = summary.get("winners", {})

        if winners.get("is_production_safe", False):
            winner = winners["production_safe_winner"]
            winner_data = summary["results"].get(winner, {})

            markdown_content += f"""
### 🏆 Production-Safe Winner: {winner}
- **Stability CCI:** {winner_data.get('stability_CCI_mean', 0):.4f} (≥0.50 required)
- **Stability Hazard:** {winner_data.get('stability_hazard_mean', 0):.4f} (≤0.20 required)  
- **Max Epochs:** {winner_data.get('max_epochs_reached', 0)} (≥{self.config['epochs_phase2']} required)
- **Family:** {winner_data.get('family', 'N/A')}
- **Intervention:** {winner_data.get('intervention', 'N/A')}
- **Collapse Rate:** {winner_data.get('collapse_rate', 0)*100:.1f}%

✅ **PRODUCTION READY** - Meets all safety and performance criteria.
"""
        else:
            best_available = winners.get("best_available", "None")
            if best_available != "None":
                best_data = summary["results"].get(best_available, {})
                limiting_metrics = winners.get("limiting_metrics", [])

                markdown_content += f"""
### ⚠️ Best Available: {best_available} (NOT PRODUCTION-SAFE)
- **Stability CCI:** {best_data.get('stability_CCI_mean', 0):.4f}
- **Stability Hazard:** {best_data.get('stability_hazard_mean', 0):.4f}
- **Max Epochs:** {best_data.get('max_epochs_reached', 0)}
- **Family:** {best_data.get('family', 'N/A')}
- **Intervention:** {best_data.get('intervention', 'N/A')}

❌ **NOT PRODUCTION-SAFE** - Limiting factors:
"""
                for factor in limiting_metrics:
                    markdown_content += f"   - {factor}\n"
            else:
                markdown_content += "\n❌ **NO VIABLE CANDIDATES** - All conditions failed production criteria.\n"

        # Add performance summary table
        if summary["results"]:
            markdown_content += """
### Performance Summary:

| Condition | Stability CCI | Stability Hazard | Max Epochs | Collapse Rate | Family | Intervention |
|-----------|---------------|------------------|------------|---------------|---------|-------------|
"""
            for label, data in summary["results"].items():
                markdown_content += f"""| {label} | {data['stability_CCI_mean']:.3f} | {data['stability_hazard_mean']:.3f} | {data['max_epochs_reached']} | {data['collapse_rate']*100:.1f}% | {data['family']} | {data['intervention']} |
"""

        # Add recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            markdown_content += """
## Next 20-30 Minute Plan

"""
            for i, rec in enumerate(recommendations, 1):
                markdown_content += f"{i}. **{rec}**\n"

        markdown_content += f"""

---

*Generated by Combined Long-Horizon Orchestrator at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        with open(f"{self.out_dir}/report/combined_long_results.md", "w") as f:
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
        bundle_name = f"combined_long_{self.timestamp}.zip"
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

        print("\n" + "=" * 80)
        print("COMBINED LONG-HORIZON ORCHESTRATOR COMPLETE")
        print("=" * 80)

        print("\n✅ EXPORTS CREATED:")
        print(f"📊 CSV: {self.out_dir}/data/")
        print("   - trajectories_long.csv")
        print("   - runs_summary.csv")
        print("   - conditions_lookup.csv")

        print(f"📋 JSON: {self.out_dir}/data/combined_long_summary.json")

        print(f"📈 PNG: {self.out_dir}/figures/")
        print("   - cci_long.png")
        print("   - hazard_long.png")
        print("   - stability_window.png")
        print("   - uplift_grid.png")

        print(f"📄 MD: {self.out_dir}/report/combined_long_results.md")
        print(f"📦 ZIP: {bundle_path}")

        print("\n⏱️ TIME REPORT:")
        print(
            f"   Used: {elapsed:.1f}s / {TIME_BUDGET_SEC}s ({elapsed/TIME_BUDGET_SEC*100:.1f}%)"
        )

        # Report any adaptive downgrades
        original_config = {
            "seeds_final": 2,
            "log_every_phase2": 10,
            "epochs_phase2": 3000,
            "agents_phase2": 48,
        }

        downgrades = []
        if self.config["seeds_final"] < original_config["seeds_final"]:
            downgrades.append(
                f"finalist seeds: {original_config['seeds_final']} → {self.config['seeds_final']}"
            )
        if self.config["log_every_phase2"] > original_config["log_every_phase2"]:
            downgrades.append(
                f"Phase2 logging: every {original_config['log_every_phase2']} → every {self.config['log_every_phase2']} epochs"
            )
        if self.config["epochs_phase2"] < original_config["epochs_phase2"]:
            downgrades.append(
                f"horizon: {original_config['epochs_phase2']} → {self.config['epochs_phase2']} epochs"
            )
        if self.config["agents_phase2"] < original_config["agents_phase2"]:
            downgrades.append(
                f"agents: {original_config['agents_phase2']} → {self.config['agents_phase2']}"
            )

        if downgrades:
            print(f"   Adaptive downgrades: {', '.join(downgrades)}")
        else:
            print("   No downgrades needed")

        # Winner analysis
        winners = summary.get("winners", {})
        print("\n🏆 WINNER ANALYSIS:")

        if winners.get("is_production_safe", False):
            winner = winners["production_safe_winner"]
            print(f"   Production-Safe Winner: {winner}")
            print("   ✅ PRODUCTION READY")
        else:
            best = winners.get("best_available", "None")
            print(f"   Best Available: {best}")
            print("   ❌ NOT PRODUCTION-SAFE")

            limiting = winners.get("limiting_metrics", [])
            if limiting:
                print(f"   Limiting factors: {', '.join(limiting)}")

        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print("\n🔧 RECOMMENDED NEXT KNOBS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        print(f"\n📁 Full results: {self.out_dir}")
        print("=" * 80)


def main():
    """Main execution function"""
    try:
        orchestrator = CombinedLongHorizonOrchestrator()

        # Phase 1: Detailed analysis
        phase1_results = orchestrator.run_phase1()

        # Select promotions for Phase 2
        promotions = orchestrator._select_phase2_promotions(phase1_results)

        # Phase 2: Extended runs
        if promotions:
            phase2_results = orchestrator.run_phase2(promotions)

            # Select and run finalists
            if phase2_results:
                finalists = orchestrator._select_finalists(phase2_results)
                if finalists:
                    orchestrator.run_finalists(finalists)

        # Analyze and export
        summary = orchestrator.analyze_and_aggregate()
        traj_df, runs_df = orchestrator.save_data_exports(summary)
        orchestrator.create_figures(traj_df, runs_df, summary)
        orchestrator.write_markdown_report(summary)

        # Create bundle and logs
        bundle_path = orchestrator.create_bundle()
        orchestrator.save_logs()

        # Final checklist
        orchestrator.print_final_checklist(bundle_path, summary)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
