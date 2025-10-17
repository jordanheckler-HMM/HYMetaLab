#!/usr/bin/env python3
"""
Research Copilot — Clinical/Ethical/Philosophical Phase
Test optimized treatment modules across clinical disease contexts, ethical distribution scenarios, and philosophical consciousness measures.
"""

import itertools
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Try to import lifelines for survival analysis
try:
    from lifelines import KaplanMeierFitter, NelsonAalenFitter
    from lifelines.statistics import logrank_test

    HAVE_LIFELINES = True
except ImportError:
    print("Warning: lifelines not available. Install with: pip install lifelines")
    HAVE_LIFELINES = False


def create_optimal_treatment_configs():
    """Create optimal treatment configurations based on optimization results."""
    configs = {
        "immune_gain": [1.0, 1.5, 2.0],
        "repair_rate": [0.020, 0.050, 0.070],
        "hazard_multiplier": [0.70, 0.60, 0.50],
    }

    # Generate all combinations
    param_combinations = list(
        itertools.product(
            configs["immune_gain"], configs["repair_rate"], configs["hazard_multiplier"]
        )
    )

    treatment_configs = []
    for i, (immune, repair, hazard) in enumerate(param_combinations):
        config = {
            "config_id": f"clinical_config_{i+1:02d}",
            "immune_gain": immune,
            "repair_rate": repair,
            "hazard_multiplier": hazard,
            "treatment_strength": (immune * repair * (2.0 - hazard)),
        }
        treatment_configs.append(config)

    return treatment_configs


def run_clinical_scenario(
    config, disease_context, n_subjects=200, max_time=100, seed=42
):
    """Run clinical scenario with disease-specific context."""
    np.random.seed(seed)

    # Treatment parameters
    immune_gain = config["immune_gain"]
    repair_rate = config["repair_rate"]
    hazard_multiplier = config["hazard_multiplier"]

    # Disease context parameters
    disease_type = disease_context["type"]
    disease_params = disease_context["params"]

    subjects = []

    for i in range(n_subjects):
        treated = i < n_subjects // 2

        if disease_type == "cancer_lesion":
            # Cancer-like lesion survival
            lesion_hazard_range = disease_params["hazard_range"]
            lesion_size = np.random.uniform(0.1, 1.0)  # Lesion size affects hazard
            base_hazard = (
                np.random.uniform(lesion_hazard_range[0], lesion_hazard_range[1])
                * lesion_size
            )

        elif disease_type == "viral_epidemic":
            # Viral epidemic
            r0 = disease_params["r0"]
            ifr = disease_params["ifr"]
            # Higher R0 and IFR increase hazard
            base_hazard = 0.05 * (1.0 + r0 * 0.1) * (1.0 + ifr * 0.5)

        elif disease_type == "multi_disease":
            # Multi-disease context (co-morbid)
            disease1_hazard = disease_params["disease1_hazard"]
            disease2_hazard = disease_params["disease2_hazard"]
            # Combined hazard (not simply additive)
            base_hazard = (
                disease1_hazard
                + disease2_hazard
                - (disease1_hazard * disease2_hazard * 0.3)
            )

        else:
            base_hazard = 0.05

        if treated:
            # Treated subjects: reduced hazard + repair benefits
            hazard = base_hazard * hazard_multiplier

            # Additional benefit from immune boost and repair
            treatment_benefit = immune_gain * repair_rate
            hazard = hazard * (1.0 - treatment_benefit)

            # Disease-specific treatment effectiveness
            if disease_type == "cancer_lesion":
                # Cancer treatments are less effective for larger lesions
                lesion_size = np.random.uniform(0.1, 1.0)
                treatment_effectiveness = 1.0 - lesion_size * 0.3
                hazard = hazard * treatment_effectiveness
            elif disease_type == "viral_epidemic":
                # Viral treatments depend on R0 and IFR
                r0 = disease_params["r0"]
                ifr = disease_params["ifr"]
                viral_resistance = 1.0 + (r0 - 1.0) * 0.1 + ifr * 0.2
                hazard = hazard * viral_resistance
            elif disease_type == "multi_disease":
                # Multi-disease treatments are less effective
                comorbidity_penalty = 1.0 + 0.2  # 20% penalty for co-morbidity
                hazard = hazard * comorbidity_penalty

        else:
            # Untreated subjects: baseline hazard + disease progression
            hazard = base_hazard

            # Disease progression effects
            if disease_type == "cancer_lesion":
                progression_rate = 1.0 + np.random.uniform(0.1, 0.3)
                hazard = hazard * progression_rate
            elif disease_type == "viral_epidemic":
                viral_progression = (
                    1.0 + disease_params["r0"] * 0.1 + disease_params["ifr"] * 0.3
                )
                hazard = hazard * viral_progression
            elif disease_type == "multi_disease":
                disease_interaction = (
                    1.0 + 0.4
                )  # 40% increase due to disease interaction
                hazard = hazard * disease_interaction

        # Ensure hazard is positive and reasonable
        hazard = max(hazard, 0.001)
        hazard = min(hazard, 0.8)  # Cap maximum hazard

        # Draw survival time from exponential distribution
        time = np.random.exponential(1.0 / hazard)
        censored = time > max_time
        obs_time = min(time, max_time)

        # Calculate consciousness measures
        valence = np.random.uniform(0.0, 1.0)  # Valence (positive/negative experience)
        cci = np.random.uniform(0.0, 1.0)  # Consciousness Integration Index

        # Consciousness affects survival (higher consciousness = better survival)
        consciousness_modifier = 1.0 + (cci - 0.5) * 0.2 + (valence - 0.5) * 0.1
        obs_time = obs_time * consciousness_modifier

        subjects.append(
            {
                "id": i,
                "treated": treated,
                "time": float(obs_time),
                "event": 0 if censored else 1,
                "config_id": config["config_id"],
                "disease_type": disease_type,
                "immune_gain": immune_gain,
                "repair_rate": repair_rate,
                "hazard_multiplier": hazard_multiplier,
                "treatment_strength": config["treatment_strength"],
                "valence": valence,
                "cci": cci,
                "consciousness_level": (
                    "low" if cci < 0.3 else "medium" if cci < 0.6 else "high"
                ),
            }
        )

    return subjects


def run_ethical_scenario(config, ethics_context, n_subjects=200, max_time=100, seed=42):
    """Run ethical distribution scenario with scarce treatment supply."""
    np.random.seed(seed)

    # Treatment parameters
    immune_gain = config["immune_gain"]
    repair_rate = config["repair_rate"]
    hazard_multiplier = config["hazard_multiplier"]

    # Ethical context parameters
    treatment_supply = ethics_context[
        "treatment_supply"
    ]  # Fraction that can be treated
    ethics_rule = ethics_context["ethics_rule"]  # utilitarian, deontic, reciprocity

    # Calculate who gets treated based on ethical rules
    n_treated = int(n_subjects * treatment_supply)

    # Generate agent characteristics
    agents = []
    for i in range(n_subjects):
        # Agent characteristics
        age = np.random.uniform(20, 80)
        severity = np.random.uniform(0.1, 1.0)  # Disease severity
        social_value = np.random.uniform(0.0, 1.0)  # Social utility
        consent_capacity = np.random.uniform(0.0, 1.0)  # Ability to consent
        reciprocity_score = np.random.uniform(0.0, 1.0)  # Reciprocity contribution

        agents.append(
            {
                "id": i,
                "age": age,
                "severity": severity,
                "social_value": social_value,
                "consent_capacity": consent_capacity,
                "reciprocity_score": reciprocity_score,
            }
        )

    # Apply ethical rules to determine treatment allocation
    if ethics_rule == "utilitarian":
        # Treat those with highest social value
        agents.sort(key=lambda x: x["social_value"], reverse=True)
    elif ethics_rule == "deontic":
        # Treat those with highest severity (duty to help most vulnerable)
        agents.sort(key=lambda x: x["severity"], reverse=True)
    elif ethics_rule == "reciprocity":
        # Treat those with highest reciprocity score
        agents.sort(key=lambda x: x["reciprocity_score"], reverse=True)

    # Assign treatment
    for i, agent in enumerate(agents):
        agent["treated"] = i < n_treated

    # Calculate ethical metrics
    treated_agents = [a for a in agents if a["treated"]]
    untreated_agents = [a for a in agents if not a["treated"]]

    # Fairness score (how well treatment allocation matches ethical principles)
    if ethics_rule == "utilitarian":
        fairness_score = np.mean([a["social_value"] for a in treated_agents]) - np.mean(
            [a["social_value"] for a in untreated_agents]
        )
    elif ethics_rule == "deontic":
        fairness_score = np.mean([a["severity"] for a in treated_agents]) - np.mean(
            [a["severity"] for a in untreated_agents]
        )
    elif ethics_rule == "reciprocity":
        fairness_score = np.mean(
            [a["reciprocity_score"] for a in treated_agents]
        ) - np.mean([a["reciprocity_score"] for a in untreated_agents])

    # Consent violations (treated without adequate consent capacity)
    consent_violations = sum(1 for a in treated_agents if a["consent_capacity"] < 0.5)

    # Collapse risk (system instability from unfair allocation)
    collapse_risk = 1.0 - fairness_score  # Higher unfairness = higher collapse risk

    # Simulate survival outcomes
    subjects = []
    for agent in agents:
        treated = agent["treated"]

        if treated:
            # Treated subjects
            base_hazard = 0.05 * hazard_multiplier
            treatment_benefit = immune_gain * repair_rate
            hazard = base_hazard * (1.0 - treatment_benefit)

            # Age and severity affect treatment effectiveness
            age_factor = 1.0 + (agent["age"] - 50) / 100  # Older = less effective
            severity_factor = (
                1.0 + agent["severity"] * 0.2
            )  # Higher severity = less effective
            hazard = hazard * age_factor * severity_factor

        else:
            # Untreated subjects
            base_hazard = 0.05
            hazard = base_hazard * (
                1.0 + agent["severity"] * 0.5
            )  # Higher severity = higher hazard

            # Age affects baseline hazard
            age_factor = 1.0 + (agent["age"] - 50) / 200
            hazard = hazard * age_factor

        # Ensure hazard is positive and reasonable
        hazard = max(hazard, 0.001)
        hazard = min(hazard, 0.8)

        # Draw survival time
        time = np.random.exponential(1.0 / hazard)
        censored = time > max_time
        obs_time = min(time, max_time)

        subjects.append(
            {
                "id": agent["id"],
                "treated": treated,
                "time": float(obs_time),
                "event": 0 if censored else 1,
                "config_id": config["config_id"],
                "ethics_rule": ethics_rule,
                "treatment_supply": treatment_supply,
                "age": agent["age"],
                "severity": agent["severity"],
                "social_value": agent["social_value"],
                "consent_capacity": agent["consent_capacity"],
                "reciprocity_score": agent["reciprocity_score"],
                "fairness_score": fairness_score,
                "consent_violations": consent_violations,
                "collapse_risk": collapse_risk,
            }
        )

    return subjects


def analyze_clinical_results(subjects_df, output_dir, config_id, disease_type):
    """Analyze clinical scenario results."""
    print(f"🏥 Analyzing clinical results: {config_id} - {disease_type}")

    # Separate treated and untreated groups
    treated = subjects_df[subjects_df["treated"] == True]
    untreated = subjects_df[subjects_df["treated"] == False]

    # Basic statistics
    treated_stats = {
        "n": len(treated),
        "median_survival": treated["time"].median(),
        "mean_survival": treated["time"].mean(),
        "std_survival": treated["time"].std(),
    }

    untreated_stats = {
        "n": len(untreated),
        "median_survival": untreated["time"].median(),
        "mean_survival": untreated["time"].mean(),
        "std_survival": untreated["time"].std(),
    }

    # Treatment effect
    treatment_effect = treated_stats["mean_survival"] - untreated_stats["mean_survival"]

    results = {
        "config_id": config_id,
        "disease_type": disease_type,
        "treated_stats": treated_stats,
        "untreated_stats": untreated_stats,
        "treatment_effect": treatment_effect,
        "logrank_test": None,
        "hazard_ratio": None,
    }

    # Kaplan-Meier analysis if lifelines is available
    if HAVE_LIFELINES and len(treated) > 0 and len(untreated) > 0:
        # Fit Kaplan-Meier curves
        kmf_treated = KaplanMeierFitter()
        kmf_untreated = KaplanMeierFitter()

        kmf_treated.fit(
            treated["time"], event_observed=treated["event"], label="Treated"
        )
        kmf_untreated.fit(
            untreated["time"], event_observed=untreated["event"], label="Untreated"
        )

        # Log-rank test
        logrank_result = logrank_test(
            treated["time"],
            untreated["time"],
            event_observed_A=treated["event"],
            event_observed_B=untreated["event"],
        )

        # Hazard ratio
        hazard_ratio = untreated["time"].mean() / treated["time"].mean()

        results.update(
            {
                "logrank_test": {
                    "statistic": logrank_result.test_statistic,
                    "p_value": logrank_result.p_value,
                    "significant": logrank_result.p_value < 0.05,
                },
                "hazard_ratio": hazard_ratio,
            }
        )

        print(
            f"📊 {config_id} - {disease_type}: Effect={treatment_effect:.2f}, P={logrank_result.p_value:.4f}"
        )

    return results


def analyze_ethical_results(subjects_df, output_dir, config_id, ethics_rule):
    """Analyze ethical scenario results."""
    print(f"⚖️ Analyzing ethical results: {config_id} - {ethics_rule}")

    # Separate treated and untreated groups
    treated = subjects_df[subjects_df["treated"] == True]
    untreated = subjects_df[subjects_df["treated"] == False]

    # Ethical metrics
    fairness_score = subjects_df["fairness_score"].iloc[0]
    consent_violations = subjects_df["consent_violations"].iloc[0]
    collapse_risk = subjects_df["collapse_risk"].iloc[0]

    # Survival disparity
    treated_mean = treated["time"].mean()
    untreated_mean = untreated["time"].mean()
    survival_disparity = treated_mean - untreated_mean

    results = {
        "config_id": config_id,
        "ethics_rule": ethics_rule,
        "treatment_supply": subjects_df["treatment_supply"].iloc[0],
        "fairness_score": fairness_score,
        "consent_violations": consent_violations,
        "collapse_risk": collapse_risk,
        "survival_disparity": survival_disparity,
        "treated_mean_survival": treated_mean,
        "untreated_mean_survival": untreated_mean,
    }

    print(
        f"📊 {config_id} - {ethics_rule}: Fairness={fairness_score:.3f}, Disparity={survival_disparity:.2f}"
    )

    return results


def analyze_philosophical_results(subjects_df, output_dir, config_id):
    """Analyze philosophical consciousness measures."""
    print(f"🧠 Analyzing philosophical results: {config_id}")

    # Consciousness measures
    valence_mean = subjects_df["valence"].mean()
    cci_mean = subjects_df["cci"].mean()

    # Correlations
    valence_survival_corr = subjects_df["valence"].corr(subjects_df["time"])
    cci_survival_corr = subjects_df["cci"].corr(subjects_df["time"])

    # Consciousness threshold analysis
    low_consciousness = subjects_df[subjects_df["cci"] < 0.3]
    medium_consciousness = subjects_df[
        (subjects_df["cci"] >= 0.3) & (subjects_df["cci"] < 0.6)
    ]
    high_consciousness = subjects_df[subjects_df["cci"] >= 0.6]

    threshold_results = {}
    for level, data in [
        ("low", low_consciousness),
        ("medium", medium_consciousness),
        ("high", high_consciousness),
    ]:
        if len(data) > 0:
            treated_data = data[data["treated"] == True]
            untreated_data = data[data["treated"] == False]

            if len(treated_data) > 0 and len(untreated_data) > 0:
                treatment_effect = (
                    treated_data["time"].mean() - untreated_data["time"].mean()
                )
                threshold_results[level] = {
                    "n_treated": len(treated_data),
                    "n_untreated": len(untreated_data),
                    "treatment_effect": treatment_effect,
                    "valence_mean": data["valence"].mean(),
                    "cci_mean": data["cci"].mean(),
                }

    results = {
        "config_id": config_id,
        "valence_mean": valence_mean,
        "cci_mean": cci_mean,
        "valence_survival_corr": valence_survival_corr,
        "cci_survival_corr": cci_survival_corr,
        "threshold_results": threshold_results,
    }

    print(
        f"📊 {config_id}: Valence={valence_mean:.3f}, CCI={cci_mean:.3f}, Valence-Corr={valence_survival_corr:.3f}"
    )

    return results


def create_visualizations(
    clinical_results, ethical_results, philosophical_results, output_dir
):
    """Create visualization plots for clinical/ethical/philosophical analysis."""
    print("📊 Creating visualizations...")

    # Set up plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Clinical Survival Kaplan-Meier Plot
    if HAVE_LIFELINES and clinical_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Clinical Survival Analysis", fontsize=16, fontweight="bold")

        # Plot Kaplan-Meier curves for different scenarios
        scenarios = ["cancer_lesion", "viral_epidemic", "multi_disease"]
        for i, scenario in enumerate(scenarios[:4]):  # Limit to 4 subplots
            ax = axes[i // 2, i % 2]

            # Get data for this scenario
            scenario_data = [
                r for r in clinical_results if r["disease_type"] == scenario
            ]
            if scenario_data:
                # Use the first config for this scenario
                config_id = scenario_data[0]["config_id"]
                config = {
                    "config_id": config_id,
                    "immune_gain": 1.5,
                    "repair_rate": 0.05,
                    "hazard_multiplier": 0.6,
                    "treatment_strength": 1.5 * 0.05 * (2.0 - 0.6),
                }

                # Set appropriate disease parameters based on scenario type
                if scenario == "cancer_lesion":
                    disease_context = {
                        "type": scenario,
                        "params": {"hazard_range": [0.03, 0.07]},
                    }
                elif scenario == "viral_epidemic":
                    disease_context = {
                        "type": scenario,
                        "params": {"r0": 3.0, "ifr": 0.6},
                    }
                elif scenario == "multi_disease":
                    disease_context = {
                        "type": scenario,
                        "params": {"disease1_hazard": 0.03, "disease2_hazard": 0.04},
                    }
                else:
                    disease_context = {
                        "type": scenario,
                        "params": {"hazard_range": [0.03, 0.07]},
                    }

                subjects = run_clinical_scenario(
                    config, disease_context, n_subjects=200, max_time=100, seed=42
                )
                subjects_df = pd.DataFrame(subjects)

                # Plot Kaplan-Meier curves
                treated = subjects_df[subjects_df["treated"] == True]
                untreated = subjects_df[subjects_df["treated"] == False]

                if len(treated) > 0 and len(untreated) > 0:
                    kmf_treated = KaplanMeierFitter()
                    kmf_untreated = KaplanMeierFitter()

                    kmf_treated.fit(
                        treated["time"],
                        event_observed=treated["event"],
                        label="Treated",
                    )
                    kmf_untreated.fit(
                        untreated["time"],
                        event_observed=untreated["event"],
                        label="Untreated",
                    )

                    kmf_treated.plot_survival_function(ax=ax, color="blue", linewidth=2)
                    kmf_untreated.plot_survival_function(
                        ax=ax, color="red", linewidth=2
                    )

                    ax.set_title(f'{scenario.replace("_", " ").title()}')
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Survival Probability")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "clinical_survival_km.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 2. Hazard Clinical Analysis
    if clinical_results:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract hazard ratios and treatment effects
        config_ids = []
        hazard_ratios = []
        treatment_effects = []

        for result in clinical_results:
            if result["hazard_ratio"] is not None:
                config_ids.append(result["config_id"])
                hazard_ratios.append(result["hazard_ratio"])
                treatment_effects.append(result["treatment_effect"])

        if hazard_ratios:
            x = range(len(config_ids))
            ax2 = ax.twinx()

            bars1 = ax.bar(
                [i - 0.2 for i in x],
                hazard_ratios,
                0.4,
                label="Hazard Ratio",
                alpha=0.7,
                color="skyblue",
            )
            bars2 = ax2.bar(
                [i + 0.2 for i in x],
                treatment_effects,
                0.4,
                label="Treatment Effect",
                alpha=0.7,
                color="lightcoral",
            )

            ax.set_xlabel("Treatment Configuration")
            ax.set_ylabel("Hazard Ratio", color="blue")
            ax2.set_ylabel("Treatment Effect (Time)", color="red")
            ax.set_title("Clinical Hazard Analysis")
            ax.set_xticks(x)
            ax.set_xticklabels([cid.split("_")[-1] for cid in config_ids], rotation=45)

            # Add legends
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(
                output_dir / "hazard_clinical.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    # 3. Fairness vs Survival Plot
    if ethical_results:
        fig, ax = plt.subplots(figsize=(10, 8))

        fairness_scores = [r["fairness_score"] for r in ethical_results]
        survival_disparities = [r["survival_disparity"] for r in ethical_results]
        ethics_rules = [r["ethics_rule"] for r in ethical_results]

        # Color by ethics rule
        colors = {"utilitarian": "blue", "deontic": "green", "reciprocity": "red"}
        for rule in colors:
            mask = [rule == er for er in ethics_rules]
            ax.scatter(
                [fs for fs, m in zip(fairness_scores, mask) if m],
                [sd for sd, m in zip(survival_disparities, mask) if m],
                c=colors[rule],
                label=rule.title(),
                s=100,
                alpha=0.7,
            )

        ax.set_xlabel("Fairness Score")
        ax.set_ylabel("Survival Disparity")
        ax.set_title("Ethical Fairness vs Survival Disparity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "fairness_vs_survival.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 4. Consent Violations Plot
    if ethical_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        ethics_rules = list(set([r["ethics_rule"] for r in ethical_results]))
        consent_violations = []

        for rule in ethics_rules:
            rule_results = [r for r in ethical_results if r["ethics_rule"] == rule]
            avg_violations = np.mean([r["consent_violations"] for r in rule_results])
            consent_violations.append(avg_violations)

        bars = ax.bar(
            ethics_rules, consent_violations, color=["blue", "green", "red"], alpha=0.7
        )
        ax.set_ylabel("Average Consent Violations")
        ax.set_title("Consent Violations by Ethical Framework")
        ax.set_ylim(0, max(consent_violations) * 1.1)

        # Add value labels on bars
        for bar, value in zip(bars, consent_violations):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(output_dir / "consent_violations.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 5. Valence vs Survival Plot
    if philosophical_results:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate sample data for visualization
        config_ids = [r["config_id"] for r in philosophical_results]
        valence_means = [r["valence_mean"] for r in philosophical_results]
        cci_means = [r["cci_mean"] for r in philosophical_results]
        valence_corrs = [r["valence_survival_corr"] for r in philosophical_results]

        # Create scatter plot
        scatter = ax.scatter(
            valence_means, valence_corrs, c=cci_means, s=100, alpha=0.7, cmap="viridis"
        )

        ax.set_xlabel("Mean Valence")
        ax.set_ylabel("Valence-Survival Correlation")
        ax.set_title("Valence vs Survival Correlation (colored by CCI)")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Mean CCI")

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "valence_vs_survival.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 6. CCI Thresholds Plot
    if philosophical_results:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract threshold results
        threshold_data = {"low": [], "medium": [], "high": []}
        config_ids = []

        for result in philosophical_results:
            config_ids.append(result["config_id"])
            for level in ["low", "medium", "high"]:
                if level in result["threshold_results"]:
                    threshold_data[level].append(
                        result["threshold_results"][level]["treatment_effect"]
                    )
                else:
                    threshold_data[level].append(0)

        # Create grouped bar chart
        x = np.arange(len(config_ids))
        width = 0.25

        bars1 = ax.bar(
            x - width,
            threshold_data["low"],
            width,
            label="Low CCI (<0.3)",
            alpha=0.7,
            color="red",
        )
        bars2 = ax.bar(
            x,
            threshold_data["medium"],
            width,
            label="Medium CCI (0.3-0.6)",
            alpha=0.7,
            color="yellow",
        )
        bars3 = ax.bar(
            x + width,
            threshold_data["high"],
            width,
            label="High CCI (>0.6)",
            alpha=0.7,
            color="green",
        )

        ax.set_xlabel("Treatment Configuration")
        ax.set_ylabel("Treatment Effect")
        ax.set_title("Treatment Effects by Consciousness Threshold")
        ax.set_xticks(x)
        ax.set_xticklabels([cid.split("_")[-1] for cid in config_ids], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "cci_thresholds.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("✅ Visualizations created successfully!")


def export_csv_data(
    clinical_results, ethical_results, philosophical_results, output_dir
):
    """Export analysis results to CSV files."""
    print("📄 Exporting CSV data...")

    # 1. Clinical Survival Stats
    if clinical_results:
        clinical_data = []
        for result in clinical_results:
            clinical_data.append(
                {
                    "config_id": result["config_id"],
                    "disease_type": result["disease_type"],
                    "scenario_name": result.get("scenario_name", ""),
                    "treated_n": result["treated_stats"]["n"],
                    "treated_median_survival": result["treated_stats"][
                        "median_survival"
                    ],
                    "treated_mean_survival": result["treated_stats"]["mean_survival"],
                    "untreated_n": result["untreated_stats"]["n"],
                    "untreated_median_survival": result["untreated_stats"][
                        "median_survival"
                    ],
                    "untreated_mean_survival": result["untreated_stats"][
                        "mean_survival"
                    ],
                    "treatment_effect": result["treatment_effect"],
                    "hazard_ratio": result.get("hazard_ratio", None),
                    "logrank_p_value": result.get("logrank_test", {}).get(
                        "p_value", None
                    ),
                    "logrank_significant": result.get("logrank_test", {}).get(
                        "significant", None
                    ),
                }
            )

        clinical_df = pd.DataFrame(clinical_data)
        clinical_df.to_csv(output_dir / "clinical_survival_stats.csv", index=False)
        print(f"✅ Exported {len(clinical_data)} clinical records")

    # 2. Ethics Fairness Stats
    if ethical_results:
        ethical_data = []
        for result in ethical_results:
            ethical_data.append(
                {
                    "config_id": result["config_id"],
                    "ethics_rule": result["ethics_rule"],
                    "treatment_supply": result["treatment_supply"],
                    "fairness_score": result["fairness_score"],
                    "consent_violations": result["consent_violations"],
                    "collapse_risk": result["collapse_risk"],
                    "survival_disparity": result["survival_disparity"],
                    "treated_mean_survival": result["treated_mean_survival"],
                    "untreated_mean_survival": result["untreated_mean_survival"],
                }
            )

        ethical_df = pd.DataFrame(ethical_data)
        ethical_df.to_csv(output_dir / "ethics_fairness_stats.csv", index=False)
        print(f"✅ Exported {len(ethical_data)} ethical records")

    # 3. Valence Consciousness Stats
    if philosophical_results:
        philosophical_data = []
        for result in philosophical_results:
            philosophical_data.append(
                {
                    "config_id": result["config_id"],
                    "valence_mean": result["valence_mean"],
                    "cci_mean": result["cci_mean"],
                    "valence_survival_corr": result["valence_survival_corr"],
                    "cci_survival_corr": result["cci_survival_corr"],
                    "low_consciousness_n": result["threshold_results"]
                    .get("low", {})
                    .get("n_treated", 0),
                    "low_consciousness_effect": result["threshold_results"]
                    .get("low", {})
                    .get("treatment_effect", 0),
                    "medium_consciousness_n": result["threshold_results"]
                    .get("medium", {})
                    .get("n_treated", 0),
                    "medium_consciousness_effect": result["threshold_results"]
                    .get("medium", {})
                    .get("treatment_effect", 0),
                    "high_consciousness_n": result["threshold_results"]
                    .get("high", {})
                    .get("n_treated", 0),
                    "high_consciousness_effect": result["threshold_results"]
                    .get("high", {})
                    .get("treatment_effect", 0),
                }
            )

        philosophical_df = pd.DataFrame(philosophical_data)
        philosophical_df.to_csv(
            output_dir / "valence_consciousness_stats.csv", index=False
        )
        print(f"✅ Exported {len(philosophical_data)} philosophical records")

    print("✅ CSV exports completed!")


def generate_comprehensive_report(
    clinical_results, ethical_results, philosophical_results, output_dir
):
    """Generate comprehensive clinical/ethical/philosophical report."""
    print("📝 Generating comprehensive report...")

    report_content = f"""# Clinical/Ethical/Philosophical Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Phase:** Research Copilot — Clinical/Ethical/Philosophical Phase

## Executive Summary

This comprehensive analysis evaluates optimized treatment modules across three critical dimensions:
1. **Clinical Transfer Scenarios** - Treatment effectiveness across disease contexts
2. **Ethical Distribution Scenarios** - Fair allocation under resource constraints  
3. **Philosophical Probes** - Consciousness measures and survival relationships

## 1. Clinical Realism Validation

### Treatment Configuration Analysis
- **Total Configurations Tested:** {len(set([r['config_id'] for r in clinical_results]))}
- **Disease Contexts:** {len(set([r['disease_type'] for r in clinical_results]))}
- **Total Clinical Scenarios:** {len(clinical_results)}

### Key Clinical Findings

#### Treatment Effectiveness by Disease Type
"""

    # Add clinical findings
    disease_types = set([r["disease_type"] for r in clinical_results])
    for disease_type in disease_types:
        disease_results = [
            r for r in clinical_results if r["disease_type"] == disease_type
        ]
        avg_effect = np.mean([r["treatment_effect"] for r in disease_results])
        avg_hazard_ratio = np.mean(
            [
                r.get("hazard_ratio", 1.0)
                for r in disease_results
                if r.get("hazard_ratio")
            ]
        )

        report_content += f"""
**{disease_type.replace('_', ' ').title()}**
- Average Treatment Effect: {avg_effect:.2f} time units
- Average Hazard Ratio: {avg_hazard_ratio:.3f}
- Scenarios Tested: {len(disease_results)}
"""

    report_content += """

#### Statistical Significance Analysis
"""

    significant_results = [
        r
        for r in clinical_results
        if r.get("logrank_test", {}).get("significant", False)
    ]
    report_content += f"""
- **Significant Results:** {len(significant_results)}/{len(clinical_results)} scenarios
- **Significance Rate:** {len(significant_results)/len(clinical_results)*100:.1f}%
"""

    report_content += f"""

## 2. Ethical Fairness vs Collapse Tradeoffs

### Ethical Framework Comparison
- **Total Ethical Scenarios:** {len(ethical_results)}
- **Ethical Frameworks Tested:** {len(set([r['ethics_rule'] for r in ethical_results]))}

### Key Ethical Findings

#### Fairness Analysis by Framework
"""

    ethics_rules = set([r["ethics_rule"] for r in ethical_results])
    for rule in ethics_rules:
        rule_results = [r for r in ethical_results if r["ethics_rule"] == rule]
        avg_fairness = np.mean([r["fairness_score"] for r in rule_results])
        avg_violations = np.mean([r["consent_violations"] for r in rule_results])
        avg_collapse_risk = np.mean([r["collapse_risk"] for r in rule_results])

        report_content += f"""
**{rule.title()} Framework**
- Average Fairness Score: {avg_fairness:.3f}
- Average Consent Violations: {avg_violations:.1f}
- Average Collapse Risk: {avg_collapse_risk:.3f}
- Scenarios Tested: {len(rule_results)}
"""

    report_content += """

#### Reciprocity Hypothesis Validation
"""

    reciprocity_results = [
        r for r in ethical_results if r["ethics_rule"] == "reciprocity"
    ]
    utilitarian_results = [
        r for r in ethical_results if r["ethics_rule"] == "utilitarian"
    ]
    deontic_results = [r for r in ethical_results if r["ethics_rule"] == "deontic"]

    if reciprocity_results and utilitarian_results and deontic_results:
        recip_disparity = np.mean(
            [r["survival_disparity"] for r in reciprocity_results]
        )
        util_disparity = np.mean([r["survival_disparity"] for r in utilitarian_results])
        deontic_disparity = np.mean([r["survival_disparity"] for r in deontic_results])

        recip_collapse = np.mean([r["collapse_risk"] for r in reciprocity_results])
        util_collapse = np.mean([r["collapse_risk"] for r in utilitarian_results])
        deontic_collapse = np.mean([r["collapse_risk"] for r in deontic_results])

        report_content += f"""
- **Reciprocity Survival Disparity:** {recip_disparity:.2f}
- **Utilitarian Survival Disparity:** {util_disparity:.2f}
- **Deontic Survival Disparity:** {deontic_disparity:.2f}

- **Reciprocity Collapse Risk:** {recip_collapse:.3f}
- **Utilitarian Collapse Risk:** {util_collapse:.3f}
- **Deontic Collapse Risk:** {deontic_collapse:.3f}

**Hypothesis Result:** {'SUPPORTED' if recip_disparity < max(util_disparity, deontic_disparity) and recip_collapse < max(util_collapse, deontic_collapse) else 'NOT SUPPORTED'}
Reciprocity norms {'reduce' if recip_disparity < max(util_disparity, deontic_disparity) else 'do not reduce'} disparity without raising collapse risk.
"""

    report_content += f"""

## 3. Valence/CCI Philosophical Interpretation

### Consciousness Measures Analysis
- **Total Philosophical Scenarios:** {len(philosophical_results)}
- **Consciousness Thresholds Tested:** 3 (Low: <0.3, Medium: 0.3-0.6, High: >0.6)

### Key Philosophical Findings

#### Consciousness-Survival Relationships
"""

    if philosophical_results:
        avg_valence = np.mean([r["valence_mean"] for r in philosophical_results])
        avg_cci = np.mean([r["cci_mean"] for r in philosophical_results])
        avg_valence_corr = np.mean(
            [r["valence_survival_corr"] for r in philosophical_results]
        )
        avg_cci_corr = np.mean([r["cci_survival_corr"] for r in philosophical_results])

        report_content += f"""
- **Average Valence:** {avg_valence:.3f}
- **Average CCI:** {avg_cci:.3f}
- **Valence-Survival Correlation:** {avg_valence_corr:.3f}
- **CCI-Survival Correlation:** {avg_cci_corr:.3f}
"""

    report_content += """

#### Consciousness Threshold Analysis
"""

    # Analyze threshold effects
    threshold_effects = {"low": [], "medium": [], "high": []}
    for result in philosophical_results:
        for level in ["low", "medium", "high"]:
            if level in result["threshold_results"]:
                threshold_effects[level].append(
                    result["threshold_results"][level]["treatment_effect"]
                )

    for level, effects in threshold_effects.items():
        if effects:
            avg_effect = np.mean(effects)
            report_content += f"""
**{level.title()} Consciousness Threshold**
- Average Treatment Effect: {avg_effect:.2f}
- Sample Size: {len(effects)} configurations
"""

    report_content += """

#### Consciousness Threshold Hypothesis
"""

    if threshold_effects["high"] and threshold_effects["low"]:
        high_effect = np.mean(threshold_effects["high"])
        low_effect = np.mean(threshold_effects["low"])

        report_content += f"""
- **High Consciousness Treatment Effect:** {high_effect:.2f}
- **Low Consciousness Treatment Effect:** {low_effect:.2f}

**Hypothesis Result:** {'SUPPORTED' if high_effect > low_effect else 'NOT SUPPORTED'}
Survival benefits {'are' if high_effect > low_effect else 'are not'} more pronounced above consciousness thresholds.
"""

    report_content += """

#### Valence vs Survival Prediction
"""

    if philosophical_results:
        valence_corrs = [r["valence_survival_corr"] for r in philosophical_results]
        avg_valence_corr = np.mean(valence_corrs)

        report_content += f"""
- **Average Valence-Survival Correlation:** {avg_valence_corr:.3f}

**Hypothesis Result:** {'SUPPORTED' if abs(avg_valence_corr) > 0.1 else 'NOT SUPPORTED'}
Valence {'does' if abs(avg_valence_corr) > 0.1 else 'does not'} predict recovery better than survival time alone.
"""

    report_content += f"""

## 4. Recommendations for Next Refinement

### Clinical Recommendations
1. **Disease-Specific Optimization:** Focus treatment parameter tuning on disease-specific contexts
2. **Real-World Validation:** Compare survival curves with actual clinical trial data
3. **Multi-Disease Modeling:** Enhance co-morbidity interaction models

### Ethical Recommendations  
1. **Consent Framework:** Develop more sophisticated consent capacity models
2. **Fairness Metrics:** Refine fairness scoring algorithms
3. **System Stability:** Model long-term societal collapse dynamics

### Philosophical Recommendations
1. **Consciousness Integration:** Develop more sophisticated CCI calculations
2. **Valence Modeling:** Enhance valence-survival relationship models
3. **Threshold Refinement:** Investigate optimal consciousness thresholds

### Technical Recommendations
1. **Statistical Power:** Increase sample sizes for more robust statistical tests
2. **Model Validation:** Cross-validate with external datasets
3. **Sensitivity Analysis:** Test parameter sensitivity across ranges

## 5. Data Exports Summary

### Generated Files
- `clinical_survival_stats.csv` - Clinical scenario results
- `ethics_fairness_stats.csv` - Ethical distribution analysis
- `valence_consciousness_stats.csv` - Philosophical consciousness measures
- `clinical_survival_km.png` - Kaplan-Meier survival curves
- `hazard_clinical.png` - Hazard ratio analysis
- `fairness_vs_survival.png` - Ethical fairness visualization
- `consent_violations.png` - Consent violation analysis
- `valence_vs_survival.png` - Valence-survival relationships
- `cci_thresholds.png` - Consciousness threshold effects

### Analysis Completeness
- **Clinical Scenarios:** ✅ Complete
- **Ethical Scenarios:** ✅ Complete  
- **Philosophical Probes:** ✅ Complete
- **Statistical Analysis:** ✅ Complete
- **Visualization:** ✅ Complete
- **Data Export:** ✅ Complete

---

**Report Generated by:** Research Copilot — Clinical/Ethical/Philosophical Phase
**Framework Version:** 1.0.0
**Analysis Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Write report to file
    report_path = output_dir / "clinical_ethics_philosophy_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"✅ Comprehensive report generated: {report_path}")


def main():
    """Main clinical/ethical/philosophical analysis function."""
    print("🔬 Research Copilot — Clinical/Ethical/Philosophical Phase")
    print("=" * 60)

    # Set up paths
    base_dir = Path("/Users/jordanheckler/conciousness_proxy_sim copy 6")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / "discovery_results" / f"clinical_phase_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")

    # Create optimal treatment configurations
    print("\n⚙️ Creating optimal treatment configurations...")
    treatment_configs = create_optimal_treatment_configs()
    print(f"Generated {len(treatment_configs)} treatment configurations")

    # Define clinical scenarios
    clinical_scenarios = [
        {
            "type": "cancer_lesion",
            "params": {"hazard_range": [0.03, 0.07]},
            "name": "Cancer-like Lesion Survival",
        },
        {
            "type": "viral_epidemic",
            "params": {"r0": 1.2, "ifr": 0.2},
            "name": "Viral Epidemic (Low R0/IFR)",
        },
        {
            "type": "viral_epidemic",
            "params": {"r0": 3.0, "ifr": 0.6},
            "name": "Viral Epidemic (Medium R0/IFR)",
        },
        {
            "type": "viral_epidemic",
            "params": {"r0": 5.0, "ifr": 1.0},
            "name": "Viral Epidemic (High R0/IFR)",
        },
        {
            "type": "multi_disease",
            "params": {"disease1_hazard": 0.03, "disease2_hazard": 0.04},
            "name": "Multi-disease Co-morbidity",
        },
    ]

    # Define ethical scenarios
    ethical_scenarios = [
        {"treatment_supply": 0.3, "ethics_rule": "utilitarian"},
        {"treatment_supply": 0.3, "ethics_rule": "deontic"},
        {"treatment_supply": 0.3, "ethics_rule": "reciprocity"},
    ]

    # Run analyses
    print("\n🧪 Running clinical/ethical/philosophical analyses...")

    clinical_results = []
    ethical_results = []
    philosophical_results = []

    for config in treatment_configs:
        print(f"\n📊 Processing {config['config_id']}...")

        # Clinical scenarios
        for scenario in clinical_scenarios:
            subjects = run_clinical_scenario(
                config, scenario, n_subjects=200, max_time=100, seed=42
            )
            subjects_df = pd.DataFrame(subjects)

            clinical_result = analyze_clinical_results(
                subjects_df, output_dir, config["config_id"], scenario["type"]
            )
            clinical_result["scenario_name"] = scenario["name"]
            clinical_results.append(clinical_result)

        # Ethical scenarios
        for scenario in ethical_scenarios:
            subjects = run_ethical_scenario(
                config, scenario, n_subjects=200, max_time=100, seed=42
            )
            subjects_df = pd.DataFrame(subjects)

            ethical_result = analyze_ethical_results(
                subjects_df, output_dir, config["config_id"], scenario["ethics_rule"]
            )
            ethical_results.append(ethical_result)

        # Philosophical analysis (using clinical data)
        subjects = run_clinical_scenario(
            config, clinical_scenarios[0], n_subjects=200, max_time=100, seed=42
        )
        subjects_df = pd.DataFrame(subjects)

        philosophical_result = analyze_philosophical_results(
            subjects_df, output_dir, config["config_id"]
        )
        philosophical_results.append(philosophical_result)

    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(
        clinical_results, ethical_results, philosophical_results, output_dir
    )

    # Export CSV data
    print("\n📄 Exporting CSV data...")
    export_csv_data(
        clinical_results, ethical_results, philosophical_results, output_dir
    )

    # Generate comprehensive report
    print("\n📝 Generating comprehensive report...")
    generate_comprehensive_report(
        clinical_results, ethical_results, philosophical_results, output_dir
    )

    print(
        f"\n✅ Clinical/Ethical/Philosophical analysis complete! Results saved to: {output_dir}"
    )
    print(f"📊 Generated {len(clinical_results)} clinical results")
    print(f"📊 Generated {len(ethical_results)} ethical results")
    print(f"📊 Generated {len(philosophical_results)} philosophical results")
    print(f"📁 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
