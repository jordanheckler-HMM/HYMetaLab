"""
Core schemas for extended simulation framework using Pydantic.
"""

from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Individual agent state with medical, cognitive, and social attributes."""

    # Physical/Medical
    energy_kJ: float = Field(default=2000.0, description="Total energy in kJ")
    mass_kg: float = Field(default=70.0, description="Body mass in kg")
    temp_K: float = Field(default=310.15, description="Body temperature in Kelvin")
    hydration: float = Field(default=0.6, description="Hydration level 0-1")
    damage: float = Field(default=0.0, description="Accumulated damage 0-1")
    immune_activation: float = Field(
        default=0.0, description="Immune system activation 0-1"
    )
    age: int = Field(default=25, description="Age in years")
    healthspan_score: float = Field(default=1.0, description="Health quality 0-1")
    lifespan_expectancy: float = Field(default=80.0, description="Expected lifespan")

    # Cognitive/Social
    cci: float = Field(default=0.7, description="Collective consciousness index 0-1")
    valence: float = Field(default=0.0, description="Affective valence -1 to 1")
    goals: list[str] = Field(
        default_factory=lambda: ["survive", "thrive"], description="Agent goals"
    )
    ethics_profile: dict[str, float] = Field(
        default_factory=lambda: {
            "fairness_weight": 0.5,
            "harm_minimization": 0.7,
            "consent_threshold": 0.8,
            "reciprocity": 0.6,
        }
    )
    memory: dict[str, Any] = Field(default_factory=dict, description="Agent memory")
    trust_map: dict[int, float] = Field(
        default_factory=dict, description="Trust levels to other agents"
    )
    self_model: dict[str, Any] = Field(
        default_factory=dict, description="Internal world model"
    )
    consent_prefs: dict[str, bool] = Field(
        default_factory=lambda: {
            "data_sharing": True,
            "medical_treatment": True,
            "social_cooperation": True,
        }
    )

    class Config:
        extra = "allow"  # Allow additional fields


class WorldState(BaseModel):
    """Global world state including resources, environment, and social structures."""

    # Resources & Environment
    resources: float = Field(default=1000.0, description="Total available resources")
    temperature: float = Field(default=295.0, description="Environmental temperature K")
    toxin_level: float = Field(default=0.0, description="Environmental toxins 0-1")
    radiation: float = Field(default=0.0, description="Radiation level 0-1")

    # Physical Fields
    fields: dict[str, float] = Field(
        default_factory=lambda: {"gravity": 9.81, "electromag": 0.0}
    )

    # Disease & Pathogens
    pathogen_pool: dict[str, float] = Field(
        default_factory=dict, description="Pathogen concentrations"
    )

    # Social Structures
    policies: dict[str, Any] = Field(
        default_factory=dict, description="Active policies"
    )
    comms_graph: dict[str, Any] = Field(
        default_factory=dict, description="Communication network"
    )
    norms_state: dict[str, float] = Field(
        default_factory=lambda: {
            "fairness_weight": 0.5,
            "harm_minimization": 0.7,
            "consent_threshold": 0.8,
            "reciprocity": 0.6,
        }
    )

    # Additional attributes needed by modules
    coordination_strength: float = Field(
        default=0.6, description="Social coordination strength"
    )
    agents: list[Any] = Field(default_factory=list, description="List of agents")

    class Config:
        extra = "allow"  # Allow additional fields


class ExperimentConfig(BaseModel):
    """Configuration for extended simulation experiments."""

    # Basic Parameters
    n_agents: int | list[int] = Field(default=100, description="Number of agents")
    timesteps: int = Field(default=1000, description="Simulation duration")
    dt: float = Field(default=1.0, description="Time step size")

    # Shocks
    shocks: list[dict[str, Any]] = Field(
        default_factory=lambda: [{"severity": 0.5, "timing": 500, "type": "external"}]
    )

    # Disease Parameters
    disease: dict[str, float | list[float]] = Field(
        default_factory=lambda: {
            "R0": 2.0,
            "IFR": 0.5,
            "incubation": 5.0,
            "vacc_rate": 0.01,
            "waning": 0.001,
        }
    )

    # Information Layer
    info: dict[str, float | list[float]] = Field(
        default_factory=lambda: {"misinfo_rate": 0.1, "trust_decay": 0.005}
    )

    # Ethics
    ethics: dict[str, str | list[str] | float | list[float]] = Field(
        default_factory=lambda: {"rule_set": "utilitarian", "mutation_rate": 0.01}
    )

    # Multi-scale
    multiscale: dict[str, float | list[float] | int] = Field(
        default_factory=lambda: {"cell_agents": 16, "coupling": 0.5}
    )

    # Energy/Thermodynamics
    energy: dict[str, float | list[float]] = Field(
        default_factory=lambda: {
            "softening": 0.05,
            "heat_loss": 0.02,
            "work_coeffs": {"metabolic": 0.8, "social": 0.2},
        }
    )

    # Additional Parameters
    noise: float | list[float] = Field(default=0.1, description="System noise level")
    seeds: list[int] = Field(
        default_factory=lambda: [42, 123, 456], description="Random seeds"
    )
    sweep_ranges: dict[str, list[Any]] = Field(
        default_factory=dict, description="Parameter sweep ranges"
    )

    # Analysis Options
    enable_uq: bool = Field(
        default=True, description="Enable uncertainty quantification"
    )
    enable_bayes: bool = Field(default=False, description="Enable Bayesian inference")
    valence_weighting: float | list[float] = Field(
        default=0.5, description="Valence influence weight"
    )


class Metrics(BaseModel):
    """Comprehensive metrics for simulation analysis."""

    # Survival & Health
    survival_rate: float = Field(default=1.0, description="Fraction surviving")
    recovery_time: float | None = Field(default=None, description="Time to recovery")
    collapse_risk: float = Field(default=0.0, description="Collapse probability")

    # Social & Economic
    gini: float = Field(default=0.3, description="Gini coefficient")
    coordination_strength: float = Field(default=0.6, description="Social coordination")

    # Cognitive
    cci_mean: float = Field(default=0.7, description="Mean collective consciousness")
    valence_mean: float = Field(default=0.0, description="Mean affective valence")

    # Physical
    bound_fraction: float = Field(default=0.8, description="Fraction of bound states")
    energy_drift: float = Field(default=0.0, description="Energy conservation drift %")

    # Information & Beliefs
    belief_convergence: float = Field(
        default=0.5, description="Belief system convergence"
    )
    info_accuracy: float = Field(default=0.8, description="Information accuracy")

    # Disease
    infection_curve: list[float] = Field(
        default_factory=list, description="Infection time series"
    )
    R_eff: float = Field(default=1.0, description="Effective reproduction number")
    basic_reproduction_est: float = Field(default=2.0, description="Estimated R0")

    # Ethics
    fairness_score: float = Field(default=0.7, description="Fairness measure")
    consent_violations: int = Field(
        default=0, description="Number of consent violations"
    )
    ethics_stability: float = Field(default=0.8, description="Ethics system stability")

    # Thermodynamics
    entropy_flux: float = Field(default=0.0, description="Entropy flux rate")

    class Config:
        arbitrary_types_allowed = True
