#!/usr/bin/env python3
"""
Consciousness Proxy Simulation

A simulation exploring consciousness-like behaviors through global workspace theory
and metacognition in artificial agents.
"""

import argparse
import csv
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FoodCell:
    """A food cell with quality and depletion state."""

    x: int
    y: int
    quality: float = 1.0  # Energy value (0.5-2.0)
    depletion: float = 0.0  # How much has been consumed (0.0-1.0)
    regeneration_rate: float = 0.02  # How fast it regenerates
    last_consumed: int = 0  # Tick when last consumed


@dataclass
class Agent:
    """An agent in the simulation with consciousness-proxy behaviors."""

    agent_id: int
    parent_id: int | None
    birth_tick: int
    death_tick: int | None = None
    children: list[int] = field(default_factory=list)
    x: int = 0
    y: int = 0
    energy: float = 30.0
    age: int = 0
    health: float = 1.0  # Health status (0.0-1.0)

    # Expanded genome with human-like traits
    genome: list[float] = field(
        default_factory=lambda: [
            0.5,
            0.3,
            0.2,
            0.4,
            0.3,  # Original: food_seek, energy_caution, exploration, memory_strength, social_tendency
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,  # Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,  # Emotions: fear, hope, frustration, curiosity, boredom
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,  # Skills: pattern_recognition, causal_reasoning, tool_use, communication, innovation
        ]
    )

    workspace_reads: int = 0
    memory: dict[str, Any] = field(default_factory=dict)  # Spatial and social memory
    confidence_history: list[float] = field(
        default_factory=list
    )  # For calibration learning

    # Emotional state
    emotions: dict[str, float] = field(
        default_factory=lambda: {
            "fear": 0.0,
            "hope": 0.5,
            "frustration": 0.0,
            "curiosity": 0.5,
            "boredom": 0.0,
        }
    )

    # Personality traits (Big Five)
    personality: dict[str, float] = field(
        default_factory=lambda: {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
    )

    # Motivational state (Maslow's hierarchy)
    motivation: dict[str, float] = field(
        default_factory=lambda: {
            "physiological": 1.0,
            "safety": 0.8,
            "social": 0.6,
            "esteem": 0.4,
            "self_actualization": 0.2,
        }
    )

    # Skills and abilities
    skills: dict[str, float] = field(
        default_factory=lambda: {
            "pattern_recognition": 0.5,
            "causal_reasoning": 0.5,
            "tool_use": 0.5,
            "communication": 0.5,
            "innovation": 0.5,
        }
    )

    # Life stage
    life_stage: str = "childhood"  # childhood, adolescence, adulthood, old_age

    # Social relationships
    relationships: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )  # agent_id -> relationship data

    # Goals and plans
    current_goals: list[dict[str, Any]] = field(default_factory=list)

    # Tools and possessions
    tools: list[str] = field(default_factory=list)

    # Cultural knowledge
    traditions: list[str] = field(default_factory=list)
    norms: dict[str, float] = field(default_factory=dict)

    # Communication
    messages_sent: int = 0
    messages_received: int = 0
    deception_count: int = 0

    # Advanced Cognitive Architecture
    working_memory: list[dict] = field(default_factory=list)
    attention_focus: int = 3  # Can focus on 3 items at once
    memory_capacity: int = 7  # Miller's rule: 7±2 items

    # Executive Control
    executive_functions: dict[str, float] = field(
        default_factory=lambda: {
            "planning": 0.5,
            "inhibition": 0.5,
            "switching": 0.5,
            "monitoring": 0.5,
        }
    )
    current_goals: list[dict] = field(default_factory=list)
    goal_stack: list[dict] = field(default_factory=list)

    # Theory of Mind
    mental_models: dict[int, dict] = field(default_factory=dict)
    deception_skills: float = 0.5
    empathy_level: float = 0.5
    lie_detection: float = 0.5

    # Symbolic Language
    vocabulary: dict[str, str] = field(
        default_factory=lambda: {
            "food": "nourishment",
            "danger": "threat",
            "friend": "ally",
            "enemy": "foe",
            "help": "assist",
            "run": "flee",
            "stay": "wait",
            "go": "move",
        }
    )
    grammar_rules: list[str] = field(
        default_factory=lambda: ["SVO", "question", "command"]
    )
    language_skills: float = 0.5
    conversation_history: list[dict] = field(default_factory=list)

    # Self-Awareness & Identity
    self_concept: dict[str, Any] = field(
        default_factory=lambda: {
            "identity": "unknown",
            "traits": [],
            "values": [],
            "beliefs": [],
        }
    )
    self_esteem: float = 0.5
    self_efficacy: float = 0.5
    identity: str = "unknown"

    # Existential Awareness
    existential_awareness: float = 0.5
    mortality_understanding: float = 0.5
    purpose_seeking: float = 0.5
    legacy_desire: float = 0.5
    spiritual_thoughts: list[str] = field(default_factory=list)

    # Advanced Social Dynamics
    relationships: dict[int, dict[str, Any]] = field(default_factory=dict)
    social_status: str = "unknown"  # leader, follower, neutral
    reputation: float = 0.5
    influence_level: float = 0.5

    # Complex Relationships
    friendships: dict[int, float] = field(
        default_factory=dict
    )  # agent_id -> friendship_level
    romantic_partners: dict[int, float] = field(
        default_factory=dict
    )  # agent_id -> love_level
    rivals: dict[int, float] = field(default_factory=dict)  # agent_id -> rivalry_level
    family_members: dict[int, str] = field(
        default_factory=dict
    )  # agent_id -> relationship_type

    # Group Dynamics
    group_membership: list[int] = field(default_factory=list)  # group_ids
    leadership_roles: list[str] = field(
        default_factory=list
    )  # roles like 'alpha', 'beta'
    dominance_hierarchy: int = 0  # position in hierarchy
    cooperation_level: float = 0.5
    conflict_resolution_skills: float = 0.5

    # Cultural Evolution
    cultural_traditions: list[str] = field(default_factory=list)
    cultural_norms: dict[str, float] = field(default_factory=dict)
    rituals_performed: list[str] = field(default_factory=list)
    cultural_knowledge: dict[str, Any] = field(default_factory=dict)

    # Altruism & Social Behavior
    altruism_level: float = 0.5
    sacrifice_count: int = 0
    help_given: int = 0
    help_received: int = 0
    social_learning: float = 0.5

    # Deep Self-Awareness
    self_reflection_history: list[dict] = field(default_factory=list)
    self_improvement_goals: list[dict] = field(default_factory=list)
    self_regulation_skills: dict[str, float] = field(
        default_factory=lambda: {
            "emotion_control": 0.5,
            "impulse_control": 0.5,
            "attention_control": 0.5,
        }
    )
    self_discipline_level: float = 0.5
    self_monitoring_frequency: float = 0.5
    self_actualization_progress: float = 0.0

    # Advanced Emotions
    complex_emotions: dict[str, float] = field(
        default_factory=lambda: {
            "love": 0.0,
            "jealousy": 0.0,
            "pride": 0.0,
            "shame": 0.0,
            "gratitude": 0.0,
            "contempt": 0.0,
            "awe": 0.0,
            "disgust": 0.0,
        }
    )
    emotional_regulation: dict[str, float] = field(
        default_factory=lambda: {
            "suppression": 0.5,
            "reappraisal": 0.5,
            "distraction": 0.5,
            "acceptance": 0.5,
        }
    )
    empathy_levels: dict[str, float] = field(
        default_factory=lambda: {
            "cognitive": 0.5,
            "emotional": 0.5,
            "compassionate": 0.5,
        }
    )
    emotional_intelligence: float = 0.5
    mood_disorders: dict[str, float] = field(
        default_factory=lambda: {"depression": 0.0, "anxiety": 0.0, "mania": 0.0}
    )
    emotional_memory: list[dict] = field(default_factory=list)

    # Scientific Thinking
    hypotheses: list[dict] = field(default_factory=list)
    experiments: list[dict] = field(default_factory=list)
    causal_reasoning_skills: dict[str, float] = field(
        default_factory=lambda: {
            "correlation": 0.5,
            "causation": 0.5,
            "counterfactual": 0.5,
        }
    )
    abstraction_level: float = 0.5
    logical_reasoning: dict[str, float] = field(
        default_factory=lambda: {"deductive": 0.5, "inductive": 0.5, "abductive": 0.5}
    )
    mathematical_skills: dict[str, float] = field(
        default_factory=lambda: {"arithmetic": 0.5, "geometry": 0.5, "statistics": 0.5}
    )
    theory_building: float = 0.5

    # Creative Expression
    artistic_works: list[dict] = field(default_factory=list)
    stories_told: list[dict] = field(default_factory=list)
    jokes_made: list[str] = field(default_factory=list)
    humor_level: float = 0.5
    imagination_capacity: float = 0.5
    aesthetic_appreciation: float = 0.5
    cultural_artifacts: list[dict] = field(default_factory=list)
    creative_collaborations: list[dict] = field(default_factory=list)

    # Creativity & Innovation
    creativity_level: float = 0.5
    innovation_skills: dict[str, float] = field(
        default_factory=lambda: {
            "tool_design": 0.5,
            "problem_solving": 0.5,
            "artistic": 0.5,
        }
    )
    creative_works: list[dict] = field(default_factory=list)

    # Physical Reality (Phase 13)
    z: int = 0  # 3D spatial coordinate
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z velocity
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z acceleration
    mass: float = 1.0  # Physical mass
    temperature: float = 37.0  # Body temperature
    pressure: float = 1.0  # Environmental pressure
    chemical_composition: dict[str, float] = field(
        default_factory=lambda: {
            "oxygen": 0.65,
            "carbon": 0.18,
            "hydrogen": 0.10,
            "nitrogen": 0.03,
            "calcium": 0.02,
            "phosphorus": 0.01,
        }
    )
    biological_processes: dict[str, float] = field(
        default_factory=lambda: {
            "metabolism": 1.0,
            "growth": 0.0,
            "reproduction": 0.0,
            "healing": 0.0,
        }
    )

    # Evolutionary Biology (Phase 14)
    dna_sequence: list[str] = field(
        default_factory=lambda: ["A", "T", "G", "C"] * 100
    )  # Simplified DNA
    genes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "appearance": ["A", "T", "G", "C"],
            "behavior": ["A", "T", "G", "C"],
            "intelligence": ["A", "T", "G", "C"],
        }
    )
    chromosomes: int = 23  # Human-like chromosome count
    mutations: list[dict] = field(default_factory=list)
    fitness_score: float = 0.5
    species_id: str = "homo_sapiens"
    generation: int = 1

    # Civilization & Culture (Phase 15)
    city_id: int | None = None
    citizenship_status: str = "citizen"  # citizen, resident, visitor, refugee
    government_role: str = "none"  # leader, official, citizen, rebel
    legal_status: str = "lawful"  # lawful, criminal, accused, convicted
    economic_class: str = "middle"  # poor, working, middle, upper, elite
    religious_beliefs: list[str] = field(default_factory=list)
    social_institutions: dict[str, str] = field(
        default_factory=lambda: {
            "family": "nuclear",
            "marriage": "single",
            "education": "basic",
        }
    )

    # Technology & Innovation (Phase 16)
    technology_tree: dict[str, bool] = field(
        default_factory=lambda: {
            "fire": False,
            "tools": False,
            "agriculture": False,
            "writing": False,
            "metallurgy": False,
        }
    )
    research_projects: list[dict] = field(default_factory=list)
    innovation_network: list[int] = field(default_factory=list)  # Connected agent IDs
    patents: list[dict] = field(default_factory=list)
    scientific_knowledge: dict[str, float] = field(
        default_factory=lambda: {
            "mathematics": 0.0,
            "physics": 0.0,
            "chemistry": 0.0,
            "biology": 0.0,
            "medicine": 0.0,
        }
    )

    # Information & Communication (Phase 17)
    information_storage: dict[str, Any] = field(default_factory=dict)
    communication_network: list[int] = field(default_factory=list)
    media_consumption: dict[str, float] = field(
        default_factory=lambda: {
            "news": 0.0,
            "entertainment": 0.0,
            "education": 0.0,
            "propaganda": 0.0,
        }
    )
    information_processing: dict[str, float] = field(
        default_factory=lambda: {
            "analysis": 0.5,
            "synthesis": 0.5,
            "interpretation": 0.5,
        }
    )
    knowledge_base: dict[str, Any] = field(default_factory=dict)
    information_security: float = 0.5

    # Psychological Reality (Phase 18)
    mental_health_status: str = (
        "healthy"  # healthy, depressed, anxious, manic, psychotic
    )
    therapy_sessions: list[dict] = field(default_factory=list)
    psychological_development_stage: str = "adult"  # child, adolescent, adult, elderly
    cognitive_biases: list[str] = field(default_factory=list)
    personality_disorders: list[str] = field(default_factory=list)
    social_psychology_traits: dict[str, float] = field(
        default_factory=lambda: {
            "conformity": 0.5,
            "obedience": 0.5,
            "group_think": 0.5,
        }
    )

    # Ecological Systems (Phase 19)
    ecological_role: str = (
        "omnivore"  # producer, herbivore, carnivore, omnivore, decomposer
    )
    food_web_position: int = 3  # Position in food chain (1=producer, 5=top predator)
    ecosystem_services: list[str] = field(default_factory=list)
    environmental_impact: float = 0.5
    biodiversity_contribution: float = 0.5
    climate_adaptation: float = 0.5

    # Health & Medicine (Phase 20)
    diseases: list[dict] = field(default_factory=list)
    medical_treatments: list[dict] = field(default_factory=list)
    healthcare_access: float = 0.5
    medical_insurance: bool = False
    public_health_status: str = "healthy"
    vaccination_status: dict[str, bool] = field(default_factory=dict)

    # Economic Systems (Phase 21)
    wealth: float = 100.0  # Economic wealth
    income: float = 10.0  # Per-tick income
    expenses: float = 5.0  # Per-tick expenses
    trade_relationships: dict[int, dict[str, Any]] = field(default_factory=dict)
    market_participation: dict[str, float] = field(
        default_factory=lambda: {"buyer": 0.5, "seller": 0.5, "investor": 0.0}
    )
    economic_cycles: dict[str, float] = field(
        default_factory=lambda: {"boom": 0.0, "bust": 0.0, "recovery": 0.0}
    )

    # Education & Learning (Phase 22)
    education_level: str = "basic"  # basic, intermediate, advanced, expert
    educational_institution: int | None = None
    learning_methods: dict[str, float] = field(
        default_factory=lambda: {
            "lecture": 0.5,
            "practice": 0.5,
            "research": 0.0,
            "mentoring": 0.0,
        }
    )
    knowledge_transfer: dict[int, dict[str, Any]] = field(default_factory=dict)
    educational_assessment: dict[str, float] = field(
        default_factory=lambda: {
            "testing": 0.0,
            "evaluation": 0.0,
            "certification": 0.0,
        }
    )
    lifelong_learning: float = 0.5

    # Cosmic & Astronomical Reality (Phase 23)
    solar_system_id: int = 1  # Which solar system the agent belongs to
    planet_id: int = 1  # Which planet in the solar system
    orbital_position: tuple[float, float] = (0.0, 0.0)  # Position in orbit
    orbital_velocity: float = 0.0  # Orbital velocity
    gravitational_field: float = 1.0  # Local gravitational strength
    cosmic_radiation: float = 0.0  # Exposure to cosmic radiation
    space_exploration: dict[str, Any] = field(default_factory=dict)
    astronomical_knowledge: dict[str, float] = field(
        default_factory=lambda: {
            "astronomy": 0.0,
            "astrophysics": 0.0,
            "cosmology": 0.0,
            "space_engineering": 0.0,
        }
    )

    # Neuroscience & Brain Simulation (Phase 24)
    neural_network: dict[str, list[dict]] = field(
        default_factory=lambda: {"neurons": [], "synapses": [], "pathways": []}
    )
    brain_regions: dict[str, float] = field(
        default_factory=lambda: {
            "cortex": 0.5,
            "hippocampus": 0.5,
            "amygdala": 0.5,
            "cerebellum": 0.5,
            "prefrontal": 0.5,
            "temporal": 0.5,
            "parietal": 0.5,
            "occipital": 0.5,
        }
    )
    neurotransmitters: dict[str, float] = field(
        default_factory=lambda: {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "gaba": 0.5,
            "glutamate": 0.5,
            "norepinephrine": 0.5,
            "acetylcholine": 0.5,
            "endorphins": 0.5,
        }
    )
    brain_plasticity: float = 0.5
    neural_oscillations: dict[str, float] = field(
        default_factory=lambda: {
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "delta": 0.0,
        }
    )
    brain_computer_interface: bool = False

    # Quantum Mechanics & Physics (Phase 25)
    quantum_state: dict[str, complex] = field(
        default_factory=lambda: {
            "superposition": complex(0.5, 0.5),
            "entanglement": complex(0.0, 0.0),
        }
    )
    quantum_uncertainty: float = 0.5
    quantum_computing: dict[str, Any] = field(default_factory=dict)
    quantum_biology: dict[str, float] = field(
        default_factory=lambda: {
            "quantum_coherence": 0.0,
            "quantum_tunneling": 0.0,
            "quantum_sensing": 0.0,
        }
    )
    quantum_consciousness: float = 0.0

    # Climate & Weather Systems (Phase 26)
    climate_zone: str = "temperate"  # tropical, temperate, arctic, desert
    weather_patterns: dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 20.0,
            "humidity": 0.5,
            "pressure": 1013.25,
            "wind_speed": 0.0,
            "precipitation": 0.0,
            "cloud_cover": 0.0,
            "visibility": 10.0,
        }
    )
    climate_change_impact: float = 0.0
    natural_disasters: list[dict] = field(default_factory=list)
    atmospheric_dynamics: dict[str, float] = field(
        default_factory=lambda: {
            "jet_stream": 0.0,
            "ocean_currents": 0.0,
            "el_nino": 0.0,
            "la_nina": 0.0,
        }
    )

    # Molecular & Cellular Biology (Phase 27)
    cellular_machinery: dict[str, dict] = field(
        default_factory=lambda: {
            "nucleus": {"dna": 100, "rna": 50},
            "mitochondria": {"energy": 100},
            "ribosomes": {"proteins": 0},
            "endoplasmic_reticulum": {"lipids": 0},
            "golgi_apparatus": {"processing": 0},
            "lysosomes": {"digestion": 0},
        }
    )
    protein_synthesis: dict[str, float] = field(
        default_factory=lambda: {
            "transcription": 0.0,
            "translation": 0.0,
            "folding": 0.0,
            "modification": 0.0,
        }
    )
    gene_expression: dict[str, float] = field(
        default_factory=lambda: {
            "regulation": 0.5,
            "epigenetics": 0.0,
            "rna_interference": 0.0,
        }
    )
    metabolic_pathways: dict[str, float] = field(
        default_factory=lambda: {
            "glycolysis": 0.0,
            "krebs_cycle": 0.0,
            "oxidative_phosphorylation": 0.0,
            "photosynthesis": 0.0,
            "fermentation": 0.0,
        }
    )
    cell_division: dict[str, float] = field(
        default_factory=lambda: {"mitosis": 0.0, "meiosis": 0.0, "cell_cycle": 0.0}
    )

    # Internet & Digital Reality (Phase 28)
    internet_presence: dict[str, Any] = field(default_factory=dict)
    social_media_profiles: dict[str, dict] = field(default_factory=dict)
    digital_identity: dict[str, Any] = field(default_factory=dict)
    virtual_worlds: list[dict] = field(default_factory=list)
    digital_economy: dict[str, float] = field(
        default_factory=lambda: {
            "cryptocurrency": 0.0,
            "nfts": 0,
            "digital_assets": 0.0,
        }
    )
    online_communities: list[dict] = field(default_factory=list)

    # Art & Cultural Expression (Phase 29)
    art_history_knowledge: dict[str, float] = field(
        default_factory=lambda: {
            "renaissance": 0.0,
            "baroque": 0.0,
            "romanticism": 0.0,
            "modernism": 0.0,
            "postmodernism": 0.0,
            "contemporary": 0.0,
        }
    )
    cultural_trends: dict[str, float] = field(
        default_factory=lambda: {
            "fashion": 0.0,
            "music": 0.0,
            "literature": 0.0,
            "film": 0.0,
            "architecture": 0.0,
            "design": 0.0,
        }
    )
    aesthetic_evolution: float = 0.5
    cultural_movements: list[dict] = field(default_factory=list)
    artistic_collaboration: dict[str, float] = field(
        default_factory=lambda: {"collective_creation": 0.0, "cultural_exchange": 0.0}
    )

    # Political Systems & Governance (Phase 30)
    political_system: str = "democracy"  # democracy, autocracy, theocracy, oligarchy
    political_party: str | None = None
    voting_record: list[dict] = field(default_factory=list)
    policy_positions: dict[str, float] = field(
        default_factory=lambda: {
            "liberal": 0.0,
            "conservative": 0.0,
            "socialist": 0.0,
            "libertarian": 0.0,
        }
    )
    international_relations: dict[str, float] = field(
        default_factory=lambda: {
            "diplomacy": 0.0,
            "trade_agreements": 0.0,
            "conflicts": 0.0,
        }
    )
    governance_structures: dict[str, float] = field(
        default_factory=lambda: {"federal": 0.0, "unitary": 0.0, "confederal": 0.0}
    )

    # Biotechnology & Genetic Engineering (Phase 31)
    genetic_modification: dict[str, Any] = field(default_factory=dict)
    crispr_technology: bool = False
    gene_therapy: list[dict] = field(default_factory=list)
    synthetic_biology: dict[str, Any] = field(default_factory=dict)
    cloning_technology: bool = False
    biotechnology: dict[str, float] = field(
        default_factory=lambda: {
            "pharmaceuticals": 0.0,
            "agriculture": 0.0,
            "medicine": 0.0,
        }
    )

    # Artificial Intelligence & Machine Learning (Phase 32)
    ai_systems: dict[str, Any] = field(default_factory=dict)
    machine_learning: dict[str, float] = field(
        default_factory=lambda: {
            "neural_networks": 0.0,
            "deep_learning": 0.0,
            "algorithms": 0.0,
        }
    )
    human_ai_interaction: dict[str, float] = field(
        default_factory=lambda: {
            "collaboration": 0.0,
            "augmentation": 0.0,
            "replacement": 0.0,
        }
    )
    ai_ethics: dict[str, float] = field(
        default_factory=lambda: {
            "bias": 0.0,
            "fairness": 0.0,
            "transparency": 0.0,
            "accountability": 0.0,
        }
    )
    agi_development: float = 0.0
    superintelligence: bool = False

    def __post_init__(self):
        """Initialize agent with random position if not set."""
        if self.x == 0 and self.y == 0:
            self.x = random.randint(0, 19)
            self.y = random.randint(0, 19)


class GlobalWorkspace:
    """Global workspace for agent information sharing."""

    def __init__(self, lesion_mode: bool = False):
        self.lesion_mode = lesion_mode
        self.data: dict[str, Any] = {}

    def write(self, key: str, value: Any) -> None:
        """Write data to workspace."""
        if not self.lesion_mode:
            self.data[key] = value

    def read(self, key: str) -> Any | None:
        """Read data from workspace."""
        if self.lesion_mode:
            return None
        return self.data.get(key)

    def broadcast_nearest_food(self, food_positions: list[tuple[int, int]]) -> None:
        """Broadcast nearest food positions to all agents."""
        if not self.lesion_mode:
            self.data["nearest_food"] = food_positions[:10]  # Limit to top 10

    def broadcast_agent_info(self, agent: Agent) -> None:
        """Broadcast agent information."""
        if not self.lesion_mode:
            self.data[f"agent_{agent.agent_id}"] = {
                "pos": (agent.x, agent.y),
                "energy": agent.energy,
            }


class World:
    """2D grid world with food cells and agents."""

    def __init__(self, grid_size: int = 20, food_regen_prob: float = 0.02):
        self.grid_size = grid_size
        self.food_regen_prob = food_regen_prob
        self.food_cells: dict[tuple[int, int], FoodCell] = {}
        self.agents: list[Agent] = []
        self.next_agent_id = 0
        self.workspace = GlobalWorkspace()
        self.tick = 0
        self.seasonal_cycle = 0  # For seasonal food variation

        # Environmental complexity
        self.terrain: dict[tuple[int, int], str] = (
            {}
        )  # 'plains', 'forest', 'mountain', 'water'
        self.weather: str = "clear"  # 'clear', 'rain', 'storm', 'fog'
        self.day_night: str = "day"  # 'day', 'night'
        self.temperature: float = 20.0  # Celsius
        self.shelters: dict[tuple[int, int], dict[str, Any]] = {}  # Safe places
        self.resources: dict[tuple[int, int], dict[str, float]] = (
            {}
        )  # Water, materials, tools

        # Cultural evolution
        self.global_traditions: list[str] = []
        self.global_norms: dict[str, float] = {}
        self.cultural_artifacts: list[dict[str, Any]] = []

        # Initialize terrain
        self._initialize_terrain()
        self._initialize_shelters()
        self._initialize_resources()

    def _initialize_terrain(self):
        """Initialize terrain types across the world."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Create terrain patches
                if random.random() < 0.1:  # 10% mountains
                    self.terrain[(x, y)] = "mountain"
                elif random.random() < 0.2:  # 20% forests
                    self.terrain[(x, y)] = "forest"
                elif random.random() < 0.05:  # 5% water
                    self.terrain[(x, y)] = "water"
                else:  # 65% plains
                    self.terrain[(x, y)] = "plains"

    def _initialize_shelters(self):
        """Initialize safe shelter locations."""
        num_shelters = max(1, self.grid_size // 10)
        for _ in range(num_shelters):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.shelters[(x, y)] = {
                "capacity": random.randint(2, 5),
                "safety_level": random.uniform(0.7, 1.0),
                "comfort": random.uniform(0.5, 1.0),
            }

    def _initialize_resources(self):
        """Initialize resource locations."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if random.random() < 0.1:  # 10% chance of resources
                    resources = {}
                    if random.random() < 0.3:  # Water
                        resources["water"] = random.uniform(0.5, 2.0)
                    if random.random() < 0.2:  # Materials
                        resources["materials"] = random.uniform(0.3, 1.5)
                    if random.random() < 0.05:  # Tools
                        resources["tools"] = random.uniform(0.1, 1.0)

                    if resources:
                        self.resources[(x, y)] = resources

    def add_agent(self, parent: Agent | None = None) -> Agent:
        """Add a new agent to the world."""
        agent_id = self.next_agent_id
        self.next_agent_id += 1

        if parent:
            # Reproduction: split energy and mutate genome
            child_energy = parent.energy / 2
            parent.energy = child_energy

            # Mutate genome
            child_genome = []
            for gene in parent.genome:
                mutation = random.gauss(0, 0.1)
                new_gene = max(0, min(1, gene + mutation))
                child_genome.append(new_gene)

            agent = Agent(
                agent_id=agent_id,
                parent_id=parent.agent_id,
                birth_tick=self.tick,
                x=parent.x,
                y=parent.y,
                energy=child_energy,
                genome=child_genome,
            )
        else:
            # Initial agent
            agent = Agent(
                agent_id=agent_id,
                parent_id=None,
                birth_tick=self.tick,
                x=random.randint(0, self.grid_size - 1),
                y=random.randint(0, self.grid_size - 1),
            )

        self.agents.append(agent)
        return agent

    def regenerate_food(self) -> None:
        """Regenerate food cells with seasonal variation and clustering."""
        # Seasonal variation
        seasonal_factor = 1.0 + 0.3 * math.sin(self.tick * 0.1)

        # Create food patches (clustering)
        if random.random() < self.food_regen_prob * seasonal_factor:
            # Create a food patch
            center_x = random.randint(0, self.grid_size - 1)
            center_y = random.randint(0, self.grid_size - 1)
            patch_size = random.randint(2, 5)

            for dx in range(-patch_size, patch_size + 1):
                for dy in range(-patch_size, patch_size + 1):
                    x = (center_x + dx) % self.grid_size
                    y = (center_y + dy) % self.grid_size

                    # Distance from center affects probability
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance <= patch_size and random.random() < (
                        1.0 - distance / patch_size
                    ):
                        if (x, y) not in self.food_cells:
                            # Create new food cell with quality variation
                            quality = random.uniform(0.5, 2.0)
                            regeneration_rate = random.uniform(0.01, 0.03)
                            self.food_cells[(x, y)] = FoodCell(
                                x, y, quality, 0.0, regeneration_rate, self.tick
                            )

        # Regenerate existing food cells
        for pos, food_cell in list(self.food_cells.items()):
            if food_cell.depletion > 0:
                # Regenerate depleted food
                food_cell.depletion = max(
                    0.0, food_cell.depletion - food_cell.regeneration_rate
                )
            elif self.tick - food_cell.last_consumed > 50:  # Remove old unused food
                del self.food_cells[pos]

    def get_food_in_radius(
        self, x: int, y: int, radius: int = 3
    ) -> list[tuple[int, int]]:
        """Get food cells within perception radius."""
        nearby_food = []
        for fx, fy in self.food_cells:
            distance = math.sqrt((fx - x) ** 2 + (fy - y) ** 2)
            if distance <= radius:
                nearby_food.append((fx, fy))
        return nearby_food

    def get_nearest_food(self, x: int, y: int) -> list[tuple[int, int]]:
        """Get nearest food cells sorted by distance."""
        if not self.food_cells:
            return []

        food_with_distances = []
        for fx, fy in self.food_cells:
            distance = math.sqrt((fx - x) ** 2 + (fy - y) ** 2)
            food_with_distances.append((distance, fx, fy))

        food_with_distances.sort()
        return [(fx, fy) for _, fx, fy in food_with_distances]

    def get_nearby_agents(self, x: int, y: int, radius: int = 2) -> list[Agent]:
        """Get agents within crowding radius."""
        nearby_agents = []
        for agent in self.agents:
            distance = math.sqrt((agent.x - x) ** 2 + (agent.y - y) ** 2)
            if distance <= radius and distance > 0:  # Exclude self
                nearby_agents.append(agent)
        return nearby_agents

    def move_agent(self, agent: Agent, dx: int, dy: int) -> None:
        """Move agent with wraparound."""
        agent.x = (agent.x + dx) % self.grid_size
        agent.y = (agent.y + dy) % self.grid_size

    def eat_food(self, agent: Agent) -> bool:
        """Agent eats food at current position with competition."""
        pos = (agent.x, agent.y)
        if pos in self.food_cells:
            food_cell = self.food_cells[pos]

            # Check for competition (multiple agents at same location)
            competing_agents = [
                a
                for a in self.agents
                if a.x == agent.x and a.y == agent.y and a != agent
            ]
            competition_factor = 1.0 / (1.0 + len(competing_agents) * 0.5)

            # Calculate energy gain based on food quality and competition
            base_energy = 6.0 * food_cell.quality * (1.0 - food_cell.depletion)
            energy_gain = base_energy * competition_factor

            agent.energy += energy_gain

            # Deplete the food cell
            food_cell.depletion = min(1.0, food_cell.depletion + 0.3)
            food_cell.last_consumed = self.tick

            # Remove completely depleted food
            if food_cell.depletion >= 1.0:
                del self.food_cells[pos]

            return True
        return False

    def remove_dead_agents(self) -> list[Agent]:
        """Remove agents with energy <= 0."""
        dead_agents = []
        alive_agents = []

        for agent in self.agents:
            if agent.energy <= 0:
                agent.death_tick = self.tick
                dead_agents.append(agent)
            else:
                alive_agents.append(agent)

        self.agents = alive_agents
        return dead_agents

    def get_reproduction_candidates(self) -> list[Agent]:
        """Get agents ready for reproduction."""
        return [agent for agent in self.agents if agent.energy >= 60.0]


class Simulation:
    """Main simulation controller."""

    def __init__(self, args):
        self.args = args
        self.world = World(args.grid, args.food_regen)
        self.world.workspace = GlobalWorkspace(args.lesion_workspace)

        # Initialize agents
        for _ in range(args.agents):
            self.world.add_agent()

        # Setup logging
        os.makedirs(args.log_dir, exist_ok=True)
        self.agents_file = open(
            os.path.join(args.log_dir, "agents_tick.csv"), "w", newline=""
        )
        self.world_file = open(
            os.path.join(args.log_dir, "world_tick.csv"), "w", newline=""
        )
        self.events_file = open(
            os.path.join(args.log_dir, "events.csv"), "w", newline=""
        )

        # Write CSV headers
        self.agents_writer = csv.writer(self.agents_file)
        self.world_writer = csv.writer(self.world_file)
        self.events_writer = csv.writer(self.events_file)

        self.agents_writer.writerow(
            [
                "tick",
                "agent_id",
                "parent_id",
                "birth_tick",
                "death_tick",
                "x",
                "y",
                "energy",
                "age",
                "health",
                "life_stage",
                "g_food_seek",
                "g_energy_caution",
                "g_exploration",
                "g_memory_strength",
                "g_social_tendency",
                "g_openness",
                "g_conscientiousness",
                "g_extraversion",
                "g_agreeableness",
                "g_neuroticism",
                "emotion_fear",
                "emotion_hope",
                "emotion_frustration",
                "emotion_curiosity",
                "emotion_boredom",
                "motivation_physiological",
                "motivation_safety",
                "motivation_social",
                "motivation_esteem",
                "motivation_self_actualization",
                "skill_pattern_recognition",
                "skill_causal_reasoning",
                "skill_tool_use",
                "skill_communication",
                "skill_innovation",
                "action",
                "score_best",
                "score_second",
                "reported_conf",
                "outcome_reward",
                "workspace_reads",
                "tools_count",
                "messages_sent",
                # Advanced cognitive features
                "working_memory_size",
                "executive_planning",
                "executive_inhibition",
                "executive_switching",
                "executive_monitoring",
                "current_goals_count",
                "mental_models_count",
                "deception_skills",
                "empathy_level",
                "lie_detection",
                "vocabulary_size",
                "language_skills",
                "conversation_history_size",
                "self_esteem",
                "self_efficacy",
                "identity",
                "existential_awareness",
                "mortality_understanding",
                "purpose_seeking",
                "legacy_desire",
                "spiritual_thoughts_count",
                "relationships_count",
                "social_status",
                "reputation",
                "influence_level",
                "creativity_level",
                "innovation_tool_design",
                "innovation_problem_solving",
                "innovation_artistic",
                "creative_works_count",
                # Advanced Social Cognition
                "friendships_count",
                "romantic_partners_count",
                "rivals_count",
                "family_members_count",
                "group_membership_count",
                "leadership_roles_count",
                "dominance_hierarchy",
                "cooperation_level",
                "conflict_resolution_skills",
                "cultural_traditions_count",
                "cultural_norms_count",
                "rituals_performed_count",
                "altruism_level",
                "sacrifice_count",
                "help_given",
                "help_received",
                "social_learning",
                # Deep Self-Awareness
                "self_reflection_history_count",
                "self_improvement_goals_count",
                "self_regulation_emotion_control",
                "self_regulation_impulse_control",
                "self_regulation_attention_control",
                "self_discipline_level",
                "self_monitoring_frequency",
                "self_actualization_progress",
                # Advanced Emotions
                "complex_emotions_love",
                "complex_emotions_jealousy",
                "complex_emotions_pride",
                "complex_emotions_shame",
                "complex_emotions_gratitude",
                "complex_emotions_contempt",
                "complex_emotions_awe",
                "complex_emotions_disgust",
                "emotional_regulation_suppression",
                "emotional_regulation_reappraisal",
                "emotional_regulation_distraction",
                "emotional_regulation_acceptance",
                "empathy_cognitive",
                "empathy_emotional",
                "empathy_compassionate",
                "emotional_intelligence",
                "mood_disorders_depression",
                "mood_disorders_anxiety",
                "mood_disorders_mania",
                "emotional_memory_count",
                # Scientific Thinking
                "hypotheses_count",
                "experiments_count",
                "causal_reasoning_correlation",
                "causal_reasoning_causation",
                "causal_reasoning_counterfactual",
                "abstraction_level",
                "logical_reasoning_deductive",
                "logical_reasoning_inductive",
                "logical_reasoning_abductive",
                "mathematical_skills_arithmetic",
                "mathematical_skills_geometry",
                "mathematical_skills_statistics",
                "theory_building",
                # Creative Expression
                "artistic_works_count",
                "stories_told_count",
                "jokes_made_count",
                "humor_level",
                "imagination_capacity",
                "aesthetic_appreciation",
                "cultural_artifacts_count",
                "creative_collaborations_count",
                # Physical Reality (Phase 13)
                "z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "mass",
                "temperature",
                "pressure",
                "chemical_oxygen",
                "chemical_carbon",
                "chemical_hydrogen",
                "chemical_nitrogen",
                "biological_metabolism",
                "biological_growth",
                "biological_reproduction",
                "biological_healing",
                # Evolutionary Biology (Phase 14)
                "dna_sequence_length",
                "genes_count",
                "chromosomes",
                "mutations_count",
                "fitness_score",
                "species_id",
                "generation",
                # Civilization & Culture (Phase 15)
                "city_id",
                "citizenship_status",
                "government_role",
                "legal_status",
                "economic_class",
                "religious_beliefs_count",
                "social_institutions_count",
                # Technology & Innovation (Phase 16)
                "technology_fire",
                "technology_tools",
                "technology_agriculture",
                "technology_writing",
                "technology_metallurgy",
                "research_projects_count",
                "innovation_network_count",
                "patents_count",
                "scientific_mathematics",
                "scientific_physics",
                "scientific_chemistry",
                "scientific_biology",
                "scientific_medicine",
                # Information & Communication (Phase 17)
                "information_storage_count",
                "communication_network_count",
                "media_news",
                "media_entertainment",
                "media_education",
                "media_propaganda",
                "information_analysis",
                "information_synthesis",
                "information_interpretation",
                "knowledge_base_count",
                "information_security",
                # Psychological Reality (Phase 18)
                "mental_health_status",
                "therapy_sessions_count",
                "psychological_development_stage",
                "cognitive_biases_count",
                "personality_disorders_count",
                "social_conformity",
                "social_obedience",
                "social_group_think",
                # Ecological Systems (Phase 19)
                "ecological_role",
                "food_web_position",
                "ecosystem_services_count",
                "environmental_impact",
                "biodiversity_contribution",
                "climate_adaptation",
                # Health & Medicine (Phase 20)
                "diseases_count",
                "medical_treatments_count",
                "healthcare_access",
                "medical_insurance",
                "public_health_status",
                "vaccination_status_count",
                # Economic Systems (Phase 21)
                "wealth",
                "income",
                "expenses",
                "trade_relationships_count",
                "market_buyer",
                "market_seller",
                "market_investor",
                "economic_boom",
                "economic_bust",
                "economic_recovery",
                # Education & Learning (Phase 22)
                "education_level",
                "educational_institution",
                "learning_lecture",
                "learning_practice",
                "learning_research",
                "learning_mentoring",
                "knowledge_transfer_count",
                "educational_testing",
                "educational_evaluation",
                "educational_certification",
                "lifelong_learning",
            ]
        )

        self.world_writer.writerow(
            ["tick", "num_agents", "avg_energy", "food_cells", "lesion_workspace"]
        )

        self.events_writer.writerow(["tick", "type", "agent_id", "payload"])

    def close_logs(self):
        """Close log files."""
        self.agents_file.close()
        self.world_file.close()
        self.events_file.close()

    def run(self):
        """Run the simulation."""
        print(
            f"Starting simulation: {self.args.ticks} ticks, {self.args.grid}x{self.args.grid} grid, {self.args.agents} agents"
        )
        if self.args.lesion_workspace:
            print("LESION MODE: Workspace disabled")

        for tick in range(self.args.ticks):
            self.world.tick = tick
            self.simulate_tick()

        self.close_logs()
        self.print_summary()

    def simulate_tick(self):
        """Simulate one tick of the world."""
        # Update environmental conditions
        self.update_environment()

        # Regenerate food
        self.world.regenerate_food()

        # Broadcast nearest food to workspace
        if self.world.agents:
            # Use first agent's position for global food broadcast (simplified)
            nearest_food = self.world.get_nearest_food(
                self.world.agents[0].x, self.world.agents[0].y
            )
            self.world.workspace.broadcast_nearest_food(nearest_food)

        # Process each agent
        for agent in self.world.agents:
            self.process_agent(agent)

        # Handle reproduction
        self.handle_reproduction()

        # Remove dead agents
        dead_agents = self.world.remove_dead_agents()
        for agent in dead_agents:
            self.events_writer.writerow(
                [self.world.tick, "death", agent.agent_id, f"energy={agent.energy}"]
            )

        # Log world state
        avg_energy = (
            sum(a.energy for a in self.world.agents) / len(self.world.agents)
            if self.world.agents
            else 0
        )
        self.world_writer.writerow(
            [
                self.world.tick,
                len(self.world.agents),
                avg_energy,
                len(self.world.food_cells),
                self.args.lesion_workspace,
            ]
        )

    def process_agent(self, agent: Agent):
        """Process one agent for one tick."""
        # Age and health effects
        agent.age += 1
        agent.health = max(0.1, agent.health - 0.001)  # Gradual health decline

        # Update life stage
        self.update_life_stage(agent)

        # Update emotional state
        self.update_emotions(agent)

        # Update motivational state (Maslow's hierarchy)
        self.update_motivation(agent)

        # Update skills through practice
        self.update_skills(agent)

        # Advanced cognitive processing
        self.update_working_memory(agent)
        self.update_executive_control(agent)
        self.update_theory_of_mind(agent)
        self.update_self_awareness(agent)
        self.update_existential_awareness(agent)

        # Phase 6: Advanced Social Cognition
        self.update_social_cognition(agent)
        self.update_complex_relationships(agent)
        self.update_group_dynamics(agent)
        self.update_cultural_evolution(agent)
        self.update_altruism(agent)

        # Phase 7: Deep Self-Awareness
        self.update_deep_self_awareness(agent)
        self.update_self_regulation(agent)
        self.update_self_improvement(agent)

        # Phase 8: Advanced Emotions
        self.update_advanced_emotions(agent)
        self.update_emotional_regulation(agent)
        self.update_empathy(agent)

        # Phase 9: Scientific Thinking
        self.update_scientific_thinking(agent)
        self.update_causal_reasoning(agent)
        self.update_logical_reasoning(agent)

        # Phase 10: Creative Expression
        self.update_creative_expression(agent)
        self.update_artistic_creation(agent)
        self.update_storytelling(agent)

        # Phase 13: Physical Reality
        self.update_physical_reality(agent)
        self.update_physics_engine(agent)
        self.update_chemical_reactions(agent)
        self.update_biological_processes(agent)

        # Phase 14: Evolutionary Biology
        self.update_evolutionary_biology(agent)
        self.update_dna_evolution(agent)
        self.update_natural_selection(agent)

        # Phase 15: Civilization & Culture
        self.update_civilization_culture(agent)
        self.update_government_systems(agent)
        self.update_legal_systems(agent)
        self.update_religious_systems(agent)

        # Phase 16: Technology & Innovation
        self.update_technology_innovation(agent)
        self.update_research_projects(agent)
        self.update_innovation_networks(agent)

        # Phase 17: Information & Communication
        self.update_information_communication(agent)
        self.update_media_systems(agent)
        self.update_knowledge_management(agent)

        # Phase 18: Psychological Reality
        self.update_psychological_reality(agent)
        self.update_mental_health(agent)
        self.update_cognitive_biases(agent)

        # Phase 19: Ecological Systems
        self.update_ecological_systems(agent)
        self.update_food_webs(agent)
        self.update_ecosystem_dynamics(agent)

        # Phase 20: Health & Medicine
        self.update_health_medicine(agent)
        self.update_disease_systems(agent)
        self.update_healthcare_systems(agent)

        # Phase 21: Economic Systems
        self.update_economic_systems(agent)
        self.update_trade_networks(agent)
        self.update_market_systems(agent)

        # Phase 22: Education & Learning
        self.update_education_learning(agent)
        self.update_educational_institutions(agent)
        self.update_knowledge_transfer(agent)

        # Phase 23: Cosmic & Astronomical Reality
        self.update_cosmic_astronomical(agent)
        self.update_orbital_mechanics(agent)
        self.update_space_exploration(agent)

        # Phase 24: Neuroscience & Brain Simulation
        self.update_neuroscience_brain(agent)
        self.update_neural_networks(agent)
        self.update_brain_plasticity(agent)

        # Phase 25: Quantum Mechanics & Physics
        self.update_quantum_mechanics(agent)
        self.update_quantum_computing(agent)
        self.update_quantum_consciousness(agent)

        # Phase 26: Climate & Weather Systems
        self.update_climate_weather(agent)
        self.update_atmospheric_dynamics(agent)
        self.update_natural_disasters(agent)

        # Phase 27: Molecular & Cellular Biology
        self.update_molecular_cellular(agent)
        self.update_protein_synthesis(agent)
        self.update_cell_division(agent)

        # Phase 28: Internet & Digital Reality
        self.update_internet_digital(agent)
        self.update_social_media(agent)
        self.update_virtual_worlds(agent)

        # Phase 29: Art & Cultural Expression
        self.update_art_cultural(agent)
        self.update_cultural_movements(agent)
        self.update_aesthetic_evolution(agent)

        # Phase 30: Political Systems & Governance
        self.update_political_governance(agent)
        self.update_elections_voting(agent)
        self.update_policy_making(agent)

        # Phase 31: Biotechnology & Genetic Engineering
        self.update_biotechnology_genetic(agent)
        self.update_crispr_gene_editing(agent)
        self.update_synthetic_biology(agent)

        # Phase 32: Artificial Intelligence & Machine Learning
        self.update_ai_machine_learning(agent)
        self.update_human_ai_interaction(agent)
        self.update_agi_development(agent)

        # Environmental effects
        self.apply_environmental_effects(agent)

        # Energy maintenance (affected by age, health, and environment)
        base_maintenance = 0.5
        age_factor = 1.0 + (agent.age / 1000.0)  # Older agents need more energy
        health_factor = 2.0 - agent.health  # Unhealthy agents need more energy

        # Environmental factors
        terrain_type = self.world.terrain.get((agent.x, agent.y), "plains")
        terrain_factor = {"plains": 1.0, "forest": 1.2, "mountain": 1.5, "water": 0.8}[
            terrain_type
        ]
        weather_factor = {"clear": 1.0, "rain": 1.3, "storm": 1.8, "fog": 1.1}[
            self.world.weather
        ]
        day_night_factor = {"day": 1.0, "night": 1.4}[self.world.day_night]

        agent.energy -= (
            base_maintenance
            * age_factor
            * health_factor
            * terrain_factor
            * weather_factor
            * day_night_factor
        )

        # Update memory (spatial and social)
        self.update_agent_memory(agent)

        # Get local perception
        nearby_food = self.world.get_food_in_radius(agent.x, agent.y, 3)
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)

        # Read from workspace
        workspace_food = self.world.workspace.read("nearest_food")
        if workspace_food is not None:
            agent.workspace_reads += 1

        # Broadcast agent info
        self.world.workspace.broadcast_agent_info(agent)

        # Decision making with advanced metacognition
        action, score_best, score_second, reported_conf = self.make_decision(
            agent, nearby_food, workspace_food, nearby_agents
        )

        # Update confidence history for calibration learning
        agent.confidence_history.append(reported_conf)
        if len(agent.confidence_history) > 100:  # Keep only recent history
            agent.confidence_history.pop(0)

        # Execute action
        outcome_reward = self.execute_action(agent, action)

        # Log agent state
        self.agents_writer.writerow(
            [
                self.world.tick,
                agent.agent_id,
                agent.parent_id,
                agent.birth_tick,
                agent.death_tick,
                agent.x,
                agent.y,
                agent.energy,
                agent.age,
                agent.health,
                agent.life_stage,
                agent.genome[0],  # food_seek
                agent.genome[1],  # energy_caution
                agent.genome[2],  # exploration
                agent.genome[3],  # memory_strength
                agent.genome[4],  # social_tendency
                agent.genome[5],  # openness
                agent.genome[6],  # conscientiousness
                agent.genome[7],  # extraversion
                agent.genome[8],  # agreeableness
                agent.genome[9],  # neuroticism
                agent.emotions["fear"],
                agent.emotions["hope"],
                agent.emotions["frustration"],
                agent.emotions["curiosity"],
                agent.emotions["boredom"],
                agent.motivation["physiological"],
                agent.motivation["safety"],
                agent.motivation["social"],
                agent.motivation["esteem"],
                agent.motivation["self_actualization"],
                agent.skills["pattern_recognition"],
                agent.skills["causal_reasoning"],
                agent.skills["tool_use"],
                agent.skills["communication"],
                agent.skills["innovation"],
                action,
                score_best,
                score_second,
                reported_conf,
                outcome_reward,
                agent.workspace_reads,
                len(agent.tools),
                agent.messages_sent,
                # Advanced cognitive features
                len(agent.working_memory),
                agent.executive_functions["planning"],
                agent.executive_functions["inhibition"],
                agent.executive_functions["switching"],
                agent.executive_functions["monitoring"],
                len(agent.current_goals),
                len(agent.mental_models),
                agent.deception_skills,
                agent.empathy_level,
                agent.lie_detection,
                len(agent.vocabulary),
                agent.language_skills,
                len(agent.conversation_history),
                agent.self_esteem,
                agent.self_efficacy,
                agent.identity,
                agent.existential_awareness,
                agent.mortality_understanding,
                agent.purpose_seeking,
                agent.legacy_desire,
                len(agent.spiritual_thoughts),
                len(agent.relationships),
                agent.social_status,
                agent.reputation,
                agent.influence_level,
                agent.creativity_level,
                agent.innovation_skills["tool_design"],
                agent.innovation_skills["problem_solving"],
                agent.innovation_skills["artistic"],
                len(agent.creative_works),
                # Advanced Social Cognition
                len(agent.friendships),
                len(agent.romantic_partners),
                len(agent.rivals),
                len(agent.family_members),
                len(agent.group_membership),
                len(agent.leadership_roles),
                agent.dominance_hierarchy,
                agent.cooperation_level,
                agent.conflict_resolution_skills,
                len(agent.cultural_traditions),
                len(agent.cultural_norms),
                len(agent.rituals_performed),
                agent.altruism_level,
                agent.sacrifice_count,
                agent.help_given,
                agent.help_received,
                agent.social_learning,
                # Deep Self-Awareness
                len(agent.self_reflection_history),
                len(agent.self_improvement_goals),
                agent.self_regulation_skills["emotion_control"],
                agent.self_regulation_skills["impulse_control"],
                agent.self_regulation_skills["attention_control"],
                agent.self_discipline_level,
                agent.self_monitoring_frequency,
                agent.self_actualization_progress,
                # Advanced Emotions
                agent.complex_emotions["love"],
                agent.complex_emotions["jealousy"],
                agent.complex_emotions["pride"],
                agent.complex_emotions["shame"],
                agent.complex_emotions["gratitude"],
                agent.complex_emotions["contempt"],
                agent.complex_emotions["awe"],
                agent.complex_emotions["disgust"],
                agent.emotional_regulation["suppression"],
                agent.emotional_regulation["reappraisal"],
                agent.emotional_regulation["distraction"],
                agent.emotional_regulation["acceptance"],
                agent.empathy_levels["cognitive"],
                agent.empathy_levels["emotional"],
                agent.empathy_levels["compassionate"],
                agent.emotional_intelligence,
                agent.mood_disorders["depression"],
                agent.mood_disorders["anxiety"],
                agent.mood_disorders["mania"],
                len(agent.emotional_memory),
                # Scientific Thinking
                len(agent.hypotheses),
                len(agent.experiments),
                agent.causal_reasoning_skills["correlation"],
                agent.causal_reasoning_skills["causation"],
                agent.causal_reasoning_skills["counterfactual"],
                agent.abstraction_level,
                agent.logical_reasoning["deductive"],
                agent.logical_reasoning["inductive"],
                agent.logical_reasoning["abductive"],
                agent.mathematical_skills["arithmetic"],
                agent.mathematical_skills["geometry"],
                agent.mathematical_skills["statistics"],
                agent.theory_building,
                # Creative Expression
                len(agent.artistic_works),
                len(agent.stories_told),
                len(agent.jokes_made),
                agent.humor_level,
                agent.imagination_capacity,
                agent.aesthetic_appreciation,
                len(agent.cultural_artifacts),
                len(agent.creative_collaborations),
                # Physical Reality (Phase 13)
                agent.z,
                agent.velocity[0],
                agent.velocity[1],
                agent.velocity[2],
                agent.mass,
                agent.temperature,
                agent.pressure,
                agent.chemical_composition["oxygen"],
                agent.chemical_composition["carbon"],
                agent.chemical_composition["hydrogen"],
                agent.chemical_composition["nitrogen"],
                agent.biological_processes["metabolism"],
                agent.biological_processes["growth"],
                agent.biological_processes["reproduction"],
                agent.biological_processes["healing"],
                # Evolutionary Biology (Phase 14)
                len(agent.dna_sequence),
                len(agent.genes),
                agent.chromosomes,
                len(agent.mutations),
                agent.fitness_score,
                agent.species_id,
                agent.generation,
                # Civilization & Culture (Phase 15)
                agent.city_id or 0,
                agent.citizenship_status,
                agent.government_role,
                agent.legal_status,
                agent.economic_class,
                len(agent.religious_beliefs),
                len(agent.social_institutions),
                # Technology & Innovation (Phase 16)
                agent.technology_tree["fire"],
                agent.technology_tree["tools"],
                agent.technology_tree["agriculture"],
                agent.technology_tree["writing"],
                agent.technology_tree["metallurgy"],
                len(agent.research_projects),
                len(agent.innovation_network),
                len(agent.patents),
                agent.scientific_knowledge["mathematics"],
                agent.scientific_knowledge["physics"],
                agent.scientific_knowledge["chemistry"],
                agent.scientific_knowledge["biology"],
                agent.scientific_knowledge["medicine"],
                # Information & Communication (Phase 17)
                len(agent.information_storage),
                len(agent.communication_network),
                agent.media_consumption["news"],
                agent.media_consumption["entertainment"],
                agent.media_consumption["education"],
                agent.media_consumption["propaganda"],
                agent.information_processing["analysis"],
                agent.information_processing["synthesis"],
                agent.information_processing["interpretation"],
                len(agent.knowledge_base),
                agent.information_security,
                # Psychological Reality (Phase 18)
                agent.mental_health_status,
                len(agent.therapy_sessions),
                agent.psychological_development_stage,
                len(agent.cognitive_biases),
                len(agent.personality_disorders),
                agent.social_psychology_traits["conformity"],
                agent.social_psychology_traits["obedience"],
                agent.social_psychology_traits["group_think"],
                # Ecological Systems (Phase 19)
                agent.ecological_role,
                agent.food_web_position,
                len(agent.ecosystem_services),
                agent.environmental_impact,
                agent.biodiversity_contribution,
                agent.climate_adaptation,
                # Health & Medicine (Phase 20)
                len(agent.diseases),
                len(agent.medical_treatments),
                agent.healthcare_access,
                agent.medical_insurance,
                agent.public_health_status,
                len(agent.vaccination_status),
                # Economic Systems (Phase 21)
                agent.wealth,
                agent.income,
                agent.expenses,
                len(agent.trade_relationships),
                agent.market_participation["buyer"],
                agent.market_participation["seller"],
                agent.market_participation["investor"],
                agent.economic_cycles["boom"],
                agent.economic_cycles["bust"],
                agent.economic_cycles["recovery"],
                # Education & Learning (Phase 22)
                agent.education_level,
                agent.educational_institution or 0,
                agent.learning_methods["lecture"],
                agent.learning_methods["practice"],
                agent.learning_methods["research"],
                agent.learning_methods["mentoring"],
                len(agent.knowledge_transfer),
                agent.educational_assessment["testing"],
                agent.educational_assessment["evaluation"],
                agent.educational_assessment["certification"],
                agent.lifelong_learning,
            ]
        )

    def update_agent_memory(self, agent: Agent):
        """Update agent's spatial and social memory."""
        memory_strength = agent.genome[3]

        # Spatial memory: remember food locations
        current_pos = (agent.x, agent.y)
        if current_pos not in agent.memory:
            agent.memory[current_pos] = {
                "food_found": 0,
                "last_visited": self.world.tick,
            }

        # Update food memory based on current perception
        nearby_food = self.world.get_food_in_radius(agent.x, agent.y, 3)
        if nearby_food:
            agent.memory[current_pos]["food_found"] += memory_strength
            agent.memory[current_pos]["last_visited"] = self.world.tick

        # Social memory: remember other agents
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)
        for other_agent in nearby_agents:
            other_id = other_agent.agent_id
            if other_id not in agent.memory:
                agent.memory[other_id] = {
                    "interactions": 0,
                    "trust": 0.5,
                    "last_seen": self.world.tick,
                }

            agent.memory[other_id]["interactions"] += 1
            agent.memory[other_id]["last_seen"] = self.world.tick

            # Update trust based on energy levels (successful agents are more trusted)
            if other_agent.energy > agent.energy:
                agent.memory[other_id]["trust"] = min(
                    1.0, agent.memory[other_id]["trust"] + 0.01
                )
            else:
                agent.memory[other_id]["trust"] = max(
                    0.0, agent.memory[other_id]["trust"] - 0.005
                )

        # Forget old memories
        forget_threshold = self.world.tick - 100
        to_forget = []
        for key, value in agent.memory.items():
            if isinstance(value, dict) and value.get("last_seen", 0) < forget_threshold:
                to_forget.append(key)

        for key in to_forget:
            del agent.memory[key]

    def update_life_stage(self, agent: Agent):
        """Update agent's life stage based on age."""
        if agent.age < 50:
            agent.life_stage = "childhood"
        elif agent.age < 150:
            agent.life_stage = "adolescence"
        elif agent.age < 500:
            agent.life_stage = "adulthood"
        else:
            agent.life_stage = "old_age"

    def update_emotions(self, agent: Agent):
        """Update agent's emotional state."""
        # Fear: increases when low energy or in dangerous situations
        if agent.energy < 20:
            agent.emotions["fear"] = min(1.0, agent.emotions["fear"] + 0.1)
        else:
            agent.emotions["fear"] = max(0.0, agent.emotions["fear"] - 0.05)

        # Hope: increases when finding food or making progress
        if agent.energy > 50:
            agent.emotions["hope"] = min(1.0, agent.emotions["hope"] + 0.05)
        else:
            agent.emotions["hope"] = max(0.0, agent.emotions["hope"] - 0.02)

        # Frustration: increases when repeatedly failing
        if len(agent.confidence_history) > 5:
            recent_failures = sum(
                1 for conf in agent.confidence_history[-5:] if conf < 0.3
            )
            if recent_failures > 3:
                agent.emotions["frustration"] = min(
                    1.0, agent.emotions["frustration"] + 0.1
                )
            else:
                agent.emotions["frustration"] = max(
                    0.0, agent.emotions["frustration"] - 0.05
                )

        # Curiosity: drives exploration
        if agent.personality["openness"] > 0.6:
            agent.emotions["curiosity"] = min(1.0, agent.emotions["curiosity"] + 0.02)

        # Boredom: increases when environment is too predictable
        if len(agent.memory) > 20:  # Been around long enough to get bored
            agent.emotions["boredom"] = min(1.0, agent.emotions["boredom"] + 0.01)

    def update_motivation(self, agent: Agent):
        """Update agent's motivational state (Maslow's hierarchy)."""
        # Physiological needs (food, water, shelter)
        if agent.energy < 30:
            agent.motivation["physiological"] = 1.0
        else:
            agent.motivation["physiological"] = max(
                0.3, agent.motivation["physiological"] - 0.01
            )

        # Safety needs
        if agent.emotions["fear"] > 0.5:
            agent.motivation["safety"] = 1.0
        else:
            agent.motivation["safety"] = max(0.2, agent.motivation["safety"] - 0.005)

        # Social needs
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 3)
        if len(nearby_agents) == 0:
            agent.motivation["social"] = min(1.0, agent.motivation["social"] + 0.05)
        else:
            agent.motivation["social"] = max(0.1, agent.motivation["social"] - 0.01)

        # Esteem needs
        if agent.energy > 60 and len(nearby_agents) > 2:
            agent.motivation["esteem"] = min(1.0, agent.motivation["esteem"] + 0.02)

        # Self-actualization
        if all(mot > 0.7 for mot in agent.motivation.values()):
            agent.motivation["self_actualization"] = min(
                1.0, agent.motivation["self_actualization"] + 0.01
            )

    def update_skills(self, agent: Agent):
        """Update agent's skills through practice and experience."""
        # Pattern recognition improves with experience
        if len(agent.memory) > 10:
            agent.skills["pattern_recognition"] = min(
                1.0, agent.skills["pattern_recognition"] + 0.001
            )

        # Causal reasoning improves with age
        if agent.age > 100:
            agent.skills["causal_reasoning"] = min(
                1.0, agent.skills["causal_reasoning"] + 0.0005
            )

        # Tool use improves with practice
        if len(agent.tools) > 0:
            agent.skills["tool_use"] = min(1.0, agent.skills["tool_use"] + 0.002)

        # Communication improves with social interaction
        if agent.messages_sent > 0:
            agent.skills["communication"] = min(
                1.0, agent.skills["communication"] + 0.001
            )

        # Innovation improves with exploration
        if agent.personality["openness"] > 0.7:
            agent.skills["innovation"] = min(1.0, agent.skills["innovation"] + 0.0005)

    def apply_environmental_effects(self, agent: Agent):
        """Apply environmental effects to agent."""
        # Weather effects
        if self.world.weather == "rain":
            agent.health = max(0.1, agent.health - 0.002)  # Rain makes you sick
        elif self.world.weather == "storm":
            agent.health = max(0.1, agent.health - 0.005)  # Storms are dangerous
            agent.emotions["fear"] = min(1.0, agent.emotions["fear"] + 0.1)

        # Day/night effects
        if self.world.day_night == "night":
            agent.emotions["fear"] = min(1.0, agent.emotions["fear"] + 0.05)

        # Shelter effects
        if (agent.x, agent.y) in self.world.shelters:
            shelter = self.world.shelters[(agent.x, agent.y)]
            agent.health = min(1.0, agent.health + shelter["safety_level"] * 0.01)
            agent.emotions["fear"] = max(0.0, agent.emotions["fear"] - 0.1)

        # Resource effects
        if (agent.x, agent.y) in self.world.resources:
            resources = self.world.resources[(agent.x, agent.y)]
            if "water" in resources:
                agent.health = min(1.0, agent.health + 0.01)
            if "tools" in resources and len(agent.tools) < 3:
                agent.tools.append(f"tool_{len(agent.tools)}")

    def update_working_memory(self, agent: Agent):
        """Update working memory system."""
        # Add current perceptions to working memory
        current_perceptions = {
            "tick": self.world.tick,
            "energy": agent.energy,
            "position": (agent.x, agent.y),
            "nearby_food": len(self.world.get_food_in_radius(agent.x, agent.y, 3)),
            "nearby_agents": len(self.world.get_nearby_agents(agent.x, agent.y, 2)),
        }

        agent.working_memory.append(current_perceptions)

        # Maintain capacity limit (Miller's rule: 7±2)
        if len(agent.working_memory) > agent.memory_capacity:
            agent.working_memory.pop(0)

        # Decay old memories
        for memory_item in agent.working_memory:
            if "decay" not in memory_item:
                memory_item["decay"] = 1.0
            memory_item["decay"] *= 0.95  # Decay over time

        # Remove fully decayed memories
        agent.working_memory = [
            item for item in agent.working_memory if item["decay"] > 0.1
        ]

    def update_executive_control(self, agent: Agent):
        """Update executive control functions."""
        # Planning: Set goals based on current needs
        if not agent.current_goals:
            if agent.energy < 30:
                agent.current_goals.append(
                    {"type": "find_food", "priority": 1.0, "steps": []}
                )
            elif agent.emotions["fear"] > 0.7:
                agent.current_goals.append(
                    {"type": "find_safety", "priority": 0.8, "steps": []}
                )
            elif agent.motivation["social"] > 0.8:
                agent.current_goals.append(
                    {"type": "find_company", "priority": 0.6, "steps": []}
                )

        # Inhibition: Suppress automatic responses when needed
        if agent.executive_functions["inhibition"] > 0.7:
            # High inhibition = more careful decisions
            agent.emotions["fear"] = min(1.0, agent.emotions["fear"] + 0.1)

        # Monitoring: Track progress toward goals
        for goal in agent.current_goals:
            if goal["type"] == "find_food" and agent.energy > 50:
                goal["completed"] = True
            elif goal["type"] == "find_safety" and agent.emotions["fear"] < 0.3:
                goal["completed"] = True

        # Remove completed goals
        agent.current_goals = [
            goal for goal in agent.current_goals if not goal.get("completed", False)
        ]

    def update_theory_of_mind(self, agent: Agent):
        """Update theory of mind - understanding others' mental states."""
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 3)

        for other_agent in nearby_agents:
            other_id = other_agent.agent_id

            if other_id not in agent.mental_models:
                agent.mental_models[other_id] = {
                    "beliefs": {},
                    "desires": {},
                    "intentions": {},
                    "trust_level": 0.5,
                    "last_updated": self.world.tick,
                }

            mental_model = agent.mental_models[other_id]

            # Update beliefs about other agent
            mental_model["beliefs"]["energy"] = other_agent.energy
            mental_model["beliefs"]["position"] = (other_agent.x, other_agent.y)
            mental_model["beliefs"]["fear"] = other_agent.emotions["fear"]

            # Infer desires based on behavior
            if other_agent.energy < 30:
                mental_model["desires"]["food"] = 1.0
            if other_agent.emotions["fear"] > 0.7:
                mental_model["desires"]["safety"] = 1.0

            # Predict intentions
            if mental_model["desires"].get("food", 0) > 0.8:
                mental_model["intentions"]["seek_food"] = 0.9
            if mental_model["desires"].get("safety", 0) > 0.8:
                mental_model["intentions"]["seek_shelter"] = 0.9

            mental_model["last_updated"] = self.world.tick

        # Deception: Manipulate others' beliefs
        if agent.deception_skills > 0.7 and random.random() < 0.1:
            # Sometimes lie about food locations
            agent.deception_count += 1

    def update_self_awareness(self, agent: Agent):
        """Update self-awareness and identity."""
        # Update self-concept based on experiences
        if agent.energy > 60:
            agent.self_concept["traits"].append("successful")
        if agent.emotions["fear"] < 0.3:
            agent.self_concept["traits"].append("brave")
        if len(agent.tools) > 2:
            agent.self_concept["traits"].append("resourceful")

        # Update self-esteem based on recent performance
        recent_successes = sum(
            1 for conf in agent.confidence_history[-10:] if conf > 0.7
        )
        if recent_successes > 5:
            agent.self_esteem = min(1.0, agent.self_esteem + 0.05)
        else:
            agent.self_esteem = max(0.0, agent.self_esteem - 0.02)

        # Update self-efficacy based on skill development
        skill_improvement = sum(agent.skills.values()) - 2.5  # Baseline is 2.5
        agent.self_efficacy = max(0.0, min(1.0, 0.5 + skill_improvement * 0.1))

        # Develop identity
        if agent.age > 50 and agent.identity == "unknown":
            traits = agent.self_concept["traits"]
            if "successful" in traits and "brave" in traits:
                agent.identity = "leader"
            elif "resourceful" in traits:
                agent.identity = "craftsperson"
            else:
                agent.identity = "survivor"

    def update_existential_awareness(self, agent: Agent):
        """Update existential awareness and mortality understanding."""
        # Increase mortality understanding with age
        if agent.age > 100:
            agent.mortality_understanding = min(
                1.0, agent.mortality_understanding + 0.01
            )

        # Increase existential awareness with experience
        if len(agent.memory) > 50:
            agent.existential_awareness = min(1.0, agent.existential_awareness + 0.005)

        # Purpose seeking based on motivation
        if agent.motivation["self_actualization"] > 0.8:
            agent.purpose_seeking = min(1.0, agent.purpose_seeking + 0.01)

        # Legacy desire increases with age
        if agent.age > 200:
            agent.legacy_desire = min(1.0, agent.legacy_desire + 0.01)

        # Spiritual thoughts emerge with existential awareness
        if agent.existential_awareness > 0.7 and random.random() < 0.05:
            thoughts = [
                "What is the meaning of existence?",
                "Why am I here?",
                "What happens after death?",
                "Is there something greater than myself?",
                "What is my purpose in life?",
            ]
            agent.spiritual_thoughts.append(random.choice(thoughts))

    def update_social_cognition(self, agent: Agent):
        """Update advanced social cognition."""
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 3)

        # Update social status based on interactions
        if len(nearby_agents) > 3:
            agent.social_status = (
                "leader" if agent.influence_level > 0.7 else "follower"
            )

        # Update reputation based on behavior
        if agent.energy > 60:
            agent.reputation = min(1.0, agent.reputation + 0.01)
        elif agent.energy < 20:
            agent.reputation = max(0.0, agent.reputation - 0.01)

        # Update influence level
        agent.influence_level = (
            agent.reputation + agent.self_esteem + agent.skills["communication"]
        ) / 3.0

    def update_complex_relationships(self, agent: Agent):
        """Update complex relationships (friendship, love, rivalry)."""
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)

        for other_agent in nearby_agents:
            other_id = other_agent.agent_id

            # Friendship development
            if other_id not in agent.friendships:
                agent.friendships[other_id] = 0.1
            else:
                # Increase friendship through positive interactions
                if agent.energy > 50 and other_agent.energy > 50:
                    agent.friendships[other_id] = min(
                        1.0, agent.friendships[other_id] + 0.05
                    )

            # Romantic relationships
            if agent.friendships.get(other_id, 0) > 0.8 and random.random() < 0.1:
                agent.romantic_partners[other_id] = min(
                    1.0, agent.romantic_partners.get(other_id, 0) + 0.1
                )

            # Rivalry development
            if agent.energy < 30 and other_agent.energy > 70:
                agent.rivals[other_id] = min(1.0, agent.rivals.get(other_id, 0) + 0.05)

    def update_group_dynamics(self, agent: Agent):
        """Update group dynamics and leadership."""
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 3)

        if len(nearby_agents) > 2:
            # Determine dominance hierarchy
            dominance_score = (
                agent.energy * 0.5
                + agent.age * 0.3
                + agent.skills["communication"] * 0.2
            )
            agent.dominance_hierarchy = int(dominance_score * 10)

            # Leadership roles
            if agent.dominance_hierarchy > 7:
                agent.leadership_roles.append("alpha")
            elif agent.dominance_hierarchy > 5:
                agent.leadership_roles.append("beta")

            # Cooperation level
            if len(nearby_agents) > 3:
                agent.cooperation_level = min(1.0, agent.cooperation_level + 0.01)

    def update_cultural_evolution(self, agent: Agent):
        """Update cultural evolution and traditions."""
        # Learn from nearby agents
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)

        for other_agent in nearby_agents:
            # Learn cultural traditions
            for tradition in other_agent.cultural_traditions:
                if tradition not in agent.cultural_traditions and random.random() < 0.1:
                    agent.cultural_traditions.append(tradition)

            # Learn cultural norms
            for norm, value in other_agent.cultural_norms.items():
                if norm not in agent.cultural_norms:
                    agent.cultural_norms[norm] = value * 0.8  # Partial learning

        # Perform rituals
        if random.random() < 0.05:
            rituals = ["greeting", "farewell", "celebration", "mourning", "blessing"]
            agent.rituals_performed.append(random.choice(rituals))

    def update_altruism(self, agent: Agent):
        """Update altruistic behavior."""
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)

        # Help others in need
        for other_agent in nearby_agents:
            if other_agent.energy < 20 and agent.energy > 40:
                if random.random() < agent.altruism_level:
                    agent.help_given += 1
                    other_agent.help_received += 1
                    agent.energy -= 10
                    other_agent.energy += 10

        # Social learning
        if len(nearby_agents) > 0:
            agent.social_learning = min(1.0, agent.social_learning + 0.01)

    def update_deep_self_awareness(self, agent: Agent):
        """Update deep self-awareness and reflection."""
        # Self-reflection
        if random.random() < agent.self_monitoring_frequency:
            reflection = {
                "tick": self.world.tick,
                "energy": agent.energy,
                "emotions": agent.emotions.copy(),
                "thoughts": f"Reflecting on my current state at tick {self.world.tick}",
            }
            agent.self_reflection_history.append(reflection)

            # Keep only recent reflections
            if len(agent.self_reflection_history) > 10:
                agent.self_reflection_history.pop(0)

        # Self-improvement goals
        if not agent.self_improvement_goals:
            if agent.energy < 40:
                agent.self_improvement_goals.append(
                    {"type": "increase_energy", "target": 60, "progress": 0}
                )
            if agent.skills["pattern_recognition"] < 0.7:
                agent.self_improvement_goals.append(
                    {
                        "type": "improve_pattern_recognition",
                        "target": 0.8,
                        "progress": 0,
                    }
                )

        # Update self-actualization progress
        skill_sum = sum(agent.skills.values())
        agent.self_actualization_progress = min(1.0, skill_sum / 5.0)

    def update_self_regulation(self, agent: Agent):
        """Update self-regulation skills."""
        # Emotion control
        if agent.emotions["fear"] > 0.8:
            agent.self_regulation_skills["emotion_control"] = min(
                1.0, agent.self_regulation_skills["emotion_control"] + 0.01
            )

        # Impulse control
        if agent.energy < 30 and random.random() < 0.5:
            agent.self_regulation_skills["impulse_control"] = min(
                1.0, agent.self_regulation_skills["impulse_control"] + 0.01
            )

        # Attention control
        if len(agent.working_memory) > 5:
            agent.self_regulation_skills["attention_control"] = min(
                1.0, agent.self_regulation_skills["attention_control"] + 0.01
            )

        # Self-discipline
        agent.self_discipline_level = sum(agent.self_regulation_skills.values()) / 3.0

    def update_self_improvement(self, agent: Agent):
        """Update self-improvement efforts."""
        for goal in agent.self_improvement_goals:
            if goal["type"] == "increase_energy" and agent.energy > goal["target"]:
                goal["progress"] = 1.0
                goal["completed"] = True
            elif (
                goal["type"] == "improve_pattern_recognition"
                and agent.skills["pattern_recognition"] > goal["target"]
            ):
                goal["progress"] = 1.0
                goal["completed"] = True

        # Remove completed goals
        agent.self_improvement_goals = [
            goal
            for goal in agent.self_improvement_goals
            if not goal.get("completed", False)
        ]

    def update_advanced_emotions(self, agent: Agent):
        """Update advanced emotions."""
        # Love
        if len(agent.romantic_partners) > 0:
            agent.complex_emotions["love"] = sum(
                agent.romantic_partners.values()
            ) / len(agent.romantic_partners)

        # Jealousy
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)
        for other_agent in nearby_agents:
            if other_agent.energy > agent.energy * 1.5:
                agent.complex_emotions["jealousy"] = min(
                    1.0, agent.complex_emotions["jealousy"] + 0.1
                )

        # Pride
        if agent.energy > 70:
            agent.complex_emotions["pride"] = min(
                1.0, agent.complex_emotions["pride"] + 0.05
            )

        # Shame
        if agent.energy < 20:
            agent.complex_emotions["shame"] = min(
                1.0, agent.complex_emotions["shame"] + 0.05
            )

        # Gratitude
        if agent.help_received > 0:
            agent.complex_emotions["gratitude"] = min(
                1.0, agent.complex_emotions["gratitude"] + 0.1
            )

    def update_emotional_regulation(self, agent: Agent):
        """Update emotional regulation strategies."""
        # Suppression
        if agent.emotions["fear"] > 0.7:
            agent.emotional_regulation["suppression"] = min(
                1.0, agent.emotional_regulation["suppression"] + 0.01
            )

        # Reappraisal
        if agent.emotions["frustration"] > 0.6:
            agent.emotional_regulation["reappraisal"] = min(
                1.0, agent.emotional_regulation["reappraisal"] + 0.01
            )

        # Distraction
        if len(agent.working_memory) > 5:
            agent.emotional_regulation["distraction"] = min(
                1.0, agent.emotional_regulation["distraction"] + 0.01
            )

        # Acceptance
        if agent.energy < 30:
            agent.emotional_regulation["acceptance"] = min(
                1.0, agent.emotional_regulation["acceptance"] + 0.01
            )

    def update_empathy(self, agent: Agent):
        """Update empathy levels."""
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)

        for other_agent in nearby_agents:
            # Cognitive empathy
            if other_agent.energy < 30:
                agent.empathy_levels["cognitive"] = min(
                    1.0, agent.empathy_levels["cognitive"] + 0.01
                )

            # Emotional empathy
            if other_agent.emotions["fear"] > 0.7:
                agent.empathy_levels["emotional"] = min(
                    1.0, agent.empathy_levels["emotional"] + 0.01
                )

            # Compassionate empathy
            if other_agent.energy < 20 and agent.energy > 40:
                agent.empathy_levels["compassionate"] = min(
                    1.0, agent.empathy_levels["compassionate"] + 0.01
                )

        # Emotional intelligence
        agent.emotional_intelligence = sum(agent.empathy_levels.values()) / 3.0

    def update_scientific_thinking(self, agent: Agent):
        """Update scientific thinking and hypothesis formation."""
        # Form hypotheses about food patterns
        if len(agent.working_memory) > 3:
            recent_food = [
                mem.get("nearby_food", 0) for mem in agent.working_memory[-3:]
            ]
            if sum(recent_food) > 2:
                hypothesis = {
                    "id": len(agent.hypotheses),
                    "type": "food_pattern",
                    "prediction": "Food will appear in this area",
                    "confidence": 0.6,
                    "tick_formed": self.world.tick,
                }
                agent.hypotheses.append(hypothesis)

        # Conduct experiments
        if random.random() < 0.1:
            experiment = {
                "id": len(agent.experiments),
                "type": "movement_test",
                "hypothesis_id": (
                    len(agent.hypotheses) - 1 if agent.hypotheses else None
                ),
                "tick_started": self.world.tick,
                "results": [],
            }
            agent.experiments.append(experiment)

    def update_causal_reasoning(self, agent: Agent):
        """Update causal reasoning skills."""
        # Correlation detection
        if len(agent.working_memory) > 5:
            agent.causal_reasoning_skills["correlation"] = min(
                1.0, agent.causal_reasoning_skills["correlation"] + 0.01
            )

        # Causation understanding
        if agent.energy > 60:
            agent.causal_reasoning_skills["causation"] = min(
                1.0, agent.causal_reasoning_skills["causation"] + 0.01
            )

        # Counterfactual reasoning
        if agent.emotions["frustration"] > 0.5:
            agent.causal_reasoning_skills["counterfactual"] = min(
                1.0, agent.causal_reasoning_skills["counterfactual"] + 0.01
            )

        # Abstraction
        if len(agent.hypotheses) > 2:
            agent.abstraction_level = min(1.0, agent.abstraction_level + 0.01)

    def update_logical_reasoning(self, agent: Agent):
        """Update logical reasoning skills."""
        # Deductive reasoning
        if agent.executive_functions["planning"] > 0.7:
            agent.logical_reasoning["deductive"] = min(
                1.0, agent.logical_reasoning["deductive"] + 0.01
            )

        # Inductive reasoning
        if len(agent.working_memory) > 4:
            agent.logical_reasoning["inductive"] = min(
                1.0, agent.logical_reasoning["inductive"] + 0.01
            )

        # Abductive reasoning
        if agent.skills["pattern_recognition"] > 0.6:
            agent.logical_reasoning["abductive"] = min(
                1.0, agent.logical_reasoning["abductive"] + 0.01
            )

        # Mathematical skills
        if agent.energy > 50:
            agent.mathematical_skills["arithmetic"] = min(
                1.0, agent.mathematical_skills["arithmetic"] + 0.01
            )

        # Theory building
        if len(agent.hypotheses) > 3:
            agent.theory_building = min(1.0, agent.theory_building + 0.01)

    def update_creative_expression(self, agent: Agent):
        """Update creative expression."""
        # Artistic creation
        if random.random() < agent.creativity_level * 0.1:
            artwork = {
                "id": len(agent.artistic_works),
                "type": random.choice(["song", "poem", "drawing", "dance"]),
                "tick_created": self.world.tick,
                "inspiration": f"Inspired by energy level {agent.energy}",
            }
            agent.artistic_works.append(artwork)

        # Humor
        if random.random() < agent.humor_level * 0.05:
            jokes = [
                "Why did the agent cross the grid? To get to the other food!",
                "I told my friend a joke about energy. It didn't have much power.",
                "What do you call an agent who can't find food? Hungry!",
            ]
            agent.jokes_made.append(random.choice(jokes))

        # Imagination
        if agent.emotions["curiosity"] > 0.7:
            agent.imagination_capacity = min(1.0, agent.imagination_capacity + 0.01)

        # Aesthetic appreciation
        if agent.energy > 60:
            agent.aesthetic_appreciation = min(1.0, agent.aesthetic_appreciation + 0.01)

    def update_artistic_creation(self, agent: Agent):
        """Update artistic creation abilities."""
        # Create cultural artifacts
        if random.random() < 0.05:
            artifact = {
                "id": len(agent.cultural_artifacts),
                "type": random.choice(
                    ["tool", "decoration", "symbol", "ritual_object"]
                ),
                "tick_created": self.world.tick,
                "cultural_significance": random.choice(
                    ["sacred", "practical", "artistic", "ceremonial"]
                ),
            }
            agent.cultural_artifacts.append(artifact)

        # Creative collaborations
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)
        if len(nearby_agents) > 0 and random.random() < 0.1:
            collaboration = {
                "id": len(agent.creative_collaborations),
                "partner_id": nearby_agents[0].agent_id,
                "type": random.choice(["art", "music", "story", "ritual"]),
                "tick_started": self.world.tick,
            }
            agent.creative_collaborations.append(collaboration)

    def update_storytelling(self, agent: Agent):
        """Update storytelling abilities."""
        # Tell stories
        if random.random() < agent.language_skills * 0.1:
            story = {
                "id": len(agent.stories_told),
                "type": random.choice(
                    ["adventure", "myth", "personal", "instructional"]
                ),
                "tick_told": self.world.tick,
                "audience_size": len(self.world.get_nearby_agents(agent.x, agent.y, 3)),
                "theme": f"Story about energy and survival at tick {self.world.tick}",
            }
            agent.stories_told.append(story)

    def update_physical_reality(self, agent: Agent):
        """Update 3D spatial environment and physics."""
        # Update 3D position based on velocity
        agent.x = (agent.x + int(agent.velocity[0])) % self.world.grid_size
        agent.y = (agent.y + int(agent.velocity[1])) % self.world.grid_size
        agent.z = max(0, min(10, agent.z + int(agent.velocity[2])))  # Limit z to 0-10

        # Update temperature based on environment
        environmental_temp = self.world.temperature
        temp_diff = environmental_temp - agent.temperature
        agent.temperature += temp_diff * 0.01  # Gradual temperature change

        # Update pressure based on altitude
        agent.pressure = 1.0 - (agent.z * 0.1)  # Pressure decreases with altitude

    def update_physics_engine(self, agent: Agent):
        """Update physics engine (gravity, momentum, collisions)."""
        # Apply gravity
        gravity = -0.1
        agent.acceleration = (
            agent.acceleration[0],
            agent.acceleration[1],
            agent.acceleration[2] + gravity,
        )

        # Update velocity based on acceleration
        agent.velocity = (
            agent.velocity[0] + agent.acceleration[0],
            agent.velocity[1] + agent.acceleration[1],
            agent.velocity[2] + agent.acceleration[2],
        )

        # Apply friction
        friction = 0.95
        agent.velocity = (
            agent.velocity[0] * friction,
            agent.velocity[1] * friction,
            agent.velocity[2] * friction,
        )

        # Reset acceleration
        agent.acceleration = (0.0, 0.0, 0.0)

    def update_chemical_reactions(self, agent: Agent):
        """Update chemical reactions and composition."""
        # Oxygen consumption for metabolism
        if agent.energy > 0:
            agent.chemical_composition["oxygen"] = max(
                0.0, agent.chemical_composition["oxygen"] - 0.001
            )
            agent.chemical_composition["carbon"] += 0.0005  # CO2 production

        # Chemical reactions based on environment
        if self.world.temperature > 40:
            # Heat stress - increased water loss
            agent.chemical_composition["hydrogen"] = max(
                0.0, agent.chemical_composition["hydrogen"] - 0.0001
            )

    def update_biological_processes(self, agent: Agent):
        """Update biological processes (metabolism, growth, reproduction)."""
        # Metabolism
        agent.biological_processes["metabolism"] = agent.energy / 100.0

        # Growth (if young)
        if agent.age < 100:
            agent.biological_processes["growth"] = min(
                1.0, agent.biological_processes["growth"] + 0.01
            )

        # Healing
        if agent.health < 1.0:
            agent.biological_processes["healing"] = min(
                1.0, agent.biological_processes["healing"] + 0.005
            )

    def update_evolutionary_biology(self, agent: Agent):
        """Update evolutionary biology and fitness."""
        # Calculate fitness score based on survival and reproduction
        survival_fitness = agent.energy / 100.0
        reproduction_fitness = len(agent.children) * 0.1
        agent.fitness_score = (survival_fitness + reproduction_fitness) / 2.0

        # Update generation
        if agent.age > 200:
            agent.generation += 1

    def update_dna_evolution(self, agent: Agent):
        """Update DNA evolution and mutations."""
        # Random mutations
        if random.random() < 0.001:  # 0.1% mutation rate
            mutation = {
                "type": random.choice(["point", "insertion", "deletion"]),
                "position": random.randint(0, len(agent.dna_sequence) - 1),
                "tick": self.world.tick,
            }
            agent.mutations.append(mutation)

            # Apply mutation to DNA
            if mutation["type"] == "point":
                agent.dna_sequence[mutation["position"]] = random.choice(
                    ["A", "T", "G", "C"]
                )

    def update_natural_selection(self, agent: Agent):
        """Update natural selection pressure."""
        # Selection pressure based on environment
        if agent.fitness_score < 0.3:
            # Low fitness - increased selection pressure
            agent.energy -= 1.0
        elif agent.fitness_score > 0.7:
            # High fitness - reduced selection pressure
            agent.energy += 0.5

    def update_civilization_culture(self, agent: Agent):
        """Update civilization and cultural systems."""
        # Assign city membership
        if agent.city_id is None:
            agent.city_id = random.randint(1, 5)  # 5 cities

        # Update citizenship status
        if agent.energy > 60:
            agent.citizenship_status = "citizen"
        elif agent.energy < 20:
            agent.citizenship_status = "refugee"

        # Update economic class
        if agent.wealth > 200:
            agent.economic_class = "upper"
        elif agent.wealth > 100:
            agent.economic_class = "middle"
        elif agent.wealth > 50:
            agent.economic_class = "working"
        else:
            agent.economic_class = "poor"

    def update_government_systems(self, agent: Agent):
        """Update government systems and roles."""
        # Assign government roles based on influence
        if agent.influence_level > 0.8:
            agent.government_role = "leader"
        elif agent.influence_level > 0.6:
            agent.government_role = "official"
        else:
            agent.government_role = "citizen"

        # Update legal status
        if agent.energy < 10:
            agent.legal_status = "criminal"  # Desperate agents may break laws
        else:
            agent.legal_status = "lawful"

    def update_legal_systems(self, agent: Agent):
        """Update legal systems and justice."""
        # Legal consequences for low energy (desperation)
        if agent.legal_status == "criminal":
            agent.reputation = max(0.0, agent.reputation - 0.01)

    def update_religious_systems(self, agent: Agent):
        """Update religious systems and beliefs."""
        # Develop religious beliefs
        if random.random() < 0.01:
            beliefs = ["monotheism", "polytheism", "animism", "atheism", "agnosticism"]
            agent.religious_beliefs.append(random.choice(beliefs))

    def update_technology_innovation(self, agent: Agent):
        """Update technology and innovation systems."""
        # Unlock technologies based on knowledge
        if agent.scientific_knowledge["mathematics"] > 0.5:
            agent.technology_tree["writing"] = True
        if agent.scientific_knowledge["physics"] > 0.3:
            agent.technology_tree["tools"] = True
        if agent.scientific_knowledge["chemistry"] > 0.4:
            agent.technology_tree["fire"] = True

    def update_research_projects(self, agent: Agent):
        """Update research projects and scientific advancement."""
        # Start research projects
        if random.random() < 0.05:
            project = {
                "id": len(agent.research_projects),
                "type": random.choice(
                    ["mathematics", "physics", "chemistry", "biology", "medicine"]
                ),
                "progress": 0.0,
                "tick_started": self.world.tick,
            }
            agent.research_projects.append(project)

        # Progress research
        for project in agent.research_projects:
            project["progress"] += 0.01
            if project["progress"] >= 1.0:
                # Research complete
                agent.scientific_knowledge[project["type"]] = min(
                    1.0, agent.scientific_knowledge[project["type"]] + 0.1
                )
                project["completed"] = True

    def update_innovation_networks(self, agent: Agent):
        """Update innovation networks and collaboration."""
        # Connect to nearby agents for innovation
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)
        for other_agent in nearby_agents:
            if other_agent.agent_id not in agent.innovation_network:
                agent.innovation_network.append(other_agent.agent_id)

    def update_information_communication(self, agent: Agent):
        """Update information and communication systems."""
        # Store information
        if random.random() < 0.1:
            info_key = f"observation_{self.world.tick}"
            agent.information_storage[info_key] = {
                "content": f"Observed environment at tick {self.world.tick}",
                "tick": self.world.tick,
                "reliability": random.random(),
            }

        # Update communication network
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 3)
        for other_agent in nearby_agents:
            if other_agent.agent_id not in agent.communication_network:
                agent.communication_network.append(other_agent.agent_id)

    def update_media_systems(self, agent: Agent):
        """Update media systems and consumption."""
        # Media consumption
        if random.random() < 0.1:
            media_type = random.choice(
                ["news", "entertainment", "education", "propaganda"]
            )
            agent.media_consumption[media_type] = min(
                1.0, agent.media_consumption[media_type] + 0.1
            )

    def update_knowledge_management(self, agent: Agent):
        """Update knowledge management and processing."""
        # Process information
        if agent.information_storage:
            agent.information_processing["analysis"] = min(
                1.0, agent.information_processing["analysis"] + 0.01
            )
            agent.information_processing["synthesis"] = min(
                1.0, agent.information_processing["synthesis"] + 0.01
            )

    def update_psychological_reality(self, agent: Agent):
        """Update psychological reality and mental health."""
        # Update mental health status
        if agent.emotions["fear"] > 0.8:
            agent.mental_health_status = "anxious"
        elif agent.emotions["frustration"] > 0.8:
            agent.mental_health_status = "depressed"
        elif agent.energy > 80:
            agent.mental_health_status = "healthy"

        # Update psychological development
        if agent.age < 50:
            agent.psychological_development_stage = "child"
        elif agent.age < 100:
            agent.psychological_development_stage = "adolescent"
        elif agent.age < 200:
            agent.psychological_development_stage = "adult"
        else:
            agent.psychological_development_stage = "elderly"

    def update_mental_health(self, agent: Agent):
        """Update mental health and therapy."""
        # Therapy sessions
        if agent.mental_health_status != "healthy" and random.random() < 0.05:
            session = {
                "tick": self.world.tick,
                "type": "therapy",
                "effectiveness": random.random(),
            }
            agent.therapy_sessions.append(session)

            # Improve mental health
            if session["effectiveness"] > 0.5:
                agent.mental_health_status = "healthy"

    def update_cognitive_biases(self, agent: Agent):
        """Update cognitive biases and thinking patterns."""
        # Develop cognitive biases
        if random.random() < 0.01:
            biases = [
                "confirmation_bias",
                "availability_heuristic",
                "anchoring_bias",
                "sunk_cost_fallacy",
            ]
            agent.cognitive_biases.append(random.choice(biases))

    def update_ecological_systems(self, agent: Agent):
        """Update ecological systems and environmental impact."""
        # Update ecological role
        if agent.energy > 70:
            agent.ecological_role = "omnivore"
        elif agent.energy < 30:
            agent.ecological_role = "herbivore"

        # Update environmental impact
        agent.environmental_impact = agent.energy / 100.0

    def update_food_webs(self, agent: Agent):
        """Update food webs and ecosystem position."""
        # Update food web position
        if agent.ecological_role == "omnivore":
            agent.food_web_position = 3
        elif agent.ecological_role == "herbivore":
            agent.food_web_position = 2
        elif agent.ecological_role == "carnivore":
            agent.food_web_position = 4

    def update_ecosystem_dynamics(self, agent: Agent):
        """Update ecosystem dynamics and services."""
        # Provide ecosystem services
        if agent.energy > 50:
            services = ["pollination", "seed_dispersal", "nutrient_cycling"]
            agent.ecosystem_services.append(random.choice(services))

    def update_health_medicine(self, agent: Agent):
        """Update health and medicine systems."""
        # Update healthcare access
        if agent.economic_class == "upper":
            agent.healthcare_access = 1.0
        elif agent.economic_class == "middle":
            agent.healthcare_access = 0.7
        else:
            agent.healthcare_access = 0.3

        # Update medical insurance
        if agent.wealth > 150:
            agent.medical_insurance = True

    def update_disease_systems(self, agent: Agent):
        """Update disease systems and health."""
        # Contract diseases
        if random.random() < 0.001:  # 0.1% disease rate
            disease = {
                "name": random.choice(["flu", "cold", "infection"]),
                "severity": random.random(),
                "tick_contracted": self.world.tick,
            }
            agent.diseases.append(disease)
            agent.health -= disease["severity"] * 0.1

    def update_healthcare_systems(self, agent: Agent):
        """Update healthcare systems and treatment."""
        # Medical treatment
        if agent.diseases and agent.healthcare_access > 0.5:
            for disease in agent.diseases:
                if random.random() < agent.healthcare_access:
                    treatment = {
                        "disease": disease["name"],
                        "effectiveness": agent.healthcare_access,
                        "tick_treated": self.world.tick,
                    }
                    agent.medical_treatments.append(treatment)
                    agent.health += treatment["effectiveness"] * 0.1
                    disease["treated"] = True

    def update_economic_systems(self, agent: Agent):
        """Update economic systems and wealth."""
        # Update income and expenses
        agent.wealth += agent.income - agent.expenses

        # Economic cycles
        if agent.wealth > 150:
            agent.economic_cycles["boom"] = min(
                1.0, agent.economic_cycles["boom"] + 0.01
            )
        elif agent.wealth < 50:
            agent.economic_cycles["bust"] = min(
                1.0, agent.economic_cycles["bust"] + 0.01
            )
        else:
            agent.economic_cycles["recovery"] = min(
                1.0, agent.economic_cycles["recovery"] + 0.01
            )

    def update_trade_networks(self, agent: Agent):
        """Update trade networks and relationships."""
        # Establish trade relationships
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)
        for other_agent in nearby_agents:
            if other_agent.agent_id not in agent.trade_relationships:
                agent.trade_relationships[other_agent.agent_id] = {
                    "goods_traded": 0,
                    "value_exchanged": 0.0,
                    "trust_level": 0.5,
                }

    def update_market_systems(self, agent: Agent):
        """Update market systems and participation."""
        # Market participation
        if agent.wealth > 100:
            agent.market_participation["buyer"] = min(
                1.0, agent.market_participation["buyer"] + 0.01
            )
            agent.market_participation["seller"] = min(
                1.0, agent.market_participation["seller"] + 0.01
            )

        if agent.wealth > 200:
            agent.market_participation["investor"] = min(
                1.0, agent.market_participation["investor"] + 0.01
            )

    def update_education_learning(self, agent: Agent):
        """Update education and learning systems."""
        # Update education level
        if agent.scientific_knowledge["mathematics"] > 0.7:
            agent.education_level = "expert"
        elif agent.scientific_knowledge["mathematics"] > 0.5:
            agent.education_level = "advanced"
        elif agent.scientific_knowledge["mathematics"] > 0.3:
            agent.education_level = "intermediate"
        else:
            agent.education_level = "basic"

        # Lifelong learning
        agent.lifelong_learning = min(1.0, agent.lifelong_learning + 0.001)

    def update_educational_institutions(self, agent: Agent):
        """Update educational institutions and enrollment."""
        # Enroll in educational institution
        if agent.education_level == "basic" and random.random() < 0.1:
            agent.educational_institution = random.randint(1, 3)  # 3 schools

        # Learning methods
        if agent.educational_institution:
            agent.learning_methods["lecture"] = min(
                1.0, agent.learning_methods["lecture"] + 0.01
            )
            agent.learning_methods["practice"] = min(
                1.0, agent.learning_methods["practice"] + 0.01
            )

    def update_knowledge_transfer(self, agent: Agent):
        """Update knowledge transfer and mentoring."""
        # Knowledge transfer to nearby agents
        nearby_agents = self.world.get_nearby_agents(agent.x, agent.y, 2)
        for other_agent in nearby_agents:
            if (
                agent.education_level == "expert"
                and other_agent.education_level == "basic"
            ):
                transfer = {
                    "from_agent": agent.agent_id,
                    "to_agent": other_agent.agent_id,
                    "knowledge_type": "mentoring",
                    "tick": self.world.tick,
                }
                agent.knowledge_transfer[other_agent.agent_id] = transfer
                other_agent.scientific_knowledge["mathematics"] = min(
                    1.0, other_agent.scientific_knowledge["mathematics"] + 0.01
                )

    def update_cosmic_astronomical(self, agent: Agent):
        """Update cosmic and astronomical reality."""
        # Update orbital position
        agent.orbital_position = (
            agent.orbital_position[0] + agent.orbital_velocity * 0.01,
            agent.orbital_position[1] + agent.orbital_velocity * 0.01,
        )

        # Update gravitational field based on distance from center
        distance_from_center = (
            agent.orbital_position[0] ** 2 + agent.orbital_position[1] ** 2
        ) ** 0.5
        agent.gravitational_field = 1.0 / (1.0 + distance_from_center * 0.1)

        # Update cosmic radiation exposure
        agent.cosmic_radiation = min(
            1.0, agent.cosmic_radiation + random.random() * 0.001
        )

    def update_orbital_mechanics(self, agent: Agent):
        """Update orbital mechanics and gravitational effects."""
        # Update orbital velocity based on gravitational field
        agent.orbital_velocity += agent.gravitational_field * 0.01

        # Apply orbital mechanics
        if agent.orbital_velocity > 1.0:
            agent.orbital_velocity = 1.0  # Limit velocity

        # Update astronomical knowledge
        if random.random() < 0.01:
            agent.astronomical_knowledge["astronomy"] = min(
                1.0, agent.astronomical_knowledge["astronomy"] + 0.01
            )

    def update_space_exploration(self, agent: Agent):
        """Update space exploration and missions."""
        # Space exploration missions
        if random.random() < 0.005:
            mission = {
                "id": len(agent.space_exploration),
                "type": random.choice(["satellite", "rover", "manned", "probe"]),
                "destination": random.choice(["moon", "mars", "asteroid", "comet"]),
                "tick_started": self.world.tick,
            }
            agent.space_exploration[f'mission_{mission["id"]}'] = mission

    def update_neuroscience_brain(self, agent: Agent):
        """Update neuroscience and brain simulation."""
        # Update brain regions based on activity
        if agent.energy > 50:
            agent.brain_regions["cortex"] = min(
                1.0, agent.brain_regions["cortex"] + 0.001
            )
            agent.brain_regions["prefrontal"] = min(
                1.0, agent.brain_regions["prefrontal"] + 0.001
            )

        # Update neurotransmitters
        if agent.emotions["hope"] > 0.5:
            agent.neurotransmitters["dopamine"] = min(
                1.0, agent.neurotransmitters["dopamine"] + 0.01
            )
        if agent.emotions["fear"] > 0.5:
            agent.neurotransmitters["norepinephrine"] = min(
                1.0, agent.neurotransmitters["norepinephrine"] + 0.01
            )

    def update_neural_networks(self, agent: Agent):
        """Update neural networks and pathways."""
        # Create new neural pathways
        if random.random() < 0.01:
            pathway = {
                "id": len(agent.neural_network["pathways"]),
                "from_region": random.choice(list(agent.brain_regions.keys())),
                "to_region": random.choice(list(agent.brain_regions.keys())),
                "strength": random.random(),
                "tick_created": self.world.tick,
            }
            agent.neural_network["pathways"].append(pathway)

        # Update brain plasticity
        agent.brain_plasticity = min(1.0, agent.brain_plasticity + 0.001)

    def update_brain_plasticity(self, agent: Agent):
        """Update brain plasticity and learning."""
        # Brain plasticity affects learning
        if agent.brain_plasticity > 0.7:
            agent.scientific_knowledge["mathematics"] = min(
                1.0, agent.scientific_knowledge["mathematics"] + 0.01
            )

        # Update neural oscillations
        agent.neural_oscillations["alpha"] = random.random() * 0.1
        agent.neural_oscillations["beta"] = random.random() * 0.1
        agent.neural_oscillations["gamma"] = random.random() * 0.1

    def update_quantum_mechanics(self, agent: Agent):
        """Update quantum mechanics and physics."""
        # Update quantum state
        agent.quantum_state["superposition"] = complex(
            random.random() * 0.1, random.random() * 0.1
        )

        # Update quantum uncertainty
        agent.quantum_uncertainty = random.random()

        # Quantum effects on decision making
        if agent.quantum_uncertainty > 0.8:
            agent.energy += 0.1  # Quantum tunneling effect

    def update_quantum_computing(self, agent: Agent):
        """Update quantum computing capabilities."""
        # Develop quantum computing
        if random.random() < 0.001:
            qubit = {
                "id": len(agent.quantum_computing),
                "state": random.choice(["|0>", "|1>", "|+>", "|->"]),
                "entanglement": random.random(),
                "tick_created": self.world.tick,
            }
            agent.quantum_computing[f'qubit_{qubit["id"]}'] = qubit

    def update_quantum_consciousness(self, agent: Agent):
        """Update quantum consciousness theories."""
        # Quantum consciousness development
        if agent.quantum_uncertainty > 0.5:
            agent.quantum_consciousness = min(1.0, agent.quantum_consciousness + 0.001)

        # Quantum biology effects
        agent.quantum_biology["quantum_coherence"] = random.random() * 0.1
        agent.quantum_biology["quantum_tunneling"] = random.random() * 0.1

    def update_climate_weather(self, agent: Agent):
        """Update climate and weather systems."""
        # Update weather patterns
        agent.weather_patterns["temperature"] += random.uniform(-0.5, 0.5)
        agent.weather_patterns["humidity"] = max(
            0.0,
            min(1.0, agent.weather_patterns["humidity"] + random.uniform(-0.01, 0.01)),
        )
        agent.weather_patterns["pressure"] += random.uniform(-1.0, 1.0)

        # Climate change impact
        agent.climate_change_impact = min(
            1.0, agent.climate_change_impact + random.random() * 0.001
        )

    def update_atmospheric_dynamics(self, agent: Agent):
        """Update atmospheric dynamics and weather systems."""
        # Update atmospheric dynamics
        agent.atmospheric_dynamics["jet_stream"] = random.uniform(-1.0, 1.0)
        agent.atmospheric_dynamics["ocean_currents"] = random.uniform(-1.0, 1.0)

        # El Niño/La Niña effects
        if random.random() < 0.01:
            agent.atmospheric_dynamics["el_nino"] = random.random()
            agent.atmospheric_dynamics["la_nina"] = random.random()

    def update_natural_disasters(self, agent: Agent):
        """Update natural disasters and extreme weather."""
        # Natural disasters
        if random.random() < 0.001:
            disaster = {
                "type": random.choice(
                    ["hurricane", "earthquake", "tsunami", "drought", "flood"]
                ),
                "severity": random.random(),
                "tick_occurred": self.world.tick,
            }
            agent.natural_disasters.append(disaster)
            agent.health -= disaster["severity"] * 0.1

    def update_molecular_cellular(self, agent: Agent):
        """Update molecular and cellular biology."""
        # Update cellular machinery
        agent.cellular_machinery["mitochondria"]["energy"] = agent.energy
        agent.cellular_machinery["nucleus"]["dna"] = len(agent.dna_sequence)

        # Protein synthesis
        if agent.energy > 30:
            agent.protein_synthesis["transcription"] = min(
                1.0, agent.protein_synthesis["transcription"] + 0.01
            )
            agent.protein_synthesis["translation"] = min(
                1.0, agent.protein_synthesis["translation"] + 0.01
            )

    def update_protein_synthesis(self, agent: Agent):
        """Update protein synthesis and gene expression."""
        # Gene expression
        agent.gene_expression["regulation"] = random.random()

        # Metabolic pathways
        if agent.energy > 50:
            agent.metabolic_pathways["glycolysis"] = min(
                1.0, agent.metabolic_pathways["glycolysis"] + 0.01
            )
            agent.metabolic_pathways["krebs_cycle"] = min(
                1.0, agent.metabolic_pathways["krebs_cycle"] + 0.01
            )

    def update_cell_division(self, agent: Agent):
        """Update cell division and cellular processes."""
        # Cell division
        if agent.age < 100 and agent.energy > 60:
            agent.cell_division["mitosis"] = min(
                1.0, agent.cell_division["mitosis"] + 0.01
            )

        # Cell cycle
        agent.cell_division["cell_cycle"] = (agent.age % 100) / 100.0

    def update_internet_digital(self, agent: Agent):
        """Update internet and digital reality."""
        # Internet presence
        if random.random() < 0.01:
            presence = {
                "website": f"agent_{agent.agent_id}.com",
                "email": f"agent_{agent.agent_id}@email.com",
                "tick_created": self.world.tick,
            }
            agent.internet_presence[f"presence_{len(agent.internet_presence)}"] = (
                presence
            )

        # Digital identity
        agent.digital_identity["username"] = f"agent_{agent.agent_id}"
        agent.digital_identity["reputation"] = agent.reputation

    def update_social_media(self, agent: Agent):
        """Update social media and online presence."""
        # Social media profiles
        if random.random() < 0.01:
            platform = random.choice(["facebook", "twitter", "instagram", "linkedin"])
            profile = {
                "platform": platform,
                "followers": random.randint(0, 1000),
                "posts": random.randint(0, 100),
                "tick_created": self.world.tick,
            }
            agent.social_media_profiles[platform] = profile

    def update_virtual_worlds(self, agent: Agent):
        """Update virtual worlds and digital environments."""
        # Virtual worlds
        if random.random() < 0.005:
            world = {
                "id": len(agent.virtual_worlds),
                "name": f"VirtualWorld_{agent.agent_id}_{self.world.tick}",
                "type": random.choice(["gaming", "social", "educational", "business"]),
                "tick_created": self.world.tick,
            }
            agent.virtual_worlds.append(world)

        # Digital economy
        agent.digital_economy["cryptocurrency"] = random.random() * 1000
        agent.digital_economy["digital_assets"] = random.random() * 100

    def update_art_cultural(self, agent: Agent):
        """Update art and cultural expression."""
        # Art history knowledge
        if random.random() < 0.01:
            period = random.choice(
                ["renaissance", "baroque", "romanticism", "modernism"]
            )
            agent.art_history_knowledge[period] = min(
                1.0, agent.art_history_knowledge[period] + 0.01
            )

        # Cultural trends
        agent.cultural_trends["fashion"] = random.random()
        agent.cultural_trends["music"] = random.random()
        agent.cultural_trends["literature"] = random.random()

    def update_cultural_movements(self, agent: Agent):
        """Update cultural movements and trends."""
        # Cultural movements
        if random.random() < 0.005:
            movement = {
                "id": len(agent.cultural_movements),
                "name": f"Movement_{agent.agent_id}_{self.world.tick}",
                "type": random.choice(["artistic", "social", "political", "religious"]),
                "tick_started": self.world.tick,
            }
            agent.cultural_movements.append(movement)

        # Aesthetic evolution
        agent.aesthetic_evolution = min(1.0, agent.aesthetic_evolution + 0.001)

    def update_aesthetic_evolution(self, agent: Agent):
        """Update aesthetic evolution and artistic development."""
        # Artistic collaboration
        agent.artistic_collaboration["collective_creation"] = random.random()
        agent.artistic_collaboration["cultural_exchange"] = random.random()

    def update_political_governance(self, agent: Agent):
        """Update political systems and governance."""
        # Political system
        if agent.influence_level > 0.8:
            agent.political_system = "democracy"
        elif agent.influence_level < 0.2:
            agent.political_system = "autocracy"

        # Political party
        if random.random() < 0.01:
            agent.political_party = random.choice(
                ["liberal", "conservative", "socialist", "libertarian"]
            )

    def update_elections_voting(self, agent: Agent):
        """Update elections and voting systems."""
        # Voting record
        if random.random() < 0.01:
            vote = {
                "election_id": f"election_{self.world.tick}",
                "candidate": f"candidate_{random.randint(1, 5)}",
                "tick_voted": self.world.tick,
            }
            agent.voting_record.append(vote)

        # Policy positions
        agent.policy_positions["liberal"] = random.random()
        agent.policy_positions["conservative"] = random.random()

    def update_policy_making(self, agent: Agent):
        """Update policy making and governance."""
        # International relations
        agent.international_relations["diplomacy"] = random.random()
        agent.international_relations["trade_agreements"] = random.random()

        # Governance structures
        agent.governance_structures["federal"] = random.random()
        agent.governance_structures["unitary"] = random.random()

    def update_biotechnology_genetic(self, agent: Agent):
        """Update biotechnology and genetic engineering."""
        # Genetic modification
        if random.random() < 0.001:
            modification = {
                "id": len(agent.genetic_modification),
                "type": random.choice(["enhancement", "therapy", "prevention"]),
                "gene": random.choice(
                    ["intelligence", "strength", "longevity", "immunity"]
                ),
                "tick_modified": self.world.tick,
            }
            agent.genetic_modification[f'mod_{modification["id"]}'] = modification

        # CRISPR technology
        if agent.scientific_knowledge["biology"] > 0.7:
            agent.crispr_technology = True

    def update_crispr_gene_editing(self, agent: Agent):
        """Update CRISPR gene editing technology."""
        # Gene therapy
        if agent.crispr_technology and random.random() < 0.001:
            therapy = {
                "id": len(agent.gene_therapy),
                "target_gene": random.choice(["disease", "enhancement", "prevention"]),
                "method": "CRISPR",
                "tick_administered": self.world.tick,
            }
            agent.gene_therapy.append(therapy)

        # Synthetic biology
        if agent.crispr_technology:
            agent.synthetic_biology["artificial_cells"] = random.random()
            agent.synthetic_biology["synthetic_organisms"] = random.random()

    def update_synthetic_biology(self, agent: Agent):
        """Update synthetic biology and biotechnology."""
        # Biotechnology
        agent.biotechnology["pharmaceuticals"] = random.random()
        agent.biotechnology["agriculture"] = random.random()
        agent.biotechnology["medicine"] = random.random()

        # Cloning technology
        if agent.scientific_knowledge["biology"] > 0.8:
            agent.cloning_technology = True

    def update_ai_machine_learning(self, agent: Agent):
        """Update artificial intelligence and machine learning."""
        # Machine learning
        agent.machine_learning["neural_networks"] = random.random()
        agent.machine_learning["deep_learning"] = random.random()
        agent.machine_learning["algorithms"] = random.random()

        # AI systems
        if random.random() < 0.01:
            ai_system = {
                "id": len(agent.ai_systems),
                "type": random.choice(
                    ["chatbot", "assistant", "autonomous", "predictive"]
                ),
                "capability": random.random(),
                "tick_created": self.world.tick,
            }
            agent.ai_systems[f'ai_{ai_system["id"]}'] = ai_system

    def update_human_ai_interaction(self, agent: Agent):
        """Update human-AI interaction and collaboration."""
        # Human-AI interaction
        agent.human_ai_interaction["collaboration"] = random.random()
        agent.human_ai_interaction["augmentation"] = random.random()
        agent.human_ai_interaction["replacement"] = random.random()

        # AI ethics
        agent.ai_ethics["bias"] = random.random()
        agent.ai_ethics["fairness"] = random.random()
        agent.ai_ethics["transparency"] = random.random()

    def update_agi_development(self, agent: Agent):
        """Update AGI development and superintelligence."""
        # AGI development
        if agent.machine_learning["deep_learning"] > 0.8:
            agent.agi_development = min(1.0, agent.agi_development + 0.001)

        # Superintelligence
        if agent.agi_development > 0.9:
            agent.superintelligence = True

    def make_decision(
        self,
        agent: Agent,
        nearby_food: list[tuple[int, int]],
        workspace_food: list[tuple[int, int]] | None,
        nearby_agents: list[Agent],
    ) -> tuple[str, float, float, float]:
        """Make decision using consciousness-proxy mechanisms."""
        actions = ["stay", "right", "left", "down", "up"]
        action_deltas = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

        scores = []

        for i, (action, (dx, dy)) in enumerate(zip(actions, action_deltas)):
            score = 0.0

            # Food seeking component
            if nearby_food:
                # Calculate distance to nearest food after move
                new_x = (agent.x + dx) % self.world.grid_size
                new_y = (agent.y + dy) % self.world.grid_size

                min_distance = min(
                    math.sqrt((fx - new_x) ** 2 + (fy - new_y) ** 2)
                    for fx, fy in nearby_food
                )
                food_score = agent.genome[0] * (
                    10.0 - min_distance
                )  # Closer = higher score
                score += food_score

            # Energy caution component
            if agent.energy < 20:  # Low energy threshold
                if action == "stay":
                    score += agent.genome[1] * 5.0  # Stay when low energy
                else:
                    score -= agent.genome[1] * 2.0  # Penalize movement when low energy

            # Exploration component
            if action != "stay":
                score += agent.genome[2] * 1.0  # Reward movement for exploration

            # Workspace influence
            if workspace_food and not self.args.lesion_workspace:
                # Use workspace information to influence decision
                workspace_bonus = 0.5
                score += workspace_bonus

            # Advanced social dynamics
            social_tendency = agent.genome[4]
            crowding_level = getattr(self.args, "crowding_strength", 0.3)

            if nearby_agents:
                num_nearby = len(nearby_agents)

                # Check if moving to this position would increase crowding
                new_x = (agent.x + dx) % self.world.grid_size
                new_y = (agent.y + dy) % self.world.grid_size
                future_nearby = self.world.get_nearby_agents(new_x, new_y, 2)
                future_crowding = len(future_nearby)

                # Crowding penalty: avoid overcrowded areas
                if future_crowding > 2:
                    crowding_penalty = crowding_level * (future_crowding - 2) * 2.0
                    score -= crowding_penalty

                # Social influence based on trust and hierarchy
                trusted_agents = 0
                dominant_agents = 0

                for other_agent in nearby_agents:
                    # Check trust from memory
                    if other_agent.agent_id in agent.memory:
                        trust = agent.memory[other_agent.agent_id]["trust"]
                        if trust > 0.6:
                            trusted_agents += 1

                    # Check dominance (age + energy)
                    dominance_score = other_agent.age * 0.1 + other_agent.energy * 0.01
                    if dominance_score > agent.age * 0.1 + agent.energy * 0.01:
                        dominant_agents += 1

                # Follow trusted agents
                if trusted_agents > num_nearby / 3:
                    social_bonus = social_tendency * 1.5
                    score += social_bonus

                # Follow dominant agents
                if dominant_agents > num_nearby / 2:
                    hierarchy_bonus = social_tendency * 1.0
                    score += hierarchy_bonus

            # Memory-based decision making
            memory_strength = agent.genome[3]
            new_pos = (
                (agent.x + dx) % self.world.grid_size,
                (agent.y + dy) % self.world.grid_size,
            )

            if new_pos in agent.memory:
                memory_data = agent.memory[new_pos]
                if memory_data["food_found"] > 0:
                    # Remember this location had food
                    memory_bonus = memory_strength * memory_data["food_found"] * 2.0
                    score += memory_bonus
                else:
                    # Avoid locations that were barren
                    memory_penalty = memory_strength * 0.5
                    score -= memory_penalty

            # Advanced metacognition: uncertainty awareness
            uncertainty_factor = 1.0
            if len(agent.confidence_history) > 10:
                # Calculate calibration error
                recent_conf = agent.confidence_history[-10:]
                avg_confidence = sum(recent_conf) / len(recent_conf)

                # If agent has been overconfident, reduce confidence
                if avg_confidence > 0.7:
                    uncertainty_factor = 0.8  # Be more cautious
                elif avg_confidence < 0.3:
                    uncertainty_factor = 1.2  # Be more confident

            score *= uncertainty_factor

            # Emotional influences
            fear_factor = 1.0 - agent.emotions["fear"] * 0.5  # Fear reduces exploration
            hope_factor = 1.0 + agent.emotions["hope"] * 0.3  # Hope increases optimism
            frustration_factor = (
                1.0 - agent.emotions["frustration"] * 0.4
            )  # Frustration reduces effectiveness
            curiosity_factor = (
                1.0 + agent.emotions["curiosity"] * 0.4
            )  # Curiosity increases exploration
            boredom_factor = (
                1.0 + agent.emotions["boredom"] * 0.2
            )  # Boredom drives change

            score *= (
                fear_factor
                * hope_factor
                * frustration_factor
                * curiosity_factor
                * boredom_factor
            )

            # Personality influences
            openness_factor = (
                1.0 + agent.personality["openness"] * 0.3
            )  # Openness increases exploration
            conscientiousness_factor = (
                1.0 + agent.personality["conscientiousness"] * 0.2
            )  # Conscientiousness increases carefulness
            extraversion_factor = (
                1.0 + agent.personality["extraversion"] * 0.2
            )  # Extraversion increases social behavior
            agreeableness_factor = (
                1.0 + agent.personality["agreeableness"] * 0.1
            )  # Agreeableness increases cooperation
            neuroticism_factor = (
                1.0 - agent.personality["neuroticism"] * 0.3
            )  # Neuroticism increases anxiety

            score *= (
                openness_factor
                * conscientiousness_factor
                * extraversion_factor
                * agreeableness_factor
                * neuroticism_factor
            )

            # Motivational influences (Maslow's hierarchy)
            physiological_factor = (
                agent.motivation["physiological"] * 2.0
            )  # Strong drive for basic needs
            safety_factor = agent.motivation["safety"] * 1.5  # Safety needs
            social_factor = agent.motivation["social"] * 1.2  # Social needs
            esteem_factor = agent.motivation["esteem"] * 1.1  # Esteem needs
            self_actualization_factor = (
                agent.motivation["self_actualization"] * 1.05
            )  # Self-actualization

            score *= (
                physiological_factor
                * safety_factor
                * social_factor
                * esteem_factor
                * self_actualization_factor
            )

            # Skill influences
            pattern_factor = (
                1.0 + agent.skills["pattern_recognition"] * 0.3
            )  # Better pattern recognition
            causal_factor = (
                1.0 + agent.skills["causal_reasoning"] * 0.2
            )  # Better reasoning
            tool_factor = 1.0 + agent.skills["tool_use"] * 0.4  # Tool use effectiveness
            communication_factor = (
                1.0 + agent.skills["communication"] * 0.2
            )  # Better communication
            innovation_factor = (
                1.0 + agent.skills["innovation"] * 0.3
            )  # Innovation ability

            score *= (
                pattern_factor
                * causal_factor
                * tool_factor
                * communication_factor
                * innovation_factor
            )

            # Life stage influences
            life_stage_factor = {
                "childhood": 0.8,  # Less capable
                "adolescence": 1.1,  # More energetic
                "adulthood": 1.0,  # Normal capability
                "old_age": 0.7,  # Reduced capability
            }[agent.life_stage]

            score *= life_stage_factor

            # Advanced cognitive influences
            # Working memory influence
            if agent.working_memory:
                recent_food_found = sum(
                    1
                    for mem in agent.working_memory[-3:]
                    if mem.get("nearby_food", 0) > 0
                )
                if recent_food_found > 1 and action != "stay":
                    score += 0.5  # Optimism from recent success

            # Executive control influence
            if agent.current_goals:
                primary_goal = max(agent.current_goals, key=lambda g: g["priority"])
                if primary_goal["type"] == "find_food" and action != "stay":
                    score += agent.executive_functions["planning"] * 0.3
                elif primary_goal["type"] == "find_safety" and action == "stay":
                    score += agent.executive_functions["inhibition"] * 0.2

            # Theory of mind influence
            for other_agent in nearby_agents:
                if other_agent.agent_id in agent.mental_models:
                    mental_model = agent.mental_models[other_agent.agent_id]
                    trust_level = mental_model["trust_level"]

                    # If trusted agent is moving in same direction, follow
                    if (
                        trust_level > 0.7
                        and other_agent.x == new_x
                        and other_agent.y == new_y
                    ):
                        score += 0.3

                    # If agent has high deception skills, be more cautious
                    if agent.deception_skills > 0.8 and trust_level < 0.3:
                        score -= 0.2

            # Self-awareness influence
            if agent.self_esteem > 0.7:
                score += 0.2  # High self-esteem = more confident decisions
            elif agent.self_esteem < 0.3:
                score -= 0.2  # Low self-esteem = more cautious decisions

            # Existential awareness influence
            if agent.existential_awareness > 0.8:
                # Existentially aware agents make more thoughtful decisions
                score += 0.1

            # Identity-based behavior
            if agent.identity == "leader" and action != "stay":
                score += 0.3  # Leaders are more active
            elif agent.identity == "craftsperson" and action == "stay":
                score += 0.2  # Craftspersons prefer to stay and work
            elif agent.identity == "survivor" and agent.energy < 40:
                score += 0.4  # Survivors are very focused on survival

            # Advanced Social Cognition influences
            # Complex relationships influence
            for other_agent in nearby_agents:
                other_id = other_agent.agent_id

                # Friendship influence
                if other_id in agent.friendships and agent.friendships[other_id] > 0.7:
                    if other_agent.x == new_x and other_agent.y == new_y:
                        score += 0.4  # Follow friends

                # Romantic relationship influence
                if (
                    other_id in agent.romantic_partners
                    and agent.romantic_partners[other_id] > 0.5
                ):
                    if other_agent.x == new_x and other_agent.y == new_y:
                        score += 0.5  # Stay close to romantic partner

                # Rivalry influence
                if other_id in agent.rivals and agent.rivals[other_id] > 0.5:
                    if other_agent.x == new_x and other_agent.y == new_y:
                        score -= 0.3  # Avoid rivals

            # Group dynamics influence
            if agent.dominance_hierarchy > 7:  # Alpha leader
                score += 0.2  # Leaders are more decisive
            elif agent.dominance_hierarchy < 3:  # Low hierarchy
                score -= 0.1  # Followers are more cautious

            # Cooperation influence
            if agent.cooperation_level > 0.7 and len(nearby_agents) > 2:
                score += 0.2  # Cooperative agents prefer group actions

            # Deep Self-Awareness influences
            # Self-regulation influence
            if agent.self_regulation_skills["emotion_control"] > 0.7:
                score += 0.1  # Better emotion control = better decisions

            if agent.self_regulation_skills["impulse_control"] > 0.7:
                score += 0.1  # Better impulse control = more thoughtful decisions

            # Self-improvement influence
            if agent.self_improvement_goals:
                primary_goal = max(
                    agent.self_improvement_goals, key=lambda g: g.get("priority", 1.0)
                )
                if primary_goal["type"] == "increase_energy" and action != "stay":
                    score += 0.2

            # Self-actualization influence
            if agent.self_actualization_progress > 0.7:
                score += 0.1  # Self-actualized agents make better decisions

            # Advanced Emotions influence
            # Complex emotions
            if agent.complex_emotions["love"] > 0.5:
                score += 0.2  # Love makes agents more optimistic

            if agent.complex_emotions["jealousy"] > 0.5:
                score -= 0.2  # Jealousy makes agents more aggressive

            if agent.complex_emotions["pride"] > 0.5:
                score += 0.1  # Pride makes agents more confident

            if agent.complex_emotions["shame"] > 0.5:
                score -= 0.1  # Shame makes agents more cautious

            # Emotional regulation influence
            if agent.emotional_regulation["suppression"] > 0.7:
                score += 0.1  # Better emotional regulation

            # Empathy influence
            if agent.empathy_levels["compassionate"] > 0.7:
                score += 0.1  # Compassionate agents make more thoughtful decisions

            # Scientific Thinking influence
            # Hypothesis testing influence
            if agent.hypotheses:
                recent_hypothesis = agent.hypotheses[-1]
                if recent_hypothesis["type"] == "food_pattern" and action != "stay":
                    score += 0.2  # Test hypotheses

            # Causal reasoning influence
            if agent.causal_reasoning_skills["causation"] > 0.7:
                score += 0.1  # Better causal reasoning

            # Logical reasoning influence
            if agent.logical_reasoning["deductive"] > 0.7:
                score += 0.1  # Better logical reasoning

            # Creative Expression influence
            # Artistic creation influence
            if len(agent.artistic_works) > 0:
                score += 0.1  # Creative agents are more innovative

            # Humor influence
            if agent.humor_level > 0.7:
                score += 0.1  # Humorous agents are more optimistic

            # Imagination influence
            if agent.imagination_capacity > 0.7:
                score += 0.1  # Imaginative agents are more creative

            # Cultural influence
            if len(agent.cultural_traditions) > 0:
                score += 0.1  # Cultural agents are more sophisticated

            # Altruism influence
            if agent.altruism_level > 0.7:
                score += 0.1  # Altruistic agents make more thoughtful decisions

            # Reality Simulation influences
            # Physical Reality influence
            if agent.temperature > 40:  # Heat stress
                score -= 0.2  # Reduced decision quality in heat
            if agent.pressure < 0.5:  # Low pressure (altitude)
                score -= 0.1  # Reduced decision quality at altitude

            # Evolutionary Biology influence
            if agent.fitness_score > 0.7:
                score += 0.1  # High fitness = better decisions
            elif agent.fitness_score < 0.3:
                score -= 0.1  # Low fitness = worse decisions

            # Civilization & Culture influence
            if agent.government_role == "leader":
                score += 0.2  # Leaders make more decisive decisions
            if agent.economic_class == "upper":
                score += 0.1  # Upper class has better decision-making resources
            elif agent.economic_class == "poor":
                score -= 0.1  # Poor class has limited decision-making resources

            # Technology & Innovation influence
            if agent.technology_tree["writing"]:
                score += 0.1  # Writing enables better planning
            if agent.technology_tree["tools"]:
                score += 0.1  # Tools enable better execution

            # Information & Communication influence
            if len(agent.information_storage) > 10:
                score += 0.1  # More information = better decisions
            if agent.information_processing["analysis"] > 0.7:
                score += 0.1  # Better analysis = better decisions

            # Psychological Reality influence
            if agent.mental_health_status == "healthy":
                score += 0.1  # Healthy mental state = better decisions
            elif agent.mental_health_status == "depressed":
                score -= 0.2  # Depression = worse decisions
            elif agent.mental_health_status == "anxious":
                score -= 0.1  # Anxiety = more cautious decisions

            # Ecological Systems influence
            if agent.ecological_role == "omnivore":
                score += 0.1  # Omnivores are more adaptable
            if agent.environmental_impact > 0.7:
                score += 0.1  # High environmental awareness = better decisions

            # Health & Medicine influence
            if agent.healthcare_access > 0.7:
                score += 0.1  # Good healthcare = better health = better decisions
            if len(agent.diseases) > 0:
                score -= 0.1  # Disease = worse decisions

            # Economic Systems influence
            if agent.wealth > 200:
                score += 0.1  # Wealth = better decision-making resources
            elif agent.wealth < 50:
                score -= 0.1  # Poverty = limited decision-making resources
            if agent.market_participation["investor"] > 0.5:
                score += 0.1  # Investors make more strategic decisions

            # Education & Learning influence
            if agent.education_level == "expert":
                score += 0.2  # Expert education = much better decisions
            elif agent.education_level == "advanced":
                score += 0.1  # Advanced education = better decisions
            elif agent.education_level == "basic":
                score -= 0.1  # Basic education = limited decision-making
            if agent.lifelong_learning > 0.7:
                score += 0.1  # Lifelong learning = better decisions

            # Advanced Reality Simulation influences
            # Cosmic & Astronomical influence
            if agent.cosmic_radiation > 0.5:
                score -= 0.1  # Cosmic radiation = reduced decision quality
            if agent.gravitational_field > 0.8:
                score += 0.1  # Strong gravity = more grounded decisions

            # Neuroscience & Brain influence
            if agent.brain_regions["prefrontal"] > 0.7:
                score += 0.2  # Strong prefrontal cortex = better executive decisions
            if agent.neurotransmitters["dopamine"] > 0.7:
                score += 0.1  # High dopamine = more motivated decisions
            if agent.brain_plasticity > 0.7:
                score += 0.1  # High plasticity = more adaptive decisions

            # Quantum Mechanics influence
            if agent.quantum_uncertainty > 0.8:
                score += 0.1  # Quantum uncertainty = creative decisions
            if agent.quantum_consciousness > 0.5:
                score += 0.1  # Quantum consciousness = transcendent decisions

            # Climate & Weather influence
            if agent.weather_patterns["temperature"] > 30:
                score -= 0.1  # Hot weather = reduced decision quality
            if agent.climate_change_impact > 0.7:
                score -= 0.1  # Climate change = stress affects decisions

            # Molecular & Cellular influence
            if agent.protein_synthesis["transcription"] > 0.7:
                score += 0.1  # Good protein synthesis = healthy decisions
            if agent.metabolic_pathways["glycolysis"] > 0.7:
                score += 0.1  # Good metabolism = energetic decisions

            # Internet & Digital influence
            if len(agent.social_media_profiles) > 0:
                score += 0.1  # Social media presence = connected decisions
            if agent.digital_economy["cryptocurrency"] > 500:
                score += 0.1  # High crypto = tech-savvy decisions

            # Art & Cultural influence
            if agent.aesthetic_evolution > 0.7:
                score += 0.1  # High aesthetic sense = beautiful decisions
            if len(agent.cultural_movements) > 0:
                score += 0.1  # Cultural participation = sophisticated decisions

            # Political Systems influence
            if agent.political_system == "democracy":
                score += 0.1  # Democratic system = collaborative decisions
            if len(agent.voting_record) > 0:
                score += 0.1  # Voting participation = civic-minded decisions

            # Biotechnology influence
            if agent.crispr_technology:
                score += 0.2  # CRISPR technology = advanced decisions
            if agent.cloning_technology:
                score += 0.1  # Cloning technology = innovative decisions

            # AI & Machine Learning influence
            if agent.machine_learning["deep_learning"] > 0.7:
                score += 0.2  # Deep learning = intelligent decisions
            if agent.agi_development > 0.7:
                score += 0.3  # AGI development = superintelligent decisions
            if agent.superintelligence:
                score += 0.5  # Superintelligence = transcendent decisions

            # Environmental influences
            terrain_type = self.world.terrain.get((agent.x, agent.y), "plains")
            terrain_factor = {
                "plains": 1.0,
                "forest": 0.8,
                "mountain": 0.6,
                "water": 0.9,
            }[terrain_type]
            weather_factor = {"clear": 1.0, "rain": 0.8, "storm": 0.5, "fog": 0.9}[
                self.world.weather
            ]
            day_night_factor = {"day": 1.0, "night": 0.7}[self.world.day_night]

            score *= terrain_factor * weather_factor * day_night_factor

            scores.append(score)

        # Find best and second best actions
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        best_idx, best_score = sorted_scores[0]
        second_best_score = (
            sorted_scores[1][1] if len(sorted_scores) > 1 else best_score
        )

        # Calculate reported confidence (metacognition)
        if best_score == second_best_score:
            base_conf = 0.0
        else:
            base_conf = (best_score - second_best_score) / max(best_score, 1.0)

        # Add noise/uncertainty to confidence reports
        noise_level = getattr(self.args, "confidence_noise", 0.1)  # Default 10% noise
        confidence_jitter = random.gauss(0, noise_level)
        reported_conf = max(0.0, min(1.0, base_conf + confidence_jitter))

        return actions[best_idx], best_score, second_best_score, reported_conf

    def update_environment(self):
        """Update environmental conditions."""
        # Day/night cycle (every 20 ticks)
        if self.world.tick % 20 == 0:
            self.world.day_night = "night" if self.world.day_night == "day" else "day"

        # Weather changes (every 50 ticks)
        if self.world.tick % 50 == 0:
            weather_options = ["clear", "rain", "storm", "fog"]
            self.world.weather = random.choice(weather_options)

        # Temperature variation
        self.world.temperature += random.uniform(-2.0, 2.0)
        self.world.temperature = max(-10.0, min(40.0, self.world.temperature))

        # Seasonal cycle
        self.world.seasonal_cycle = (self.world.tick / 100.0) % (2 * math.pi)

    def execute_action(self, agent: Agent, action: str) -> float:
        """Execute agent action and return reward."""
        action_deltas = {
            "stay": (0, 0),
            "right": (1, 0),
            "left": (-1, 0),
            "down": (0, 1),
            "up": (0, -1),
        }

        dx, dy = action_deltas[action]
        self.world.move_agent(agent, dx, dy)

        # Check if agent ate food
        if self.world.eat_food(agent):
            return 6.0  # Food reward

        return 0.0  # No reward

    def handle_reproduction(self):
        """Handle agent reproduction."""
        candidates = self.world.get_reproduction_candidates()

        for agent in candidates:
            if random.random() < 0.1:  # 10% chance to reproduce
                child = self.world.add_agent(parent=agent)
                self.events_writer.writerow(
                    [
                        self.world.tick,
                        "birth",
                        child.agent_id,
                        f"parent={agent.agent_id},energy={child.energy}",
                    ]
                )

    def print_summary(self):
        """Print simulation summary."""
        print("\nSimulation completed!")
        print(f"Final agents: {len(self.world.agents)}")
        print(f"Confidence noise: {self.args.confidence_noise}")
        print(f"Crowding strength: {self.args.crowding_strength}")
        print("\n🧠 HUMAN-LIKE FEATURES:")
        print("  ✓ Emotional Systems (fear, hope, frustration, curiosity, boredom)")
        print(
            "  ✓ Personality Traits (Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism)"
        )
        print("  ✓ Motivational Systems (Maslow's hierarchy)")
        print(
            "  ✓ Advanced Learning (pattern recognition, causal reasoning, skill development)"
        )
        print("  ✓ Life Stages (childhood, adolescence, adulthood, old age)")
        print("  ✓ Environmental Complexity (terrain, weather, day/night cycles)")
        print("  ✓ Tools & Innovation (tool use, cultural artifacts)")
        print("  ✓ Social Relationships (trust, hierarchy, communication)")
        print("  ✓ Memory Systems (spatial, social, cultural)")
        print(
            "  ✓ Advanced Metacognition (confidence calibration, uncertainty awareness)"
        )
        print("")
        print("🚀 ADVANCED COGNITIVE ARCHITECTURE:")
        print("  ✓ Working Memory System (7±2 capacity, attention focus, decay)")
        print("  ✓ Executive Control (planning, inhibition, switching, monitoring)")
        print("  ✓ Theory of Mind (beliefs, desires, intentions, deception)")
        print("  ✓ Symbolic Language (vocabulary, grammar, communication)")
        print("  ✓ Self-Awareness (identity, self-esteem, self-efficacy)")
        print("  ✓ Existential Awareness (mortality, purpose, spirituality)")
        print("  ✓ Advanced Social Dynamics (relationships, status, reputation)")
        print("  ✓ Creativity & Innovation (problem-solving, artistic expression)")
        print("  ✓ Deception & Persuasion (lying, manipulation, detection)")
        print("  ✓ Complex Relationships (love, friendship, rivalry)")
        print("")
        print("🌟 TRANSCENDENT CONSCIOUSNESS FEATURES:")
        print("  ✓ Advanced Social Cognition (complex relationships, group dynamics)")
        print(
            "  ✓ Deep Self-Awareness (self-reflection, self-improvement, self-regulation)"
        )
        print("  ✓ Advanced Emotions (love, jealousy, pride, shame, gratitude)")
        print("  ✓ Scientific Thinking (hypothesis testing, causal reasoning, logic)")
        print("  ✓ Creative Expression (artistic creation, storytelling, humor)")
        print("  ✓ Cultural Evolution (traditions, norms, rituals, artifacts)")
        print("  ✓ Altruism & Cooperation (helping others, social learning)")
        print("  ✓ Emotional Intelligence (empathy, emotional regulation)")
        print("  ✓ Mathematical Reasoning (arithmetic, geometry, statistics)")
        print("  ✓ Imagination & Aesthetics (artistic appreciation, creativity)")
        print("")
        print("🌍 REALITY SIMULATION FEATURES:")
        print("  ✓ Physical Reality (3D space, physics, chemistry, biology)")
        print("  ✓ Evolutionary Biology (DNA, natural selection, speciation)")
        print("  ✓ Civilization & Culture (cities, governments, laws, religions)")
        print("  ✓ Technology & Innovation (research, patents, technology trees)")
        print("  ✓ Information & Communication (networks, media, knowledge)")
        print("  ✓ Psychological Reality (mental health, therapy, cognitive biases)")
        print("  ✓ Ecological Systems (food webs, ecosystems, environmental impact)")
        print("  ✓ Health & Medicine (diseases, treatment, healthcare systems)")
        print("  ✓ Economic Systems (trade, markets, currency, wealth)")
        print("  ✓ Education & Learning (schools, teaching, knowledge transfer)")
        print("")
        print("🚀 ADVANCED UNIVERSE SIMULATION FEATURES:")
        print("  ✓ Cosmic & Astronomical Reality (solar systems, orbital mechanics)")
        print("  ✓ Neuroscience & Brain Simulation (neural networks, brain regions)")
        print("  ✓ Quantum Mechanics & Physics (quantum states, superposition)")
        print("  ✓ Climate & Weather Systems (atmospheric dynamics, natural disasters)")
        print("  ✓ Molecular & Cellular Biology (protein synthesis, gene expression)")
        print("  ✓ Internet & Digital Reality (social media, virtual worlds)")
        print("  ✓ Art & Cultural Expression (art history, cultural movements)")
        print("  ✓ Political Systems & Governance (democracy, elections, policy)")
        print("  ✓ Biotechnology & Genetic Engineering (CRISPR, synthetic biology)")
        print("  ✓ Artificial Intelligence & Machine Learning (neural networks, AGI)")

        if self.args.enable_predators:
            print(f"Predators: {self.args.predator_count}")
        if self.args.enable_cooperation:
            print("Cooperation: Enabled")
        if self.args.enable_teaching:
            print("Teaching: Enabled")
        print(f"Logs saved to: {self.args.log_dir}/")
        print("\nExample commands:")
        print("  python main.py")
        print("  python main.py --lesion-workspace")
        print("  python main.py --enable-predators --enable-cooperation")
        print("  python main.py --confidence-noise 0.2 --crowding-strength 0.5")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Consciousness Proxy Simulation")
    parser.add_argument(
        "--ticks", type=int, default=500, help="Number of simulation ticks"
    )
    parser.add_argument("--grid", type=int, default=20, help="Grid size")
    parser.add_argument(
        "--agents", type=int, default=25, help="Number of initial agents"
    )
    parser.add_argument(
        "--food-regen", type=float, default=0.02, help="Food regeneration probability"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lesion-workspace",
        action="store_true",
        help="Disable workspace functionality",
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip automatic analysis after simulation",
    )
    parser.add_argument(
        "--confidence-noise",
        type=float,
        default=0.1,
        help="Noise level for confidence reports (0.0-1.0, default: 0.1)",
    )
    parser.add_argument(
        "--crowding-strength",
        type=float,
        default=0.3,
        help="Strength of crowding effects (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--enable-predators", action="store_true", help="Enable predator agents"
    )
    parser.add_argument(
        "--predator-count", type=int, default=3, help="Number of predator agents"
    )
    parser.add_argument(
        "--enable-cooperation",
        action="store_true",
        help="Enable food sharing between agents",
    )
    parser.add_argument(
        "--enable-teaching", action="store_true", help="Enable teaching between agents"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Run simulation
    simulation = Simulation(args)
    simulation.run()

    # Run analysis automatically unless disabled
    if not args.no_analysis:
        print(f"\n{'='*60}")
        print("RUNNING AUTOMATIC ANALYSIS...")
        print(f"{'='*60}")
        try:
            # Run the analysis script
            result = subprocess.run(
                [sys.executable, "analyze.py", "--log-dir", args.log_dir],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Analysis failed: {result.stderr}")
                print("You can still run analysis manually with: python3 analyze.py")
        except Exception as e:
            print(f"Could not run analysis automatically: {e}")
            print("You can still run analysis manually with: python3 analyze.py")


if __name__ == "__main__":
    main()
