---
title: API.md
date: 2025-10-16
version: draft
Gee6251a1c9
---

# Extended Simulation Framework - API Documentation

## Core Classes

### AgentState
Individual agent state with medical, cognitive, and social attributes.

```python
class AgentState(BaseModel):
    # Physical/Medical
    energy_kJ: float = 2000.0
    mass_kg: float = 70.0
    temp_K: float = 310.15
    hydration: float = 0.6
    damage: float = 0.0
    immune_activation: float = 0.0
    age: int = 25
    healthspan_score: float = 1.0
    lifespan_expectancy: float = 80.0
    
    # Cognitive/Social
    cci: float = 0.7
    valence: float = 0.0
    goals: List[str] = ["survive", "thrive"]
    ethics_profile: Dict[str, float] = {...}
    memory: Dict[str, Any] = {}
    trust_map: Dict[int, float] = {}
    self_model: Dict[str, Any] = {}
    consent_prefs: Dict[str, bool] = {...}
```

### WorldState
Global world state including resources, environment, and social structures.

```python
class WorldState(BaseModel):
    # Resources & Environment
    resources: float = 1000.0
    temperature: float = 295.0
    toxin_level: float = 0.0
    radiation: float = 0.0
    
    # Physical Fields
    fields: Dict[str, float] = {"gravity": 9.81, "electromag": 0.0}
    
    # Disease & Pathogens
    pathogen_pool: Dict[str, float] = {}
    
    # Social Structures
    policies: Dict[str, Any] = {}
    comms_graph: Dict[str, Any] = {}
    norms_state: Dict[str, float] = {...}
    coordination_strength: float = 0.6
    agents: List[Any] = []
```

### ExperimentConfig
Configuration for extended simulation experiments.

```python
class ExperimentConfig(BaseModel):
    # Basic Parameters
    n_agents: Union[int, List[int]] = 100
    timesteps: int = 1000
    dt: float = 1.0
    
    # Shocks
    shocks: List[Dict[str, Any]] = [...]
    
    # Module Configurations
    disease: Dict[str, Union[float, List[float]]] = {...}
    info: Dict[str, Union[float, List[float]]] = {...}
    ethics: Dict[str, Union[str, List[str], float, List[float]]] = {...}
    multiscale: Dict[str, Union[float, List[float], int]] = {...}
    energy: Dict[str, Union[float, List[float]]] = {...}
    
    # Analysis Options
    enable_uq: bool = True
    enable_bayes: bool = False
    valence_weighting: Union[float, List[float]] = 0.5
```

## Main Functions

### run_extended(config: Dict[str, Any]) -> Dict[str, Any]

Main entry point for extended simulation sweeps.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing:
  - `n_agents`: Number of agents (int or list for sweeps)
  - `timesteps`: Simulation duration
  - `dt`: Time step size
  - `shocks`: List of shock specifications
  - `disease`: Disease module parameters
  - `info`: Information layer parameters
  - `ethics`: Ethics module parameters
  - `multiscale`: Multi-scale module parameters
  - `energy`: Energy/thermodynamics parameters
  - `enable_uq`: Enable uncertainty quantification
  - `enable_bayes`: Enable Bayesian inference
  - `use_parallel`: Enable parallel processing
  - `n_workers`: Number of parallel workers
  - `limit_history`: Optimize memory usage
  - `seeds`: List of random seeds

**Returns:**
- `output_dir` (str): Path to results directory
- `n_simulations` (int): Number of simulations completed
- `summary` (Dict[str, Any]): Summary statistics including:
  - `n_simulations`: Total simulations
  - `successful_simulations`: Valid simulations
  - `energy_valid_simulations`: Energy-conserving simulations
  - `avg_survival_rate`: Average survival rate
  - `avg_cci`: Average collective consciousness index
  - `avg_valence`: Average valence
  - `avg_energy_drift`: Average energy drift
  - `module_success_rates`: Success rates per module
  - `analysis_results`: UQ and Bayesian results

**Example:**
```python
config = {
    "n_agents": [50, 100],
    "timesteps": 1000,
    "disease": {"R0": 2.0, "IFR": 0.1},
    "enable_uq": True,
    "use_parallel": True,
    "n_workers": 4
}

result = run_extended(config)
print(f"Results: {result['n_simulations']} simulations")
print(f"Output: {result['output_dir']}")
```

### run_single_simulation(params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]

Run a single simulation with specific parameters.

**Parameters:**
- `params` (Dict[str, Any]): Specific parameter combination
- `config` (Dict[str, Any]): Base configuration

**Returns:**
- `final_metrics` (Dict[str, Any]): Final simulation metrics
- `energy_valid` (bool): Energy conservation status
- `simulation_valid` (bool): Overall validation status
- `energy_drift` (float): Energy drift percentage
- `metrics_history` (List[Dict]): Time series metrics
- `step_results_summary` (Dict[str, str]): Module success status
- `params` (Dict[str, Any]): Parameters used

## Module Functions

### Energy Thermodynamics

#### init_energy(agent: AgentState) -> None
Initialize energy stores and metabolic parameters.

**Parameters:**
- `agent`: Agent to initialize

**Effects:**
- Sets `agent.bmr`: Basal metabolic rate (kJ/day)
- Sets `agent.glycogen_kJ`: Liver glycogen stores
- Sets `agent.fat_kJ`: Fat stores
- Sets `agent.total_energy_kJ`: Total energy
- Sets `agent.metabolic_rate`: Metabolic rate (kJ/hour)
- Sets `agent.work_efficiency`: Work efficiency

#### step_energy(agent: AgentState, world: WorldState, dt: float) -> Dict[str, float]
Step energy dynamics with conservation.

**Parameters:**
- `agent`: Agent to update
- `world`: World state
- `dt`: Time step

**Returns:**
- `work_done`: Work performed (kJ)
- `heat_dissipated`: Heat dissipated (kJ)
- `entropy_proxy`: Entropy proxy
- `net_energy`: Net energy change (kJ)
- `energy_intake`: Energy intake (kJ)

#### conservation_check(population: List[AgentState]) -> float
Check energy conservation across population.

**Parameters:**
- `population`: List of agents

**Returns:**
- Energy drift percentage

### Agent Health

#### init_health(agent: AgentState) -> None
Initialize health parameters.

**Effects:**
- Sets `agent.health_score`: Health quality (0-1)
- Sets `agent.damage`: Accumulated damage (0-1)
- Sets `agent.immune_activation`: Immune activation (0-1)
- Sets `agent.fatigue`: Fatigue level (0-1)
- Sets `agent.metabolic_efficiency`: Metabolic efficiency
- Sets `agent.repair_rate`: Repair rate
- Sets `agent.immune_response_rate`: Immune response rate
- Sets `agent.age_factor`: Age-related factor

#### step_health(agent: AgentState, world: WorldState, dt: float) -> Dict[str, float]
Step health dynamics.

**Returns:**
- `damage`: Current damage level
- `healing`: Healing amount
- `immune_activation`: Immune activation level
- `fatigue`: Fatigue level
- `health_score`: Overall health score
- `metabolic_cost`: Metabolic cost

#### apply_lesion(agent: AgentState, severity: float) -> None
Apply damage/lesion to agent.

**Parameters:**
- `agent`: Agent to damage
- `severity`: Damage severity (0-1)

#### heal(agent: AgentState, rate: float) -> None
Apply healing to agent.

**Parameters:**
- `agent`: Agent to heal
- `rate`: Healing rate

### Disease Epidemic

#### init_disease(agent: AgentState, disease_config: Dict[str, Any]) -> None
Initialize disease state for agent.

**Parameters:**
- `agent`: Agent to initialize
- `disease_config`: Disease configuration

**Effects:**
- Sets `agent.disease_state`: DiseaseState object with:
  - `susceptible`: Susceptible status
  - `exposed`: Exposed status
  - `infectious`: Infectious status
  - `recovered`: Recovered status
  - `vaccinated`: Vaccinated status
  - `comorbidity_factor`: Comorbidity multiplier

#### step_disease(agents: List[AgentState], world: WorldState, dt: float, disease_config: Dict[str, Any]) -> Dict[str, Any]
Step disease dynamics across population.

**Parameters:**
- `agents`: List of agents
- `world`: World state
- `dt`: Time step
- `disease_config`: Disease configuration

**Returns:**
- `S`: Susceptible count
- `E`: Exposed count
- `I`: Infectious count
- `R`: Recovered count
- `V`: Vaccinated count
- `R_eff`: Effective reproduction number
- `new_infections`: New infections
- `deaths`: Deaths
- `recoveries`: Recoveries
- `infection_rate`: Infection rate

### Information Layer

#### init_info_layer(agents: List[AgentState], world: WorldState) -> None
Initialize information layer and communication network.

**Effects:**
- Creates `world.comms_graph`: NetworkX communication graph
- Initializes `agent.trust_map`: Trust relationships

#### step_info_layer(agents: List[AgentState], world: WorldState, dt: float, info_config: Dict[str, Any]) -> Dict[str, Any]
Step information propagation.

**Returns:**
- `avg_trust`: Average trust level
- `info_accuracy`: Information accuracy
- `misinformation_count`: Misinformation count
- `messages_sent`: Messages sent
- `echo_chamber_index`: Echo chamber clustering
- `network_density`: Network density

### Ethics & Norms

#### init_ethics(agents: List[AgentState], world: WorldState, ethics_config: Dict[str, Any]) -> None
Initialize ethics and norms.

**Effects:**
- Sets `world.norms_state`: Global norms
- Initializes `agent.ethics_profile`: Individual ethics

#### step_ethics(agents: List[AgentState], world: WorldState, dt: float, ethics_config: Dict[str, Any]) -> Dict[str, Any]
Step ethics and norms evolution.

**Returns:**
- `fairness_score`: Fairness measure
- `consent_violations`: Consent violations
- `ethics_diversity`: Ethics diversity
- `ethics_stability`: Ethics stability
- `avg_fairness`: Average fairness
- `avg_harm_minimization`: Average harm minimization
- `avg_consent_threshold`: Average consent threshold
- `avg_reciprocity`: Average reciprocity

### Multi-scale Coupling

#### init_multiscale(agents: List[AgentState], world: WorldState, multiscale_config: Dict[str, Any]) -> None
Initialize multi-scale coupling.

**Effects:**
- Creates `agent.cells`: List of CellAgent objects
- Sets `agent.coupling_coeff`: Coupling coefficient
- Sets `agent.cell_energy_total`: Total cell energy
- Sets `agent.cell_damage_total`: Total cell damage

#### step_multiscale(agents: List[AgentState], world: WorldState, dt: float, multiscale_config: Dict[str, Any]) -> Dict[str, Any]
Step multi-scale dynamics.

**Returns:**
- `cell_coherence_micro`: Micro-scale coherence
- `cell_coherence_macro`: Macro-scale coherence
- `collapse_propagation_delay`: Collapse delay
- `avg_cell_health`: Average cell health
- `coordination_strength`: Coordination strength

### Phenomenology

#### init_phenomenology(agents: List[AgentState], world: WorldState) -> None
Initialize phenomenology tracking.

**Effects:**
- Sets `agent.valence`: Affective valence
- Sets `agent.prediction_error`: Prediction error
- Sets `agent.social_standing`: Social standing
- Sets `agent.valence_history`: Valence history

#### step_phenomenology(agents: List[AgentState], world: WorldState, dt: float, valence_weighting: float) -> Dict[str, Any]
Step phenomenology and valence dynamics.

**Returns:**
- `valence_mean`: Mean valence
- `valence_std`: Valence standard deviation
- `valence_correlation_with_cci`: Valence-CCI correlation
- `avg_prediction_error`: Average prediction error
- `well_being_mean`: Mean well-being
- `well_being_std`: Well-being standard deviation
- `cci_mean`: Mean CCI

### Self-Modeling

#### init_self_modeling(agents: List[AgentState], world: WorldState) -> None
Initialize self-modeling for agents.

**Effects:**
- Sets `agent.self_model`: Kalman-like state estimator
- Sets `agent.meta_prediction`: Meta-prediction tracking

#### step_self_modeling(agents: List[AgentState], world: WorldState, dt: float) -> Dict[str, Any]
Step self-modeling dynamics.

**Returns:**
- `self_calibration_mean`: Mean self-calibration
- `self_calibration_std`: Self-calibration standard deviation
- `surprise_rate_mean`: Mean surprise rate
- `surprise_rate_std`: Surprise rate standard deviation
- `planning_horizon_mean`: Mean planning horizon
- `planning_horizon_std`: Planning horizon standard deviation
- `model_confidence_mean`: Mean model confidence

## Utility Functions

### create_output_dir(run_id: str) -> str
Create timestamped output directory with subdirectories.

**Parameters:**
- `run_id`: Unique run identifier

**Returns:**
- Path to output directory

### save_results(data: Any, filepath: str, format: str) -> None
Save results in specified format.

**Parameters:**
- `data`: Data to save
- `filepath`: Output file path
- `format`: Format ("csv", "json", "npy")

### config_hash(config: Dict[str, Any]) -> str
Generate hash for configuration reproducibility.

**Parameters:**
- `config`: Configuration dictionary

**Returns:**
- MD5 hash string

### validate_energy_conservation(energy_history: List[float], tolerance: float = 0.01) -> bool
Validate energy conservation within tolerance.

**Parameters:**
- `energy_history`: Energy time series
- `tolerance`: Maximum allowed drift

**Returns:**
- True if conservation valid

### compute_bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple
Compute bootstrap confidence intervals.

**Parameters:**
- `data`: Data array
- `n_bootstrap`: Number of bootstrap samples
- `alpha`: Significance level

**Returns:**
- (lower_bound, upper_bound) tuple

## Analysis Functions

### run_uq(results_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]
Run uncertainty quantification analysis.

**Parameters:**
- `results_df`: Simulation results DataFrame
- `params`: Parameter ranges

**Returns:**
- `bootstrap_results`: Bootstrap confidence intervals
- `sensitivity_results`: Sensitivity indices
- `tornado_data`: Tornado chart data
- `fragile_parameters`: Most sensitive parameters

### run_bayes_infer(observed_data: Optional[pd.DataFrame], config: Dict[str, Any]) -> Dict[str, Any]
Run Bayesian parameter inference.

**Parameters:**
- `observed_data`: Observed data (optional)
- `config`: Configuration

**Returns:**
- `method`: Inference method used
- `posterior_summary`: Posterior parameter summaries
- `convergence_diagnostics`: Convergence metrics
- `posterior_predictive_checks`: Model validation

## Error Handling

### Common Exceptions

#### ModuleInitializationError
Raised when module initialization fails.

#### EnergyConservationError
Raised when energy conservation is violated.

#### SimulationValidationError
Raised when simulation validation fails.

### Error Recovery

The framework includes comprehensive error handling:

1. **Module-level errors**: Individual module failures don't stop the simulation
2. **Graceful degradation**: Failed modules provide fallback results
3. **Validation tracking**: All simulations are validated and marked
4. **Detailed logging**: Comprehensive error logging and reporting

## Performance Considerations

### Memory Optimization
- `limit_history`: may Reduces memory usage for long simulations
- Efficient data structures with Pydantic
- Streaming data processing for large datasets

### Parallel Processing
- `use_parallel`: Enables multi-core processing
- `n_workers`: Controls parallel worker count
- Automatic load balancing

### Scalability
- Vectorized operations where possible
- Efficient parameter sweep generation
- Optimized data structures

## Best Practices

### Configuration
1. Start with simple configurations
2. Use parameter sweeps for exploration
3. Enable parallel processing for large sweeps
4. Validate results with uncertainty quantification

### Development
1. Follow module interface patterns
2. Include comprehensive error handling
3. Add unit tests for new modules
4. Document all parameters and outputs

### Analysis
1. Check energy conservation warnings
2. Validate simulation success rates
3. Use uncertainty quantification for robustness
4. Generate comprehensive reports






## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
