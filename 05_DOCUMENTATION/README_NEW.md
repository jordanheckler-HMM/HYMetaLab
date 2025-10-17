---
title: README_NEW.md
date: 2025-10-16
version: draft
checksum: 948ac596aa49
---

# Consciousness Simulation with Decision-Level Instrumentation

A comprehensive agent-based simulation for studying consciousness, metacognition, and decision-making with detailed logging and analysis capabilities.

## Features

- **Decision-Level Instrumentation**: Per-decision logging with confidence, workspace usage, and outcome tracking
- **Global Workspace Theory**: Information integration with conflict detection and broadcasting
- **Metacognitive Confidence**: Confidence calibration with noise and decay mechanisms
- **Volatility & Partial Observability**: Configurable environmental uncertainty
- **Workspace Lesion Experiments**: A/B testing with workspace enabled/disabled
- **Innovation & Culture Systems**: Lightweight but active cultural transmission
- **Predictability Probe**: Compatibilist test for free will vs determinism
- **Comprehensive Analysis**: Automated plotting and statistical analysis

## Setup

### Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- PyYAML

### Installation

```bash
pip install numpy pandas matplotlib pyyaml
```

## Project Structure

```
sim/
  __init__.py          # Package initialization
  config.py           # Configuration management
  world.py            # World environment
  agents.py           # Agent implementation
  workspace.py        # Global workspace
  logging_io.py       # Logging utilities
  innovations.py      # Innovation system
  culture.py          # Culture system
  experiments.py      # Experiment runner
  run_experiments.py  # CLI interface

analysis/
  quick_report.py     # Analysis and plotting

configs/
  baseline.yaml       # Baseline configuration
  lesion.yaml         # Workspace lesion experiment
  volatility.yaml     # High volatility experiment

outputs/              # Generated results (created automatically)
```

## Usage

### Running Experiments

```bash
# Run baseline experiment
python -m sim.run_experiments --config configs/baseline.yaml

# Run lesion experiment (workspace disabled)
python -m sim.run_experiments --config configs/lesion.yaml

# Run high volatility experiment
python -m sim.run_experiments --config configs/volatility.yaml
```

### Generating Analysis Reports

```bash
# Generate quick analysis report
python analysis/quick_report.py --run outputs/run_1234_2000t_80a
```

## Output Files

Each experiment generates the following files in `outputs/<run_name>/`:

### JSONL Logs
- **decisions.jsonl**: Per-decision logging with confidence, workspace usage, outcomes
- **integration.jsonl**: Workspace integration events and conflicts
- **innovations.jsonl**: Innovation creation and diffusion
- **culture.jsonl**: Cultural transmission events

### CSV Data
- **lifespans.csv**: Agent lifespan statistics
- **time_series.csv**: Population-level metrics over time
- **predictability_summary.csv**: Free will vs determinism analysis

### Plots (in quick_plots/)
- **population.png**: Population dynamics over time
- **avg_consciousness.png**: Consciousness evolution
- **innovation.png**: Innovation rate over time
- **conf_hist.png**: Confidence distribution histogram
- **reliability_curve.png**: Confidence calibration curve

### Bundle
- **<run_name>_bundle.zip**: Complete results package for upload

## Data Schemas

### decisions.jsonl
```json
{
  "tick": 123,
  "agent_id": "A-42",
  "state_hash": "sha1:...",
  "workspace_reads": 3,
  "workspace_writes": 2,
  "lookahead_depth": 1,
  "candidates": [{"action":"move_e","score":0.61},{"action":"eat","score":0.55}],
  "chosen_action": "move_e",
  "reported_conf": 0.47,
  "outcome_reward": -1.0,
  "prediction_model_p": 0.38,
  "rng_seed_local": 9876543
}
```

### integration.jsonl
```json
{
  "tick": 123,
  "agent_id": "A-42",
  "conflicts": 1,
  "conflict_types": ["planner_vs_reflex"],
  "broadcasts": 2,
  "resolution_time_ms": 0.12
}
```

### innovations.jsonl
```json
{
  "tick": 210,
  "agent_id": "A-17",
  "invention_id": "inv-5",
  "parents": ["inv-1","inv-3"],
  "novelty_score": 0.63,
  "utility_score": 0.40,
  "adopted_by": 7,
  "diffusion_radius": 2
}
```

### culture.jsonl
```json
{
  "tick": 310,
  "agent_id": "A-11",
  "meme_id": "share_food",
  "action": "adopt",
  "source_id": "A-07",
  "trust_change": 0.12,
  "reputation": 0.68,
  "norm_violations": 0
}
```

## Configuration Options

### Core Parameters
- `ticks`: Simulation duration
- `n_agents`: Number of agents
- `seed`: Random seed for reproducibility

### Workspace Costs
- `ws_cost`: Energy cost per workspace read/write
- `lookahead_cost`: Energy cost per lookahead depth

### Metacognition
- `conf_noise_std`: Standard deviation of confidence noise
- `metacog_decay`: Decay rate for metacognitive confidence

### World Parameters
- `volatility_period`: Ticks between volatility shocks
- `volatility_strength`: Strength of volatility shocks
- `observe_radius`: Agent observation radius

### Workspace Settings
- `workspace_enabled`: Enable/disable global workspace

## Key Classes

### Agent
- `perceive()`: Process environmental observations
- `propose_actions()`: Generate action candidates with scores
- `choose_action()`: Select action with confidence calculation
- `confidence()`: Calculate metacognitive confidence
- `apply_costs()`: Apply workspace and lookahead costs

### Workspace
- `read()`: Read from global workspace
- `write()`: Write to global workspace with conflict detection
- `broadcast()`: Broadcast messages to all agents

### World
- `get_observation()`: Get local observation for agent
- `consume_food()`: Consume food at position
- `step()`: Advance world state

## Research Applications

This simulation enables research into:

1. **Consciousness Emergence**: How does consciousness arise from simple rules?
2. **Free Will vs Determinism**: Do agents have genuine free will?
3. **Metacognitive Calibration**: How accurate is agent confidence?
4. **Information Integration**: How does the global workspace create unity?
5. **Innovation Diffusion**: How do innovations spread through populations?
6. **Cultural Transmission**: How do cultural memes evolve and spread?
7. **Predictability**: Can agent behavior be predicted?

## Analysis Capabilities

- **Calibration Analysis**: Confidence vs success rate
- **Population Dynamics**: Agent survival and reproduction
- **Innovation Metrics**: Creation and diffusion rates
- **Cultural Evolution**: Meme transmission and trust networks
- **Predictability Testing**: Free will vs determinism analysis
- **Workspace Integration**: Conflict detection and resolution

## Example Commands

```bash
# Run all three experiments
python -m sim.run_experiments --config configs/baseline.yaml
python -m sim.run_experiments --config configs/lesion.yaml
python -m sim.run_experiments --config configs/volatility.yaml

# Generate analysis for each
python analysis/quick_report.py --run outputs/run_1234_2000t_80a
python analysis/quick_report.py --run outputs/run_1234_2000t_80a_lesion
python analysis/quick_report.py --run outputs/run_1234_2000t_80a_volatility

# Upload ZIP bundles
# outputs/run_1234_2000t_80a_bundle.zip
# outputs/run_1234_2000t_80a_lesion_bundle.zip
# outputs/run_1234_2000t_80a_volatility_bundle.zip
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Check that config files exist in `configs/`
3. **Permission Errors**: Ensure write permissions for `outputs/` directory
4. **Memory Issues**: may Reduce `n_agents` or `ticks` for large experiments

### Performance Tips

- Use smaller `n_agents` for quick testing
- may Reduce `ticks` for rapid iteration
- Monitor memory usage for long experiments
- Use `--no-zip` flag to skip ZIP creation for faster runs

## Contributing

This simulation is designed for research purposes. Key areas for extension:

1. **Advanced Metacognition**: More sophisticated confidence models
2. **Complex Environments**: Multi-dimensional worlds
3. **Social Networks**: Explicit network structures
4. **Learning Algorithms**: Adaptive behavior over time
5. **Neural Integration**: More realistic brain models

## License

This project is designed for academic and research use. Please cite appropriately if used in publications.


## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.

## Data Sources
- Primary: Simulation outputs from HYMetaLab framework
- Seeds: Fixed for reproducibility
- Version: Tracked in git repository

## References
1. HYMetaLab Framework Documentation (internal)
2. Guardian v4 Validation System
3. Reproducibility standards per SOP v1.1

**Citation Format:** Author (Year). Title. Framework/Journal.
