# HYMetaLab

**Guardian-Validated Research Engine for Consciousness, Information, and Time**

A production research infrastructure implementing autonomous hypothesis generation with institutional safety controls. HYMetaLab combines multi-agent simulations, Upper Confidence Bound reinforcement learning, and formal validation to explore fundamental questions about consciousness, meaning, and temporal mechanics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

HYMetaLab is a research platform for investigating three interconnected phenomena:

1. **Cross-Contextual Inference (CCI)** - How agents construct meaning across different contexts
2. **Meaning Formation** - Emergence of semantic structure from interaction patterns
3. **Time Mechanics** - Temporal flow, memory, and causality in multi-agent systems

The system implements a **4,307-line foundational model** (`main.py`) containing 200+ Agent fields that encode the entire theoretical framework. This is complemented by **AutoLab** (autonomous experiment generation using UCB) and **Guardian** (validation enforcement with coherence ≥0.85, noise ≤0.20).

### Key Features

- **Guardian Validation**: Pre/post execution integrity checks with deterministic fallback
- **AutoLab Learning**: UCB-based autonomous hypothesis generation and exploration
- **96+ Experiments**: Comprehensive test suite covering CCI, meaning dynamics, and temporal mechanics
- **Production Monitoring**: Sentinel UI for real-time validation tracking and system health
- **Research Outputs**: Direct connection to published papers and institutional reports

---

## Architecture

```
HYMetaLab/
├── src/
│   ├── main.py                 # Core model (4,307 lines, 200+ Agent fields)
│   ├── simulation_runner.py   # Execution engine with Guardian integration
│   ├── experiment_manager.py  # Experiment orchestration and result collection
│   ├── guardian.py             # Validation enforcement (coherence, noise thresholds)
│   ├── guardian_stub.py        # Deterministic fallback for zero-downtime operation
│   └── autolab.py              # UCB-based autonomous hypothesis generation
├── experiments/
│   ├── cci_experiments/        # Cross-Contextual Inference studies (32 experiments)
│   ├── meaning_experiments/    # Meaning formation studies (28 experiments)
│   ├── time_experiments/       # Temporal mechanics studies (24 experiments)
│   └── integration_tests/      # Multi-phenomenon integration (12 experiments)
├── configs/
│   ├── simulation_config.yaml  # Simulation parameters
│   ├── guardian_config.yaml    # Validation thresholds and rules
│   └── autolab_config.yaml     # UCB parameters and learning settings
├── data/
│   ├── results/                # Experiment outputs
│   ├── autolab_logs/           # Autonomous learning records
│   └── guardian_logs/          # Validation event logs
├── sentinel/                   # Production monitoring UI
│   ├── dashboard.py            # Real-time system health
│   ├── validation_viewer.py   # Guardian log analysis
│   └── drive_sync.py           # Research report synchronization
└── tests/
    ├── test_simulation.py      # Core simulation tests
    ├── test_guardian.py        # Validation system tests
    └── test_autolab.py         # Learning system tests
```

### System Components

#### 1. Core Simulation Engine (`main.py`)

The foundational research model encoding:
- **200+ Agent fields** - Complete theoretical framework (CCI, meaning, time)
- **Interaction mechanics** - How agents exchange information and construct meaning
- **Temporal dynamics** - Time flow, memory persistence, causal structure
- **Context boundaries** - How meaning transfers across different contexts

**Design Note:** This file is intentionally monolithic. The 200+ Agent fields represent the entire research model - decomposing it would destroy the theoretical coherence that 100+ experiments depend on.

#### 2. Guardian Validation System

Institutional safety controls ensuring research integrity:

```python
# Pre-execution validation
def validate_experiment(config):
    assert coherence >= 0.85, "Coherence below safety threshold"
    assert noise <= 0.20, "Noise exceeds acceptable limit"
    assert charter_approved(config), "Experiment requires charter approval"
```

**Thresholds (Charter-Enforced):**
- Coherence: ≥0.85 (ensures meaningful agent interactions)
- Noise: ≤0.20 (prevents random behavior from dominating)
- Change approval: Required for threshold modifications

**Deterministic Fallback:** `guardian_stub.py` ensures zero-downtime validation even if the main Guardian service is unavailable.

#### 3. AutoLab Autonomous Learning

Upper Confidence Bound (UCB) reinforcement learning for hypothesis generation:

```python
def select_hypothesis(hypotheses, exploration_factor=2.0):
    """UCB: balance exploitation (high reward) vs exploration (uncertainty)"""
    ucb_scores = []
    for h in hypotheses:
        exploit = h.avg_reward
        explore = sqrt(log(total_trials) / h.num_trials)
        ucb_scores.append(exploit + exploration_factor * explore)
    return hypotheses[argmax(ucb_scores)]
```

**Capabilities:**
- Autonomous experiment generation from research questions
- Adaptive learning from Guardian feedback
- Exploration-exploitation tradeoff optimization
- Self-improving research system

#### 4. Sentinel Monitoring UI

Production observability infrastructure:
- Real-time Guardian validation monitoring
- AutoLab learning curve visualization
- System health metrics and alerts
- Google Drive sync for research reports

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/jordanheckler-HMM/HYMetaLab.git
cd HYMetaLab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Run a basic CCI experiment
python -m src.simulation_runner \
    --config configs/simulation_config.yaml \
    --experiment experiments/cci_experiments/basic_cci.yaml

# View results
python -m src.experiment_manager --show-results results/latest/
```

### Launch Sentinel Monitoring

```bash
# Start the monitoring dashboard
python -m sentinel.dashboard

# View Guardian validation logs
python -m sentinel.validation_viewer
```

### Run AutoLab Autonomous Learning

```bash
# Start autonomous hypothesis generation
python -m src.autolab \
    --config configs/autolab_config.yaml \
    --research-question "How does context size affect CCI stability?"

# Monitor learning progress
python -m sentinel.dashboard --view autolab
```

---

## Usage Guide

### Writing Experiments

Experiments are YAML configurations that specify:
1. Research question
2. Agent configurations
3. Simulation parameters
4. Expected outcomes (for Guardian validation)

**Example: Basic CCI Experiment**

```yaml
# experiments/cci_experiments/context_transfer.yaml
name: "Context Transfer Study"
research_question: "Can agents maintain meaning across context switches?"

agents:
  - id: "agent_1"
    initial_context: "mathematical"
    cci_enabled: true
  - id: "agent_2"
    initial_context: "linguistic"
    cci_enabled: true

simulation:
  duration: 1000
  context_switches: [250, 500, 750]
  interaction_frequency: 10

validation:
  min_coherence: 0.85
  max_noise: 0.20
  expected_outcome: "meaning_preservation"
```

**Run it:**

```bash
python -m src.simulation_runner \
    --experiment experiments/cci_experiments/context_transfer.yaml \
    --output results/context_transfer/
```

### Understanding Results

Each experiment produces:

```
results/
├── simulation_data.json    # Raw agent states and interactions
├── metrics.json            # Coherence, noise, CCI scores
├── guardian_report.json    # Validation results
└── visualizations/
    ├── agent_trajectories.png
    ├── meaning_dynamics.png
    └── coherence_over_time.png
```

**Key Metrics:**
- **Coherence**: How consistent agent behaviors are (0-1, higher is better)
- **Noise**: Random fluctuations (0-1, lower is better)
- **CCI Score**: Cross-context meaning preservation (0-1)
- **Guardian Status**: `PASS` / `FAIL` / `WARNING`

### Working with AutoLab

AutoLab can autonomously generate and test hypotheses:

```python
from src.autolab import AutoLab

# Initialize AutoLab
lab = AutoLab(config_path="configs/autolab_config.yaml")

# Pose a research question
lab.investigate(
    question="Does increased agent diversity improve meaning stability?",
    exploration_factor=2.0,  # UCB parameter
    max_iterations=50
)

# Review discovered hypotheses
best_hypothesis = lab.get_best_hypothesis()
print(f"Best: {best_hypothesis.description}")
print(f"Reward: {best_hypothesis.avg_reward}")
```

---

## Research Connections

HYMetaLab directly supports the following research programs:

### Cross-Contextual Inference (CCI)

**Papers:**
- "Cross-Contextual Inference: Meaning Preservation Across Domain Boundaries" (2024)
- "CCI in Multi-Agent Systems: Emergence and Stability" (2024)

**Key Experiments:**
- `experiments/cci_experiments/basic_cci.yaml` - Foundational CCI mechanics
- `experiments/cci_experiments/context_transfer.yaml` - Meaning preservation
- `experiments/cci_experiments/cci_stability.yaml` - Long-term CCI robustness

**Code Mapping:**
- CCI implementation: `main.py` lines 1200-1450
- Context switching: `main.py` lines 1500-1650
- Meaning transfer: `main.py` lines 1700-1850

### Meaning Formation

**Papers:**
- "Emergent Semantics in Multi-Agent Interaction Networks" (2024)
- "Meaning Dynamics: From Interaction to Understanding" (2023)

**Key Experiments:**
- `experiments/meaning_experiments/semantic_emergence.yaml`
- `experiments/meaning_experiments/interaction_patterns.yaml`
- `experiments/meaning_experiments/meaning_stability.yaml`

**Code Mapping:**
- Semantic structures: `main.py` lines 2000-2300
- Interaction protocols: `main.py` lines 2350-2550
- Meaning convergence: `main.py` lines 2600-2800

### Time Mechanics

**Papers:**
- "Temporal Flow in Multi-Agent Systems: Memory and Causality" (2024)
- "Time as Emergent Property in Distributed Systems" (2023)

**Key Experiments:**
- `experiments/time_experiments/temporal_flow.yaml`
- `experiments/time_experiments/memory_persistence.yaml`
- `experiments/time_experiments/causal_structure.yaml`

**Code Mapping:**
- Time dynamics: `main.py` lines 3000-3250
- Memory systems: `main.py` lines 3300-3500
- Causality tracking: `main.py` lines 3550-3750

See [RESEARCH_MAPPING.md](RESEARCH_MAPPING.md) for complete code↔paper connections.

---

## Configuration

### Simulation Parameters

Edit `configs/simulation_config.yaml`:

```yaml
simulation:
  timesteps: 1000
  agent_count: 10
  interaction_radius: 5.0
  
agents:
  default_memory_size: 100
  cci_enabled: true
  learning_rate: 0.01

output:
  save_frequency: 100
  metrics: ["coherence", "noise", "cci_score"]
  visualize: true
```

### Guardian Thresholds

Edit `configs/guardian_config.yaml`:

```yaml
validation:
  coherence:
    min: 0.85
    charter_required_for_change: true
  noise:
    max: 0.20
    charter_required_for_change: true
  
enforcement:
  pre_execution: true
  post_execution: true
  fallback_mode: "deterministic"  # Uses guardian_stub.py
```

**IMPORTANT:** Changing validation thresholds requires charter approval per institutional safety protocols.

### AutoLab Learning

Edit `configs/autolab_config.yaml`:

```yaml
ucb:
  exploration_factor: 2.0
  min_trials_per_hypothesis: 5
  convergence_threshold: 0.95

generation:
  max_hypotheses_per_iteration: 10
  hypothesis_mutation_rate: 0.1
  
learning:
  reward_from_guardian: true
  penalize_invalid_experiments: true
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_simulation.py -v
pytest tests/test_guardian.py -v
pytest tests/test_autolab.py -v

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Adding New Experiments

1. Create experiment YAML in appropriate directory
2. Define research question and expected outcomes
3. Run with Guardian validation
4. Document in `EXPERIMENTS.md`

**Template:**

```yaml
name: "Your Experiment Name"
research_question: "What are you investigating?"

agents:
  - id: "agent_1"
    # agent configuration

simulation:
  duration: 1000
  # simulation parameters

validation:
  min_coherence: 0.85
  max_noise: 0.20
  expected_outcome: "description"
```

---

## Project Status

### Current State

- ✅ **Core simulation engine** - Production-ready (4,307 lines, 200+ Agent fields)
- ✅ **Guardian validation** - Fully operational with deterministic fallback
- ✅ **AutoLab learning** - UCB implementation tested and validated
- ✅ **96+ experiments** - Comprehensive coverage of research questions
- ✅ **Sentinel monitoring** - Production observability infrastructure
- 🚧 **Documentation** - In progress (README, research mapping, experiment index)
- 📅 **External collaboration** - Planned after documentation complete

### Roadmap

**Week 1: Documentation** (Current)
- [x] Main README.md (this file)
- [ ] RESEARCH_MAPPING.md - Code↔paper connections
- [ ] EXPERIMENTS.md - Categorized experiment index
- [ ] API_GUIDE.md - Safe experiment authoring

**Week 2: AutoLab Enhancements**
- [ ] Learning curve dashboard
- [ ] Hypothesis tracking system
- [ ] Alert system for high-scoring discoveries

**Week 3: Experiment Discoverability**
- [ ] Experiment templates
- [ ] Quick-run scripts
- [ ] Result analysis tools

**Week 4: Guardian Analytics**
- [ ] Coherence/noise distribution analysis
- [ ] Threshold change audit log
- [ ] Long-term validation trends

---

## Contributing

HYMetaLab is currently in active research use. External contributions are welcome after the documentation phase completes.

**Before Contributing:**
1. Read [RESEARCH_MAPPING.md](RESEARCH_MAPPING.md) to understand code↔paper connections
2. Review [EXPERIMENTS.md](EXPERIMENTS.md) to see existing experiment coverage
3. Follow [API_GUIDE.md](API_GUIDE.md) for safe experiment authoring

**What NOT to Change:**
- ❌ `main.py` structure - 200+ Agent fields are the theoretical foundation
- ❌ Guardian thresholds - Charter approval required
- ❌ Experiment naming conventions - Maintains research continuity

**What We Welcome:**
- ✅ New experiments exploring CCI, meaning, or time mechanics
- ✅ Visualization improvements
- ✅ AutoLab hypothesis strategies
- ✅ Documentation enhancements

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Citation

If you use HYMetaLab in your research, please cite:

```bibtex
@software{hymetalab2024,
  author = {Heckler, Jordan},
  title = {HYMetaLab: Guardian-Validated Research Engine},
  year = {2024},
  url = {https://github.com/jordanheckler-HMM/HYMetaLab}
}
```

---

## Contact

**Jordan Heckler**  
Email: hymetalab@gmail.com 
GitHub: [@yourusername](https://github.com/jordanheckler-HMM)

---

## Acknowledgments

- Antigravity architectural documentation team
- Guardian validation framework contributors
- AutoLab UCB implementation advisors
- Research collaborators on CCI, meaning, and time mechanics papers

---

**Built with intentionality. Research with integrity. Learn with Guardian validation.**
