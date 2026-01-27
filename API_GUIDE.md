# HYMetaLab API Guide

**Safe experiment authoring and best practices**

This guide shows you how to write experiments, extend the simulation framework, and work safely with HYMetaLab's core systems. Follow these patterns to maintain research integrity and Guardian compliance.

---

## Quick Navigation

- [Experiment Authoring](#experiment-authoring)
- [Working with Agents](#working-with-agents)
- [Guardian Integration](#guardian-integration)
- [AutoLab Integration](#autolab-integration)
- [Data Access Patterns](#data-access-patterns)
- [Safety Guidelines](#safety-guidelines)
- [Common Pitfalls](#common-pitfalls)

---

## Experiment Authoring

### Basic Experiment Structure

Every experiment is a YAML file with five required sections:

```yaml
# 1. Metadata
name: "Your Experiment Name"
category: "cci_experiments"  # or meaning_experiments, time_experiments, integration_tests
research_question: "Clear, testable hypothesis"

# 2. Agent Configuration
agents:
  count: 10
  # agent-specific parameters

# 3. Simulation Parameters
simulation:
  duration: 1000
  # simulation-specific parameters

# 4. Validation Criteria
validation:
  min_coherence: 0.85
  max_noise: 0.20
  # expected outcomes

# 5. Output Configuration
output:
  save_frequency: 100
  visualizations: ["trajectories", "metrics"]
```

### Minimal Working Example

```yaml
name: "Minimal CCI Test"
category: "cci_experiments"
research_question: "Does CCI emerge in minimal configuration?"

agents:
  count: 5
  cci_enabled: true

simulation:
  duration: 500

validation:
  min_coherence: 0.85
  max_noise: 0.20

output:
  save_frequency: 100
```

**Run it:**
```bash
python -m src.simulation_runner --experiment minimal_cci_test.yaml
```

### Adding Custom Parameters

Extend the basic structure with research-specific parameters:

```yaml
agents:
  count: 10
  cci_enabled: true
  learning_rate: 0.01        # Custom: how fast agents adapt
  memory_depth: 50           # Custom: how much history retained
  initial_context: "math"    # Custom: starting domain

simulation:
  duration: 1000
  interaction_frequency: 10  # Custom: interactions per timestep
  context_switches: [250, 500, 750]  # Custom: when to switch contexts
  perturbations:             # Custom: disruptions to test robustness
    - {type: "noise", time: 400, intensity: 0.3}
    - {type: "agent_removal", time: 600, count: 2}
```

**Accessing custom parameters in code:**

```python
# In simulation_runner.py or custom analysis
config = load_yaml("your_experiment.yaml")
learning_rate = config['agents']['learning_rate']
context_switches = config['simulation']['context_switches']
```

---

## Working with Agents

### Agent API

Agents in HYMetaLab expose these key methods:

```python
from src.main import Agent

# Create an agent
agent = Agent(
    agent_id="agent_1",
    initial_context="mathematical",
    cci_enabled=True,
    memory_depth=50
)

# Core methods
agent.perceive(environment)           # Gather information from environment
agent.interact(other_agent)           # Exchange information with another agent
agent.update_internal_state()         # Process information and update state
agent.cross_contextual_inference()    # Perform CCI if enabled

# Measurement methods
coherence = agent.measure_coherence()
semantic_state = agent.get_semantic_state()
temporal_state = agent.get_temporal_state()
```

### Agent Lifecycle

```python
# 1. Initialization
agent = Agent(agent_id="agent_1", cci_enabled=True)

# 2. Per-Timestep Loop
for t in range(simulation_duration):
    # Perceive environment
    observations = agent.perceive(environment)
    
    # Interact with other agents
    for other in nearby_agents:
        agent.interact(other)
    
    # Update internal state
    agent.update_internal_state()
    
    # Optional: perform CCI if context changed
    if context_changed:
        agent.cross_contextual_inference(new_context)
    
    # Record metrics
    metrics[t] = {
        'coherence': agent.measure_coherence(),
        'cci_score': agent.cci_score,
        'semantic_state': agent.get_semantic_state()
    }

# 3. Cleanup
results = agent.get_full_history()
```

### Custom Agent Behaviors

Extend agents with custom behavior:

```python
from src.main import Agent

class CustomAgent(Agent):
    def __init__(self, *args, custom_param=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_param = custom_param
        self.custom_metric = 0.0
    
    def update_internal_state(self):
        """Override to add custom logic"""
        # Call parent implementation
        super().update_internal_state()
        
        # Add custom behavior
        self.custom_metric = self.calculate_custom_metric()
        
    def calculate_custom_metric(self):
        """Your custom measurement logic"""
        return sum(self.semantic_state.values()) / len(self.semantic_state)
```

**Use in experiments:**

```python
# In your experiment runner
from your_module import CustomAgent

agent = CustomAgent(
    agent_id="agent_1",
    custom_param=42,
    cci_enabled=True
)
```

### Agent Communication Patterns

#### Pairwise Communication
```python
def pairwise_interaction(agent_a, agent_b):
    """Bidirectional information exchange"""
    # A shares with B
    message_ab = agent_a.create_message()
    agent_b.receive_message(message_ab)
    
    # B shares with A
    message_ba = agent_b.create_message()
    agent_a.receive_message(message_ba)
```

#### Broadcast Communication
```python
def broadcast_interaction(sender, receivers):
    """One-to-many information sharing"""
    message = sender.create_message()
    for receiver in receivers:
        receiver.receive_message(message)
```

#### Network-Based Communication
```python
def network_interaction(agent, network):
    """Interact with network neighbors only"""
    neighbors = network.get_neighbors(agent.id)
    for neighbor in neighbors:
        pairwise_interaction(agent, neighbor)
```

---

## Guardian Integration

### Guardian Validation Workflow

Every experiment runs through Guardian validation:

```
Pre-Execution → Run Simulation → Post-Execution → Results
     ↓                                ↓
 Validate config              Validate metrics
 Check thresholds             Check coherence ≥0.85
 Charter compliance           Check noise ≤0.20
```

### Pre-Execution Validation

Guardian validates your experiment configuration before running:

```python
from src.guardian import Guardian

guardian = Guardian(config_path="configs/guardian_config.yaml")

# Validate experiment before running
validation_result = guardian.validate_pre_execution(experiment_config)

if not validation_result.passed:
    print(f"Validation failed: {validation_result.errors}")
    # Fix configuration and retry
else:
    # Safe to run experiment
    run_experiment(experiment_config)
```

**What Guardian checks:**
- Parameter bounds (e.g., agent_count > 0)
- Required fields present
- Threshold compliance
- Charter approval for special cases

### Post-Execution Validation

Guardian validates results after experiment completes:

```python
# After simulation
results = run_simulation(config)

# Validate results
validation = guardian.validate_post_execution(results)

if validation.passed:
    print("✓ Experiment passed Guardian validation")
    print(f"  Coherence: {validation.coherence:.3f} (≥0.85 required)")
    print(f"  Noise: {validation.noise:.3f} (≤0.20 required)")
else:
    print("✗ Experiment failed Guardian validation")
    print(f"  Reasons: {validation.failure_reasons}")
```

### Setting Custom Thresholds

**IMPORTANT:** Threshold changes require charter approval.

```yaml
# In your experiment YAML
validation:
  min_coherence: 0.85      # Standard threshold
  max_noise: 0.20          # Standard threshold
  
  # Custom thresholds (requires charter approval)
  custom_thresholds:
    min_cci_score: 0.75
    max_semantic_drift: 0.15
    
  charter_approval: "CHARTER-2024-001"  # Required for custom thresholds
```

**Checking approval:**
```python
if 'charter_approval' in config['validation']:
    approval_id = config['validation']['charter_approval']
    if not guardian.verify_charter_approval(approval_id):
        raise PermissionError("Invalid charter approval")
```

### Guardian Fallback Mode

If Guardian service is unavailable, the system uses deterministic fallback:

```python
from src.guardian_stub import GuardianStub

try:
    guardian = Guardian()
except ConnectionError:
    # Use deterministic fallback
    guardian = GuardianStub()
    print("Warning: Using Guardian fallback mode")
```

GuardianStub provides basic validation without external dependencies, ensuring zero-downtime operation.

---

## AutoLab Integration

### Generating Experiments Autonomously

AutoLab uses UCB (Upper Confidence Bound) to generate and test hypotheses:

```python
from src.autolab import AutoLab

# Initialize AutoLab
lab = AutoLab(config_path="configs/autolab_config.yaml")

# Pose research question
lab.investigate(
    question="How does network topology affect CCI convergence speed?",
    exploration_factor=2.0,    # UCB exploration parameter
    max_iterations=50,         # Maximum experiments to run
    min_confidence=0.95        # Stop when confident
)

# Get results
best_hypothesis = lab.get_best_hypothesis()
print(f"Best finding: {best_hypothesis.description}")
print(f"Confidence: {best_hypothesis.confidence:.2f}")
print(f"Reward: {best_hypothesis.avg_reward:.3f}")
```

### UCB Algorithm

AutoLab balances exploitation (high reward) vs exploration (uncertainty):

```python
def ucb_score(hypothesis, total_trials, exploration_factor=2.0):
    """
    Upper Confidence Bound formula
    
    UCB = exploit + explore
        = avg_reward + c * sqrt(log(total) / trials)
    """
    exploit = hypothesis.avg_reward
    explore = exploration_factor * sqrt(log(total_trials) / hypothesis.num_trials)
    return exploit + explore

# Select hypothesis with highest UCB score
selected = max(hypotheses, key=lambda h: ucb_score(h, total_trials))
```

### Creating AutoLab-Compatible Hypotheses

Make your experiments discoverable by AutoLab:

```python
from src.autolab import Hypothesis

class CustomHypothesis(Hypothesis):
    def __init__(self, description, parameters):
        super().__init__(description)
        self.parameters = parameters
    
    def generate_experiment_config(self):
        """Convert hypothesis to experiment YAML"""
        return {
            'name': self.description,
            'category': 'autolab_generated',
            'agents': {
                'count': self.parameters['agent_count'],
                # ... other params from hypothesis
            },
            'simulation': {
                'duration': self.parameters['duration'],
                # ... other params
            },
            'validation': {
                'min_coherence': 0.85,
                'max_noise': 0.20
            }
        }
    
    def evaluate_results(self, results):
        """Calculate reward for UCB learning"""
        # Higher reward = better performance
        coherence = results['metrics']['coherence']
        convergence_speed = results['metrics']['convergence_time']
        
        # Combine metrics into single reward
        reward = coherence * (1000 / convergence_speed)  # Fast + coherent = high reward
        return reward
```

### AutoLab Feedback Loop

```python
# AutoLab learning cycle
while not converged:
    # 1. Select hypothesis using UCB
    hypothesis = select_hypothesis(hypotheses, total_trials)
    
    # 2. Generate experiment from hypothesis
    experiment_config = hypothesis.generate_experiment_config()
    
    # 3. Run experiment
    results = run_simulation(experiment_config)
    
    # 4. Validate with Guardian
    validation = guardian.validate_post_execution(results)
    
    # 5. Calculate reward
    if validation.passed:
        reward = hypothesis.evaluate_results(results)
    else:
        reward = 0.0  # Penalize invalid experiments
    
    # 6. Update hypothesis statistics
    hypothesis.update(reward)
    
    # 7. Check convergence
    converged = check_convergence(hypotheses)
```

---

## Data Access Patterns

### Reading Experiment Results

Results are saved in structured JSON format:

```python
import json

# Load results
with open('data/results/cci_experiments/basic_cci/simulation_data.json') as f:
    data = json.load(f)

# Access agent states
agent_states = data['agents']
for agent_id, states in agent_states.items():
    print(f"Agent {agent_id}:")
    for t, state in enumerate(states):
        print(f"  t={t}: coherence={state['coherence']:.3f}")

# Access metrics
metrics = data['metrics']
coherence_over_time = metrics['coherence']
noise_over_time = metrics['noise']
```

### Structured Result Format

```python
{
  "experiment_name": "basic_cci",
  "timestamp": "2024-01-15T14:30:00Z",
  "config": {
    # Full experiment configuration
  },
  "agents": {
    "agent_1": [
      {"t": 0, "coherence": 0.45, "cci_score": 0.0, "context": "math"},
      {"t": 1, "coherence": 0.48, "cci_score": 0.12, "context": "math"},
      # ... timestep data
    ],
    "agent_2": [ /* ... */ ]
  },
  "metrics": {
    "coherence": [0.45, 0.48, 0.51, ...],
    "noise": [0.18, 0.17, 0.16, ...],
    "cci_score": [0.0, 0.12, 0.24, ...]
  },
  "guardian_validation": {
    "pre_execution": {"passed": true},
    "post_execution": {
      "passed": true,
      "coherence": 0.89,
      "noise": 0.15
    }
  }
}
```

### Efficient Data Processing

For large datasets, use incremental processing:

```python
import json

def process_large_results(filepath):
    """Process results without loading entire file into memory"""
    with open(filepath) as f:
        # Load only metrics (small)
        data = json.load(f)
        metrics = data['metrics']
        
        # Process incrementally
        coherence_avg = sum(metrics['coherence']) / len(metrics['coherence'])
        noise_avg = sum(metrics['noise']) / len(metrics['noise'])
        
        return coherence_avg, noise_avg

# For very large files, use streaming
def stream_process(filepath):
    with open(filepath) as f:
        for line in f:
            # Process line by line
            pass
```

### Cross-Experiment Analysis

Compare results across multiple experiments:

```python
import pandas as pd
from pathlib import Path

def compare_experiments(experiment_dir):
    """Load and compare multiple experiment runs"""
    results = []
    
    for exp_path in Path(experiment_dir).glob("*/simulation_data.json"):
        with open(exp_path) as f:
            data = json.load(f)
            results.append({
                'name': data['experiment_name'],
                'coherence': data['metrics']['coherence'][-1],  # Final value
                'noise': data['metrics']['noise'][-1],
                'cci_score': data['metrics']['cci_score'][-1]
            })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    return df

# Usage
df = compare_experiments("data/results/cci_experiments/")
print(df.describe())
print(df.sort_values('cci_score', ascending=False))
```

---

## Safety Guidelines

### DO: Safe Practices

✅ **Always validate with Guardian before running experiments**
```python
validation = guardian.validate_pre_execution(config)
if validation.passed:
    run_experiment(config)
```

✅ **Use standard thresholds (coherence ≥0.85, noise ≤0.20)**
```yaml
validation:
  min_coherence: 0.85
  max_noise: 0.20
```

✅ **Document custom parameters in experiment description**
```yaml
name: "CCI with Custom Learning Rate"
research_question: "How does learning_rate=0.05 affect CCI emergence?"
```

✅ **Version control your experiments**
```bash
git add experiments/cci_experiments/my_experiment.yaml
git commit -m "Add experiment testing learning rate effect"
```

✅ **Save results with timestamps**
```python
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"results/my_experiment/{timestamp}/"
```

✅ **Check Guardian logs after experiments**
```python
guardian_log = data['guardian_validation']
if not guardian_log['post_execution']['passed']:
    print("Warning: Experiment failed validation")
```

### DON'T: Dangerous Patterns

❌ **Don't modify main.py without understanding cascading effects**
```python
# BAD: Changing core Agent structure
class Agent:
    def __init__(self):
        # Removing this field breaks 100+ experiments
        # self.cci_score = 0.0  
        pass
```

❌ **Don't bypass Guardian validation**
```python
# BAD: Running without validation
run_experiment(config)  # Skips safety checks!

# GOOD: Always validate
if guardian.validate_pre_execution(config).passed:
    run_experiment(config)
```

❌ **Don't change thresholds without charter approval**
```yaml
# BAD: Custom thresholds without approval
validation:
  min_coherence: 0.60  # Too low! Requires charter approval
```

❌ **Don't create circular dependencies**
```python
# BAD: Agent A depends on Agent B depends on Agent A
agent_a.set_dependency(agent_b)
agent_b.set_dependency(agent_a)  # Circular!
```

❌ **Don't modify results files manually**
```bash
# BAD: Editing results
vim data/results/cci_experiments/basic_cci/simulation_data.json

# GOOD: Re-run experiment if needed
python -m src.simulation_runner --experiment basic_cci.yaml
```

❌ **Don't use mutable default arguments**
```python
# BAD: Mutable default
def create_agent(params={}):
    params['id'] = generate_id()  # Modifies shared default!
    return Agent(**params)

# GOOD: Use None
def create_agent(params=None):
    if params is None:
        params = {}
    params['id'] = generate_id()
    return Agent(**params)
```

---

## Common Pitfalls

### Pitfall 1: Insufficient Simulation Duration

**Problem:** Simulation too short to observe phenomena
```yaml
# BAD: Too short
simulation:
  duration: 100  # CCI emergence takes ~300 cycles
```

**Solution:** Use research-appropriate durations
```yaml
# GOOD: Based on literature
simulation:
  duration: 1000  # Allows full CCI emergence and stabilization
```

### Pitfall 2: Forgetting Context in CCI Experiments

**Problem:** No context switches to test CCI
```yaml
# BAD: No context switches
agents:
  cci_enabled: true
simulation:
  duration: 1000
  # Missing: context_switches
```

**Solution:** Define context switches explicitly
```yaml
# GOOD: Tests CCI properly
agents:
  cci_enabled: true
  initial_context: "mathematical"
simulation:
  duration: 1000
  context_switches: [250, 500, 750]  # Switch at these timesteps
```

### Pitfall 3: Ignoring Guardian Failures

**Problem:** Not checking validation results
```python
# BAD: Ignoring Guardian
results = run_experiment(config)
save_results(results)  # Might be invalid!
```

**Solution:** Always check Guardian status
```python
# GOOD: Check before saving
results = run_experiment(config)
validation = guardian.validate_post_execution(results)

if validation.passed:
    save_results(results)
    print("✓ Valid results saved")
else:
    print(f"✗ Invalid results: {validation.failure_reasons}")
    # Don't save invalid data
```

### Pitfall 4: Memory Depth Too Small

**Problem:** Insufficient memory for temporal experiments
```yaml
# BAD: Memory too small for causal inference
agents:
  memory_depth: 10  # Paper shows ≥50 required
```

**Solution:** Use research-validated depths
```yaml
# GOOD: Based on memory paper findings
agents:
  memory_depth: 50  # Minimum for accurate causal inference
```

### Pitfall 5: Missing Output Configuration

**Problem:** No metrics saved
```yaml
# BAD: No output config
validation:
  min_coherence: 0.85
# Missing: output section
```

**Solution:** Always specify output
```yaml
# GOOD: Complete output config
validation:
  min_coherence: 0.85

output:
  save_frequency: 100
  metrics: ["coherence", "noise", "cci_score"]
  visualizations: ["trajectories", "metrics"]
```

### Pitfall 6: Hardcoding File Paths

**Problem:** Non-portable paths
```python
# BAD: Hardcoded absolute path
results = load_json("/Users/jordan/HYMetaLab/results/cci/data.json")
```

**Solution:** Use relative paths and Path objects
```python
# GOOD: Portable path handling
from pathlib import Path

project_root = Path(__file__).parent.parent
results_dir = project_root / "data" / "results" / "cci"
results = load_json(results_dir / "data.json")
```

### Pitfall 7: Not Using AutoLab for Parameter Sweeps

**Problem:** Manual parameter exploration
```python
# BAD: Manual loop (inefficient)
for learning_rate in [0.001, 0.005, 0.01, 0.05, 0.1]:
    config['agents']['learning_rate'] = learning_rate
    run_experiment(config)
```

**Solution:** Let AutoLab explore intelligently
```python
# GOOD: AutoLab finds optimal parameters
lab = AutoLab()
lab.investigate(
    question="What learning rate optimizes CCI emergence?",
    exploration_factor=2.0
)
best = lab.get_best_hypothesis()
```

---

## Advanced Patterns

### Pattern 1: Experiment Inheritance

Extend existing experiments:

```yaml
# base_cci.yaml
name: "Base CCI"
agents:
  count: 10
  cci_enabled: true
simulation:
  duration: 1000

---
# extended_cci.yaml (inherits from base_cci.yaml)
extends: "base_cci.yaml"
name: "Extended CCI with Custom Network"
agents:
  network_topology: "scale_free"  # Adds to base config
simulation:
  duration: 2000  # Overrides base config
```

### Pattern 2: Multi-Stage Experiments

Run experiments in phases:

```python
def multi_stage_experiment(config):
    # Stage 1: Initialization (no perturbations)
    stage1_config = config.copy()
    stage1_config['simulation']['duration'] = 500
    results_stage1 = run_simulation(stage1_config)
    
    # Stage 2: Perturbation (inject noise)
    stage2_config = config.copy()
    stage2_config['simulation']['duration'] = 500
    stage2_config['simulation']['initial_state'] = results_stage1['final_state']
    stage2_config['simulation']['perturbations'] = [
        {'type': 'noise', 'time': 100, 'intensity': 0.3}
    ]
    results_stage2 = run_simulation(stage2_config)
    
    # Stage 3: Recovery (remove perturbations)
    stage3_config = config.copy()
    stage3_config['simulation']['duration'] = 500
    stage3_config['simulation']['initial_state'] = results_stage2['final_state']
    results_stage3 = run_simulation(stage3_config)
    
    return {
        'stage1': results_stage1,
        'stage2': results_stage2,
        'stage3': results_stage3
    }
```

### Pattern 3: Parallel Experiment Execution

Run multiple experiments concurrently:

```python
from multiprocessing import Pool
from pathlib import Path

def run_experiment_wrapper(experiment_path):
    """Wrapper for parallel execution"""
    config = load_yaml(experiment_path)
    results = run_simulation(config)
    validation = guardian.validate_post_execution(results)
    return experiment_path.name, validation.passed

def run_experiments_parallel(experiment_dir, num_processes=4):
    """Run all experiments in directory in parallel"""
    experiment_paths = list(Path(experiment_dir).glob("*.yaml"))
    
    with Pool(num_processes) as pool:
        results = pool.map(run_experiment_wrapper, experiment_paths)
    
    # Summarize results
    passed = sum(1 for _, valid in results if valid)
    print(f"Completed {len(results)} experiments: {passed} passed, {len(results)-passed} failed")
    
    return results
```

---

## Debugging Experiments

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='debug.log'
)

logger = logging.getLogger('HYMetaLab')
logger.debug("Starting experiment with config: %s", config)
```

### Visualize Intermediate States

```python
def debug_visualization(agent, timestep):
    """Visualize agent state during simulation"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(agent.coherence_history)
    plt.title('Coherence Over Time')
    
    plt.subplot(2, 2, 2)
    plt.plot(agent.cci_score_history)
    plt.title('CCI Score Over Time')
    
    plt.subplot(2, 2, 3)
    plt.imshow(agent.semantic_state_matrix)
    plt.title(f'Semantic State at t={timestep}')
    
    plt.subplot(2, 2, 4)
    plt.plot(agent.noise_history)
    plt.title('Noise Over Time')
    
    plt.tight_layout()
    plt.savefig(f'debug/agent_{agent.id}_t{timestep}.png')
    plt.close()
```

### Validate Intermediate States

```python
def validate_during_simulation(agent, timestep):
    """Check agent state is valid during simulation"""
    assert agent.coherence >= 0.0 and agent.coherence <= 1.0, \
        f"Invalid coherence at t={timestep}: {agent.coherence}"
    
    assert len(agent.memory) <= agent.memory_depth, \
        f"Memory overflow at t={timestep}"
    
    if agent.cci_enabled:
        assert agent.cci_score >= 0.0, \
            f"Invalid CCI score at t={timestep}: {agent.cci_score}"
```

---

## Testing Your Experiments

### Unit Test Template

```python
import unittest
from src.simulation_runner import run_simulation
from src.guardian import Guardian

class TestMyExperiment(unittest.TestCase):
    def setUp(self):
        """Setup before each test"""
        self.guardian = Guardian()
        self.base_config = {
            'name': 'test_experiment',
            'category': 'cci_experiments',
            'agents': {'count': 5, 'cci_enabled': True},
            'simulation': {'duration': 100},
            'validation': {'min_coherence': 0.85, 'max_noise': 0.20}
        }
    
    def test_experiment_passes_validation(self):
        """Test that experiment passes Guardian validation"""
        results = run_simulation(self.base_config)
        validation = self.guardian.validate_post_execution(results)
        self.assertTrue(validation.passed)
    
    def test_cci_emergence(self):
        """Test that CCI emerges in sufficient time"""
        results = run_simulation(self.base_config)
        final_cci = results['metrics']['cci_score'][-1]
        self.assertGreater(final_cci, 0.5, "CCI should emerge")
    
    def test_coherence_maintained(self):
        """Test that coherence stays above threshold"""
        results = run_simulation(self.base_config)
        min_coherence = min(results['metrics']['coherence'])
        self.assertGreaterEqual(min_coherence, 0.85)

if __name__ == '__main__':
    unittest.main()
```

---

## Getting Help

**Configuration issues?**
→ Check [EXPERIMENTS.md](EXPERIMENTS.md) for templates

**Research questions?**
→ See [RESEARCH_MAPPING.md](RESEARCH_MAPPING.md) for code↔paper connections

**Guardian failures?**
→ Review validation logs in `data/guardian_logs/`

**AutoLab questions?**
→ Check AutoLab logs in `data/autolab_logs/`

**Still stuck?**
→ Open an issue with your experiment config and error logs

---

## Summary: Experiment Authoring Checklist

Before running your experiment:

- [ ] Research question is clear and testable
- [ ] Experiment YAML has all required sections
- [ ] Agent count and parameters are reasonable
- [ ] Simulation duration is sufficient for phenomenon
- [ ] Guardian thresholds are set (or use defaults)
- [ ] Output configuration specifies metrics to track
- [ ] Pre-execution validation passes
- [ ] Experiment is version controlled

After running your experiment:

- [ ] Post-execution validation passed
- [ ] Results are saved with timestamp
- [ ] Guardian logs are checked
- [ ] Visualizations are generated
- [ ] Results are compared with expected outcomes
- [ ] Experiment is documented (if successful)
- [ ] Results are linked to papers (if applicable)

---

**Last Updated:** 2024-01-26  
**Maintained By:** Research Team  
**Related Docs:** [README.md](README.md), [RESEARCH_MAPPING.md](RESEARCH_MAPPING.md), [EXPERIMENTS.md](EXPERIMENTS.md)
