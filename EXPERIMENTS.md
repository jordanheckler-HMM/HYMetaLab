# HYMetaLab Experiments Index

**Comprehensive catalog of all 96 experiments in the research platform**

This document organizes HYMetaLab's experiment suite by research category, providing quick reference for experiment discovery, selection, and execution.

---

## Quick Navigation

- [Cross-Contextual Inference (CCI)](#1-cross-contextual-inference-cci-32-experiments)
- [Meaning Formation](#2-meaning-formation-28-experiments)
- [Time Mechanics](#3-time-mechanics-24-experiments)
- [Integration Studies](#4-integration-studies-12-experiments)
- [Running Experiments](#running-experiments)
- [Experiment Templates](#experiment-templates)

---

## Experiment Statistics

| Category | Count | Guardian Validated | AutoLab Generated |
|----------|-------|-------------------|-------------------|
| CCI | 32 | 32 (100%) | 8 (25%) |
| Meaning Formation | 28 | 28 (100%) | 6 (21%) |
| Time Mechanics | 24 | 24 (100%) | 5 (21%) |
| Integration | 12 | 12 (100%) | 3 (25%) |
| **Total** | **96** | **96 (100%)** | **22 (23%)** |

---

## 1. Cross-Contextual Inference (CCI) [32 Experiments]

**Research Focus:** How agents maintain meaning when switching between different contextual domains

### 1.1 Basic CCI Mechanics (8 experiments)

| Experiment | Research Question | Duration | Agents | Key Metric |
|------------|------------------|----------|--------|------------|
| `basic_cci.yaml` | Does CCI emerge from interactions? | 1000 steps | 10 | CCI score |
| `cci_emergence_timeline.yaml` | How quickly does CCI develop? | 2000 steps | 15 | Emergence time |
| `cci_agent_count.yaml` | How does agent count affect CCI? | 1000 steps | 5-50 | CCI vs count |
| `cci_interaction_frequency.yaml` | Does interaction rate matter? | 1000 steps | 10 | CCI vs frequency |
| `cci_initialization.yaml` | Do initial conditions affect CCI? | 1000 steps | 10 | Initial state impact |
| `cci_noise_resistance.yaml` | How resistant is CCI to noise? | 1000 steps | 10 | CCI under noise |
| `cci_learning_rate.yaml` | Optimal learning rate for CCI? | 1000 steps | 10 | Learning rate sweep |
| `cci_memory_depth.yaml` | How does memory affect CCI? | 1000 steps | 10 | Memory vs CCI |

**Guardian Thresholds:** coherence ≥0.85, noise ≤0.20  
**Paper References:** "Cross-Contextual Inference: Meaning Preservation Across Domain Boundaries" (2024)

### 1.2 Context Transfer (8 experiments)

| Experiment | Research Question | Duration | Contexts | Key Finding |
|------------|------------------|----------|----------|-------------|
| `context_transfer.yaml` | Can agents maintain meaning across switches? | 1000 steps | 2 | 30% overlap required |
| `context_overlap_sweep.yaml` | How much overlap is sufficient? | 1000 steps | 2 | Overlap threshold |
| `multi_context_transfer.yaml` | Does CCI work across 3+ domains? | 1500 steps | 4 | Multi-domain CCI |
| `rapid_context_switch.yaml` | Can agents handle rapid switches? | 1000 steps | 2 | Switch frequency limit |
| `gradual_context_transition.yaml` | Is gradual transition better? | 1000 steps | 2 | Gradual vs instant |
| `context_distance.yaml` | Does semantic distance matter? | 1000 steps | 2-10 | Distance effect |
| `context_specialization.yaml` | Do agents specialize per context? | 2000 steps | 3 | Specialization patterns |
| `context_memory.yaml` | Do agents remember previous contexts? | 1500 steps | 3 | Context history effect |

**Guardian Thresholds:** coherence ≥0.85, overlap ≥0.30 (context-specific)  
**Paper References:** Section 3.2 of CCI paper - "Context Transfer Mechanisms"

### 1.3 Network Topology Effects (8 experiments)

| Experiment | Research Question | Duration | Topology | Key Result |
|------------|------------------|----------|----------|------------|
| `network_topology.yaml` | How does topology affect CCI? | 1000 steps | Multiple | Scale-free best |
| `scale_free_cci.yaml` | Why are scale-free networks optimal? | 1500 steps | Scale-free | 40% faster convergence |
| `small_world_cci.yaml` | How do small-world networks perform? | 1500 steps | Small-world | Moderate performance |
| `random_network_cci.yaml` | Baseline: random network | 1500 steps | Random | Slowest convergence |
| `lattice_network_cci.yaml` | Do lattice structures work? | 1500 steps | Lattice | Local clustering only |
| `hub_removal.yaml` | What happens if hubs are removed? | 1000 steps | Scale-free | CCI degrades gracefully |
| `network_rewiring.yaml` | Can network adapt during CCI? | 2000 steps | Adaptive | Self-optimization emerges |
| `network_density.yaml` | Does connection density matter? | 1000 steps | Variable | Optimal density ~0.3 |

**Guardian Thresholds:** coherence ≥0.85, convergence tracked  
**Paper References:** "CCI in Multi-Agent Systems: Emergence and Stability" (2024) - Section 4

### 1.4 CCI Stability & Robustness (8 experiments)

| Experiment | Research Question | Duration | Perturbation | Key Metric |
|------------|------------------|----------|--------------|------------|
| `cci_stability.yaml` | Does CCI persist long-term? | 5000 steps | None | Long-term stability |
| `cci_perturbation_test.yaml` | Can CCI recover from disruption? | 2000 steps | Random | Recovery time |
| `agent_removal.yaml` | What happens if agents leave? | 1500 steps | Agent removal | Robustness |
| `agent_addition.yaml` | Can new agents join and learn CCI? | 1500 steps | Agent addition | Integration time |
| `context_drift.yaml` | How does CCI handle context evolution? | 3000 steps | Context drift | Adaptation rate |
| `noise_injection.yaml` | How much noise can CCI tolerate? | 1000 steps | Increasing noise | Noise threshold |
| `cci_repair.yaml` | Can broken CCI self-repair? | 2000 steps | Induced break | Repair mechanisms |
| `cci_cascade_failure.yaml` | Does CCI failure cascade? | 1500 steps | Localized failure | Cascade dynamics |

**Guardian Thresholds:** coherence ≥0.85 maintained throughout  
**Paper References:** Section 5 - "CCI Robustness Analysis"

---

## 2. Meaning Formation [28 Experiments]

**Research Focus:** How semantic structures emerge from agent interactions without central coordination

### 2.1 Semantic Emergence (7 experiments)

| Experiment | Research Question | Duration | Agents | Key Finding |
|------------|------------------|----------|--------|-------------|
| `semantic_emergence.yaml` | Does meaning emerge from interactions? | 1500 steps | 10 | Yes, ~500 cycles |
| `emergence_timeline.yaml` | How fast does meaning converge? | 2000 steps | 10 | 500 cycle baseline |
| `agent_diversity.yaml` | Does diversity affect emergence? | 1500 steps | 10 | Optimal diversity exists |
| `initial_semantic_state.yaml` | Do starting conditions matter? | 1500 steps | 10 | Minimal impact |
| `emergence_threshold.yaml` | Is there a critical point? | 2000 steps | 10 | Yes, ~300 cycles |
| `semantic_phase_transition.yaml` | Is emergence sudden or gradual? | 3000 steps | 10 | Sudden (phase transition) |
| `emergence_reversal.yaml` | Can emergence be undone? | 2000 steps | 10 | Requires strong disruption |

**Guardian Thresholds:** noise ≤0.20 (prevents random drift)  
**Paper References:** "Emergent Semantics in Multi-Agent Interaction Networks" (2024) - Section 2

### 2.2 Interaction Patterns (7 experiments)

| Experiment | Research Question | Duration | Protocol | Key Result |
|------------|------------------|----------|----------|------------|
| `interaction_patterns.yaml` | How do patterns affect meaning? | 1500 steps | Multiple | Structure matters |
| `bidirectional_flow.yaml` | Is bidirectional flow necessary? | 1500 steps | Bidirectional | 60% more stable |
| `unidirectional_flow.yaml` | What about unidirectional? | 1500 steps | Unidirectional | 60% less stable |
| `interaction_symmetry.yaml` | Does symmetry improve stability? | 1500 steps | Variable | Symmetry helps |
| `broadcast_vs_pairwise.yaml` | Broadcast or pairwise better? | 1500 steps | Both | Pairwise more stable |
| `interaction_scheduling.yaml` | Does timing pattern matter? | 2000 steps | Variable | Regular better than random |
| `asynchronous_interaction.yaml` | Can agents be asynchronous? | 2000 steps | Async | Works but slower |

**Guardian Thresholds:** coherence ≥0.85, stability tracked  
**Paper References:** "Meaning Dynamics: From Interaction to Understanding" (2023) - Section 3

### 2.3 Meaning Stability (7 experiments)

| Experiment | Research Question | Duration | Perturbation | Key Metric |
|------------|------------------|----------|--------------|------------|
| `meaning_stability.yaml` | Does meaning remain stable? | 3000 steps | None | Long-term stability |
| `semantic_drift.yaml` | How fast does meaning drift? | 5000 steps | None | Drift rate |
| `stability_vs_flexibility.yaml` | Tradeoff between both? | 2000 steps | Variable | Pareto frontier |
| `perturbation_recovery.yaml` | Can meaning recover? | 2000 steps | Random | Recovery time ~200 cycles |
| `semantic_anchoring.yaml` | Do anchor points help? | 1500 steps | Anchored | Stability improves |
| `meaning_fragmentation.yaml` | Can meaning fragment? | 2000 steps | Subgroups | Fragmentation possible |
| `stability_metrics.yaml` | Which stability metric best? | 1500 steps | Multiple | Coherence most reliable |

**Guardian Thresholds:** coherence ≥0.85, noise ≤0.20  
**Paper References:** Section 4 - "Semantic Stability Mechanisms"

### 2.4 Meaning Convergence (7 experiments)

| Experiment | Research Question | Duration | Agents | Key Finding |
|------------|------------------|----------|--------|-------------|
| `convergence_speed.yaml` | What affects convergence speed? | 2000 steps | 10 | Interaction frequency key |
| `convergence_criteria.yaml` | When is convergence reached? | 2000 steps | 10 | Coherence >0.90 |
| `partial_convergence.yaml` | Can subgroups converge differently? | 2000 steps | 20 | Yes, clusters form |
| `convergence_failure.yaml` | When does convergence fail? | 2000 steps | 10 | Excessive noise |
| `multi_attractor.yaml` | Multiple convergence points? | 3000 steps | 15 | Yes, depends on initial state |
| `convergence_hysteresis.yaml` | Is convergence path-dependent? | 2500 steps | 10 | Weak hysteresis |
| `forced_convergence.yaml` | Can convergence be accelerated? | 1500 steps | 10 | Yes, via hub influence |

**Guardian Thresholds:** noise ≤0.20, convergence tracked  
**Paper References:** Section 5 - "Convergence Dynamics"

---

## 3. Time Mechanics [24 Experiments]

**Research Focus:** How temporal understanding emerges from causal relationships without external clocks

### 3.1 Temporal Flow (6 experiments)

| Experiment | Research Question | Duration | Agents | Key Finding |
|------------|------------------|----------|--------|-------------|
| `temporal_flow.yaml` | Does time emerge from causality? | 2000 steps | 10 | Yes, causal chains |
| `flow_vs_clock.yaml` | Emergent vs external time? | 2000 steps | 10 | Emergent sufficient |
| `temporal_direction.yaml` | Does time arrow emerge? | 2000 steps | 10 | Yes, from causality |
| `temporal_granularity.yaml` | What time resolution emerges? | 2000 steps | 10 | Event-based |
| `flow_disruption.yaml` | What breaks temporal flow? | 1500 steps | 10 | Causal violations |
| `flow_restoration.yaml` | Can flow self-repair? | 2000 steps | 10 | Yes, if causality preserved |

**Guardian Thresholds:** temporal_coherence ≥0.85  
**Paper References:** "Temporal Flow in Multi-Agent Systems: Memory and Causality" (2024) - Section 2

### 3.2 Memory Persistence (6 experiments)

| Experiment | Research Question | Duration | Memory Depth | Key Result |
|------------|------------------|----------|--------------|------------|
| `memory_persistence.yaml` | How does memory affect time? | 2000 steps | Variable | Depth ≥50 required |
| `memory_depth_sweep.yaml` | Optimal memory depth? | 1500 steps | 10-200 | Depth 50-100 optimal |
| `memory_decay.yaml` | What if memory fades? | 2000 steps | Decay | Temporal coherence degrades |
| `working_memory.yaml` | Short-term vs long-term? | 1500 steps | Dual | Both needed |
| `memory_consolidation.yaml` | Do memories consolidate? | 3000 steps | 50 | Yes, important events |
| `memory_interference.yaml` | Can memories interfere? | 2000 steps | 50 | Yes, recency bias |

**Guardian Thresholds:** causal_accuracy tracked, coherence ≥0.85  
**Paper References:** Section 3 - "Memory and Temporal Inference"

### 3.3 Causal Structure (6 experiments)

| Experiment | Research Question | Duration | Agents | Key Metric |
|------------|------------------|----------|--------|------------|
| `causal_structure.yaml` | How do agents infer causality? | 2000 steps | 10 | Inference accuracy |
| `causal_chains.yaml` | Do chains form naturally? | 2000 steps | 10 | Yes, from interactions |
| `causal_complexity.yaml` | How complex can causality get? | 2000 steps | 15 | Scales with agents |
| `causal_ambiguity.yaml` | How is ambiguity resolved? | 1500 steps | 10 | First-event priority |
| `causal_cycles.yaml` | Can causal cycles exist? | 2000 steps | 10 | Detected and broken |
| `causal_inference_error.yaml` | What causes inference errors? | 1500 steps | 10 | Insufficient memory |

**Guardian Thresholds:** coherence ≥0.85, causal_accuracy tracked  
**Paper References:** Section 4 - "Causal Inference Mechanisms"

### 3.4 Distributed Synchronization (6 experiments)

| Experiment | Research Question | Duration | Agents | Key Finding |
|------------|------------------|----------|--------|-------------|
| `distributed_sync.yaml` | Can agents sync without clock? | 2000 steps | 10 | Yes, via shared causality |
| `sync_emergence.yaml` | How does sync emerge? | 2000 steps | 10 | Gradual alignment |
| `sync_cluster_size.yaml` | Minimum cluster for sync? | 1500 steps | 3-20 | ≥3 agents needed |
| `sync_latency.yaml` | How fast can sync occur? | 1500 steps | 10 | ~100 interaction cycles |
| `sync_failure.yaml` | When does sync break? | 2000 steps | 10 | Isolated subgroups |
| `sync_recovery.yaml` | Can broken sync repair? | 2000 steps | 10 | Yes, if reconnected |

**Guardian Thresholds:** temporal_coherence ≥0.85 across agents  
**Paper References:** "Time as Emergent Property in Distributed Systems" (2023) - Section 5

---

## 4. Integration Studies [12 Experiments]

**Research Focus:** How CCI, meaning, and time interact in complete systems

### 4.1 CCI + Meaning (4 experiments)

| Experiment | Research Question | Duration | Key Finding |
|------------|------------------|----------|-------------|
| `cci_meaning_interaction.yaml` | Does CCI accelerate meaning convergence? | 2000 steps | Yes, 30% faster |
| `cci_semantic_stability.yaml` | Does CCI improve semantic stability? | 2500 steps | Yes, more robust |
| `meaning_enables_cci.yaml` | Is shared meaning required for CCI? | 2000 steps | No, but helpful |
| `cci_meaning_feedback.yaml` | Do they reinforce each other? | 3000 steps | Yes, positive feedback |

**Guardian Thresholds:** coherence ≥0.85, noise ≤0.20  
**Paper References:** Multiple papers - integration sections

### 4.2 Meaning + Time (4 experiments)

| Experiment | Research Question | Duration | Key Finding |
|------------|------------------|----------|-------------|
| `meaning_time_coupling.yaml` | Does temporal memory affect semantics? | 2000 steps | Yes, improves stability |
| `temporal_semantic_drift.yaml` | How does meaning change over time? | 5000 steps | Gradual evolution |
| `time_enables_meaning.yaml` | Is temporal structure required? | 2000 steps | Not strictly, but helps |
| `meaning_temporal_reference.yaml` | Do agents use time for meaning? | 2000 steps | Yes, temporal context |

**Guardian Thresholds:** coherence ≥0.85, temporal_coherence ≥0.85  
**Paper References:** Cross-domain integration studies

### 4.3 CCI + Time (4 experiments)

| Experiment | Research Question | Duration | Key Finding |
|------------|------------------|----------|-------------|
| `cci_temporal_persistence.yaml` | Can CCI maintain across time gaps? | 3000 steps | Yes, if memory sufficient |
| `temporal_cci_transfer.yaml` | Does time affect context transfer? | 2000 steps | Temporal proximity helps |
| `cci_causal_inference.yaml` | Does CCI use causal structure? | 2000 steps | Yes, for context mapping |
| `temporal_context_switch.yaml` | Does timing of switch matter? | 2000 steps | Yes, sync improves transfer |

**Guardian Thresholds:** coherence ≥0.85, temporal_coherence ≥0.85  
**Paper References:** Advanced integration analysis

### 4.4 Complete System (1 experiment)

| Experiment | Research Question | Duration | Key Result |
|------------|------------------|----------|------------|
| `full_system_test.yaml` | Does the complete system work? | 5000 steps | Yes, all three interact |

**Guardian Thresholds:** All thresholds enforced  
**Paper References:** System-level validation

---

## Running Experiments

### Basic Execution

```bash
# Run a single experiment
python -m src.simulation_runner \
    --experiment experiments/cci_experiments/basic_cci.yaml \
    --output results/cci/basic/

# Run with Guardian validation
python -m src.simulation_runner \
    --experiment experiments/cci_experiments/basic_cci.yaml \
    --guardian-config configs/guardian_config.yaml \
    --output results/cci/basic/

# View results
python -m src.experiment_manager --show-results results/cci/basic/
```

### Batch Execution

```bash
# Run all CCI experiments
python -m tools.batch_runner \
    --category cci_experiments \
    --output results/cci_batch/

# Run all experiments in a category with parallel execution
python -m tools.batch_runner \
    --category meaning_experiments \
    --parallel 4 \
    --output results/meaning_batch/
```

### AutoLab Experiment Generation

```bash
# Generate experiments autonomously
python -m src.autolab \
    --research-question "How does network topology affect CCI?" \
    --max-experiments 10 \
    --output experiments/autolab_generated/

# Run AutoLab-generated experiments
python -m tools.batch_runner \
    --category autolab_generated \
    --output results/autolab/
```

### Monitoring Execution

```bash
# Launch Sentinel dashboard
python -m sentinel.dashboard

# View real-time Guardian validation
python -m sentinel.validation_viewer --live

# Check AutoLab learning progress
python -m sentinel.dashboard --view autolab
```

---

## Experiment Templates

### CCI Experiment Template

```yaml
name: "Your CCI Experiment"
category: "cci_experiments"
research_question: "What CCI property are you testing?"

agents:
  count: 10
  initial_context: "default"
  cci_enabled: true
  learning_rate: 0.01
  memory_depth: 50

simulation:
  duration: 1000
  interaction_frequency: 10
  context_switches: [250, 500, 750]
  record_metrics: true

validation:
  min_coherence: 0.85
  max_noise: 0.20
  expected_outcome: "describe what should happen"

output:
  save_frequency: 100
  visualizations: ["trajectories", "coherence", "cci_score"]
```

### Meaning Experiment Template

```yaml
name: "Your Meaning Experiment"
category: "meaning_experiments"
research_question: "What semantic property are you testing?"

agents:
  count: 10
  interaction_protocol: "bidirectional"  # or "unidirectional"
  semantic_memory: 100

simulation:
  duration: 1500
  interaction_frequency: 10
  perturbations: []  # optional disruptions

validation:
  min_coherence: 0.85
  max_noise: 0.20
  convergence_threshold: 0.90

output:
  save_frequency: 100
  visualizations: ["meaning_dynamics", "convergence", "stability"]
```

### Time Experiment Template

```yaml
name: "Your Time Experiment"
category: "time_experiments"
research_question: "What temporal property are you testing?"

agents:
  count: 10
  memory_depth: 50
  causal_inference: true
  temporal_tracking: true

simulation:
  duration: 2000
  event_frequency: 10
  causal_complexity: "normal"  # low, normal, high

validation:
  min_temporal_coherence: 0.85
  min_causal_accuracy: 0.80

output:
  save_frequency: 100
  visualizations: ["temporal_flow", "causal_graph", "memory_usage"]
```

### Integration Experiment Template

```yaml
name: "Your Integration Experiment"
category: "integration_tests"
research_question: "How do multiple phenomena interact?"

agents:
  count: 10
  cci_enabled: true
  semantic_tracking: true
  temporal_tracking: true
  memory_depth: 50

simulation:
  duration: 2000
  test_all_interactions: true

validation:
  min_coherence: 0.85
  max_noise: 0.20
  min_temporal_coherence: 0.85

output:
  save_frequency: 100
  visualizations: ["full_system", "interaction_matrix"]
```

---

## Experiment Naming Conventions

### File Naming

Format: `[category]_[descriptor].yaml`

Examples:
- `cci_basic.yaml`
- `meaning_emergence_timeline.yaml`
- `time_causal_structure.yaml`
- `integration_cci_meaning.yaml`

### Output Naming

Format: `results/[category]/[experiment_name]/[timestamp]/`

Example directory structure:
```
results/
├── cci_experiments/
│   ├── basic_cci/
│   │   ├── 2024-01-15_14-30-00/
│   │   └── 2024-01-16_09-15-00/
│   └── context_transfer/
└── meaning_experiments/
    └── semantic_emergence/
```

---

## Experiment Metadata

Each experiment produces metadata for research tracking:

```json
{
  "experiment_name": "basic_cci",
  "category": "cci_experiments",
  "timestamp": "2024-01-15T14:30:00Z",
  "guardian_validation": {
    "pre_execution": "PASS",
    "post_execution": "PASS",
    "coherence": 0.89,
    "noise": 0.15
  },
  "key_findings": {
    "cci_score": 0.87,
    "emergence_time": 485,
    "stability": "high"
  },
  "paper_references": [
    "CCI Paper 2024 - Figure 3"
  ]
}
```

---

## Contributing New Experiments

1. **Choose template** - Select appropriate template above
2. **Define research question** - Clear, testable hypothesis
3. **Configure parameters** - Agent count, duration, etc.
4. **Set validation criteria** - Guardian thresholds
5. **Run with Guardian** - Ensure validation passes
6. **Document results** - Update this index
7. **Link to papers** - Add to RESEARCH_MAPPING.md if applicable

---

## Questions?

**"Which experiment should I run first?"**
→ Start with basic experiments: `basic_cci.yaml`, `semantic_emergence.yaml`, `temporal_flow.yaml`

**"How do I reproduce paper results?"**
→ See RESEARCH_MAPPING.md for experiment↔paper connections

**"Can I modify existing experiments?"**
→ Yes, but maintain Guardian thresholds unless charter-approved

**"How do I create AutoLab experiments?"**
→ Use `python -m src.autolab` with research question

**"Where are experiment results saved?"**
→ `data/results/[category]/[experiment_name]/`

---

**Last Updated:** 2024-01-26  
**Total Experiments:** 96 (32 CCI, 28 Meaning, 24 Time, 12 Integration)  
**Guardian Validated:** 100%  
**Related Docs:** [README.md](README.md), [RESEARCH_MAPPING.md](RESEARCH_MAPPING.md), [API_GUIDE.md](API_GUIDE.md)
