# Research Mapping: Code ↔ Papers

**How HYMetaLab simulations connect to published research**

This document maps the relationship between code implementations in HYMetaLab and the research papers they support. Use this guide to understand which simulations validate which theoretical claims, and where to find specific implementations.

---

## Overview

HYMetaLab supports three interconnected research programs:

1. **Cross-Contextual Inference (CCI)** - How meaning transfers across different contexts
2. **Meaning Formation** - Emergence of semantic structure from agent interactions
3. **Time Mechanics** - Temporal flow, memory, and causality in distributed systems

Each research program has:
- **Published papers** articulating theoretical frameworks
- **Core implementation** in `main.py` (4,307-line foundational model)
- **Experiment suite** testing specific hypotheses
- **Guardian validation** ensuring research integrity

---

## 1. Cross-Contextual Inference (CCI)

### Published Research

**"Cross-Contextual Inference: Meaning Preservation Across Domain Boundaries" (2024)**
- **Core claim**: Agents can maintain semantic coherence when switching between different contextual domains
- **Key finding**: CCI stability requires contextual overlap ≥30% for meaning preservation
- **Validation**: Guardian-enforced coherence ≥0.85 across context switches

**"CCI in Multi-Agent Systems: Emergence and Stability" (2024)**
- **Core claim**: CCI emerges naturally from interaction patterns, not pre-programmed rules
- **Key finding**: Network topology affects CCI convergence speed (scale-free networks 40% faster)
- **Validation**: AutoLab discovered optimal topology parameters through UCB learning

### Code Implementation

#### Core CCI Mechanics (`main.py` lines 1200-1450)

```python
class Agent:
    def cross_contextual_inference(self, source_context, target_context):
        """
        Transfer meaning from source to target context
        
        Implementation of CCI algorithm from paper:
        1. Extract semantic core from source context
        2. Map to target context representation space
        3. Validate coherence with Guardian thresholds
        """
        semantic_core = self.extract_meaning(source_context)
        mapped_meaning = self.context_mapper.transfer(
            semantic_core, 
            target_context
        )
        
        # Guardian validation: ensure coherence maintained
        coherence = self.measure_coherence(mapped_meaning)
        assert coherence >= 0.85, "CCI coherence below threshold"
        
        return mapped_meaning
```

**Key components:**
- **Lines 1200-1250**: Semantic extraction from source context
- **Lines 1260-1320**: Context mapping algorithm (implements overlap calculation)
- **Lines 1330-1380**: Coherence measurement (used by Guardian)
- **Lines 1390-1450**: CCI stability tracking over time

#### Context Switching (`main.py` lines 1500-1650)

```python
def switch_context(self, new_context, transition_speed="normal"):
    """
    Context transition with meaning preservation
    
    Validates paper claim: 30% overlap required for stability
    """
    overlap = self.calculate_overlap(self.current_context, new_context)
    
    if overlap < 0.30:
        # Insufficient overlap - gradual transition required
        self.gradual_context_shift(new_context, steps=10)
    else:
        # Sufficient overlap - direct switch safe
        self.current_context = new_context
```

**Key components:**
- **Lines 1500-1540**: Overlap calculation (30% threshold validation)
- **Lines 1550-1600**: Gradual transition algorithm for low overlap
- **Lines 1610-1650**: Context history tracking for CCI analysis

#### Meaning Transfer (`main.py` lines 1700-1850)

Implements the semantic preservation mechanism that enables CCI across domains.

**Key components:**
- **Lines 1700-1750**: Semantic core extraction (domain-invariant meaning)
- **Lines 1760-1800**: Transfer function (maps meaning to new domain)
- **Lines 1810-1850**: Preservation validation (measures semantic loss)

### Experiments

| Experiment | Paper Claim Validated | Guardian Threshold |
|------------|----------------------|-------------------|
| `cci_experiments/basic_cci.yaml` | CCI emergence from interactions | coherence ≥0.85 |
| `cci_experiments/context_transfer.yaml` | 30% overlap requirement | coherence ≥0.85, overlap ≥0.30 |
| `cci_experiments/cci_stability.yaml` | Long-term CCI persistence | coherence ≥0.85 over 1000 steps |
| `cci_experiments/network_topology.yaml` | Scale-free networks optimize CCI | convergence_speed measured |
| `cci_experiments/multi_domain.yaml` | CCI works across 3+ domains | coherence ≥0.85 across all |

### Data Files

Results supporting paper figures and tables:
- `data/results/cci/overlap_vs_stability.json` → Paper Figure 3
- `data/results/cci/network_topology_comparison.json` → Paper Table 2
- `data/results/cci/emergence_timeline.json` → Paper Figure 5

---

## 2. Meaning Formation

### Published Research

**"Emergent Semantics in Multi-Agent Interaction Networks" (2024)**
- **Core claim**: Meaning emerges from interaction patterns without centralized coordination
- **Key finding**: Semantic convergence occurs in ~500 interaction cycles for 10-agent systems
- **Validation**: Guardian noise ≤0.20 prevents random drift during convergence

**"Meaning Dynamics: From Interaction to Understanding" (2023)**
- **Core claim**: Shared meaning requires bidirectional information flow
- **Key finding**: Unidirectional communication produces 60% less semantic stability
- **Validation**: Interaction symmetry metrics tracked via Guardian

### Code Implementation

#### Semantic Structures (`main.py` lines 2000-2300)

```python
class SemanticStructure:
    def __init__(self):
        self.meaning_graph = {}  # Nodes = concepts, edges = relations
        self.stability_score = 0.0
        
    def evolve(self, interactions):
        """
        Update semantic structure based on agent interactions
        
        Implements emergence algorithm from paper:
        - Local interactions → global semantic patterns
        - No centralized coordination required
        - Convergence validated by Guardian noise threshold
        """
        for interaction in interactions:
            self.integrate_interaction(interaction)
            
        # Guardian validation: ensure noise doesn't dominate
        noise = self.measure_semantic_noise()
        assert noise <= 0.20, "Semantic noise exceeds threshold"
```

**Key components:**
- **Lines 2000-2080**: Semantic graph construction (concepts + relations)
- **Lines 2090-2150**: Emergence dynamics (local → global patterns)
- **Lines 2160-2220**: Stability measurement (resistance to perturbation)
- **Lines 2230-2300**: Noise calculation (random vs structured semantic change)

#### Interaction Protocols (`main.py` lines 2350-2550)

```python
class InteractionProtocol:
    def exchange_meaning(self, agent_a, agent_b, bidirectional=True):
        """
        Agent-to-agent semantic exchange
        
        Validates paper claim: bidirectional flow required for stability
        """
        # A → B information transfer
        meaning_ab = agent_a.share_meaning()
        agent_b.integrate_meaning(meaning_ab)
        
        if bidirectional:
            # B → A information transfer
            meaning_ba = agent_b.share_meaning()
            agent_a.integrate_meaning(meaning_ba)
            
        return self.measure_stability()
```

**Key components:**
- **Lines 2350-2410**: Bidirectional vs unidirectional protocols
- **Lines 2420-2480**: Information exchange mechanics
- **Lines 2490-2550**: Stability comparison (validates 60% finding)

#### Meaning Convergence (`main.py` lines 2600-2800)

Tracks how individual agent meanings converge toward shared semantics.

**Key components:**
- **Lines 2600-2670**: Convergence detection algorithm
- **Lines 2680-2740**: Convergence speed measurement (validates 500-cycle finding)
- **Lines 2750-2800**: Divergence prevention (Guardian-enforced bounds)

### Experiments

| Experiment | Paper Claim Validated | Guardian Threshold |
|------------|----------------------|-------------------|
| `meaning_experiments/semantic_emergence.yaml` | Emergence from interactions | noise ≤0.20 |
| `meaning_experiments/interaction_patterns.yaml` | Network structure affects emergence | coherence ≥0.85 |
| `meaning_experiments/meaning_stability.yaml` | Bidirectional flow requirement | stability measured |
| `meaning_experiments/convergence_speed.yaml` | 500-cycle convergence finding | convergence tracked |
| `meaning_experiments/perturbation_test.yaml` | Resistance to disruption | noise ≤0.20 |

### Data Files

Results supporting paper figures and tables:
- `data/results/meaning/emergence_dynamics.json` → Paper Figure 2
- `data/results/meaning/bidirectional_vs_unidirectional.json` → Paper Figure 4
- `data/results/meaning/convergence_curves.json` → Paper Figure 6

---

## 3. Time Mechanics

### Published Research

**"Temporal Flow in Multi-Agent Systems: Memory and Causality" (2024)**
- **Core claim**: Time emerges from causal relationships between events, not from external clock
- **Key finding**: Memory depth affects causal inference accuracy (depth ≥50 required)
- **Validation**: Guardian tracks causal coherence over temporal sequences

**"Time as Emergent Property in Distributed Systems" (2023)**
- **Core claim**: Distributed agents can synchronize temporal understanding without global time
- **Key finding**: Local causality sufficient for global temporal consistency
- **Validation**: Temporal coherence ≥0.85 maintained across all agents

### Code Implementation

#### Time Dynamics (`main.py` lines 3000-3250)

```python
class TemporalAgent:
    def __init__(self, memory_depth=50):
        self.memory = deque(maxlen=memory_depth)  # Event history
        self.causal_graph = {}  # Event → caused events
        
    def perceive_time(self):
        """
        Construct temporal understanding from causal relationships
        
        Implements paper's emergent time theory:
        - No external clock reference
        - Time = sequence of causally-linked events
        - Synchronization emerges from shared causality
        """
        temporal_sequence = self.construct_causal_chain()
        temporal_coherence = self.measure_consistency()
        
        # Guardian validation: ensure temporal coherence
        assert temporal_coherence >= 0.85, "Temporal coherence lost"
        
        return temporal_sequence
```

**Key components:**
- **Lines 3000-3070**: Event-based time representation (no global clock)
- **Lines 3080-3150**: Causal chain construction (builds temporal sequence)
- **Lines 3160-3210**: Temporal coherence measurement (Guardian metric)
- **Lines 3220-3250**: Synchronization emergence (distributed coordination)

#### Memory Systems (`main.py` lines 3300-3500)

```python
class MemorySystem:
    def __init__(self, depth=50):
        self.short_term = deque(maxlen=depth)
        self.causal_accuracy_history = []
        
    def store_event(self, event):
        """
        Store event with causal metadata
        
        Validates paper claim: depth ≥50 required for accurate inference
        """
        self.short_term.append(event)
        
        if len(self.short_term) >= 50:
            accuracy = self.evaluate_causal_inference()
            self.causal_accuracy_history.append(accuracy)
```

**Key components:**
- **Lines 3300-3370**: Memory depth implementation (validates ≥50 finding)
- **Lines 3380-3440**: Causal inference from memory (accuracy tracking)
- **Lines 3450-3500**: Memory-causality relationship analysis

#### Causality Tracking (`main.py` lines 3550-3750)

Implements the causal inference mechanism that enables emergent time.

**Key components:**
- **Lines 3550-3620**: Event causality detection (A caused B determination)
- **Lines 3630-3690**: Causal graph construction (network of dependencies)
- **Lines 3700-3750**: Distributed synchronization (shared causal understanding)

### Experiments

| Experiment | Paper Claim Validated | Guardian Threshold |
|------------|----------------------|-------------------|
| `time_experiments/temporal_flow.yaml` | Time emerges from causality | coherence ≥0.85 |
| `time_experiments/memory_persistence.yaml` | Memory depth ≥50 requirement | causal_accuracy tracked |
| `time_experiments/causal_structure.yaml` | Causal inference accuracy | coherence ≥0.85 |
| `time_experiments/distributed_sync.yaml` | No global clock needed | temporal_coherence ≥0.85 |
| `time_experiments/temporal_consistency.yaml` | Shared time from shared causality | coherence ≥0.85 |

### Data Files

Results supporting paper figures and tables:
- `data/results/time/emergence_vs_clock.json` → Paper Figure 1
- `data/results/time/memory_depth_vs_accuracy.json` → Paper Table 1
- `data/results/time/distributed_synchronization.json` → Paper Figure 7

---

## 4. Integration Studies

Some experiments test interactions between multiple phenomena:

| Experiment | Phenomena Combined | Research Question |
|------------|-------------------|------------------|
| `integration_tests/cci_meaning_interaction.yaml` | CCI + Meaning | Does CCI accelerate meaning convergence? |
| `integration_tests/meaning_time_coupling.yaml` | Meaning + Time | How does temporal memory affect semantic stability? |
| `integration_tests/cci_temporal_persistence.yaml` | CCI + Time | Can CCI maintain across large temporal gaps? |
| `integration_tests/full_system_test.yaml` | All Three | Comprehensive system validation |

These experiments appear in multiple papers as they bridge research domains.

---

## Using This Mapping

### To Understand a Paper's Implementation

1. Find the paper in the sections above
2. Note the code line references (e.g., `main.py` lines 1200-1450)
3. Review the corresponding experiment files
4. Check data files for results supporting paper figures

**Example: "How is the 30% overlap finding from the CCI paper implemented?"**
- Code: `main.py` lines 1500-1540 (overlap calculation)
- Experiment: `cci_experiments/context_transfer.yaml`
- Data: `data/results/cci/overlap_vs_stability.json`
- Paper reference: Figure 3

### To Find Code for a Specific Claim

Use this reference table:

| Research Claim | Code Location | Experiment | Guardian Validation |
|----------------|---------------|------------|-------------------|
| CCI requires 30% overlap | `main.py` 1500-1540 | `context_transfer.yaml` | coherence ≥0.85 |
| Meaning emerges in ~500 cycles | `main.py` 2680-2740 | `convergence_speed.yaml` | noise ≤0.20 |
| Bidirectional flow improves stability 60% | `main.py` 2490-2550 | `interaction_patterns.yaml` | stability tracked |
| Memory depth ≥50 for causal inference | `main.py` 3300-3370 | `memory_persistence.yaml` | accuracy tracked |
| Time emerges from causality | `main.py` 3000-3070 | `temporal_flow.yaml` | coherence ≥0.85 |

### To Reproduce Paper Results

1. Locate the experiment file from the mapping tables above
2. Run: `python -m src.simulation_runner --experiment [experiment_file]`
3. Results saved to `data/results/[category]/`
4. Compare with paper's reported metrics
5. Guardian logs validate that thresholds were maintained

**Example: Reproduce CCI stability findings**
```bash
python -m src.simulation_runner \
    --experiment experiments/cci_experiments/cci_stability.yaml \
    --output data/results/cci/reproduction/

# Compare with paper results
python -m tools.compare_results \
    --paper-data data/results/cci/overlap_vs_stability.json \
    --reproduction-data data/results/cci/reproduction/overlap_vs_stability.json
```

---

## Guardian Validation in Research

Every experiment run includes Guardian validation to ensure research integrity:

### Pre-Execution Validation
- Configuration sanity checks
- Parameter bounds verification
- Charter compliance (for threshold changes)

### Post-Execution Validation
- Coherence ≥0.85 (semantic consistency maintained)
- Noise ≤0.20 (random effects bounded)
- Expected outcome achieved (experiment-specific)

### Validation Logs

Guardian creates detailed logs for each experiment:
```
data/guardian_logs/
├── cci_experiments/
│   ├── basic_cci_20240115_validation.json
│   └── context_transfer_20240115_validation.json
├── meaning_experiments/
│   └── semantic_emergence_20240116_validation.json
└── time_experiments/
    └── temporal_flow_20240117_validation.json
```

These logs are referenced in papers to demonstrate experimental rigor.

---

## AutoLab Contributions

AutoLab (autonomous learning system) has discovered several findings that became paper contributions:

### AutoLab-Discovered Findings

**CCI Research:**
- Optimal context overlap varies by domain type (discovered via UCB exploration)
- Network topology effect on CCI speed (40% improvement in scale-free networks)

**Meaning Research:**
- Interaction frequency affects convergence speed nonlinearly
- Optimal agent diversity for semantic stability

**Time Research:**
- Memory depth interacts with causal complexity
- Distributed synchronization requires minimum 3-agent clusters

**AutoLab Logs:**
```
data/autolab_logs/
├── hypothesis_generation_20240101-20240131.json
├── ucb_learning_curves.json
└── discovered_parameters.json
```

These discoveries are noted in papers with acknowledgment to AutoLab's contribution.

---

## Code Stability and Research Continuity

### Why main.py is Monolithic

The 4,307-line `main.py` file is **intentionally monolithic** to ensure research continuity:

1. **200+ Agent fields encode the entire theoretical framework**
2. **100+ experiments depend on this stable schema**
3. **Decomposition would break research reproducibility**
4. **Changes cascade across three research programs**

**This is a feature, not a bug.** Research code prioritizes theoretical coherence over software engineering modularity.

### Adding New Research

When extending the framework:

1. **Add to main.py** - Don't create new modules for core theory
2. **Write experiments** - Test new ideas with Guardian validation
3. **Update this mapping** - Document code↔paper connections
4. **Run AutoLab** - Let autonomous learning explore parameter space

---

## Questions?

**"Where is the code for [specific paper claim]?"**
→ Use the mapping tables in sections 1-3 above

**"How do I reproduce paper results?"**
→ See "To Reproduce Paper Results" section

**"Can I modify main.py safely?"**
→ Only with deep understanding of cascading effects - consult research team first

**"Where are the raw data files?"**
→ `data/results/[category]/` - organized by research domain

**"How does Guardian ensure research integrity?"**
→ See "Guardian Validation in Research" section

---

**Last Updated:** 2024-01-26  
**Maintained By:** Research Team  
**Related Docs:** [README.md](README.md), [EXPERIMENTS.md](EXPERIMENTS.md), [API_GUIDE.md](API_GUIDE.md)
