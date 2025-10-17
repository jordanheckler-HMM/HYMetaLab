# AutoLab Lite — Autonomous Hypothesis Testing

## Overview

**AutoLab Lite** is an autonomous experiment runner that:
- ✅ Generates hypotheses automatically
- ✅ Pre-validates with Guardian
- ✅ Executes experiments via lab_techs_runner.py
- ✅ Learns from success/failure via simple UCB algorithm
- ✅ Maintains knowledge base in autolab/knowledge.json

**No external services required** — fully local, safe to stop anytime (Ctrl+C).

---

## Quick Start

```bash
# Run 1 cycle (default)
python3 auto_lab.py

# Run 5 cycles
python3 auto_lab.py 5

# Run 10 cycles and let it explore
python3 auto_lab.py 10
```

---

## Knowledge Base After 4 Runs

```json
{
  "runs": [
    {"ts": "2025-10-17T03:58:03Z", "hypothesis": "baseline: demo-run", "success": true},
    {"ts": "2025-10-17T03:58:05Z", "hypothesis": "baseline: demo-run", "success": true},
    {"ts": "2025-10-17T03:58:05Z", "hypothesis": "η sweep 0.10–0.20, n=40", "success": false},
    {"ts": "2025-10-17T03:58:05Z", "hypothesis": "baseline: demo-run", "success": true}
  ],
  "hypotheses": {
    "baseline: demo-run": {"score": 0.829, "n": 3},
    "η sweep 0.10–0.20, n=40": {"score": 0.35, "n": 1}
  }
}
```

**Interpretation**:
- baseline has 82.9% success score (3/3 runs passed)
- η sweep has 35% score (1 run, failed Guardian check)
- AutoLab learns to favor baseline but will still explore others

---

## How It Works

### 1. UCB Selection Algorithm
```
utility = score + 0.2 * sqrt(log(total_runs) / (hypothesis_runs))
```
Balances exploitation (high score) vs exploration (under-tested).

### 2. Guardian Pre-Check
Every hypothesis validated before execution:
- Coherence ≥ 0.85
- Noise ≤ 0.20
- Verdict != "block"

### 3. Learning Update
```python
new_score = 0.7 * old_score + 0.3 * (1.0 if success else 0.0)
```
Gradual learning from outcomes.

---

## Features

✅ **Fully Autonomous** — No human intervention  
✅ **Guardian-Validated** — Every run pre-checked  
✅ **Exploration/Exploitation** — UCB algorithm  
✅ **Persistent Learning** — Knowledge survives restarts  
✅ **Safe Shutdown** — Ctrl+C anytime  

---

## Charter Compliance

✅ **Charter v2.0**: All experiments Guardian-validated  
✅ **SOP v1.1**: Standard operational procedures  
✅ **Integrity**: No experiments without validation  
✅ **Reproducibility**: Full audit trail  

---

*"Integrity → Resilience → Meaning"*  
*HYMetaLab AutoLab Lite v1.0*

