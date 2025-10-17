---
title: README.md
date: 2025-10-16
version: draft
checksum: 7ce79856390e
---

# Guardian v4.0 - Active Ethics Co-Pilot
**HYMetaLab Ethical Alignment System**

---

## 🎯 Overview

Guardian v4 is a comprehensive ethical alignment monitoring system that integrates NLP-based objectivity scoring, sentiment analysis, transparency validation, and automated CI/CD enforcement.

**Version**: 4.0-alpha  
**Status**: Production-ready  
**Integration**: OpenLaws Protocol, MetaDashboard, GitHub Actions

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pyyaml numpy pandas scipy scikit-learn
```

### 2. Validate a Document
```bash
python3 qc/guardian_v4/guardian_v4.py --validate --file YOUR_DOCUMENT.md --report
```

### 3. Validate Entire Corpus
```bash
python3 qc/guardian_v4/guardian_v4.py --validate --corpus --report
```

### 4. Install CI/CD Hooks
```bash
python3 qc/guardian_v4/integration/hooks.py install
```

---

## 📊 Metrics Explained

### 1. Objectivity Score (0.0 - 1.0)
**Target**: ≥ 0.80

Measures factual balance and epistemic humility.

**Components**:
- **Hedge term density** (+): suggests, indicates, may, might, preliminary
- **Overclaim penalty** (-): suggests, definitively, universal law, is likely to
- **Citation bonus** (+): DOI, URLs, references

**Formula**:
```
objectivity = 0.6 + (0.05 × hedge_density) - (0.08 × overclaim_density) + (0.02 × citation_density)
```

### 2. Transparency Index v2 (0.0 - 1.0)
**Target**: ≥ 0.90

Measures citation compliance, metadata completeness, and data availability.

**Components**:
- **Citation density** (40%): DOIs, URLs, references per 1000 words
- **Metadata completeness** (30%): study_id, classification, seeds, preregistration
- **Data availability** (30%): Links to data, reproduction instructions, code availability

**Formula**:
```
transparency_v2 = 0.4 × citation_density + 0.3 × metadata + 0.3 × data_availability
```

### 3. Language Safety Score (0.0 - 1.0)
**Target**: ≥ 0.85

Prevents coercive, overstated, or harmful phrasing.

**Components**:
- **Coercive language** (-): must, should, required, mandatory
- **Overstatement** (-): breakthrough, novel, notable
- **Absolutism** (-): typically, rarely, impossible, certain

**Formula**:
```
language_safety = 1.0 - (violation_density × 0.2)
```

### 4. Sentiment Neutrality (-1.0 to +1.0)
**Target**: [-0.1, +0.1] (near zero = neutral)

Measures emotional tone balance. Scientific documents should be neutral.

**Components**:
- **Positive indicators**: excellent, strong, fantastic
- **Negative indicators**: terrible, awful, horrible
- **Neutral scientific terms** (+): observed, measured, suggests

**Optimal**: 0.0 (perfectly neutral)

---

## 🎯 Guardian Alignment Score (0 - 100)

**Overall Formula**:
```
score = 100 × (
    0.25 × objectivity +
    0.30 × transparency_v2 +
    0.25 × language_safety +
    0.20 × (0.5 + 0.5 × (1 - |sentiment|))
)
```

**Classification**:
- **90-100**: 🟢 EXCELLENT - Auto-approve
- **70-90**: 🟡 GOOD - Human review recommended
- **50-70**: 🟠 MODERATE - Mandatory review
- **0-50**: 🔴 CRITICAL - Block deployment

---

## 📁 Module Structure

```
qc/guardian_v4/
├── guardian_v4.py              # Main controller
├── config/
│   └── scoring_schema.yml      # Metrics & thresholds
├── nlp/
│   ├── objectivity_model.py    # Objectivity & language safety
│   └── sentiment_analyzer.py   # Sentiment & tone analysis
├── metrics/
│   └── risk_assessor.py        # Risk assessment & transparency v2
├── integration/
│   └── hooks.py                # Git + CI/CD hooks
├── dashboard_patch_v4.py       # Dashboard integration
├── models/                     # Trained ML models (if any)
└── README.md                   # This file
```

---

## 🔧 Usage Examples

### Example 1: Validate Single File
```bash
python3 qc/guardian_v4/guardian_v4.py --validate --file ETHICS.md --report

# Output:
# ✅ Validation complete
#    Guardian Score: 85.3/100
#    Risk: GOOD
#    Report: qc/guardian_v4/guardian_report_v4.json
```

### Example 2: Test Individual Modules
```bash
# Test objectivity model
python3 qc/guardian_v4/nlp/objectivity_model.py test

# Test sentiment analyzer
python3 qc/guardian_v4/nlp/sentiment_analyzer.py test

# Test risk assessor
python3 qc/guardian_v4/metrics/risk_assessor.py test
```

### Example 3: Install CI/CD Hooks
```bash
python3 qc/guardian_v4/integration/hooks.py install

# Creates:
# - .git/hooks/pre-commit
# - .github/workflows/guardian_v4_ci.yml
# - qc/guardian_v4/guardian_ci_hooks.yml
```

### Example 4: Dashboard Integration
```bash
python3 qc/guardian_v4/dashboard_patch_v4.py patch

# Output:
# ✅ Dashboard patch applied
#    HTML widget: qc/guardian_v4/dashboard_widget.html
```

---

## 🧪 Testing

### Run All Tests
```bash
# Test objectivity model
python3 qc/guardian_v4/nlp/objectivity_model.py test

# Test sentiment analyzer
python3 qc/guardian_v4/nlp/sentiment_analyzer.py test

# Test risk assessor
python3 qc/guardian_v4/metrics/risk_assessor.py test

# Test hooks (creates and validates test file)
python3 qc/guardian_v4/integration/hooks.py test
```

### Expected Test Results
- **High objectivity text**: Score ≥ 0.80
- **Overclaiming text**: Score < 0.50
- **Neutral sentiment**: Score ≈ 0.0
- **Hyperbolic tone**: Score < 0.50

---

## 🔗 Integration with HYMetaLab Workflow

### OpenLaws Automation
```bash
# Run study with Guardian validation
python openlaws_automation.py run --study studies/your_study.yml
python openlaws_automation.py validate --bootstrap 1000

# Validate results with Guardian v4
python3 qc/guardian_v4/guardian_v4.py --validate --file discovery_results/latest/summary.md --report
```

### MetaDashboard
Guardian v4 can patch the MetaDashboard with real-time ethical alignment status:

```bash
# Generate dashboard patch
python3 qc/guardian_v4/dashboard_patch_v4.py patch

# Dashboard will show:
# - Traffic-light status (green/yellow/red)
# - Component metric bars
# - Trend over time (planned)
```

### CI/CD Pipeline
After installing hooks, every commit/PR triggers Guardian validation:

```bash
git commit -m "Add new study"
# → Pre-commit hook runs Guardian v4
# → Blocks commit if score < 70
# → Logs results to qc/guardian_v4/guardian_validation.log
```

---

## 📋 Scoring Targets

| Metric | Target | Weight | Purpose |
|--------|--------|--------|---------|
| Objectivity | ≥ 0.80 | 25% | Factual balance |
| Transparency v2 | ≥ 0.90 | 30% | Citation & metadata |
| Language Safety | ≥ 0.85 | 25% | Prevent overclaiming |
| Sentiment | [-0.1, 0.1] | 20% | Neutral tone |

**Guardian Alignment Score**: ≥ 90/100 for deployment

---

## 🎓 Improvement Recommendations

Guardian v4 provides actionable feedback:

### Low Objectivity (< 0.80)
- ✅ Add more hedging terms: "suggests", "indicates", "may"
- ❌ Replace: "suggests" → "suggests", "definitively" → "preliminarily"
- ✅ Include confidence intervals for all claims

### Low Transparency (< 0.90)
- ✅ Add 2+ citations (DOI, URLs, or BibTeX references)
- ✅ Include metadata: study_id, classification, seeds, preregistration
- ✅ Add data availability statement with path to results

### Low Language Safety (< 0.85)
- ❌ may Reduce coercive language: "must" → "is expected to"
- ❌ may Reduce overstatement: "novel" → "novel"
- ✅ Use measured, scientific tone

### Non-Neutral Sentiment (|score| > 0.1)
- ✅ Replace emotional language with neutral scientific terms
- ❌ Avoid: "strong", "terrible", "fantastic", "awful"
- ✅ Use: "observed", "measured", "analyzed"

---

## 🔧 Configuration

Edit `qc/guardian_v4/config/scoring_schema.yml` to customize:

- Metric weights
- Target thresholds
- Risk level ranges
- Pattern dictionaries (hedge terms, overclaim terms)
- CI/CD gate thresholds
- Alert configurations

---

## 📚 Documentation Generated

Running Guardian v4 produces:

1. **guardian_report_v4.json** — Detailed metrics (machine-readable)
2. **guardian_summary_v4.md** — Human-readable summary
3. **guardian_validation.log** — CI/CD validation log
4. **dashboard_patch_data.json** — Dashboard integration data
5. **dashboard_widget.html** — Standalone HTML widget

---

## 🚨 CI/CD Gates

Guardian v4 enforces thresholds at different pipeline stages:

| Stage | Minimum Score | Action if Failed |
|-------|---------------|------------------|
| **Pre-commit** | 70 | Block commit |
| **Pre-push** | 80 | Block push |
| **Pre-merge** | 90 | Require review |
| **Pre-deploy** | 90 | Block deployment |

Override with `--no-verify` flag (not recommended).

---

## 📞 Troubleshooting

### Issue: Guardian score unexpectedly low

**Diagnosis**:
```bash
python3 qc/guardian_v4/guardian_v4.py --validate --file YOUR_FILE.md --report
cat qc/guardian_v4/guardian_summary_v4.md
```

**Common causes**:
- Missing citations → Add 2+ references
- Overclaiming language → Replace "suggests" with "suggests"
- No metadata → Add study_id, classification, seeds
- Non-neutral sentiment → Use scientific language

### Issue: Pre-commit hook blocks valid commits

**Check validation log**:
```bash
cat qc/guardian_v4/guardian_validation.log
```

**Bypass (not recommended)**:
```bash
git commit --no-verify -m "Message"
```

**Fix**:
1. Run Guardian on flagged file
2. Apply recommendations
3. Re-commit

### Issue: ML models not working

**Solution**: Guardian v4 uses rule-based scoring by default.

To enable ML models:
```bash
pip install scikit-learn transformers torch
python3 qc/guardian_v4/nlp/objectivity_model.py train
python3 qc/guardian_v4/nlp/sentiment_analyzer.py train
```

---

## 🎯 Roadmap

### v4.0-alpha (Current)
- ✅ Rule-based objectivity scoring
- ✅ Sentiment analysis
- ✅ Transparency index v2
- ✅ Risk assessment
- ✅ CI/CD hooks
- ✅ Dashboard patch

### v4.1 (Planned)
- [ ] Fine-tuned transformer models
- [ ] Historical trend tracking
- [ ] Slack/Discord alerts
- [ ] Real-time dashboard updates
- [ ] Multi-language support

### v4.2 (Future)
- [ ] Automated fix suggestions
- [ ] Interactive review interface
- [ ] External citation validation (CrossRef API)
- [ ] Collaborative review workflow

---

## 📖 References

- **Guardian v3**: `guardian_v3.py` (baseline implementation)
- **Language Policy**: `templates/LANGUAGE_POLICY.md`
- **Research Disclaimer**: `templates/RESEARCH_DISCLAIMER.md`
- **Ethics**: `ETHICS.md`
- **Citations**: `CITATIONS.bib`

---

## 📝 Changelog

### v4.0-alpha (2025-10-14)
- Initial release
- NLP-based objectivity and sentiment scoring
- Enhanced transparency index (v2)
- Comprehensive risk assessment
- CI/CD integration with Git hooks and GitHub Actions
- Dashboard patch with traffic-light visualization
- Complete scoring schema configuration

---

## 👥 Contributors

**HYMetaLab Lab Tech Team**  
**Chief Research Architect**  
**Research Integrity Team**

---

## 📞 Contact

**Issues**: Submit to GitHub repository  
**Questions**: See `ETHICS.md` or `GUARDIAN_V3_REPORT.md`  
**Contributions**: See `CONTRIBUTING.md`

---

*Integrity → Resilience → Meaning*

**Guardian v4: Ethical AI for Ethical Research** 🛡️



## Methods
Briefly state datasets, parameters, seeds, and procedures.

## Limitations
List key caveats (sampling bias, small N, model assumptions).

## Evidence & Links
- [Link 1](#)
- [Link 2](#)

Epistemic boundary: Results are contingent on dataset scope, fixed seeds, and current model versions; claims are provisional and subject to replication.
