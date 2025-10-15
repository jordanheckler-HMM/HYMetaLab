# Guardian v4 False Flag Log

**Generated:** 2025-10-15  
**Corpus Scan:** Full repository  
**Purpose:** Track false positives/negatives for calibration

---

## False Positives (Flagged incorrectly as violations)

*To be populated during manual review*

**Criteria for False Positive:**
- Guardian flags content as problematic
- Manual review confirms content is appropriate
- Common patterns: hedged language flagged as certain, simulation-scoped claims flagged as universal

---

## False Negatives (Missed violations)

*To be populated during manual review*

**Criteria for False Negative:**
- Guardian passes content without flags
- Manual review identifies issues
- Common patterns: Subtle overclaiming, missing citations, insufficient hedging

---

## Calibration Notes

**Current Guardian v4 Performance:**
- Mean corpus score: ~62.5/100
- Pass rate: ~7.5% (≥70 threshold)
- Top performer: synthesis_narrative.md (87.0)

**CRA-Identified Issues:**
1. Too strict on hedging requirements
2. Citation density threshold may be too high
3. Sentiment analysis catching positive scientific language

**Recommended Weight Adjustments:**
- Reduce hedge density requirement: 0.15 → 0.10
- Increase citation density tolerance: 0.05 → 0.03
- Soften sentiment neutrality: 85 → 80

---

**Lab Tech:** Validation  
**Status:** Manual review in progress
