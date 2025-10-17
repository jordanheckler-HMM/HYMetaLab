#!/usr/bin/env python3
"""
TruthLens v4 - Domain Adapter
Normalizes Truth Index across science/social/metaphysical domains

v4 Domain Adapter: Cross-domain truth alignment within ±5%
"""
from pathlib import Path

import numpy as np


class DomainAdapter:
    """
    Adapt Truth Index scoring to domain-specific norms
    Normalize across science, social science, and metaphysical domains
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Domain keywords for classification
        self.domain_keywords = {
            "science": [
                "experiment",
                "hypothesis",
                "methodology",
                "statistical",
                "measured",
                "laboratory",
                "control",
                "variable",
                "sample",
                "protocol",
                "empirical",
                "observation",
                "replication",
                "quantitative",
                "data",
                "physics",
                "chemistry",
                "biology",
                "neuroscience",
                "genetics",
            ],
            "social": [
                "society",
                "culture",
                "behavior",
                "population",
                "survey",
                "interview",
                "ethnography",
                "qualitative",
                "participant",
                "social",
                "psychology",
                "sociology",
                "anthropology",
                "economics",
                "political",
                "participants",
                "respondents",
                "cohort",
                "demographic",
                "community",
            ],
            "metaphysical": [
                "ontology",
                "existence",
                "reality",
                "consciousness",
                "being",
                "metaphysics",
                "philosophical",
                "conceptual",
                "abstract",
                "theoretical",
                "essence",
                "nature",
                "fundamental",
                "principle",
                "logic",
                "argument",
                "reasoning",
                "proposition",
                "axiom",
                "thought",
            ],
        }

        # Domain-specific scaling parameters
        # Science: high citation expectations, strong statistical focus
        # Social: moderate citations, mixed methods
        # Metaphysical: lower citation rate, emphasis on logical coherence
        self.domain_scales = self.config.get(
            "domain_scales",
            {
                "science": {
                    "citation_weight": 1.15,  # Higher weight for citations
                    "clarity_weight": 1.0,  # Standard clarity
                    "causal_weight": 0.95,  # Less weight on causal language
                    "baseline_shift": -0.02,  # Slightly higher bar (reduced from -0.05)
                },
                "social": {
                    "citation_weight": 1.0,  # Standard weight
                    "clarity_weight": 1.05,  # Slightly higher clarity weight
                    "causal_weight": 1.0,  # Standard causal weight
                    "baseline_shift": 0.0,  # No shift (reference domain)
                },
                "metaphysical": {
                    "citation_weight": 0.85,  # Lower citation expectations
                    "clarity_weight": 1.15,  # Higher clarity/argument quality
                    "causal_weight": 1.05,  # Higher weight on logical reasoning
                    "baseline_shift": 0.02,  # Slightly lower bar (reduced from 0.05)
                },
            },
        )

        # Variance guardrails
        self.target_variance = self.config.get("target_variance", 0.05)  # ±5%
        self.calibration_samples = {}  # Store calibration data

    def detect_domain(self, text: str) -> tuple[str, dict]:
        """
        Detect document domain from text

        Args:
            text: Document text

        Returns:
            Tuple of (domain_name, confidence_scores)
        """
        text_lower = text.lower()

        # Count keyword matches for each domain
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by keyword count
            score = matches / len(keywords)
            domain_scores[domain] = score

        # Detect domain with highest score
        if not domain_scores:
            return "social", domain_scores  # Default to social

        primary_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])

        # If scores are very close, default to social (middle ground)
        max_score = domain_scores[primary_domain]
        if max_score < 0.05:  # Very low match
            return "social", domain_scores

        return primary_domain, domain_scores

    def adapt_truth_index(
        self, raw_ti: float, domain: str, components: dict = None
    ) -> dict:
        """
        Adapt raw Truth Index to domain-specific norms

        Args:
            raw_ti: Raw truth index from core calculation
            domain: Detected domain (science/social/metaphysical)
            components: Component scores (clarity, citations, causality)

        Returns:
            Dictionary with adapted Ti and metadata
        """
        if domain not in self.domain_scales:
            # Unknown domain, return raw
            return {
                "adapted_ti": raw_ti,
                "raw_ti": raw_ti,
                "domain": domain,
                "adaptation_applied": False,
            }

        scales = self.domain_scales[domain]

        # If components provided, re-weight them
        if components:
            clarity = components.get("claim_clarity", 0)
            citations = components.get("citation_presence", 0)
            causality = components.get("causal_tokens", 0)

            # Apply domain-specific weights
            adapted_ti = (
                clarity * 0.35 * scales["clarity_weight"]
                + citations * 0.40 * scales["citation_weight"]
                + causality * 0.25 * scales["causal_weight"]
            )

            # Normalize back to [0,1] range
            total_weight = (
                0.35 * scales["clarity_weight"]
                + 0.40 * scales["citation_weight"]
                + 0.25 * scales["causal_weight"]
            )
            adapted_ti = adapted_ti / total_weight

            # Apply baseline shift
            adapted_ti = adapted_ti + scales["baseline_shift"]

            # Clip to [0, 1]
            adapted_ti = max(0.0, min(1.0, adapted_ti))
        else:
            # Simple scaling without components
            adapted_ti = raw_ti + scales["baseline_shift"]
            adapted_ti = max(0.0, min(1.0, adapted_ti))

        return {
            "adapted_ti": adapted_ti,
            "raw_ti": raw_ti,
            "domain": domain,
            "domain_scales": scales,
            "adaptation_applied": True,
        }

    def compute_cross_domain_variance(
        self, domain_scores: dict[str, list[float]]
    ) -> dict:
        """
        Compute variance across domains

        Args:
            domain_scores: Dict mapping domain name to list of Ti scores

        Returns:
            Variance metrics
        """
        # Compute mean for each domain
        domain_means = {
            domain: np.mean(scores) if scores else 0.0
            for domain, scores in domain_scores.items()
        }

        # Overall mean
        all_scores = [score for scores in domain_scores.values() for score in scores]
        overall_mean = np.mean(all_scores) if all_scores else 0.0

        # Variance across domain means
        if len(domain_means) > 1:
            mean_values = list(domain_means.values())
            cross_domain_variance = np.var(mean_values)
            cross_domain_std = np.std(mean_values)
        else:
            cross_domain_variance = 0.0
            cross_domain_std = 0.0

        # Max deviation from overall mean
        max_deviation = (
            max(abs(mean - overall_mean) for mean in domain_means.values())
            if domain_means
            else 0.0
        )

        # Relative variance (as percentage)
        relative_variance = (
            (max_deviation / overall_mean * 100) if overall_mean > 0 else 0.0
        )

        return {
            "domain_means": domain_means,
            "overall_mean": overall_mean,
            "cross_domain_variance": cross_domain_variance,
            "cross_domain_std": cross_domain_std,
            "max_deviation": max_deviation,
            "relative_variance_pct": relative_variance,
            "passes_threshold": relative_variance <= 5.0,  # ±5% target
        }

    def grade_adapted_ti(self, adapted_ti: float) -> str:
        """
        Grade adapted Ti with same thresholds as v1

        Args:
            adapted_ti: Adapted truth index

        Returns:
            Grade string
        """
        if adapted_ti >= 0.85:
            return "EXCELLENT"
        elif adapted_ti >= 0.75:
            return "GOOD"
        elif adapted_ti >= 0.60:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"

    def analyze_document(
        self, text: str, raw_ti: float, components: dict = None
    ) -> dict:
        """
        Complete domain-aware analysis

        Args:
            text: Document text
            raw_ti: Raw truth index
            components: Component scores

        Returns:
            Complete analysis with domain adaptation
        """
        # Detect domain
        domain, domain_scores = self.detect_domain(text)

        # Adapt Ti
        adaptation = self.adapt_truth_index(raw_ti, domain, components)

        # Grade
        grade = self.grade_adapted_ti(adaptation["adapted_ti"])

        return {
            "domain": domain,
            "domain_confidence": domain_scores,
            "raw_ti": raw_ti,
            "adapted_ti": adaptation["adapted_ti"],
            "grade": grade,
            "adaptation": adaptation,
            "passes_threshold": adaptation["adapted_ti"] >= 0.60,
        }


def main():
    """CLI for domain adapter"""
    import argparse

    parser = argparse.ArgumentParser(description="TruthLens v4 Domain Adapter")
    parser.add_argument(
        "command", choices=["test", "detect"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File to analyze")

    args = parser.parse_args()

    adapter = DomainAdapter()

    if args.command == "detect":
        if not args.file:
            print("❌ Error: --file required")
            return

        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        domain, scores = adapter.detect_domain(text)

        print(f"🎨 Domain Detection: {file_path.name}")
        print(f"\n   Primary domain: {domain}")
        print("\n   Confidence scores:")
        for d, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"     • {d}: {score:.3f}")

    elif args.command == "test":
        print("🧪 Running domain adapter tests...")

        # Test 1: Domain detection
        science_text = "The experiment measured statistical significance (p<0.001) using laboratory protocols."
        social_text = "Survey participants from diverse populations provided qualitative interview data."
        meta_text = "The ontological argument examines fundamental principles of existence and consciousness."

        print("\nTest 1 (Domain detection):")
        for label, text in [
            ("Science", science_text),
            ("Social", social_text),
            ("Metaphysical", meta_text),
        ]:
            domain, _ = adapter.detect_domain(text)
            print(
                f"   {label}: {domain} ✅"
                if domain.startswith(label.lower()[:3])
                else f"   {label}: {domain} ⚠️"
            )

        # Test 2: Ti adaptation
        print("\nTest 2 (Ti adaptation):")
        raw_ti = 0.70

        for domain in ["science", "social", "metaphysical"]:
            result = adapter.adapt_truth_index(raw_ti, domain)
            print(f"   {domain}: {raw_ti:.3f} → {result['adapted_ti']:.3f}")

        # Test 3: Cross-domain variance
        print("\nTest 3 (Cross-domain variance):")
        test_scores = {
            "science": [0.72, 0.74, 0.73],
            "social": [0.71, 0.73, 0.72],
            "metaphysical": [0.70, 0.72, 0.71],
        }

        variance = adapter.compute_cross_domain_variance(test_scores)
        print(f"   Overall mean: {variance['overall_mean']:.3f}")
        print(f"   Relative variance: {variance['relative_variance_pct']:.2f}%")
        print(
            f"   Passes ±5% threshold: {'✅' if variance['passes_threshold'] else '❌'}"
        )

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
