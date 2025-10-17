#!/usr/bin/env python3
"""
MeaningForge v4 - Domain Calibrator
Levels Meaning Quotient across science/social/metaphysical domains

v4 Calibrator: Cross-domain MQ variance ≤5%
"""
import sys
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from meaning_core import MeaningCore


class DomainCalibrator:
    """
    Calibrate Meaning Quotient across domains

    Adapts MQ scoring to domain-specific expectations:
    - Science: Lower transformative/resonance expectations (data-focused)
    - Social: Balanced expectations (reference domain)
    - Metaphysical: Higher transformative/conceptual expectations
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.meaning_core = MeaningCore()

        # Domain keywords (from TruthLens domain_adapter)
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

        # Domain-specific calibration scales
        # Science: Lower expectations for resonance/transformative (practical focus)
        # Social: Reference domain (no adjustment)
        # Metaphysical: Higher expectations for transformative (conceptual depth)
        self.domain_scales = self.config.get(
            "domain_scales",
            {
                "science": {
                    "relevance_weight": 1.05,  # Slightly higher weight on practical value
                    "resonance_weight": 0.95,  # Slightly lower resonance expectations
                    "transformative_weight": 0.98,  # Slightly lower transformative expectations
                    "baseline_shift": 0.005,  # Tiny boost (calibrated for ≤5%)
                },
                "social": {
                    "relevance_weight": 1.0,  # Standard
                    "resonance_weight": 1.0,  # Standard (reference)
                    "transformative_weight": 1.0,  # Standard
                    "baseline_shift": 0.0,  # No adjustment (reference domain)
                },
                "metaphysical": {
                    "relevance_weight": 0.95,  # Slightly lower practical expectations
                    "resonance_weight": 1.02,  # Slightly higher resonance expectations
                    "transformative_weight": 1.05,  # Slightly higher transformative expectations
                    "baseline_shift": 0.055,  # Boost to compensate for lower base MQ
                },
            },
        )

        self.target_variance = self.config.get("target_variance", 0.05)  # ≤5%

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

        # If scores are very low, default to social (middle ground)
        max_score = domain_scores[primary_domain]
        if max_score < 0.05:
            return "social", domain_scores

        return primary_domain, domain_scores

    def calibrate_mq(self, raw_mq: float, domain: str, components: dict = None) -> dict:
        """
        Calibrate MQ to domain-specific norms

        Args:
            raw_mq: Raw meaning quotient
            domain: Detected domain
            components: Component scores (relevance, resonance, transformative)

        Returns:
            Calibrated MQ and metadata
        """
        if domain not in self.domain_scales:
            return {
                "calibrated_mq": raw_mq,
                "raw_mq": raw_mq,
                "domain": domain,
                "calibration_applied": False,
            }

        scales = self.domain_scales[domain]

        # If components provided, re-weight them
        if components:
            relevance = components.get("relevance", 0)
            resonance = components.get("resonance", 0)
            transformative = components.get("transformative_potential", 0)

            # Apply domain-specific weights
            calibrated_mq = (
                relevance * 0.35 * scales["relevance_weight"]
                + resonance * 0.30 * scales["resonance_weight"]
                + transformative * 0.35 * scales["transformative_weight"]
            )

            # Normalize back to [0,1] range
            total_weight = (
                0.35 * scales["relevance_weight"]
                + 0.30 * scales["resonance_weight"]
                + 0.35 * scales["transformative_weight"]
            )
            calibrated_mq = calibrated_mq / total_weight

            # Apply baseline shift
            calibrated_mq = calibrated_mq + scales["baseline_shift"]

            # Clip to [0, 1]
            calibrated_mq = max(0.0, min(1.0, calibrated_mq))
        else:
            # Simple scaling without components
            calibrated_mq = raw_mq + scales["baseline_shift"]
            calibrated_mq = max(0.0, min(1.0, calibrated_mq))

        return {
            "calibrated_mq": calibrated_mq,
            "raw_mq": raw_mq,
            "domain": domain,
            "domain_scales": scales,
            "calibration_applied": True,
        }

    def grade_calibrated_mq(self, calibrated_mq: float) -> str:
        """
        Grade calibrated MQ

        Args:
            calibrated_mq: Calibrated meaning quotient

        Returns:
            Grade string
        """
        if calibrated_mq >= 0.85:
            return "EXCELLENT"
        elif calibrated_mq >= 0.75:
            return "GOOD"
        elif calibrated_mq >= 0.60:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"

    def analyze_document(
        self, text: str, raw_mq: float, components: dict = None
    ) -> dict:
        """
        Complete domain-aware MQ analysis

        Args:
            text: Document text
            raw_mq: Raw meaning quotient
            components: Component scores

        Returns:
            Complete analysis with domain calibration
        """
        # Detect domain
        domain, domain_scores = self.detect_domain(text)

        # Calibrate MQ
        calibration = self.calibrate_mq(raw_mq, domain, components)

        # Grade
        grade = self.grade_calibrated_mq(calibration["calibrated_mq"])

        return {
            "domain": domain,
            "domain_confidence": domain_scores,
            "raw_mq": raw_mq,
            "calibrated_mq": calibration["calibrated_mq"],
            "grade": grade,
            "calibration": calibration,
            "passes_threshold": calibration["calibrated_mq"] >= 0.60,
        }

    def compute_cross_domain_variance(
        self, domain_scores: dict[str, list[float]]
    ) -> dict:
        """
        Compute variance across domains

        Args:
            domain_scores: Dict mapping domain name to list of MQ scores

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
            "passes_threshold": relative_variance <= 5.0,  # ≤5% target
        }


def main():
    """CLI for domain calibrator"""
    import argparse

    parser = argparse.ArgumentParser(description="MeaningForge v4 Domain Calibrator")
    parser.add_argument(
        "command", choices=["test", "detect"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File to analyze")

    args = parser.parse_args()

    calibrator = DomainCalibrator()

    if args.command == "detect":
        if not args.file:
            print("❌ Error: --file required")
            return

        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        domain, scores = calibrator.detect_domain(text)

        print(f"🎯 Domain Detection: {file_path.name}")
        print(f"\n   Primary domain: {domain}")
        print("\n   Confidence scores:")
        for d, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"     • {d}: {score:.3f}")

    elif args.command == "test":
        print("🧪 Testing Domain Calibrator...")

        # Test 1: Domain detection
        science_text = "The experiment measured quantum coherence using laboratory protocols and statistical analysis."
        social_text = "Survey participants from diverse populations provided qualitative interview data on social behavior."
        meta_text = "The ontological argument examines fundamental principles of existence and consciousness through philosophical reasoning."

        print("\nTest 1 (Domain detection):")
        for label, text in [
            ("Science", science_text),
            ("Social", social_text),
            ("Metaphysical", meta_text),
        ]:
            domain, _ = calibrator.detect_domain(text)
            match = domain.startswith(label.lower()[:3])
            print(f"   {label}: {domain} {'✅' if match else '⚠️'}")

        # Test 2: MQ calibration
        print("\nTest 2 (MQ calibration):")
        raw_mq = 0.70

        for domain in ["science", "social", "metaphysical"]:
            result = calibrator.calibrate_mq(raw_mq, domain)
            print(f"   {domain}: {raw_mq:.3f} → {result['calibrated_mq']:.3f}")

        # Test 3: Cross-domain variance
        print("\nTest 3 (Cross-domain variance):")
        test_scores = {
            "science": [0.72, 0.74, 0.73],
            "social": [0.71, 0.73, 0.72],
            "metaphysical": [0.70, 0.72, 0.71],
        }

        variance = calibrator.compute_cross_domain_variance(test_scores)
        print(f"   Overall mean: {variance['overall_mean']:.3f}")
        print(f"   Relative variance: {variance['relative_variance_pct']:.2f}%")
        print(
            f"   Passes ≤5% threshold: {'✅' if variance['passes_threshold'] else '❌'}"
        )

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
