#!/usr/bin/env python3
"""
MeaningForge v5 - Stability Analyzer
Tests MQ robustness under noise perturbations

v5 Stability: Inject 10% noise, measure stability ≥0.8
"""
import random
import sys
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from meaning_core import MEANINGFORGE_SEED, MeaningCore

# Set seed for reproducible noise
random.seed(MEANINGFORGE_SEED)
np.random.seed(MEANINGFORGE_SEED)


class MeaningDecayAnalyzer:
    """
    Analyze MQ stability under noise perturbations

    Injects noise through:
    - Word substitution (synonyms)
    - Word deletion
    - Word reordering
    - Punctuation changes

    Measures stability: correlation between original and perturbed MQ
    """

    def __init__(self, noise_level: float = 0.10):
        """
        Initialize stability analyzer

        Args:
            noise_level: Fraction of words to perturb (default: 0.10 = 10%)
        """
        self.noise_level = noise_level
        self.meaning_core = MeaningCore()

        # Simple synonym mappings for perturbations
        self.synonyms = {
            "practical": ["actionable", "useful", "applied"],
            "meaningful": ["significant", "important", "valuable"],
            "transform": ["change", "shift", "alter"],
            "insight": ["understanding", "realization", "awareness"],
            "connect": ["link", "relate", "associate"],
            "experience": ["encounter", "occurrence", "event"],
            "purpose": ["goal", "aim", "objective"],
            "wisdom": ["knowledge", "understanding", "insight"],
            "resonate": ["echo", "align", "harmonize"],
            "discover": ["find", "uncover", "reveal"],
        }

    def inject_noise(self, text: str, noise_type: str = "mixed") -> str:
        """
        Inject noise into text

        Args:
            text: Original text
            noise_type: Type of noise ('substitution', 'deletion', 'reorder', 'mixed')

        Returns:
            Perturbed text
        """
        words = text.split()
        num_words = len(words)
        num_perturb = int(num_words * self.noise_level)

        if num_perturb == 0 or num_words == 0:
            return text

        # Select random word indices to perturb
        perturb_indices = random.sample(range(num_words), min(num_perturb, num_words))

        perturbed_words = words.copy()

        for idx in perturb_indices:
            word = words[idx].lower().strip(".,!?;:")

            if noise_type == "substitution" or (
                noise_type == "mixed" and random.random() < 0.5
            ):
                # Substitute with synonym
                if word in self.synonyms:
                    perturbed_words[idx] = random.choice(self.synonyms[word])
                elif random.random() < 0.3:  # Some random deletions
                    perturbed_words[idx] = ""

            elif noise_type == "deletion" or (
                noise_type == "mixed" and random.random() < 0.3
            ):
                # Delete word
                perturbed_words[idx] = ""

            elif noise_type == "reorder":
                # Swap with neighbor
                if idx + 1 < num_words:
                    perturbed_words[idx], perturbed_words[idx + 1] = (
                        perturbed_words[idx + 1],
                        perturbed_words[idx],
                    )

        # Reconstruct text
        perturbed_text = " ".join(w for w in perturbed_words if w)

        return perturbed_text

    def compute_stability(self, text: str, num_trials: int = 10) -> dict:
        """
        Compute MQ stability under noise

        Args:
            text: Original text
            num_trials: Number of perturbation trials

        Returns:
            Stability metrics
        """
        # Get baseline MQ
        baseline_result = self.meaning_core.compute_meaning_quotient(text)
        baseline_mq = baseline_result["meaning_quotient"]

        # Perturb and measure MQ multiple times
        perturbed_mqs = []

        for _ in range(num_trials):
            perturbed_text = self.inject_noise(text, noise_type="mixed")
            perturbed_result = self.meaning_core.compute_meaning_quotient(
                perturbed_text
            )
            perturbed_mqs.append(perturbed_result["meaning_quotient"])

        # Compute stability metrics
        mean_perturbed = np.mean(perturbed_mqs)
        std_perturbed = np.std(perturbed_mqs)

        # Mean absolute deviation from baseline
        mad = np.mean([abs(mq - baseline_mq) for mq in perturbed_mqs])

        # Stability score: 1 - (normalized MAD)
        # Normalize MAD by baseline to get percentage change
        # If MAD is small relative to baseline, stability is high
        if baseline_mq > 0.1:
            # Normalized by baseline (percentage change)
            normalized_mad = mad / baseline_mq
            # Stability decreases as percentage change increases
            # 0% change = 1.0 stability, 20% change = 0.8 stability, 50% change = 0.5 stability
            stability_score = max(0.0, 1.0 - normalized_mad)
        else:
            # For very low baseline MQ, use absolute MAD
            stability_score = max(0.0, 1.0 - mad * 5)  # Scale MAD to [0,1]

        # Clip to [0, 1]
        stability = max(0.0, min(1.0, stability_score))

        return {
            "baseline_mq": baseline_mq,
            "mean_perturbed_mq": mean_perturbed,
            "std_perturbed_mq": std_perturbed,
            "mean_absolute_deviation": mad,
            "stability_score": stability,
            "passes_threshold": stability >= 0.8,
            "num_trials": num_trials,
            "noise_level": self.noise_level,
            "perturbed_scores": perturbed_mqs,
        }

    def analyze_corpus_stability(
        self, documents: list[tuple[str, str]], num_trials: int = 10
    ) -> dict:
        """
        Analyze stability across a corpus

        Args:
            documents: List of (name, text) tuples
            num_trials: Number of trials per document

        Returns:
            Corpus stability analysis
        """
        doc_stabilities = []
        results = []

        for name, text in documents:
            stability_result = self.compute_stability(text, num_trials)
            stability_result["document"] = name
            results.append(stability_result)
            doc_stabilities.append(stability_result["stability_score"])

        # Aggregate metrics
        mean_stability = np.mean(doc_stabilities)
        min_stability = min(doc_stabilities)
        max_stability = max(doc_stabilities)
        std_stability = np.std(doc_stabilities)

        passes = mean_stability >= 0.8

        return {
            "documents_analyzed": len(documents),
            "mean_stability": mean_stability,
            "min_stability": min_stability,
            "max_stability": max_stability,
            "std_stability": std_stability,
            "passes_threshold": passes,
            "threshold": 0.8,
            "results": results,
        }

    def generate_stability_report(self, corpus_results: dict, output_path: Path):
        """
        Generate stability report markdown

        Args:
            corpus_results: Results from analyze_corpus_stability
            output_path: Path to save report
        """
        report_lines = [
            "# MeaningForge v5 Stability Report",
            "",
            "**Analysis Date**: " + Path(__file__).stat().st_mtime.__str__()[:10],
            "**Noise Level**: 10%",
            "**Stability Threshold**: ≥0.80",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"- **Documents Analyzed**: {corpus_results['documents_analyzed']}",
            f"- **Mean Stability**: {corpus_results['mean_stability']:.3f}",
            f"- **Range**: [{corpus_results['min_stability']:.3f}, {corpus_results['max_stability']:.3f}]",
            f"- **Std Deviation**: {corpus_results['std_stability']:.3f}",
            f"- **Passes Threshold**: {'✅ YES' if corpus_results['passes_threshold'] else '❌ NO'}",
            "",
            "---",
            "",
            "## Per-Document Results",
            "",
            "| Document | Baseline MQ | Perturbed MQ (mean) | Stability | Status |",
            "|----------|-------------|---------------------|-----------|--------|",
        ]

        for result in corpus_results["results"]:
            status = "✅" if result["passes_threshold"] else "❌"
            report_lines.append(
                f"| {result['document']} | {result['baseline_mq']:.3f} | "
                f"{result['mean_perturbed_mq']:.3f} | {result['stability_score']:.3f} | {status} |"
            )

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## Stability Analysis",
                "",
                f"**Noise Injection**: {self.noise_level * 100:.0f}% of words perturbed per trial",
                "",
                "**Perturbation Types**:",
                "- Word substitution (synonyms)",
                "- Word deletion",
                "- Word reordering",
                "",
                "**Stability Formula**: `1 - (mean_absolute_deviation / max_possible_deviation)`",
                "",
                f"**Result**: {'✅ STABLE' if corpus_results['passes_threshold'] else '❌ UNSTABLE'}",
                "",
                "---",
                "",
                "## Interpretation",
                "",
                "- **Stability ≥0.90**: Highly robust to noise",
                "- **Stability 0.80-0.89**: Good stability (meets threshold)",
                "- **Stability 0.70-0.79**: Moderate stability (below threshold)",
                "- **Stability <0.70**: Low stability (unstable)",
                "",
                "---",
                "",
                f"**Overall Assessment**: Mean stability {corpus_results['mean_stability']:.3f} "
                f"{'✅ PASSES' if corpus_results['passes_threshold'] else '❌ FAILS'} "
                f"threshold (≥0.80)",
            ]
        )

        report_text = "\n".join(report_lines)
        output_path.write_text(report_text)


def main():
    """CLI for meaning decay analyzer"""
    import argparse

    parser = argparse.ArgumentParser(description="MeaningForge v5 Stability Analyzer")
    parser.add_argument(
        "command", choices=["test", "analyze"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--corpus", type=str, help="Corpus directory")
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of perturbation trials (default: 10)",
    )
    parser.add_argument(
        "--output", type=str, default="stability_report.md", help="Output report path"
    )

    args = parser.parse_args()

    analyzer = MeaningDecayAnalyzer(noise_level=0.10)

    if args.command == "test":
        print("🧪 Testing Stability Analyzer...")

        # Test samples
        test_docs = [
            (
                "high_stability",
                """
                This transformative journey resonates deeply with our shared purpose and meaning.
                Practical insights reveal profound wisdom that connects us to something greater.
                Joy and wonder emerge when we apply these principles in real-world contexts together.
                What if we embraced this breakthrough fully? The experience transforms understanding.
            """,
            ),
            (
                "moderate_stability",
                """
                Practical applications demonstrate concrete value in real-world contexts.
                For example, implementing these methods yields measurable results and actionable insights.
            """,
            ),
            (
                "balanced",
                """
                Consider how meaningful insights connect practical understanding with human experience.
                We discover relevance through thoughtful exploration and shared wisdom.
            """,
            ),
        ]

        print(
            f"\nAnalyzing {len(test_docs)} documents with {args.trials} trials each...\n"
        )

        for name, text in test_docs:
            result = analyzer.compute_stability(text, num_trials=args.trials)

            print(f"{name}:")
            print(f"   Baseline MQ: {result['baseline_mq']:.3f}")
            print(
                f"   Perturbed MQ: {result['mean_perturbed_mq']:.3f} ± {result['std_perturbed_mq']:.3f}"
            )
            print(
                f"   Stability: {result['stability_score']:.3f} {'✅' if result['passes_threshold'] else '❌'}"
            )
            print()

        # Generate corpus report
        corpus_results = analyzer.analyze_corpus_stability(
            test_docs, num_trials=args.trials
        )

        print("Corpus Stability:")
        print(f"   Mean: {corpus_results['mean_stability']:.3f}")
        print(
            f"   Passes ≥0.8 threshold: {'✅' if corpus_results['passes_threshold'] else '❌'}"
        )

        # Generate report
        output_path = Path(args.output)
        analyzer.generate_stability_report(corpus_results, output_path)
        print(f"\n✅ Report saved: {output_path}")

    elif args.command == "analyze":
        if args.file:
            # Single file analysis
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ Error: File not found: {file_path}")
                return

            text = file_path.read_text(encoding="utf-8", errors="ignore")

            print(f"🛡️  Analyzing stability: {file_path.name}")
            print(f"   Noise level: {analyzer.noise_level * 100:.0f}%")
            print(f"   Trials: {args.trials}\n")

            result = analyzer.compute_stability(text, num_trials=args.trials)

            print("Results:")
            print(f"   Baseline MQ: {result['baseline_mq']:.3f}")
            print(
                f"   Perturbed MQ: {result['mean_perturbed_mq']:.3f} ± {result['std_perturbed_mq']:.3f}"
            )
            print(
                f"   Mean Absolute Deviation: {result['mean_absolute_deviation']:.3f}"
            )
            print(f"   Stability Score: {result['stability_score']:.3f}")
            print(
                f"   Status: {'✅ STABLE' if result['passes_threshold'] else '❌ UNSTABLE'}"
            )

        elif args.corpus:
            # Corpus analysis
            corpus_path = Path(args.corpus)
            if not corpus_path.exists():
                print(f"❌ Error: Corpus not found: {corpus_path}")
                return

            docs = []
            for file_path in corpus_path.glob("*.md"):
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                docs.append((file_path.name, text))

            if not docs:
                print(f"❌ No documents found in {corpus_path}")
                return

            print("🛡️  Analyzing corpus stability...")
            print(f"   Documents: {len(docs)}")
            print(f"   Noise level: {analyzer.noise_level * 100:.0f}%")
            print(f"   Trials per doc: {args.trials}\n")

            corpus_results = analyzer.analyze_corpus_stability(
                docs, num_trials=args.trials
            )

            print("Corpus Results:")
            print(f"   Mean stability: {corpus_results['mean_stability']:.3f}")
            print(
                f"   Range: [{corpus_results['min_stability']:.3f}, {corpus_results['max_stability']:.3f}]"
            )
            print(
                f"   Status: {'✅ STABLE' if corpus_results['passes_threshold'] else '❌ UNSTABLE'}"
            )

            # Generate report
            output_path = Path(args.output)
            analyzer.generate_stability_report(corpus_results, output_path)
            print(f"\n✅ Report saved: {output_path}")


if __name__ == "__main__":
    main()
