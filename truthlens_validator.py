#!/usr/bin/env python3
"""
TruthLens v5 - Reproducibility Validator
Ensures ≥99% reproducibility and <1s/doc runtime

v5 Repro Standard: 5-run variance check + performance validation
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from truthlens_core import TruthLensCore


class TruthLensValidator:
    """
    Validate TruthLens reproducibility and performance

    Acceptance Criteria:
    - Reproducibility: ≥99% (std dev < 0.01 on Ti scores)
    - Runtime: <1s per document average
    - Consistency: Same results across 5 runs
    """

    def __init__(self, config_path: Path = None):
        self.config_path = config_path
        self.results = []
        self.validation_timestamp = datetime.now().isoformat()

    def run_reproducibility_test(
        self, test_documents: list[Path], num_runs: int = 5
    ) -> dict:
        """
        Run reproducibility test across multiple runs

        Args:
            test_documents: List of document paths to test
            num_runs: Number of runs to perform (default: 5)

        Returns:
            Reproducibility metrics
        """
        print(
            f"🔄 Running reproducibility test ({num_runs} runs on {len(test_documents)} docs)..."
        )

        # Store Ti scores for each document across runs
        doc_scores = {str(doc): [] for doc in test_documents}

        for run_idx in range(num_runs):
            print(f"   Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)

            # Reinitialize TruthLens for each run
            tl = TruthLensCore(config_path=self.config_path)

            for doc_path in test_documents:
                try:
                    text = doc_path.read_text(encoding="utf-8", errors="ignore")
                    result = tl.compute_truth_index(text)
                    ti = result["truth_index"]
                    doc_scores[str(doc_path)].append(ti)
                except Exception as e:
                    print(f"\n⚠️  Error processing {doc_path.name}: {e}")
                    doc_scores[str(doc_path)].append(None)

            print("✓")

        # Compute reproducibility metrics
        variances = []
        std_devs = []
        max_deviations = []

        for doc_path, scores in doc_scores.items():
            # Filter out None values
            valid_scores = [s for s in scores if s is not None]

            if len(valid_scores) >= 2:
                variance = np.var(valid_scores)
                std_dev = np.std(valid_scores)
                max_dev = max(valid_scores) - min(valid_scores)

                variances.append(variance)
                std_devs.append(std_dev)
                max_deviations.append(max_dev)

        mean_variance = np.mean(variances) if variances else 0.0
        mean_std_dev = np.mean(std_devs) if std_devs else 0.0
        max_std_dev = max(std_devs) if std_devs else 0.0
        mean_max_deviation = np.mean(max_deviations) if max_deviations else 0.0

        # Reproducibility score (inverse of variance, scaled to percentage)
        # 99% = std dev < 0.01
        reproducibility_pct = max(0, min(100, 100 * (1 - mean_std_dev * 100)))

        passes_reproducibility = mean_std_dev < 0.01  # 99% threshold

        return {
            "num_runs": num_runs,
            "num_documents": len(test_documents),
            "mean_variance": mean_variance,
            "mean_std_dev": mean_std_dev,
            "max_std_dev": max_std_dev,
            "mean_max_deviation": mean_max_deviation,
            "reproducibility_pct": reproducibility_pct,
            "passes_reproducibility": passes_reproducibility,
            "doc_scores": doc_scores,
        }

    def run_performance_test(self, test_documents: list[Path]) -> dict:
        """
        Run performance test (runtime measurement)

        Args:
            test_documents: List of document paths to test

        Returns:
            Performance metrics
        """
        print(f"⚡ Running performance test on {len(test_documents)} docs...")

        tl = TruthLensCore(config_path=self.config_path)

        runtimes = []

        for doc_path in test_documents:
            try:
                text = doc_path.read_text(encoding="utf-8", errors="ignore")

                start_time = time.time()
                result = tl.compute_truth_index(text)
                end_time = time.time()

                runtime = end_time - start_time
                runtimes.append(runtime)

            except Exception as e:
                print(f"⚠️  Error processing {doc_path.name}: {e}")

        if not runtimes:
            return {
                "num_documents": len(test_documents),
                "mean_runtime": 0.0,
                "passes_performance": False,
            }

        mean_runtime = np.mean(runtimes)
        median_runtime = np.median(runtimes)
        max_runtime = max(runtimes)
        min_runtime = min(runtimes)

        passes_performance = mean_runtime < 1.0  # <1s per doc

        print(f"   Mean runtime: {mean_runtime:.3f}s per doc")

        return {
            "num_documents": len(test_documents),
            "mean_runtime": mean_runtime,
            "median_runtime": median_runtime,
            "max_runtime": max_runtime,
            "min_runtime": min_runtime,
            "total_runtime": sum(runtimes),
            "passes_performance": passes_performance,
            "runtimes": runtimes,
        }

    def validate(
        self, test_documents: list[Path], num_runs: int = 5, output_path: Path = None
    ) -> dict:
        """
        Complete validation (reproducibility + performance)

        Args:
            test_documents: Documents to validate
            num_runs: Number of reproducibility runs
            output_path: Path to save validation report

        Returns:
            Complete validation report
        """
        print("\n" + "=" * 70)
        print("🔄 TruthLens v5 Reproducibility Validator")
        print("=" * 70)

        # Run tests
        repro_results = self.run_reproducibility_test(test_documents, num_runs)
        perf_results = self.run_performance_test(test_documents)

        # Overall pass/fail
        passes_all = (
            repro_results["passes_reproducibility"]
            and perf_results["passes_performance"]
        )

        # Build report
        validation_report = {
            "timestamp": self.validation_timestamp,
            "version": "5.0-repro-standard",
            "test_documents": [str(p) for p in test_documents],
            "reproducibility": {
                "num_runs": repro_results["num_runs"],
                "mean_std_dev": repro_results["mean_std_dev"],
                "max_std_dev": repro_results["max_std_dev"],
                "reproducibility_pct": repro_results["reproducibility_pct"],
                "passes": repro_results["passes_reproducibility"],
                "threshold": "<0.01 std dev (≥99% reproducibility)",
            },
            "performance": {
                "num_documents": perf_results["num_documents"],
                "mean_runtime": perf_results["mean_runtime"],
                "median_runtime": perf_results["median_runtime"],
                "passes": perf_results["passes_performance"],
                "threshold": "<1.0s per document",
            },
            "overall": {
                "passes_all": passes_all,
                "passes_reproducibility": repro_results["passes_reproducibility"],
                "passes_performance": perf_results["passes_performance"],
            },
        }

        # Display results
        print("\n" + "=" * 70)
        print("📊 VALIDATION RESULTS")
        print("=" * 70)

        print("\n   Reproducibility:")
        print(f"     • Runs: {repro_results['num_runs']}")
        print(f"     • Mean std dev: {repro_results['mean_std_dev']:.6f}")
        print(f"     • Reproducibility: {repro_results['reproducibility_pct']:.2f}%")
        print(
            f"     • Status: {'✅ PASS' if repro_results['passes_reproducibility'] else '❌ FAIL'}"
        )

        print("\n   Performance:")
        print(f"     • Documents: {perf_results['num_documents']}")
        print(f"     • Mean runtime: {perf_results['mean_runtime']:.3f}s/doc")
        print(f"     • Median runtime: {perf_results['median_runtime']:.3f}s/doc")
        print(
            f"     • Status: {'✅ PASS' if perf_results['passes_performance'] else '❌ FAIL'}"
        )

        print(f"\n   Overall: {'✅ PASS' if passes_all else '❌ FAIL'}")

        # Save report (convert numpy types to native Python)
        if output_path:
            # Convert numpy bools to Python bools
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                    return (
                        float(obj)
                        if isinstance(obj, (np.floating, np.integer))
                        else bool(obj)
                    )
                return obj

            clean_report = convert_numpy(validation_report)
            output_path.write_text(json.dumps(clean_report, indent=2))
            print(f"\n✅ Validation report saved: {output_path}")

        print("=" * 70 + "\n")

        return validation_report


def main():
    """CLI for TruthLens validator"""
    import argparse

    parser = argparse.ArgumentParser(
        description="TruthLens v5 Reproducibility Validator"
    )
    parser.add_argument(
        "command", choices=["validate", "quick"], help="Command to execute"
    )
    parser.add_argument("--corpus", type=str, help="Path to corpus directory")
    parser.add_argument(
        "--config", type=str, default="truth_index_v1.yml", help="Configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of reproducibility runs (default: 5)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of documents to test"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = None

    # Initialize validator
    validator = TruthLensValidator(config_path=config_path)

    if args.command == "quick":
        # Quick validation with small corpus
        print("🚀 Quick validation mode")

        # Use test_corpus_v4 if available
        corpus_path = Path("test_corpus_v4")
        if not corpus_path.exists():
            print("❌ No test corpus found")
            return

        docs = list(corpus_path.glob("*.md"))[:5]  # First 5 docs

        if not docs:
            print("❌ No documents found in corpus")
            return

        report = validator.validate(docs, num_runs=3, output_path=Path(args.output))

    elif args.command == "validate":
        if not args.corpus:
            print("❌ Error: --corpus required for validate command")
            return

        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"❌ Error: Corpus not found: {corpus_path}")
            return

        # Collect documents
        docs = []
        for ext in ["*.md", "*.txt"]:
            docs.extend(corpus_path.glob(ext))

        if args.limit:
            docs = docs[: args.limit]

        if not docs:
            print(f"❌ No documents found in {corpus_path}")
            return

        print(f"📁 Found {len(docs)} documents")

        report = validator.validate(
            docs, num_runs=args.runs, output_path=Path(args.output)
        )

        # Exit with error code if validation fails
        if not report["overall"]["passes_all"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
