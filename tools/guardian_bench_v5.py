#!/usr/bin/env python3
"""
Guardian v5 Benchmarking Tool
Measures performance, determinism, and calibration across document corpus
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# Add Guardian v4 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "qc" / "guardian_v4"))

from guardian_v4 import GuardianV4


class GuardianBenchmarkV5:
    """
    Comprehensive benchmarking for Guardian v5 stabilization
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.benchmark_dir = self.root / "qc" / "guardian_v4" / "benchmarks"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

        self.guardian = GuardianV4(self.root)

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents"""
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except:
            return "error"

    def find_target_documents(self) -> list:
        """Find all markdown files in target directories"""
        patterns = {
            "docs": self.root / "docs",
            "qc": self.root / "qc",
            "templates": self.root / "templates",
            "discovery_results": self.root / "discovery_results",
        }

        documents = []
        exclude_patterns = ["node_modules", ".venv", ".git", "backups"]

        # Scan each target directory
        for category, base_path in patterns.items():
            if not base_path.exists():
                continue

            for md_file in base_path.rglob("*.md"):
                # Skip excluded paths
                if any(ex in str(md_file) for ex in exclude_patterns):
                    continue

                # Skip QC_REPORT files (contain violations as metadata)
                if "QC_REPORT" in md_file.name:
                    continue

                documents.append(
                    {
                        "category": category,
                        "path": md_file,
                        "relative_path": str(md_file.relative_to(self.root)),
                        "size_bytes": md_file.stat().st_size,
                    }
                )

        return documents

    def benchmark_single_document(self, doc_info: dict, run_number: int = 1) -> dict:
        """
        Benchmark validation of a single document
        Returns: {score, components, runtime_ms, file_hash}
        """
        file_path = doc_info["path"]

        # Compute file hash for tracking
        file_hash = self.compute_file_hash(file_path)

        # Time the validation
        start_time = time.perf_counter()

        try:
            result = self.guardian.validate_document(file_path)

            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000

            if "error" in result:
                return {
                    "file": doc_info["relative_path"],
                    "category": doc_info["category"],
                    "file_hash": file_hash,
                    "run_number": run_number,
                    "error": result["error"],
                    "runtime_ms": runtime_ms,
                }

            return {
                "file": doc_info["relative_path"],
                "category": doc_info["category"],
                "file_hash": file_hash,
                "run_number": run_number,
                "guardian_score": result["guardian_alignment_score"],
                "objectivity": result["metrics"]["objectivity_score"],
                "transparency": result["metrics"]["transparency_index_v2"],
                "language_safety": result["metrics"]["language_safety_score"],
                "sentiment": result["metrics"]["sentiment_neutrality"],
                "risk_level": result["risk_assessment"]["risk_level"],
                "runtime_ms": runtime_ms,
                "passes_threshold": result["passes_threshold"],
            }

        except Exception as e:
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000

            return {
                "file": doc_info["relative_path"],
                "category": doc_info["category"],
                "file_hash": file_hash,
                "run_number": run_number,
                "error": str(e),
                "runtime_ms": runtime_ms,
            }

    def run_determinism_test(self, doc_info: dict, n_runs: int = 5) -> dict:
        """
        Test determinism by running validation n times on same document
        Measures variance in scores
        """
        results = []

        for i in range(n_runs):
            result = self.benchmark_single_document(doc_info, run_number=i + 1)
            if "error" not in result:
                results.append(result)

        if not results:
            return {"file": doc_info["relative_path"], "error": "All runs failed"}

        # Compute variance metrics
        scores = [r["guardian_score"] for r in results]

        variance_metrics = {
            "file": doc_info["relative_path"],
            "n_runs": len(results),
            "mean_score": mean(scores),
            "stdev_score": stdev(scores) if len(scores) > 1 else 0.0,
            "min_score": min(scores),
            "max_score": max(scores),
            "score_range": max(scores) - min(scores),
            "variance_pct": (
                (stdev(scores) / mean(scores) * 100)
                if len(scores) > 1 and mean(scores) > 0
                else 0.0
            ),
            "is_deterministic": (stdev(scores) if len(scores) > 1 else 0.0) < 0.02,
            "mean_runtime_ms": mean([r["runtime_ms"] for r in results]),
            "all_results": results,
        }

        return variance_metrics

    def run_full_benchmark(self, max_docs: int = None, n_runs: int = 1) -> dict:
        """
        Run benchmark across all target documents
        """
        print("🔍 Guardian v5 Benchmark: Finding documents...")
        documents = self.find_target_documents()

        if max_docs:
            documents = documents[:max_docs]

        print(f"✅ Found {len(documents)} documents to benchmark")

        # Run benchmarks
        results = []

        for i, doc_info in enumerate(documents):
            print(
                f"[{i+1}/{len(documents)}] {doc_info['relative_path'][:60]:<60}",
                end="\r",
            )

            if n_runs > 1:
                # Determinism test
                result = self.run_determinism_test(doc_info, n_runs=n_runs)
            else:
                # Single run benchmark
                result = self.benchmark_single_document(doc_info)

            results.append(result)

        print()  # New line after progress

        # Aggregate statistics
        valid_results = [r for r in results if "error" not in r]

        if valid_results and n_runs == 1:
            scores = [r["guardian_score"] for r in valid_results]
            runtimes = [r["runtime_ms"] for r in valid_results]

            aggregate = {
                "total_documents": len(documents),
                "successful_validations": len(valid_results),
                "failed_validations": len(documents) - len(valid_results),
                "mean_score": mean(scores),
                "stdev_score": stdev(scores) if len(scores) > 1 else 0.0,
                "min_score": min(scores),
                "max_score": max(scores),
                "median_runtime_ms": sorted(runtimes)[len(runtimes) // 2],
                "mean_runtime_ms": mean(runtimes),
                "max_runtime_ms": max(runtimes),
                "passing_rate": sum(
                    1 for r in valid_results if r.get("passes_threshold")
                )
                / len(valid_results),
                "results": valid_results,
            }
        elif valid_results and n_runs > 1:
            # Determinism test aggregate
            variances = [
                r["variance_pct"] for r in valid_results if "variance_pct" in r
            ]
            deterministic_count = sum(
                1 for r in valid_results if r.get("is_deterministic")
            )

            aggregate = {
                "total_documents": len(documents),
                "successful_tests": len(valid_results),
                "deterministic_documents": deterministic_count,
                "determinism_rate": (
                    deterministic_count / len(valid_results) if valid_results else 0.0
                ),
                "mean_variance_pct": mean(variances) if variances else 0.0,
                "max_variance_pct": max(variances) if variances else 0.0,
                "reproducibility_pct": (
                    (deterministic_count / len(valid_results) * 100)
                    if valid_results
                    else 0.0
                ),
                "results": valid_results,
            }
        else:
            aggregate = {
                "total_documents": len(documents),
                "successful_validations": 0,
                "results": results,
            }

        return aggregate

    def save_baseline(self, results: dict, baseline_name: str = "v5_baseline"):
        """Save benchmark results as baseline CSV"""
        csv_path = self.benchmark_dir / f"{baseline_name}.csv"

        # Extract flat results
        if "results" in results:
            flat_results = results["results"]
        else:
            flat_results = []

        if not flat_results:
            print("⚠️  No results to save")
            return

        # Write CSV
        fieldnames = list(flat_results[0].keys())
        if "all_results" in fieldnames:
            fieldnames.remove("all_results")  # Too nested for CSV

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_results)

        print(f"✅ Baseline saved: {csv_path}")
        print(f"   Records: {len(flat_results)}")

        # Also save aggregate JSON
        json_path = self.benchmark_dir / f"{baseline_name}_aggregate.json"
        with open(json_path, "w") as f:
            # Remove detailed results for aggregate JSON
            aggregate_summary = {k: v for k, v in results.items() if k != "results"}
            json.dump(aggregate_summary, f, indent=2)

        print(f"✅ Aggregate saved: {json_path}")

        return csv_path


def main():
    """CLI interface for benchmarking"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v5 Benchmarking Tool")
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Run determinism test (5 runs per document)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to benchmark",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="v5_baseline",
        help="Baseline name for saving results",
    )

    args = parser.parse_args()

    bench = GuardianBenchmarkV5()

    print("=" * 70)
    print("🔍 Guardian v5 Benchmark")
    print("=" * 70)

    if args.determinism:
        print("\n📊 Running determinism test (5 runs per document)...")
        results = bench.run_full_benchmark(max_docs=args.max_docs, n_runs=5)

        print("\n✅ Determinism test complete")
        print(f"   Documents tested: {results['total_documents']}")
        print(f"   Successful: {results.get('successful_tests', 0)}")
        print(f"   Deterministic: {results.get('deterministic_documents', 0)}")
        print(f"   Reproducibility: {results.get('reproducibility_pct', 0):.1f}%")
        print(f"   Mean variance: {results.get('mean_variance_pct', 0):.3f}%")
        print(f"   Max variance: {results.get('max_variance_pct', 0):.3f}%")

        # Check acceptance criteria
        if results.get("reproducibility_pct", 0) >= 99.0:
            print("\n✅ PASS: Reproducibility ≥ 99%")
        else:
            print("\n❌ FAIL: Reproducibility < 99%")
            sys.exit(1)

        if results.get("max_variance_pct", 100) < 2.0:
            print("✅ PASS: Variance < 2%")
        else:
            print("❌ FAIL: Variance ≥ 2%")
            sys.exit(1)

    else:
        print("\n📊 Running performance benchmark (single run per document)...")
        results = bench.run_full_benchmark(max_docs=args.max_docs, n_runs=1)

        print("\n✅ Benchmark complete")
        print(f"   Documents: {results['total_documents']}")
        print(f"   Successful: {results.get('successful_validations', 0)}")
        print(f"   Mean score: {results.get('mean_score', 0):.1f}/100")
        print(f"   Passing rate: {results.get('passing_rate', 0)*100:.1f}%")
        print(f"   Median runtime: {results.get('median_runtime_ms', 0):.0f}ms")
        print(f"   Mean runtime: {results.get('mean_runtime_ms', 0):.0f}ms")
        print(f"   Max runtime: {results.get('max_runtime_ms', 0):.0f}ms")

        # Check performance criteria
        if results.get("median_runtime_ms", 10000) < 1000:
            print("\n✅ PASS: Median runtime < 1s")
        else:
            print("\n❌ FAIL: Median runtime ≥ 1s")
            sys.exit(1)

    # Save baseline
    bench.save_baseline(results, baseline_name=args.baseline)

    print("\n📁 Outputs:")
    print(f"   Baseline CSV: qc/guardian_v4/benchmarks/{args.baseline}.csv")
    print(
        f"   Aggregate JSON: qc/guardian_v4/benchmarks/{args.baseline}_aggregate.json"
    )


if __name__ == "__main__":
    main()
