#!/usr/bin/env python3
"""
TruthLens v5 - Performance and Reproducibility Test Suite
Validates runtime and determinism thresholds

v5 Repro Standard: ≥99% reproducibility, <1s/doc runtime
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthlens_core import TruthLensCore


@pytest.fixture
def truthlens():
    """Create TruthLensCore instance"""
    return TruthLensCore()


@pytest.fixture
def sample_text():
    """Sample research text"""
    return """
    The study demonstrates significant effects (n=200, p<0.001) as shown by Smith et al. (2020).
    Results indicate strong correlation (r=0.65, 95% CI [0.60, 0.70]) per Jones (2019).
    Analysis reveals patterns consistent with theory (doi.org/10.1234/abc).
    """


@pytest.fixture
def longer_text():
    """Longer research text for performance testing"""
    return """
    # Research on Cooperation Dynamics
    
    The study shows that openness significantly increases cooperation in social networks 
    (Smith et al. 2020). We found a strong correlation between openness and cooperation 
    (r=0.65, p<0.001, n=200).
    
    ## Methods
    
    The data demonstrates clear patterns as shown in Figure 1 and Table 2. Previous work 
    suggests similar results in organizational contexts (Jones et al. 2019, doi.org/10.1234/cooperation).
    
    ## Results
    
    Results indicate that shared meaning enhances coordination (Garcia & Martinez 2023). 
    Analysis reveals significant effects across multiple domains (F(2,197)=45.2, p<0.001) 
    as presented in https://example.com/study.
    
    Evidence from field studies suggests strong support for this hypothesis (Lee 2021, 
    arxiv.org/abs/2103.12345). The findings demonstrate robustness across different 
    populations (Brown 2018).
    
    ## Discussion
    
    Research shows that resilience improves when cooperation increases (Wilson 2022). 
    This finding is consistent with theoretical predictions (Davis et al. 2020, Figure 3).
    """


class TestReproducibility:
    """Test reproducibility (determinism)"""

    def test_perfect_determinism_single_text(self, truthlens, sample_text):
        """Same text should produce identical Ti across runs"""
        ti_scores = []

        for _ in range(10):
            result = truthlens.compute_truth_index(sample_text)
            ti_scores.append(result["truth_index"])

        # Should be perfectly deterministic (zero variance)
        std_dev = np.std(ti_scores)

        assert std_dev < 1e-10, f"Non-deterministic behavior detected: std={std_dev}"

    def test_reproducibility_threshold(self, truthlens):
        """Variance should be <0.01 (99% reproducibility)"""
        texts = [
            "Study shows effects (p<0.001, Smith 2020).",
            "Results indicate patterns (Jones 2019, Figure 1).",
            "Analysis reveals trends (r=0.65, Lee 2021).",
        ]

        # Run each text multiple times
        all_std_devs = []

        for text in texts:
            ti_scores = []
            for _ in range(5):
                result = truthlens.compute_truth_index(text)
                ti_scores.append(result["truth_index"])

            std_dev = np.std(ti_scores)
            all_std_devs.append(std_dev)

        mean_std_dev = np.mean(all_std_devs)

        # 99% reproducibility = std dev < 0.01
        assert mean_std_dev < 0.01, f"Reproducibility below 99%: std={mean_std_dev}"

    def test_component_consistency(self, truthlens, sample_text):
        """Component scores should be consistent"""
        results = []

        for _ in range(5):
            result = truthlens.compute_truth_index(sample_text)
            results.append(
                {
                    "clarity": result.get("clarity", result.get("claim_clarity", 0)),
                    "citations": result.get(
                        "citations", result.get("citation_presence", 0)
                    ),
                    "causality": result.get(
                        "causality", result.get("causal_tokens", 0)
                    ),
                }
            )

        # Check each component
        for component in ["clarity", "citations", "causality"]:
            values = [r[component] for r in results]
            std_dev = np.std(values)
            assert std_dev < 1e-10, f"{component} not deterministic: std={std_dev}"


class TestPerformance:
    """Test runtime performance"""

    def test_runtime_single_doc_threshold(self, truthlens, longer_text):
        """Single document should process in <1s"""
        start_time = time.time()
        result = truthlens.compute_truth_index(longer_text)
        end_time = time.time()

        runtime = end_time - start_time

        assert runtime < 1.0, f"Runtime {runtime:.3f}s exceeds 1s threshold"

    def test_runtime_multiple_docs(self, truthlens):
        """Multiple documents should average <1s each"""
        texts = [
            "Study shows effects (n=100, p<0.001) per Smith (2020) and Jones (2019).",
            "Results indicate correlation (r=0.65) as shown in Figure 1 (Lee 2021).",
            "Analysis reveals patterns (doi.org/10.1234/abc, Brown 2018).",
            "Data demonstrates trends (95% CI [0.60, 0.70], Garcia 2022).",
            "Research suggests mechanisms (Wilson 2021, arxiv.org/abs/2103).",
        ] * 10  # 50 texts total

        start_time = time.time()
        for text in texts:
            result = truthlens.compute_truth_index(text)
        end_time = time.time()

        total_runtime = end_time - start_time
        avg_runtime = total_runtime / len(texts)

        assert (
            avg_runtime < 1.0
        ), f"Average runtime {avg_runtime:.3f}s exceeds 1s threshold"

    def test_performance_stability(self, truthlens, sample_text):
        """Runtime should be stable across multiple runs"""
        runtimes = []

        for _ in range(20):
            start_time = time.time()
            result = truthlens.compute_truth_index(sample_text)
            end_time = time.time()

            runtimes.append(end_time - start_time)

        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)

        # Runtime should have low variance (coefficient of variation < 0.5)
        cv = std_runtime / mean_runtime if mean_runtime > 0 else 0

        assert cv < 0.5, f"Runtime unstable: CV={cv:.3f}"


class TestScalability:
    """Test scalability with larger document counts"""

    def test_batch_processing_efficiency(self, truthlens):
        """Batch processing should maintain performance"""
        num_docs = 100

        texts = [
            f"Study {i} shows results (p<0.001, Smith {2020+i%5})."
            for i in range(num_docs)
        ]

        start_time = time.time()
        for text in texts:
            result = truthlens.compute_truth_index(text)
        end_time = time.time()

        total_runtime = end_time - start_time
        avg_runtime = total_runtime / num_docs

        print(f"\nBatch Processing ({num_docs} docs):")
        print(f"   Total: {total_runtime:.2f}s")
        print(f"   Average: {avg_runtime:.4f}s/doc")
        print(f"   Throughput: {num_docs/total_runtime:.1f} docs/s")

        assert (
            avg_runtime < 1.0
        ), f"Batch average {avg_runtime:.3f}s exceeds 1s threshold"

    def test_memory_efficiency(self, truthlens):
        """Should not accumulate excessive state"""
        import sys

        # Get initial size
        initial_size = sys.getsizeof(truthlens)

        # Process many documents
        for i in range(100):
            text = f"Study shows effects (Smith {2020+i})."
            result = truthlens.compute_truth_index(text)

        # Get final size
        final_size = sys.getsizeof(truthlens)

        # Should not grow significantly
        size_growth = final_size - initial_size

        assert size_growth < 1000000, f"Memory growth {size_growth} bytes too large"


class TestValidationReport:
    """Test validation report generation"""

    def test_validation_report_structure(self):
        """Validation report should have required fields"""
        from truthlens_validator import TruthLensValidator

        validator = TruthLensValidator()

        # Use test_corpus_v4 if available
        corpus_path = Path(__file__).parent.parent / "test_corpus_v4"

        if not corpus_path.exists():
            pytest.skip("Test corpus not found")

        docs = list(corpus_path.glob("*.md"))[:3]

        if not docs:
            pytest.skip("No documents in test corpus")

        report = validator.validate(docs, num_runs=3, output_path=None)

        # Check required fields
        assert "timestamp" in report
        assert "version" in report
        assert "reproducibility" in report
        assert "performance" in report
        assert "overall" in report

        # Check reproducibility fields
        assert "mean_std_dev" in report["reproducibility"]
        assert "reproducibility_pct" in report["reproducibility"]
        assert "passes" in report["reproducibility"]

        # Check performance fields
        assert "mean_runtime" in report["performance"]
        assert "passes" in report["performance"]

        # Check overall
        assert "passes_all" in report["overall"]


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🔄 TruthLens v5 Performance & Reproducibility Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Reproducibility: Determinism and variance <0.01")
    print("  • Performance: Runtime <1s per document")
    print("  • Scalability: Batch processing efficiency")
    print("  • Validation: Report generation and structure")
    print("\nAcceptance Criteria:")
    print("  • Reproducibility: ≥99% (std dev <0.01)")
    print("  • Runtime: <1s per document average")
    print("  • Scalability: 100+ docs maintaining performance")
    print("\nRun with: pytest tests/test_truthlens_perf_v5.py -v -s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])
