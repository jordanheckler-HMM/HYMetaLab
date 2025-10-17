#!/usr/bin/env python3
"""
Guardian v5 Runtime Performance Tests
Ensures <1000ms per document validation
"""
import sys
import time
from pathlib import Path
from statistics import mean, median

import pytest

# Add Guardian v4 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "qc" / "guardian_v4"))

from guardian_v4 import GuardianV4

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    "README.md",
    "ETHICS.md",
    "templates/RESEARCH_DISCLAIMER.md",
]


@pytest.fixture
def guardian():
    """Create Guardian v4 instance"""
    return GuardianV4()


def test_single_document_runtime(guardian):
    """Test that single document validation completes in <1s"""
    test_file = Path("README.md")
    if not test_file.exists():
        pytest.skip("README.md not found")

    start = time.perf_counter()
    result = guardian.validate_document(test_file)
    end = time.perf_counter()

    runtime_ms = (end - start) * 1000

    print(f"\n   File: {test_file.name}")
    print(f"   Runtime: {runtime_ms:.1f}ms")

    assert runtime_ms < 1000, f"Runtime {runtime_ms:.1f}ms exceeds 1000ms threshold"
    assert "error" not in result, f"Validation failed: {result.get('error')}"


@pytest.mark.parametrize("test_file", SAMPLE_DOCUMENTS)
def test_multiple_documents_runtime(guardian, test_file):
    """Test runtime on multiple sample documents"""
    file_path = Path(test_file)
    if not file_path.exists():
        pytest.skip(f"{test_file} not found")

    start = time.perf_counter()
    result = guardian.validate_document(file_path)
    end = time.perf_counter()

    runtime_ms = (end - start) * 1000

    assert runtime_ms < 1000, f"{test_file}: Runtime {runtime_ms:.1f}ms exceeds 1000ms"


def test_corpus_median_runtime(guardian):
    """Test that median runtime across corpus is <1s"""
    # Find available test documents
    available_docs = [Path(f) for f in SAMPLE_DOCUMENTS if Path(f).exists()]

    if len(available_docs) < 2:
        pytest.skip("Insufficient test documents")

    runtimes = []

    for doc in available_docs:
        start = time.perf_counter()
        result = guardian.validate_document(doc)
        end = time.perf_counter()

        if "error" not in result:
            runtime_ms = (end - start) * 1000
            runtimes.append(runtime_ms)

    if not runtimes:
        pytest.fail("No successful validations")

    median_runtime = median(runtimes)
    mean_runtime = mean(runtimes)
    max_runtime = max(runtimes)

    print(f"\n   Documents: {len(runtimes)}")
    print(f"   Median runtime: {median_runtime:.1f}ms")
    print(f"   Mean runtime: {mean_runtime:.1f}ms")
    print(f"   Max runtime: {max_runtime:.1f}ms")

    assert (
        median_runtime < 1000
    ), f"Median runtime {median_runtime:.1f}ms exceeds 1000ms"


def test_large_corpus_performance(guardian):
    """
    Test performance on larger sample (n≥10 if available)
    Acceptance: median < 1000ms
    """
    root = Path(".")

    # Find markdown files
    md_files = []
    for pattern in ["docs/*.md", "templates/*.md", "qc/*.md"]:
        md_files.extend(list(root.glob(pattern)))

    # Exclude large files
    md_files = [f for f in md_files if f.stat().st_size < 1_000_000]  # < 1MB
    md_files = md_files[:20]  # Cap at 20 for test speed

    if len(md_files) < 10:
        pytest.skip("Insufficient documents for large corpus test")

    runtimes = []

    for doc in md_files:
        start = time.perf_counter()
        try:
            result = guardian.validate_document(doc)
            end = time.perf_counter()

            if "error" not in result:
                runtime_ms = (end - start) * 1000
                runtimes.append(runtime_ms)
        except:
            pass

    if len(runtimes) < 10:
        pytest.skip(f"Only {len(runtimes)} successful validations")

    median_runtime = median(runtimes)

    print(f"\n   Large corpus test (n={len(runtimes)})")
    print(f"   Median runtime: {median_runtime:.1f}ms")

    assert (
        median_runtime < 1000
    ), f"Median {median_runtime:.1f}ms exceeds 1000ms on n={len(runtimes)} docs"


def test_no_performance_regression():
    """
    Test that performance hasn't regressed
    Compare with baseline if available
    """
    baseline_file = Path("qc/guardian_v4/benchmarks/v5_baseline_aggregate.json")

    if not baseline_file.exists():
        pytest.skip("No baseline for comparison")

    import json

    baseline = json.loads(baseline_file.read_text())

    # Run quick performance check
    guardian = GuardianV4()
    test_file = Path("README.md")

    if not test_file.exists():
        pytest.skip("README.md not found")

    runtimes = []
    for _ in range(5):
        start = time.perf_counter()
        guardian.validate_document(test_file)
        end = time.perf_counter()
        runtimes.append((end - start) * 1000)

    current_median = median(runtimes)
    baseline_median = baseline.get("median_runtime_ms", 1000)

    # Allow up to 20% regression (or +200ms, whichever is larger)
    tolerance = max(baseline_median * 0.20, 200)

    print(f"\n   Current median: {current_median:.1f}ms")
    print(f"   Baseline median: {baseline_median:.1f}ms")
    print(f"   Tolerance: {tolerance:.1f}ms")

    assert (
        current_median <= baseline_median + tolerance
    ), f"Performance regression: {current_median:.1f}ms vs baseline {baseline_median:.1f}ms"
