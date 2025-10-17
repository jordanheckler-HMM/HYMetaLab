#!/usr/bin/env python3
"""
Guardian v5 Determinism Tests
Ensures <±2% variance across multiple runs on same document
"""
import sys
from pathlib import Path
from statistics import mean, stdev

import pytest

# Add Guardian v4 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "qc" / "guardian_v4"))

from guardian_v4 import GuardianV4

# Test fixtures
TEST_DOCUMENTS = [
    "README.md",
    "ETHICS.md",
    "templates/RESEARCH_DISCLAIMER.md",
    "templates/LANGUAGE_POLICY.md",
]


@pytest.fixture
def guardian():
    """Create Guardian v4 instance"""
    return GuardianV4()


def test_determinism_single_document(guardian):
    """
    Test that same file produces same score across 5 runs
    Acceptance: stdev < 0.02 (2% of typical 70/100 score)
    """
    test_file = Path("README.md")
    if not test_file.exists():
        pytest.skip("README.md not found")

    scores = []

    for run in range(5):
        result = guardian.validate_document(test_file)
        if "error" not in result:
            scores.append(result["guardian_alignment_score"])

    assert len(scores) == 5, "Some runs failed"

    score_stdev = stdev(scores) if len(scores) > 1 else 0.0
    score_mean = mean(scores)
    variance_pct = (score_stdev / score_mean * 100) if score_mean > 0 else 0.0

    print(f"\n   File: {test_file.name}")
    print(f"   Mean score: {score_mean:.2f}")
    print(f"   Stdev: {score_stdev:.4f}")
    print(f"   Variance: {variance_pct:.3f}%")

    assert (
        score_stdev < 0.02
    ), f"Score variance {score_stdev:.4f} exceeds 0.02 threshold"
    assert variance_pct < 2.0, f"Variance {variance_pct:.2f}% exceeds 2% threshold"


@pytest.mark.parametrize("test_file", TEST_DOCUMENTS)
def test_determinism_multiple_documents(guardian, test_file):
    """
    Test determinism across multiple documents
    Each document should have <2% variance across runs
    """
    file_path = Path(test_file)
    if not file_path.exists():
        pytest.skip(f"{test_file} not found")

    scores = []

    for run in range(5):
        result = guardian.validate_document(file_path)
        if "error" not in result:
            scores.append(result["guardian_alignment_score"])

    if len(scores) < 5:
        pytest.fail(f"Only {len(scores)}/5 runs succeeded")

    score_stdev = stdev(scores)
    score_mean = mean(scores)
    variance_pct = (score_stdev / score_mean * 100) if score_mean > 0 else 0.0

    assert (
        variance_pct < 2.0
    ), f"{test_file}: Variance {variance_pct:.2f}% exceeds 2% threshold"


def test_component_determinism(guardian):
    """Test that individual components are deterministic"""
    test_file = Path("README.md")
    if not test_file.exists():
        pytest.skip("README.md not found")

    components = {
        "objectivity": [],
        "transparency": [],
        "language_safety": [],
        "sentiment": [],
    }

    for run in range(5):
        result = guardian.validate_document(test_file)
        if "error" not in result:
            components["objectivity"].append(result["metrics"]["objectivity_score"])
            components["transparency"].append(
                result["metrics"]["transparency_index_v2"]
            )
            components["language_safety"].append(
                result["metrics"]["language_safety_score"]
            )
            components["sentiment"].append(result["metrics"]["sentiment_neutrality"])

    # Check each component
    for component_name, values in components.items():
        if len(values) < 5:
            continue

        comp_stdev = stdev(values)
        comp_mean = mean(values)
        variance_pct = (comp_stdev / comp_mean * 100) if comp_mean > 0 else 0.0

        print(
            f"\n   {component_name}: mean={comp_mean:.3f}, stdev={comp_stdev:.5f}, var={variance_pct:.3f}%"
        )

        # Stricter threshold for components
        assert (
            comp_stdev < 0.01
        ), f"{component_name}: stdev {comp_stdev:.5f} exceeds 0.01"


def test_reproducibility_rate():
    """
    Test that >99% of documents produce reproducible scores
    This is the acceptance criterion for v5 stabilization
    """
    guardian = GuardianV4()

    # Sample subset of documents
    test_docs = [Path(f) for f in TEST_DOCUMENTS if Path(f).exists()]

    if len(test_docs) < 2:
        pytest.skip("Insufficient test documents")

    deterministic_count = 0

    for doc in test_docs:
        scores = []
        for run in range(5):
            result = guardian.validate_document(doc)
            if "error" not in result:
                scores.append(result["guardian_alignment_score"])

        if len(scores) == 5:
            score_stdev = stdev(scores)
            if score_stdev < 0.02:  # Deterministic
                deterministic_count += 1

    reproducibility_rate = (deterministic_count / len(test_docs)) * 100

    print(f"\n   Deterministic documents: {deterministic_count}/{len(test_docs)}")
    print(f"   Reproducibility rate: {reproducibility_rate:.1f}%")

    assert (
        reproducibility_rate >= 99.0
    ), f"Reproducibility {reproducibility_rate:.1f}% < 99%"
