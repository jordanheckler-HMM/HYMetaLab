#!/usr/bin/env python3
"""
OriginChain v3 Integrator Test Suite
Tests for cross-system integration and JSON ingestion
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration_core import IntegrationCore


@pytest.fixture
def integration_core():
    """IntegrationCore fixture"""
    return IntegrationCore()


@pytest.fixture
def sample_guardian_data():
    """Sample Guardian JSON data"""
    return {
        "version": "v3",
        "safety_score": 0.85,
        "harm_score": 0.1,
        "risk_level": "low",
        "violence_score": 0.05,
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_truthlens_data():
    """Sample TruthLens JSON data"""
    return {
        "version": "v5",
        "truth_score": 0.92,
        "evidence_quality": 0.9,
        "logical_coherence": 0.95,
        "source_reliability": 0.88,
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_meaningforge_data():
    """Sample MeaningForge JSON data"""
    return {
        "version": "v5",
        "meaning_quotient": 0.88,
        "relevance": 0.85,
        "resonance": 0.9,
        "transformative_potential": 0.9,
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_originchain_data():
    """Sample OriginChain JSON data"""
    return {
        "version": "v2",
        "emergence_quotient": 0.78,
        "complexity": 0.75,
        "novelty": 0.8,
        "interconnectedness": 0.8,
        "metadata": {"source": "test"},
    }


class TestJSONIngestion:
    """Test JSON ingestion from each system"""

    def test_ingest_guardian(self, integration_core, sample_guardian_data):
        """Test Guardian JSON ingestion"""
        metrics = integration_core.ingest_guardian_json(sample_guardian_data)

        assert metrics.system_name == "Guardian"
        assert metrics.version == "v3"
        assert 0.0 <= metrics.primary_score <= 1.0
        assert (
            "harm_score" in metrics.components or "safety_score" in metrics.components
        )

    def test_ingest_truthlens(self, integration_core, sample_truthlens_data):
        """Test TruthLens JSON ingestion"""
        metrics = integration_core.ingest_truthlens_json(sample_truthlens_data)

        assert metrics.system_name == "TruthLens"
        assert metrics.version == "v5"
        assert 0.0 <= metrics.primary_score <= 1.0
        assert metrics.primary_score == 0.92

    def test_ingest_meaningforge(self, integration_core, sample_meaningforge_data):
        """Test MeaningForge JSON ingestion"""
        metrics = integration_core.ingest_meaningforge_json(sample_meaningforge_data)

        assert metrics.system_name == "MeaningForge"
        assert metrics.version == "v5"
        assert 0.0 <= metrics.primary_score <= 1.0
        assert metrics.primary_score == 0.88

    def test_ingest_originchain(self, integration_core, sample_originchain_data):
        """Test OriginChain JSON ingestion"""
        metrics = integration_core.ingest_originchain_json(sample_originchain_data)

        assert metrics.system_name == "OriginChain"
        assert metrics.version == "v2"
        assert 0.0 <= metrics.primary_score <= 1.0
        assert metrics.primary_score == 0.78

    def test_ingest_json_string(self, integration_core):
        """Test ingesting JSON from string"""
        json_str = '{"truth_score": 0.9, "version": "v5"}'
        metrics = integration_core.ingest_truthlens_json(json_str)

        assert metrics.primary_score == 0.9

    def test_ingest_json_file(self, integration_core, sample_truthlens_data):
        """Test ingesting JSON from file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_truthlens_data, f)
            temp_path = Path(f.name)

        try:
            metrics = integration_core.ingest_truthlens_json(temp_path)
            assert metrics.primary_score == 0.92
        finally:
            temp_path.unlink()


class TestIntegratedAnalysis:
    """Test integrated cross-system analysis"""

    def test_ingest_all_systems(
        self,
        integration_core,
        sample_guardian_data,
        sample_truthlens_data,
        sample_meaningforge_data,
        sample_originchain_data,
    ):
        """Test ingesting data from all systems"""
        analysis = integration_core.ingest_all(
            document_id="test_doc",
            text="Sample document text for testing...",
            guardian_data=sample_guardian_data,
            truthlens_data=sample_truthlens_data,
            meaningforge_data=sample_meaningforge_data,
            originchain_data=sample_originchain_data,
        )

        assert analysis.document_id == "test_doc"
        assert analysis.guardian is not None
        assert analysis.truthlens is not None
        assert analysis.meaningforge is not None
        assert analysis.originchain is not None

    def test_partial_system_ingestion(
        self, integration_core, sample_truthlens_data, sample_meaningforge_data
    ):
        """Test ingesting data from subset of systems"""
        analysis = integration_core.ingest_all(
            document_id="partial_doc",
            text="Partial system test...",
            truthlens_data=sample_truthlens_data,
            meaningforge_data=sample_meaningforge_data,
        )

        assert analysis.truthlens is not None
        assert analysis.meaningforge is not None
        assert analysis.guardian is None
        assert analysis.originchain is None

    def test_integrated_scoring(
        self,
        integration_core,
        sample_guardian_data,
        sample_truthlens_data,
        sample_meaningforge_data,
        sample_originchain_data,
    ):
        """Test integrated score computation"""
        analysis = integration_core.ingest_all(
            document_id="score_test",
            text="Test scoring...",
            guardian_data=sample_guardian_data,
            truthlens_data=sample_truthlens_data,
            meaningforge_data=sample_meaningforge_data,
            originchain_data=sample_originchain_data,
        )

        assert 0.0 <= analysis.composite_score <= 1.0
        assert 0.0 <= analysis.integrity_index <= 1.0
        assert 0.0 <= analysis.meaning_emergence_index <= 1.0
        assert 0.0 <= analysis.overall_quality <= 1.0

    def test_integrity_index(
        self, integration_core, sample_guardian_data, sample_truthlens_data
    ):
        """Test integrity index (Guardian + TruthLens)"""
        analysis = integration_core.ingest_all(
            document_id="integrity_test",
            text="Test integrity...",
            guardian_data=sample_guardian_data,
            truthlens_data=sample_truthlens_data,
        )

        # Integrity index should be average of Guardian and TruthLens
        expected = (
            analysis.guardian.primary_score + analysis.truthlens.primary_score
        ) / 2
        assert abs(analysis.integrity_index - expected) < 0.01

    def test_meaning_emergence_index(
        self, integration_core, sample_meaningforge_data, sample_originchain_data
    ):
        """Test meaning-emergence index (MeaningForge + OriginChain)"""
        analysis = integration_core.ingest_all(
            document_id="meaning_emergence_test",
            text="Test meaning emergence...",
            meaningforge_data=sample_meaningforge_data,
            originchain_data=sample_originchain_data,
        )

        # Meaning-emergence index should be average of MeaningForge and OriginChain
        expected = (
            analysis.meaningforge.primary_score + analysis.originchain.primary_score
        ) / 2
        assert abs(analysis.meaning_emergence_index - expected) < 0.01


class TestPerformance:
    """Test performance and sync time"""

    def test_sync_time_measured(self, integration_core, sample_guardian_data):
        """Test sync time is measured"""
        analysis = integration_core.ingest_all(
            document_id="time_test",
            text="Test sync time...",
            guardian_data=sample_guardian_data,
        )

        assert analysis.sync_time > 0
        assert analysis.sync_time < 1.0  # Should be well under 1 second

    def test_single_system_under_1s(self, integration_core, sample_truthlens_data):
        """Test single system ingestion is under 1 second"""
        start = time.time()

        for _ in range(100):
            integration_core.ingest_all(
                document_id=f"doc_{_}",
                text="Quick test...",
                truthlens_data=sample_truthlens_data,
            )

        elapsed = time.time() - start
        avg_time = elapsed / 100

        assert avg_time < 1.0, f"Average sync time {avg_time:.3f}s exceeds 1s"

    def test_all_systems_under_1s(
        self,
        integration_core,
        sample_guardian_data,
        sample_truthlens_data,
        sample_meaningforge_data,
        sample_originchain_data,
    ):
        """Test all-system ingestion is under 1 second"""
        analysis = integration_core.ingest_all(
            document_id="perf_test",
            text="Performance test...",
            guardian_data=sample_guardian_data,
            truthlens_data=sample_truthlens_data,
            meaningforge_data=sample_meaningforge_data,
            originchain_data=sample_originchain_data,
        )

        assert (
            analysis.sync_time < 1.0
        ), f"Sync time {analysis.sync_time:.3f}s exceeds 1s threshold"

    def test_batch_ingestion_performance(self, integration_core):
        """Test batch ingestion performance"""
        batch = []
        for i in range(50):
            batch.append(
                {
                    "document_id": f"batch_doc_{i}",
                    "text": f"Batch test document {i}...",
                    "truthlens": {"truth_score": 0.8 + i * 0.001},
                    "meaningforge": {"meaning_quotient": 0.75 + i * 0.001},
                }
            )

        start = time.time()
        results = integration_core.ingest_batch(batch)
        elapsed = time.time() - start

        assert len(results) == 50
        assert elapsed < 50.0  # Total batch should be reasonable

        # Check individual sync times
        for result in results:
            assert result.sync_time < 1.0


class TestBatchProcessing:
    """Test batch processing"""

    def test_batch_ingestion(self, integration_core):
        """Test batch ingestion"""
        batch = [
            {
                "document_id": "doc1",
                "text": "First document...",
                "truthlens": {"truth_score": 0.9},
            },
            {
                "document_id": "doc2",
                "text": "Second document...",
                "meaningforge": {"meaning_quotient": 0.85},
            },
        ]

        results = integration_core.ingest_batch(batch)

        assert len(results) == 2
        assert results[0].document_id == "doc1"
        assert results[1].document_id == "doc2"


class TestQueryAndExport:
    """Test query and export functionality"""

    def test_get_analysis(self, integration_core, sample_truthlens_data):
        """Test retrieving analysis by ID"""
        integration_core.ingest_all(
            document_id="retrieve_test",
            text="Test retrieval...",
            truthlens_data=sample_truthlens_data,
        )

        retrieved = integration_core.get_analysis("retrieve_test")

        assert retrieved is not None
        assert retrieved.document_id == "retrieve_test"

    def test_get_all_analyses(self, integration_core, sample_truthlens_data):
        """Test retrieving all analyses"""
        for i in range(3):
            integration_core.ingest_all(
                document_id=f"doc_{i}",
                text=f"Document {i}...",
                truthlens_data=sample_truthlens_data,
            )

        all_analyses = integration_core.get_all_analyses()

        assert len(all_analyses) == 3

    def test_export_analysis(self, integration_core, sample_truthlens_data):
        """Test exporting single analysis"""
        integration_core.ingest_all(
            document_id="export_test",
            text="Test export...",
            truthlens_data=sample_truthlens_data,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            integration_core.export_analysis("export_test", temp_path)
            assert temp_path.exists()

            # Verify exported data
            exported = json.loads(temp_path.read_text())
            assert exported["document_id"] == "export_test"
        finally:
            temp_path.unlink()

    def test_export_all(self, integration_core, sample_truthlens_data):
        """Test exporting all analyses"""
        for i in range(2):
            integration_core.ingest_all(
                document_id=f"doc_{i}",
                text=f"Document {i}...",
                truthlens_data=sample_truthlens_data,
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            integration_core.export_all(temp_path)
            assert temp_path.exists()

            # Verify exported data
            exported = json.loads(temp_path.read_text())
            assert exported["total_analyses"] == 2
            assert len(exported["analyses"]) == 2
        finally:
            temp_path.unlink()


class TestPerformanceStats:
    """Test performance statistics"""

    def test_performance_stats(self, integration_core, sample_truthlens_data):
        """Test performance statistics generation"""
        for i in range(10):
            integration_core.ingest_all(
                document_id=f"stats_doc_{i}",
                text=f"Stats test {i}...",
                truthlens_data=sample_truthlens_data,
            )

        stats = integration_core.get_performance_stats()

        assert "total_analyses" in stats
        assert stats["total_analyses"] == 10
        assert "mean_sync_time" in stats
        assert "max_sync_time" in stats
        assert "min_sync_time" in stats
        assert "under_1s" in stats
        assert "under_1s_rate" in stats

        # All should be under 1s
        assert stats["under_1s"] == 10
        assert stats["under_1s_rate"] == 1.0


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_components(self, integration_core):
        """Test handling of empty component dictionaries"""
        metrics = integration_core.ingest_truthlens_json({"version": "v5"})

        assert metrics.system_name == "TruthLens"
        assert 0.0 <= metrics.primary_score <= 1.0

    def test_missing_version(self, integration_core):
        """Test handling of missing version field"""
        metrics = integration_core.ingest_guardian_json({"safety_score": 0.8})

        assert metrics.version is not None


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("OriginChain v3 Integrator Test Suite Summary")
    print("=" * 70)
    print("✅ JSON Ingestion: Guardian, TruthLens, MeaningForge, OriginChain")
    print("✅ Integrated Analysis: Cross-system scoring and indices")
    print("✅ Performance: <1s sync time verification")
    print("✅ Batch Processing: Multi-document ingestion")
    print("✅ Query & Export: Retrieval and JSON export")
    print("✅ Performance Stats: Metrics and monitoring")
    print("✅ Edge Cases: Empty data, missing fields")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
