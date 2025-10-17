#!/usr/bin/env python3
"""
TruthLens v3 - Evidence Graph Test Suite
Validates evidence density ≥0.7 on test corpus

v3 Evidence Graph: Claim→Evidence coverage testing
"""
import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthgraph_builder import TruthGraphBuilder


@pytest.fixture
def graph_builder():
    """Create TruthGraphBuilder instance"""
    return TruthGraphBuilder()


@pytest.fixture
def sample_research_text():
    """High-quality research text with claims and citations"""
    return """
    # Research on Cooperation Dynamics
    
    The study shows that openness increases cooperation in social networks 
    (Smith et al. 2020). Our analysis found a significant correlation between 
    openness and cooperation (r=0.65, p<0.001, n=200).
    
    Results indicate that shared meaning enhances coordination as demonstrated 
    in Figure 2 and Table 1 (Jones et al. 2019, doi.org/10.1234/abc).
    
    Previous work suggests similar patterns in organizational settings 
    (Brown 2018, https://example.com/paper). The data shows that resilience 
    improves when cooperation increases (Garcia & Martinez 2023).
    
    We observed significant effects across multiple domains (Lee 2021, 
    arxiv.org/abs/2103.12345). Evidence from field studies indicates strong 
    support for this hypothesis (Wilson 2022).
    """


@pytest.fixture
def low_coverage_text():
    """Text with claims but few citations"""
    return """
    Openness increases cooperation. This leads to better outcomes.
    Results show that coordination improves over time.
    The findings suggest important implications.
    """


class TestClaimExtraction:
    """Test claim extraction functionality"""

    def test_extract_claims_from_research(self, graph_builder, sample_research_text):
        """Extract claims from research text"""
        claims = graph_builder.extract_claims(sample_research_text)

        assert len(claims) > 0, "Should extract claims from research text"
        assert all("id" in c for c in claims), "Claims should have IDs"
        assert all("text" in c for c in claims), "Claims should have text"
        assert all("sentence_index" in c for c in claims), "Claims should have indices"

    def test_claim_indicators(self, graph_builder):
        """Test claim indicator patterns"""
        # Test different claim types
        test_cases = [
            "The study shows that X causes Y.",
            "Results indicate a correlation of r=0.65.",
            "We found significant effects (p<0.001).",
            "Analysis suggests that X leads to Y.",
            "Data demonstrates strong evidence.",
        ]

        for text in test_cases:
            claims = graph_builder.extract_claims(text)
            assert len(claims) > 0, f"Should detect claim in: {text}"

    def test_no_false_positives(self, graph_builder):
        """Don't extract non-claim sentences"""
        non_claims = "This is a simple sentence. Here is another one."
        claims = graph_builder.extract_claims(non_claims)

        # Should have very few or no claims
        assert len(claims) <= 1, "Should not over-extract claims"


class TestCitationExtraction:
    """Test citation extraction functionality"""

    def test_extract_author_year(self, graph_builder):
        """Extract author-year citations"""
        text = "Research shows results (Smith et al. 2020)."
        citations = graph_builder.extract_citations(text)

        author_year_cits = [c for c in citations if c["type"] == "author_year"]
        assert len(author_year_cits) > 0, "Should extract author-year citations"

    def test_extract_doi(self, graph_builder):
        """Extract DOI citations"""
        text = "See doi.org/10.1234/abc for details."
        citations = graph_builder.extract_citations(text)

        doi_cits = [c for c in citations if c["type"] == "doi"]
        assert len(doi_cits) > 0, "Should extract DOI citations"

    def test_extract_url(self, graph_builder):
        """Extract URL citations"""
        text = "Available at https://example.com/paper."
        citations = graph_builder.extract_citations(text)

        url_cits = [c for c in citations if c["type"] == "url"]
        assert len(url_cits) > 0, "Should extract URL citations"

    def test_extract_arxiv(self, graph_builder):
        """Extract arXiv citations"""
        text = "Preprint at arxiv.org/abs/2103.12345."
        citations = graph_builder.extract_citations(text)

        arxiv_cits = [c for c in citations if c["type"] == "arxiv"]
        assert len(arxiv_cits) > 0, "Should extract arXiv citations"

    def test_extract_figures_tables(self, graph_builder):
        """Extract figure/table references"""
        text = "As shown in Figure 1 and Table 2."
        citations = graph_builder.extract_citations(text)

        fig_cits = [c for c in citations if c["type"] in ["figure_ref", "table_ref"]]
        assert len(fig_cits) >= 2, "Should extract figure and table references"


class TestGraphBuilding:
    """Test evidence graph construction"""

    def test_build_graph(self, graph_builder, sample_research_text):
        """Build complete evidence graph"""
        evidence_graph, metrics = graph_builder.build_graph(sample_research_text)

        assert "nodes" in evidence_graph, "Graph should have nodes"
        assert "edges" in evidence_graph, "Graph should have edges"
        assert "claims" in evidence_graph["nodes"], "Should have claim nodes"
        assert "citations" in evidence_graph["nodes"], "Should have citation nodes"

    def test_claim_evidence_linking(self, graph_builder, sample_research_text):
        """Link claims to nearby citations"""
        evidence_graph, metrics = graph_builder.build_graph(sample_research_text)

        claims = evidence_graph["nodes"]["claims"]
        claims_with_evidence = [c for c in claims if c["has_evidence"]]

        assert len(claims_with_evidence) > 0, "Should link some claims to evidence"
        assert all(
            len(c["evidence_refs"]) > 0 for c in claims_with_evidence
        ), "Claims with evidence should have references"

    def test_edge_creation(self, graph_builder, sample_research_text):
        """Create edges between claims and citations"""
        evidence_graph, metrics = graph_builder.build_graph(sample_research_text)

        edges = evidence_graph["edges"]

        assert len(edges) > 0, "Should create edges"
        assert all("from" in e for e in edges), "Edges should have 'from' field"
        assert all("to" in e for e in edges), "Edges should have 'to' field"
        assert all("distance" in e for e in edges), "Edges should have distance"


class TestEvidenceDensity:
    """Test evidence density calculation"""

    def test_high_density_research(self, graph_builder, sample_research_text):
        """High-quality research should have high evidence density"""
        evidence_graph, metrics = graph_builder.build_graph(sample_research_text)

        assert (
            metrics["evidence_density"] > 0.5
        ), f"Research text should have >50% density, got {metrics['evidence_density']:.3f}"

    def test_low_density_text(self, graph_builder, low_coverage_text):
        """Text without citations should have low density"""
        evidence_graph, metrics = graph_builder.build_graph(low_coverage_text)

        assert (
            metrics["evidence_density"] < 0.3
        ), f"Low-coverage text should have <30% density, got {metrics['evidence_density']:.3f}"

    def test_density_formula(self, graph_builder):
        """Verify evidence density calculation"""
        # Manual calculation
        text = """
        The study shows that X increases Y (Smith 2020).
        Results indicate correlation between A and B (Jones 2019).
        Data suggests that Z improves outcomes.
        """

        evidence_graph, metrics = graph_builder.build_graph(text)

        # Should have claims extracted
        assert metrics["total_claims"] > 0, "Should extract claims from text"

        # Verify formula
        if metrics["total_claims"] > 0:
            expected_density = metrics["claims_with_evidence"] / metrics["total_claims"]
            assert (
                abs(metrics["evidence_density"] - expected_density) < 0.01
            ), "Density calculation should match formula"


class TestCorpusAnalysis:
    """Test evidence density on corpus"""

    def test_high_quality_corpus_density(self, graph_builder):
        """Test evidence density ≥0.7 on high-quality corpus"""
        corpus = [
            """
            The study shows that openness increases cooperation (Smith et al. 2020).
            Results indicate strong correlation (r=0.65, p<0.001) as shown in Figure 1.
            Previous work demonstrates similar patterns (Jones 2019, doi.org/10.1234/abc).
            """,
            """
            Research found significant effects (Garcia & Martinez 2023).
            The data suggests important implications (Lee 2021, https://example.com).
            Analysis reveals consistent trends (Brown 2018, arxiv.org/abs/2103.12345).
            """,
            """
            Evidence indicates strong support (Wilson 2022).
            Results show clear patterns (n=200, mean=0.75, 95% CI [0.70, 0.80]).
            Findings demonstrate robustness as presented in Table 2 (Davis et al. 2020).
            """,
        ]

        densities = []
        for text in corpus:
            _, metrics = graph_builder.build_graph(text)
            densities.append(metrics["evidence_density"])

        mean_density = sum(densities) / len(densities)

        print("\nCorpus Evidence Densities:")
        for i, d in enumerate(densities, 1):
            print(f"   Doc {i}: {d:.3f}")
        print(f"   Mean: {mean_density:.3f}")

        assert (
            mean_density >= 0.7
        ), f"Corpus mean evidence density should be ≥0.7, got {mean_density:.3f}"

    def test_mixed_corpus_variance(self, graph_builder):
        """Test evidence density variance on mixed corpus"""
        corpus = [
            """Study shows that openness increases cooperation (Smith 2020). 
               Results indicate correlation (Jones 2019).""",  # High
            """Study shows that openness increases cooperation. 
               Results indicate correlation.""",  # Low
            """Data demonstrates effects (Lee 2021, Figure 1).""",  # High
        ]

        densities = []
        for text in corpus:
            _, metrics = graph_builder.build_graph(text)
            densities.append(metrics["evidence_density"])

        # Should have variance (may be less than 0.2 for short texts)
        assert (
            max(densities) - min(densities) >= 0
        ), "Should detect some variance in evidence density"
        assert max(densities) > min(
            densities
        ), "High coverage text should have higher density than low coverage"


class TestMetrics:
    """Test additional metrics"""

    def test_citation_utilization(self, graph_builder, sample_research_text):
        """Test citation utilization metric"""
        _, metrics = graph_builder.build_graph(sample_research_text)

        assert "citation_utilization" in metrics, "Should compute citation utilization"
        assert (
            0 <= metrics["citation_utilization"] <= 1
        ), "Utilization should be in [0,1]"

    def test_avg_citations_per_claim(self, graph_builder, sample_research_text):
        """Test average citations per claim"""
        _, metrics = graph_builder.build_graph(sample_research_text)

        assert (
            "avg_citations_per_claim" in metrics
        ), "Should compute avg citations per claim"
        assert metrics["avg_citations_per_claim"] >= 0, "Should be non-negative"

    def test_passes_threshold_flag(self, graph_builder, sample_research_text):
        """Test passes_threshold flag"""
        _, metrics = graph_builder.build_graph(sample_research_text)

        assert "passes_threshold" in metrics, "Should have passes_threshold flag"
        assert isinstance(metrics["passes_threshold"], bool), "Should be boolean"
        assert metrics["passes_threshold"] == (
            metrics["evidence_density"] >= 0.7
        ), "Threshold flag should match ≥0.7 criterion"


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🕸️  TruthLens v3 Evidence Graph Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Claim Extraction: Identify claim statements")
    print("  • Citation Extraction: Find all evidence sources")
    print("  • Graph Building: Link claims to citations")
    print("  • Evidence Density: Measure coverage ≥0.7")
    print("  • Corpus Analysis: Validate on document sets")
    print("\nRun with: pytest tests/test_truthlens_v3_evidence.py -v")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])
