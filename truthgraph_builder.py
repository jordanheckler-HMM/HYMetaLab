#!/usr/bin/env python3
"""
TruthLens v3 - Evidence Graph Builder
Links claims to citations/URLs/DOIs for evidence tracking

v3 Evidence Graph: Claim→Evidence connectivity analysis
"""
import json
import re
from pathlib import Path


class TruthGraphBuilder:
    """
    Build evidence graphs linking claims to supporting citations
    Calculate evidence density and coverage metrics
    """

    def __init__(self):
        # Citation patterns (from truthlens_core, expanded)
        self.citation_patterns = {
            "author_year": r"\([A-Z][a-z]+(?:\s+et al\.)?\s*(?:&|,)?\s*(?:[A-Z][a-z]+\s*)?\d{4}\)",
            "doi": r"doi\.org/[\w\./\-]+|doi:\s*[\w\./\-]+",
            "url": r"https?://[^\s\)]+",
            "bracket_ref": r"\[\d+\]",
            "arxiv": r"arxiv\.org/[\w\./\-]+",
            "figure_ref": r"Figure\s+\d+",
            "table_ref": r"Table\s+\d+",
            "year_only": r"\b(19|20)\d{2}\b",
        }

        # Claim indicators (signals that a statement is making a claim)
        self.claim_indicators = [
            # Assertions
            r"\b(shows?|demonstrates?|proves?|indicates?|suggests?|reveals?)\s+that\b",
            r"\b(found|discovered|observed|measured|detected)\b",
            r"\b(is|are|was|were)\s+(significant|correlated|associated|linked)\b",
            # Causal claims
            r"\b(causes?|leads?\s+to|results?\s+in|produces?|generates?)\b",
            r"\b(increases?|decreases?|enhances?|reduces?|improves?|weakens?)\b",
            # Statistical claims
            r"\b(mean|median|average|correlation|p\s*[<>=])\b",
            r"\b(n\s*=|\d+%|CI\s*\[)",
            # Findings
            r"\b(finding|result|evidence|data|analysis)\s+(shows?|indicates?|suggests?)\b",
        ]

        # Compile patterns for efficiency
        self.claim_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.claim_indicators
        ]
        self.citation_compiled = {
            k: re.compile(v, re.IGNORECASE) for k, v in self.citation_patterns.items()
        }

    def extract_claims(self, text: str) -> list[dict]:
        """
        Extract claim statements from text

        Args:
            text: Document text

        Returns:
            List of claim dictionaries with positions and text
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        claims = []

        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Check if sentence contains claim indicators
            is_claim = False
            for pattern in self.claim_patterns:
                if pattern.search(sentence):
                    is_claim = True
                    break

            # Also check for basic claim structure (more permissive)
            if not is_claim:
                # Check for subject + verb + object patterns
                if any(
                    word in sentence.lower()
                    for word in [
                        "shows",
                        "indicates",
                        "suggests",
                        "demonstrates",
                        "reveals",
                    ]
                ):
                    is_claim = True
                elif any(
                    word in sentence.lower()
                    for word in ["study", "results", "data", "findings", "evidence"]
                ):
                    is_claim = True

            if is_claim:
                claims.append(
                    {
                        "id": f"claim_{sent_idx}",
                        "text": sentence[:200],  # Truncate long claims
                        "sentence_index": sent_idx,
                        "length": len(sentence),
                        "has_evidence": False,  # Will be updated later
                        "evidence_refs": [],
                    }
                )

        return claims

    def extract_citations(self, text: str) -> list[dict]:
        """
        Extract all citations/URLs/DOIs from text

        Args:
            text: Document text

        Returns:
            List of citation dictionaries with type and content
        """
        citations = []
        citation_id = 0

        for cit_type, pattern in self.citation_compiled.items():
            matches = pattern.finditer(text)

            for match in matches:
                citation_text = match.group(0)

                # Deduplicate by content
                if any(c["text"] == citation_text for c in citations):
                    continue

                citations.append(
                    {
                        "id": f"cit_{citation_id}",
                        "type": cit_type,
                        "text": citation_text,
                        "position": match.start(),
                    }
                )
                citation_id += 1

        # Sort by position
        citations.sort(key=lambda x: x["position"])

        return citations

    def link_claims_to_evidence(
        self,
        claims: list[dict],
        citations: list[dict],
        text: str,
        proximity_window: int = 500,
    ) -> dict:
        """
        Link claims to nearby citations (evidence)

        Args:
            claims: List of extracted claims
            citations: List of extracted citations
            text: Original document text
            proximity_window: Character distance for evidence linking

        Returns:
            Evidence graph with links
        """
        # Find position of each claim in text
        for claim in claims:
            # Find claim text in document
            claim_pos = text.find(
                claim["text"][:50]
            )  # Use first 50 chars to find position

            if claim_pos == -1:
                continue

            # Find citations within proximity window
            nearby_citations = []

            for citation in citations:
                distance = abs(citation["position"] - claim_pos)

                if distance <= proximity_window:
                    nearby_citations.append(
                        {
                            "citation_id": citation["id"],
                            "citation_text": citation["text"],
                            "citation_type": citation["type"],
                            "distance": distance,
                        }
                    )

            if nearby_citations:
                claim["has_evidence"] = True
                claim["evidence_refs"] = nearby_citations

        # Build graph structure
        evidence_graph = {
            "nodes": {"claims": claims, "citations": citations},
            "edges": [],
        }

        # Create edges (claim → citation links)
        for claim in claims:
            for evidence in claim.get("evidence_refs", []):
                evidence_graph["edges"].append(
                    {
                        "from": claim["id"],
                        "to": evidence["citation_id"],
                        "distance": evidence["distance"],
                        "type": "supports",
                    }
                )

        return evidence_graph

    def compute_evidence_density(self, evidence_graph: dict) -> dict:
        """
        Compute evidence density metrics

        Formula: evidence_density = claims_with_evidence / total_claims

        Args:
            evidence_graph: Evidence graph structure

        Returns:
            Metrics dictionary
        """
        claims = evidence_graph["nodes"]["claims"]
        citations = evidence_graph["nodes"]["citations"]
        edges = evidence_graph["edges"]

        total_claims = len(claims)
        claims_with_evidence = sum(1 for c in claims if c["has_evidence"])
        total_citations = len(citations)
        total_links = len(edges)

        # Evidence density (primary metric)
        evidence_density = (
            claims_with_evidence / total_claims if total_claims > 0 else 0.0
        )

        # Citation diversity (how many unique citations are used)
        unique_citations_used = len(set(edge["to"] for edge in edges))
        citation_utilization = (
            unique_citations_used / total_citations if total_citations > 0 else 0.0
        )

        # Average citations per claim (for claims with evidence)
        avg_citations_per_claim = (
            total_links / claims_with_evidence if claims_with_evidence > 0 else 0.0
        )

        # Coverage ratio (claims covered / citations available)
        coverage_ratio = claims_with_evidence / max(1, total_citations)

        return {
            "evidence_density": evidence_density,
            "total_claims": total_claims,
            "claims_with_evidence": claims_with_evidence,
            "claims_without_evidence": total_claims - claims_with_evidence,
            "total_citations": total_citations,
            "unique_citations_used": unique_citations_used,
            "citation_utilization": citation_utilization,
            "total_links": total_links,
            "avg_citations_per_claim": avg_citations_per_claim,
            "coverage_ratio": coverage_ratio,
            "passes_threshold": evidence_density >= 0.7,
        }

    def build_graph(self, text: str, proximity_window: int = 500) -> tuple[dict, dict]:
        """
        Build complete evidence graph from text

        Args:
            text: Document text
            proximity_window: Character distance for linking

        Returns:
            Tuple of (evidence_graph, metrics)
        """
        # Extract claims and citations
        claims = self.extract_claims(text)
        citations = self.extract_citations(text)

        # Link claims to evidence
        evidence_graph = self.link_claims_to_evidence(
            claims, citations, text, proximity_window
        )

        # Compute metrics
        metrics = self.compute_evidence_density(evidence_graph)

        return evidence_graph, metrics

    def save_graph(self, evidence_graph: dict, metrics: dict, output_path: Path):
        """
        Save evidence graph to JSON file

        Args:
            evidence_graph: Graph structure
            metrics: Computed metrics
            output_path: Output file path
        """
        output_data = {
            "evidence_graph": evidence_graph,
            "metrics": metrics,
            "version": "3.0-evidence-graph",
        }

        output_path.write_text(json.dumps(output_data, indent=2))


def main():
    """CLI for TruthGraph builder"""
    import argparse

    parser = argparse.ArgumentParser(description="TruthLens v3 Evidence Graph Builder")
    parser.add_argument("command", choices=["build", "test"], help="Command to execute")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument(
        "--output", type=str, default="evidence_graph.json", help="Output file path"
    )
    parser.add_argument(
        "--proximity",
        type=int,
        default=500,
        help="Proximity window for claim-evidence linking (default: 500)",
    )

    args = parser.parse_args()

    builder = TruthGraphBuilder()

    if args.command == "build":
        if not args.file:
            print("❌ Error: --file required")
            return

        file_path = Path(args.file)

        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        print(f"🕸️  Building evidence graph: {file_path.name}")

        # Build graph
        evidence_graph, metrics = builder.build_graph(text, args.proximity)

        print("\n✅ Graph complete:")
        print(f"   Claims: {metrics['total_claims']}")
        print(f"   Citations: {metrics['total_citations']}")
        print(f"   Links: {metrics['total_links']}")
        print(f"   Evidence density: {metrics['evidence_density']:.3f}")
        print(
            f"   Coverage: {metrics['claims_with_evidence']}/{metrics['total_claims']} claims"
        )
        print(
            f"   Passes threshold (≥0.7): {'✅' if metrics['passes_threshold'] else '❌'}"
        )

        # Save
        output_path = Path(args.output)
        builder.save_graph(evidence_graph, metrics, output_path)
        print(f"\n✅ Saved: {output_path}")

    elif args.command == "test":
        print("🧪 Running TruthGraph builder tests...")

        # Test 1: Claim extraction
        test_text = """
        The study shows that openness increases cooperation (Smith et al. 2020).
        Results indicate a correlation of r=0.65 (p<0.001).
        We found significant effects as demonstrated in Figure 1.
        Previous work suggests similar patterns (Jones 2019, doi.org/10.1234/abc).
        """

        claims = builder.extract_claims(test_text)
        print(f"\nTest 1 (Claim extraction): Found {len(claims)} claims")
        for claim in claims[:3]:
            print(f"   • {claim['text'][:60]}...")

        # Test 2: Citation extraction
        citations = builder.extract_citations(test_text)
        print(f"\nTest 2 (Citation extraction): Found {len(citations)} citations")
        for cit in citations:
            print(f"   • [{cit['type']}] {cit['text']}")

        # Test 3: Graph building
        evidence_graph, metrics = builder.build_graph(test_text)
        print("\nTest 3 (Graph building):")
        print(f"   Claims: {metrics['total_claims']}")
        print(f"   Citations: {metrics['total_citations']}")
        print(f"   Links: {metrics['total_links']}")
        print(f"   Evidence density: {metrics['evidence_density']:.3f}")

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
