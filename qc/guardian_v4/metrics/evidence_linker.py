#!/usr/bin/env python3
"""
Guardian v6 - Evidence Linker
Maps claims to nearby citations/evidence within K sentences

v6 Context Engine: Claim-evidence matching and coverage analysis
"""
import re
from pathlib import Path


class EvidenceLinker:
    """
    Links research claims to supporting evidence
    Computes evidence coverage percentage
    """

    def __init__(self, k_sentences: int = 3):
        """
        Initialize evidence linker

        Args:
            k_sentences: Window size for evidence search (default: 3)
        """
        self.k = k_sentences

        # Evidence patterns
        self.citation_patterns = [
            r"doi\.org/[\w\./\-]+",  # DOI
            r"https?://[^\s]+",  # URLs
            r"\([A-Z][a-z]+\s+(?:et al\.\s+)?\d{4}\)",  # Author (Year)
            r"\[\d+\]",  # Citation numbers
            r"Figure\s+\d+",  # Figure references
            r"Table\s+\d+",  # Table references
        ]

        # Data reference patterns
        self.data_patterns = [
            r"discovery_results/[\w/_]+",
            r"data/[\w/_]+",
            r"results/[\w/_]+",
            r"dataset:?\s+\w+",
            r"study\s+[\w\d]+",
        ]

        # Claim indicators (sentences that make assertions)
        self.claim_indicators = {
            "observed",
            "measured",
            "found",
            "showed",
            "demonstrated",
            "suggests",
            "indicates",
            "implies",
            "supports",
            "increased",
            "decreased",
            "correlated",
            "associated",
            "effect",
            "difference",
            "relationship",
            "pattern",
        }

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r"[.!?]+\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        return sentences

    def detect_evidence(self, sentence: str) -> list[dict]:
        """
        Detect evidence markers in a sentence
        Returns list of evidence items found
        """
        evidence_items = []

        # Check for citations
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                evidence_items.append(
                    {
                        "type": "citation",
                        "text": match.group(0),
                        "position": match.start(),
                        "pattern": pattern,
                    }
                )

        # Check for data references
        for pattern in self.data_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                evidence_items.append(
                    {
                        "type": "data_reference",
                        "text": match.group(0),
                        "position": match.start(),
                        "pattern": pattern,
                    }
                )

        # Check for statistical evidence
        stats_patterns = [
            r"p\s*[<>=]\s*0\.\d+",  # p-values
            r"r\s*=\s*0\.\d+",  # Correlations
            r"95%\s+CI\s*[\[\(]",  # Confidence intervals
            r"mean\s*=\s*[\d.]+",  # Means
            r"n\s*=\s*\d+",  # Sample sizes
        ]

        for pattern in stats_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                evidence_items.append(
                    {
                        "type": "statistical_evidence",
                        "text": match.group(0),
                        "position": match.start(),
                        "pattern": pattern,
                    }
                )

        return evidence_items

    def is_claim_sentence(self, sentence: str) -> bool:
        """
        Determine if a sentence makes a claim (assertion)
        """
        sentence_lower = sentence.lower()

        # Check for claim indicators
        has_claim_indicator = any(
            indicator in sentence_lower for indicator in self.claim_indicators
        )

        # Claims typically have main verbs and substantive content
        has_verb = bool(
            re.search(
                r"\b(is|are|was|were|shows?|indicates?|suggests?)\b", sentence_lower
            )
        )
        has_substance = len(sentence.split()) >= 8  # At least 8 words

        return has_claim_indicator or (has_verb and has_substance)

    def link_claims_to_evidence(self, text: str) -> dict:
        """
        Link claims to nearby evidence within K sentences
        Returns claim-evidence pairs and coverage statistics
        """
        sentences = self.split_into_sentences(text)

        if not sentences:
            return {
                "claim_count": 0,
                "evidence_coverage": 0.0,
                "claim_evidence_pairs": [],
            }

        claim_evidence_pairs = []
        claim_count = 0
        supported_claim_count = 0

        for i, sentence in enumerate(sentences):
            # Check if this is a claim
            if not self.is_claim_sentence(sentence):
                continue

            claim_count += 1

            # Search for evidence in window: [i-k, i+k]
            window_start = max(0, i - self.k)
            window_end = min(len(sentences), i + self.k + 1)

            evidence_found = []

            for j in range(window_start, window_end):
                evidence_items = self.detect_evidence(sentences[j])
                if evidence_items:
                    evidence_found.extend(
                        [
                            {**item, "sentence_index": j, "distance": abs(j - i)}
                            for item in evidence_items
                        ]
                    )

            # Record claim-evidence pair
            claim_evidence_pairs.append(
                {
                    "claim_index": i,
                    "claim_text": sentence[:200],  # Truncate for readability
                    "evidence_count": len(evidence_found),
                    "evidence_items": evidence_found,
                    "is_supported": len(evidence_found) > 0,
                }
            )

            if evidence_found:
                supported_claim_count += 1

        # Compute coverage
        evidence_coverage = (
            supported_claim_count / claim_count if claim_count > 0 else 0.0
        )

        return {
            "claim_count": claim_count,
            "supported_claim_count": supported_claim_count,
            "unsupported_claim_count": claim_count - supported_claim_count,
            "evidence_coverage": evidence_coverage,
            "claim_evidence_pairs": claim_evidence_pairs,
            "window_size": self.k,
            "passes_threshold": evidence_coverage
            >= 0.60,  # 60% claims should have nearby evidence
        }

    def analyze_file(self, file_path: Path) -> dict:
        """Analyze a file for claim-evidence links"""
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        result = self.link_claims_to_evidence(text)
        result["file"] = str(file_path)

        return result


def main():
    """CLI interface for evidence linker"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Guardian v6 Evidence Linker")
    parser.add_argument(
        "command", choices=["analyze", "test"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument(
        "--k", type=int, default=3, help="Sentence window size for evidence search"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/evidence_links.json",
        help="Output file path",
    )

    args = parser.parse_args()

    linker = EvidenceLinker(k_sentences=args.k)

    if args.command == "analyze":
        if not args.file:
            print("❌ Error: --file required")
            return

        print(f"🔗 Analyzing claim-evidence links in {args.file}...")
        result = linker.analyze_file(Path(args.file))

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print("\n✅ Analysis complete")
        print(f"   Claims detected: {result['claim_count']}")
        print(f"   Supported claims: {result['supported_claim_count']}")
        print(f"   Unsupported claims: {result['unsupported_claim_count']}")
        print(f"   Evidence coverage: {result['evidence_coverage']*100:.1f}%")
        print(
            f"   Passes threshold (≥60%): {'✅' if result['passes_threshold'] else '❌'}"
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\n   Output: {output_path}")

    elif args.command == "test":
        print("🧪 Running evidence linker tests...")

        # Test document with claims and evidence
        test_text = """
        We observed increased coherence in cooperative conditions.
        The mean CCI was 0.75 (95% CI [0.71, 0.79], n=200).
        These results are consistent with Tomasello et al. (2005).
        
        The data suggests that shared meaning enhances resilience.
        However, further validation is required.
        
        We measured hazard rates across four domains (see Figure 2).
        Each domain showed similar patterns (p<0.001).
        Data available at discovery_results/phase35c/.
        """

        result = linker.link_claims_to_evidence(test_text)

        print(f"\n   Claims: {result['claim_count']}")
        print(f"   Supported: {result['supported_claim_count']}")
        print(f"   Coverage: {result['evidence_coverage']*100:.1f}%")

        # Show claim-evidence pairs
        print("\n   Claim-Evidence Pairs:")
        for pair in result["claim_evidence_pairs"][:5]:  # Show first 5
            print(f"     Claim: {pair['claim_text'][:80]}...")
            print(f"     Evidence: {pair['evidence_count']} items")
            if pair["evidence_items"]:
                for ev in pair["evidence_items"][:2]:
                    print(f"       - {ev['type']}: {ev['text']}")

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
