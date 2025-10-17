#!/usr/bin/env python3
"""
Guardian v7 - Consistency Checker
Detects cross-document contradictions using corpus index

v7 Memory & Consistency: Cross-document validation
"""
import json
import re
from datetime import datetime
from pathlib import Path


class ConsistencyChecker:
    """
    Check document consistency against corpus index
    Detect contradictions and compute Continuity Score
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.memory_dir = self.root / "qc" / "guardian_v4" / "memory"
        self.index_path = self.memory_dir / "index_v7.json"

        # Recency weights (for scoring)
        self.recency_halflife_days = 30  # Claims older than 30 days have half weight

    def load_index(self) -> dict:
        """Load corpus index"""
        if not self.index_path.exists():
            return {}

        try:
            return json.load(open(self.index_path))
        except Exception:
            return {}

    def extract_claims_from_text(self, text: str) -> list[dict]:
        """
        Extract claims from text (simplified version of indexer logic)
        """
        # Import patterns from indexer
        claim_patterns = {
            "openness": ["openness", "open", "transparency", "sharing"],
            "cooperation": ["cooperation", "collaborate", "collective", "shared"],
            "competition": ["competition", "competitive", "contest", "rivalry"],
            "resilience": ["resilience", "robust", "stability", "survive"],
            "coherence": ["coherence", "cci", "collective coherence", "integration"],
            "hazard": ["hazard", "risk", "threat", "failure"],
            "meaning": ["meaning", "purpose", "significance", "value"],
            "causality": ["causal", "cause", "effect", "mechanism"],
        }

        positive_stance = {
            "increases",
            "improves",
            "enhances",
            "boosts",
            "strengthens",
            "promotes",
            "supports",
            "facilitates",
            "positively",
            "benefits",
        }

        negative_stance = {
            "decreases",
            "reduces",
            "weakens",
            "harms",
            "undermines",
            "inhibits",
            "negatively",
            "damages",
            "impairs",
            "degrades",
        }

        # Split into sentences
        sentences = re.split(r"[.!?]+\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        claims = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            words = set(re.findall(r"\b\w+\b", sentence_lower))

            # Detect stance
            positive_count = len(words & positive_stance)
            negative_count = len(words & negative_stance)

            if positive_count > negative_count:
                stance = "positive"
            elif negative_count > positive_count:
                stance = "negative"
            else:
                continue  # Skip neutral

            # Detect claim key
            for key, patterns in claim_patterns.items():
                if any(pattern in sentence_lower for pattern in patterns):
                    claims.append(
                        {"text": sentence[:200], "claim_key": key, "stance": stance}
                    )
                    break  # Only one key per claim

        return claims

    def compute_recency_weight(self, modified_at: str) -> float:
        """
        Compute recency weight for a claim
        Uses exponential decay: weight = 2^(-days / halflife)

        Args:
            modified_at: ISO timestamp

        Returns:
            Weight between 0 and 1
        """
        try:
            modified_date = datetime.fromisoformat(modified_at)
            now = datetime.now()
            days_old = (now - modified_date).days

            # Exponential decay
            weight = 2 ** (-days_old / self.recency_halflife_days)
            return max(0.1, min(1.0, weight))  # Clamp to [0.1, 1.0]
        except Exception:
            return 0.5  # Default weight

    def detect_contradictions(
        self, document_claims: list[dict], file_path: Path = None
    ) -> list[dict]:
        """
        Detect contradictions between document claims and corpus index

        Args:
            document_claims: List of claims from current document
            file_path: Optional path to current document (to exclude from comparison)

        Returns:
            List of contradiction dictionaries
        """
        index = self.load_index()

        if not index:
            return []  # No index to check against

        claim_key_index = index.get("claim_key_index", {})
        contradictions = []

        # For each claim in the document
        for doc_claim in document_claims:
            claim_key = doc_claim["claim_key"]
            doc_stance = doc_claim["stance"]

            # Opposing stance
            opposing_stance = "negative" if doc_stance == "positive" else "positive"

            # Find claims in corpus with same key but opposite stance
            corpus_claims = claim_key_index.get(claim_key, [])

            for corpus_claim in corpus_claims:
                # Skip if same file
                if file_path and corpus_claim["file"] == str(file_path):
                    continue

                # Check if opposing stance
                if corpus_claim["stance"] == opposing_stance:
                    # Found contradiction!
                    recency_weight = self.compute_recency_weight(
                        corpus_claim.get("modified_at", datetime.now().isoformat())
                    )

                    contradictions.append(
                        {
                            "claim_key": claim_key,
                            "document_claim": doc_claim["text"],
                            "document_stance": doc_stance,
                            "corpus_claim": corpus_claim["text"],
                            "corpus_stance": corpus_claim["stance"],
                            "corpus_file": corpus_claim["file"],
                            "recency_weight": recency_weight,
                        }
                    )

        return contradictions

    def compute_continuity_score(
        self, contradictions: list[dict], total_claims: int
    ) -> float:
        """
        Compute Continuity Score: measures consistency across corpus

        Formula: 1 - (weighted_contradictions / total_claims)

        Args:
            contradictions: List of detected contradictions
            total_claims: Total number of claims in document

        Returns:
            Score between 0 and 1 (1 = perfect continuity)
        """
        if total_claims == 0:
            return 1.0  # No claims = no contradictions

        # Weight contradictions by recency
        weighted_contradictions = sum(
            c.get("recency_weight", 1.0) for c in contradictions
        )

        # Compute rate
        contradiction_rate = weighted_contradictions / total_claims

        # Continuity score
        score = max(0.0, 1.0 - contradiction_rate)

        return score

    def check_document_consistency(self, text: str, file_path: Path = None) -> dict:
        """
        Check consistency of a document against corpus

        Args:
            text: Document text
            file_path: Optional path to document

        Returns:
            Consistency report dictionary
        """
        # Extract claims from document
        document_claims = self.extract_claims_from_text(text)

        if not document_claims:
            return {
                "claim_count": 0,
                "contradiction_count": 0,
                "contradictions": [],
                "continuity_score": 1.0,
                "status": "no_claims",
            }

        # Detect contradictions
        contradictions = self.detect_contradictions(document_claims, file_path)

        # Compute continuity score
        continuity_score = self.compute_continuity_score(
            contradictions, len(document_claims)
        )

        # Classify status
        if continuity_score >= 0.90:
            status = "excellent"
        elif continuity_score >= 0.75:
            status = "good"
        elif continuity_score >= 0.60:
            status = "moderate"
        else:
            status = "poor"

        return {
            "claim_count": len(document_claims),
            "contradiction_count": len(contradictions),
            "contradictions": contradictions,
            "continuity_score": continuity_score,
            "status": status,
            "passes_threshold": continuity_score >= 0.75,
        }


def main():
    """CLI interface for consistency checker"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v7 Consistency Checker")
    parser.add_argument("command", choices=["check", "test"], help="Command to execute")
    parser.add_argument("--file", type=str, help="File to check")
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/consistency_report.json",
        help="Output file path",
    )

    args = parser.parse_args()

    checker = ConsistencyChecker()

    if args.command == "check":
        if not args.file:
            print("❌ Error: --file required")
            return

        file_path = Path(args.file)

        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        print(f"🔍 Checking consistency: {file_path}")

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        result = checker.check_document_consistency(text, file_path)

        print("\n✅ Consistency check complete")
        print(f"   Claims: {result['claim_count']}")
        print(f"   Contradictions: {result['contradiction_count']}")
        print(f"   Continuity Score: {result['continuity_score']:.2f}")
        print(f"   Status: {result['status'].upper()}")

        if result["contradictions"]:
            print("\n   ⚠️  Contradictions found:")
            for i, contr in enumerate(result["contradictions"][:5], 1):
                print(f"\n   {i}. Claim key: {contr['claim_key']}")
                print(
                    f"      This doc ({contr['document_stance']}): {contr['document_claim'][:60]}..."
                )
                print(
                    f"      Corpus ({contr['corpus_stance']}): {contr['corpus_claim'][:60]}..."
                )
                print(f"      Source: {contr['corpus_file']}")

        # Save report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\n   Saved: {output_path}")

    elif args.command == "test":
        print("🧪 Running consistency checker tests...")

        # Test 1: Document with no contradictions
        test1 = """
        Openness increases cooperation in agent societies.
        Cooperation enhances collective resilience.
        Resilience reduces hazard rates significantly.
        """

        result1 = checker.check_document_consistency(test1)
        print(
            f"\nTest 1 (Consistent): Score = {result1['continuity_score']:.2f}, Contradictions = {result1['contradiction_count']}"
        )

        # Test 2: Document with contradictions
        test2 = """
        Openness decreases cooperation among agents.
        Competition improves system performance.
        Cooperation weakens overall resilience.
        """

        result2 = checker.check_document_consistency(test2)
        print(
            f"Test 2 (Contradictory): Score = {result2['continuity_score']:.2f}, Contradictions = {result2['contradiction_count']}"
        )

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
