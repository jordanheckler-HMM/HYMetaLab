#!/usr/bin/env python3
"""
Guardian v6 - Claim Type Classifier
Classifies research claims as: empirical | interpretive | speculative

v6 Context Engine: Paragraph-level reasoning with evidence tracking
"""
import re
from pathlib import Path


class ClaimClassifier:
    """
    Classify research claims into three categories:
    - EMPIRICAL: Data-driven, observable, testable
    - INTERPRETIVE: Analysis of data, pattern recognition
    - SPECULATIVE: Hypotheses, predictions, untested ideas
    """

    def __init__(self):
        # Empirical indicators (data-driven)
        self.empirical_indicators = {
            "measured",
            "observed",
            "recorded",
            "collected",
            "tested",
            "validated",
            "confirmed",
            "verified",
            "demonstrated",
            "showed",
            "found",
            "detected",
            "identified",
            "n=",
            "p<",
            "p=",
            "r=",
            "χ²",
            "t=",
            "f=",
            "mean",
            "median",
            "std",
            "variance",
            "correlation",
            "regression",
            "anova",
            "bootstrap",
            "ci",
            "dataset",
            "sample",
            "survey",
            "experiment",
        }

        # Interpretive indicators (analysis/reasoning)
        self.interpretive_indicators = {
            "suggests",
            "indicates",
            "implies",
            "reflects",
            "consistent with",
            "supports",
            "aligns with",
            "pattern",
            "trend",
            "relationship",
            "association",
            "appears to",
            "seems to",
            "tends to",
            "analysis shows",
            "results indicate",
            "data suggest",
            "interpretation",
            "framework",
            "model",
            "theory",
        }

        # Speculative indicators (hypotheses/predictions)
        self.speculative_indicators = {
            "may",
            "might",
            "could",
            "would",
            "should",
            "hypothesize",
            "predict",
            "expect",
            "anticipate",
            "propose",
            "conjecture",
            "speculate",
            "if",
            "whether",
            "potentially",
            "possibly",
            "future",
            "further research",
            "warrants investigation",
            "remains to be seen",
            "unclear",
            "unknown",
            "hypothesis",
            "prediction",
        }

        # Evidence markers (citations, data references)
        self.evidence_markers = {
            "doi:",
            "http://",
            "https://",
            "(19",
            "(20",  # Years
            "et al.",
            "figure",
            "table",
            "appendix",
            "see",
            "as shown",
            "as described",
            "discovery_results/",
            "data/",
            "results/",
        }

    def classify_sentence(self, sentence: str) -> dict:
        """
        Classify a single sentence as empirical, interpretive, or speculative
        Returns classification with confidence scores
        """
        sentence_lower = sentence.lower()

        # Count indicators for each type
        empirical_score = sum(
            1 for indicator in self.empirical_indicators if indicator in sentence_lower
        )

        interpretive_score = sum(
            1
            for indicator in self.interpretive_indicators
            if indicator in sentence_lower
        )

        speculative_score = sum(
            1
            for indicator in self.speculative_indicators
            if indicator in sentence_lower
        )

        # Check for evidence markers (boosts empirical)
        evidence_count = sum(
            1 for marker in self.evidence_markers if marker in sentence_lower
        )

        if evidence_count > 0:
            empirical_score += evidence_count * 2  # Strong boost for citations

        # Determine primary classification
        total_score = empirical_score + interpretive_score + speculative_score

        if total_score == 0:
            # No clear indicators - default to interpretive (most common in research)
            return {
                "classification": "interpretive",
                "confidence": 0.3,
                "scores": {"empirical": 0.0, "interpretive": 0.3, "speculative": 0.0},
                "evidence_present": False,
            }

        # Normalize scores to probabilities
        scores = {
            "empirical": empirical_score / total_score,
            "interpretive": interpretive_score / total_score,
            "speculative": speculative_score / total_score,
        }

        # Get classification (highest score)
        classification = max(scores.items(), key=lambda x: x[1])[0]
        confidence = scores[classification]

        return {
            "classification": classification,
            "confidence": confidence,
            "scores": scores,
            "evidence_present": evidence_count > 0,
            "evidence_count": evidence_count,
        }

    def classify_paragraph(self, paragraph: str) -> dict:
        """
        Classify a paragraph (multiple sentences)
        Aggregates sentence-level classifications
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", paragraph)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return {"classification": "unknown", "confidence": 0.0, "sentence_count": 0}

        # Classify each sentence
        sentence_results = []
        for sentence in sentences:
            result = self.classify_sentence(sentence)
            sentence_results.append(result)

        # Aggregate classifications
        classification_counts = {"empirical": 0, "interpretive": 0, "speculative": 0}
        total_confidence = 0.0
        total_evidence = 0

        for result in sentence_results:
            cls = result.get("classification", "interpretive")
            # Defensive: map unknown/empty classifications to 'interpretive'
            if cls not in classification_counts:
                cls = "interpretive"
            classification_counts[cls] += 1
            total_confidence += result.get("confidence", 0.0)
            total_evidence += result.get("evidence_count", 0)

        # Determine paragraph classification (majority vote)
        primary_classification = max(classification_counts.items(), key=lambda x: x[1])[
            0
        ]

        # Average confidence
        avg_confidence = total_confidence / len(sentence_results)

        return {
            "classification": primary_classification,
            "confidence": avg_confidence,
            "sentence_count": len(sentence_results),
            "classification_distribution": classification_counts,
            "evidence_present": total_evidence > 0,
            "total_evidence_markers": total_evidence,
            "sentences": sentence_results,
        }

    def classify_document(self, text: str) -> dict:
        """
        Classify entire document
        Returns paragraph-level classifications and document summary
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]

        if not paragraphs:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "paragraph_count": 0,
            }

        # Classify each paragraph
        paragraph_results = []
        for i, paragraph in enumerate(paragraphs):
            result = self.classify_paragraph(paragraph)
            result["paragraph_index"] = i
            paragraph_results.append(result)

        # Aggregate document-level statistics
        classification_counts = {"empirical": 0, "interpretive": 0, "speculative": 0}
        total_evidence = 0
        total_confidence = 0.0

        for result in paragraph_results:
            cls = result.get("classification", "interpretive")
            if cls not in classification_counts:
                cls = "interpretive"
            classification_counts[cls] += 1
            total_evidence += result.get("total_evidence_markers", 0)
            total_confidence += result.get("confidence", 0.0)

        # Document classification (majority of paragraphs)
        doc_classification = max(classification_counts.items(), key=lambda x: x[1])[0]

        # Calculate ratios
        total_paragraphs = len(paragraph_results)
        classification_ratios = {
            k: v / total_paragraphs for k, v in classification_counts.items()
        }

        avg_confidence = (
            total_confidence / total_paragraphs if total_paragraphs > 0 else 0.0
        )

        # Evidence coverage (% of paragraphs with evidence)
        paragraphs_with_evidence = sum(
            1 for p in paragraph_results if p.get("evidence_present")
        )
        evidence_coverage = (
            paragraphs_with_evidence / total_paragraphs if total_paragraphs > 0 else 0.0
        )

        return {
            "document_classification": doc_classification,
            "document_confidence": avg_confidence,
            "paragraph_count": total_paragraphs,
            "classification_distribution": classification_counts,
            "classification_ratios": classification_ratios,
            "total_evidence_markers": total_evidence,
            "evidence_coverage": evidence_coverage,
            "paragraphs_with_evidence": paragraphs_with_evidence,
            "paragraphs": paragraph_results,
        }

    def analyze_file(self, file_path: Path) -> dict:
        """Analyze a file and return claim classification results"""
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        result = self.classify_document(text)
        result["file"] = str(file_path)

        return result


def main():
    """CLI interface for claim classifier"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Guardian v6 Claim Classifier")
    parser.add_argument(
        "command", choices=["classify", "test"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File to classify")
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/claim_classification.json",
        help="Output file path",
    )

    args = parser.parse_args()

    classifier = ClaimClassifier()

    if args.command == "classify":
        if not args.file:
            print("❌ Error: --file required for classify command")
            return

        print(f"📊 Classifying claims in {args.file}...")
        result = classifier.analyze_file(Path(args.file))

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print("\n✅ Classification complete")
        print(f"   Document type: {result['document_classification'].upper()}")
        print(f"   Confidence: {result['document_confidence']:.2f}")
        print(f"   Paragraphs: {result['paragraph_count']}")
        print("\n   Distribution:")
        for claim_type, count in result["classification_distribution"].items():
            pct = result["classification_ratios"][claim_type] * 100
            print(f"     - {claim_type}: {count} ({pct:.1f}%)")
        print(f"\n   Evidence coverage: {result['evidence_coverage']*100:.1f}%")
        print(
            f"   Paragraphs with evidence: {result['paragraphs_with_evidence']}/{result['paragraph_count']}"
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\n   Output: {output_path}")

    elif args.command == "test":
        print("🧪 Running claim classifier tests...")

        # Test case 1: Empirical claim
        test1 = "We measured CCI across 200 agents (mean=0.75, 95% CI [0.71, 0.79]) using bootstrap resampling (n=1000). The data showed significant improvement (p<0.001)."
        result1 = classifier.classify_sentence(test1)
        print(
            f"\nTest 1 (Empirical): {result1['classification']} (conf={result1['confidence']:.2f})"
        )
        print(
            f"  Scores: empirical={result1['scores']['empirical']:.2f}, interpretive={result1['scores']['interpretive']:.2f}, speculative={result1['scores']['speculative']:.2f}"
        )

        # Test case 2: Interpretive claim
        test2 = "These results suggest that openness may enhance collective coherence. The pattern is consistent with theories of shared intentionality."
        result2 = classifier.classify_sentence(test2)
        print(
            f"\nTest 2 (Interpretive): {result2['classification']} (conf={result2['confidence']:.2f})"
        )

        # Test case 3: Speculative claim
        test3 = "Future research could explore whether this effect generalizes to larger populations. It remains unclear if the mechanism applies universally."
        result3 = classifier.classify_sentence(test3)
        print(
            f"\nTest 3 (Speculative): {result3['classification']} (conf={result3['confidence']:.2f})"
        )

        # Test case 4: Paragraph with mixed claims
        test4 = """We observed increased coherence in cooperative conditions (ΔCCI=0.045, p<0.01).
        This suggests that shared meaning contexts enhance resilience.
        Future studies might investigate the mechanisms underlying this effect."""

        result4 = classifier.classify_paragraph(test4)
        print(
            f"\nTest 4 (Paragraph): {result4['classification']} (conf={result4['confidence']:.2f})"
        )
        print(f"  Distribution: {result4['classification_distribution']}")

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
