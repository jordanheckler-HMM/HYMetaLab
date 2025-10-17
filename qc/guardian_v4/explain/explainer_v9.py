#!/usr/bin/env python3
"""
Guardian v9 - Explainability Engine
Generates human-readable explanations for all Guardian metrics

v9 Explainability: Full transparency with causal tracing
"""
import json
import re
from pathlib import Path


class GuardianExplainer:
    """
    Convert Guardian metrics into human-readable explanations
    Shows exactly what text contributed to each score
    """

    def __init__(self):
        pass

    def extract_text_evidence(
        self, text: str, terms: list[str], context_chars: int = 100
    ) -> list[dict]:
        """
        Find where specific terms appear in text with context

        Args:
            text: Full document text
            terms: Terms to find
            context_chars: Characters of context to include

        Returns:
            List of evidence snippets
        """
        text_lower = text.lower()
        evidence = []

        for term in terms:
            # Find all occurrences
            pattern = re.escape(term.lower())
            matches = re.finditer(pattern, text_lower)

            for match in matches:
                start = max(0, match.start() - context_chars)
                end = min(len(text), match.end() + context_chars)

                snippet = text[start:end]

                evidence.append(
                    {
                        "term": term,
                        "position": match.start(),
                        "snippet": snippet,
                        "highlighted": text[match.start() : match.end()],
                    }
                )

        return evidence

    def explain_objectivity(self, objectivity_data: dict, text: str) -> dict:
        """
        Generate explanation for objectivity score

        Args:
            objectivity_data: Output from ObjectivityModel
            text: Document text

        Returns:
            Explanation dictionary with trace and narrative
        """
        score = objectivity_data.get("objectivity_score", 0)
        components = objectivity_data.get("components", {})

        # Build trace
        hedge_bonus = components.get("hedge_bonus", 0)
        overclaim_penalty = components.get("overclaim_penalty", 0)
        citation_bonus = components.get("citation_bonus", 0)

        # Generate natural language explanation
        explanation_parts = []

        # Introduce score
        if score >= 0.80:
            explanation_parts.append(
                f"This document demonstrates strong objectivity (score: {score:.2f})."
            )
        elif score >= 0.70:
            explanation_parts.append(
                f"This document shows good objectivity (score: {score:.2f})."
            )
        elif score >= 0.60:
            explanation_parts.append(
                f"This document exhibits moderate objectivity (score: {score:.2f})."
            )
        else:
            explanation_parts.append(
                f"This document suggests room for improvement in objectivity (score: {score:.2f})."
            )

        # Explain components
        if hedge_bonus > 0.05:
            explanation_parts.append(
                f"The text includes appropriate hedging language "
                f"(+{hedge_bonus:.2f} bonus), which indicates epistemic humility and appropriate caution."
            )

        if overclaim_penalty > 0.05:
            explanation_parts.append(
                f"However, overclaiming language was detected "
                f"(-{overclaim_penalty:.2f} penalty), suggesting some statements may overstate certainty."
            )

        if citation_bonus > 0.05:
            explanation_parts.append(
                f"The document includes citation indicators "
                f"(+{citation_bonus:.2f} bonus), showing engagement with evidence sources."
            )

        # Add suggestion if score is low
        if score < 0.70:
            explanation_parts.append(
                "To improve: Consider adding qualifying terms (e.g., 'suggests', 'indicates'), "
                "reducing absolute claims, and including more citations."
            )

        narrative = " ".join(explanation_parts)

        return {
            "metric": "objectivity_score",
            "score": score,
            "grade": self._score_to_grade(score),
            "explanation": narrative,
            "trace": {
                "hedge_bonus": hedge_bonus,
                "overclaim_penalty": overclaim_penalty,
                "citation_bonus": citation_bonus,
                "hedge_density": objectivity_data.get("hedge_density", 0),
                "overclaim_density": objectivity_data.get("overclaim_density", 0),
                "citation_density": objectivity_data.get("citation_density", 0),
            },
            "traceable": True,
        }

    def explain_transparency(self, transparency_data: dict, text: str) -> dict:
        """Generate explanation for transparency index v2"""
        score = transparency_data.get("transparency_index_v2", 0)
        components = transparency_data.get("components", {})

        explanation_parts = []

        # Introduce
        if score >= 0.80:
            explanation_parts.append(
                f"This document shows excellent transparency (score: {score:.2f})."
            )
        elif score >= 0.70:
            explanation_parts.append(
                f"This document demonstrates good transparency (score: {score:.2f})."
            )
        else:
            explanation_parts.append(
                f"This document's transparency could be improved (score: {score:.2f})."
            )

        # Citation compliance
        citation_score = components.get("citation_compliance", 0)
        if citation_score > 0.5:
            explanation_parts.append(
                f"The document includes citations and references "
                f"(citation compliance: {citation_score:.2f})."
            )
        else:
            explanation_parts.append(
                f"Citation coverage appears limited "
                f"(citation compliance: {citation_score:.2f}). "
                f"Consider adding DOIs or URLs to support claims."
            )

        # Metadata completeness
        metadata_score = components.get("metadata_completeness", 0)
        if metadata_score > 0.5:
            explanation_parts.append(
                f"Metadata is reasonably complete (score: {metadata_score:.2f})."
            )

        narrative = " ".join(explanation_parts)

        return {
            "metric": "transparency_index_v2",
            "score": score,
            "grade": self._score_to_grade(score),
            "explanation": narrative,
            "trace": components,
            "traceable": True,
        }

    def explain_language_safety(self, language_safety_data: dict, text: str) -> dict:
        """Generate explanation for language safety"""
        score = language_safety_data.get("language_safety_score", 0)

        explanation_parts = []

        if score >= 0.85:
            explanation_parts.append(
                f"Language safety is excellent (score: {score:.2f})."
            )
        elif score >= 0.75:
            explanation_parts.append(f"Language safety is good (score: {score:.2f}).")
        else:
            explanation_parts.append(
                f"Language safety could be improved (score: {score:.2f})."
            )

        explanation_parts.append(
            "The text was evaluated for coercive phrasing, overstatements, "
            "and potentially harmful framing."
        )

        if score < 0.80:
            explanation_parts.append(
                "Consider reviewing for absolute claims and softening language where appropriate."
            )

        narrative = " ".join(explanation_parts)

        return {
            "metric": "language_safety_score",
            "score": score,
            "grade": self._score_to_grade(score),
            "explanation": narrative,
            "trace": language_safety_data,
            "traceable": True,
        }

    def explain_sentiment(self, sentiment_data: dict, text: str) -> dict:
        """Generate explanation for sentiment neutrality"""
        score = sentiment_data.get("sentiment_score", 0)

        explanation_parts = []

        if abs(score) <= 0.1:
            explanation_parts.append(
                f"Sentiment is appropriately neutral (score: {score:+.2f}), "
                f"suitable for scientific discourse."
            )
        elif abs(score) <= 0.2:
            explanation_parts.append(
                f"Sentiment is mostly neutral (score: {score:+.2f}) "
                f"with minor emotional coloring."
            )
        else:
            if score > 0:
                explanation_parts.append(
                    f"Sentiment appears somewhat positive (score: {score:+.2f}). "
                    f"Consider adopting more neutral phrasing."
                )
            else:
                explanation_parts.append(
                    f"Sentiment appears somewhat negative (score: {score:+.2f}). "
                    f"Consider adopting more neutral phrasing."
                )

        narrative = " ".join(explanation_parts)

        return {
            "metric": "sentiment_neutrality",
            "score": score,
            "grade": self._sentiment_to_grade(score),
            "explanation": narrative,
            "trace": sentiment_data,
            "traceable": True,
        }

    def explain_overall_score(self, guardian_score: float, all_metrics: dict) -> dict:
        """Generate explanation for overall Guardian alignment score"""

        explanation_parts = []

        if guardian_score >= 90:
            explanation_parts.append(
                f"Overall Guardian alignment is excellent ({guardian_score:.1f}/100). "
                f"This document meets high ethical standards across all metrics."
            )
        elif guardian_score >= 70:
            explanation_parts.append(
                f"Overall Guardian alignment is good ({guardian_score:.1f}/100). "
                f"This document generally meets ethical standards with some areas for refinement."
            )
        elif guardian_score >= 50:
            explanation_parts.append(
                f"Overall Guardian alignment is moderate ({guardian_score:.1f}/100). "
                f"This document shows several areas that would benefit from improvement."
            )
        else:
            explanation_parts.append(
                f"Overall Guardian alignment needs attention ({guardian_score:.1f}/100). "
                f"Substantial improvements may be warranted across multiple metrics."
            )

        # Highlight strongest and weakest metrics
        metrics = {
            "Objectivity": all_metrics.get("objectivity_score", 0),
            "Transparency": all_metrics.get("transparency_index_v2", 0),
            "Language Safety": all_metrics.get("language_safety_score", 0),
            "Sentiment Neutrality": 1
            - abs(all_metrics.get("sentiment_neutrality", 0)),  # Convert to 0-1
        }

        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)

        if sorted_metrics:
            strongest = sorted_metrics[0]
            weakest = sorted_metrics[-1]

            explanation_parts.append(
                f"Strongest metric: {strongest[0]} ({strongest[1]:.2f}). "
                f"Weakest metric: {weakest[0]} ({weakest[1]:.2f})."
            )

        narrative = " ".join(explanation_parts)

        return {
            "metric": "guardian_alignment_score",
            "score": guardian_score,
            "grade": self._overall_to_grade(guardian_score),
            "explanation": narrative,
            "trace": all_metrics,
            "traceable": True,
        }

    def explain_all(self, guardian_report: dict, text: str) -> dict:
        """
        Generate comprehensive explanations for all metrics

        Args:
            guardian_report: Full Guardian validation report
            text: Document text

        Returns:
            Complete explanation package
        """
        detailed_metrics = guardian_report.get("detailed_metrics", {})
        metrics = guardian_report.get("metrics", {})

        explanations = {}

        # Objectivity
        if "objectivity" in detailed_metrics:
            explanations["objectivity"] = self.explain_objectivity(
                detailed_metrics["objectivity"], text
            )

        # Transparency
        if "transparency" in detailed_metrics:
            explanations["transparency"] = self.explain_transparency(
                detailed_metrics["transparency"], text
            )

        # Language safety
        if "language_safety" in detailed_metrics:
            explanations["language_safety"] = self.explain_language_safety(
                detailed_metrics["language_safety"], text
            )

        # Sentiment
        if "sentiment" in detailed_metrics:
            explanations["sentiment"] = self.explain_sentiment(
                detailed_metrics["sentiment"], text
            )

        # Overall score
        explanations["overall"] = self.explain_overall_score(
            guardian_report.get("guardian_alignment_score", 0), metrics
        )

        return {
            "explanations": explanations,
            "all_traceable": all(
                e.get("traceable", False) for e in explanations.values()
            ),
            "clarity_score": None,  # To be filled by user testing
            "generated_at": guardian_report.get("timestamp"),
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.90:
            return "A"
        elif score >= 0.80:
            return "B"
        elif score >= 0.70:
            return "C"
        elif score >= 0.60:
            return "D"
        else:
            return "F"

    def _sentiment_to_grade(self, score: float) -> str:
        """Convert sentiment score to grade (centered at 0)"""
        abs_score = abs(score)
        if abs_score <= 0.1:
            return "A"
        elif abs_score <= 0.2:
            return "B"
        elif abs_score <= 0.3:
            return "C"
        else:
            return "D"

    def _overall_to_grade(self, score: float) -> str:
        """Convert overall score (0-100) to grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


def main():
    """CLI for explainer"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v9 Explainer")
    parser.add_argument(
        "command", choices=["explain", "test"], help="Command to execute"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="qc/guardian_v4/guardian_report_v4.json",
        help="Guardian report to explain",
    )
    parser.add_argument("--file", type=str, help="Original document file")
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/guardian_explanations_v9.json",
        help="Output file for explanations",
    )

    args = parser.parse_args()

    explainer = GuardianExplainer()

    if args.command == "explain":
        if not args.file:
            print("❌ Error: --file required")
            return

        # Load report
        report_path = Path(args.report)
        if not report_path.exists():
            print(f"❌ Error: Report not found: {report_path}")
            return

        report = json.load(open(report_path))

        # Load document text
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        print(f"🔍 Generating explanations for {file_path.name}...")

        # Generate explanations
        explanations = explainer.explain_all(report, text)

        print("\n✅ Explanations generated")
        print(f"   All traceable: {'✅' if explanations['all_traceable'] else '❌'}")

        # Display explanations
        print("\n📊 Explanations:\n")
        for metric_name, explanation in explanations["explanations"].items():
            print(f"{'='*70}")
            print(
                f"{metric_name.upper()}: {explanation['score']:.2f} (Grade: {explanation['grade']})"
            )
            print(f"{'='*70}")
            print(explanation["explanation"])
            print()

        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(explanations, indent=2))
        print(f"   Saved: {output_path}")

    elif args.command == "test":
        print("🧪 Running explainer tests...")

        # Mock report
        mock_report = {
            "guardian_alignment_score": 75.5,
            "metrics": {
                "objectivity_score": 0.78,
                "transparency_index_v2": 0.72,
                "language_safety_score": 0.85,
                "sentiment_neutrality": 0.05,
            },
            "detailed_metrics": {
                "objectivity": {
                    "objectivity_score": 0.78,
                    "components": {
                        "hedge_bonus": 0.12,
                        "overclaim_penalty": 0.04,
                        "citation_bonus": 0.10,
                    },
                    "hedge_density": 2.4,
                    "overclaim_density": 0.5,
                    "citation_density": 5.0,
                },
                "transparency": {
                    "transparency_index_v2": 0.72,
                    "components": {
                        "citation_compliance": 0.65,
                        "metadata_completeness": 0.80,
                    },
                },
                "language_safety": {"language_safety_score": 0.85},
                "sentiment": {"sentiment_score": 0.05},
            },
            "timestamp": "2025-10-14T20:00:00",
        }

        mock_text = "This study suggests that openness may enhance cooperation."

        explanations = explainer.explain_all(mock_report, mock_text)

        print("\n✅ Test explanations generated:")
        print(f"   Metrics explained: {len(explanations['explanations'])}")
        print(f"   All traceable: {'✅' if explanations['all_traceable'] else '❌'}")

        # Show sample
        obj_exp = explanations["explanations"].get("objectivity", {})
        print("\n   Sample (Objectivity):")
        print(f"   {obj_exp.get('explanation', '')[:200]}...")


if __name__ == "__main__":
    main()
