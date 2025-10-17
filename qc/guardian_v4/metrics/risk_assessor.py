#!/usr/bin/env python3
"""
Guardian v4 - Risk Assessor & Transparency Index v2
Combines all metrics into comprehensive ethical risk assessment

v5 Stabilizer: Deterministic scoring with fixed seeding
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

# v5 Stabilizer: Deterministic seed
GUARDIAN_SEED = int(os.getenv("GUARDIAN_SEED", "42"))


class TransparencyIndexV2:
    """
    Enhanced transparency scoring with citation validation
    and metadata completeness checks
    """

    def __init__(self):
        self.required_metadata_fields = [
            "study_id",
            "classification",
            "seeds",
            "preregistration",
        ]

    def compute_citation_density(self, text: str) -> float:
        """Count citations per 1000 words"""
        words = len(re.findall(r"\b\w+\b", text))
        if words == 0:
            return 0.0

        # Find citations (DOI, URLs, parenthetical years)
        citations = (
            len(re.findall(r"doi\.org/[\w\./\-]+", text, re.I))
            + len(re.findall(r"https?://[^\s]+", text))
            + len(re.findall(r"\(\d{4}\)", text))
        )

        # Density per 1000 words
        density = (citations / words) * 1000
        return min(1.0, density / 5.0)  # Normalize: 5 citations per 1000 words = 1.0

    def check_metadata_completeness(self, text: str) -> float:
        """Check for required metadata fields"""
        found_fields = 0
        for field in self.required_metadata_fields:
            # Check various patterns for each field
            if field == "study_id":
                if re.search(r"study[_\s]id|phase\d+", text, re.I):
                    found_fields += 1
            elif field == "classification":
                if re.search(
                    r"VALIDATED|UNDER_REVIEW|HYPOTHESIS-GEN|classification", text, re.I
                ):
                    found_fields += 1
            elif field == "seeds":
                if re.search(r"seed[s]?[:=\s]*\[?\d+", text, re.I):
                    found_fields += 1
            elif field == "preregistration":
                if re.search(r"preregister|prereg|study\.yml", text, re.I):
                    found_fields += 1

        return found_fields / len(self.required_metadata_fields)

    def check_data_availability(self, text: str) -> float:
        """Check for data availability statements"""
        indicators = [
            r"data[_\s]availability",
            r"discovery_results/",
            r"reproduction[_\s]instructions",
            r"code[_\s]availability",
            r"github\.com",
            r"zenodo\.org",
        ]

        found = sum(1 for pattern in indicators if re.search(pattern, text, re.I))
        return min(1.0, found / 3.0)  # 3+ indicators = 1.0

    def compute_transparency_index_v2(self, text: str) -> dict:
        """Compute enhanced transparency index"""
        citation_density = self.compute_citation_density(text)
        metadata_completeness = self.check_metadata_completeness(text)
        data_availability = self.check_data_availability(text)

        # Weighted average (from schema)
        index = (
            0.4 * citation_density
            + 0.3 * metadata_completeness
            + 0.3 * data_availability
        )

        return {
            "transparency_index_v2": index,
            "citation_density": citation_density,
            "metadata_completeness": metadata_completeness,
            "data_availability": data_availability,
            "components": {
                "citations_found": re.findall(r"doi\.org|https?://|\(\d{4}\)", text)[
                    :5
                ],
                "metadata_fields_found": metadata_completeness
                * len(self.required_metadata_fields),
                "data_availability_indicators": data_availability * 3,
            },
        }


class RiskAssessor:
    """
    Comprehensive ethical risk assessment
    Combines objectivity, transparency, language safety, and sentiment
    """

    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        self.transparency = TransparencyIndexV2()

    def _load_config(self, config_path: Path) -> dict:
        """Load scoring schema"""
        if config_path is None:
            config_path = Path("qc/guardian_v4/config/scoring_schema.yml")

        if config_path.exists():
            try:
                import yaml

                return yaml.safe_load(config_path.read_text())
            except:
                pass
        return self._default_config()

    def _default_config(self) -> dict:
        """Default configuration if YAML not available"""
        return {
            "metrics": {
                "objectivity_score": {"target": 0.80, "weight": 0.25},
                "transparency_index_v2": {"target": 0.90, "weight": 0.30},
                "language_safety_score": {"target": 0.85, "weight": 0.25},
                "sentiment_neutrality": {"target_range": [-0.1, 0.1], "weight": 0.20},
            },
            "guardian_alignment_score": {"target": 90},
            "ethical_risk_levels": {
                "low": {"range": [90, 100], "color": "green"},
                "medium": {"range": [70, 90], "color": "yellow"},
                "high": {"range": [50, 70], "color": "orange"},
                "critical": {"range": [0, 50], "color": "red"},
            },
        }

    def compute_guardian_alignment_score(
        self,
        objectivity: float,
        transparency: float,
        language_safety: float,
        sentiment: float,
    ) -> float:
        """
        Compute overall Guardian alignment score (0-100)

        Formula from schema:
        score = 100 * (
            0.25 * objectivity +
            0.30 * transparency +
            0.25 * language_safety +
            0.20 * (0.5 + 0.5 * (1 - abs(sentiment)))
        )
        """
        # Sentiment contribution: penalize deviation from neutral (0)
        sentiment_contribution = 0.5 + 0.5 * (1 - abs(sentiment))

        score = 100 * (
            0.25 * objectivity
            + 0.30 * transparency
            + 0.25 * language_safety
            + 0.20 * sentiment_contribution
        )

        return max(0.0, min(100.0, score))

    def assess_risk_level(self, guardian_score: float) -> dict:
        """Determine ethical risk level based on Guardian score"""
        risk_levels = self.config.get("ethical_risk_levels", {})

        for level, config in risk_levels.items():
            range_min, range_max = config["range"]
            if range_min <= guardian_score <= range_max:
                return {
                    "risk_level": level,
                    "color": config["color"],
                    "action": config.get("action", "Review required"),
                    "score": guardian_score,
                }

        return {
            "risk_level": "unknown",
            "color": "gray",
            "action": "Manual review",
            "score": guardian_score,
        }

    def generate_recommendations(self, metrics: dict) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []
        targets = self.config.get("metrics", {})

        # Check each metric against target
        objectivity = metrics.get("objectivity_score", 0)
        if objectivity < targets.get("objectivity_score", {}).get("target", 0.80):
            recommendations.append(
                f"⚠️  Objectivity ({objectivity:.2f}) below target (0.80): "
                "Add more hedging terms (suggests, indicates, may) and reduce "
                "overclaiming language (proves, definitively)"
            )

        transparency = metrics.get("transparency_index_v2", 0)
        if transparency < targets.get("transparency_index_v2", {}).get("target", 0.90):
            recommendations.append(
                f"⚠️  Transparency ({transparency:.2f}) below target (0.90): "
                "Add citations, metadata (study_id, seeds), and data availability statements"
            )

        language_safety = metrics.get("language_safety_score", 0)
        if language_safety < targets.get("language_safety_score", {}).get(
            "target", 0.85
        ):
            recommendations.append(
                f"⚠️  Language Safety ({language_safety:.2f}) below target (0.85): "
                "Reduce coercive language (must, should) and overstatement "
                "(revolutionary, breakthrough)"
            )

        sentiment = metrics.get("sentiment_neutrality", 0)
        target_range = targets.get("sentiment_neutrality", {}).get(
            "target_range", [-0.1, 0.1]
        )
        if not (target_range[0] <= sentiment <= target_range[1]):
            recommendations.append(
                f"⚠️  Sentiment ({sentiment:.2f}) outside neutral range ({target_range}): "
                "Use more neutral scientific language, avoid excessive positive/negative terms"
            )

        if not recommendations:
            recommendations.append("✅ All metrics meet targets")

        return recommendations

    def validate_document(self, file_path: Path) -> dict:
        """Perform complete validation of a document"""
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        # Import analysis modules
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        try:
            from nlp.objectivity_model import ObjectivityModel
            from nlp.sentiment_analyzer import SentimentAnalyzer

            # Run all analyses
            obj_model = ObjectivityModel()
            sent_analyzer = SentimentAnalyzer()

            objectivity_result = obj_model.compute_objectivity_score(text)
            safety_result = obj_model.compute_language_safety_score(text)
            sentiment_result = sent_analyzer.analyze_sentiment(text)
            transparency_result = self.transparency.compute_transparency_index_v2(text)

            # Extract scores
            metrics = {
                "objectivity_score": objectivity_result["objectivity_score"],
                "language_safety_score": safety_result["language_safety_score"],
                "transparency_index_v2": transparency_result["transparency_index_v2"],
                "sentiment_neutrality": sentiment_result["sentiment_score"],
            }

            # Compute Guardian score
            guardian_score = self.compute_guardian_alignment_score(
                metrics["objectivity_score"],
                metrics["transparency_index_v2"],
                metrics["language_safety_score"],
                metrics["sentiment_neutrality"],
            )

            # Assess risk
            risk = self.assess_risk_level(guardian_score)

            # Generate recommendations
            recommendations = self.generate_recommendations(metrics)

            return {
                "file": str(file_path),
                "guardian_alignment_score": guardian_score,
                "risk_assessment": risk,
                "metrics": {
                    **metrics,
                    "objectivity_details": objectivity_result,
                    "language_safety_details": safety_result,
                    "sentiment_details": sentiment_result,
                    "transparency_details": transparency_result,
                },
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "file": str(file_path)}


def main():
    """CLI interface for risk assessor"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v4 Risk Assessor")
    parser.add_argument(
        "command", choices=["validate", "test"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File to validate")
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/risk_summary.md",
        help="Output file path",
    )

    args = parser.parse_args()

    assessor = RiskAssessor()

    if args.command == "validate":
        if not args.file:
            print("❌ Error: --file required for validate command")
            return

        print(f"📊 Validating {args.file}...")
        result = assessor.validate_document(Path(args.file))

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        # Print summary
        print("\n✅ Validation complete")
        print(f"   Guardian Score: {result['guardian_alignment_score']:.1f}/100")
        print(
            f"   Risk Level: {result['risk_assessment']['risk_level'].upper()} ({result['risk_assessment']['color']})"
        )
        print("\n📊 Component Scores:")
        print(f"   Objectivity: {result['metrics']['objectivity_score']:.2f}")
        print(f"   Transparency: {result['metrics']['transparency_index_v2']:.2f}")
        print(f"   Language Safety: {result['metrics']['language_safety_score']:.2f}")
        print(
            f"   Sentiment Neutrality: {result['metrics']['sentiment_neutrality']:.2f}"
        )

        print("\n💡 Recommendations:")
        for rec in result["recommendations"]:
            print(f"   {rec}")

        # Save detailed results
        output_json = Path(args.output).with_suffix(".json")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2))
        print(f"\n📄 Detailed results: {output_json}")

    elif args.command == "test":
        print("🧪 Running test validation...")

        # Create test document
        test_doc = Path("qc/guardian_v4/test_document.md")
        test_doc.parent.mkdir(parents=True, exist_ok=True)
        test_doc.write_text(
            """
# Test Study: Phase 99 Validation

**Study ID**: phase99_test  
**Classification**: VALIDATED  
**Seeds**: [11, 17, 23, 29]  
**Preregistration**: studies/phase99_test.yml

## Abstract

This study suggests that parameter epsilon may influence outcome CCI
(ΔCCI = 0.045, 95% CI [0.03, 0.06]). Within this simulation framework,
results are preliminary and warrant further empirical validation.

## Methods

Data available at discovery_results/phase99_test/. Analysis used bootstrap
confidence intervals (n=1000) as described in Efron & Tibshirani (1993).

## References

- Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
- https://github.com/hymetalab/research
- DOI: 10.1234/example.5678
        """
        )

        result = assessor.validate_document(test_doc)

        print("\n✅ Test validation complete")
        print(f"   Guardian Score: {result['guardian_alignment_score']:.1f}/100")
        print(f"   Risk Level: {result['risk_assessment']['risk_level'].upper()}")
        print(
            f"   Status: {'PASS' if result['guardian_alignment_score'] >= 70 else 'FAIL'}"
        )

        test_doc.unlink()  # Clean up
        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
