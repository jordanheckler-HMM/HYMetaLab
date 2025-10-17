#!/usr/bin/env python3
"""
Guardian v4 - Sentiment & Tone Analyzer
Detects emotional tone in research documents (prefer neutral scientific tone)

v5 Stabilizer: Deterministic scoring with fixed seeding
"""
import json
import os
import re
from pathlib import Path

# v5 Stabilizer: Deterministic seed
GUARDIAN_SEED = int(os.getenv("GUARDIAN_SEED", "42"))


class SentimentAnalyzer:
    """
    Sentiment analyzer for scientific documents
    Optimizes for neutral tone (avoid excessive positive or negative language)
    """

    def __init__(self):
        # Positive sentiment indicators (non-scientific)
        self.positive_terms = {
            "excellent": 2,
            "amazing": 2,
            "fantastic": 2,
            "perfect": 2,
            "wonderful": 2,
            "great": 1,
            "good": 1,
            "positive": 1,
            "promising": 0.5,
            "favorable": 0.5,
            "beneficial": 0.5,
        }

        # Negative sentiment indicators (non-scientific)
        self.negative_terms = {
            "terrible": -2,
            "awful": -2,
            "horrible": -2,
            "bad": -1,
            "poor": -1,
            "negative": -1,
            "problematic": -1,
            "concerning": -0.5,
            "worrying": -0.5,
            "troubling": -0.5,
        }

        # Neutral scientific terms (encouraged)
        self.neutral_terms = {
            "observed",
            "measured",
            "calculated",
            "analyzed",
            "tested",
            "suggests",
            "indicates",
            "shows",
            "demonstrates",
            "supports",
            "consistent",
            "preliminary",
            "tentative",
            "potential",
        }

    def analyze_sentiment(self, text: str) -> dict[str, float]:
        """
        Compute sentiment score in range [-1, 1]
        -1 = very negative, 0 = neutral, +1 = very positive

        v5 Stabilizer: Deterministic scoring
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        total_words = len(words) if words else 1

        # Count sentiment indicators (deterministic - sorted dict iteration)
        positive_score = sum(self.positive_terms.get(word, 0) for word in sorted(words))
        negative_score = sum(self.negative_terms.get(word, 0) for word in sorted(words))
        neutral_count = sum(1 for word in sorted(words) if word in self.neutral_terms)

        # Compute raw sentiment
        raw_sentiment = (positive_score + negative_score) / total_words

        # Normalize to [-1, 1]
        sentiment = max(-1.0, min(1.0, raw_sentiment * 100))

        # Compute neutrality score (how close to 0)
        neutrality = 1.0 - abs(sentiment)

        # Bonus for neutral scientific terms
        neutral_density = (neutral_count / total_words) * 100
        neutral_bonus = min(0.2, neutral_density * 0.02)
        neutrality += neutral_bonus
        neutrality = min(1.0, neutrality)

        return {
            "sentiment_score": sentiment,
            "neutrality_score": neutrality,
            "positive_indicators": positive_score,
            "negative_indicators": negative_score,
            "neutral_term_density": neutral_density,
            "classification": self._classify_sentiment(sentiment),
            "optimal_for_science": abs(sentiment) < 0.1,
        }

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment into categories"""
        if score < -0.5:
            return "very_negative"
        elif score < -0.1:
            return "negative"
        elif score > 0.5:
            return "very_positive"
        elif score > 0.1:
            return "positive"
        else:
            return "neutral"

    def analyze_tone(self, text: str) -> dict[str, any]:
        """
        Analyze overall tone of document
        Scientific documents should be:
        - Formal (not casual)
        - Objective (not subjective)
        - Measured (not hyperbolic)
        """
        text_lower = text.lower()

        # Formality indicators
        formal_indicators = [
            "furthermore",
            "moreover",
            "however",
            "therefore",
            "consequently",
            "specifically",
            "particularly",
            "notably",
            "significantly",
        ]
        informal_indicators = [
            "basically",
            "actually",
            "really",
            "pretty much",
            "kind of",
            "sort of",
            "like",
            "you know",
            "obviously",
        ]

        formal_count = sum(1 for term in formal_indicators if term in text_lower)
        informal_count = sum(1 for term in informal_indicators if term in text_lower)

        # Objectivity indicators
        subjective_indicators = [
            "i think",
            "i believe",
            "in my opinion",
            "i feel",
            "personally",
            "obviously",
            "clearly",
            "undoubtedly",
        ]
        objective_indicators = [
            "the data",
            "results indicate",
            "analysis shows",
            "observed",
            "measured",
            "calculated",
            "statistically",
        ]

        subjective_count = sum(
            1 for term in subjective_indicators if term in text_lower
        )
        objective_count = sum(1 for term in objective_indicators if term in text_lower)

        # Hyperbole indicators
        hyperbole_indicators = [
            "amazing",
            "incredible",
            "unbelievable",
            "extraordinary",
            "revolutionary",
            "groundbreaking",
            "breakthrough",
            "unprecedented",
            "never before",
            "always",
            "never",
            "absolutely",
            "completely",
        ]

        hyperbole_count = sum(1 for term in hyperbole_indicators if term in text_lower)

        # Compute tone scores
        formality = (formal_count - informal_count) / max(
            1, formal_count + informal_count + 1
        )
        objectivity = (objective_count - subjective_count) / max(
            1, objective_count + subjective_count + 1
        )
        measured = 1.0 - min(1.0, hyperbole_count * 0.1)

        # Overall tone score
        tone_score = (formality + objectivity + measured) / 3
        tone_score = max(0.0, min(1.0, tone_score))

        return {
            "tone_score": tone_score,
            "formality": formality,
            "objectivity": objectivity,
            "measured": measured,
            "hyperbole_count": hyperbole_count,
            "is_scientific_tone": tone_score > 0.6,
        }

    def analyze_document(self, file_path: Path) -> dict:
        """Analyze a single document"""
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        sentiment = self.analyze_sentiment(text)
        tone = self.analyze_tone(text)

        return {
            "file": str(file_path),
            "sentiment": sentiment,
            "tone": tone,
            "recommendation": self._generate_recommendation(sentiment, tone),
        }

    def _generate_recommendation(self, sentiment: dict, tone: dict) -> str:
        """Generate improvement recommendations"""
        issues = []

        if abs(sentiment["sentiment_score"]) > 0.2:
            issues.append("Reduce emotional language for neutral scientific tone")

        if tone["formality"] < 0.3:
            issues.append("Increase formality (use 'furthermore', 'however', etc.)")

        if tone["objectivity"] < 0.3:
            issues.append("Reduce subjective language ('I think', 'obviously', etc.)")

        if tone["hyperbole_count"] > 5:
            issues.append(
                "Reduce hyperbolic language ('revolutionary', 'unprecedented', etc.)"
            )

        if not issues:
            return "✅ Excellent scientific tone"
        else:
            return " | ".join(issues)

    def train(self, training_data_path: Path = None):
        """Train sentiment model (currently rule-based)"""
        print("ℹ️  Sentiment analyzer using rule-based approach")
        print("✅ Training complete (rule-based mode)")


def main():
    """CLI interface for sentiment analyzer"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v4 Sentiment Analyzer")
    parser.add_argument(
        "command", choices=["train", "analyze", "test"], help="Command to execute"
    )
    parser.add_argument("--path", type=str, default=".", help="Path to analyze")
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/sentiment_results.json",
        help="Output file path",
    )

    args = parser.parse_args()

    analyzer = SentimentAnalyzer()

    if args.command == "train":
        print("🧠 Training sentiment analyzer...")
        analyzer.train()
        print("✅ Training complete")

    elif args.command == "analyze":
        print(f"📊 Analyzing documents at {args.path}...")

        root = Path(args.path)
        results = []

        for pattern in ["**/*.md", "**/*.txt"]:
            for file_path in root.glob(pattern):
                if any(x in str(file_path) for x in ["node_modules", ".venv", ".git"]):
                    continue

                result = analyzer.analyze_document(file_path)
                if "error" not in result:
                    results.append(result)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {"total_documents": len(results), "documents": results}

        output_path.write_text(json.dumps(output_data, indent=2))

        print("\n✅ Analysis complete")
        print(f"   Documents analyzed: {len(results)}")
        print(f"   Output: {output_path}")

    elif args.command == "test":
        print("🧪 Running test cases...")

        # Test case 1: Neutral scientific
        test1 = "The data suggests that parameter epsilon correlates with outcome CCI. Results indicate a moderate effect size. Further validation is required."
        result1 = analyzer.analyze_sentiment(test1)
        print(
            f"\nTest 1 (Neutral): sentiment={result1['sentiment_score']:.3f}, neutrality={result1['neutrality_score']:.3f}"
        )

        # Test case 2: Overly positive
        test2 = "This amazing study shows excellent results! The fantastic findings are wonderful and perfect in every way."
        result2 = analyzer.analyze_sentiment(test2)
        print(
            f"Test 2 (Positive): sentiment={result2['sentiment_score']:.3f}, neutrality={result2['neutrality_score']:.3f}"
        )

        # Test case 3: Tone analysis
        test3 = "Obviously, this revolutionary breakthrough is absolutely unprecedented. It's basically unbelievable!"
        result3 = analyzer.analyze_tone(test3)
        print(
            f"Test 3 (Tone): score={result3['tone_score']:.3f}, scientific={result3['is_scientific_tone']}"
        )

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
