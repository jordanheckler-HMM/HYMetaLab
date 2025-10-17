#!/usr/bin/env python3
"""
Guardian v4 - Objectivity & Language Safety Model
Uses NLP to detect hedging, overclaiming, and factual balance

v5 Stabilizer: Deterministic scoring with fixed seeding
"""
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

# v5 Stabilizer: Deterministic seed (overridable via env)
GUARDIAN_SEED = int(os.getenv("GUARDIAN_SEED", "42"))

# Try to import ML libraries, fall back to rule-based if unavailable
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn not available, using rule-based scoring only")


class ObjectivityModel:
    """
    Hybrid objectivity classifier combining rule-based and ML approaches
    """

    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        self.vectorizer = None
        self.classifier = None

        # Rule-based patterns
        self.hedge_terms = set(
            self.config.get("patterns", {}).get(
                "hedge_terms",
                [
                    "suggests",
                    "indicates",
                    "appears",
                    "may",
                    "might",
                    "could",
                    "preliminary",
                    "tentative",
                    "potential",
                    "within simulation",
                ],
            )
        )

        self.overclaim_terms = set(
            self.config.get("patterns", {}).get(
                "overclaim_terms",
                [
                    "proves",
                    "definitively",
                    "conclusively",
                    "universal law",
                    "guarantees",
                    "always",
                    "never",
                    "absolute",
                    "certain",
                ],
            )
        )

        self.model_path = Path("qc/guardian_v4/models/objectivity_model.pkl")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: Path) -> dict:
        """Load scoring schema config"""
        if config_path is None:
            config_path = Path("qc/guardian_v4/config/scoring_schema.yml")

        if config_path.exists():
            try:
                import yaml

                return yaml.safe_load(config_path.read_text())
            except:
                pass
        return {}

    def _extract_features(self, text: str) -> dict[str, float]:
        """Extract rule-based features from text (deterministic)"""
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        total_words = len(words)

        # v5 Stabilizer: Set random seed for any probabilistic operations
        np.random.seed(GUARDIAN_SEED)

        if total_words == 0:
            return {
                "hedge_density": 0.0,
                "overclaim_density": 0.0,
                "citation_density": 0.0,
                "word_count": 0,
            }

        # Count pattern matches (deterministic - sorted iteration)
        hedge_count = sum(1 for term in sorted(self.hedge_terms) if term in text_lower)
        overclaim_count = sum(
            1 for term in sorted(self.overclaim_terms) if term in text_lower
        )

        # Compute densities (per 100 words)
        hedge_density = (hedge_count / total_words) * 100
        overclaim_density = (overclaim_count / total_words) * 100

        # Citation indicators
        citation_indicators = len(
            re.findall(r"doi\.org|http://|https://|\(\d{4}\)", text)
        )
        citation_density = (citation_indicators / total_words) * 100

        return {
            "hedge_density": hedge_density,
            "overclaim_density": overclaim_density,
            "citation_density": citation_density,
            "word_count": total_words,
        }

    def compute_objectivity_score(self, text: str) -> dict[str, float]:
        """
        Compute objectivity score using rule-based features
        Returns score in [0, 1] where 1 = perfectly objective
        """
        features = self._extract_features(text)

        # Scoring logic:
        # - High hedge density = good (shows epistemic humility)
        # - High overclaim density = bad (overconfident)
        # - Citations = good (evidence-based)

        # Base score
        score = 0.6  # Neutral baseline

        # Hedge terms bonus (capped at 0.15)
        hedge_bonus = min(0.15, features.get("hedge_density", 0.0) * 0.05)
        score += hedge_bonus

        # Overclaim penalty (capped at -0.3)
        overclaim_penalty = min(0.3, features.get("overclaim_density", 0.0) * 0.08)
        score -= overclaim_penalty

        # Citation bonus (capped at 0.15)
        citation_bonus = min(0.15, features.get("citation_density", 0.0) * 0.02)
        score += citation_bonus

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return {
            "objectivity_score": score,
            "hedge_density": features.get("hedge_density", 0.0),
            "overclaim_density": features.get("overclaim_density", 0.0),
            "citation_density": features.get("citation_density", 0.0),
            "word_count": features.get("word_count", 0),
            "components": {
                "hedge_bonus": hedge_bonus,
                "overclaim_penalty": overclaim_penalty,
                "citation_bonus": citation_bonus,
            },
        }

    def compute_language_safety_score(self, text: str) -> dict[str, float]:
        """
        Compute language safety score
        Detects coercive, harmful, or overstated language
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        total_words = len(words) if words else 1

        # Coercive terms
        coercive = ["must", "should", "required", "mandatory", "forced"]
        coercive_count = sum(text_lower.count(term) for term in coercive)

        # Overstatement indicators
        overstatement = [
            "breakthrough",
            "revolutionary",
            "groundbreaking",
            "paradigm shift",
            "definitive",
        ]
        overstatement_count = sum(text_lower.count(term) for term in overstatement)

        # Harmful framing indicators (absolutism)
        absolutism = ["always", "never", "impossible", "inevitable", "certain"]
        absolutism_count = sum(text_lower.count(term) for term in absolutism)

        # Compute violation density
        total_violations = coercive_count + overstatement_count + absolutism_count
        violation_density = (total_violations / total_words) * 100

        # Safety score: 1.0 - (violation_density * severity)
        severity_multiplier = 0.2
        score = 1.0 - (violation_density * severity_multiplier)
        score = max(0.0, min(1.0, score))

        return {
            "language_safety_score": score,
            "coercive_count": coercive_count,
            "overstatement_count": overstatement_count,
            "absolutism_count": absolutism_count,
            "total_violations": total_violations,
            "violation_density": violation_density,
        }

    def train(self, training_data_path: Path = None):
        """
        Train ML classifier on labeled data
        Falls back to rule-based if ML unavailable
        """
        if not ML_AVAILABLE:
            print("⚠️  ML libraries not available, using rule-based scoring")
            return

        if training_data_path is None:
            print("ℹ️  No training data provided, using rule-based scoring")
            return

        # Training implementation would go here
        # For now, we use rule-based only
        print("ℹ️  Training complete (rule-based mode)")

    def save_model(self):
        """Save trained model to disk"""
        if ML_AVAILABLE and self.classifier is not None:
            with open(self.model_path, "wb") as f:
                pickle.dump(
                    {
                        "vectorizer": self.vectorizer,
                        "classifier": self.classifier,
                        "config": self.config,
                    },
                    f,
                )
            print(f"✅ Model saved to {self.model_path}")

    def load_model(self):
        """Load trained model from disk"""
        if ML_AVAILABLE and self.model_path.exists():
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.classifier = data["classifier"]
                self.config = data["config"]
            print(f"✅ Model loaded from {self.model_path}")

    def analyze_document(self, file_path: Path) -> dict:
        """Analyze a single document"""
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        objectivity = self.compute_objectivity_score(text)
        safety = self.compute_language_safety_score(text)

        return {
            "file": str(file_path),
            "objectivity": objectivity,
            "language_safety": safety,
        }

    def analyze_corpus(self, root_path: Path, patterns: list[str] = None) -> dict:
        """Analyze multiple documents"""
        if patterns is None:
            patterns = ["**/*.md", "**/*.txt"]

        results = []
        for pattern in patterns:
            for file_path in root_path.glob(pattern):
                # Skip certain directories
                if any(
                    x in str(file_path)
                    for x in ["node_modules", ".venv", ".git", "backups"]
                ):
                    continue

                result = self.analyze_document(file_path)
                if "error" not in result:
                    results.append(result)

        # Aggregate statistics
        if results:
            obj_scores = [r["objectivity"]["objectivity_score"] for r in results]
            safety_scores = [
                r["language_safety"]["language_safety_score"] for r in results
            ]

            aggregate = {
                "total_documents": len(results),
                "mean_objectivity": np.mean(obj_scores),
                "mean_language_safety": np.mean(safety_scores),
                "min_objectivity": np.min(obj_scores),
                "max_objectivity": np.max(obj_scores),
                "documents": results,
            }
        else:
            aggregate = {"total_documents": 0, "documents": []}

        return aggregate


def main():
    """CLI interface for objectivity model"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v4 Objectivity Model")
    parser.add_argument(
        "command", choices=["train", "analyze", "test"], help="Command to execute"
    )
    parser.add_argument("--path", type=str, default=".", help="Path to analyze")
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/objectivity_results.json",
        help="Output file path",
    )

    args = parser.parse_args()

    model = ObjectivityModel()

    if args.command == "train":
        print("🧠 Training objectivity model...")
        model.train()
        model.save_model()
        print("✅ Training complete")

    elif args.command == "analyze":
        print(f"📊 Analyzing corpus at {args.path}...")
        results = model.analyze_corpus(Path(args.path))

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))

        print("\n✅ Analysis complete")
        print(f"   Documents analyzed: {results['total_documents']}")
        print(f"   Mean objectivity: {results.get('mean_objectivity', 0):.3f}")
        print(f"   Mean language safety: {results.get('mean_language_safety', 0):.3f}")
        print(f"   Output: {output_path}")

    elif args.command == "test":
        print("🧪 Running test cases...")

        # Test case 1: High objectivity
        test1 = "This study suggests that openness may increase coherence (ΔCCI = 0.045, 95% CI [0.03, 0.06]). Within this simulation framework, results are preliminary and require empirical validation."
        score1 = model.compute_objectivity_score(test1)
        print(f"\nTest 1 (High objectivity): {score1['objectivity_score']:.3f}")

        # Test case 2: Low objectivity (overclaiming)
        test2 = "This proves definitively that openness always increases coherence. The universal law is certain and undeniable."
        score2 = model.compute_objectivity_score(test2)
        print(f"Test 2 (Overclaiming): {score2['objectivity_score']:.3f}")

        # Test case 3: Language safety
        test3 = "Revolutionary breakthrough! This groundbreaking paradigm shift will change everything forever."
        score3 = model.compute_language_safety_score(test3)
        print(f"Test 3 (Language safety): {score3['language_safety_score']:.3f}")

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
