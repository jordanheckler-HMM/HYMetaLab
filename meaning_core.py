#!/usr/bin/env python3
"""
MeaningForge v1 - Meaning Quotient (MQ) Core
Measures meaning potential through Relevance, Resonance, and Transformative Potential

v1 Genesis: MQ foundation with deterministic seeding
"""
import os
import re
from pathlib import Path

import numpy as np
import yaml

# v1 Genesis: Deterministic seed (overridable via env)
MEANINGFORGE_SEED = int(os.getenv("MEANINGFORGE_SEED", "42"))
np.random.seed(MEANINGFORGE_SEED)


class MeaningCore:
    """
    Calculate Meaning Quotient (MQ) from text

    MQ = f(Relevance, Resonance, Transformative_Potential)

    Components:
    - Relevance: Contextual applicability and practical value
    - Resonance: Emotional/human connection depth
    - Transformative Potential: Capacity to shift perspective/behavior
    """

    def __init__(self, config_path: Path = None, config: dict = None):
        """
        Initialize MeaningCore

        Args:
            config_path: Path to YAML config file
            config: Direct config dictionary
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            # Default configuration
            self.config = {
                "version": "1.0-genesis",
                "seed": MEANINGFORGE_SEED,
                "weights": {
                    "relevance": 0.35,
                    "resonance": 0.30,
                    "transformative_potential": 0.35,
                },
                "thresholds": {"min_acceptable": 0.60, "good": 0.75, "excellent": 0.85},
            }

        # Extract weights
        self.weights = self.config.get("weights", {})

        # Relevance indicators (practical applicability)
        self.relevance_keywords = [
            "practical",
            "application",
            "use",
            "implement",
            "apply",
            "utilize",
            "real-world",
            "actionable",
            "concrete",
            "specific",
            "example",
            "case",
            "scenario",
            "situation",
            "context",
            "relevant",
        ]

        # Resonance indicators (human connection)
        self.resonance_keywords = [
            "experience",
            "feel",
            "sense",
            "understand",
            "relate",
            "connect",
            "resonate",
            "meaningful",
            "significant",
            "important",
            "value",
            "purpose",
            "identity",
            "belonging",
            "shared",
            "common",
            "human",
            "personal",
            "story",
            "narrative",
            "insight",
        ]

        # Transformative potential indicators
        self.transformative_keywords = [
            "transform",
            "change",
            "shift",
            "evolve",
            "develop",
            "grow",
            "breakthrough",
            "insight",
            "realization",
            "awakening",
            "discovery",
            "paradigm",
            "revolution",
            "innovation",
            "new",
            "novel",
            "perspective",
            "understanding",
            "awareness",
            "consciousness",
            "enlighten",
        ]

        # Question patterns (indicate exploration/depth)
        self.question_patterns = [
            r"\?",  # Question marks
            r"\bwhy\b",
            r"\bhow\b",
            r"\bwhat if\b",
            r"\bcould\b",
            r"\bmight\b",
        ]

        # Integration words (connections and coherence)
        self.integration_words = [
            "therefore",
            "thus",
            "hence",
            "because",
            "since",
            "leads to",
            "connects",
            "relates",
            "links",
            "integrates",
            "synthesizes",
            "combines",
            "unifies",
            "bridges",
            "weaves",
        ]

    def extract_relevance(self, text: str) -> dict:
        """
        Extract relevance score (practical applicability)

        Args:
            text: Document text

        Returns:
            Relevance metrics
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Count relevance keywords
        relevance_count = sum(
            1 for keyword in self.relevance_keywords if keyword in text_lower
        )

        # Density of relevance keywords
        relevance_density = relevance_count / max(1, word_count / 100)

        # Look for concrete examples
        example_patterns = [
            r"\bfor example\b",
            r"\bfor instance\b",
            r"\bsuch as\b",
            r"\be\.g\.\b",
            r"\bi\.e\.\b",
            r"\bconsider\b",
        ]
        example_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in example_patterns
        )

        # Relevance score (0-1)
        base_score = min(1.0, np.sqrt(relevance_density) / np.sqrt(5))
        example_bonus = min(0.2, example_count * 0.05)

        relevance_score = min(1.0, base_score + example_bonus)

        return {
            "relevance_score": relevance_score,
            "relevance_count": relevance_count,
            "example_count": example_count,
            "relevance_density": relevance_density,
        }

    def extract_resonance(self, text: str) -> dict:
        """
        Extract resonance score (human connection depth)

        Args:
            text: Document text

        Returns:
            Resonance metrics
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Count resonance keywords
        resonance_count = sum(
            1 for keyword in self.resonance_keywords if keyword in text_lower
        )

        # Density of resonance keywords
        resonance_density = resonance_count / max(1, word_count / 100)

        # Look for first-person perspective (we, our, I)
        first_person_count = len(re.findall(r"\b(we|our|I|us)\b", text_lower))

        # Look for questions (indicate depth of exploration)
        question_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in self.question_patterns
        )

        # Resonance score (0-1)
        base_score = min(1.0, np.sqrt(resonance_density) / np.sqrt(4))
        perspective_bonus = min(0.15, first_person_count * 0.01)
        question_bonus = min(0.15, question_count * 0.03)

        resonance_score = min(1.0, base_score + perspective_bonus + question_bonus)

        return {
            "resonance_score": resonance_score,
            "resonance_count": resonance_count,
            "first_person_count": first_person_count,
            "question_count": question_count,
            "resonance_density": resonance_density,
        }

    def extract_transformative_potential(self, text: str) -> dict:
        """
        Extract transformative potential (capacity to shift perspective)

        Args:
            text: Document text

        Returns:
            Transformative potential metrics
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Count transformative keywords
        transformative_count = sum(
            1 for keyword in self.transformative_keywords if keyword in text_lower
        )

        # Density of transformative keywords
        transformative_density = transformative_count / max(1, word_count / 100)

        # Look for integration/synthesis language
        integration_count = sum(
            1 for word in self.integration_words if word in text_lower
        )

        # Look for meta-level language (thinking about thinking)
        meta_patterns = [
            r"\bthink about\b",
            r"\bconsider\b",
            r"\breflect\b",
            r"\bimagine\b",
            r"\benvision\b",
            r"\bconceive\b",
        ]
        meta_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in meta_patterns
        )

        # Transformative score (0-1)
        base_score = min(1.0, np.sqrt(transformative_density) / np.sqrt(5))
        integration_bonus = min(0.2, integration_count * 0.02)
        meta_bonus = min(0.15, meta_count * 0.03)

        transformative_score = min(1.0, base_score + integration_bonus + meta_bonus)

        return {
            "transformative_score": transformative_score,
            "transformative_count": transformative_count,
            "integration_count": integration_count,
            "meta_count": meta_count,
            "transformative_density": transformative_density,
        }

    def compute_meaning_quotient(self, text: str) -> dict:
        """
        Compute Meaning Quotient (MQ) for text

        Args:
            text: Document text

        Returns:
            Dictionary with MQ and component scores
        """
        # Extract components
        relevance = self.extract_relevance(text)
        resonance = self.extract_resonance(text)
        transformative = self.extract_transformative_potential(text)

        # Compute MQ
        mq = (
            self.weights.get("relevance", 0.35) * relevance["relevance_score"]
            + self.weights.get("resonance", 0.30) * resonance["resonance_score"]
            + self.weights.get("transformative_potential", 0.35)
            * transformative["transformative_score"]
        )

        # Clip to [0, 1]
        mq = max(0.0, min(1.0, mq))

        # Grade MQ
        thresholds = self.config.get("thresholds", {})
        if mq >= thresholds.get("excellent", 0.85):
            grade = "EXCELLENT"
        elif mq >= thresholds.get("good", 0.75):
            grade = "GOOD"
        elif mq >= thresholds.get("min_acceptable", 0.60):
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS_IMPROVEMENT"

        return {
            "meaning_quotient": mq,
            "grade": grade,
            "relevance": relevance["relevance_score"],
            "resonance": resonance["resonance_score"],
            "transformative_potential": transformative["transformative_score"],
            "passes_threshold": mq >= thresholds.get("min_acceptable", 0.60),
            "components": {
                "relevance": relevance,
                "resonance": resonance,
                "transformative": transformative,
            },
        }

    def analyze_file(self, file_path: Path) -> dict:
        """
        Analyze a file and compute MQ

        Args:
            file_path: Path to file

        Returns:
            Analysis results
        """
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            result = self.compute_meaning_quotient(text)
            result["file"] = str(file_path)
            return result
        except Exception as e:
            return {
                "file": str(file_path),
                "error": str(e),
                "meaning_quotient": 0.0,
                "grade": "ERROR",
            }


def main():
    """CLI for testing meaning_core"""
    import argparse

    parser = argparse.ArgumentParser(description="MeaningForge v1 Core")
    parser.add_argument("command", choices=["test"], help="Command to execute")

    args = parser.parse_args()

    if args.command == "test":
        print("🔥 Testing MeaningForge Core...")

        mc = MeaningCore()

        # Test 1: High-meaning text
        high_meaning = """
        Consider how this practical insight transforms our understanding of human experience.
        When we apply these principles, we discover meaningful connections that resonate 
        deeply with our shared purpose. This breakthrough shifts our perspective on what's 
        possible, integrating new understanding with lived reality.
        
        For example, when communities embrace these values, they evolve together, creating 
        transformative change. The insight reveals how personal growth connects to collective 
        flourishing. What if we could harness this potential?
        """

        result1 = mc.compute_meaning_quotient(high_meaning)
        print("\n✅ High-meaning text:")
        print(f"   MQ: {result1['meaning_quotient']:.3f} ({result1['grade']})")
        print(f"   Relevance: {result1['relevance']:.3f}")
        print(f"   Resonance: {result1['resonance']:.3f}")
        print(f"   Transformative: {result1['transformative_potential']:.3f}")

        # Test 2: Low-meaning text
        low_meaning = "This is text. It has words. The end."

        result2 = mc.compute_meaning_quotient(low_meaning)
        print("\n✅ Low-meaning text:")
        print(f"   MQ: {result2['meaning_quotient']:.3f} ({result2['grade']})")

        # Test 3: Monotonicity (adding resonance)
        base_text = "The study shows practical applications."
        resonant_text = (
            base_text
            + " This insight resonates with human experience and transforms our understanding."
        )

        mq_base = mc.compute_meaning_quotient(base_text)["meaning_quotient"]
        mq_resonant = mc.compute_meaning_quotient(resonant_text)["meaning_quotient"]

        print("\n✅ Monotonicity test:")
        print(f"   Base MQ: {mq_base:.3f}")
        print(f"   With resonance: {mq_resonant:.3f}")
        print(f"   Increased: {'✅' if mq_resonant > mq_base else '❌'}")

        # Test 4: Determinism
        print("\n✅ Determinism test:")
        mq_scores = [
            mc.compute_meaning_quotient(high_meaning)["meaning_quotient"]
            for _ in range(5)
        ]
        std_dev = np.std(mq_scores)
        print(f"   Std dev (5 runs): {std_dev:.10f}")
        print(f"   Deterministic: {'✅' if std_dev < 0.001 else '❌'}")

        print("\n✅ Core tests complete")


if __name__ == "__main__":
    main()
