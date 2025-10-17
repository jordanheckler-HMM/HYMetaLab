#!/usr/bin/env python3
"""
OriginChain v1 - Emergence Quotient (EQ) Core
Measures emergence potential through Complexity, Novelty, and Interconnectedness

v1 Genesis: EQ foundation with deterministic seeding
"""
import os
import re
from pathlib import Path

import numpy as np
import yaml

# v1 Genesis: Deterministic seed (overridable via env)
ORIGINCHAIN_SEED = int(os.getenv("ORIGINCHAIN_SEED", "42"))
np.random.seed(ORIGINCHAIN_SEED)


class OriginChainCore:
    """
    Calculate Emergence Quotient (EQ) from text

    EQ = f(Complexity, Novelty, Interconnectedness)

    Components:
    - Complexity: Structural richness and layered understanding
    - Novelty: New patterns, ideas, and perspectives
    - Interconnectedness: Relational density and network effects
    """

    def __init__(self, config_path: Path = None, config: dict = None):
        """
        Initialize OriginChainCore

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
                "seed": ORIGINCHAIN_SEED,
                "weights": {
                    "complexity": 0.35,
                    "novelty": 0.30,
                    "interconnectedness": 0.35,
                },
                "thresholds": {"min_acceptable": 0.70, "good": 0.80, "excellent": 0.90},
            }

        # Extract weights
        self.weights = self.config.get("weights", {})

        # Complexity indicators (structural richness)
        self.complexity_keywords = [
            "complex",
            "nuanced",
            "multifaceted",
            "layered",
            "sophisticated",
            "intricate",
            "elaborate",
            "nested",
            "hierarchical",
            "structured",
            "dimensional",
            "multidimensional",
            "facet",
            "aspect",
            "dimension",
            "level",
            "layer",
            "depth",
            "spectrum",
            "gradient",
        ]

        # Novelty indicators (new patterns/ideas)
        self.novelty_keywords = [
            "new",
            "novel",
            "unprecedented",
            "innovative",
            "original",
            "creative",
            "unique",
            "fresh",
            "emerging",
            "nascent",
            "pioneer",
            "breakthrough",
            "revolutionary",
            "cutting-edge",
            "frontier",
            "unexplored",
            "untapped",
            "uncharted",
            "discovery",
            "invention",
        ]

        # Interconnectedness indicators (relational density)
        self.interconnection_keywords = [
            "connect",
            "link",
            "relate",
            "network",
            "web",
            "mesh",
            "interplay",
            "interaction",
            "interdependent",
            "interconnect",
            "feedback",
            "loop",
            "cycle",
            "reciprocal",
            "mutual",
            "system",
            "ecosystem",
            "synergy",
            "emergence",
            "holistic",
        ]

        # Evolution/emergence patterns
        self.emergence_patterns = [
            r"\bemerge[sd]?\b",
            r"\bevol(ve|ution)\b",
            r"\bunfold[s]?\b",
            r"\bself-\w+",
            r"\bauto-\w+",
            r"\bco-\w+",
            r"\b(bottom-up|top-down)\b",
            r"\bfeedback loop\b",
        ]

    def extract_complexity(self, text: str) -> dict:
        """
        Extract complexity score (structural richness)

        Args:
            text: Document text

        Returns:
            Complexity metrics
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Count complexity keywords
        complexity_count = sum(
            1 for keyword in self.complexity_keywords if keyword in text_lower
        )

        # Density of complexity keywords
        complexity_density = complexity_count / max(1, word_count / 100)

        # Structural complexity (nested clauses, sentence variety)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if sentences:
            # Sentence length variance (more variance = more complex structure)
            lengths = [len(s.split()) for s in sentences]
            length_variance = np.std(lengths) / (np.mean(lengths) + 1) if lengths else 0
            structural_bonus = min(0.2, length_variance * 0.1)
        else:
            structural_bonus = 0.0

        # Multi-level language (if-then, when-then, therefore)
        multi_level_patterns = [
            r"\bif\b.*\bthen\b",
            r"\bwhen\b.*\bthen\b",
            r"\btherefore\b",
        ]
        multi_level_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in multi_level_patterns
        )
        multi_level_bonus = min(0.15, multi_level_count * 0.03)

        # Complexity score (0-1)
        base_score = min(1.0, np.sqrt(complexity_density) / np.sqrt(4))
        complexity_score = min(1.0, base_score + structural_bonus + multi_level_bonus)

        return {
            "complexity_score": complexity_score,
            "complexity_count": complexity_count,
            "complexity_density": complexity_density,
            "structural_variance": length_variance if sentences else 0,
            "multi_level_count": multi_level_count,
        }

    def extract_novelty(self, text: str) -> dict:
        """
        Extract novelty score (new patterns/ideas)

        Args:
            text: Document text

        Returns:
            Novelty metrics
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Count novelty keywords
        novelty_count = sum(
            1 for keyword in self.novelty_keywords if keyword in text_lower
        )

        # Density of novelty keywords
        novelty_density = novelty_count / max(1, word_count / 100)

        # Look for emergence patterns
        emergence_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in self.emergence_patterns
        )

        # Question-driven exploration (indicates novelty-seeking)
        question_count = len(re.findall(r"\?", text))
        question_bonus = min(0.15, question_count * 0.02)

        # Novelty score (0-1)
        base_score = min(1.0, np.sqrt(novelty_density) / np.sqrt(4))
        emergence_bonus = min(0.2, emergence_count * 0.04)

        novelty_score = min(1.0, base_score + emergence_bonus + question_bonus)

        return {
            "novelty_score": novelty_score,
            "novelty_count": novelty_count,
            "novelty_density": novelty_density,
            "emergence_patterns": emergence_count,
            "question_count": question_count,
        }

    def extract_interconnectedness(self, text: str) -> dict:
        """
        Extract interconnectedness score (relational density)

        Args:
            text: Document text

        Returns:
            Interconnectedness metrics
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Count interconnection keywords
        interconnection_count = sum(
            1 for keyword in self.interconnection_keywords if keyword in text_lower
        )

        # Density of interconnection keywords
        interconnection_density = interconnection_count / max(1, word_count / 100)

        # Relational language (between, among, with, through)
        relational_patterns = [
            r"\bbetween\b",
            r"\bamong\b",
            r"\bwith\b",
            r"\bthrough\b",
            r"\bacross\b",
        ]
        relational_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in relational_patterns
        )
        relational_bonus = min(0.2, relational_count * 0.02)

        # Network/system thinking
        network_terms = ["network", "system", "web", "ecosystem", "feedback"]
        network_count = sum(1 for term in network_terms if term in text_lower)
        network_bonus = min(0.15, network_count * 0.05)

        # Interconnectedness score (0-1)
        base_score = min(1.0, np.sqrt(interconnection_density) / np.sqrt(5))
        interconnectedness_score = min(
            1.0, base_score + relational_bonus + network_bonus
        )

        return {
            "interconnectedness_score": interconnectedness_score,
            "interconnection_count": interconnection_count,
            "interconnection_density": interconnection_density,
            "relational_count": relational_count,
            "network_count": network_count,
        }

    def compute_emergence_quotient(self, text: str) -> dict:
        """
        Compute Emergence Quotient (EQ) for text

        Args:
            text: Document text

        Returns:
            Dictionary with EQ and component scores
        """
        # Extract components
        complexity = self.extract_complexity(text)
        novelty = self.extract_novelty(text)
        interconnectedness = self.extract_interconnectedness(text)

        # Compute EQ
        eq = (
            self.weights.get("complexity", 0.35) * complexity["complexity_score"]
            + self.weights.get("novelty", 0.30) * novelty["novelty_score"]
            + self.weights.get("interconnectedness", 0.35)
            * interconnectedness["interconnectedness_score"]
        )

        # Clip to [0, 1]
        eq = max(0.0, min(1.0, eq))

        # Grade EQ
        thresholds = self.config.get("thresholds", {})
        if eq >= thresholds.get("excellent", 0.90):
            grade = "EXCELLENT"
        elif eq >= thresholds.get("good", 0.80):
            grade = "GOOD"
        elif eq >= thresholds.get("min_acceptable", 0.70):
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS_IMPROVEMENT"

        return {
            "emergence_quotient": eq,
            "grade": grade,
            "complexity": complexity["complexity_score"],
            "novelty": novelty["novelty_score"],
            "interconnectedness": interconnectedness["interconnectedness_score"],
            "passes_threshold": eq >= thresholds.get("min_acceptable", 0.70),
            "components": {
                "complexity": complexity,
                "novelty": novelty,
                "interconnectedness": interconnectedness,
            },
        }

    def analyze_file(self, file_path: Path) -> dict:
        """
        Analyze a file and compute EQ

        Args:
            file_path: Path to file

        Returns:
            Analysis results
        """
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            result = self.compute_emergence_quotient(text)
            result["file"] = str(file_path)
            return result
        except Exception as e:
            return {
                "file": str(file_path),
                "error": str(e),
                "emergence_quotient": 0.0,
                "grade": "ERROR",
            }


def main():
    """CLI for testing originchain_core"""
    import argparse

    parser = argparse.ArgumentParser(description="OriginChain v1 Core")
    parser.add_argument("command", choices=["test"], help="Command to execute")

    args = parser.parse_args()

    if args.command == "test":
        print("🌱 Testing OriginChain Core...")

        oc = OriginChainCore()

        # Test 1: High-emergence text
        high_emergence = """
        Novel patterns emerge through complex interconnected systems that evolve organically.
        The multifaceted interplay between components creates unprecedented feedback loops.
        When self-organizing networks interact, they generate innovative emergent properties
        that transcend individual elements. This sophisticated web of relationships reveals
        layered dimensions of meaning across multiple levels of analysis.
        
        For example, ecosystems demonstrate how simple rules produce intricate behaviors
        through bottom-up emergence. What new patterns might unfold when we explore these
        uncharted territories of complexity and interconnection?
        """

        result1 = oc.compute_emergence_quotient(high_emergence)
        print("\n✅ High-emergence text:")
        print(f"   EQ: {result1['emergence_quotient']:.3f} ({result1['grade']})")
        print(f"   Complexity: {result1['complexity']:.3f}")
        print(f"   Novelty: {result1['novelty']:.3f}")
        print(f"   Interconnectedness: {result1['interconnectedness']:.3f}")

        # Test 2: Low-emergence text
        low_emergence = "This is basic text. It has simple words. The end."

        result2 = oc.compute_emergence_quotient(low_emergence)
        print("\n✅ Low-emergence text:")
        print(f"   EQ: {result2['emergence_quotient']:.3f} ({result2['grade']})")

        # Test 3: Monotonicity (adding interconnectedness)
        base_text = "Novel patterns show complex structures."
        interconnected_text = (
            base_text
            + " These systems interconnect through feedback loops that link multiple networks in synergistic relationships."
        )

        eq_base = oc.compute_emergence_quotient(base_text)["emergence_quotient"]
        eq_inter = oc.compute_emergence_quotient(interconnected_text)[
            "emergence_quotient"
        ]

        print("\n✅ Monotonicity test:")
        print(f"   Base EQ: {eq_base:.3f}")
        print(f"   With interconnectedness: {eq_inter:.3f}")
        print(f"   Increased: {'✅' if eq_inter > eq_base else '❌'}")

        # Test 4: Determinism
        print("\n✅ Determinism test:")
        eq_scores = [
            oc.compute_emergence_quotient(high_emergence)["emergence_quotient"]
            for _ in range(5)
        ]
        std_dev = np.std(eq_scores)
        print(f"   Std dev (5 runs): {std_dev:.10f}")
        print(f"   Deterministic: {'✅' if std_dev < 0.001 else '❌'}")

        print("\n✅ Core tests complete")


if __name__ == "__main__":
    main()
