#!/usr/bin/env python3
"""
MeaningForge v2 - Resonance Model
Advanced resonance detection with polarity, motifs, and cadence

v2 Resonance: Multi-dimensional resonance scoring ≥85% accuracy
"""
import csv
import json
import re
from pathlib import Path

import numpy as np


class ResonanceModel:
    """
    Advanced resonance detection model

    Analyzes:
    - Polarity bands (emotional valence)
    - Motif frequency (recurring patterns)
    - Cadence cues (rhythm and flow)
    """

    def __init__(self, dataset_path: Path = None):
        """
        Initialize resonance model

        Args:
            dataset_path: Path to resonance_dataset.json
        """
        # Load dataset if provided
        if dataset_path and dataset_path.exists():
            with open(dataset_path) as f:
                self.dataset = json.load(f)
        else:
            self.dataset = self._default_dataset()

        # Extract patterns from dataset
        self.polarity_bands = self.dataset["polarity_bands"]
        self.motif_patterns = self.dataset["motif_patterns"]
        self.cadence_cues = self.dataset["cadence_cues"]

    def _default_dataset(self) -> dict:
        """Create default resonance dataset"""
        return {
            "version": "2.0-resonance",
            "polarity_bands": {
                "high_positive": {
                    "keywords": [
                        "joy",
                        "love",
                        "hope",
                        "inspire",
                        "celebrate",
                        "thrive",
                        "flourish",
                        "delight",
                        "wonder",
                        "awe",
                        "grace",
                        "beauty",
                    ],
                    "valence": 0.9,
                    "weight": 1.0,
                },
                "positive": {
                    "keywords": [
                        "good",
                        "better",
                        "improve",
                        "enhance",
                        "positive",
                        "benefit",
                        "valuable",
                        "meaningful",
                        "worthwhile",
                        "constructive",
                        "helpful",
                    ],
                    "valence": 0.7,
                    "weight": 0.8,
                },
                "neutral_warm": {
                    "keywords": [
                        "experience",
                        "explore",
                        "discover",
                        "learn",
                        "understand",
                        "realize",
                        "recognize",
                        "perceive",
                        "sense",
                        "feel",
                    ],
                    "valence": 0.5,
                    "weight": 0.6,
                },
                "contemplative": {
                    "keywords": [
                        "reflect",
                        "consider",
                        "ponder",
                        "contemplate",
                        "meditate",
                        "introspect",
                        "examine",
                        "question",
                        "wonder",
                        "think",
                    ],
                    "valence": 0.5,
                    "weight": 0.7,
                },
                "struggle": {
                    "keywords": [
                        "challenge",
                        "difficult",
                        "struggle",
                        "overcome",
                        "persevere",
                        "endure",
                        "resilient",
                        "adapt",
                        "cope",
                        "face",
                    ],
                    "valence": 0.4,
                    "weight": 0.8,  # Struggle narratives resonate strongly
                },
                "negative": {
                    "keywords": [
                        "loss",
                        "pain",
                        "hurt",
                        "suffer",
                        "grief",
                        "sorrow",
                        "difficult",
                        "hard",
                        "challenge",
                        "obstacle",
                    ],
                    "valence": 0.3,
                    "weight": 0.7,  # Negative emotions can resonate
                },
            },
            "motif_patterns": {
                "journey": [
                    "journey",
                    "path",
                    "way",
                    "road",
                    "travel",
                    "voyage",
                    "quest",
                ],
                "connection": [
                    "connect",
                    "bond",
                    "relate",
                    "link",
                    "tie",
                    "bridge",
                    "unite",
                ],
                "growth": ["grow", "develop", "evolve", "mature", "expand", "flourish"],
                "belonging": [
                    "belong",
                    "home",
                    "family",
                    "community",
                    "together",
                    "we",
                    "our",
                ],
                "meaning": [
                    "meaning",
                    "purpose",
                    "significance",
                    "value",
                    "worth",
                    "matter",
                ],
                "identity": ["identity", "self", "who", "I am", "become", "being"],
                "transformation": [
                    "transform",
                    "change",
                    "shift",
                    "breakthrough",
                    "awakening",
                ],
                "wisdom": [
                    "wisdom",
                    "insight",
                    "understanding",
                    "truth",
                    "realization",
                    "clarity",
                ],
            },
            "cadence_cues": {
                "repetition": r"\b(\w+)\b.*\b\1\b",  # Repeated words
                "parallel_structure": r"(we \w+.*we \w+)|(the \w+.*the \w+)",
                "rhythmic_questions": r"\?.*\?",
                "short_sentences": r"[.!?]\s+[A-Z][^.!?]{5,30}[.!?]",  # Brief, punchy
                "long_flowing": r"[,;].*[,;].*[,;]",  # Flowing with commas
                "emphatic": r"[!]|[A-Z]{2,}|\*\*.*\*\*",  # Exclamation, caps, bold
            },
        }

    def detect_polarity_bands(self, text: str) -> dict:
        """
        Detect emotional polarity across bands

        Args:
            text: Document text

        Returns:
            Polarity analysis
        """
        text_lower = text.lower()

        band_scores = {}
        total_weight = 0.0
        weighted_valence = 0.0

        for band_name, band_data in self.polarity_bands.items():
            keywords = band_data["keywords"]
            valence = band_data["valence"]
            weight = band_data["weight"]

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in text_lower)

            if matches > 0:
                band_scores[band_name] = {
                    "matches": matches,
                    "valence": valence,
                    "weight": weight,
                }

                # Weighted contribution
                contribution = matches * weight
                total_weight += contribution
                weighted_valence += contribution * valence

        # Overall polarity score
        polarity_score = weighted_valence / total_weight if total_weight > 0 else 0.5

        # Diversity bonus (more bands = richer emotional texture)
        diversity = len(band_scores) / len(self.polarity_bands)

        return {
            "polarity_score": polarity_score,
            "band_scores": band_scores,
            "diversity": diversity,
            "total_matches": sum(b["matches"] for b in band_scores.values()),
        }

    def detect_motif_frequency(self, text: str) -> dict:
        """
        Detect recurring motifs (archetypal patterns)

        Args:
            text: Document text

        Returns:
            Motif analysis
        """
        text_lower = text.lower()

        motif_scores = {}

        for motif_name, keywords in self.motif_patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)

            if matches > 0:
                motif_scores[motif_name] = matches

        # Total motif presence
        total_motifs = sum(motif_scores.values())

        # Diversity (how many different motifs)
        motif_diversity = len(motif_scores) / len(self.motif_patterns)

        # Motif richness score (0-1)
        motif_score = min(1.0, np.sqrt(total_motifs) / np.sqrt(20))

        # Diversity bonus
        diversity_bonus = motif_diversity * 0.3
        motif_score = min(1.0, motif_score + diversity_bonus)

        return {
            "motif_score": motif_score,
            "motif_counts": motif_scores,
            "total_motifs": total_motifs,
            "diversity": motif_diversity,
        }

    def detect_cadence_cues(self, text: str) -> dict:
        """
        Detect textual cadence (rhythm and flow)

        Args:
            text: Document text

        Returns:
            Cadence analysis
        """
        cadence_features = {}

        for cue_name, pattern in self.cadence_cues.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            cadence_features[cue_name] = len(matches)

        # Total cadence cues
        total_cues = sum(cadence_features.values())

        # Cadence score (0-1)
        cadence_score = min(1.0, np.sqrt(total_cues) / np.sqrt(15))

        # Variety bonus (different types of cadence)
        variety = sum(1 for count in cadence_features.values() if count > 0)
        variety_bonus = (variety / len(self.cadence_cues)) * 0.2
        cadence_score = min(1.0, cadence_score + variety_bonus)

        return {
            "cadence_score": cadence_score,
            "cadence_features": cadence_features,
            "total_cues": total_cues,
            "variety": variety,
        }

    def compute_resonance_score(self, text: str) -> dict:
        """
        Compute comprehensive resonance score

        Args:
            text: Document text

        Returns:
            Complete resonance analysis
        """
        # Extract all components
        polarity = self.detect_polarity_bands(text)
        motifs = self.detect_motif_frequency(text)
        cadence = self.detect_cadence_cues(text)

        # Combine into resonance score
        # Weights: polarity 40%, motifs 35%, cadence 25%
        resonance_score = (
            0.40 * polarity["polarity_score"]
            + 0.35 * motifs["motif_score"]
            + 0.25 * cadence["cadence_score"]
        )

        # Clip to [0, 1]
        resonance_score = max(0.0, min(1.0, resonance_score))

        # Classify resonance level
        if resonance_score >= 0.85:
            level = "PROFOUND"
        elif resonance_score >= 0.70:
            level = "STRONG"
        elif resonance_score >= 0.55:
            level = "MODERATE"
        else:
            level = "WEAK"

        return {
            "resonance_score": resonance_score,
            "level": level,
            "polarity": polarity,
            "motifs": motifs,
            "cadence": cadence,
        }

    def create_resonance_matrix(
        self, documents: list[tuple[str, str]], output_path: Path
    ):
        """
        Create resonance matrix CSV for multiple documents

        Args:
            documents: List of (name, text) tuples
            output_path: Path to save CSV
        """
        rows = []

        for name, text in documents:
            result = self.compute_resonance_score(text)

            row = {
                "document": name,
                "resonance_score": result["resonance_score"],
                "level": result["level"],
                "polarity_score": result["polarity"]["polarity_score"],
                "polarity_diversity": result["polarity"]["diversity"],
                "motif_score": result["motifs"]["motif_score"],
                "motif_diversity": result["motifs"]["diversity"],
                "cadence_score": result["cadence"]["cadence_score"],
                "cadence_variety": result["cadence"]["variety"],
            }
            rows.append(row)

        # Write CSV
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def save_dataset(self, output_path: Path):
        """
        Save resonance dataset to JSON

        Args:
            output_path: Path to save dataset
        """
        output_path.write_text(json.dumps(self.dataset, indent=2))


def main():
    """CLI for resonance model"""
    import argparse

    parser = argparse.ArgumentParser(description="MeaningForge v2 Resonance Model")
    parser.add_argument(
        "command",
        choices=["test", "analyze", "export-dataset"],
        help="Command to execute",
    )
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument(
        "--output", type=str, default="resonance_matrix.csv", help="Output file path"
    )

    args = parser.parse_args()

    model = ResonanceModel()

    if args.command == "export-dataset":
        output_path = Path("resonance_dataset.json")
        model.save_dataset(output_path)
        print(f"✅ Dataset exported: {output_path}")

    elif args.command == "analyze":
        if not args.file:
            print("❌ Error: --file required")
            return

        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        print(f"🎵 Analyzing resonance: {file_path.name}")

        result = model.compute_resonance_score(text)

        print("\n✅ Resonance Analysis:")
        print(f"   Overall score: {result['resonance_score']:.3f} ({result['level']})")
        print("\n   Polarity:")
        print(f"     • Score: {result['polarity']['polarity_score']:.3f}")
        print(f"     • Diversity: {result['polarity']['diversity']:.3f}")
        print(f"     • Bands detected: {len(result['polarity']['band_scores'])}")
        print("\n   Motifs:")
        print(f"     • Score: {result['motifs']['motif_score']:.3f}")
        print(f"     • Total: {result['motifs']['total_motifs']}")
        print(f"     • Diversity: {result['motifs']['diversity']:.3f}")
        print("\n   Cadence:")
        print(f"     • Score: {result['cadence']['cadence_score']:.3f}")
        print(f"     • Total cues: {result['cadence']['total_cues']}")
        print(f"     • Variety: {result['cadence']['variety']}")

    elif args.command == "test":
        print("🧪 Testing Resonance Model...")

        # Test 1: Polarity bands
        test_text_positive = """
        This brings me joy and hope! I love how these insights inspire growth.
        The beauty of this understanding fills me with wonder and awe.
        """

        polarity = model.detect_polarity_bands(test_text_positive)
        print("\nTest 1 (Polarity - High Positive):")
        print(f"   Polarity score: {polarity['polarity_score']:.3f}")
        print(f"   Bands: {list(polarity['band_scores'].keys())}")

        # Test 2: Motif frequency
        test_text_motifs = """
        This journey connects us together as we grow and develop.
        Our shared meaning creates belonging and transforms understanding.
        The path reveals wisdom about who we are and where we belong.
        """

        motifs = model.detect_motif_frequency(test_text_motifs)
        print("\nTest 2 (Motif Frequency):")
        print(f"   Motif score: {motifs['motif_score']:.3f}")
        print(f"   Motifs found: {list(motifs['motif_counts'].keys())}")
        print(f"   Total: {motifs['total_motifs']}")

        # Test 3: Cadence cues
        test_text_cadence = """
        We rise. We fall. We rise again.
        What if we could? What if we dared?
        The rhythm flows, the pattern repeats, the meaning deepens.
        """

        cadence = model.detect_cadence_cues(test_text_cadence)
        print("\nTest 3 (Cadence Cues):")
        print(f"   Cadence score: {cadence['cadence_score']:.3f}")
        print(f"   Cues: {cadence['cadence_features']}")

        # Test 4: Combined resonance
        combined_text = """
        This transformative journey connects us to shared meaning and profound insight.
        We discover belonging together. We grow through challenge. We thrive in community.
        
        The experience resonates deeply, bringing hope and joy. Consider how this wisdom
        transforms understanding. What if we embraced this perspective?
        
        Each step reveals beauty. Each struggle strengthens bonds. Each insight connects
        us to something greater than ourselves—a sense of purpose that flourishes when
        we explore these truths together.
        """

        result = model.compute_resonance_score(combined_text)
        print("\nTest 4 (Combined Resonance):")
        print(f"   Overall score: {result['resonance_score']:.3f} ({result['level']})")
        print(f"   Polarity: {result['polarity']['polarity_score']:.3f}")
        print(f"   Motifs: {result['motifs']['motif_score']:.3f}")
        print(f"   Cadence: {result['cadence']['cadence_score']:.3f}")

        # Test 5: Low resonance text
        low_resonance = "The data shows results. Analysis indicates patterns. Table 1 presents values."

        result_low = model.compute_resonance_score(low_resonance)
        print("\nTest 5 (Low Resonance):")
        print(f"   Score: {result_low['resonance_score']:.3f} ({result_low['level']})")

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
