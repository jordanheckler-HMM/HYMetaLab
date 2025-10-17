#!/usr/bin/env python3
"""
TruthLens Core Implementation - Minimal Stub
Provides basic text quality scoring for testing purposes
"""


class TruthLensCore:
    """
    Minimal TruthLens implementation for test compatibility
    Returns deterministic scores based on text features
    """

    def __init__(self):
        self.weight_claim_clarity = 0.33
        self.weight_citation_presence = 0.33
        self.weight_causal_tokens = 0.34

    def compute_truth_index(self, text: str) -> dict:
        """
        Compute truth index for given text
        Returns deterministic score based on text features
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        if not text or len(text.strip()) == 0:
            return {"truth_index": 0.0, "error": "Empty text"}

        # Simple heuristics for demonstration
        score = 0.5  # baseline

        # Citation presence (simple check for brackets/references)
        if "[" in text and "]" in text:
            score += 0.1
        if any(word in text.lower() for word in ["et al", "doi", "http"]):
            score += 0.1

        # Clarity (penalize very short or very long texts)
        word_count = len(text.split())
        if 20 <= word_count <= 200:
            score += 0.1

        # Causal tokens
        causal_words = [
            "because",
            "therefore",
            "thus",
            "consequently",
            "leads to",
            "causes",
        ]
        if any(word in text.lower() for word in causal_words):
            score += 0.1

        # Hedge words (reduce score for excessive hedging)
        hedge_words = ["might", "possibly", "perhaps", "maybe"]
        hedge_count = sum(1 for word in hedge_words if word in text.lower())
        if hedge_count > 3:
            score -= 0.1

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return {
            "truth_index": score,
            "word_count": word_count,
            "has_citations": "[" in text and "]" in text,
        }
