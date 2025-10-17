#!/usr/bin/env python3
"""
TruthLens v2 - Causal Parser
Extracts causal relationships and detects contradictions

v2 Causal Consistency: Cause→Effect chain tracking
"""
import re


class CausalParser:
    """
    Parse causal relationships from text
    Detect contradictions in causal claims
    """

    def __init__(self):
        # Causal templates (cause → effect patterns)
        # ORDERED by priority: explicit connectives first, then modifiers
        self.causal_templates = [
            # If-then (highest priority - most explicit)
            (r"if\s+(\w+(?:\s+\w+){0,4})\s+then\s+(\w+(?:\s+\w+){0,4})", "if_then"),
            (r"when\s+(\w+(?:\s+\w+){0,4})\s+then\s+(\w+(?:\s+\w+){0,4})", "when_then"),
            # Because/since (effect because cause - reversed)
            (r"(\w+(?:\s+\w+){0,4})\s+because\s+(\w+(?:\s+\w+){0,4})", "because"),
            (r"(\w+(?:\s+\w+){0,4})\s+since\s+(\w+(?:\s+\w+){0,4})", "since"),
            (r"(\w+(?:\s+\w+){0,4})\s+due\s+to\s+(\w+(?:\s+\w+){0,4})", "due_to"),
            # Therefore/thus (cause therefore effect)
            (r"(\w+(?:\s+\w+){0,4})\s+therefore\s+(\w+(?:\s+\w+){0,4})", "therefore"),
            (r"(\w+(?:\s+\w+){0,4})\s+thus\s+(\w+(?:\s+\w+){0,4})", "thus"),
            (r"(\w+(?:\s+\w+){0,4})\s+hence\s+(\w+(?:\s+\w+){0,4})", "hence"),
            (
                r"(\w+(?:\s+\w+){0,4})\s+consequently\s+(\w+(?:\s+\w+){0,4})",
                "consequently",
            ),
            # Direct causation
            (r"(\w+(?:\s+\w+){0,4})\s+causes?\s+(\w+(?:\s+\w+){0,4})", "causes"),
            (r"(\w+(?:\s+\w+){0,4})\s+leads?\s+to\s+(\w+(?:\s+\w+){0,4})", "leads_to"),
            (
                r"(\w+(?:\s+\w+){0,4})\s+results?\s+in\s+(\w+(?:\s+\w+){0,4})",
                "results_in",
            ),
            (r"(\w+(?:\s+\w+){0,4})\s+produces?\s+(\w+(?:\s+\w+){0,4})", "produces"),
            (r"(\w+(?:\s+\w+){0,4})\s+generates?\s+(\w+(?:\s+\w+){0,4})", "generates"),
            # Effect modifiers (lowest priority - can match many things)
            (
                r"(\w+(?:\s+\w+){0,4})\s+(increases?|enhances?|improves?|boosts?|strengthens?|promotes?)\s+(\w+(?:\s+\w+){0,4})",
                "positive_effect",
            ),
            (
                r"(\w+(?:\s+\w+){0,4})\s+(decreases?|reduces?|weakens?|harms?|undermines?|inhibits?)\s+(\w+(?:\s+\w+){0,4})",
                "negative_effect",
            ),
        ]

        # Negation markers
        self.negation_markers = {
            "not",
            "no",
            "n't",
            "never",
            "neither",
            "nor",
            "without",
            "lack",
            "absent",
            "fails to",
            "does not",
            "cannot",
            "won't",
            "didn't",
            "doesn't",
            "don't",
        }

        # Effect modifiers (increases/decreases)
        self.effect_modifiers = {
            "positive": {
                "increases",
                "enhances",
                "improves",
                "boosts",
                "strengthens",
                "promotes",
            },
            "negative": {
                "decreases",
                "reduces",
                "weakens",
                "harms",
                "undermines",
                "inhibits",
            },
        }

    def detect_negation(self, text: str) -> bool:
        """
        Detect if statement contains negation

        Args:
            text: Text to check

        Returns:
            True if negation detected
        """
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        return bool(words & self.negation_markers)

    def extract_effect_polarity(self, effect_text: str) -> str:
        """
        Determine if effect is positive, negative, or neutral

        Args:
            effect_text: Effect part of causal statement

        Returns:
            'positive', 'negative', or 'neutral'
        """
        effect_lower = effect_text.lower()

        for word in self.effect_modifiers["positive"]:
            if word in effect_lower:
                return "positive"

        for word in self.effect_modifiers["negative"]:
            if word in effect_lower:
                return "negative"

        return "neutral"

    def parse_causal_statement(self, sentence: str) -> list[dict]:
        """
        Parse a sentence for causal relationships

        Args:
            sentence: Single sentence to parse

        Returns:
            List of causal pairs found
        """
        causal_pairs = []

        # Try each template
        for pattern, relation_type in self.causal_templates:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)

            for match in matches:
                # Handle patterns with modifiers (group 2 is the modifier)
                if relation_type in ["positive_effect", "negative_effect"]:
                    cause = match.group(1).strip()
                    modifier = match.group(2).strip()
                    effect = match.group(3).strip()

                    # Extract polarity from relation type
                    polarity = (
                        "positive" if relation_type == "positive_effect" else "negative"
                    )
                else:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    modifier = None
                    polarity = None

                # Handle reversed relations (because/since)
                if relation_type in ["because", "since", "due_to"]:
                    # Effect because cause → swap
                    cause, effect = effect, cause

                # Detect negation
                has_negation = self.detect_negation(sentence)

                # Determine effect polarity (if not already set by template)
                if polarity is None:
                    polarity = self.extract_effect_polarity(
                        effect if modifier is None else modifier + " " + effect
                    )

                causal_pairs.append(
                    {
                        "cause": cause.lower(),
                        "effect": effect.lower(),
                        "relation": relation_type,
                        "polarity": polarity,
                        "negated": has_negation,
                        "original_text": sentence[:100],  # Truncate for readability
                    }
                )

        return causal_pairs

    def parse_document(self, text: str, k_window: int = 3) -> dict:
        """
        Parse entire document for causal relationships

        Args:
            text: Full document text
            k_window: Sentence window for pairwise checks

        Returns:
            Dictionary with causal pairs and analysis
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        all_causal_pairs = []

        # Extract causal pairs from each sentence
        for i, sentence in enumerate(sentences):
            pairs = self.parse_causal_statement(sentence)
            for pair in pairs:
                pair["sentence_index"] = i
                all_causal_pairs.append(pair)

        return {
            "causal_pairs": all_causal_pairs,
            "causal_pair_count": len(all_causal_pairs),
            "sentences_analyzed": len(sentences),
        }

    def detect_contradiction(self, pair1: dict, pair2: dict) -> dict:
        """
        Check if two causal pairs contradict each other

        Args:
            pair1: First causal pair
            pair2: Second causal pair

        Returns:
            Contradiction analysis or None if no contradiction
        """
        # Same cause?
        cause_match = self._similar_terms(pair1["cause"], pair2["cause"])

        if not cause_match:
            return None

        # Same effect?
        effect_match = self._similar_terms(pair1["effect"], pair2["effect"])

        if not effect_match:
            return None

        # Check for contradictory polarities or negations
        contradictory = False
        reason = ""

        # Negation contradiction (one negated, one not)
        if pair1["negated"] != pair2["negated"]:
            contradictory = True
            reason = "One statement negates the relationship"

        # Polarity contradiction (both non-neutral but opposing)
        if (
            not contradictory
            and pair1["polarity"] != "neutral"
            and pair2["polarity"] != "neutral"
        ):
            if pair1["polarity"] != pair2["polarity"]:
                contradictory = True
                reason = (
                    f"Opposing polarities: {pair1['polarity']} vs {pair2['polarity']}"
                )

        # Check if relation types themselves indicate contradiction
        if not contradictory:
            if pair1["relation"] in ["positive_effect"] and pair2["relation"] in [
                "negative_effect"
            ]:
                contradictory = True
                reason = f"Opposing relation types: {pair1['relation']} vs {pair2['relation']}"
            elif pair1["relation"] in ["negative_effect"] and pair2["relation"] in [
                "positive_effect"
            ]:
                contradictory = True
                reason = f"Opposing relation types: {pair1['relation']} vs {pair2['relation']}"

        # Check for polarized words in effect text
        if not contradictory:
            # Look for harms/leads in pair texts
            effect1_text = pair1.get("original_text", "").lower()
            effect2_text = pair2.get("original_text", "").lower()

            positive_words = [
                "leads to",
                "promotes",
                "enhances",
                "improves",
                "strengthens",
                "boosts",
            ]
            negative_words = [
                "harms",
                "damages",
                "undermines",
                "weakens",
                "inhibits",
                "reduces",
            ]

            has_positive_1 = any(word in effect1_text for word in positive_words)
            has_negative_1 = any(word in effect1_text for word in negative_words)
            has_positive_2 = any(word in effect2_text for word in positive_words)
            has_negative_2 = any(word in effect2_text for word in negative_words)

            if (has_positive_1 and has_negative_2) or (
                has_negative_1 and has_positive_2
            ):
                contradictory = True
                reason = "Opposing effect indicators in text"

        if contradictory:
            return {
                "pair1": pair1,
                "pair2": pair2,
                "reason": reason,
                "type": "causal_contradiction",
            }

        return None

    def _similar_terms(self, term1: str, term2: str, threshold: float = 0.5) -> bool:
        """
        Check if two terms are similar (simple word overlap)

        Args:
            term1: First term
            term2: Second term
            threshold: Similarity threshold (lowered to 0.5 for better matching)

        Returns:
            True if terms are similar
        """
        words1 = set(re.findall(r"\b\w+\b", term1.lower()))
        words2 = set(re.findall(r"\b\w+\b", term2.lower()))

        # Remove common stop words
        stop_words = {"the", "a", "an", "to", "of", "in", "on", "at"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return False

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union if union > 0 else 0

        # Also check for exact substring match
        if term1.lower() in term2.lower() or term2.lower() in term1.lower():
            return True

        return similarity >= threshold

    def find_contradictions(
        self, causal_pairs: list[dict], k_window: int = 3
    ) -> list[dict]:
        """
        Find contradictions within causal pairs

        Args:
            causal_pairs: List of causal pairs from parse_document
            k_window: Maximum sentence distance to check

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Pairwise comparison within window
        for i, pair1 in enumerate(causal_pairs):
            for j, pair2 in enumerate(causal_pairs[i + 1 :], start=i + 1):
                # Check if within window
                sent_distance = abs(pair1["sentence_index"] - pair2["sentence_index"])

                if sent_distance > k_window:
                    continue

                # Check for contradiction
                contradiction = self.detect_contradiction(pair1, pair2)

                if contradiction:
                    contradiction["sentence_distance"] = sent_distance
                    contradictions.append(contradiction)

        return contradictions

    def compute_causal_continuity_score(
        self, causal_pair_count: int, contradiction_count: int
    ) -> float:
        """
        Compute continuity score for causal reasoning

        Formula: 1 - (contradictions / max(1, causal_pairs))

        Args:
            causal_pair_count: Total causal pairs found
            contradiction_count: Contradictions detected

        Returns:
            Continuity score [0, 1]
        """
        if causal_pair_count == 0:
            return 1.0  # No causal pairs = no contradictions

        contradiction_rate = contradiction_count / causal_pair_count

        return max(0.0, 1.0 - contradiction_rate)


def main():
    """CLI for causal parser"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="TruthLens v2 Causal Parser")
    parser.add_argument("command", choices=["parse", "test"], help="Command to execute")
    parser.add_argument("--file", type=str, help="File to parse")
    parser.add_argument(
        "--output", type=str, default="contradiction_map.json", help="Output file path"
    )

    args = parser.parse_args()

    cp = CausalParser()

    if args.command == "parse":
        if not args.file:
            print("❌ Error: --file required")
            return

        from pathlib import Path

        file_path = Path(args.file)

        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        print(f"🔗 Parsing causal relationships: {file_path.name}")

        # Parse document
        parse_result = cp.parse_document(text)

        print("\n✅ Parsing complete:")
        print(f"   Sentences: {parse_result['sentences_analyzed']}")
        print(f"   Causal pairs: {parse_result['causal_pair_count']}")

        # Find contradictions
        contradictions = cp.find_contradictions(parse_result["causal_pairs"])

        print(f"   Contradictions: {len(contradictions)}")

        # Compute continuity
        continuity = cp.compute_causal_continuity_score(
            parse_result["causal_pair_count"], len(contradictions)
        )

        print(f"   Continuity score: {continuity:.3f}")

        # Show sample pairs
        if parse_result["causal_pairs"]:
            print("\n   Sample causal pairs:")
            for pair in parse_result["causal_pairs"][:5]:
                negation = " (negated)" if pair["negated"] else ""
                print(f"     • {pair['cause']} → {pair['effect']}{negation}")

        # Show contradictions
        if contradictions:
            print("\n   ⚠️  Contradictions detected:")
            for contr in contradictions[:3]:
                print(f"     • {contr['reason']}")
                print(
                    f"       Pair 1: {contr['pair1']['cause']} → {contr['pair1']['effect']}"
                )
                print(
                    f"       Pair 2: {contr['pair2']['cause']} → {contr['pair2']['effect']}"
                )

        # Save output
        output_data = {
            "file": str(file_path),
            "parse_result": parse_result,
            "contradictions": contradictions,
            "continuity_score_v2": continuity,
        }

        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\n✅ Saved: {output_path}")

    elif args.command == "test":
        print("🧪 Running causal parser tests...")

        # Test 1: Simple causal statement
        test1 = "Openness causes cooperation."
        pairs1 = cp.parse_causal_statement(test1)
        print(f"\nTest 1 (Simple causation): Found {len(pairs1)} pairs")
        if pairs1:
            print(f"   {pairs1[0]['cause']} → {pairs1[0]['effect']}")

        # Test 2: Negation
        test2 = "Openness does not cause cooperation."
        pairs2 = cp.parse_causal_statement(test2)
        print(f"\nTest 2 (Negation): Found {len(pairs2)} pairs")
        if pairs2:
            print(f"   Negated: {pairs2[0]['negated']}")

        # Test 3: Multiple relations
        test3 = "Cooperation leads to resilience because shared meaning enhances coordination."
        pairs3 = cp.parse_causal_statement(test3)
        print(f"\nTest 3 (Multiple relations): Found {len(pairs3)} pairs")
        for pair in pairs3:
            print(f"   {pair['cause']} → {pair['effect']} ({pair['relation']})")

        # Test 4: Contradiction detection
        test4a = "Openness increases cooperation."
        test4b = "Openness decreases cooperation."

        pairs4a = cp.parse_causal_statement(test4a)
        pairs4b = cp.parse_causal_statement(test4b)

        if pairs4a and pairs4b:
            # Add dummy sentence indices
            pairs4a[0]["sentence_index"] = 0
            pairs4b[0]["sentence_index"] = 1

            contr = cp.detect_contradiction(pairs4a[0], pairs4b[0])
            print("\nTest 4 (Contradiction detection):")
            print(
                f"   Pair 1: {pairs4a[0]['cause']} → {pairs4a[0]['effect']} (polarity: {pairs4a[0]['polarity']})"
            )
            print(
                f"   Pair 2: {pairs4b[0]['cause']} → {pairs4b[0]['effect']} (polarity: {pairs4b[0]['polarity']})"
            )
            print(f"   Contradiction: {'✅ Detected' if contr else '❌ Not detected'}")
            if contr:
                print(f"   Reason: {contr['reason']}")

        # Test 5: If-then
        test5 = "If cooperation increases then resilience improves."
        pairs5 = cp.parse_causal_statement(test5)
        print(f"\nTest 5 (If-then): Found {len(pairs5)} pairs")
        if pairs5:
            print(f"   {pairs5[0]['cause']} → {pairs5[0]['effect']}")

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
