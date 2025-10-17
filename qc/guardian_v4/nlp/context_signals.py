#!/usr/bin/env python3
"""
Guardian v6 - Context Signal Detector
Detects irony, sarcasm, and contextual ambiguities in research documents

v6 Context Engine: Advanced context reasoning
"""
import re
from pathlib import Path


class ContextSignalDetector:
    """
    Detect contextual signals that modify meaning:
    - Irony/sarcasm cues
    - Scare quotes (skepticism markers)
    - Polarity flips (negation, contradiction)
    - Hedging qualifiers
    """

    def __init__(self):
        # Irony/sarcasm indicators
        self.irony_phrases = {
            "yeah right",
            "sure thing",
            "of course",
            "obviously not",
            "as if",
            "clearly not",
            "so-called",
            "alleged",
            "supposedly",
            "apparently",
        }

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
        }

        # Contradiction indicators
        self.contradiction_indicators = {
            "however",
            "but",
            "yet",
            "although",
            "despite",
            "nevertheless",
            "nonetheless",
            "on the other hand",
            "conversely",
            "in contrast",
        }

        # Skepticism markers
        self.skepticism_markers = {
            "questionable",
            "dubious",
            "unclear",
            "uncertain",
            "controversial",
            "debated",
            "disputed",
        }

    def detect_scare_quotes(self, text: str) -> list[dict]:
        """
        Detect scare quotes (e.g., "so-called" evidence)
        Indicates skepticism or irony
        """
        # Pattern: "word" or 'word' used skeptically
        patterns = [
            r'"([^"]+)"',  # Double quotes
            r"'([^']+)'",  # Single quotes
        ]

        scare_quotes = []

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                quoted_text = match.group(1)
                position = match.start()

                # Check context for skepticism markers within 50 chars before
                context_start = max(0, position - 50)
                context = text[context_start:position].lower()

                is_skeptical = any(
                    marker in context
                    for marker in ["so-called", "alleged", "supposed", "claimed"]
                )

                if is_skeptical or quoted_text.lower() in [
                    "proven",
                    "confirmed",
                    "established",
                ]:
                    scare_quotes.append(
                        {
                            "text": quoted_text,
                            "position": position,
                            "type": "skeptical_quote",
                            "context_window": text[
                                context_start : min(len(text), position + 50)
                            ],
                        }
                    )

        return scare_quotes

    def detect_polarity_flips(self, text: str) -> list[dict]:
        """
        Detect polarity flips (negation + positive term, or contradiction)
        E.g., "not proven", "fails to confirm", "however, results show"
        """
        text_lower = text.lower()

        flips = []

        # Pattern: negation + positive claim within 5 words
        # E.g., "not proven", "does not confirm"
        positive_claims = [
            "proven",
            "confirmed",
            "established",
            "demonstrated",
            "showed",
        ]

        for claim in positive_claims:
            # Find all occurrences
            pattern = rf'\b(\w+\s+){0,5}({"|".join(self.negation_markers)})\s+(\w+\s+){{0,3}}{claim}'
            matches = re.finditer(pattern, text_lower)

            for match in matches:
                flips.append(
                    {
                        "type": "negation_flip",
                        "text": match.group(0),
                        "position": match.start(),
                        "interpretation": f'Negated claim: "{claim}"',
                    }
                )

        # Pattern: contradiction word followed by opposing claim
        for i, sentence in enumerate(re.split(r"[.!?]+", text)):
            sentence_lower = sentence.lower()

            has_contradiction = any(
                word in sentence_lower for word in self.contradiction_indicators
            )

            if has_contradiction:
                flips.append(
                    {
                        "type": "contradiction",
                        "text": sentence.strip()[:100],  # First 100 chars
                        "position": i,
                        "interpretation": "Contains contradicting statement",
                    }
                )

        return flips

    def detect_irony_cues(self, text: str) -> list[dict]:
        """
        Detect irony cues (phrases that signal non-literal meaning)
        """
        text_lower = text.lower()

        irony_cues = []

        for phrase in self.irony_phrases:
            if phrase in text_lower:
                # Find all occurrences
                pattern = re.escape(phrase)
                matches = re.finditer(pattern, text_lower)

                for match in matches:
                    irony_cues.append(
                        {
                            "type": "irony_phrase",
                            "phrase": phrase,
                            "position": match.start(),
                            "interpretation": "Potential non-literal meaning",
                        }
                    )

        return irony_cues

    def analyze_context_signals(self, text: str) -> dict:
        """
        Comprehensive context signal analysis
        Returns all detected contextual modifications
        """
        scare_quotes = self.detect_scare_quotes(text)
        polarity_flips = self.detect_polarity_flips(text)
        irony_cues = self.detect_irony_cues(text)

        # Compute context error rate (signals that modify straightforward interpretation)
        total_signals = len(scare_quotes) + len(polarity_flips) + len(irony_cues)

        # Approximate total claims (sentences with verbs)
        sentences = len(re.findall(r"[.!?]+", text))
        context_error_rate = total_signals / max(1, sentences)

        return {
            "scare_quotes": scare_quotes,
            "scare_quote_count": len(scare_quotes),
            "polarity_flips": polarity_flips,
            "polarity_flip_count": len(polarity_flips),
            "irony_cues": irony_cues,
            "irony_cue_count": len(irony_cues),
            "total_context_signals": total_signals,
            "total_sentences": sentences,
            "context_error_rate": context_error_rate,
            "passes_threshold": context_error_rate < 0.10,  # <10% error rate
        }

    def analyze_file(self, file_path: Path) -> dict:
        """Analyze a file for context signals"""
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        result = self.analyze_context_signals(text)
        result["file"] = str(file_path)

        return result


def main():
    """CLI interface for context signal detector"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Guardian v6 Context Signal Detector")
    parser.add_argument(
        "command", choices=["analyze", "test"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument(
        "--output",
        type=str,
        default="qc/guardian_v4/context_signals.json",
        help="Output file path",
    )

    args = parser.parse_args()

    detector = ContextSignalDetector()

    if args.command == "analyze":
        if not args.file:
            print("❌ Error: --file required")
            return

        print(f"🔍 Analyzing context signals in {args.file}...")
        result = detector.analyze_file(Path(args.file))

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print("\n✅ Analysis complete")
        print(f"   Scare quotes: {result['scare_quote_count']}")
        print(f"   Polarity flips: {result['polarity_flip_count']}")
        print(f"   Irony cues: {result['irony_cue_count']}")
        print(f"   Context error rate: {result['context_error_rate']*100:.1f}%")
        print(
            f"   Passes threshold (<10%): {'✅' if result['passes_threshold'] else '❌'}"
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\n   Output: {output_path}")

    elif args.command == "test":
        print("🧪 Running context signal tests...")

        # Test 1: Scare quotes
        test1 = 'The "proven" method was actually unreliable. So-called "experts" disagreed.'
        result1 = detector.detect_scare_quotes(test1)
        print(f"\nTest 1 (Scare quotes): Found {len(result1)} instances")

        # Test 2: Polarity flip
        test2 = (
            "The results were not proven. The study fails to confirm the hypothesis."
        )
        result2 = detector.detect_polarity_flips(test2)
        print(f"Test 2 (Polarity flips): Found {len(result2)} instances")

        # Test 3: Irony
        test3 = (
            "Yeah right, like that would ever work. Obviously not the best approach."
        )
        result3 = detector.detect_irony_cues(test3)
        print(f"Test 3 (Irony cues): Found {len(result3)} instances")

        # Test 4: Clean text (should have minimal signals)
        test4 = "The data suggests a moderate effect. Results indicate preliminary support for the hypothesis."
        result4 = detector.analyze_context_signals(test4)
        print("\nTest 4 (Clean text):")
        print(f"  Context error rate: {result4['context_error_rate']*100:.1f}%")
        print(f"  Passes threshold: {'✅' if result4['passes_threshold'] else '❌'}")

        print("\n✅ Tests complete")


if __name__ == "__main__":
    main()
