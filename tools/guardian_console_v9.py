#!/usr/bin/env python3
"""
Guardian v9 - Interactive Console
TUI for exploring Guardian reports with explanations

v9 Explainability: Interactive audit trail exploration
"""
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "qc" / "guardian_v4"))

try:
    from explain.explainer_v9 import GuardianExplainer

    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False


class GuardianConsole:
    """
    Interactive console for exploring Guardian validation results
    """

    def __init__(self, report_path: Path = None):
        if report_path is None:
            report_path = Path("qc/guardian_v4/guardian_report_v4.json")

        self.report_path = report_path
        self.report = None
        self.explanations = None

        if EXPLAINER_AVAILABLE:
            self.explainer = GuardianExplainer()
        else:
            self.explainer = None

    def load_report(self) -> bool:
        """Load Guardian report"""
        if not self.report_path.exists():
            print(f"❌ Report not found: {self.report_path}")
            return False

        try:
            self.report = json.load(open(self.report_path))
            print(f"✅ Loaded report: {self.report_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading report: {e}")
            return False

    def display_summary(self):
        """Display report summary"""
        if not self.report:
            print("❌ No report loaded")
            return

        print("\n" + "=" * 70)
        print("🛡️  GUARDIAN VALIDATION SUMMARY")
        print("=" * 70)

        file_name = self.report.get("file", "unknown")
        score = self.report.get("guardian_alignment_score", 0)
        risk = self.report.get("risk_assessment", {}).get("risk_level", "unknown")
        timestamp = self.report.get("timestamp", "unknown")

        print(f"\nFile: {file_name}")
        print(f"Guardian Score: {score:.1f}/100")
        print(f"Risk Level: {risk.upper()}")
        print(f"Validated: {timestamp}")

        # Display metrics
        metrics = self.report.get("metrics", {})
        print("\n📊 Core Metrics:")
        print(f"   • Objectivity:      {metrics.get('objectivity_score', 0):.2f}")
        print(f"   • Transparency v2:  {metrics.get('transparency_index_v2', 0):.2f}")
        print(f"   • Language Safety:  {metrics.get('language_safety_score', 0):.2f}")
        print(f"   • Sentiment:        {metrics.get('sentiment_neutrality', 0):+.2f}")

        # v6+ features
        if "claim_type_accuracy" in metrics:
            print("\n🧠 v6 Context Metrics:")
            print(f"   • Claim Accuracy:   {metrics.get('claim_type_accuracy', 0):.2f}")
            print(f"   • Evidence Coverage: {metrics.get('evidence_coverage', 0):.2f}")
            print(
                f"   • Context Error Rate: {metrics.get('context_error_rate', 0):.2f}"
            )

        # v7+ features
        if "continuity_score" in metrics:
            print("\n🔗 v7 Memory Metrics:")
            print(f"   • Continuity Score: {metrics.get('continuity_score', 0):.2f}")

        # v8+ features
        if "confidence_intervals" in self.report:
            print("\n📈 v8 Confidence Intervals:")
            ci = self.report["confidence_intervals"]
            for metric, intervals in ci.items():
                if not intervals.get("insufficient_data"):
                    print(
                        f"   • {metric}: [{intervals['lower']:.2f}, {intervals['upper']:.2f}]"
                    )

        print()

    def display_metric_details(self, metric_name: str):
        """Display detailed breakdown for a specific metric"""
        if not self.report:
            print("❌ No report loaded")
            return

        print("\n" + "=" * 70)
        print(f"📊 {metric_name.upper()} - DETAILED BREAKDOWN")
        print("=" * 70)

        metrics = self.report.get("metrics", {})
        detailed = self.report.get("detailed_metrics", {})

        # Find metric data
        metric_data = None
        score = metrics.get(metric_name, None)

        # Map metric names to detailed_metrics keys
        metric_map = {
            "objectivity_score": "objectivity",
            "transparency_index_v2": "transparency",
            "language_safety_score": "language_safety",
            "sentiment_neutrality": "sentiment",
        }

        if metric_name in metric_map:
            metric_data = detailed.get(metric_map[metric_name], {})

        if score is not None:
            print(f"\nScore: {score:.3f}")

            # Show components if available
            if metric_data and "components" in metric_data:
                print("\nComponents:")
                for comp_name, comp_value in metric_data["components"].items():
                    print(f"   • {comp_name}: {comp_value:.3f}")

            # Show confidence interval if available
            if "confidence_intervals" in self.report:
                ci = self.report["confidence_intervals"].get(metric_name, {})
                if ci and not ci.get("insufficient_data"):
                    print(
                        f"\n95% Confidence Interval: [{ci['lower']:.3f}, {ci['upper']:.3f}]"
                    )
                    width = ci["upper"] - ci["lower"]
                    print(f"Uncertainty: ±{width/2:.3f}")

        else:
            print(f"\n⚠️  Metric '{metric_name}' not found in report")

        print()

    def display_recommendations(self):
        """Display Guardian recommendations"""
        if not self.report:
            print("❌ No report loaded")
            return

        recommendations = self.report.get("recommendations", [])

        print("\n" + "=" * 70)
        print("💡 RECOMMENDATIONS")
        print("=" * 70)

        if not recommendations:
            print("\n✅ No specific recommendations - document meets standards")
        else:
            print()
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")

        print()

    def display_explanations(self):
        """Display human-readable explanations"""
        if not self.report:
            print("❌ No report loaded")
            return

        if not self.explainer:
            print("❌ Explainer not available")
            return

        # Load document text if available
        file_path = Path(self.report.get("file", ""))
        if not file_path.exists():
            print(f"⚠️  Original file not found: {file_path}")
            text = ""
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        # Generate explanations
        explanations = self.explainer.explain_all(self.report, text)

        print("\n" + "=" * 70)
        print("📖 HUMAN-READABLE EXPLANATIONS")
        print("=" * 70)

        for metric_name, explanation in explanations["explanations"].items():
            print(f"\n{'-'*70}")
            print(
                f"{metric_name.upper()}: {explanation['score']:.2f} (Grade: {explanation['grade']})"
            )
            print(f"{'-'*70}")
            print(explanation["explanation"])

        print()

    def interactive_menu(self):
        """Main interactive menu"""
        print("\n" + "=" * 70)
        print("🛡️  GUARDIAN CONSOLE v9")
        print("=" * 70)
        print("\n1. View Summary")
        print("2. View Metric Details")
        print("3. View Recommendations")
        print("4. View Explanations")
        print("5. Reload Report")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            self.display_summary()
        elif choice == "2":
            metric = input("Enter metric name (e.g., objectivity_score): ").strip()
            self.display_metric_details(metric)
        elif choice == "3":
            self.display_recommendations()
        elif choice == "4":
            self.display_explanations()
        elif choice == "5":
            self.load_report()
        elif choice == "6":
            print("\n👋 Goodbye!")
            return False
        else:
            print("\n⚠️  Invalid choice")

        return True  # Continue

    def run(self):
        """Run interactive console"""
        # Load report
        if not self.load_report():
            return

        # Show summary first
        self.display_summary()

        # Interactive loop
        while True:
            try:
                cont = self.interactive_menu()
                if not cont:
                    break
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v9 Interactive Console")
    parser.add_argument(
        "--report",
        type=str,
        default="qc/guardian_v4/guardian_report_v4.json",
        help="Path to Guardian report",
    )

    args = parser.parse_args()

    console = GuardianConsole(report_path=Path(args.report))
    console.run()


if __name__ == "__main__":
    main()
