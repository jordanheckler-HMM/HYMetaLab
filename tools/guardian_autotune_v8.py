#!/usr/bin/env python3
"""
Guardian v8 - Auto-Tuner
Weekly retraining based on override patterns

v8 Auto-Calibration: Learn from human corrections
"""
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "qc" / "guardian_v4"))

try:
    from overrides.override_logger import OverrideLogger

    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False


class GuardianAutoTuner:
    """
    Automatically tune Guardian thresholds based on override history
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.config_dir = self.root / "qc" / "guardian_v4" / "config"
        self.thresholds_path = self.config_dir / "thresholds_v8.yml"
        self.schema_v5_path = self.config_dir / "scoring_schema_v5.yml"
        self.schema_v8_path = self.config_dir / "scoring_schema_v8.yml"

        if LOGGER_AVAILABLE:
            self.logger = OverrideLogger()
        else:
            self.logger = None

        # Load constraints
        self.load_thresholds()

    def load_thresholds(self):
        """Load threshold configuration"""
        if not self.thresholds_path.exists():
            self.thresholds = {}
            self.constraints = {
                "min_weight_change": -0.01,
                "max_weight_change": 0.01,
                "max_allowed_drift": 0.01,
            }
            return

        with open(self.thresholds_path) as f:
            config = yaml.safe_load(f)

        self.thresholds = config
        self.constraints = config.get("calibration", {}).get(
            "constraints",
            {
                "min_weight_change": -0.01,
                "max_weight_change": 0.01,
                "max_allowed_drift": 0.01,
            },
        )

    def analyze_overrides(self, days: int = 7) -> dict:
        """
        Analyze override patterns from last N days

        Returns:
            Analysis dictionary with suggested adjustments
        """
        if not self.logger:
            return {"error": "Override logger not available"}

        overrides = self.logger.get_overrides(days=days)

        if len(overrides) < self.thresholds.get("calibration", {}).get(
            "autotune", {}
        ).get("min_overrides_required", 5):
            return {
                "insufficient_data": True,
                "overrides_count": len(overrides),
                "min_required": 5,
            }

        # Group by component
        components = {}
        for override in overrides:
            comp = override["component"]
            if comp not in components:
                components[comp] = {
                    "count": 0,
                    "deltas": [],
                    "avg_delta": 0.0,
                    "std_delta": 0.0,
                }

            components[comp]["count"] += 1
            components[comp]["deltas"].append(override["delta"])

        # Compute statistics
        for comp, data in components.items():
            deltas = data["deltas"]
            data["avg_delta"] = sum(deltas) / len(deltas)

            if len(deltas) > 1:
                variance = sum((d - data["avg_delta"]) ** 2 for d in deltas) / (
                    len(deltas) - 1
                )
                data["std_delta"] = variance**0.5

            del data["deltas"]  # Don't need full list

        return {
            "overrides_analyzed": len(overrides),
            "days_analyzed": days,
            "components": components,
            "timestamp": datetime.now().isoformat(),
        }

    def suggest_adjustments(self, analysis: dict) -> dict:
        """
        Suggest weight adjustments based on override analysis

        Args:
            analysis: Output from analyze_overrides()

        Returns:
            Suggested adjustments dict
        """
        if "insufficient_data" in analysis:
            return analysis

        if "error" in analysis:
            return analysis

        suggestions = {}
        learning_rate = (
            self.thresholds.get("calibration", {})
            .get("autotune", {})
            .get("learning_rate", 0.5)
        )

        for comp, data in analysis.get("components", {}).items():
            avg_delta = data["avg_delta"]
            count = data["count"]

            # Suggest adjustment proportional to average delta
            # But scale by learning rate and constrain
            raw_adjustment = avg_delta * learning_rate

            # Apply constraints
            min_change = self.constraints.get("min_weight_change", -0.01)
            max_change = self.constraints.get("max_weight_change", 0.01)

            suggested_adjustment = max(min_change, min(max_change, raw_adjustment))

            # Calculate confidence based on sample size and consistency
            confidence = min(1.0, count / 10.0)  # More overrides = more confidence
            if data.get("std_delta", 0) > 0:
                confidence *= 1.0 - min(1.0, data["std_delta"] / abs(avg_delta + 0.001))

            suggestions[comp] = {
                "suggested_adjustment": suggested_adjustment,
                "confidence": confidence,
                "based_on_overrides": count,
                "avg_human_correction": avg_delta,
            }

        return {
            "suggestions": suggestions,
            "analysis": analysis,
            "learning_rate": learning_rate,
            "constraints": self.constraints,
        }

    def apply_adjustments(self, suggestions: dict, dry_run: bool = True) -> dict:
        """
        Apply suggested adjustments to scoring schema

        Args:
            suggestions: Output from suggest_adjustments()
            dry_run: If True, don't actually write files

        Returns:
            Updated schema and metadata
        """
        if not self.schema_v5_path.exists():
            return {"error": "Base schema (v5) not found"}

        # Load base schema (v5)
        with open(self.schema_v5_path) as f:
            schema = yaml.safe_load(f)

        # Apply adjustments
        adjustments_applied = {}
        total_drift = 0.0

        for comp, suggestion in suggestions.get("suggestions", {}).items():
            adjustment = suggestion["suggested_adjustment"]

            # Find component in schema
            if comp in [
                "objectivity_score",
                "transparency_index_v2",
                "language_safety_score",
                "sentiment_neutrality",
            ]:
                metric_def = schema.get("metrics", {}).get(comp, {})
                current_weight = metric_def.get("weight", 0.25)
                new_weight = current_weight + adjustment

                # Clamp to valid weight range (weights must sum to ~1.0)
                # Keep weights between 0.10 and 0.40 for safety
                new_weight = max(0.10, min(0.40, new_weight))

                # Record adjustment
                actual_adjustment = new_weight - current_weight
                adjustments_applied[comp] = {
                    "old_weight": current_weight,
                    "new_weight": new_weight,
                    "adjustment": actual_adjustment,
                    "confidence": suggestion["confidence"],
                }

                total_drift += abs(actual_adjustment)

                # Update schema
                if not dry_run:
                    schema["metrics"][comp]["weight"] = new_weight

        # Check drift constraint (per-component, not total)
        max_drift_per_component = self.constraints.get("max_allowed_drift", 0.01)
        within_drift_limit = all(
            abs(adj["adjustment"])
            <= max_drift_per_component + 1e-9  # Add tiny tolerance for FP
            for adj in adjustments_applied.values()
        )

        # Update metadata
        if not dry_run:
            schema["version"] = "8.0-auto"
            schema["name"] = "Guardian v8 Auto-Calibrated Schema"

            # Add calibration entry to changelog
            if "changelog" not in schema:
                schema["changelog"] = []

            schema["changelog"].insert(
                0,
                {
                    "version": "8.0-auto",
                    "date": datetime.now().isoformat(),
                    "changes": [
                        f"Auto-calibrated based on {suggestions.get('analysis', {}).get('overrides_analyzed', 0)} overrides",
                        f"Total drift: {total_drift:.4f} (limit: {max_drift:.4f})",
                        *[
                            f"{comp}: {adj['old_weight']:.3f} → {adj['new_weight']:.3f}"
                            for comp, adj in adjustments_applied.items()
                        ],
                    ],
                },
            )

            # Write new schema
            with open(self.schema_v8_path, "w") as f:
                yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

        return {
            "adjustments": adjustments_applied,
            "total_drift": total_drift,
            "max_allowed_drift": max_drift_per_component,
            "within_drift_limit": within_drift_limit,
            "dry_run": dry_run,
            "output_path": str(self.schema_v8_path) if not dry_run else None,
        }


def main():
    """CLI for auto-tuner"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v8 Auto-Tuner")
    parser.add_argument(
        "command",
        nargs="?",
        default="analyze",
        choices=["analyze", "suggest", "apply"],
        help="Command to execute",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Days of overrides to analyze"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show adjustments without applying"
    )

    args = parser.parse_args()

    tuner = GuardianAutoTuner()

    if args.command == "analyze":
        print("📊 Analyzing override patterns...")
        analysis = tuner.analyze_overrides(days=args.days)

        if "insufficient_data" in analysis:
            print(
                f"\n⚠️  Insufficient data: {analysis['overrides_count']} overrides (need ≥{analysis['min_required']})"
            )
            return

        if "error" in analysis:
            print(f"\n❌ Error: {analysis['error']}")
            return

        print("\n✅ Analysis complete:")
        print(f"   Overrides analyzed: {analysis['overrides_analyzed']}")
        print(f"   Days: {analysis['days_analyzed']}")

        print("\n   Components:")
        for comp, data in analysis.get("components", {}).items():
            print(f"     • {comp}: {data['count']} overrides")
            print(f"       Avg correction: {data['avg_delta']:+.3f}")
            print(f"       Std deviation: {data.get('std_delta', 0):.3f}")

    elif args.command == "suggest":
        print("💡 Generating adjustment suggestions...")
        analysis = tuner.analyze_overrides(days=args.days)
        suggestions = tuner.suggest_adjustments(analysis)

        if "insufficient_data" in suggestions:
            print(
                f"\n⚠️  Insufficient data: {suggestions['overrides_count']} overrides (need ≥{suggestions['min_required']})"
            )
            return

        print("\n✅ Suggestions:")
        for comp, sugg in suggestions.get("suggestions", {}).items():
            print(f"   • {comp}:")
            print(f"     Adjustment: {sugg['suggested_adjustment']:+.4f}")
            print(f"     Confidence: {sugg['confidence']:.2f}")
            print(f"     Based on: {sugg['based_on_overrides']} overrides")

    elif args.command == "apply":
        print("🔧 Applying auto-calibration...")

        if args.dry_run:
            print("   (DRY RUN - no files will be modified)")

        analysis = tuner.analyze_overrides(days=args.days)
        suggestions = tuner.suggest_adjustments(analysis)
        result = tuner.apply_adjustments(suggestions, dry_run=args.dry_run)

        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return

        print("\n✅ Calibration complete:")
        print(
            f"   Total drift: {result['total_drift']:.4f} (limit: {result['max_allowed_drift']:.4f})"
        )
        print(f"   Within limits: {'✅' if result['within_drift_limit'] else '❌'}")

        print("\n   Adjustments:")
        for comp, adj in result.get("adjustments", {}).items():
            print(
                f"     • {comp}: {adj['old_weight']:.3f} → {adj['new_weight']:.3f} ({adj['adjustment']:+.4f})"
            )
            print(f"       Confidence: {adj['confidence']:.2f}")

        if not args.dry_run:
            print(f"\n   Output: {result['output_path']}")
        else:
            print("\n   ℹ️  Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
