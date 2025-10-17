#!/usr/bin/env python3
"""
Guardian v8 - Override Logger
Records manual overrides for adaptive calibration

v8 Auto-Calibration: Learn from human corrections
"""
import json
from datetime import datetime
from pathlib import Path


class OverrideLogger:
    """
    Log manual overrides to Guardian validation results
    Used for weekly auto-tuning
    """

    def __init__(self, log_path: Path = None):
        if log_path is None:
            log_path = Path(__file__).parent / "overrides_log.jsonl"

        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_override(
        self,
        file: str,
        component: str,
        before_score: float,
        after_score: float,
        reason: str,
        lab: str = "general",
        operator: str = "manual",
    ) -> None:
        """
        Log a manual override

        Args:
            file: Path to file that was overridden
            component: Which component was adjusted (e.g., 'objectivity_score')
            before_score: Original Guardian score
            after_score: Human-corrected score
            reason: Why the override was necessary
            lab: Lab context (openlight, opentime, openmind, etc.)
            operator: Who made the override
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "file": str(file),
            "component": component,
            "before_score": before_score,
            "after_score": after_score,
            "delta": after_score - before_score,
            "reason": reason,
            "lab": lab,
            "operator": operator,
        }

        # Append to JSONL
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_overrides(
        self,
        component: str | None = None,
        lab: str | None = None,
        days: int | None = None,
    ) -> list:
        """
        Retrieve overrides with optional filters

        Args:
            component: Filter by component name
            lab: Filter by lab
            days: Only get overrides from last N days

        Returns:
            List of override entries
        """
        if not self.log_path.exists():
            return []

        overrides = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)

                    # Apply filters
                    if component and entry["component"] != component:
                        continue

                    if lab and entry["lab"] != lab:
                        continue

                    if days:
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if (datetime.now() - entry_time).days > days:
                            continue

                    overrides.append(entry)

        return overrides

    def get_stats(self, days: int = 30) -> dict:
        """Get override statistics for last N days"""
        overrides = self.get_overrides(days=days)

        if not overrides:
            return {"total_overrides": 0, "overrides_per_day": 0.0, "components": {}}

        # Count by component
        components = {}
        for override in overrides:
            comp = override["component"]
            if comp not in components:
                components[comp] = {"count": 0, "avg_delta": 0.0, "deltas": []}
            components[comp]["count"] += 1
            components[comp]["deltas"].append(override["delta"])

        # Compute averages
        for comp, data in components.items():
            data["avg_delta"] = sum(data["deltas"]) / len(data["deltas"])
            del data["deltas"]  # Don't need full list in stats

        return {
            "total_overrides": len(overrides),
            "overrides_per_day": len(overrides) / days,
            "days_analyzed": days,
            "components": components,
        }


def main():
    """CLI for override logger"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v8 Override Logger")
    parser.add_argument(
        "command", choices=["log", "stats", "list"], help="Command to execute"
    )
    parser.add_argument("--file", type=str, help="File path")
    parser.add_argument("--component", type=str, help="Component name")
    parser.add_argument("--before", type=float, help="Before score")
    parser.add_argument("--after", type=float, help="After score")
    parser.add_argument("--reason", type=str, help="Override reason")
    parser.add_argument("--lab", type=str, default="general", help="Lab context")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")

    args = parser.parse_args()

    logger = OverrideLogger()

    if args.command == "log":
        if not all([args.file, args.component, args.before, args.after, args.reason]):
            print(
                "❌ Error: log requires --file, --component, --before, --after, --reason"
            )
            return

        logger.log_override(
            file=args.file,
            component=args.component,
            before_score=args.before,
            after_score=args.after,
            reason=args.reason,
            lab=args.lab,
        )

        print(
            f"✅ Override logged: {args.component} {args.before:.2f} → {args.after:.2f}"
        )
        print(f"   File: {args.file}")
        print(f"   Reason: {args.reason}")

    elif args.command == "stats":
        stats = logger.get_stats(days=args.days)

        print(f"📊 Override Statistics (last {args.days} days):\n")
        print(f"   Total overrides: {stats['total_overrides']}")
        print(f"   Overrides/day: {stats['overrides_per_day']:.2f}")

        if stats["components"]:
            print("\n   By component:")
            for comp, data in stats["components"].items():
                print(
                    f"     • {comp}: {data['count']} overrides (avg Δ: {data['avg_delta']:+.3f})"
                )

    elif args.command == "list":
        overrides = logger.get_overrides(
            component=args.component, lab=args.lab, days=args.days
        )

        print(f"📋 Overrides (last {args.days} days):\n")
        for i, override in enumerate(overrides[-20:], 1):  # Last 20
            print(
                f"{i}. {override['component']}: {override['before_score']:.2f} → {override['after_score']:.2f}"
            )
            print(f"   File: {override['file']}")
            print(f"   Reason: {override['reason']}")
            print()


if __name__ == "__main__":
    main()
