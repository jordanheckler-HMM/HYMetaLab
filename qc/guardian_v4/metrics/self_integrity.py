#!/usr/bin/env python3
"""
Guardian v10 - Self-Integrity Monitor
Tracks Guardian's own consistency and rule change verification

v10 Meta-Audit: Self-auditing and internal consistency
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml


class SelfIntegrityMonitor:
    """
    Monitor Guardian's self-consistency across versions and time
    Compute Self-Integrity Score
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.config_dir = self.root / "qc" / "guardian_v4" / "config"
        self.dualrun_dir = self.root / "qc" / "guardian_v4" / "dualrun"

        self.metadata_path = self.config_dir / "self_integrity_metadata.json"

        # Load or initialize metadata
        self.load_metadata()

    def load_metadata(self):
        """Load self-integrity metadata"""
        if self.metadata_path.exists():
            try:
                self.metadata = json.load(open(self.metadata_path))
            except Exception:
                self.metadata = self._init_metadata()
        else:
            self.metadata = self._init_metadata()

    def _init_metadata(self) -> dict:
        """Initialize metadata structure"""
        return {
            "version": "10.0",
            "initialized_at": datetime.now().isoformat(),
            "schema_hashes": {},
            "change_log": [],
            "self_integrity_history": [],
            "unverified_changes": 0,
        }

    def compute_schema_hash(self, schema_path: Path) -> str:
        """Compute SHA256 hash of schema file"""
        if not schema_path.exists():
            return "missing"

        try:
            content = schema_path.read_text()
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            return "error"

    def check_schema_change(self, schema_name: str) -> dict:
        """
        Check if schema has changed and if change is logged

        Args:
            schema_name: Name of schema file (e.g., 'scoring_schema_v5.yml')

        Returns:
            Change detection results
        """
        schema_path = self.config_dir / schema_name
        current_hash = self.compute_schema_hash(schema_path)

        # Get previous hash
        previous_hash = self.metadata["schema_hashes"].get(schema_name)

        if previous_hash is None:
            # First time seeing this schema
            self.metadata["schema_hashes"][schema_name] = current_hash
            self.save_metadata()

            return {
                "schema": schema_name,
                "changed": False,
                "first_time": True,
                "hash": current_hash,
            }

        if current_hash == previous_hash:
            # No change
            return {"schema": schema_name, "changed": False, "hash": current_hash}

        # Schema has changed - check if logged
        change_logged = self._is_change_logged(schema_name, current_hash)

        return {
            "schema": schema_name,
            "changed": True,
            "previous_hash": previous_hash[:8],
            "current_hash": current_hash[:8],
            "change_logged": change_logged,
            "verified": change_logged,
        }

    def _is_change_logged(self, schema_name: str, new_hash: str) -> bool:
        """
        Check if schema change has a changelog entry

        Looks for:
        1. Entry in self_integrity_metadata change_log
        2. OR entry in schema's own changelog section
        """
        # Check metadata change log
        for entry in self.metadata.get("change_log", []):
            if entry.get("schema") == schema_name and entry.get("hash") == new_hash:
                return True

        # Check schema's internal changelog
        schema_path = self.config_dir / schema_name
        if schema_path.exists():
            try:
                schema_data = yaml.safe_load(open(schema_path))
                changelog = schema_data.get("changelog", [])

                # If there's a recent changelog entry (within 7 days), consider it logged
                if changelog:
                    latest_entry = (
                        changelog[0] if isinstance(changelog, list) else changelog
                    )
                    if isinstance(latest_entry, dict):
                        entry_date_str = latest_entry.get("date", "")
                        try:
                            entry_date = datetime.fromisoformat(entry_date_str)
                            if (datetime.now() - entry_date).days <= 7:
                                return True
                        except Exception:
                            pass
            except Exception:
                pass

        return False

    def log_schema_change(
        self, schema_name: str, new_hash: str, reason: str, operator: str = "auto"
    ):
        """Log a verified schema change"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "schema": schema_name,
            "hash": new_hash,
            "reason": reason,
            "operator": operator,
        }

        self.metadata["change_log"].append(entry)
        self.metadata["schema_hashes"][schema_name] = new_hash
        self.save_metadata()

    def compute_inconsistency_from_dualrun(self, dualrun_report: dict) -> float:
        """
        Compute inconsistency rate from dual-run comparison

        Args:
            dualrun_report: Output from GuardianDualRunner

        Returns:
            Inconsistency rate (0 to 1)
        """
        comparison = dualrun_report.get("comparison", {})

        if comparison.get("comparison_failed"):
            return 0.5  # Unknown, assume moderate

        return comparison.get("inconsistency_rate", 0.0)

    def compute_self_integrity_score(
        self, inconsistency_rate: float = 0.0, unverified_changes: int = 0
    ) -> dict:
        """
        Compute Self-Integrity Score

        Formula: score = 1 - inconsistency_rate - (unverified_changes * 0.05)

        Args:
            inconsistency_rate: Rate of inconsistencies detected (0-1)
            unverified_changes: Number of unverified schema changes

        Returns:
            Self-Integrity score and details
        """
        # Penalty for unverified changes (5% per change)
        unverified_penalty = min(0.50, unverified_changes * 0.05)

        # Compute score
        score = max(0.0, 1.0 - inconsistency_rate - unverified_penalty)

        # Classify
        if score >= 0.95:
            status = "EXCELLENT"
        elif score >= 0.90:
            status = "GOOD"
        elif score >= 0.80:
            status = "MODERATE"
        else:
            status = "NEEDS_ATTENTION"

        result = {
            "self_integrity_score": score,
            "status": status,
            "components": {
                "base_integrity": 1.0,
                "inconsistency_rate": inconsistency_rate,
                "inconsistency_penalty": inconsistency_rate,
                "unverified_changes": unverified_changes,
                "unverified_penalty": unverified_penalty,
            },
            "passes_threshold": score >= 0.95,
            "timestamp": datetime.now().isoformat(),
        }

        # Record in history
        self.metadata["self_integrity_history"].append(result)
        self.save_metadata()

        return result

    def get_rolling_window_score(self, days: int = 180) -> dict:
        """
        Compute self-integrity over rolling window

        Args:
            days: Window size (default: 6 months)

        Returns:
            Aggregated self-integrity metrics
        """
        cutoff = datetime.now() - timedelta(days=days)

        recent_scores = []
        for entry in self.metadata.get("self_integrity_history", []):
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time >= cutoff:
                    recent_scores.append(entry["self_integrity_score"])
            except Exception:
                continue

        if not recent_scores:
            return {
                "window_days": days,
                "samples": 0,
                "avg_score": None,
                "min_score": None,
                "meets_threshold": False,
            }

        avg_score = sum(recent_scores) / len(recent_scores)
        min_score = min(recent_scores)

        return {
            "window_days": days,
            "samples": len(recent_scores),
            "avg_score": avg_score,
            "min_score": min_score,
            "meets_threshold": min_score >= 0.95,
        }

    def save_metadata(self):
        """Save metadata to file"""
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2))


def main():
    """CLI for self-integrity monitor"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v10 Self-Integrity Monitor")
    parser.add_argument(
        "command", choices=["check", "compute", "history"], help="Command to execute"
    )
    parser.add_argument("--schema", type=str, help="Schema file to check")
    parser.add_argument("--dualrun", type=str, help="Path to dual-run report")
    parser.add_argument("--days", type=int, default=180, help="Rolling window in days")

    args = parser.parse_args()

    monitor = SelfIntegrityMonitor()

    if args.command == "check":
        if not args.schema:
            print("❌ Error: --schema required")
            return

        print(f"🔍 Checking schema: {args.schema}")
        result = monitor.check_schema_change(args.schema)

        print("\n✅ Check complete:")
        print(f"   Changed: {result.get('changed', False)}")

        if result.get("changed"):
            print(f"   Previous hash: {result.get('previous_hash', 'N/A')}")
            print(f"   Current hash: {result.get('current_hash', 'N/A')}")
            print(f"   Logged: {'✅' if result.get('change_logged') else '❌'}")
            print(f"   Verified: {'✅' if result.get('verified') else '❌ UNVERIFIED'}")

    elif args.command == "compute":
        print("📊 Computing Self-Integrity Score...")

        # Load latest dual-run if available
        inconsistency_rate = 0.0
        if args.dualrun:
            dualrun_path = Path(args.dualrun)
            if dualrun_path.exists():
                dualrun_report = json.load(open(dualrun_path))
                inconsistency_rate = monitor.compute_inconsistency_from_dualrun(
                    dualrun_report
                )

        # Check for unverified changes
        unverified = 0
        for schema_name in [
            "scoring_schema_v5.yml",
            "scoring_schema_v6.yml",
            "thresholds_v8.yml",
        ]:
            check = monitor.check_schema_change(schema_name)
            if check.get("changed") and not check.get("verified"):
                unverified += 1

        # Compute score
        result = monitor.compute_self_integrity_score(
            inconsistency_rate=inconsistency_rate, unverified_changes=unverified
        )

        print(f"\n✅ Self-Integrity Score: {result['self_integrity_score']:.3f}")
        print(f"   Status: {result['status']}")
        print(
            f"   Passes threshold (≥0.95): {'✅' if result['passes_threshold'] else '❌'}"
        )

        print("\n   Components:")
        for comp, val in result["components"].items():
            print(f"     • {comp}: {val:.3f}")

    elif args.command == "history":
        print(f"📈 Self-Integrity History (last {args.days} days):")
        window = monitor.get_rolling_window_score(days=args.days)

        print(f"\n   Samples: {window['samples']}")
        if window["samples"] > 0:
            print(f"   Average score: {window['avg_score']:.3f}")
            print(f"   Minimum score: {window['min_score']:.3f}")
            print(f"   Meets threshold: {'✅' if window['meets_threshold'] else '❌'}")
        else:
            print("   ⚠️  No data in window")


if __name__ == "__main__":
    main()
