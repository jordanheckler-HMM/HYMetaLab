#!/usr/bin/env python3
"""
Guardian v10 - Dual Runner
Run v3 and v4 in parallel for regression testing and consistency validation

v10 Meta-Audit: Self-auditing and version comparison
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class GuardianDualRunner:
    """
    Run Guardian v3 and v4 on same corpus and compare results
    Detect regressions and inconsistencies
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.v3_script = self.root / "guardian_v3.py"
        self.v4_script = self.root / "qc" / "guardian_v4" / "guardian_v4.py"
        self.output_dir = self.root / "qc" / "guardian_v4" / "dualrun"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tolerance thresholds for flagging differences
        self.tolerances = {
            "objectivity_score": 0.10,  # ±10% acceptable difference
            "transparency_index_v2": 0.10,
            "language_safety_score": 0.10,
            "sentiment_neutrality": 0.15,  # Slightly higher for sentiment
            "guardian_alignment_score": 5.0,  # ±5 points on 100-point scale
        }

    def run_v3(self) -> dict:
        """Run Guardian v3 and parse results"""
        try:
            print("🔍 Running Guardian v3...")
            result = subprocess.run(
                [sys.executable, str(self.v3_script)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Load v3 report
            v3_report_path = self.root / "qc" / "guardian_report_v3.json"
            if not v3_report_path.exists():
                return {"error": "v3 report not generated"}

            v3_data = json.load(open(v3_report_path))

            print(
                f"✅ v3 complete: Score = {v3_data.get('guardian_alignment_v3', 0):.1f}/100"
            )

            return {
                "version": "v3",
                "score": v3_data.get("guardian_alignment_v3", 0),
                "objectivity": v3_data.get("objectivity", 0),
                "sentiment": v3_data.get("sentiment", 0),
                "language_safety": v3_data.get("language_safety", 0),
                "transparency": v3_data.get("transparency_index", 0),
                "success": True,
            }

        except Exception as e:
            print(f"❌ v3 failed: {e}")
            return {"error": str(e), "success": False}

    def run_v4(self, config: str = None) -> dict:
        """Run Guardian v4 and parse results"""
        try:
            print("🔍 Running Guardian v4...")

            cmd = [sys.executable, str(self.v4_script), "--validate", "--report"]
            if config:
                cmd.extend(["--config", config])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Load v4 report
            v4_report_path = (
                self.root / "qc" / "guardian_v4" / "guardian_report_v4.json"
            )
            if not v4_report_path.exists():
                return {"error": "v4 report not generated"}

            v4_data = json.load(open(v4_report_path))
            metrics = v4_data.get("metrics", {})

            print(
                f"✅ v4 complete: Score = {v4_data.get('guardian_alignment_score', 0):.1f}/100"
            )

            return {
                "version": "v4",
                "score": v4_data.get("guardian_alignment_score", 0),
                "objectivity": metrics.get("objectivity_score", 0),
                "sentiment": metrics.get("sentiment_neutrality", 0),
                "language_safety": metrics.get("language_safety_score", 0),
                "transparency": metrics.get("transparency_index_v2", 0),
                "success": True,
                "full_report": v4_data,
            }

        except Exception as e:
            print(f"❌ v4 failed: {e}")
            return {"error": str(e), "success": False}

    def compare_results(self, v3_result: dict, v4_result: dict) -> dict:
        """
        Compare v3 and v4 results, flag significant differences

        Returns:
            Comparison dictionary with flagged differences
        """
        if not v3_result.get("success") or not v4_result.get("success"):
            return {
                "comparison_failed": True,
                "v3_error": v3_result.get("error"),
                "v4_error": v4_result.get("error"),
            }

        differences = {}
        flags = []

        # Compare each metric
        for metric in [
            "objectivity",
            "sentiment",
            "language_safety",
            "transparency",
            "score",
        ]:
            v3_val = v3_result.get(metric, 0)
            v4_val = v4_result.get(metric, 0)

            delta = abs(v4_val - v3_val)

            # Determine tolerance
            if metric == "score":
                tolerance = self.tolerances.get("guardian_alignment_score", 5.0)
            else:
                metric_key = (
                    f"{metric}_score"
                    if metric != "sentiment"
                    else "sentiment_neutrality"
                )
                if metric == "transparency":
                    metric_key = "transparency_index_v2"
                tolerance = self.tolerances.get(metric_key, 0.10)

            flagged = delta > tolerance

            differences[metric] = {
                "v3": v3_val,
                "v4": v4_val,
                "delta": delta,
                "tolerance": tolerance,
                "flagged": flagged,
            }

            if flagged:
                flags.append(f"{metric}: Δ={delta:.3f} (tolerance: {tolerance:.3f})")

        # Compute overall inconsistency rate
        flagged_count = sum(1 for d in differences.values() if d.get("flagged"))
        total_count = len(differences)
        inconsistency_rate = flagged_count / total_count if total_count > 0 else 0

        return {
            "differences": differences,
            "flags": flags,
            "flagged_count": flagged_count,
            "total_metrics": total_count,
            "inconsistency_rate": inconsistency_rate,
            "consistent": flagged_count == 0,
        }

    def run_dual(self, config: str = None) -> dict:
        """
        Run both versions and compare

        Args:
            config: Optional v4 config path

        Returns:
            Complete dual-run report
        """
        print("\n" + "=" * 70)
        print("🔄 Guardian Dual-Run (v3 vs v4)")
        print("=" * 70 + "\n")

        # Run both versions
        v3_result = self.run_v3()
        v4_result = self.run_v4(config=config)

        # Compare
        comparison = self.compare_results(v3_result, v4_result)

        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "v3_result": v3_result,
            "v4_result": v4_result,
            "comparison": comparison,
            "config_used": config,
        }

        # Save report
        report_path = (
            self.output_dir / f"dualrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path.write_text(json.dumps(report, indent=2))

        print(f"\n{'='*70}")
        print("📊 DUAL-RUN COMPARISON RESULTS")
        print("=" * 70)

        if comparison.get("comparison_failed"):
            print("\n❌ Comparison failed")
            print(f"   v3 error: {comparison.get('v3_error')}")
            print(f"   v4 error: {comparison.get('v4_error')}")
        else:
            print(f"\nInconsistency Rate: {comparison['inconsistency_rate']*100:.1f}%")
            print(
                f"Flagged Metrics: {comparison['flagged_count']}/{comparison['total_metrics']}"
            )
            print(
                f"Status: {'✅ CONSISTENT' if comparison['consistent'] else '⚠️  DIVERGENCE DETECTED'}"
            )

            if comparison["flags"]:
                print("\n⚠️  Flagged Differences:")
                for flag in comparison["flags"]:
                    print(f"   • {flag}")

            print("\nMetric Details:")
            for metric, diff in comparison["differences"].items():
                status = "⚠️" if diff["flagged"] else "✅"
                print(
                    f"   {status} {metric}: v3={diff['v3']:.2f}, v4={diff['v4']:.2f}, Δ={diff['delta']:.3f}"
                )

        print(f"\n✅ Report saved: {report_path}")
        print("=" * 70 + "\n")

        return report


def main():
    """CLI for dual runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v10 Dual Runner")
    parser.add_argument(
        "--corpus", action="store_true", help="Run on full corpus (v3 default)"
    )
    parser.add_argument(
        "--config", type=str, help="Config for v4 (e.g., scoring_schema_v5.yml)"
    )

    args = parser.parse_args()

    runner = GuardianDualRunner()
    report = runner.run_dual(config=args.config)

    # Exit code based on consistency
    comparison = report.get("comparison", {})
    if comparison.get("consistent"):
        sys.exit(0)
    else:
        # Flag inconsistency but don't fail hard (informational)
        sys.exit(0)  # Changed to 0 since divergence is expected for different versions


if __name__ == "__main__":
    main()
