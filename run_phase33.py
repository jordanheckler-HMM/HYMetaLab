#!/usr/bin/env python3
"""Runner for Phase 33 — Cooperative Meaning Fields.

Quick execution script for the preregistered study.
"""

import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from adapters.phase33_coop_meaning import run_adapter


def main():
    """Execute Phase 33 study."""
    print("🚀 Phase 33 — Cooperative Meaning Fields Runner")
    print("=" * 60)
    print()

    # Load study configuration
    study_path = Path(__file__).parent / "studies" / "phase33_coop_meaning.yml"

    if not study_path.exists():
        print(f"❌ Error: Study file not found: {study_path}")
        sys.exit(1)

    print(f"📖 Loading study: {study_path}")
    with open(study_path) as f:
        study_config = yaml.safe_load(f)

    print(f"   Study ID: {study_config['study_id']}")
    print(f"   Version: {study_config['version']}")
    print(f"   Preregistered: {study_config['prereg_date']}")
    print()

    # Prepare output directory
    output_dir_template = study_config["exports"]["outdir"]
    output_dir = Path(
        output_dir_template.replace("${study_id}", study_config["study_id"])
    )

    print(f"📁 Output directory: {output_dir}")
    print()

    # Run adapter
    try:
        result = run_adapter(study_config, output_dir)

        print()
        print("=" * 60)
        print("✅ Study execution complete!")
        print()
        print("📊 Results:")
        print(f"   Status: {result['status']}")
        print(f"   Total runs: {result['total_runs']}")
        print(f"   Output directory: {result['output_dir']}")
        print()

        # Display hypothesis test results
        if "summary" in result and "hypothesis_test" in result["summary"]:
            hyp = result["summary"]["hypothesis_test"]
            print("📈 Hypothesis Test:")
            print(f"   Mean CCI gain: {hyp['mean_CCI_gain']:.4f} (threshold: ≥0.03)")
            print(
                f"   Mean hazard Δ: {hyp['mean_hazard_delta']:.4f} (threshold: ≤-0.01)"
            )
            print()

            for metric in hyp["metrics_met"]:
                status = "✅" if metric["passed"] else "❌"
                print(
                    f"   {status} {metric['name']}: {metric['value']:.4f} {metric['rule']}"
                )

            print()
            if hyp["all_passed"]:
                print("🎉 All validation metrics passed!")
            else:
                print("⚠️  Some validation metrics did not meet criteria")

        print()
        print("📁 Generated files:")
        for name, path in result["files"].items():
            print(f"   • {name}: {path}")

        print()
        print("🔬 Next steps:")
        print("   1. Review results in CSV file")
        print(
            "   2. Run validation: python openlaws_automation.py validate --study studies/phase33_coop_meaning.yml"
        )
        print(
            "   3. Generate report: python openlaws_automation.py report --study studies/phase33_coop_meaning.yml"
        )

        return 0

    except Exception as e:
        print()
        print(f"❌ Error during study execution: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
