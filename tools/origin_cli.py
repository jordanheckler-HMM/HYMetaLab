#!/usr/bin/env python3
"""
OriginChain v1 CLI Tool
Command-line interface for Emergence Quotient (EQ) analysis

Usage:
    python tools/origin_cli.py analyze <file_or_dir> --output evolution_map.json
    python tools/origin_cli.py test
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from originchain_core import OriginChainCore


def analyze_path(path: Path, oc: OriginChainCore, recursive: bool = False):
    """
    Analyze file or directory

    Args:
        path: Path to analyze
        oc: OriginChainCore instance
        recursive: Recursively analyze directories

    Returns:
        List of analysis results
    """
    results = []

    if path.is_file():
        # Single file
        result = oc.analyze_file(path)
        results.append(result)
    elif path.is_dir():
        # Directory
        pattern = "**/*.md" if recursive else "*.md"
        for file_path in path.glob(pattern):
            if file_path.is_file():
                result = oc.analyze_file(file_path)
                results.append(result)

    return results


def generate_evolution_map(results: list, output_path: Path):
    """
    Generate evolution_map.json from analysis results

    Args:
        results: List of analysis results
        output_path: Path to write evolution_map.json
    """

    # Helper to convert numpy types to Python types
    def convert_numpy(obj):
        """Recursively convert numpy types to Python native types"""
        import numpy as np

        if isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        else:
            return obj

    # Sort by EQ (descending)
    sorted_results = sorted(
        results, key=lambda x: x.get("emergence_quotient", 0), reverse=True
    )

    # Compute statistics
    eq_scores = [r["emergence_quotient"] for r in results if "emergence_quotient" in r]

    evolution_map = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0-genesis",
        "analysis": {
            "total_documents": len(results),
            "mean_eq": sum(eq_scores) / len(eq_scores) if eq_scores else 0,
            "min_eq": min(eq_scores) if eq_scores else 0,
            "max_eq": max(eq_scores) if eq_scores else 0,
            "passes_threshold": sum(1 for eq in eq_scores if eq >= 0.7),
            "fails_threshold": sum(1 for eq in eq_scores if eq < 0.7),
        },
        "documents": [],
    }

    # Add documents
    for result in sorted_results:
        doc_entry = {
            "file": result.get("file", "unknown"),
            "emergence_quotient": result.get("emergence_quotient", 0),
            "grade": result.get("grade", "UNKNOWN"),
            "passes_threshold": result.get("passes_threshold", False),
            "components": {
                "complexity": result.get("complexity", 0),
                "novelty": result.get("novelty", 0),
                "interconnectedness": result.get("interconnectedness", 0),
            },
        }
        evolution_map["documents"].append(doc_entry)

    # Convert numpy types before JSON serialization
    evolution_map = convert_numpy(evolution_map)

    # Write to file
    with open(output_path, "w") as f:
        json.dump(evolution_map, f, indent=2)

    print(f"\n✅ Evolution map written to: {output_path}")
    print(f"   Documents analyzed: {len(results)}")
    print(f"   Mean EQ: {evolution_map['analysis']['mean_eq']:.3f}")
    print(
        f"   Passes threshold (≥0.7): {evolution_map['analysis']['passes_threshold']}"
    )


def cmd_analyze(args):
    """Execute analyze command"""
    path = Path(args.path)

    if not path.exists():
        print(f"❌ Error: Path not found: {path}")
        return 1

    # Initialize OriginChainCore
    config_path = Path("emergence_index_v1.yml")
    if config_path.exists():
        oc = OriginChainCore(config_path=config_path)
        print(f"✅ Loaded config from: {config_path}")
    else:
        oc = OriginChainCore()
        print("✅ Using default config")

    print(f"\n🌱 Analyzing: {path}")

    # Analyze
    results = analyze_path(path, oc, recursive=args.recursive)

    if not results:
        print("⚠️  No files found to analyze")
        return 1

    # Generate evolution map
    output_path = Path(args.output) if args.output else Path("evolution_map.json")
    generate_evolution_map(results, output_path)

    return 0


def cmd_test(args):
    """Execute test command"""
    print("🌱 Testing OriginChain CLI...")

    # Initialize
    oc = OriginChainCore()

    # Test text
    test_text = """
    Novel patterns emerge through complex interconnected systems that evolve organically.
    The multifaceted interplay between components creates unprecedented feedback loops.
    When self-organizing networks interact, they generate innovative emergent properties.
    """

    result = oc.compute_emergence_quotient(test_text)

    print("\n✅ Test result:")
    print(f"   EQ: {result['emergence_quotient']:.3f} ({result['grade']})")
    print(f"   Complexity: {result['complexity']:.3f}")
    print(f"   Novelty: {result['novelty']:.3f}")
    print(f"   Interconnectedness: {result['interconnectedness']:.3f}")
    print(f"   Passes threshold: {'✅' if result['passes_threshold'] else '❌'}")

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="OriginChain v1 CLI - Emergence Quotient Analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze files for EQ")
    analyze_parser.add_argument("path", help="File or directory to analyze")
    analyze_parser.add_argument(
        "-o",
        "--output",
        default="evolution_map.json",
        help="Output file for evolution map",
    )
    analyze_parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recursively analyze directories"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run basic test")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "test":
        return cmd_test(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
