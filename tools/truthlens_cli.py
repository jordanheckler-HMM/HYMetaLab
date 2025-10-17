#!/usr/bin/env python3
"""
TruthLens v1 - Command Line Interface
Analyze documents for Truth Index (Ti)

v1 Genesis: Simple, deterministic, reproducible
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthlens_core import TruthLensCore

# v2 Causal Consistency imports
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from causal_parser import CausalParser

    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# v3 Evidence Graph imports
try:
    from truthgraph_builder import TruthGraphBuilder

    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False

# v4 Domain Adapter imports
try:
    from domain_adapter import DomainAdapter

    V4_AVAILABLE = True
except ImportError:
    V4_AVAILABLE = False


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="TruthLens v1 - Truth Index Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python3 tools/truthlens_cli.py --file README.md
  
  # Use custom config
  python3 tools/truthlens_cli.py --file paper.md --config truth_index_v1.yml
  
  # Analyze multiple files
  python3 tools/truthlens_cli.py --corpus "docs/**/*.md"
  
  # Save to custom location
  python3 tools/truthlens_cli.py --file paper.md --output results/ti_report.json
        """,
    )

    parser.add_argument("--file", type=str, help="Single file to analyze")
    parser.add_argument(
        "--corpus",
        type=str,
        help='Glob pattern for multiple files (e.g., "docs/**/*.md")',
    )
    parser.add_argument(
        "--config",
        type=str,
        default="truth_index_v1.yml",
        help="Configuration file (default: truth_index_v1.yml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="truth_index.json",
        help="Output file (default: truth_index.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--causal", action="store_true", help="Enable v2 causal consistency analysis"
    )
    parser.add_argument(
        "--evidence", action="store_true", help="Enable v3 evidence graph analysis"
    )
    parser.add_argument(
        "--domain", action="store_true", help="Enable v4 domain adaptation"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠️  Config not found: {config_path}")
        print("   Using default configuration")
        config_path = None

    # Initialize TruthLens
    version_str = "v1 Genesis"
    if args.domain and V4_AVAILABLE:
        version_str = "v4 Domain Adapter"
    elif args.evidence and V3_AVAILABLE:
        version_str = "v3 Evidence Graph"
    elif args.causal and V2_AVAILABLE:
        version_str = "v2 Causal Consistency"

    print(f"🔬 TruthLens {version_str}")
    print(f"   Config: {config_path if config_path else 'default'}\n")

    tl = TruthLensCore(config_path=config_path)

    # Initialize causal parser if enabled
    cp = None
    if args.causal:
        if V2_AVAILABLE:
            cp = CausalParser()
            print("   ✅ Causal analysis enabled")
        else:
            print("   ⚠️  Causal parser not available (skipping)")

    # Initialize evidence graph builder if enabled
    gb = None
    if args.evidence:
        if V3_AVAILABLE:
            gb = TruthGraphBuilder()
            print("   ✅ Evidence graph enabled")
        else:
            print("   ⚠️  Evidence graph not available (skipping)")

    # Initialize domain adapter if enabled
    da = None
    if args.domain:
        if V4_AVAILABLE:
            da = DomainAdapter()
            print("   ✅ Domain adaptation enabled")
        else:
            print("   ⚠️  Domain adapter not available (skipping)")

    if cp or gb or da:
        print()

    # Collect files to analyze
    files = []

    if args.file:
        files.append(Path(args.file))
    elif args.corpus:
        root = Path.cwd()
        files = list(root.glob(args.corpus))
        print(f"   Found {len(files)} files matching pattern\n")
    else:
        print("❌ Error: Either --file or --corpus required")
        parser.print_help()
        return

    # Analyze files
    results = []

    for i, file_path in enumerate(files, 1):
        if args.verbose or len(files) > 1:
            print(f"[{i}/{len(files)}] {file_path.name}... ", end="", flush=True)

        result = tl.analyze_file(file_path)

        if "error" in result:
            if args.verbose:
                print(f"❌ Error: {result['error']}")
            continue

        # v2 Causal analysis (if enabled)
        if cp:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                parse_result = cp.parse_document(text)
                contradictions = cp.find_contradictions(parse_result["causal_pairs"])
                continuity_v2 = cp.compute_causal_continuity_score(
                    parse_result["causal_pair_count"], len(contradictions)
                )

                result["v2_causal"] = {
                    "causal_pairs": parse_result["causal_pairs"],
                    "causal_pair_count": parse_result["causal_pair_count"],
                    "contradictions": contradictions,
                    "contradiction_count": len(contradictions),
                    "continuity_score_v2": continuity_v2,
                }
            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Causal analysis failed: {e}")

        # v3 Evidence graph (if enabled)
        if gb:
            try:
                if "text" not in locals():
                    text = file_path.read_text(encoding="utf-8", errors="ignore")

                evidence_graph, graph_metrics = gb.build_graph(text)

                result["v3_evidence"] = {
                    "evidence_density": graph_metrics["evidence_density"],
                    "total_claims": graph_metrics["total_claims"],
                    "claims_with_evidence": graph_metrics["claims_with_evidence"],
                    "total_citations": graph_metrics["total_citations"],
                    "total_links": graph_metrics["total_links"],
                    "passes_threshold": graph_metrics["passes_threshold"],
                    "graph_file": str(file_path.stem) + "_evidence_graph.json",
                }

                # Save evidence graph separately
                graph_output = (
                    Path(args.output).parent / result["v3_evidence"]["graph_file"]
                )
                gb.save_graph(evidence_graph, graph_metrics, graph_output)

            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Evidence graph failed: {e}")

        # v4 Domain adaptation (if enabled)
        if da:
            try:
                if "text" not in locals():
                    text = file_path.read_text(encoding="utf-8", errors="ignore")

                # Get component scores for adaptation
                components = {
                    "claim_clarity": result.get("claim_clarity", 0),
                    "citation_presence": result.get("citation_presence", 0),
                    "causal_tokens": result.get("causal_tokens", 0),
                }

                domain_analysis = da.analyze_document(
                    text, result["truth_index"], components
                )

                result["v4_domain"] = {
                    "domain": domain_analysis["domain"],
                    "domain_confidence": domain_analysis["domain_confidence"],
                    "raw_ti": domain_analysis["raw_ti"],
                    "adapted_ti": domain_analysis["adapted_ti"],
                    "grade": domain_analysis["grade"],
                    "passes_threshold": domain_analysis["passes_threshold"],
                }

                # Update main Ti with adapted value
                if domain_analysis["adaptation"]["adaptation_applied"]:
                    result["truth_index_adapted"] = domain_analysis["adapted_ti"]
                    result["grade_adapted"] = domain_analysis["grade"]

            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Domain adaptation failed: {e}")

        results.append(result)

        if args.verbose or len(files) > 1:
            print(f"Ti: {result['truth_index']:.3f} ({result['grade']})")

    # Compute aggregate statistics
    if results:
        ti_scores = [r["truth_index"] for r in results]

        aggregate = {
            "count": len(results),
            "mean_ti": sum(ti_scores) / len(ti_scores),
            "min_ti": min(ti_scores),
            "max_ti": max(ti_scores),
            "std_ti": (
                (
                    sum((ti - sum(ti_scores) / len(ti_scores)) ** 2 for ti in ti_scores)
                    / len(ti_scores)
                )
                ** 0.5
                if len(ti_scores) > 1
                else 0.0
            ),
            "passes_threshold_count": sum(
                1 for r in results if r.get("passes_threshold")
            ),
            "timestamp": datetime.now().isoformat(),
        }

        # Display summary
        print(f"\n{'='*70}")
        print("📊 TRUTH INDEX SUMMARY")
        print(f"{'='*70}")
        print(f"\n   Documents analyzed: {aggregate['count']}")
        print(f"   Mean Ti: {aggregate['mean_ti']:.3f}")
        print(f"   Range: [{aggregate['min_ti']:.3f}, {aggregate['max_ti']:.3f}]")
        print(f"   Std deviation: {aggregate['std_ti']:.4f}")
        print(
            f"   Passes threshold (≥0.60): {aggregate['passes_threshold_count']}/{aggregate['count']}"
        )

        # Check acceptance criteria
        meets_mean = aggregate["mean_ti"] >= 0.60
        meets_variance = aggregate["std_ti"] < 0.02

        print("\n   Acceptance Criteria:")
        print(
            f"     • Mean Ti ≥0.60: {'✅' if meets_mean else '❌'} ({aggregate['mean_ti']:.3f})"
        )
        print(
            f"     • Std < 0.02: {'✅' if meets_variance else '❌'} ({aggregate['std_ti']:.4f})"
        )

        # v3 Evidence graph aggregate statistics
        if gb and any("v3_evidence" in r for r in results):
            evidence_densities = [
                r["v3_evidence"]["evidence_density"]
                for r in results
                if "v3_evidence" in r
            ]

            if evidence_densities:
                mean_evidence_density = sum(evidence_densities) / len(
                    evidence_densities
                )
                passes_evidence_threshold = sum(
                    1 for ed in evidence_densities if ed >= 0.7
                )

                aggregate["v3_evidence"] = {
                    "mean_evidence_density": mean_evidence_density,
                    "passes_evidence_threshold_count": passes_evidence_threshold,
                    "total_docs_analyzed": len(evidence_densities),
                }

                print("\n   v3 Evidence Graph:")
                print(f"     • Mean evidence density: {mean_evidence_density:.3f}")
                print(
                    f"     • Passes threshold (≥0.7): {passes_evidence_threshold}/{len(evidence_densities)}"
                )
                print(
                    f"     • Acceptance: {'✅' if mean_evidence_density >= 0.7 else '❌'}"
                )

        # v4 Domain adaptation aggregate statistics
        if da and any("v4_domain" in r for r in results):
            # Collect adapted Ti scores by domain
            domain_scores = {}
            for r in results:
                if "v4_domain" in r:
                    domain = r["v4_domain"]["domain"]
                    adapted_ti = r["v4_domain"]["adapted_ti"]
                    if domain not in domain_scores:
                        domain_scores[domain] = []
                    domain_scores[domain].append(adapted_ti)

            if domain_scores:
                # Compute cross-domain variance
                variance_metrics = da.compute_cross_domain_variance(domain_scores)

                aggregate["v4_domain"] = {
                    "domain_scores": domain_scores,
                    "domain_means": variance_metrics["domain_means"],
                    "overall_mean": variance_metrics["overall_mean"],
                    "relative_variance_pct": variance_metrics["relative_variance_pct"],
                    "passes_threshold": variance_metrics["passes_threshold"],
                }

                print("\n   v4 Domain Adaptation:")
                print(f"     • Domains analyzed: {len(domain_scores)}")
                for domain, scores in sorted(domain_scores.items()):
                    print(
                        f"       - {domain}: {len(scores)} docs, mean Ti = {np.mean(scores):.3f}"
                    )
                print(
                    f"     • Overall mean (adapted): {variance_metrics['overall_mean']:.3f}"
                )
                print(
                    f"     • Cross-domain variance: {variance_metrics['relative_variance_pct']:.2f}%"
                )
                print(
                    f"     • Passes ±5% threshold: {'✅' if variance_metrics['passes_threshold'] else '❌'}"
                )

        # Save results
        output_data = {
            "results": (
                results if len(results) <= 100 else results[:100]
            ),  # Limit for file size
            "aggregate": aggregate,
            "config": {
                "version": tl.config.get("version"),
                "seed": int(tl.config.get("seed", 42)),  # Ensure int
                "weights": tl.config.get("weights", {}),
                "thresholds": tl.config.get("thresholds", {}),
            },
            "acceptance": {
                "meets_mean_threshold": bool(meets_mean),
                "meets_variance_threshold": bool(meets_variance),
                "all_criteria_met": bool(meets_mean and meets_variance),
            },
        }

        output_path = Path(args.output)

        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                return super().default(obj)

        output_path.write_text(json.dumps(output_data, indent=2, cls=NumpyEncoder))

        print(f"\n✅ Results saved: {output_path}")
        print(f"{'='*70}\n")

    else:
        print("❌ No results generated")


if __name__ == "__main__":
    main()
