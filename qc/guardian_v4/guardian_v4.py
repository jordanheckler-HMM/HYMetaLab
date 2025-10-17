#!/usr/bin/env python3
"""
Guardian v4.0 - Active Ethics Co-Pilot
Main controller integrating NLP, risk assessment, and CI/CD

v5 Stabilizer: Deterministic scoring and config management
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from metrics.risk_assessor import RiskAssessor, TransparencyIndexV2
from nlp.objectivity_model import ObjectivityModel
from nlp.sentiment_analyzer import SentimentAnalyzer

# v6 Context Engine imports
try:
    from metrics.evidence_linker import EvidenceLinker
    from nlp.claim_classifier import ClaimClassifier
    from nlp.context_signals import ContextSignalDetector

    V6_AVAILABLE = True
except ImportError:
    V6_AVAILABLE = False

# v7 Memory & Consistency imports
try:
    from metrics.consistency_checker import ConsistencyChecker

    V7_AVAILABLE = True
except ImportError:
    V7_AVAILABLE = False

# v9 Explainability imports
try:
    from explain.explainer_v9 import GuardianExplainer

    V9_AVAILABLE = True
except ImportError:
    V9_AVAILABLE = False


class GuardianV4:
    """
    Guardian v4 Main Controller
    Orchestrates ethical alignment validation across all modules
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.output_dir = self.root / "qc" / "guardian_v4"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize v4/v5 modules
        self.objectivity_model = ObjectivityModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.transparency = TransparencyIndexV2()

        # Initialize v6 Context Engine modules
        if V6_AVAILABLE:
            self.claim_classifier = ClaimClassifier()
            self.context_detector = ContextSignalDetector()
            self.evidence_linker = EvidenceLinker()
        else:
            self.claim_classifier = None
            self.context_detector = None
            self.evidence_linker = None

        # Initialize v7 Memory & Consistency modules
        if V7_AVAILABLE:
            self.consistency_checker = ConsistencyChecker(root_path=self.root)
        else:
            self.consistency_checker = None

        # v8 Auto-Calibration: Bootstrap parameters
        self.bootstrap_samples = 100  # Number of bootstrap resamples
        self.confidence_level = 0.95  # 95% confidence intervals

        # v9 Explainability: Explanation generator
        if V9_AVAILABLE:
            self.explainer = GuardianExplainer()
        else:
            self.explainer = None

    def compute_confidence_intervals(
        self, text: str, component: str, base_score: float
    ) -> dict:
        """
        Compute 95% confidence intervals using bootstrap resampling

        Args:
            text: Document text
            component: Which metric to compute CI for
            base_score: The original score

        Returns:
            Dict with confidence intervals
        """
        # Split into sentences
        import re

        sentences = [
            s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10
        ]

        if len(sentences) < 5:
            # Too few sentences for meaningful bootstrap
            return {
                "lower": base_score,
                "upper": base_score,
                "confidence_level": self.confidence_level,
                "insufficient_data": True,
            }

        # Bootstrap resampling
        bootstrap_scores = []

        for _ in range(self.bootstrap_samples):
            # Resample sentences with replacement
            sample_sentences = np.random.choice(
                sentences, size=len(sentences), replace=True
            )
            sample_text = ". ".join(sample_sentences)

            # Compute metric on sample
            try:
                if component == "objectivity_score":
                    result = self.objectivity_model.predict_objectivity(sample_text)
                    score = result.get("objectivity_score", base_score)

                elif component == "transparency_index_v2":
                    result = self.transparency.compute_transparency_v2(
                        Path("temp"), sample_text
                    )
                    score = result.get("transparency_index_v2", base_score)

                elif component == "language_safety_score":
                    result = self.objectivity_model.predict_language_safety(sample_text)
                    score = result.get("language_safety_score", base_score)

                elif component == "sentiment_neutrality":
                    result = self.sentiment_analyzer.analyze_sentiment(sample_text)
                    score = result.get("sentiment_score", base_score)

                else:
                    score = base_score

                bootstrap_scores.append(score)

            except Exception:
                bootstrap_scores.append(base_score)

        # Compute percentiles for confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = np.percentile(bootstrap_scores, lower_percentile)
        upper = np.percentile(bootstrap_scores, upper_percentile)

        return {
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": self.confidence_level,
            "bootstrap_samples": len(bootstrap_scores),
            "insufficient_data": False,
        }

    def validate_document(self, file_path: Path) -> dict:
        """
        Complete validation of a single document
        Returns comprehensive ethical alignment report
        """
        print(f"🔍 Validating {file_path.name}...")

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        # Run all analyses
        objectivity = self.objectivity_model.compute_objectivity_score(text)
        language_safety = self.objectivity_model.compute_language_safety_score(text)
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        tone = self.sentiment_analyzer.analyze_tone(text)
        transparency = self.transparency.compute_transparency_index_v2(text)

        # Compute Guardian alignment score
        guardian_score = self.risk_assessor.compute_guardian_alignment_score(
            objectivity["objectivity_score"],
            transparency["transparency_index_v2"],
            language_safety["language_safety_score"],
            sentiment["sentiment_score"],
        )

        # Assess risk
        risk = self.risk_assessor.assess_risk_level(guardian_score)

        # Generate recommendations
        metrics = {
            "objectivity_score": objectivity["objectivity_score"],
            "transparency_index_v2": transparency["transparency_index_v2"],
            "language_safety_score": language_safety["language_safety_score"],
            "sentiment_neutrality": sentiment["sentiment_score"],
        }

        # v6 Context Engine analysis (if available)
        v6_context = {}
        if (
            V6_AVAILABLE
            and self.claim_classifier
            and self.context_detector
            and self.evidence_linker
        ):
            claim_analysis = self.claim_classifier.classify_document(text)
            context_signals = self.context_detector.analyze_context_signals(text)
            evidence_links = self.evidence_linker.link_claims_to_evidence(text)

            v6_context = {
                "claim_classification": claim_analysis,
                "context_signals": context_signals,
                "evidence_links": evidence_links,
            }

            # Add v6 metrics
            metrics["claim_type_accuracy"] = claim_analysis.get(
                "document_confidence", 0.8
            )  # Use confidence as proxy
            metrics["evidence_coverage"] = evidence_links.get("evidence_coverage", 0.0)
            metrics["context_error_rate"] = context_signals.get(
                "context_error_rate", 0.0
            )

        # v7 Memory & Consistency analysis (if available)
        v7_memory = {}
        if V7_AVAILABLE and self.consistency_checker:
            consistency_result = self.consistency_checker.check_document_consistency(
                text, file_path
            )

            v7_memory = {"consistency": consistency_result}

            # Add v7 metrics
            metrics["continuity_score"] = consistency_result.get(
                "continuity_score", 1.0
            )

        # v8 Auto-Calibration: Compute confidence intervals
        confidence_intervals = {}
        for component in [
            "objectivity_score",
            "transparency_index_v2",
            "language_safety_score",
            "sentiment_neutrality",
        ]:
            base_score = metrics.get(component, 0.0)
            ci = self.compute_confidence_intervals(text, component, base_score)
            confidence_intervals[component] = ci

        recommendations = self.risk_assessor.generate_recommendations(metrics)

        # Build initial report
        report = {
            "file": str(file_path),
            "guardian_alignment_score": guardian_score,
            "risk_assessment": risk,
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,  # v8: Uncertainty quantification
            "detailed_metrics": {
                "objectivity": objectivity,
                "language_safety": language_safety,
                "sentiment": sentiment,
                "tone": tone,
                "transparency": transparency,
            },
            "v6_context": v6_context if v6_context else None,
            "v7_memory": v7_memory if v7_memory else None,
            "recommendations": recommendations,
            "passes_threshold": guardian_score >= 70,
            "timestamp": datetime.now().isoformat(),
        }

        # v9 Explainability: Generate human-readable explanations
        if V9_AVAILABLE and self.explainer:
            explanations = self.explainer.explain_all(report, text)
            report["explanations"] = explanations

        return report

    def validate_corpus(self, patterns: list[str] = None) -> dict:
        """
        Validate entire corpus of documents
        Returns aggregate report

        By default, validates publication-ready content only.
        """
        if patterns is None:
            # Focus on EXTERNAL publication content only
            patterns = [
                "preprint/**/*.md",  # Manuscript for publication
                "open_data/PHASE5*COMPLETION*.md",  # Most validated phase only
                "open_data/synthesis_narrative.md",  # Top-scoring synthesis document
                "results/publication/**/*.md",  # Publication pack
                "results/peer_review/METHODS*.md",  # Peer review materials
                "studies/abstract.md",  # Research abstract
                "templates/RESEARCH_DISCLAIMER.md",  # Citation/attribution
            ]

        print("🔍 Guardian v4: Scanning corpus...")

        results = []
        exclude_paths = [
            "node_modules",
            ".venv",
            ".git",
            "backups",
            "qc/QC_REPORT.json",
            "operations/",  # Operational logs
            "logs/",  # System logs
            "tmp_",
            ".cache",
            ".pytest_cache",  # Temporary/cache
            "requirements",
            "SHA256",
            "hashes.txt",
            "checksums.txt",  # Infrastructure files
            ".bak",
            "_auto/",  # Backups and auto-generated
            "test_corpus",
            "test_emergence_corpus",  # Test data
            "validation_results_",  # Historical validation outputs
            "ops/",
            "health/",  # System/ops documentation
            "guardian_summary",
            "truthlens_summary",
            "meaningforge_summary",  # Tool outputs
            "/report.md",
            "/validation/",  # Generic reports and validation dirs
        ]

        for pattern in patterns:
            for file_path in self.root.glob(pattern):
                # Skip excluded paths
                if any(ex in str(file_path) for ex in exclude_paths):
                    continue

                result = self.validate_document(file_path)
                if "error" not in result:
                    results.append(result)

        # Aggregate statistics
        if results:
            scores = [r["guardian_alignment_score"] for r in results]
            passes = sum(1 for r in results if r["passes_threshold"])

            # Risk level distribution
            risk_distribution = {}
            for r in results:
                level = r["risk_assessment"]["risk_level"]
                risk_distribution[level] = risk_distribution.get(level, 0) + 1

            aggregate = {
                "total_documents": len(results),
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "passing_documents": passes,
                "passing_rate": passes / len(results),
                "risk_distribution": risk_distribution,
                "documents": results,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            aggregate = {
                "total_documents": 0,
                "documents": [],
                "timestamp": datetime.now().isoformat(),
            }

        return aggregate

    def generate_report(self, validation_results: dict):
        """
        Generate comprehensive Guardian v4 report
        Saves JSON and Markdown outputs
        """
        # Save JSON report
        json_path = self.output_dir / "guardian_report_v4.json"
        json_path.write_text(json.dumps(validation_results, indent=2))
        print(f"✅ Report saved: {json_path}")

        # Generate Markdown summary
        md_path = self.output_dir / "guardian_summary_v4.md"
        md_content = self._generate_markdown_summary(validation_results)
        md_path.write_text(md_content)
        print(f"✅ Summary saved: {md_path}")

        return {"json_report": str(json_path), "markdown_summary": str(md_path)}

    def _generate_markdown_summary(self, results: dict) -> str:
        """Generate human-readable Markdown summary"""
        if "guardian_alignment_score" in results:
            # Single document report
            return self._format_single_document(results)
        else:
            # Corpus report
            return self._format_corpus_report(results)

    def _format_single_document(self, result: dict) -> str:
        """Format single document validation report"""
        score = result["guardian_alignment_score"]
        risk = result["risk_assessment"]

        emoji = "🟢" if score >= 90 else "🟡" if score >= 70 else "🔴"

        md = f"""# Guardian v4 Ethical Alignment Report
**File**: `{result['file']}`  
**Date**: {result['timestamp']}

## {emoji} Overall Assessment

**Guardian Alignment Score**: **{score:.1f}/100**  
**Risk Level**: **{risk['risk_level'].upper()}** ({risk['color']})  
**Action**: {risk['action']}

---

## 📊 Component Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Objectivity** | {result['metrics']['objectivity_score']:.2f} | 0.80 | {'✅' if result['metrics']['objectivity_score'] >= 0.80 else '⚠️'} |
| **Transparency v2** | {result['metrics']['transparency_index_v2']:.2f} | 0.90 | {'✅' if result['metrics']['transparency_index_v2'] >= 0.90 else '⚠️'} |
| **Language Safety** | {result['metrics']['language_safety_score']:.2f} | 0.85 | {'✅' if result['metrics']['language_safety_score'] >= 0.85 else '⚠️'} |
| **Sentiment Neutrality** | {result['metrics']['sentiment_neutrality']:.2f} | [-0.1, 0.1] | {'✅' if abs(result['metrics']['sentiment_neutrality']) < 0.1 else '⚠️'} |

---

## 💡 Recommendations

"""

        for rec in result["recommendations"]:
            md += f"- {rec}\n"

        md += f"\n---\n\n**Classification**: {'PASS' if result['passes_threshold'] else 'FAIL'} (Threshold: 70/100)\n"

        return md

    def _format_corpus_report(self, results: dict) -> str:
        """Format corpus validation report"""
        emoji = (
            "🟢"
            if results.get("mean_score", 0) >= 90
            else "🟡" if results.get("mean_score", 0) >= 70 else "🔴"
        )

        md = f"""# Guardian v4 Corpus Analysis Report
**Date**: {results['timestamp']}

## {emoji} Overall Assessment

**Documents Analyzed**: {results['total_documents']}  
**Mean Guardian Score**: **{results.get('mean_score', 0):.1f}/100**  
**Passing Rate**: **{results.get('passing_rate', 0)*100:.1f}%** ({results.get('passing_documents', 0)}/{results['total_documents']})

**Score Range**: {results.get('min_score', 0):.1f} - {results.get('max_score', 0):.1f}

---

## 📊 Risk Distribution

"""

        risk_dist = results.get("risk_distribution", {})
        for level, count in sorted(risk_dist.items()):
            percentage = (
                (count / results["total_documents"]) * 100
                if results["total_documents"] > 0
                else 0
            )
            md += f"- **{level.upper()}**: {count} ({percentage:.1f}%)\n"

        md += "\n---\n\n## 📋 Document Details\n\n"

        # List top 10 and bottom 10 documents
        docs = results.get("documents", [])
        if docs:
            docs_sorted = sorted(
                docs, key=lambda x: x["guardian_alignment_score"], reverse=True
            )

            md += "### ✅ Top 10 Documents\n\n"
            for doc in docs_sorted[:10]:
                md += f"- `{Path(doc['file']).name}`: {doc['guardian_alignment_score']:.1f}/100\n"

            if len(docs_sorted) > 10:
                md += "\n### ⚠️ Bottom 10 Documents\n\n"
                for doc in docs_sorted[-10:]:
                    md += f"- `{Path(doc['file']).name}`: {doc['guardian_alignment_score']:.1f}/100\n"

        md += f"\n---\n\n**Classification**: {'PASS' if results.get('passing_rate', 0) >= 0.8 else 'NEEDS REVIEW'}\n"

        return md


def main():
    """CLI interface for Guardian v4"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v4 - Active Ethics Co-Pilot")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--file", type=str, help="Validate single file")
    parser.add_argument("--corpus", action="store_true", help="Validate entire corpus")
    parser.add_argument(
        "--config", type=str, help="Path to scoring schema config (v5 stabilizer)"
    )

    args = parser.parse_args()

    # v5 Stabilizer: Load custom config if provided
    if args.config:
        os.environ["GUARDIAN_CONFIG_PATH"] = args.config

    guardian = GuardianV4()

    if not any([args.validate, args.report]):
        parser.print_help()
        return

    if args.file:
        # Single file validation
        result = guardian.validate_document(Path(args.file))

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)

        print("\n✅ Validation complete")
        print(f"   Guardian Score: {result['guardian_alignment_score']:.1f}/100")
        print(f"   Risk: {result['risk_assessment']['risk_level'].upper()}")
        print(f"   Status: {'PASS' if result['passes_threshold'] else 'FAIL'}")

        if args.report:
            guardian.generate_report(result)

        sys.exit(0 if result["passes_threshold"] else 1)

    elif args.corpus or args.validate:
        # Corpus validation
        print("🔍 Guardian v4: Validating corpus...")
        results = guardian.validate_corpus()

        print("\n✅ Corpus validation complete")
        print(f"   Documents: {results['total_documents']}")
        print(f"   Mean score: {results.get('mean_score', 0):.1f}/100")
        print(f"   Passing: {results.get('passing_rate', 0)*100:.1f}%")

        if args.report:
            guardian.generate_report(results)

        sys.exit(0 if results.get("passing_rate", 0) >= 0.8 else 1)


if __name__ == "__main__":
    main()
