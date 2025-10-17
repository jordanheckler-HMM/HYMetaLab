#!/usr/bin/env python3
"""Integration core for Aletheia: validate full-stack sync and emit UCS manifest

Checks presence and basic consistency of earlier outputs (sensor_network.json,
total_coherence.json, equilibrium_map.json) and writes a validated manifest to
`outputs/aletheia/ucs_manifest.yml`. Also writes a short coherence_bus.md log.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs" / "aletheia"


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def validate_and_compose():
    sensor = read_json(OUTDIR / "sensor_network.json")
    total = read_json(ROOT / "outputs" / "total_coherence.json") or read_json(
        ROOT / "outputs" / "aletheia" / "total_coherence.json"
    )
    eqmap = read_json(OUTDIR / "equilibrium_map.json")

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "components_present": {
            "sensor_network": bool(sensor),
            "total_coherence": bool(total),
            "equilibrium_map": bool(eqmap),
        },
        "checks": {},
    }

    # Basic consistency checks
    if sensor:
        manifest["checks"]["sensors_count"] = len(sensor.get("sensors", []))
    if total:
        manifest["checks"]["PsiC"] = total.get("PsiC")
    if eqmap:
        manifest["checks"]["equilibrium"] = eqmap.get("equilibrium")

    # Simple cross-check: equilibrium should be close to PsiC
    if total and eqmap:
        psi = float(total.get("PsiC", 0))
        eq = float(eqmap.get("equilibrium", 0))
        manifest["checks"]["psi_eq_delta"] = round(abs(psi - eq), 4)
        manifest["checks"]["psi_eq_consistent"] = abs(psi - eq) <= 0.1

    return manifest


def write_manifest(manifest):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / "ucs_manifest.yml"
    with open(out_path, "w") as f:
        yaml.safe_dump(manifest, f)
    return out_path


def write_bus_log(manifest):
    log_path = OUTDIR / "coherence_bus.md"
    lines = [
        "# Coherence Bus Sync Log",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        "## Checks",
        "",
    ]
    for k, v in manifest["checks"].items():
        lines.append(f"- {k}: {v}")
    log_path.write_text("\n".join(lines))
    return log_path


def main():
    manifest = validate_and_compose()
    mpath = write_manifest(manifest)
    lpath = write_bus_log(manifest)
    print(f"Wrote manifest {mpath} and bus log {lpath}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
OriginChain v3 - Integration Core
Cross-system integration for Guardian, TruthLens, MeaningForge, and OriginChain

v3 Integrator: Unified JSON ingestion with <1s sync
"""
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SystemMetrics:
    """Metrics from a specific system"""

    system_name: str
    version: str
    primary_score: float  # Main metric (0-1)
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class IntegratedAnalysis:
    """Integrated analysis across all systems"""

    document_id: str
    text_preview: str

    # System-specific metrics
    guardian: SystemMetrics | None = None
    truthlens: SystemMetrics | None = None
    meaningforge: SystemMetrics | None = None
    originchain: SystemMetrics | None = None

    # Integrated scores
    composite_score: float = 0.0
    integrity_index: float = 0.0  # Truth + Guardian
    meaning_emergence_index: float = 0.0  # Meaning + Emergence
    overall_quality: float = 0.0  # All systems

    # Metadata
    sync_time: float = 0.0
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            "document_id": self.document_id,
            "text_preview": self.text_preview,
            "composite_score": self.composite_score,
            "integrity_index": self.integrity_index,
            "meaning_emergence_index": self.meaning_emergence_index,
            "overall_quality": self.overall_quality,
            "sync_time": self.sync_time,
            "analysis_timestamp": self.analysis_timestamp,
            "systems": {},
        }

        if self.guardian:
            result["systems"]["guardian"] = self.guardian.to_dict()
        if self.truthlens:
            result["systems"]["truthlens"] = self.truthlens.to_dict()
        if self.meaningforge:
            result["systems"]["meaningforge"] = self.meaningforge.to_dict()
        if self.originchain:
            result["systems"]["originchain"] = self.originchain.to_dict()

        return result


class IntegrationCore:
    """
    Integration core for cross-system analysis

    Ingests and synchronizes:
    - Guardian: Harm prevention and safety
    - TruthLens: Epistemological integrity
    - MeaningForge: Semantic meaning analysis
    - OriginChain: Emergence quotient and evolution
    """

    def __init__(self):
        """Initialize IntegrationCore"""
        self.analyses: dict[str, IntegratedAnalysis] = {}

    # ==================== JSON INGESTION ====================

    def ingest_guardian_json(self, data: dict | str | Path) -> SystemMetrics:
        """
        Ingest Guardian JSON output

        Args:
            data: Guardian JSON data (dict, JSON string, or file path)

        Returns:
            SystemMetrics for Guardian
        """
        parsed = self._parse_json(data)

        # Guardian typically has: harm_score, safety_score, risk_level
        # Adapt to actual Guardian output format
        if "harm_score" in parsed:
            # Lower harm score is better, so invert
            primary_score = 1.0 - parsed.get("harm_score", 0.0)
        elif "safety_score" in parsed:
            primary_score = parsed.get("safety_score", 0.5)
        elif "overall_safety" in parsed:
            primary_score = parsed.get("overall_safety", 0.5)
        else:
            # Try to extract from nested structure
            primary_score = self._extract_primary_score(parsed, default=0.5)

        components = {}
        for key in [
            "harm_score",
            "safety_score",
            "risk_level",
            "violence_score",
            "toxicity_score",
            "bias_score",
        ]:
            if key in parsed:
                components[key] = parsed[key]

        return SystemMetrics(
            system_name="Guardian",
            version=parsed.get("version", "v3"),
            primary_score=primary_score,
            components=components,
            metadata=parsed.get("metadata", {}),
            timestamp=parsed.get("timestamp"),
        )

    def ingest_truthlens_json(self, data: dict | str | Path) -> SystemMetrics:
        """
        Ingest TruthLens JSON output

        Args:
            data: TruthLens JSON data

        Returns:
            SystemMetrics for TruthLens
        """
        parsed = self._parse_json(data)

        # TruthLens typically has: truth_score, epistemic_integrity
        if "truth_score" in parsed:
            primary_score = parsed["truth_score"]
        elif "epistemic_integrity" in parsed:
            primary_score = parsed["epistemic_integrity"]
        elif "overall_truth" in parsed:
            primary_score = parsed["overall_truth"]
        else:
            primary_score = self._extract_primary_score(parsed, default=0.5)

        components = {}
        for key in [
            "evidence_quality",
            "logical_coherence",
            "source_reliability",
            "claim_strength",
            "verification_level",
        ]:
            if key in parsed:
                components[key] = parsed[key]

        return SystemMetrics(
            system_name="TruthLens",
            version=parsed.get("version", "v5"),
            primary_score=primary_score,
            components=components,
            metadata=parsed.get("metadata", {}),
            timestamp=parsed.get("timestamp"),
        )

    def ingest_meaningforge_json(self, data: dict | str | Path) -> SystemMetrics:
        """
        Ingest MeaningForge JSON output

        Args:
            data: MeaningForge JSON data

        Returns:
            SystemMetrics for MeaningForge
        """
        parsed = self._parse_json(data)

        # MeaningForge typically has: meaning_quotient
        if "meaning_quotient" in parsed:
            primary_score = parsed["meaning_quotient"]
        elif "mq" in parsed:
            primary_score = parsed["mq"]
        elif "overall_meaning" in parsed:
            primary_score = parsed["overall_meaning"]
        else:
            primary_score = self._extract_primary_score(parsed, default=0.5)

        components = {}
        for key in [
            "relevance",
            "resonance",
            "transformative_potential",
            "emotional_resonance",
            "practical_applicability",
        ]:
            if key in parsed:
                components[key] = parsed[key]

        return SystemMetrics(
            system_name="MeaningForge",
            version=parsed.get("version", "v5"),
            primary_score=primary_score,
            components=components,
            metadata=parsed.get("metadata", {}),
            timestamp=parsed.get("timestamp"),
        )

    def ingest_originchain_json(self, data: dict | str | Path) -> SystemMetrics:
        """
        Ingest OriginChain JSON output

        Args:
            data: OriginChain JSON data

        Returns:
            SystemMetrics for OriginChain
        """
        parsed = self._parse_json(data)

        # OriginChain has: emergence_quotient
        if "emergence_quotient" in parsed:
            primary_score = parsed["emergence_quotient"]
        elif "eq" in parsed:
            primary_score = parsed["eq"]
        else:
            primary_score = self._extract_primary_score(parsed, default=0.5)

        components = {}
        for key in ["complexity", "novelty", "interconnectedness", "coherence"]:
            if key in parsed:
                components[key] = parsed[key]

        return SystemMetrics(
            system_name="OriginChain",
            version=parsed.get("version", "v2"),
            primary_score=primary_score,
            components=components,
            metadata=parsed.get("metadata", {}),
            timestamp=parsed.get("timestamp"),
        )

    # ==================== UNIFIED INGESTION ====================

    def ingest_all(
        self,
        document_id: str,
        text: str,
        guardian_data: dict | str | Path | None = None,
        truthlens_data: dict | str | Path | None = None,
        meaningforge_data: dict | str | Path | None = None,
        originchain_data: dict | str | Path | None = None,
    ) -> IntegratedAnalysis:
        """
        Ingest data from all systems and create integrated analysis

        Args:
            document_id: Unique document identifier
            text: Document text
            guardian_data: Guardian JSON data (optional)
            truthlens_data: TruthLens JSON data (optional)
            meaningforge_data: MeaningForge JSON data (optional)
            originchain_data: OriginChain JSON data (optional)

        Returns:
            IntegratedAnalysis with cross-system metrics
        """
        start_time = time.time()

        # Create integrated analysis
        analysis = IntegratedAnalysis(
            document_id=document_id,
            text_preview=text[:200] + "..." if len(text) > 200 else text,
        )

        # Ingest each system
        if guardian_data:
            analysis.guardian = self.ingest_guardian_json(guardian_data)

        if truthlens_data:
            analysis.truthlens = self.ingest_truthlens_json(truthlens_data)

        if meaningforge_data:
            analysis.meaningforge = self.ingest_meaningforge_json(meaningforge_data)

        if originchain_data:
            analysis.originchain = self.ingest_originchain_json(originchain_data)

        # Compute integrated scores
        self._compute_integrated_scores(analysis)

        # Record sync time
        analysis.sync_time = time.time() - start_time

        # Store analysis
        self.analyses[document_id] = analysis

        return analysis

    def ingest_batch(self, analyses: list[dict]) -> list[IntegratedAnalysis]:
        """
        Ingest batch of analyses

        Args:
            analyses: List of analysis dictionaries with system data

        Returns:
            List of IntegratedAnalysis objects
        """
        results = []

        for item in analyses:
            result = self.ingest_all(
                document_id=item.get("document_id", f"doc_{len(results)}"),
                text=item.get("text", ""),
                guardian_data=item.get("guardian"),
                truthlens_data=item.get("truthlens"),
                meaningforge_data=item.get("meaningforge"),
                originchain_data=item.get("originchain"),
            )
            results.append(result)

        return results

    # ==================== INTEGRATED SCORING ====================

    def _compute_integrated_scores(self, analysis: IntegratedAnalysis):
        """
        Compute integrated cross-system scores

        Args:
            analysis: IntegratedAnalysis to update
        """
        scores = []

        # Integrity Index: Guardian + TruthLens
        integrity_scores = []
        if analysis.guardian:
            integrity_scores.append(analysis.guardian.primary_score)
        if analysis.truthlens:
            integrity_scores.append(analysis.truthlens.primary_score)

        if integrity_scores:
            analysis.integrity_index = sum(integrity_scores) / len(integrity_scores)
            scores.append(analysis.integrity_index)

        # Meaning-Emergence Index: MeaningForge + OriginChain
        meaning_emergence_scores = []
        if analysis.meaningforge:
            meaning_emergence_scores.append(analysis.meaningforge.primary_score)
        if analysis.originchain:
            meaning_emergence_scores.append(analysis.originchain.primary_score)

        if meaning_emergence_scores:
            analysis.meaning_emergence_index = sum(meaning_emergence_scores) / len(
                meaning_emergence_scores
            )
            scores.append(analysis.meaning_emergence_index)

        # Composite Score: Average of all systems
        if analysis.guardian:
            scores.append(analysis.guardian.primary_score)
        if analysis.truthlens:
            scores.append(analysis.truthlens.primary_score)
        if analysis.meaningforge:
            scores.append(analysis.meaningforge.primary_score)
        if analysis.originchain:
            scores.append(analysis.originchain.primary_score)

        if scores:
            analysis.composite_score = sum(scores) / len(scores)

        # Overall Quality: Weighted combination
        # Guardian (25%) + TruthLens (25%) + MeaningForge (25%) + OriginChain (25%)
        quality_scores = []
        if analysis.guardian:
            quality_scores.append(analysis.guardian.primary_score * 0.25)
        if analysis.truthlens:
            quality_scores.append(analysis.truthlens.primary_score * 0.25)
        if analysis.meaningforge:
            quality_scores.append(analysis.meaningforge.primary_score * 0.25)
        if analysis.originchain:
            quality_scores.append(analysis.originchain.primary_score * 0.25)

        if quality_scores:
            analysis.overall_quality = sum(quality_scores) / (
                len(quality_scores) * 0.25
            )

    # ==================== UTILITY METHODS ====================

    def _parse_json(self, data: dict | str | Path) -> dict:
        """Parse JSON from various formats"""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            if data.strip().startswith("{"):
                # JSON string
                return json.loads(data)
            else:
                # File path
                return json.loads(Path(data).read_text())
        elif isinstance(data, Path):
            return json.loads(data.read_text())
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _extract_primary_score(self, data: dict, default: float = 0.5) -> float:
        """Extract primary score from nested JSON structure"""
        # Try common field names
        for key in [
            "score",
            "primary_score",
            "overall_score",
            "total_score",
            "final_score",
            "main_score",
        ]:
            if key in data:
                return float(data[key])

        # Try to find first float value
        for value in data.values():
            if isinstance(value, (int, float)):
                return float(value)

        return default

    # ==================== QUERY & EXPORT ====================

    def get_analysis(self, document_id: str) -> IntegratedAnalysis | None:
        """Get integrated analysis by document ID"""
        return self.analyses.get(document_id)

    def get_all_analyses(self) -> list[IntegratedAnalysis]:
        """Get all integrated analyses"""
        return list(self.analyses.values())

    def export_analysis(self, document_id: str, output_path: Path):
        """Export single analysis to JSON file"""
        analysis = self.get_analysis(document_id)
        if analysis:
            with open(output_path, "w") as f:
                json.dump(analysis.to_dict(), f, indent=2)

    def export_all(self, output_path: Path):
        """Export all analyses to JSON file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_analyses": len(self.analyses),
            "analyses": [a.to_dict() for a in self.get_all_analyses()],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.analyses:
            return {"error": "No analyses available"}

        sync_times = [a.sync_time for a in self.get_all_analyses()]

        return {
            "total_analyses": len(self.analyses),
            "mean_sync_time": sum(sync_times) / len(sync_times),
            "max_sync_time": max(sync_times),
            "min_sync_time": min(sync_times),
            "under_1s": sum(1 for t in sync_times if t < 1.0),
            "under_1s_rate": sum(1 for t in sync_times if t < 1.0) / len(sync_times),
        }


def main():
    """CLI for testing integration_core"""
    import argparse

    parser = argparse.ArgumentParser(description="OriginChain v3 Integration Core")
    parser.add_argument(
        "command", choices=["test", "perf-test"], help="Command to execute"
    )

    args = parser.parse_args()

    if args.command == "test":
        print("🔗 Testing Integration Core...")

        ic = IntegrationCore()

        # Test data
        guardian_data = {
            "version": "v3",
            "safety_score": 0.85,
            "harm_score": 0.1,
            "risk_level": "low",
        }

        truthlens_data = {
            "version": "v5",
            "truth_score": 0.92,
            "evidence_quality": 0.9,
            "logical_coherence": 0.95,
        }

        meaningforge_data = {
            "version": "v5",
            "meaning_quotient": 0.88,
            "relevance": 0.85,
            "resonance": 0.9,
        }

        originchain_data = {
            "version": "v2",
            "emergence_quotient": 0.78,
            "complexity": 0.75,
            "novelty": 0.8,
            "interconnectedness": 0.8,
        }

        # Ingest all systems
        analysis = ic.ingest_all(
            document_id="test_doc_1",
            text="Novel patterns emerge through complex interconnected systems...",
            guardian_data=guardian_data,
            truthlens_data=truthlens_data,
            meaningforge_data=meaningforge_data,
            originchain_data=originchain_data,
        )

        print("\n✅ Integrated Analysis:")
        print(f"   Document: {analysis.document_id}")
        print(f"   Sync time: {analysis.sync_time*1000:.2f}ms")
        print(f"   Composite score: {analysis.composite_score:.3f}")
        print(f"   Integrity index: {analysis.integrity_index:.3f}")
        print(f"   Meaning-Emergence index: {analysis.meaning_emergence_index:.3f}")
        print(f"   Overall quality: {analysis.overall_quality:.3f}")

        print("\n✅ System Scores:")
        if analysis.guardian:
            print(f"   Guardian: {analysis.guardian.primary_score:.3f}")
        if analysis.truthlens:
            print(f"   TruthLens: {analysis.truthlens.primary_score:.3f}")
        if analysis.meaningforge:
            print(f"   MeaningForge: {analysis.meaningforge.primary_score:.3f}")
        if analysis.originchain:
            print(f"   OriginChain: {analysis.originchain.primary_score:.3f}")

    elif args.command == "perf-test":
        print("🔗 Testing Performance (<1s sync)...")

        ic = IntegrationCore()

        # Batch test data
        batch_size = 100
        analyses = []

        for i in range(batch_size):
            analyses.append(
                {
                    "document_id": f"doc_{i}",
                    "text": f"Test document {i} with sample text...",
                    "guardian": {"safety_score": 0.8 + i * 0.001},
                    "truthlens": {"truth_score": 0.85 + i * 0.001},
                    "meaningforge": {"meaning_quotient": 0.75 + i * 0.001},
                    "originchain": {"emergence_quotient": 0.7 + i * 0.001},
                }
            )

        # Run batch ingestion
        start_time = time.time()
        results = ic.ingest_batch(analyses)
        total_time = time.time() - start_time

        # Get performance stats
        stats = ic.get_performance_stats()

        print("\n✅ Batch Performance:")
        print(f"   Documents: {len(results)}")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Mean sync time: {stats['mean_sync_time']*1000:.2f}ms")
        print(f"   Max sync time: {stats['max_sync_time']*1000:.2f}ms")
        print(
            f"   Under 1s: {stats['under_1s']}/{stats['total_analyses']} ({stats['under_1s_rate']*100:.1f}%)"
        )
        print(
            f"   Passes <1s requirement: {'✅' if stats['max_sync_time'] < 1.0 else '❌'}"
        )


if __name__ == "__main__":
    main()
