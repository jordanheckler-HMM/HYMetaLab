#!/usr/bin/env python3
"""
HYMetaLab Auto-Theorist Enhanced
Custom version that parses Phase 33c and FIS studies to generate next-wave hypotheses
"""
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(".")
DISCOVERY = ROOT / "discovery_results"
STUDIES_DIR = ROOT / "studies"
STUDIES_DIR.mkdir(exist_ok=True)

print("🤖 Auto-Theorist Enhanced - HYMetaLab Hypothesis Generator")
print("=" * 80)

# ============================================================================
# STEP 1: Load validated studies from recent publication pack
# ============================================================================
print("\n[1/4] Loading validated discovery results...")

validated_studies = []

# Phase 33c studies
for phase33_dir in sorted(DISCOVERY.glob("phase33c_coop_meaning_*"), reverse=True)[:3]:
    summary_file = phase33_dir / "summary.json"
    if summary_file.exists():
        try:
            data = json.loads(summary_file.read_text())
            validated_studies.append(
                {
                    "id": "phase33c",
                    "path": str(phase33_dir),
                    "summary": data,
                    "type": "cooperative_meaning",
                }
            )
            print(f"  ✓ Phase 33c: {phase33_dir.name}")
        except:
            pass

# FIS studies
for fis_type in [
    "fis_trust_hope_stabilizers",
    "fis_ai_safety_toolkit",
    "fis_emotion_cci_stabilizers",
]:
    for fis_dir in sorted(DISCOVERY.glob(f"{fis_type}_*"), reverse=True)[:1]:
        summary_file = fis_dir / "summary.json"
        if summary_file.exists():
            try:
                data = json.loads(summary_file.read_text())
                validated_studies.append(
                    {
                        "id": fis_type,
                        "path": str(fis_dir),
                        "summary": data,
                        "type": "fis_validation",
                    }
                )
                print(f"  ✓ {fis_type}: {fis_dir.name}")
            except:
                pass

print(f"\n✅ Loaded {len(validated_studies)} validated studies")

# ============================================================================
# STEP 2: Analyze validated findings and identify patterns
# ============================================================================
print("\n[2/4] Analyzing validated findings...")

insights = []

for study in validated_studies:
    summary = study["summary"]

    # Extract hypothesis test results
    if "hypothesis_test" in summary:
        ht = summary["hypothesis_test"]
        delta_cci = ht.get("mean_CCI_gain", 0)
        delta_hazard = ht.get("mean_hazard_delta", 0)
        all_passed = ht.get("all_passed", False)

        insights.append(
            {
                "study_id": study["id"],
                "study_type": study["type"],
                "delta_cci": delta_cci,
                "delta_hazard": delta_hazard,
                "validated": all_passed,
                "path": study["path"],
            }
        )

        status = "✓ VALIDATED" if all_passed else "⚠ Partial"
        print(
            f"  {status} {study['id']}: ΔCCI={delta_cci:.4f}, Δhazard={delta_hazard:.4f}"
        )

# Sort by performance
insights.sort(key=lambda x: x["delta_cci"], reverse=True)

# ============================================================================
# STEP 3: Generate hypotheses for next-wave experiments
# ============================================================================
print("\n[3/4] Generating hypotheses for next experiments...")

hypotheses = []
timestamp = int(time.time())

# Hypothesis 1: Scale up validated interventions
if any(i["validated"] for i in insights):
    best = max(insights, key=lambda x: x["delta_cci"])
    hypotheses.append(
        {
            "rank": 1,
            "id": f"autotheorist_scale_validated_{timestamp}",
            "title": "Scale Validated Interventions: Multi-Mechanism Integration",
            "hypothesis": f"Combining validated mechanisms (trust+hope+AI safety) will produce "
            f"synergistic effects exceeding individual gains (best single: ΔCCI={best['delta_cci']:.4f})",
            "rationale": "Best-performing validated study provides lower bound; integration may amplify effects",
            "method": "synthesis",
            "prediction": {
                "delta_cci_min": 0.05,  # Higher threshold for combined effects
                "delta_hazard_max": -0.02,
                "confidence": "high",
            },
            "sweep_parameters": {
                "epsilon": [0.0010, 0.0012, 0.0015],
                "trust_delta": [0.07, 0.08, 0.09],
                "hope_delta": [0.05, 0.06, 0.07],
                "ai_safety_boost": [0.02, 0.03, 0.04],
            },
            "seeds": [11, 17, 23, 29],
            "priority": "high",
        }
    )

# Hypothesis 2: Explore non-linear epsilon effects
hypotheses.append(
    {
        "rank": 2,
        "id": f"autotheorist_epsilon_nonlinear_{timestamp}",
        "title": "Non-Linear Openness Effects: Threshold Discovery",
        "hypothesis": "Openness (ε) exhibits threshold effects where benefits accelerate beyond ε≈0.0015",
        "rationale": "Linear tests showed consistent gains; non-linear exploration may reveal optimal operating points",
        "method": "parameter_expansion",
        "prediction": {
            "delta_cci_min": 0.04,
            "delta_hazard_max": -0.015,
            "confidence": "moderate",
        },
        "sweep_parameters": {
            "epsilon": [0.0008, 0.0012, 0.0015, 0.0018, 0.0020, 0.0025],
            "rho": [0.0828],  # Hold constant at optimal
            "noise": [0.03, 0.05, 0.07],  # Test robustness to noise
        },
        "seeds": [11, 17, 23, 29],
        "priority": "high",
    }
)

# Hypothesis 3: Timing-dependent interventions
hypotheses.append(
    {
        "rank": 3,
        "id": f"autotheorist_timing_critical_{timestamp}",
        "title": "Critical Window Hypothesis: Pre-Shock vs Post-Shock",
        "hypothesis": "Interventions applied BEFORE shock are 2-3x more effective than post-shock applications",
        "rationale": "All validated studies used pre-shock setup; post-shock rescue may require different parameters",
        "method": "temporal_analysis",
        "prediction": {
            "delta_cci_min": 0.03,
            "delta_hazard_max": -0.01,
            "confidence": "moderate",
        },
        "sweep_parameters": {
            "intervention_epoch": [800, 950, 1000, 1050, 1200],  # 1000 = shock
            "epsilon": [0.0012, 0.0015],
            "rescue_strength": [0.5, 1.0, 1.5],
        },
        "seeds": [11, 17, 23, 29],
        "priority": "medium",
    }
)

# Hypothesis 4: Agent diversity effects
hypotheses.append(
    {
        "rank": 4,
        "id": f"autotheorist_diversity_{timestamp}",
        "title": "Diversity Amplification: Heterogeneous Agent Benefits",
        "hypothesis": "Increasing agent diversity (skill, preference, strategy) amplifies resilience gains",
        "rationale": "Homogeneous agents in validated studies; real systems have heterogeneity",
        "method": "generalization_test",
        "prediction": {
            "delta_cci_min": 0.035,
            "delta_hazard_max": -0.012,
            "confidence": "moderate",
        },
        "sweep_parameters": {
            "diversity_index": [0.0, 0.2, 0.4, 0.6, 0.8],
            "epsilon": [0.0012],
            "agents": [150, 200, 250],  # Test scale sensitivity
        },
        "seeds": [11, 17, 23, 29],
        "priority": "medium",
    }
)

# Hypothesis 5: Cross-domain generalization
hypotheses.append(
    {
        "rank": 5,
        "id": f"autotheorist_crossdomain_{timestamp}",
        "title": "Cross-Domain Generalization: Economic Shocks",
        "hypothesis": "Validated social mechanisms generalize to economic shock scenarios with modified parameters",
        "rationale": "Extend beyond social shocks to economic/resource shocks for broader applicability",
        "method": "domain_transfer",
        "prediction": {
            "delta_cci_min": 0.025,  # Lower threshold for new domain
            "delta_hazard_max": -0.008,
            "confidence": "low",
        },
        "sweep_parameters": {
            "shock_type": ["social", "economic", "resource", "combined"],
            "epsilon": [0.0010, 0.0012, 0.0015],
            "economic_coupling": [0.3, 0.5, 0.7],
        },
        "seeds": [11, 17, 23, 29, 31],  # Extra seed for domain transfer
        "priority": "low",
    }
)

print(f"✅ Generated {len(hypotheses)} hypotheses")
for h in hypotheses:
    print(f"   {h['rank']}. {h['title']} (priority: {h['priority']})")

# ============================================================================
# STEP 4: Export preregistered study YAMLs
# ============================================================================
print("\n[4/4] Exporting preregistered study YAMLs...")

generated_yamls = []

for hyp in hypotheses[:3]:  # Top 3 get YAMLs
    yaml_path = STUDIES_DIR / f"{hyp['id']}.yml"

    yaml_content = {
        "study_id": hyp["id"],
        "title": hyp["title"],
        "version": "2.0",
        "prereg_date": datetime.now().strftime("%Y-%m-%d"),
        "preregistered": True,
        "hypothesis": {
            "text": hyp["hypothesis"],
            "rationale": hyp["rationale"],
            "method": hyp["method"],
            "rank": hyp["rank"],
        },
        "constants": {
            "seeds": hyp["seeds"],
            "agents": 200,
            "epochs": 1500,
            "noise": 0.05,
            "shock": {"epoch": 1000, "severity": 0.5},
            "analysis_window": [960, 1040],
        },
        "sweep": hyp["sweep_parameters"],
        "validation": {
            "metrics": [
                {
                    "name": "mean_CCI_gain",
                    "rule": f">= {hyp['prediction']['delta_cci_min']}",
                },
                {
                    "name": "mean_hazard_delta",
                    "rule": f"<= {hyp['prediction']['delta_hazard_max']}",
                },
            ],
            "bootstrap_ci": 1000,
            "confidence": hyp["prediction"]["confidence"],
        },
        "exports": {
            "outdir": f"discovery_results/{hyp['id']}/",
            "require_columns": ["CCI", "hazard", "survival_rate", "collapse_risk"],
        },
        "metadata": {
            "generated_by": "auto_theorist_enhanced.py",
            "generated_at": datetime.now().isoformat(),
            "priority": hyp["priority"],
            "based_on": [i["study_id"] for i in insights if i["validated"]],
        },
        "adapter": "adapters/universal_resilience_adapter.py",
    }

    yaml_path.write_text(
        yaml.safe_dump(yaml_content, sort_keys=False, default_flow_style=False)
    )
    generated_yamls.append(str(yaml_path))
    print(f"  ✓ {yaml_path.name}")

# ============================================================================
# Export queue JSON
# ============================================================================
queue_data = {
    "version": "2.0_enhanced",
    "generated_at": datetime.now().isoformat(),
    "generator": "auto_theorist_enhanced.py",
    "based_on_studies": len(validated_studies),
    "hypotheses": hypotheses,
    "generated_yamls": generated_yamls,
}

queue_file = ROOT / "autotheorist_queue.json"
queue_file.write_text(json.dumps(queue_data, indent=2))

print(f"\n✅ Queue saved: {queue_file}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ Auto-Theorist Enhanced Complete")
print("=" * 80)
print(f"📊 Analyzed: {len(validated_studies)} validated studies")
print(f"💡 Generated: {len(hypotheses)} hypotheses")
print(f"📝 Created: {len(generated_yamls)} preregistered YAMLs")
print("🎯 Priority breakdown:")
print(f"   - High: {sum(1 for h in hypotheses if h['priority']=='high')}")
print(f"   - Medium: {sum(1 for h in hypotheses if h['priority']=='medium')}")
print(f"   - Low: {sum(1 for h in hypotheses if h['priority']=='low')}")
print("\n📁 Outputs:")
print(f"   - {queue_file}")
for yf in generated_yamls:
    print(f"   - {yf}")
print("\n🚀 Next steps:")
print("   1. Review hypotheses in autotheorist_queue.json")
print("   2. Select studies to run from studies/ directory")
print("   3. Execute: python openlaws_automation.py run --study studies/[study_id].yml")
print("=" * 80)
