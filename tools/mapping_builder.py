#!/usr/bin/env python3
"""
HYMetaLab Mapping Builder
Creates mapping.yml that links standardized datasets to Universal Resilience Law framework
Phase 3: Mapping & Validation
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Standardized dataset directory")
parser.add_argument("--output", required=True, help="Output mapping.yml path")
args = parser.parse_args()

print("🗺️  HYMetaLab Mapping Builder v1.0")
print("=" * 80)

DATASET_DIR = Path(args.dataset)
OUTPUT_PATH = Path(args.output)

# Load standardized datasets to inspect schema
standardized_files = list(DATASET_DIR.glob("*_standardized.csv"))

if not standardized_files:
    print(f"❌ No standardized datasets found in {DATASET_DIR}")
    exit(1)

print(f"📊 Found {len(standardized_files)} standardized datasets:")
for f in standardized_files:
    print(f"   - {f.name}")

# Load first dataset to get schema
sample_df = pd.read_csv(standardized_files[0])
columns = list(sample_df.columns)

print("\n✅ Standard schema detected:")
print(f"   Columns: {len(columns)}")
print(f"   Sample: {columns[:5]}...")

# Create mapping.yml
mapping = {
    "mapping_version": "1.0",
    "generated_at": datetime.now().isoformat(),
    "framework": "Universal Resilience Law",
    "formula": "R ∝ (ε × CCI) / η",
    "datasets": [],
    "column_mappings": {
        "trust_score": {
            "framework_component": "Connection (C)",
            "description": "Social trust proxy for interpersonal connection strength",
            "normalization": "0-1 scale",
            "weight": 0.3,
            "resilience_contribution": "High trust increases network resilience to shocks",
        },
        "wellbeing_score": {
            "framework_component": "Coherence (I)",
            "description": "Life satisfaction proxy for internal coherence",
            "normalization": "0-1 scale",
            "weight": 0.25,
            "resilience_contribution": "Well-being buffers against hazard accumulation",
        },
        "collaboration_score": {
            "framework_component": "Connection (C)",
            "description": "Cooperative behavior proxy for collective action capacity",
            "normalization": "0-1 scale",
            "weight": 0.25,
            "resilience_contribution": "Collaboration amplifies coordinated shock responses",
        },
        "information_access": {
            "framework_component": "Information Flow (I)",
            "description": "Knowledge sharing proxy for information efficiency",
            "normalization": "0-1 scale",
            "weight": 0.2,
            "resilience_contribution": "Information access reduces uncertainty and enables adaptation",
        },
    },
    "cci_calculation": {
        "formula": "CCI = w1·trust + w2·wellbeing + w3·collaboration + w4·information",
        "weights": {
            "trust_score": 0.3,
            "wellbeing_score": 0.25,
            "collaboration_score": 0.25,
            "information_access": 0.2,
        },
        "normalization": "All inputs normalized to [0,1]; output in [0,1]",
        "validation": "CCI must be in [0,1]; reject if out of bounds",
    },
    "hypothesis_tests": [],
    "quality_gates": {
        "minimum_records": 100,
        "maximum_missing_pct": 0.10,
        "cci_range": [0.0, 1.0],
        "correlation_threshold": 0.15,  # Minimum correlation between CCI proxies
    },
    "metadata": {
        "pipeline": "Guardian → TruthLens → MeaningForge → OriginChain → Aletheia",
        "openlaws_version": "1.0",
        "guardian_version": "v4",
        "preregistration": "open_data/preregister.yml",
    },
}

# Add each dataset
for f in standardized_files:
    ds_id = f.stem.replace("_standardized", "")
    mapping["datasets"].append(
        {
            "id": ds_id,
            "file": str(f.name),
            "columns": columns,
            "cci_columns": [
                "trust_score",
                "wellbeing_score",
                "collaboration_score",
                "information_access",
            ],
        }
    )

# Add hypothesis tests
for f in standardized_files:
    ds_id = f.stem.replace("_standardized", "")
    mapping["hypothesis_tests"].append(
        {
            "dataset": ds_id,
            "H0": "CCI has no relationship with resilience indicators",
            "H1": "Higher CCI predicts improved resilience (lower hazard, higher survival)",
            "test_metrics": ["cci_trust_correlation", "cci_wellbeing_correlation"],
            "acceptance_threshold": "correlation > 0.15 (small effect size)",
            "preregistered": True,
        }
    )

# Write mapping.yml
with open(OUTPUT_PATH, "w") as f:
    yaml.dump(mapping, f, sort_keys=False, default_flow_style=False)

print(f"\n✅ Mapping generated: {OUTPUT_PATH}")
print(f"   Datasets mapped: {len(mapping['datasets'])}")
print(f"   Column mappings: {len(mapping['column_mappings'])}")
print(f"   Hypothesis tests: {len(mapping['hypothesis_tests'])}")
print("\n📁 Next step: Guardian v4 validation")
print(
    f"   python3 qc/guardian_v4/guardian_v4.py --validate --file {OUTPUT_PATH} --report"
)
