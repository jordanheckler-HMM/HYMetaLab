#!/usr/bin/env python3
"""
HYMetaLab Preregistration Manifest Generator
Creates preregister.yml for Guardian v4 validation
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="preregister.yml", help="Output file type")
args = parser.parse_args()

print("📝 HYMetaLab Preregistration Manifest Generator")
print("=" * 80)

# Load datasets manifest
manifest_path = Path("open_data/datasets_manifest.yml")
with open(manifest_path) as f:
    datasets_manifest = yaml.safe_load(f)

# Load ETL processing log
log_path = Path("open_data/etl_processing_log.json")
if log_path.exists():
    with open(log_path) as f:
        etl_log = json.load(f)
else:
    etl_log = {"processed_datasets": []}

# Create preregistration document
prereg = {
    "preregistration": {
        "project": datasets_manifest["project"],
        "version": datasets_manifest["version"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "guardian_version": "v4",
        "openlaws_version": "1.0",
    },
    "datasets": [],
    "methodology": {
        "pipeline": datasets_manifest["pipeline"],
        "etl_tool": "etl_open.py v1.0",
        "standardization": {
            "schema": "HYMetaLab Standard Schema v1.0",
            "required_columns": [
                "dataset_id",
                "record_id",
                "timestamp",
                "country",
                "sample_weight",
            ],
            "cci_mappings": [
                "trust_score → Connection proxy",
                "wellbeing_score → Coherence proxy",
                "collaboration_score → Connection proxy",
                "information_access → Information flow proxy",
            ],
        },
        "quality_checks": [
            "License compatibility (tools/license_checker.py)",
            "Missing data threshold < 10%",
            "CCI mappability verification",
            "SHA256 integrity hashing",
        ],
    },
    "hypotheses": [],
    "governance": datasets_manifest.get("governance", {}),
    "integrity": {
        "deterministic_processing": True,
        "full_provenance": True,
        "sha256_checksums": True,
        "guardian_threshold_phase2": 80,
        "guardian_threshold_publication": 90,
    },
}

# Add each dataset
for ds_cfg in datasets_manifest["datasets"]:
    ds_entry = {
        "id": ds_cfg["id"],
        "name": ds_cfg["name"],
        "theme": ds_cfg["theme"],
        "license": ds_cfg["source"]["license"],
        "size_gb": ds_cfg["source"]["size_estimate_gb"],
        "cci_mapping": ds_cfg["relevance_to_framework"]["cci_mapping"],
        "hypothesis_potential": ds_cfg["relevance_to_framework"][
            "hypothesis_potential"
        ],
        "status": ds_cfg.get("intake_status", "pending"),
    }
    prereg["datasets"].append(ds_entry)

# Add hypotheses for each dataset
for ds_cfg in datasets_manifest["datasets"]:
    hyp = {
        "dataset": ds_cfg["id"],
        "hypothesis": ds_cfg["relevance_to_framework"]["hypothesis_potential"],
        "prediction": "Validated dataset will enable testing Universal Resilience Law predictions",
        "metrics": [
            "delta_cci",
            "delta_hazard",
            "trust_correlation",
            "wellbeing_correlation",
        ],
    }
    prereg["hypotheses"].append(hyp)

# Write preregistration YAML
output_path = Path("open_data") / args.type
with open(output_path, "w") as f:
    yaml.dump(prereg, f, sort_keys=False, default_flow_style=False)

print(f"✅ Preregistration manifest generated: {output_path}")
print(f"   Datasets: {len(prereg['datasets'])}")
print(f"   Hypotheses: {len(prereg['hypotheses'])}")
print("\n📁 Next step: Run Guardian v4 validation")
print(
    f"   python3 qc/guardian_v4/guardian_v4.py --validate --file {output_path} --report"
)
