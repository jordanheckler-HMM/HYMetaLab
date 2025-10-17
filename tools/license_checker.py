#!/usr/bin/env python3
"""
HYMetaLab License Checker
Verifies open-source license compatibility for dataset integration
Guardian v4 + OpenLaws Standards
"""
import sys
from pathlib import Path

import yaml

# Approved open licenses for HYMetaLab integration
APPROVED_LICENSES = {
    "CC-BY-4.0": {"commercial": True, "attribution": True, "share_alike": False},
    "CC-BY": {"commercial": True, "attribution": True, "share_alike": False},
    "CC-BY-SA": {"commercial": True, "attribution": True, "share_alike": True},
    "CC-BY-NC": {
        "commercial": False,
        "attribution": True,
        "share_alike": False,
    },  # Research only
    "CC-BY-NC-SA": {
        "commercial": False,
        "attribution": True,
        "share_alike": True,
    },  # Research only
    "CC0": {"commercial": True, "attribution": False, "share_alike": False},
    "MIT": {"commercial": True, "attribution": True, "share_alike": False},
    "Public Domain": {"commercial": True, "attribution": False, "share_alike": False},
}

RESEARCH_ONLY_LICENSES = ["CC-BY-NC", "CC-BY-NC-SA"]

print("🔒 HYMetaLab License Checker v1.0")
print("=" * 80)

# Load datasets manifest
manifest_path = Path("open_data/datasets_manifest.yml")

if not manifest_path.exists():
    print(f"❌ Error: Manifest not found at {manifest_path}")
    print("   Run Phase 1 dataset registration first.")
    sys.exit(1)

with open(manifest_path) as f:
    manifest = yaml.safe_load(f)

print(f"📋 Loaded manifest: {manifest['project']}")
print(f"   Version: {manifest['version']}")
print(f"   Datasets: {len(manifest['datasets'])}")
print()

# Check each dataset
results = []
all_approved = True

for i, dataset in enumerate(manifest["datasets"], 1):
    ds_id = dataset["id"]
    ds_name = dataset["name"]
    license_str = dataset["source"]["license"]

    print(f"\n[{i}/{len(manifest['datasets'])}] {ds_id}")
    print(f"   Name: {ds_name}")
    print(f"   License: {license_str}")

    # Parse license (handle variants)
    license_normalized = None
    for approved_lic in APPROVED_LICENSES.keys():
        if approved_lic.lower() in license_str.lower():
            license_normalized = approved_lic
            break

    if license_normalized:
        lic_info = APPROVED_LICENSES[license_normalized]
        is_research_only = license_normalized in RESEARCH_ONLY_LICENSES

        status = "✅ APPROVED"
        if is_research_only:
            status += " (RESEARCH ONLY)"

        print(f"   Status: {status}")
        print("   Details:")
        print(
            f"     - Commercial use: {'❌ No' if not lic_info['commercial'] else '✅ Yes'}"
        )
        print(
            f"     - Attribution required: {'✅ Yes' if lic_info['attribution'] else '❌ No'}"
        )
        print(f"     - Share-alike: {'✅ Yes' if lic_info['share_alike'] else '❌ No'}")

        results.append(
            {
                "id": ds_id,
                "license": license_normalized,
                "approved": True,
                "research_only": is_research_only,
                "details": lic_info,
            }
        )
    else:
        print("   Status: ❌ NOT APPROVED")
        print(f"   Reason: License '{license_str}' not in approved list")
        all_approved = False

        results.append(
            {
                "id": ds_id,
                "license": license_str,
                "approved": False,
                "research_only": None,
                "details": None,
            }
        )

# Summary
print("\n" + "=" * 80)
print("📊 LICENSE CHECK SUMMARY")
print("=" * 80)

approved_count = sum(1 for r in results if r["approved"])
research_only_count = sum(1 for r in results if r.get("research_only", False))

print(f"Total datasets: {len(results)}")
print(f"Approved: {approved_count}/{len(results)}")
print(f"Research-only licenses: {research_only_count}")

if research_only_count > 0:
    print(
        f"\n⚠️  WARNING: {research_only_count} dataset(s) restricted to research use only"
    )
    print("   These datasets cannot be used for commercial applications")

if all_approved:
    print("\n✅ All datasets have approved licenses!")
    print("   Proceeding to Phase 2 (Ingestion & Standardization)")
    sys.exit(0)
else:
    print("\n❌ Some datasets have unapproved licenses")
    print("   Remove or replace these datasets before proceeding")
    sys.exit(1)
