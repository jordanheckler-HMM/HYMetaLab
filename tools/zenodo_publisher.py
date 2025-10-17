#!/usr/bin/env python3
"""
Zenodo Publisher — Automated Publication Pipeline
Creates deposition, uploads files, publishes, and updates dashboard.

Requires:
  - ZENODO_API_TOKEN environment variable
  - Replication package ZIP file

Usage:
  python3 tools/zenodo_publisher.py --package path/to/file.zip [--sandbox]

Environment:
  ZENODO_API_TOKEN   Required. Get from zenodo.org/account/settings/applications

Flags:
  --sandbox          Use sandbox.zenodo.org instead of production
  --dry-run          Show what would be done without executing
"""

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("Error: requests library required. Install: pip3 install requests")
    sys.exit(1)


class ZenodoPublisher:
    def __init__(self, api_token: str, sandbox: bool = False):
        self.api_token = api_token
        self.base_url = (
            "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
        )
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.sandbox = sandbox

    def create_deposition(self, metadata: dict) -> tuple[str, str]:
        """Create new deposition and return (deposition_id, concept_doi)"""
        url = f"{self.base_url}/deposit/depositions"

        data = {"metadata": metadata}
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()

        result = response.json()
        deposition_id = result["id"]
        concept_doi = result.get("conceptdoi", "").replace("https://doi.org/", "")

        print(f"✅ Created deposition: {deposition_id}")
        print(f"   Concept DOI: {concept_doi}")

        return str(deposition_id), concept_doi

    def upload_file(self, deposition_id: str, filepath: pathlib.Path) -> None:
        """Upload file to deposition"""
        url = f"{self.base_url}/deposit/depositions/{deposition_id}/files"

        with open(filepath, "rb") as f:
            data = {"name": filepath.name}
            files = {"file": f}
            response = requests.post(url, data=data, files=files, headers=self.headers)
            response.raise_for_status()

        print(
            f"✅ Uploaded: {filepath.name} ({filepath.stat().st_size / 1024 / 1024:.2f} MB)"
        )

    def publish_deposition(self, deposition_id: str) -> str:
        """Publish deposition and return version DOI"""
        url = f"{self.base_url}/deposit/depositions/{deposition_id}/actions/publish"

        response = requests.post(url, headers=self.headers)
        response.raise_for_status()

        result = response.json()
        version_doi = result.get("doi", "").replace("https://doi.org/", "")

        print(f"✅ Published deposition: {deposition_id}")
        print(f"   Version DOI: {version_doi}")

        return version_doi

    def update_dashboard(self, concept_doi: str, version_doi: str) -> None:
        """Update dashboard_data.json with DOIs"""
        dash_path = pathlib.Path("dashboard_data.json")
        data = json.loads(dash_path.read_text())

        data["zenodo_concept_doi"] = concept_doi
        data["zenodo_version_doi"] = version_doi
        data["zenodo_status"] = "published"

        dash_path.write_text(json.dumps(data, indent=2) + "\n")
        print("✅ Updated dashboard_data.json")

    def update_integrity_log(self, concept_doi: str, version_doi: str) -> None:
        """Add entry to SHA256_log.md"""
        log_path = pathlib.Path("docs/integrity/SHA256_log.md")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d")

        entry = f"\n- {timestamp}: Published to Zenodo. Concept: {concept_doi}, Version: {version_doi}\n"

        with open(log_path, "a") as f:
            f.write(entry)

        print("✅ Updated docs/integrity/SHA256_log.md")


def get_metadata_from_dashboard() -> dict:
    """Extract metadata from dashboard for Zenodo"""
    dash_path = pathlib.Path("dashboard_data.json")
    data = json.loads(dash_path.read_text())

    phase4 = data.get("phase4_open_data_integration", {})

    metadata = {
        "title": "Phase 4 Open-Source Data Integration - Replication Package",
        "upload_type": "dataset",
        "description": (
            "Replication package for Phase 4 Open-Source Data Integration study. "
            "Includes standardized datasets, configuration manifests, synthesis outputs, "
            "and analysis scripts. Guardian v4 score: 87.0/100. "
            "TruthLens: 1.000. MeaningForge: 1.000."
        ),
        "creators": [
            {
                "name": "HYMetaLab",
                "affiliation": "Heck Yeah Simulation Research Initiative",
            }
        ],
        "keywords": [
            "open data",
            "reproducibility",
            "ethics validation",
            "truth index",
            "semantic analysis",
            "data integration",
        ],
        "access_right": "open",
        "license": "CC-BY-4.0",
        "version": "1.0",
    }

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Publish package to Zenodo")
    parser.add_argument("--package", required=True, help="Path to ZIP package")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    args = parser.parse_args()

    # Get API token
    api_token = os.getenv("ZENODO_API_TOKEN")
    if not api_token:
        print("Error: ZENODO_API_TOKEN environment variable not set")
        print("Get token from: https://zenodo.org/account/settings/applications")
        sys.exit(1)

    # Verify package exists
    package_path = pathlib.Path(args.package)
    if not package_path.exists():
        print(f"Error: Package not found: {package_path}")
        sys.exit(1)

    print("🚀 Zenodo Publisher")
    print(f"   Mode: {'SANDBOX' if args.sandbox else 'PRODUCTION'}")
    print(f"   Package: {package_path}")
    print(f"   Size: {package_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    if args.dry_run:
        print("🔍 DRY RUN - No changes will be made")
        print()
        print("Would execute:")
        print("  1. Create deposition with metadata")
        print(f"  2. Upload {package_path.name}")
        print("  3. Publish deposition")
        print("  4. Update dashboard_data.json")
        print("  5. Update docs/integrity/SHA256_log.md")
        return

    try:
        publisher = ZenodoPublisher(api_token, args.sandbox)
        metadata = get_metadata_from_dashboard()

        print("📝 Step 1/5: Creating deposition...")
        deposition_id, concept_doi = publisher.create_deposition(metadata)
        print()

        print("📤 Step 2/5: Uploading package...")
        publisher.upload_file(deposition_id, package_path)
        print()

        print("🚀 Step 3/5: Publishing deposition...")
        version_doi = publisher.publish_deposition(deposition_id)
        print()

        print("📊 Step 4/5: Updating dashboard...")
        publisher.update_dashboard(concept_doi, version_doi)
        print()

        print("📝 Step 5/5: Updating integrity log...")
        publisher.update_integrity_log(concept_doi, version_doi)
        print()

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  ✅ ZENODO PUBLICATION COMPLETE                             ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        print(f"🔗 Concept DOI: {concept_doi}")
        print(f"📄 Version DOI: {version_doi}")
        print(f"🌐 URL: https://doi.org/{version_doi}")
        print()
        print("✅ Dashboard and logs updated automatically")

    except requests.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response is not None:
            print(f"   Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
