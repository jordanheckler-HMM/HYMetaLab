#!/usr/bin/env python3
"""
Create a replication packet for a completed study.

Usage: python tools/make_replication_packet.py --study STUDY_ID --zip OUTPUT.zip
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def create_replication_packet(study_id, output_zip):
    """Create a comprehensive replication packet."""
    print(f"Creating replication packet for: {study_id}")
    print("=" * 60)

    # Base paths
    study_results = Path(f"results/discovery_results/{study_id}")

    # Try timestamped directory if standard path doesn't exist
    if not study_results.exists():
        discovery_dir = Path("discovery_results")
        if discovery_dir.exists():
            timestamped_dirs = sorted(
                discovery_dir.glob(f"{study_id}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if timestamped_dirs:
                study_results = timestamped_dirs[0]
                print(f"ℹ Using timestamped results: {study_results}")

    study_file = Path(f"studies/{study_id}.yml")
    temp_dir = Path(f"temp_replication_{study_id}")

    # Create temp directory
    temp_dir.mkdir(exist_ok=True)

    # Copy study configuration
    if study_file.exists():
        shutil.copy(study_file, temp_dir / f"{study_id}.yml")
        print(f"✓ Copied study configuration: {study_file}")
    else:
        print(f"⚠ Study file not found: {study_file}")

    # Copy results directory if it exists
    if study_results.exists():
        shutil.copytree(study_results, temp_dir / "results", dirs_exist_ok=True)
        print(f"✓ Copied results: {study_results}")
    else:
        print(f"⚠ Results directory not found: {study_results}")

    # Create metadata file
    metadata = {
        "study_id": study_id,
        "created": datetime.now().isoformat(),
        "package_version": "1.0",
        "contents": {"study_config": f"{study_id}.yml", "results": "results/"},
    }

    with open(temp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Created metadata.json")

    # Create ZIP archive
    output_path = Path(output_zip)
    shutil.make_archive(str(output_path.with_suffix("")), "zip", temp_dir)
    print(f"✓ Created archive: {output_path}")

    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    print("✓ Cleaned up temporary files")

    print("=" * 60)
    print(f"✅ Replication packet created: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create study replication packet")
    parser.add_argument("--study", required=True, help="Study ID")
    parser.add_argument("--zip", required=True, help="Output ZIP filename")

    args = parser.parse_args()

    output_path = create_replication_packet(args.study, args.zip)

    print()
    print("Next steps:")
    print(f"  shasum -a 256 {output_path}")
    print(f"  # Upload {output_path} to Zenodo or archive")


if __name__ == "__main__":
    main()
