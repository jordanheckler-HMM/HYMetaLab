"""Create a sanitized `github_release/` folder containing only code and docs safe to publish.

Usage:
  python -m tools.prepare_github_release

The script copies a curated list of files and directories into `github_release/` while omitting
large/run-time output directories (e.g., `discovery_results/`, `outputs/`). It also optionally
exports public artifacts from a chosen run using `tools/export_public_artifacts.py`.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(".").resolve()
OUT = ROOT / "github_release"

# Curated include list (relative to repo root). Edit this list to add/remove files.
INCLUDES = [
    "adapters",
    "studies",
    "tools/export_public_artifacts.py",
    "tools/prepare_github_release.py",
    "requirements.txt",
    "README.md",
    ".github/workflows/ci.yml",
    "LICENSE",
    "CONTRIBUTING.md",
    "openlaws_automation.py",
    "studies/creator_emergence.yml",
]

# Paths that should never be copied
EXCLUDE_DIRS = ["discovery_results", "outputs", "results"]


def copy_item(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_release(export_run: str = None):
    if OUT.exists():
        print(f"Removing existing {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for rel in INCLUDES:
        s = ROOT / rel
        if not s.exists():
            print(f"[warn] {s} not found — skipping")
            continue
        # reject copying from excluded dirs
        if any(part in EXCLUDE_DIRS for part in s.parts):
            print(f"[skip] {s} is in exclude list")
            continue
        dst = OUT / rel
        try:
            copy_item(s, dst)
            print(f"Copied {s} -> {dst}")
        except Exception as e:
            print(f"Error copying {s}: {e}")

    # add a small release README
    release_readme = OUT / "README.md"
    release_readme.write_text(
        "# Public release bundle\n\nThis folder contains the curated code and docs meant for public sharing.\n"
    )

    # optional: export public artifacts from a run
    if export_run:
        from tools.export_public_artifacts import export

        src_run = ROOT / export_run
        if src_run.exists():
            export(src_run, out_base=OUT / "public_artifacts")
        else:
            print(f"[warn] export_run {src_run} not found")

    print("\nRelease prepared at:", OUT)
    return OUT


if __name__ == "__main__":
    export_run = None
    if len(sys.argv) > 1:
        export_run = sys.argv[1]
    build_release(export_run)
