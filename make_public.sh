#!/usr/bin/env bash
set -euo pipefail

# Usage: ./make_public.sh <study_yaml> <branch-name> [--export-run <discovery_run_dir>]
STUDY=${1:-}
BRANCH=${2:-public-release}
EXPORT_RUN_DIR=${3:-}

if [ -z "$STUDY" ]; then
  echo "Usage: $0 <study_yaml> <branch-name> [export_run_dir]"
  exit 1
fi

python3 openlaws_automation.py auto --study "$STUDY"

if [ -n "$EXPORT_RUN_DIR" ]; then
  python3 -m tools.export_public_artifacts "$EXPORT_RUN_DIR"
fi

python3 tools/prepare_github_release.py

cd github_release
git init || true
git checkout -b "$BRANCH" || git checkout "$BRANCH" || true
git add -A
git commit -m "Public release bundle: $(date -u +%Y%m%d)" || true
echo "Prepared branch '$BRANCH' in github_release/. Review and push to your remote with:" 
echo "  cd github_release && git remote add origin git@github.com:<org>/<repo>.git && git push -u origin $BRANCH"