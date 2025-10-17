#!/usr/bin/env bash
# Convenience wrapper for Zenodo Link Guard
# Usage: ./tools/run_zenodo_guard.sh [--watch]

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--watch" ]]; then
    echo "🔍 Watching Zenodo status (checking every 5 minutes)..."
    echo "   Press Ctrl+C to stop"
    echo ""
    while true; do
        python3 tools/zenodo_link_guard.py
        sleep 300  # 5 minutes
    done
else
    echo "🔍 Checking Zenodo status (one-time)..."
    python3 tools/zenodo_link_guard.py
fi
