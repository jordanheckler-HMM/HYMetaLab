#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== AutoLab Desktop smoke check =="
echo "Project: $ROOT_DIR"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not installed."
  exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required but not installed."
  exit 1
fi

echo "1) Installing Node dependencies"
npm install --no-audit --no-fund

echo "2) Running Rust compile check"
cargo check --manifest-path src-tauri/Cargo.toml

if [[ "${1:-}" != "--build" ]]; then
  echo "Smoke check complete. To run full bundle build:"
  echo "  npm run smoke:build"
  exit 0
fi

echo "3) Running full Tauri build"
npm run tauri:build

BUNDLE_DIR="src-tauri/target/release/bundle"
if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "Build finished but bundle directory not found at $BUNDLE_DIR"
  exit 1
fi

echo "4) Bundle artifacts"
find "$BUNDLE_DIR" -maxdepth 3 -type f | sed "s|^|  - |"
echo "Smoke build complete."
