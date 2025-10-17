#!/bin/zsh
set -euo pipefail

echo "🧹 Cleaning up old terminals and zombie python processes..."
pkill -f openlaws_automation.py 2>/dev/null || true
pkill -f python 2>/dev/null || true

echo "🔧 Resetting VSCode shell environment..."
export SHELL="/bin/zsh"
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "export PYTHONPATH=\"$PWD:\$PYTHONPATH\"" >> .env 2>/dev/null || true

# make sure we’re in project root
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# re-create folders if needed
mkdir -p adapters ops studies exports data meta || true

# if a venv exists, activate it; if not, create one
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "🐍 Creating new virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
fi

# sanity-check python
echo "Python path: $(which python)"
python -V || true

# confirm adapter + ops imports
python - <<'PY'
print("⚙️  Checking imports...")
try:
    import adapters.sim_adapter as a
    print(" - adapters.sim_adapter:", hasattr(a,"run_sim"))
except Exception as e:
    print(" - adapter issue:", e)
try:
    import ops.maintenance as m
    print(" - ops.maintenance:", hasattr(m,"main"))
except Exception as e:
    print(" - maintenance issue:", e)
print("✅ Environment looks good if you see True above.")
PY

echo ""
echo "💡 Terminal reset complete."
echo "Try running:"
echo "  python ops/maintenance.py health"
echo "Then re-run your study, e.g.:"
echo "  python openlaws_automation.py run --study fis_trust_hope_stabilizers --epsilon 0.001 --seeds 11 --shock 0.5"
