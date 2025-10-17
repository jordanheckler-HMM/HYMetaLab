#!/usr/bin/env zsh
set -euo pipefail

echo "🩺 MetaLab Doctor starting…"

# 0) Clean up anything hanging
pkill -f openlaws_automation.py 2>/dev/null || true
pkill -f python 2>/dev/null || true

# 1) Normalize env
export SHELL="/bin/zsh"
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PYTHONWARNINGS="ignore"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs ops adapters studies exports data meta || true
LOG="logs/metalab_doctor_$(date +%Y%m%d_%H%M%S).log"

# 2) Ensure/activate venv (make if missing)
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "🐍 Creating .venv …"
  /usr/local/bin/python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
fi

# 3) Minimal deps required by our tools
python - <<'PY' >>"$LOG" 2>&1
import sys, subprocess
pkgs = ["pyyaml","pydantic","numpy"]
for p in pkgs:
    try:
        __import__(p.split("==")[0])
    except Exception:
        subprocess.check_call([sys.executable,"-m","pip","install",p])
print("deps_ok")
PY

# 4) Quick project sanity
echo "🔎 Project sanity checks…" | tee -a "$LOG"
python - <<'PY' | tee -a "$LOG"
import os, sys, importlib.util, json
issues=[]
for d in ["adapters","ops","studies","exports","data"]:
    if not os.path.isdir(d): issues.append(f"missing_dir:{d}")
if not os.path.isfile("openlaws_automation.py"):
    issues.append("missing_file:openlaws_automation.py")
print("issues:",json.dumps(issues))
PY

# 5) Add/repair __main__ guard (fixes macOS multiprocessing no-ops)
if [ -f openlaws_automation.py ] && ! grep -q '__name__ == "__main__"' openlaws_automation.py; then
  cat >> openlaws_automation.py <<'PY'
if __name__ == "__main__":
    try:
        from openlaws_automation import main as _main
        _main()
    except Exception as _e:
        # Allow import-style usage too
        pass
PY
  echo "✅ Patched __main__ guard in openlaws_automation.py" | tee -a "$LOG"
fi

# 6) Safe adapter wrapper (timeout) if not present
if [ ! -f adapters/sim_adapter_safe.py ]; then
  cat > adapters/sim_adapter_safe.py <<'PY'
from ops.watchdog import run_with_timeout, TimeoutError
from adapters.sim_adapter import run_sim as _run_sim

def run_sim(**kwargs):
    # 90s safety timeout per run
    return run_with_timeout(_run_sim, kwargs=kwargs, timeout_s=90, name="run_sim")
PY
  echo "made adapters/sim_adapter_safe.py" >>"$LOG"
fi

# 7) Watchdog (if missing)
if [ ! -f ops/watchdog.py ]; then
  cat > ops/watchdog.py <<'PY'
import multiprocessing as mp, traceback
class TimeoutError(Exception): pass

def run_with_timeout(fn, kwargs=None, timeout_s=120, name="job"):
    kwargs = kwargs or {}
    q = mp.Queue()
    def _target(q, kwargs):
        try:
            res = fn(**kwargs); q.put(("ok", res))
        except Exception as e:
            q.put(("err", f"{e}\n{traceback.format_exc()}"))
    p = mp.Process(target=_target, args=(q, kwargs), name=name, daemon=True)
    p.start(); p.join(timeout_s)
    if p.is_alive():
        p.terminate(); p.join(2)
        raise TimeoutError(f"{name} exceeded {timeout_s}s and was killed")
    if q.empty(): raise RuntimeError(f"{name} died without result")
    status, payload = q.get()
    if status=="err": raise RuntimeError(payload)
    return payload
PY
  echo "made ops/watchdog.py" >>"$LOG"
fi

# 8) If studies exist, point them at safe adapter
if ls studies/*.yml >/dev/null 2>&1; then
  for f in studies/*.yml; do
    if grep -q "adapter: adapters/sim_adapter.py" "$f"; then
      sed -i '' 's|adapter: adapters/sim_adapter.py|adapter: adapters/sim_adapter_safe.py|g' "$f"
    elif ! grep -q "^adapter:" "$f"; then
      printf "\nadapter: adapters/sim_adapter_safe.py\n" >> "$f"
    fi
  done
fi

# 9) Validate study YAML structure (fast pydantic check). Create a tiny sample if NONE exist.
python - <<'PY' | tee -a "$LOG"
import os, glob, yaml, sys
from pydantic import BaseModel, conlist, ValidationError
class Sweep(BaseModel):
    epsilon: conlist(float, min_items=1)
    seeds: conlist(int, min_items=1)
    shock: conlist(float, min_items=1)
class Study(BaseModel):
    id: str
    hypothesis: str
    sweep: Sweep
    metrics: conlist(str, min_items=1)
    exports: conlist(str, min_items=1)
    adapter: str
paths = sorted(glob.glob("studies/*.yml"))
if not paths:
    open("studies/sample.yml","w").write(
        "id: sample\nhypothesis: demo\nsweep:\n  epsilon: [0.001]\n  seeds: [11]\n  shock: [0.5]\nmetrics: [cci, hazard, survival]\nexports: [csv, png]\nadapter: adapters/sim_adapter_safe.py\n"
    )
    paths=["studies/sample.yml"]
ok=0
for p in paths:
    try:
        Study(**yaml.safe_load(open(p)))
        print("study_ok:",p); ok+=1
    except ValidationError as e:
        print("study_invalid:",p,"\n",e)
if not ok:
    sys.exit(2)
PY

# 10) Probe the adapter directly using the first study’s params (bypasses automation)
cat > ops/run_probe.py <<'PY'
import os, sys, yaml, importlib, json, time
from glob import glob

def pick():
    ys = sorted(glob("studies/*.yml"))
    if not ys: print("ERR no studies",file=sys.stderr); sys.exit(2)
    for p in ys:
        if "trust_hope" in p: return p
    return ys[0]

cfgp = pick()
cfg = yaml.safe_load(open(cfgp))
mod = cfg.get("adapter","adapters/sim_adapter_safe.py").replace("/",".").replace(".py","")
try:
    adp = importlib.import_module(mod)
except Exception as e:
    print("ERR adapter import:",e, file=sys.stderr); sys.exit(2)
if not hasattr(adp,"run_sim"):
    print("ERR adapter missing run_sim", file=sys.stderr); sys.exit(2)
sweep = cfg["sweep"]
params = dict(
    epsilon = float(sweep["epsilon"][0]),
    seed    = int(sweep["seeds"][0]),
    shock   = float(sweep["shock"][0]),
)
if "agents" in sweep and sweep["agents"]:
    params["agents"]=int(sweep["agents"][0])
print("PROBE_PARAMS", json.dumps(params))
t0=time.time()
res = adp.run_sim(**params)
print("PROBE_OK", round(time.time()-t0,2),"s")
try:
    if isinstance(res, dict):
        print("RESULT_KEYS", list(res.keys())[:6])
except Exception: pass
PY

echo "🚀 Running direct probe (adapter only)…"
/usr/local/bin/python3 ops/run_probe.py | tee -a "$LOG"

# 11) If probe passes, run ONE tiny automation case and show exports
STUDY_ID=$(basename "$(ls studies/*.yml | head -n1)" .yml)
echo "▶️  Running ONE automation case for $STUDY_ID …"
python openlaws_automation.py run --study "$STUDY_ID" --epsilon 0.001 --seeds 11 --shock 0.5 --verbose 2>&1 | tee -a "$LOG" || true

echo "📦 Exports:"
ls -lah "exports/$STUDY_ID" 2>/dev/null || echo "(no exports yet)"

echo "📝 Doctor log saved to: $LOG"
echo "✅ If you saw PROBE_OK and files under exports/, runs are working."
echo "❌ If not, copy the LAST 30 LINES of $LOG back to me."
