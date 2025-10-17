# HYMetaLab — Quick Start Guide

## 🚀 One-Click Launch (macOS)

**Desktop Launcher**: Double-click `HYMetaLab_Start.command` on your Desktop

This will automatically:
1. ✅ Start Sentinel API (Terminal 1) → http://localhost:8000
2. ✅ Start Sentinel UI (Terminal 2) → http://localhost:8501
3. ✅ Run Lab Techs runner (Terminal 3)
4. ✅ Open dashboard in your browser

---

## 📋 Manual Launch

### Option 1: Full Stack (API + UI + Runner)

**Terminal 1 — Sentinel API:**
```bash
cd ~/HYMetaLab/HYMetaLab_Sentinel
make api
# or: uvicorn apps.sentinel_api:app --reload
```

**Terminal 2 — Sentinel UI:**
```bash
cd ~/HYMetaLab/HYMetaLab_Sentinel
make ui
# or: streamlit run apps/sentinel_ui.py
```

**Terminal 3 — Lab Techs Runner:**
```bash
cd ~/HYMetaLab
python3 lab_techs_runner.py
```

### Option 2: Standalone Mode (No API)

```bash
cd ~/HYMetaLab
python3 lab_techs_runner.py
```

Uses local Guardian stub (no API required).

---

## 🧪 Run Tests

```bash
cd ~/HYMetaLab
pytest tests/test_runner_integration.py -v
```

Expected: 4/4 tests passing

---

## 🛠️ Customize Lab Techs Runner

Edit `lab_techs_runner.py` around line 48:

```python
def run_task() -> dict:
    """
    Replace with your experiment logic:
    - Import your modules
    - Run simulations
    - Process data
    """
    # Example: run a script
    import subprocess
    result = subprocess.run(
        ["python3", "your_experiment.py"],
        capture_output=True
    )
    return {"output": result.stdout, "exitcode": result.returncode}
```

---

## 📊 Access Dashboards

| Service | URL | Purpose |
|---------|-----|---------|
| **Sentinel API** | http://localhost:8000 | Validation endpoint |
| **Sentinel UI** | http://localhost:8501 | Monitoring dashboard |
| **API Health** | http://localhost:8000/health | Status check |
| **API Docs** | http://localhost:8000/docs | FastAPI interactive docs |

---

## 🔍 Check Logs

```bash
# Task execution log
cat Task_Log.md

# Latest integrity alert (if any)
cat dev/logs/Integrity_Alert.md

# Run reports
cat Run_Report.md

# Checksums
cat Checksums.csv
```

---

## 🎯 Validation Thresholds

**Current settings** (`lab_techs_runner.py`):
```python
PASS_MIN = 0.85   # Coherence threshold
NOISE_MAX = 0.20  # Noise threshold
```

**To adjust** (requires Lab Manager approval):
```python
PASS_MIN = 0.90   # Stricter
NOISE_MAX = 0.10  # Tighter noise bounds
```

---

## 🔄 Git Workflow

```bash
# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feat/my-experiment

# Make changes, then commit
git add .
git commit -m "feat: description"

# Push to GitHub
git push -u origin feat/my-experiment

# Create PR on GitHub
```

---

## 📦 Update Dependencies

```bash
# HYMetaLab Sentinel
cd ~/HYMetaLab/HYMetaLab_Sentinel
pip install -r requirements.txt --upgrade

# Main project (if needed)
cd ~/HYMetaLab
pip install -r requirements.txt --upgrade
```

---

## 🛑 Stop Services

Press `Ctrl+C` in each terminal window, or:

```bash
# Kill all uvicorn processes
pkill -f uvicorn

# Kill all streamlit processes
pkill -f streamlit
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Ensure you're in project root
cd ~/HYMetaLab
python3 -c "from tools.guardian_client import validate; print('✅ OK')"
```

### Port Already in Use
```bash
# Find process using port 8000
lsof -ti:8000 | xargs kill -9

# Find process using port 8501
lsof -ti:8501 | xargs kill -9
```

### API Connection Failed
This is expected! Guardian client automatically falls back to local stub when API is offline.

---

## 📚 Documentation

- **Operational Status**: `OPERATIONAL_STATUS.md`
- **Integration Summary**: `INTEGRATION_SUMMARY.md`
- **Reproducibility**: `Repro_Test.md`
- **Run Reports**: `Run_Report.md`
- **Task Log**: `Task_Log.md`

---

## 🎓 Example Workflow

```bash
# 1. Start services (use launcher or manual)
# 2. Verify API is running
curl http://localhost:8000/health

# 3. Run your experiment
python3 lab_techs_runner.py

# 4. Check results
cat Task_Log.md
tail -20 Run_Report.md

# 5. View in dashboard
open http://localhost:8501

# 6. Commit results (if good)
git add .
git commit -m "results: experiment XYZ completed"
```

---

## 🏆 Success Criteria

✅ **API Health**: `curl http://localhost:8000/health` returns `{"status":"ok"}`  
✅ **Tests Pass**: `pytest -q` shows 4 passed  
✅ **Runner Executes**: Sees `✅ Guardian PASS` output  
✅ **Dashboard Loads**: http://localhost:8501 accessible  
✅ **Logs Updated**: `Task_Log.md` has new entries  

---

## 🆘 Support

- **GitHub Issues**: https://github.com/jordanheckler-HMM/HYMetaLab/issues
- **CI/CD Status**: https://github.com/jordanheckler-HMM/HYMetaLab/actions
- **Documentation**: See README.md and docs in this repo

---

*"Integrity → Resilience → Meaning"*  
*HYMetaLab Lab Techs v2.0*
