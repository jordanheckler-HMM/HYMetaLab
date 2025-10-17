# HYMetaLab Desktop Launchers

Three macOS one-click launchers are available on your Desktop for convenient HYMetaLab operations.

---

## 🚀 HYMetaLab_AutoStart.command

**Purpose**: Autonomous experimentation autopilot

**What it does**:
1. Opens Terminal
2. Runs `auto_lab.py` for 5 cycles (default)
3. Shows knowledge base summary
4. Reports completion

**Usage**:
- **Double-click** to start 5 autonomous experiment cycles
- Watch Terminal for real-time progress
- Close window when "Done" appears

**Customization**:
Edit the file to change:
```bash
CYCLES=5        # Change to 10, 20, 50, etc.
BASE_DIR="..."  # Update if repo moves
```

**Features**:
- ✅ UCB hypothesis selection
- ✅ Guardian pre-validation
- ✅ Persistent knowledge accumulation
- ✅ Safe to interrupt (Ctrl+C)
- ✅ Shows results on completion

**Perfect for**: Quick autonomous experimentation sessions

---

## 🏁 HYMetaLab_Start.command

**Purpose**: Full HYMetaLab infrastructure startup

**What it does**:
1. Starts Sentinel API (FastAPI on port 8000)
2. Starts Sentinel UI (Streamlit on port 8501)
3. Opens dashboard in browser
4. Runs lab_techs_runner.py once

**Usage**:
- **Double-click** to launch full stack
- Three Terminal windows open automatically
- Dashboard opens at `http://localhost:8501`

**Features**:
- ✅ Argv-safe AppleScript
- ✅ Port checking before opening browser
- ✅ macOS notifications
- ✅ Auto-installs dependencies

**Perfect for**: Development, interactive experimentation, dashboard monitoring

---

## 🛑 HYMetaLab_Stop.command

**Purpose**: Graceful shutdown of Sentinel services

**What it does**:
1. Stops Sentinel API (uvicorn)
2. Stops Sentinel UI (streamlit)
3. Confirms shutdown

**Usage**:
- **Double-click** to stop services
- Safe to run anytime
- Cleans up background processes

**Perfect for**: Ending development sessions, freeing ports

---

## Launcher Comparison

| Launcher | Purpose | Opens Terminal? | Runs What? | Best For |
|----------|---------|-----------------|------------|----------|
| **AutoStart** | Autonomous experiments | Yes (1 window) | auto_lab.py | Hands-free experimentation |
| **Start** | Full infrastructure | Yes (3 windows) | API + UI + Runner | Development & monitoring |
| **Stop** | Shutdown services | No | Cleanup | Ending sessions |

---

## Workflow Examples

### Quick Autonomous Experimentation
```
1. Double-click HYMetaLab_AutoStart.command
2. Watch experiments run
3. Close Terminal when done
4. Check autolab/knowledge.json for results
```

### Development Session
```
1. Double-click HYMetaLab_Start.command
2. Use dashboard at http://localhost:8501
3. Run experiments via Terminal or UI
4. Double-click HYMetaLab_Stop.command when done
```

### Long Autonomous Session (Terminal)
```bash
# Start 50-cycle autonomous run in background
cd ~/conciousness_proxy_sim\ copy\ 6
nohup python3 auto_lab.py 50 > autolab_log.txt 2>&1 &

# Monitor progress
tail -f autolab_log.txt

# Or check knowledge base
watch -n 10 'cat autolab/knowledge.json | python3 -m json.tool | tail -30'
```

---

## File Locations

All launchers are located at:
```
~/Desktop/HYMetaLab_AutoStart.command   (846 bytes)
~/Desktop/HYMetaLab_Start.command       (2.0 KB)
~/Desktop/HYMetaLab_Stop.command        (154 bytes)
```

**Quarantine attributes removed** — No macOS security warnings!

---

## Troubleshooting

### "auto_lab.py not found"
- Edit `BASE_DIR` in launcher to correct path
- Current: `~/conciousness_proxy_sim copy 6`

### macOS Security Warning
```bash
# Run this if needed:
xattr -d com.apple.quarantine ~/Desktop/HYMetaLab_*.command
```

### Launchers Not Executable
```bash
chmod +x ~/Desktop/HYMetaLab_*.command
```

### Port Already in Use (Start launcher)
```bash
# Stop existing services first
./HYMetaLab_Stop.command

# Or manually:
pkill -f uvicorn
pkill -f streamlit
```

---

## Integration with AutoLab

**AutoStart launcher** integrates seamlessly with:
- `auto_lab.py` — UCB hypothesis selection
- `lab_techs_runner.py` — Guardian-gated execution
- `tools/guardian_client.py` — Safety validation
- `autolab/knowledge.json` — Learning state

**Architecture**:
```
HYMetaLab_AutoStart.command
    ↓
  auto_lab.py (UCB loop)
    ↓
  ├─→ Guardian pre-check
  ├─→ lab_techs_runner.py --hypothesis "..."
  └─→ Knowledge base update
```

---

## Advanced Usage

### Custom Cycle Count
Edit launcher and change `CYCLES=5` to any value:
```bash
CYCLES=20   # 20 autonomous experiments
CYCLES=100  # Long exploration session
```

### Multiple Launchers
Create variants for different scenarios:
```bash
# Copy and customize
cp ~/Desktop/HYMetaLab_AutoStart.command ~/Desktop/HYMetaLab_AutoStart_Long.command

# Edit new file: CYCLES=50
```

### Notification on Completion
Add to launcher before last line:
```bash
osascript -e 'display notification "AutoLab complete!" with title "HYMetaLab"'
```

---

## Charter Compliance

All launchers maintain:
- ✅ **Charter v2.0**: Guardian validation enforced
- ✅ **SOP v1.1**: Standard procedures followed
- ✅ **Integrity**: All experiments validated
- ✅ **Reproducibility**: Full audit trails
- ✅ **Safety**: Graceful shutdown supported

---

*"Integrity → Resilience → Meaning"*  
*HYMetaLab Desktop Launchers v1.0*

