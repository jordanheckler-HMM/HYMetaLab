# AutoLab Desktop (Tauri Wrapper)

This wraps `apps/autolab_studio.py` in a desktop shell so you can launch and use AutoLab Studio as a native app.

## What it does
- Starts `streamlit run apps/autolab_studio.py` automatically.
- Auto-detects a Python interpreter that has `streamlit` installed.
- Picks an available localhost port (prefers `8787`, then scans upward).
- Waits for the local server and opens the desktop window to that runtime URL.
- Opens a Tauri desktop window pointed at the local Streamlit app.
- Stops the Streamlit child process when the app exits.
- Shows startup errors in-app (instead of a blank screen) and writes backend logs to `<workspace>/logs/`.
- Adds native app menu actions:
  - Open Workspace Folder
  - Open Runs Folder
  - Reload UI
  - Restart Backend

## Prerequisites
- Node.js + npm
- Rust + Cargo
- Python 3 with Streamlit installed in your environment

## Run (dev)
```bash
cd apps/autolab_desktop
npm install
npm run tauri:dev
```

## Build desktop app
```bash
cd apps/autolab_desktop
npm install
npm run tauri:build
```

## Smoke checks
```bash
cd apps/autolab_desktop
npm run smoke:check
```

Full bundle smoke build:
```bash
cd apps/autolab_desktop
npm run smoke:build
```

## Optional environment variables
- `AUTOLAB_PYTHON`: Python executable to use (default: `python3`)
- `AUTOLAB_STUDIO_PATH`: absolute path to `autolab_studio.py`
- `AUTOLAB_PORT`: preferred starting port for Streamlit scan (default: `8787`)
- `AUTOLAB_WORKSPACE`: workspace folder used by "Open Workspace/Runs" menu actions
