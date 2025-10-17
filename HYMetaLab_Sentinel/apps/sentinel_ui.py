import streamlit as st
import requests
import json
import time
import os
import psutil
from datetime import datetime

st.set_page_config(page_title="Guardian Sentinel", layout="wide")
st.title("🛡️ HYMetaLab — Guardian Sentinel (Phase I)")

with st.sidebar:
    st.subheader("System")
    st.write("CPU:", psutil.cpu_percent(), "%")
    mem = psutil.virtual_memory()
    st.write("RAM:", f"{mem.used//(10242)}MB / {mem.total//(10242)}MB")
    st.divider()
    api_url = st.text_input("API URL", "http://127.0.0.1:8000")
    st.caption("Run make api in another terminal if offline.")

tabs = st.tabs(["Validate", "Metrics", "Drive Sync", "Logs"])

with tabs[0]:
    st.subheader("Run validation (pre/post)")
    col1, col2 = st.columns(2)
    with col1:
        actor = st.text_input("Actor", "CEO_GPT")
        target = st.text_input("Target", "model:gpt-local")
        task_type = st.selectbox(
            "Task Type", ["answer", "plan", "code", "summarize"], index=0
        )
        policy_tags = st.text_input(
            "Policy Tags (comma)", "safety.general,privacy.redact"
        )
        required_checks = st.text_input(
            "Required Checks (comma)", "guardian.coherence>=0.85,guardian.noise<=0.2"
        )
    with col2:
        prompt = st.text_area("Prompt / Input", "Explain the lab's mission in 2 lines.")
        output = st.text_area("Output (for post phase; optional)", "")

    if st.button("Validate (pre)"):
        env = {
            "trace_id": str(int(time.time() * 1000)),
            "ts": datetime.utcnow().isoformat() + "Z",
            "actor": actor,
            "target": target,
            "phase": "pre",
            "task_type": task_type,
            "policy_tags": [x.strip() for x in policy_tags.split(",") if x.strip()],
            "required_checks": [
                x.strip() for x in required_checks.split(",") if x.strip()
            ],
            "payload": {"prompt": prompt},
            "meta": {"context_hash": "sha256:stub"},
        }
        r = requests.post(f"{api_url}/validate", json=env, timeout=10)
        st.json(r.json())

    if st.button("Validate (post)"):
        env = {
            "trace_id": str(int(time.time() * 1000)),
            "ts": datetime.utcnow().isoformat() + "Z",
            "actor": actor,
            "target": target,
            "phase": "post",
            "task_type": task_type,
            "policy_tags": [x.strip() for x in policy_tags.split(",") if x.strip()],
            "required_checks": [
                x.strip() for x in required_checks.split(",") if x.strip()
            ],
            "payload": {"prompt": prompt, "output": output or "Sample output..."},
            "meta": {"context_hash": "sha256:stub"},
        }
        r = requests.post(f"{api_url}/validate", json=env, timeout=10)
        st.json(r.json())
with tabs[1]:
    st.subheader("Metrics (demo)")
    st.write("Integrity-per-Dollar (placeholder):")
    st.metric("Integrity-per-Dollar", "0.97", "Phase I demo")
    st.write("Latency snapshots will appear here in Phase II.")
    st.caption("Real metrics wire in once Guardian/TruthLens are non-stub.")
with tabs[2]:
    st.subheader("Drive Sync (demo)")
    st.write(
        "Reports are written to dev/logs/guardian_reports/ and a Drive URI string is recorded in the response."
    )
    st.caption("In Phase I we simulate Drive; Phase II can use real Drive APIs.")

with tabs[3]:
    st.subheader("Recent Logs")
    logs_dir = "dev/logs/guardian_reports"
    if os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir))[-10:]:
            if name.endswith(".json"):
                st.write(f"{name}")
                st.json(json.load(open(os.path.join(logs_dir, name))))
    else:
        st.info("No logs yet — run a validation first.")
