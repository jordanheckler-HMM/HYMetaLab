# HYMetaLab — Guardian Sentinel (Phase I)

**What this is:** a local watchdog that validates agents/models, logs runs, computes checksums, and syncs reports.  
**Phase I:** stubs + dashboards + CI hooks (no paid services, ≤25 GB).

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make api          # starts FastAPI
make ui           # starts Streamlit UI
make validate     # runs Guardian stub demo
```

Structure
apps/           # FastAPI + Streamlit
tools/          # sync, checksums, guardian/truthlens stubs
dev/            # CI, logs, metrics
state/          # sqlite (runtime)
Phase II (future)
Swap stubs with real Guardian/TruthLens evaluators.
Wire into Aletheia bus (pre/post envelopes + verdict loop).
