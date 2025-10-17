import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Reality Loop Lite — Minimal", layout="wide")

# NO st.button gating and NO @st.cache on compute_once/trajectory
# Streamlit re-runs the script on every interaction automatically.

# robust import: works when run from repo root or apps/
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apps.loop_lite_core import compute_once, trajectory  # safe either way

try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

# ---------------- Controls
cA, cB, cC, cD = st.columns([1, 1, 1, 1])
with cA:
    trust = st.slider("Trust", 0.0, 1.0, 0.55, key="trust")
with cB:
    hope = st.slider("Hope", 0.0, 1.0, 0.55, key="hope")
with cC:
    meaning = st.slider("Meaning", 0.0, 1.0, 0.55, key="meaning")
with cD:
    sensitivity = st.slider("Sensitivity", 1.0, 2.5, 1.6, key="sens")

# ---------------- Compute (no buttons; recompute every slider move)
dcci, dhaz = compute_once(trust, hope, meaning, sensitivity=sensitivity)
dhaz_display = -dhaz  # higher = safer for users

if dcci >= 0.06 and dhaz <= -0.02:
    status_emoji, status_label = "🌈", "Thriving"
elif dcci >= 0.03 and dhaz <= -0.01:
    status_emoji, status_label = "☀️", "Stable Growth"
else:
    status_emoji, status_label = "🌧", "Fragile"

st.markdown(f"## {status_emoji} {status_label}")

colA, colB = st.columns(2)
cci_bar = min(max(dcci / 0.10, 0.0), 1.0)  # vs 0.10 cap
risk_bar = min(max(dhaz_display / 0.12, 0.0), 1.0)  # vs 0.12 cap
with colA:
    st.caption(f"ΔCCI Progress (≥ 0.030). Now: {dcci:.3f}")
    st.progress(cci_bar)
with colB:
    st.caption(f"Risk Reduction (higher = safer). Now: {dhaz_display:.3f}")
    st.progress(risk_bar)

st.write(
    f"Trust {trust*100:.0f}% • Hope {hope*100:.0f}% • Meaning {meaning*100:.0f}% • Sens {sensitivity:.2f}"
)

# Trajectory view (ΔCCI vs Risk Reduction)
xs, ys = trajectory(trust, hope, meaning, steps=28, alpha=0.30, sensitivity=sensitivity)
ys_display = [-y for y in ys]

fig2d = go.Figure()
fig2d.add_trace(go.Scatter(x=xs, y=ys_display, mode="lines+markers", name="Trajectory"))
fig2d.add_hline(y=0.01, line=dict(color="gray", dash="dot", width=1))
fig2d.add_vline(x=0.03, line=dict(color="gray", dash="dot", width=1))
fig2d.update_layout(
    template="plotly_dark",
    height=420,
    xaxis_title="ΔCCI (higher is better)",
    yaxis_title="Risk Reduction (higher is safer)",
    xaxis=dict(range=[-0.005, 0.12]),
    yaxis=dict(range=[-0.005, 0.12]),
)
st.plotly_chart(fig2d, use_container_width=True)
