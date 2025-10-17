from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Reality Loop Lite v1.0", page_icon="🔄", layout="wide")

# Custom dark theme with Matrix-style green accents
st.markdown(
    """
<style>
    .stApp { 
        background-color: #000000; 
        color: #00FF66; 
        font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; 
    }
    h1, h2, h3 { 
        color: #00FF66 !important; 
        font-weight: 600;
    }
    .stMetric { 
        background-color: #0a0a0a; 
        padding: 10px; 
        border-radius: 8px; 
        border: 1px solid #00FF66;
    }
    .stMetric label { 
        color: #00FF66 !important; 
    }
    .stMetric .metric-value { 
        color: #00FF66 !important; 
    }
    .stButton>button { 
        background-color: #003300; 
        color: #00FF66; 
        border: 2px solid #00FF66;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover { 
        background-color: #00FF66; 
        color: #000000; 
        box-shadow: 0 0 20px #00FF66;
    }
    .stMarkdown { 
        color: #00FF66; 
    }
    .stInfo { 
        background-color: #001a0a; 
        border-left: 4px solid #00FF66; 
    }
    .stSuccess { 
        background-color: #002200; 
        border-left: 4px solid #00FF66; 
    }
    .stWarning { 
        background-color: #1a1a00; 
        border-left: 4px solid #FFCC00; 
        color: #FFCC00;
    }
    hr { 
        border-color: #00FF66 !important; 
        opacity: 0.3;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Import core computation module
from loop_lite_core import compute_once, trajectory

# Initialize session state for scenario presets
if "trust" not in st.session_state:
    st.session_state.trust = 0.55
if "hope" not in st.session_state:
    st.session_state.hope = 0.55
if "meaning" not in st.session_state:
    st.session_state.meaning = 0.55

st.title("🔄 Reality Loop Lite v1.0")
st.markdown("**Nonlinear Dynamics of Trust, Hope & Meaning**")

# Scenario preset buttons
st.markdown("**Quick Scenarios:**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🔴 Crisis"):
        st.session_state.trust = 0.15
        st.session_state.hope = 0.20
        st.session_state.meaning = 0.30
        st.rerun()
with c2:
    if st.button("🟡 Neutral"):
        st.session_state.trust = 0.50
        st.session_state.hope = 0.50
        st.session_state.meaning = 0.50
        st.rerun()
with c3:
    if st.button("🟢 Optimistic"):
        st.session_state.trust = 0.85
        st.session_state.hope = 0.90
        st.session_state.meaning = 0.85
        st.rerun()
with c4:
    if st.button("🤝 Team Cohesion"):
        st.session_state.trust = 0.70
        st.session_state.hope = 0.80
        st.session_state.meaning = 0.90
        st.rerun()

st.markdown("---")

# Main controls
colA, colB, colC, colD = st.columns([1, 1, 1, 1])
with colA:
    trust = st.slider("Trust", 0.0, 1.0, st.session_state.trust, key="trust_slider")
with colB:
    hope = st.slider("Hope", 0.0, 1.0, st.session_state.hope, key="hope_slider")
with colC:
    meaning = st.slider(
        "Meaning", 0.0, 1.0, st.session_state.meaning, key="meaning_slider"
    )
with colD:
    sensitivity = st.slider("Sensitivity", 1.0, 2.5, 1.6, 0.1)

# Update session state
st.session_state.trust = trust
st.session_state.hope = hope
st.session_state.meaning = meaning

st.write(
    f"**Trust {trust*100:.0f}% • Hope {hope*100:.0f}% • Meaning {meaning*100:.0f}% • Sensitivity {sensitivity:.2f}**"
)

st.markdown("---")

# Compute current state and trajectory
dcci, risk_reduction = compute_once(trust, hope, meaning, sensitivity=sensitivity)
xs, ys = trajectory(trust, hope, meaning, steps=28, alpha=0.28, sensitivity=sensitivity)

# Display current metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "ΔCCI",
        f"{dcci:.4f}",
        delta=f"{dcci*100:+.1f}%",
        help="Higher is better (0-1 range)",
    )
with col2:
    st.metric(
        "Risk Reduction",
        f"{risk_reduction:.4f}",
        delta=f"{risk_reduction*100:+.1f}%",
        help="Higher = safer (risk reduced)",
    )
with col3:
    status = (
        "🌈 Thriving" if dcci > 0.06 else ("☀️ Stable" if dcci > 0.03 else "🌧 Fragile")
    )
    st.metric("Status", status)
with col4:
    health = (
        "Excellent"
        if risk_reduction > 0.05
        else ("Good" if risk_reduction > 0.02 else "Caution")
    )
    st.metric("Health", health)

# Status feedback
if dcci > 0.06:
    st.success(
        "🌈 **System Thriving** — High coherence detected! Trust, Hope, and Meaning are well-aligned."
    )
elif dcci > 0.03:
    st.info("☀️ **Stable Growth** — System is stable and showing positive development.")
else:
    st.warning(
        "🌧 **Fragile State** — Low coherence. Consider interventions to boost Trust, Hope, or Meaning."
    )

st.markdown("---")

# Main trajectory plot
st.subheader("📈 Reality Loop Trajectory")
st.markdown("System evolution in ΔCCI-Risk Reduction phase space")

fig = go.Figure()

# Trajectory line
fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        name="Trajectory",
        line=dict(color="#00FF66", width=3),
        marker=dict(size=6, color="#00FF66", line=dict(color="#003300", width=1)),
    )
)

# Current position
fig.add_trace(
    go.Scatter(
        x=[dcci],
        y=[risk_reduction],
        mode="markers",
        name="Current",
        marker=dict(
            size=20, color="#FFCC00", symbol="star", line=dict(color="#FFFFFF", width=2)
        ),
    )
)

# Reference lines
fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"))
fig.add_vline(
    x=0.03,
    line=dict(color="rgba(0,255,102,0.5)", width=1, dash="dot"),
    annotation_text="Target ΔCCI",
)

# Target zone (shaded region) - flipped for new convention
fig.add_shape(
    type="rect",
    x0=0.03,
    y0=0.01,
    x1=0.12,
    y1=0.12,
    fillcolor="rgba(0,255,102,0.1)",
    line=dict(width=0),
    layer="below",
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="ΔCCI (Collective Coherence Index) →",
    yaxis_title="ΔHazard (Risk Reduction ↑)",
    xaxis=dict(range=[-0.01, 0.12], showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
    yaxis=dict(range=[-0.03, 0.12], showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
    hovermode="closest",
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"**Current:** ΔCCI={dcci:.4f} | Risk Reduction={risk_reduction:.4f}  •  **Target Band:** ΔCCI≥0.030, Risk Reduction≥0.010"
)

# Enhanced dual-metric feedback
if dcci >= 0.06 and risk_reduction >= 0.02:
    st.markdown("### 🌈 Thriving — high coherence, strong risk reduction")
    st.success(
        "System is in optimal state with strong coherence and significant risk reduction."
    )
elif dcci >= 0.03 and risk_reduction >= 0.01:
    st.markdown("### ☀️ Stable Growth — trending healthier")
    st.info(
        "System shows positive development with good coherence and measurable risk reduction."
    )
else:
    st.markdown("### 🌧 Fragile — increase trust/hope or reduce coupling")
    st.warning(
        "System needs attention. Try boosting Trust or Hope, or adjust sensitivity to reduce coupling penalty."
    )

# Diagnostics expander
with st.expander("🔍 Diagnostics & Tips"):
    st.write(
        "**How it works:** Inputs drive ΔCCI nonlinearly; low trust & hope create a coupling penalty."
    )

    st.markdown("**ΔCCI Progress** (Target: 0.10 max scale)")
    st.progress(min(max(dcci / 0.10, 0), 1.0), text=f"ΔCCI: {dcci:.3f} / 0.10")

    st.markdown("**Risk Reduction Progress** (Target: 0.12 max scale)")
    st.progress(
        min(max(risk_reduction / 0.12, 0), 1.0),
        text=f"Risk Reduction: {risk_reduction:.3f} / 0.12",
    )

    st.info(
        "💡 **Pro Tip:** Push Trust & Hope above 70%, then raise Sensitivity until ☀️/🌈 appears."
    )

    st.markdown("**Understanding the Metrics:**")
    st.markdown(
        """
    - **ΔCCI** measures collective coherence improvement over baseline (0.5)
    - **Risk Reduction** tracks how much collapse risk is reduced (positive = safer)
    - **Coupling penalty** activates when both Trust & Hope are low
    - **Sensitivity** controls how responsive the system is to changes
    """
    )

    st.markdown("---")
    st.markdown("**Quick Export:**")

    # Quick CSV download
    quick_df = pd.DataFrame(
        [
            {
                "trust": trust,
                "hope": hope,
                "meaning": meaning,
                "sensitivity": sensitivity,
                "dcci": dcci,
                "risk_reduction": risk_reduction,
            }
        ]
    )

    st.download_button(
        "📥 Download Current Values CSV",
        quick_df.to_csv(index=False).encode(),
        "reality_loop_snapshot.csv",
        "text/csv",
        help="Quick export of current slider values and computed metrics",
    )

st.markdown("---")

# Export functionality
st.subheader("📥 Export Current State")

snapshot_df = pd.DataFrame(
    [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trust": trust,
            "hope": hope,
            "meaning": meaning,
            "sensitivity": sensitivity,
            "ΔCCI": dcci,
            "risk_reduction": risk_reduction,
            "status": (
                "Thriving" if dcci > 0.06 else ("Stable" if dcci > 0.03 else "Fragile")
            ),
        }
    ]
)

# Trajectory data
trajectory_df = pd.DataFrame({"step": range(len(xs)), "ΔCCI": xs, "risk_reduction": ys})

col1, col2 = st.columns(2)
with col1:
    snapshot_csv = snapshot_df.to_csv(index=False).encode()
    st.download_button(
        label="📊 Download Current State CSV",
        data=snapshot_csv,
        file_name=f"reality_loop_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Export current Trust, Hope, Meaning, ΔCCI, and Risk Reduction values",
    )
with col2:
    trajectory_csv = trajectory_df.to_csv(index=False).encode()
    st.download_button(
        label="📈 Download Trajectory CSV",
        data=trajectory_csv,
        file_name=f"reality_loop_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Export full trajectory data",
    )

st.markdown("---")

# Temporary QA block (remove before shipping)
low = compute_once(0.1, 0.1, 0.1, sensitivity=1.8)
high = compute_once(0.9, 0.9, 0.9, sensitivity=1.8)
# FIXED: Core module now returns positive risk reduction (higher = safer)
visible = (high[0] - low[0] >= 0.04) and (
    (high[1] - low[1]) >= 0.01
)  # ΔCCI lift ≥.04, risk reduction lift ≥.01
st.caption(
    f"QA Visible Change: {'✅ PASS' if visible else '❌ FAIL'}  | low={low}  high={high}"
)

st.markdown(
    "**Reality Loop Lite v1.0** | HYMetaLab Simulation Suite | Powered by Nonlinear Dynamics"
)
