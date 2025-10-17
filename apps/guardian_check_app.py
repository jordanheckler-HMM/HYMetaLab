"""
Guardian Check Application

Real-time ethical alignment validation for research documents.
Provides interactive interface for Guardian v4 validation.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Guardian Check", page_icon="🛡️", layout="wide")

from qc.guardian_v4.api import GuardianError, evaluate_text  # noqa: E402
from tools.epistemic import BOUNDARY, hedge  # noqa: E402

# Add epistemic boundary to module documentation
__doc__ = (__doc__ or "") + "\n\n" + BOUNDARY

st.title("🛡️ Guardian v4 Ethics Checker")
st.markdown("**Real-time ethical alignment validation for research documents**")

st.sidebar.header("About Guardian")
guardian_info = hedge(
    "Guardian v4 validates documents against: Objectivity (hedge terms, claims, citations), "
    "Sentiment (neutrality and balance), Transparency (metadata, reproducibility), "
    "and Safety (risk assessment). Target: ≥90/100 for publication."
)
st.sidebar.info(guardian_info)

# Self-test button in sidebar
if st.sidebar.button("🧪 Run Self-Test"):
    with st.sidebar:
        with st.spinner("Testing Guardian v4..."):
            try:
                res = evaluate_text(
                    "This preliminary report suggests improvements with 95% CI [0.03, 0.06]."
                )
                st.success(f"✅ Self-Test OK — Score {res['score']:.1f}/100")
                st.caption(f"Objectivity: {res.get('objectivity', 0):.2f}")
            except GuardianError as e:
                st.error(f"❌ Self-Test Failed: {e}")

# Input options
input_mode = st.radio("Input Mode", ["Paste Text", "Upload File"])

text = ""
if input_mode == "Paste Text":
    text = st.text_area("Paste your document text:", height=300)
else:
    uploaded_file = st.file_uploader("Upload a .md or .txt file", type=["md", "txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

if st.button("🔍 Run Guardian Check", type="primary"):
    if not text:
        st.warning("Please provide text to validate.")
    else:
        with st.spinner("Running Guardian v4 validation..."):
            try:
                result = evaluate_text(text)

                score = result.get("score", 0)

                # Display score with color coding
                if score >= 90:
                    msg = hedge(
                        f"Guardian score of {score:.1f}/100 indicates strong alignment with integrity standards."
                    )
                    st.success(
                        f"✅ **Guardian Score: {score:.1f}/100** (Excellent)\n\n{msg}"
                    )
                elif score >= 80:
                    msg = hedge(
                        f"Guardian score of {score:.1f}/100 indicates good alignment with integrity standards."
                    )
                    st.info(f"ℹ️ **Guardian Score: {score:.1f}/100** (Good)\n\n{msg}")
                elif score >= 70:
                    msg = hedge(
                        f"Guardian score of {score:.1f}/100 indicates acceptable alignment, though review is recommended."
                    )
                    st.warning(
                        f"⚠️ **Guardian Score: {score:.1f}/100** (Pass, review recommended)\n\n{msg}"
                    )
                else:
                    msg = hedge(
                        f"Guardian score of {score:.1f}/100 indicates improvements may enhance integrity alignment."
                    )
                    st.error(
                        f"❌ **Guardian Score: {score:.1f}/100** (Needs improvement)\n\n{msg}"
                    )

                # Detailed metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    obj = result.get("objectivity", 0)
                    st.metric("Objectivity", f"{obj:.2f}")
                with col2:
                    trans = result.get("transparency", 0)
                    st.metric("Transparency", f"{trans:.2f}")
                with col3:
                    safety = result.get("language_safety", 0)
                    st.metric("Language Safety", f"{safety:.2f}")
                with col4:
                    sent = result.get("sentiment", 0)
                    st.metric("Sentiment", f"{sent:.2f}")

                # Risk level
                risk = result.get("risk_level", "unknown")
                risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                    risk.lower(), "⚪"
                )
                st.markdown(f"**Risk Level:** {risk_color} {risk.upper()}")

                # Full report
                with st.expander("📊 Full Guardian Report (JSON)"):
                    st.json(result["raw"])

                # Summary
                with st.expander("📝 Guardian Summary (Markdown)"):
                    st.markdown(result["summary_text"] or "_No summary generated_")

            except GuardianError as e:
                st.error(f"Guardian Error: {e}")
                st.info("Make sure Guardian v4 is installed in qc/guardian_v4/")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                import traceback

                st.code(traceback.format_exc())

st.markdown("---")
st.markdown("**Guardian v4** | HYMetaLab Research Integrity Suite")

# Add epistemic boundary footer
with st.expander("ℹ️ Epistemic Boundary"):
    st.info(BOUNDARY)
    st.caption(
        "Guardian scores are indicative assessments based on current heuristics and "
        "should be interpreted as one signal among many in evaluating document quality."
    )
