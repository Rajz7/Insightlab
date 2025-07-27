# utils.py
import os, streamlit as st

def save_and_report(fig, title, insight_text, filename=None):
    os.makedirs("plots", exist_ok=True)
    if not filename:
        sanitized = title.replace(" ", "_").replace(":", "").replace(",", "")
        filename = f"plots/{sanitized}.png"
    fig.savefig(filename)
    sec = {"title": title, "image": filename, "text": insight_text}
    st.session_state.setdefault('report_sections', []).append(sec)
    st.success("✅ Added to report!")

