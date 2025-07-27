import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime
import google.generativeai as genai
import numpy as np
from utils import save_and_report

st.session_state.setdefault('report_sections', [])

# ——— Gemini Setup ———
GENAI_API_KEY = "AIzaSyB-4YQD_RHpJCR3KAe83mvD7TNmQJLPS60"
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def call_gemini_api(prompt: str) -> str:
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"[AI error]: {e}"

# ——— PDF Export Helper ———
def sanitize_text(text):
    return text.encode('latin-1', 'replace').decode('latin-1')
from fpdf import FPDF
from datetime import datetime
import os

def save_pdf(sections: list[dict], filename="eda_report.pdf") -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)

    # Add Unicode-capable font
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError("DejaVuSans.ttf not found. Please add it to the project folder.")

    pdf.add_font("DejaVu", "", font_path, uni=True)

    # Title page
    pdf.add_page()
    pdf.set_font("DejaVu", "", 20)
    pdf.cell(0, 10, "Exploratory Data Analysis Report", ln=True, align="C")
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8,
             f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}",
             ln=True, align="C")
    pdf.ln(10)

    # Sections
    for sec in sections:
        pdf.set_font("DejaVu", "", 14)
        pdf.multi_cell(0, 8, sec["title"])
        pdf.set_font("DejaVu", "", 12)
        pdf.multi_cell(0, 6, sec["text"])
        if sec.get("image"):
            try:
                pdf.image(sec["image"], w=160)
                pdf.ln(5)
            except Exception:
                pass
        pdf.ln(8)
        pdf.add_page()

    pdf.output(filename)
    return filename


# ——— Report Builder ———
def show():
    st.title("📑 AI Report")

    if "dataframe" not in st.session_state:
        st.warning("⚠️ Please upload and clean your dataset first.")
        return
    df = st.session_state["dataframe"]

    rows, cols = df.shape


    # ——— 1) Dataset Overview ———
    st.subheader("📊 Dataset Overview")
    
    # Build a schema list for the AI
    col_info = "\n".join(f"- {c}: {str(dt)}" for c, dt in df.dtypes.items())
    overview_prompt = f"""
    You are a data analyst. Here is the schema of a dataset with {rows} rows and {cols} columns:
    {col_info}

    Summarize the overall structure, variable types, and anything else that jumps out.
    """
    overview = call_gemini_api(overview_prompt.strip())
    st.write(overview)
    sections = [{"title": "Dataset Overview", "text": overview, "image": None}]

    # ——— 2) Visualizations Collected ———
    saved = st.session_state.get("report_sections", [])
    st.subheader("🖼️ Collected Analysis Sections")
    if not saved:
        st.info("You haven't added any visuals yet. Go back to the Visualizations page and click 📄 Add to Report under each chart you want.")
    else:
        for sec in saved:
            st.markdown(f"### {sec['title']}")
            st.write(sec["text"])
            if sec.get("image"):
                st.image(sec["image"], use_container_width=True)
            sections.append(sec)

    # ——— 3) Key Insights & Hypotheses ———
    st.subheader("🔍 Key Insights & Hypotheses")
    insight_prompt = (
        f"We have completed EDA on a dataset of {rows} rows × {cols} columns. "
        "Based on the above analyses, list 3–5 concise hypotheses or next‑step recommendations."
    )
    key_insights = call_gemini_api(insight_prompt)
    st.write(key_insights)
    sections.append({"title": "Key Insights & Hypotheses", "text": key_insights, "image": None})

    # 4) Download PDF
    st.subheader("📥 Download Full Report")
    if st.button("Generate & Download PDF"):
        pdf_path = save_pdf(sections)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name=pdf_path,
                mime="application/pdf"
            )
        st.success("✅ Your PDF report is ready!")

    
    st.success("🏁 Report generation complete!")