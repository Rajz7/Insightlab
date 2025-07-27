import streamlit as st
import pandas as pd
from PIL import Image
import os
import shutil

def clear_plot_folder():
    plot_folder = "plots"
    if os.path.exists(plot_folder):
        for filename in os.listdir(plot_folder):
            file_path = os.path.join(plot_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def show():
    st.title("🏠 Welcome to Insight Lab")
    
    st.write("""
    **Insight Lab** is an interactive tool that helps you analyze datasets step by step. Follow the structured process:
    
    1️⃣ **Before Cleaning** - Explore the raw dataset.
    
    2️⃣ **Data Cleaning** - Fix missing values, outliers, and data types.
    
    3️⃣ **After Cleaning** - Review the cleaned dataset.
    
    4️⃣ **Visualization** - Generate insightful plots.
    
    5️⃣ **Hypothesis & Report** - Get AI-generated insights & download a PDF report.
    """)
    
    st.subheader("📌 How It Works")

    flowchart = Image.open("assets/InsightLab3.jpg")  # Ensure the correct path
    st.image(flowchart, caption="EDA Process Flow")

    st.write("### 📂 Upload a CSV File to Start")
    uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
    
    if uploaded_file:
        try:
            clear_plot_folder()
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("❌ The uploaded file is empty. Please upload a valid CSV file.")
                return
            
            # Store DataFrame in session state
            filename = uploaded_file.name
            st.session_state['filename'] = filename
            st.session_state['uploaded_file'] = uploaded_file  # Store file reference
            st.session_state['original_df'] = df.copy()  # Store a copy of original dataset
            st.session_state['dataframe'] = df.copy() 
            
            st.success("✅ File uploaded successfully! Now navigate to 'Before Cleaning' to explore it.")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
