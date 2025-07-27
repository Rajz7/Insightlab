import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def show():
    st.title("📊 Before Data Cleaning")

    # Check if dataset exists in session state
    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No file uploaded! Please go to the Home page and upload a CSV file first.")
        return

    df = st.session_state['dataframe']

    st.info(f"🧮 Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Overview")
    st.write(df.describe())

    def dataframe_info(df):
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.notnull().sum().values,
            'Dtype': df.dtypes.values
        })
        return info_df

    st.subheader("📊 Dataset Info")
    st.dataframe(dataframe_info(df))


    st.subheader("❗ Missing Values")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]  # Only plot those with missing data

    if missing_values.empty:
        st.success("✅ No missing values detected!")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=missing_values.index, y=missing_values.values, ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel("Columns")
        plt.ylabel("Missing Values Count")
        st.pyplot(fig)

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.empty:
        st.warning("⚠️ No numeric columns available for boxplot.")
    else:
        # Create boxplot for numeric columns
        st.subheader("📦 Boxplot of Numeric Features (Outlier Detection)")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=numeric_df, ax=ax)
        plt.title("Boxplot of Numeric Features")
        plt.xticks(rotation=45)
        plt.xlabel("Columns")
        plt.ylabel("Value Range")
        st.pyplot(fig)
    
    st.write("➡️ **Proceed to the 'Data Cleaning' page to clean the dataset!**")