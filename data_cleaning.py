import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ——————————————————————————————
# 🧠 Helper Functions
# ——————————————————————————————

def drop_high_missing(df, threshold):
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()
    df_cleaned = df.drop(columns=cols_to_drop)
    return df_cleaned, cols_to_drop

def impute_missing(df, num_strategy, cat_strategy, constant):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].isnull().sum() == 0:
            continue
        if df_copy[col].dtype in ['int64', 'float64']:
            if num_strategy == "mean":
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif num_strategy == "median":
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            else:
                df_copy[col].fillna(constant, inplace=True)
        else:
            if cat_strategy == "mode":
                mode_val = df_copy[col].mode()
                df_copy[col].fillna(mode_val[0] if not mode_val.empty else constant, inplace=True)
            else:
                df_copy[col].fillna(constant, inplace=True)
    return df_copy

def remove_duplicates(df):
    before = len(df)
    df_clean = df.drop_duplicates()
    removed = before - len(df_clean)
    return df_clean, removed

def handle_outliers(df, k=1.5):
    numeric_df = df.select_dtypes(include='number')
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((numeric_df < (Q1 - k * IQR)) | (numeric_df > (Q3 + k * IQR))).any(axis=1)
    df_filtered = df[condition]
    outliers_removed = len(df) - len(df_filtered)
    return df_filtered, outliers_removed

# ——————————————————————————————
# 🚿 Cleaning Page Main Function
# ——————————————————————————————

def show():
    st.sidebar.header("⚙️ Cleaning Options")
    missing_thresh = st.sidebar.slider("Drop columns if missing ≥ (%)", 0, 100, 50) / 100
    num_strategy = st.sidebar.selectbox("Numeric Imputation", ["median", "mean", "constant"])
    cat_strategy = st.sidebar.selectbox("Categorical Imputation", ["mode", "constant"])
    fill_constant = st.sidebar.text_input("Constant Value", "Unknown")
    iqr_multiplier = st.sidebar.number_input("Outlier IQR Multiplier", 1.0, 3.0, step=0.1)

    st.title("🛠 Data Cleaning")

    # Check if data is uploaded
    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No dataset found. Please upload a CSV file from the Home page.")
        return

    if 'history' not in st.session_state:
        st.session_state.history = [st.session_state['dataframe'].copy()]

    df = st.session_state['dataframe']

    st.subheader("📂 Current Data Preview")
    st.dataframe(df.head())

    st.subheader("⚡Run all steps at once")

    col1, col2= st.columns(2)

    with col1:
        if st.button("🚀 Run All Steps"):
            df_cleaned, dropped_cols = drop_high_missing(df, missing_thresh)
            df_cleaned = impute_missing(df_cleaned, num_strategy, cat_strategy, fill_constant)
            df_cleaned, dupes = remove_duplicates(df_cleaned)
            df_cleaned, outliers = handle_outliers(df_cleaned, iqr_multiplier)
            st.session_state['dataframe'] = df_cleaned
            st.session_state.history.append(df_cleaned.copy())
            dropped_cols_str = ', '.join(dropped_cols) if dropped_cols else 'None'
            st.success(
                f"Dropped {len(dropped_cols)} columns, columns: {dropped_cols_str}\n\n"
                f"Removed {dupes} duplicates, "
                f"Removed {outliers} outliers."
            )
        
    with col2:
        if st.button("↩️ Undo Last"):
            if len(st.session_state.history) > 1:
                st.session_state.history.pop()
                st.session_state['dataframe'] = st.session_state.history[-1]
                st.success("Reverted to previous version.")
                st.rerun()

    st.subheader("𓊍 Or Run Each Step Individually")
    st.write("Click the buttons below to handle specific data cleaning tasks.")    

    if st.button("🧹 Handle Missing Values"):
        df_cleaned, dropped = drop_high_missing(df, missing_thresh)
        df_cleaned = impute_missing(df_cleaned, num_strategy, cat_strategy, fill_constant)
        st.session_state['dataframe'] = df_cleaned
        st.session_state.history.append(df_cleaned.copy())
        st.success(f"Handled missing values. Dropped columns: {dropped}")
    
    if st.button("🗑 Remove Duplicates"):
            df_cleaned, removed = remove_duplicates(df)
            st.session_state['dataframe'] = df_cleaned
            st.session_state.history.append(df_cleaned.copy())
            st.success(f"Removed {removed} duplicate rows.")

    if st.button("🔄 Fix Data Types"):
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include='object'):
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='raise')
            except:
                df_copy[col] = df_copy[col].astype('category')
        for col in df_copy.select_dtypes(include='float64'):
            if df_copy[col].dropna().apply(float.is_integer).all():
                df_copy[col] = df_copy[col].astype('int64')
        st.session_state['dataframe'] = df_copy
        st.session_state.history.append(df_copy.copy())
        st.success("✅ Converted column types.")

    if st.button("📉 Remove Outliers"):
        df_filtered, removed = handle_outliers(df, iqr_multiplier)
        st.session_state['dataframe'] = df_filtered
        st.session_state.history.append(df_filtered.copy())
        st.success(f"Removed {removed} rows with outliers.")

    if st.button("💾 Save Cleaned Data"):
        st.success("✅ Cleaned data saved! Proceed to the next page.")
        # You can export to CSV or keep it for downstream use

    # Final preview
    st.subheader("📂 Preview Cleaned Data")
    st.dataframe(st.session_state['dataframe'].head())

    st.subheader("🔍 Custom Query (Python code)")
    query = st.text_area("Type a pandas command using `df`:")

    if st.button("▶ Run Query"):
        try:
            df = st.session_state['dataframe']  # Reference the actual df
            exec(query, {"pd": pd, "np": np}, {"df": df})  # Mutates df in-place
            st.session_state.history.append(df.copy())     # Save to history
            st.success("✅ Query executed successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error: {e}")

