import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show():
    st.title("✅ After Data Cleaning")

    # Ensure original dataset is available
    if 'original_df' in st.session_state:
        original_df = st.session_state['original_df']  # Use the stored dataset
    else:
        st.warning("⚠️ No original dataset found. Please re-upload the file.")
        return

    # Ensure cleaned dataset is available
    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No cleaned dataset found! Please complete the Data Cleaning process first.")
        return

    # Load the cleaned dataset
    cleaned_df = st.session_state['dataframe']

    # Display dataset preview
    st.subheader("📂 Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head())

    # Row & Column Count Comparison
    st.subheader("🔍 Dataset Shape Before vs After Cleaning")
    col1, col2 = st.columns(2)

    with col1:
        st.write("📊 **Before Cleaning**")
        st.write(f"Rows: {original_df.shape[0]}")
        st.write(f"Columns: {original_df.shape[1]}")

    with col2:
        st.write("✅ **After Cleaning**")
        st.write(f"Rows: {cleaned_df.shape[0]}")
        st.write(f"Columns: {cleaned_df.shape[1]}")

    # Missing Value Comparison
    st.subheader("❗ Missing Values: Before vs. After Cleaning")
    original_missing = original_df.isnull().sum()
    cleaned_missing = cleaned_df.isnull().sum()
    missing_comparison = pd.DataFrame({"Before Cleaning": original_missing, "After Cleaning": cleaned_missing})

    # Check if there are missing values to display
    if missing_comparison.sum().sum() > 0:
        st.bar_chart(missing_comparison)
    else:
        st.write("✅ No missing values in both datasets.")

    # Data Distribution Before vs After Cleaning
    st.subheader("📉 Data Distribution Before vs After Cleaning")
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("📌 Select a Numeric Column to Compare", numeric_cols)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.histplot(original_df[selected_col].dropna(), kde=True, color="red", ax=axes[0])
        axes[0].set_title(f"Before Cleaning: {selected_col}")

        sns.histplot(cleaned_df[selected_col].dropna(), kde=True, color="green", ax=axes[1])
        axes[1].set_title(f"After Cleaning: {selected_col}")

        st.pyplot(fig)
    else:
        st.write("⚠️ No numeric columns found for distribution comparison.")

    # Download Cleaned Data
    st.subheader("📥 Download Cleaned Dataset")
    cleaned_filename = "cleaned_" + st.session_state.get('filename', 'dataset.csv')
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📩 Download CSV",
        data=csv,
        file_name=cleaned_filename,
        mime="text/csv"
    )

    st.success("✅ The dataset is now cleaned and ready for visualization!")
