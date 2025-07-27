import streamlit as st
from streamlit_option_menu import option_menu

def main():
    st.set_page_config(page_title="Insight Lab", page_icon="🔍")

    st.title("🔍 Insight Lab ")
    st.subheader("*AI-Driven Data Insights*")

    
    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation", 
            options=["Home", "Before Cleaning", "Data Cleaning", "After Cleaning", "Visualization", "Ai Powered Report Generation"],
            icons=["house", "clipboard-data", "tools", "check-circle", "bar-chart", "file-earmark-text"],
            menu_icon="cast",
            default_index=0
        )
    
    if selected == "Home":
        import home
        home.show()
    elif selected == "Before Cleaning":
        import before_cleaning
        before_cleaning.show()
    elif selected == "Data Cleaning":
        import data_cleaning
        data_cleaning.show()
    elif selected == "After Cleaning":
        import after_cleaning
        after_cleaning.show()
    elif selected == "Visualization":
        import visualization
        visualization.show()
    elif selected == "Ai Powered Report Generation":
        import Ai_report
        Ai_report.show()

if __name__ == "__main__":
    main()
