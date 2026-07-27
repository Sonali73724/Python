import streamlit as st

def render_sidebar():
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🌍 GeoAI Console")
    st.sidebar.markdown("**Remote Sensing Research Platform**")

    st.sidebar.page_link("app.py", label="Dashboard", icon="🎯")
    st.sidebar.page_link("pages/01_Dataset.py", label="Dataset", icon="🗄️")
    st.sidebar.page_link("pages/02_Training.py", label="Training", icon="🧠")
    st.sidebar.page_link("pages/03_Evaluation.py", label="Evaluation", icon="📈")
    st.sidebar.page_link("pages/04_Comparison.py", label="Comparison", icon="⚖️")
    st.sidebar.page_link("pages/05_Change_Detection.py", label="Change Detection", icon="🌎")
    st.sidebar.page_link("pages/06_Error_Analysis.py", label="Error Analysis", icon="⚠️")
    st.sidebar.page_link("pages/07_UC_Merced.py", label="UC Merced", icon="✈️")
    st.sidebar.page_link("pages/08_About.py", label="About", icon="ℹ️")

    st.sidebar.divider()