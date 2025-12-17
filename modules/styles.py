import streamlit as st


def hide_sidebar_nav():
    """Hide auto-generated pages navigation sidebar"""
    st.markdown("""
    <style>
    /* Hide auto-generated pages navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


def custom_metric_style():
    """Custom style for metrics"""
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
