from __future__ import annotations

import streamlit as st

from .pages import business_insights, market_overview, model_performance


st.set_page_config(page_title="Egypt Apartments", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["Market Overview", "Model Performance", "Business Insights"],
    index=0,
)

if page == "Market Overview":
    market_overview.show()
elif page == "Model Performance":
    model_performance.show()
else:
    business_insights.show()
