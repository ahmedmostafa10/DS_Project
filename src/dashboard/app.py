from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from src.dashboard.pages import business_insights, market_overview, model_performance

# Ensure project root is on sys.path so absolute imports like `from src...` work
# when Streamlit runs the script via the CLI (e.g. `poetry run streamlit run ...`).
# app.py is at `src/dashboard/app.py` so parents[2] is the project root.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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
