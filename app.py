"""Signal Watch dashboard for Streamlit.

Shows the latest analysis from docs/forecast.json, which the weekly GitHub
Action rebuilds from fresh data. The sidebar re-runs the gut check for any year
instantly, without refitting the model.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

import model

ROOT = Path(__file__).parent
DASHBOARD = ROOT / "docs" / "index.html"
FORECAST = ROOT / "docs" / "forecast.json"

st.set_page_config(page_title="Signal Watch", page_icon="📡", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 0; max-width: 1320px; }
      header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_forecast(mtime: float) -> dict:
    return json.loads(FORECAST.read_text())


def render_dashboard(result: dict) -> str:
    return re.sub(
        r'fetch\("forecast.json".*?\}\);\n',
        lambda _: "render(" + json.dumps(result) + ");\n",
        DASHBOARD.read_text(),
        flags=re.S,
    )


with st.sidebar:
    st.header("Test your gut")
    st.caption("Pick the year you think the shift starts. The model checks it against the data.")
    hypothesis = st.slider("Your call", 2020, 2040, 2027, key="hypothesis")
    st.caption(
        "Data comes from OpenAlex, Wikipedia pageviews and FRED. The repo's GitHub Action "
        "refreshes it and refits the model every Monday."
    )

result = model.retarget(load_forecast(FORECAST.stat().st_mtime), hypothesis)

if hasattr(st, "iframe"):
    st.iframe(render_dashboard(result), height=3200)
else:  # Streamlit releases before st.iframe existed
    import streamlit.components.v1 as components

    components.html(render_dashboard(result), height=3200, scrolling=True)
