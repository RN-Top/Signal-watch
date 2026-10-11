"""Signal Watch dashboard for Streamlit.

Runs the Monte Carlo forecast from model.py and renders the dashboard in
docs/index.html with the results. Signal readings can be adjusted live from the
sidebar to explore "what if" scenarios.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

import model

DASHBOARD = Path(__file__).parent / "docs" / "index.html"

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


@st.cache_data(ttl=6 * 3600, show_spinner="Reading live signals…")
def live_readings() -> dict[str, float]:
    """Fetch live signals once every few hours; return {signal_id: value}."""
    readings = {}
    for signal_id, fetch in model.FETCHERS.items():
        try:
            value = fetch()
        except Exception:  # network trouble should never break the dashboard
            value = None
        if value is not None:
            readings[signal_id] = value
    return readings


@st.cache_data(show_spinner="Running Monte Carlo ensemble…")
def run_forecast(readings: tuple[tuple[str, float], ...], live: tuple[str, ...], draws: int) -> dict:
    model.collect_signals(offline=True, overrides=dict(readings))
    for signal_id in live:
        model.SIGNALS[signal_id].source = "live"
    today = datetime.now(timezone.utc).date()
    return model.run_ensemble(draws, today, seed=42)


def render_dashboard(forecast: dict) -> str:
    html = DASHBOARD.read_text()
    html = re.sub(
        r'fetch\("forecast.json".*?\}\);\n',
        "render(" + json.dumps(forecast) + ");\n",
        html,
        flags=re.S,
    )
    return html


# Sidebar: tune the signals ------------------------------------------------
live = live_readings()
with st.sidebar:
    st.header("Signal controls")
    st.caption(
        "Each signal runs from 0 (quiet) to 1 (extreme). Move a slider to see how "
        "the forecast shifts. Live feeds start at their latest reading."
    )
    readings = {}
    for signal in model.SIGNALS.values():
        start = live.get(signal.id, signal.baseline)
        label = signal.name + (" · live" if signal.id in live else "")
        readings[signal.id] = st.slider(label, 0.0, 1.0, float(round(start, 2)), 0.01, key=signal.id)
    draws = st.select_slider("Simulated futures", [5_000, 20_000, 50_000, 100_000], value=20_000)
    if st.button("Reset to defaults"):
        for signal in model.SIGNALS.values():
            st.session_state.pop(signal.id, None)
        st.rerun()

unchanged_live = tuple(sid for sid, v in live.items() if abs(readings[sid] - round(v, 2)) < 1e-9)
forecast = run_forecast(tuple(sorted(readings.items())), unchanged_live, draws)

if hasattr(st, "iframe"):
    st.iframe(render_dashboard(forecast), height=1720)
else:  # Streamlit releases before st.iframe existed
    import streamlit.components.v1 as components

    components.html(render_dashboard(forecast), height=1720, scrolling=True)
