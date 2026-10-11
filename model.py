"""Signal Watch: a forecaster for an AI-driven societal shift around 2027.

The model tracks competing outcome scenarios for what the shift could turn
into. Every run is a Monte Carlo ensemble: signal readings and model weights
are sampled from their uncertainty, scenarios interact through a cross-impact
matrix, and each scenario gets a probability with a P10-P90 band plus a
quarter-by-quarter trajectory. Results go to a dated Markdown report, a CSV
history, and docs/forecast.json, which feeds the dashboard in docs/.

Usage:
    python model.py                     # fetch live signals where possible
    python model.py --offline           # skip network calls
    python model.py --signals my.json   # override signal readings (0.0 - 1.0)
    python model.py --draws 50000       # ensemble size (default 20000)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HORIZON = date(2027, 12, 31)
TRAJECTORY_END = date(2028, 12, 31)
REPORTS_DIR = Path("reports")
HISTORY_FILE = REPORTS_DIR / "history.csv"
DASHBOARD_DATA = Path("docs") / "forecast.json"

# How strongly a signal moves a scenario's log-odds when it is at 0.0 or 1.0.
SENSITIVITY = 3.0
# Relative uncertainty applied to every weight in each ensemble draw.
WEIGHT_JITTER = 0.25
# How strongly scenarios reinforce or suppress each other.
CROSS_IMPACT_SCALE = 2.0


@dataclass
class Signal:
    """A leading indicator, normalized to 0.0 (quiet) .. 1.0 (extreme)."""

    id: str
    name: str
    baseline: float
    spread: float  # standard deviation of our uncertainty about the reading
    source: str = "baseline"
    value: float | None = None

    @property
    def current(self) -> float:
        return self.baseline if self.value is None else self.value


@dataclass
class Scenario:
    """A possible outcome of the shift, with weights on each signal (-1 .. 1)."""

    id: str
    name: str
    description: str
    prior: float
    weights: dict[str, float] = field(default_factory=dict)


SIGNALS: dict[str, Signal] = {
    s.id: s
    for s in [
        Signal("research_velocity", "AI research velocity", 0.6, 0.08),
        Signal("capability_jumps", "Frontier capability jumps", 0.7, 0.12),
        Signal("compute_investment", "Compute and data-center investment", 0.8, 0.08),
        Signal("labor_displacement", "AI-driven labor displacement", 0.4, 0.15),
        Signal("policy_activity", "AI legislation and regulation", 0.5, 0.12),
        Signal("market_concentration", "Concentration of AI power", 0.7, 0.10),
        Signal("public_concern", "Public concern about AI", 0.6, 0.12),
    ]
}

SCENARIOS: list[Scenario] = [
    Scenario(
        "labor_transformation",
        "Labor and income transformation",
        "AI automates a large share of knowledge work, forcing new income "
        "systems (UBI pilots at national scale, shorter work weeks, AI dividends).",
        prior=0.25,
        weights={
            "capability_jumps": 0.8,
            "labor_displacement": 1.0,
            "research_velocity": 0.4,
            "public_concern": 0.4,
        },
    ),
    Scenario(
        "governance_order",
        "New AI governance order",
        "Binding international AI rules, licensing of frontier models, or a "
        "new global institution for AI oversight is established.",
        prior=0.20,
        weights={
            "policy_activity": 1.0,
            "public_concern": 0.7,
            "capability_jumps": 0.5,
            "market_concentration": 0.3,
        },
    ),
    Scenario(
        "power_concentration",
        "Concentration of power",
        "A few AI labs and states pull far ahead, becoming the de facto new "
        "system of economic and geopolitical power.",
        prior=0.30,
        weights={
            "market_concentration": 1.0,
            "compute_investment": 0.8,
            "capability_jumps": 0.6,
            "policy_activity": -0.4,
        },
    ),
    Scenario(
        "scientific_acceleration",
        "Scientific and economic acceleration",
        "AI-driven discovery (science, medicine, energy) sharply accelerates "
        "growth and becomes the dominant story of the shift.",
        prior=0.25,
        weights={
            "research_velocity": 1.0,
            "capability_jumps": 0.8,
            "compute_investment": 0.5,
            "labor_displacement": -0.2,
        },
    ),
    Scenario(
        "backlash_slowdown",
        "Backlash and slowdown",
        "Public backlash, incidents, or an investment pullback stall AI "
        "deployment and the expected shift is delayed past 2027.",
        prior=0.15,
        weights={
            "public_concern": 1.0,
            "policy_activity": 0.5,
            "compute_investment": -0.6,
            "capability_jumps": -0.5,
        },
    ),
]

# Cross-impact matrix: CROSS_IMPACT[a][b] is how much scenario b becoming more
# likely pushes scenario a up (+) or down (-). This is the core idea of
# cross-impact analysis from futures studies: outcomes are not independent.
CROSS_IMPACT: dict[str, dict[str, float]] = {
    "labor_transformation": {"scientific_acceleration": 0.4, "backlash_slowdown": -0.5},
    "governance_order": {"power_concentration": 0.5, "backlash_slowdown": 0.4},
    "power_concentration": {"governance_order": -0.5, "scientific_acceleration": 0.3},
    "scientific_acceleration": {"backlash_slowdown": -0.6, "power_concentration": 0.2},
    "backlash_slowdown": {"labor_transformation": 0.5, "power_concentration": 0.3},
}


# ---------------------------------------------------------------------------
# Live signal fetchers. Each returns a 0..1 value or None if unavailable.
# ---------------------------------------------------------------------------

def _arxiv_count(start: date, end: date) -> int | None:
    query = (
        "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND "
        f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    )
    resp = requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": query, "max_results": 0},
        timeout=30,
    )
    resp.raise_for_status()
    match = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", resp.text)
    return int(match.group(1)) if match else None


def fetch_research_velocity() -> float | None:
    """Last 30 days of AI papers vs. the same window one year earlier."""
    today = datetime.now(timezone.utc).date()
    recent = _arxiv_count(today - timedelta(days=30), today)
    year_ago = _arxiv_count(today - timedelta(days=395), today - timedelta(days=365))
    if not recent or not year_ago:
        return None
    growth = recent / year_ago - 1.0  # e.g. 0.3 means +30% year over year
    # Map growth to 0..1: 0% -> 0.5, +50% or more -> ~0.95.
    return round(1 / (1 + math.exp(-growth * 6)), 3)


FETCHERS = {
    "research_velocity": fetch_research_velocity,
}


def collect_signals(offline: bool, overrides: dict[str, float]) -> None:
    for signal in SIGNALS.values():
        signal.value, signal.source = None, "baseline"
    if not offline:
        for signal_id, fetch in FETCHERS.items():
            try:
                value = fetch()
            except requests.RequestException as exc:
                print(f"warning: could not fetch {signal_id}: {exc}")
                value = None
            if value is not None:
                SIGNALS[signal_id].value = value
                SIGNALS[signal_id].source = "live"
    for signal_id, value in overrides.items():
        if signal_id not in SIGNALS:
            print(f"warning: unknown signal in overrides: {signal_id}")
            continue
        SIGNALS[signal_id].value = max(0.0, min(1.0, float(value)))
        SIGNALS[signal_id].source = "override"


# ---------------------------------------------------------------------------
# Monte Carlo ensemble
# ---------------------------------------------------------------------------

def _sample_signal(rng: np.random.Generator, signal: Signal, n: int) -> np.ndarray:
    """Draw signal readings from a Beta distribution with the signal's mean and spread."""
    m = min(max(signal.current, 0.01), 0.99)
    var = min(signal.spread**2, m * (1 - m) * 0.9)
    k = m * (1 - m) / var - 1
    return rng.beta(m * k, (1 - m) * k, size=n)


def _quarters(start: date, end: date) -> list[str]:
    labels, y, q = [], start.year, (start.month - 1) // 3 + 1
    while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
        labels.append(f"{y}-Q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return labels


def run_ensemble(draws: int, today: date, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    signal_draws = {sid: _sample_signal(rng, s, draws) for sid, s in SIGNALS.items()}

    priors = np.array([s.prior for s in SCENARIOS])
    log_odds = np.tile(np.log(priors / (1 - priors)), (draws, 1))
    contributions: dict[str, dict[str, np.ndarray]] = {}
    for i, scenario in enumerate(SCENARIOS):
        contributions[scenario.id] = {}
        for sid, weight in scenario.weights.items():
            w = weight * (1 + WEIGHT_JITTER * rng.standard_normal(draws))
            push = w * (signal_draws[sid] - 0.5) * SENSITIVITY
            log_odds[:, i] += push
            contributions[scenario.id][sid] = push

    # One round of cross-impact adjustment between scenarios.
    base = 1 / (1 + np.exp(-log_odds))
    ids = [s.id for s in SCENARIOS]
    adjusted = log_odds.copy()
    for i, a in enumerate(ids):
        for b, strength in CROSS_IMPACT.get(a, {}).items():
            j = ids.index(b)
            adjusted[:, i] += strength * (base[:, j] - priors[j]) * CROSS_IMPACT_SCALE
    prob = 1 / (1 + np.exp(-adjusted))

    # Trajectory: convert the by-horizon probability into a constant
    # quarterly hazard and project cumulative probability quarter by quarter.
    quarters = _quarters(today, TRAJECTORY_END)
    horizon_q = len(_quarters(today, HORIZON))
    steps = np.arange(1, len(quarters) + 1)

    scenarios_out = []
    for i, scenario in enumerate(SCENARIOS):
        p = prob[:, i]
        hazard = 1 - (1 - p) ** (1 / horizon_q)
        cumulative = 1 - (1 - hazard[:, None]) ** steps[None, :]
        p10, p50, p90 = np.percentile(cumulative, [10, 50, 90], axis=0)
        crossing = next((quarters[k] for k, v in enumerate(p50) if v >= 0.5), None)
        drivers = sorted(
            ((SIGNALS[sid].name, float(push.mean())) for sid, push in contributions[scenario.id].items()),
            key=lambda d: abs(d[1]),
            reverse=True,
        )
        scenarios_out.append({
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "probability": float(p.mean()),
            "p10": float(np.percentile(p, 10)),
            "p90": float(np.percentile(p, 90)),
            "tipping_point": crossing,
            "drivers": [{"signal": n, "push": round(v, 3)} for n, v in drivers[:3]],
            "trajectory": {
                "p10": [round(float(v), 4) for v in p10],
                "p50": [round(float(v), 4) for v in p50],
                "p90": [round(float(v), 4) for v in p90],
            },
        })
    scenarios_out.sort(key=lambda s: s["probability"], reverse=True)

    # Shift Proximity Index: overall intensity of the signals, 0-100.
    spi = np.mean(np.stack(list(signal_draws.values())), axis=0) * 100

    return {
        "generated": today.isoformat(),
        "horizon": HORIZON.isoformat(),
        "draws": draws,
        "shift_proximity_index": {
            "value": round(float(spi.mean()), 1),
            "p10": round(float(np.percentile(spi, 10)), 1),
            "p90": round(float(np.percentile(spi, 90)), 1),
        },
        "quarters": quarters,
        "scenarios": scenarios_out,
        "signals": [
            {"id": s.id, "name": s.name, "value": round(s.current, 3),
             "spread": s.spread, "source": s.source}
            for s in SIGNALS.values()
        ],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_report(forecast: dict) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    spi = forecast["shift_proximity_index"]
    lines = [
        f"# Signal Watch forecast, {forecast['generated']}",
        "",
        f"**Shift Proximity Index: {spi['value']} / 100** (P10-P90: {spi['p10']}-{spi['p90']}). "
        f"Horizon {forecast['horizon']}, {forecast['draws']:,} Monte Carlo draws.",
        "",
        "## Scenarios",
        "",
        "| Scenario | Likelihood by horizon | P10-P90 | Tipping point (50%) | Top drivers |",
        "|---|---|---|---|---|",
    ]
    for s in forecast["scenarios"]:
        drivers = ", ".join(f"{d['signal']} ({d['push']:+.2f})" for d in s["drivers"])
        lines.append(
            f"| {s['name']} | {s['probability']:.0%} | {s['p10']:.0%}-{s['p90']:.0%} "
            f"| {s['tipping_point'] or 'beyond 2028'} | {drivers} |"
        )
    lines += ["", "## Signals", "", "| Signal | Reading | Uncertainty | Source |", "|---|---|---|---|"]
    for s in forecast["signals"]:
        lines.append(f"| {s['name']} | {s['value']:.2f} | ±{s['spread']:.2f} | {s['source']} |")
    lines.append("")

    path = REPORTS_DIR / f"{forecast['generated']}.md"
    path.write_text("\n".join(lines))
    return path


def append_history(forecast: dict) -> list[dict]:
    """Append this run to the CSV history, replacing any earlier run from the same day."""
    REPORTS_DIR.mkdir(exist_ok=True)
    rows = []
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != forecast["generated"]]
    for s in forecast["scenarios"]:
        rows.append({"date": forecast["generated"], "scenario": s["id"],
                     "probability": f"{s['probability']:.4f}"})
    with HISTORY_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "scenario", "probability"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_dashboard_data(forecast: dict, history: list[dict]) -> None:
    DASHBOARD_DATA.parent.mkdir(exist_ok=True)
    forecast = {**forecast, "history": history}
    DASHBOARD_DATA.write_text(json.dumps(forecast, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="skip live data fetches")
    parser.add_argument("--signals", type=Path, help="JSON file of signal overrides (0..1)")
    parser.add_argument("--draws", type=int, default=20000, help="Monte Carlo ensemble size")
    parser.add_argument("--seed", type=int, help="random seed for reproducible runs")
    args = parser.parse_args()

    overrides = json.loads(args.signals.read_text()) if args.signals else {}
    collect_signals(args.offline, overrides)

    today = datetime.now(timezone.utc).date()
    forecast = run_ensemble(args.draws, today, args.seed)
    report = write_report(forecast)
    history = append_history(forecast)
    write_dashboard_data(forecast, history)

    spi = forecast["shift_proximity_index"]
    print(f"Signal Watch | horizon {forecast['horizon']} | Shift Proximity Index {spi['value']}/100\n")
    for s in forecast["scenarios"]:
        tip = s["tipping_point"] or "beyond 2028"
        print(f"  {s['probability']:>4.0%}  [{s['p10']:.0%}-{s['p90']:.0%}]  {s['name']}  (tipping point: {tip})")
    print(f"\nReport: {report}\nDashboard data: {DASHBOARD_DATA}")


if __name__ == "__main__":
    main()
