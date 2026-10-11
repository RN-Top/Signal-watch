"""Signal Watch: a forecaster for an AI-driven societal shift around 2027.

The model tracks a handful of competing outcome scenarios for what the shift
could turn into. Each scenario starts from a prior probability and is pushed up
or down by leading indicators (signals). Every run writes a dated Markdown
report and appends to a CSV history, so the forecast can be watched over time.

Usage:
    python app.py                 # fetch live signals where possible, write report
    python app.py --offline       # skip network calls, use baseline/override values
    python app.py --signals my.json   # override any signal value (0.0 - 1.0)
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

import requests

HORIZON = "2027-12-31"
REPORTS_DIR = Path("reports")
HISTORY_FILE = REPORTS_DIR / "history.csv"

# How strongly a signal moves a scenario's log-odds when it is at 0.0 or 1.0.
SENSITIVITY = 3.0


@dataclass
class Signal:
    """A leading indicator, normalized to 0.0 (quiet) .. 1.0 (extreme)."""

    id: str
    name: str
    baseline: float
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
        Signal("research_velocity", "AI research velocity (arXiv cs.AI/cs.LG/cs.CL)", 0.6),
        Signal("capability_jumps", "Frontier model capability jumps", 0.7),
        Signal("compute_investment", "AI compute and data-center investment", 0.8),
        Signal("labor_displacement", "AI-attributed layoffs and hiring freezes", 0.4),
        Signal("policy_activity", "AI legislation and regulatory action", 0.5),
        Signal("market_concentration", "Concentration of AI power in few labs/states", 0.7),
        Signal("public_concern", "Public concern and sentiment about AI", 0.6),
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
# Model
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def score(scenario: Scenario) -> dict:
    """Probability that the scenario defines the shift by the horizon."""
    log_odds = _logit(scenario.prior)
    drivers = []
    for signal_id, weight in scenario.weights.items():
        signal = SIGNALS[signal_id]
        push = weight * (signal.current - 0.5) * SENSITIVITY
        log_odds += push
        drivers.append((signal.name, push))
    drivers.sort(key=lambda d: abs(d[1]), reverse=True)
    return {
        "scenario": scenario,
        "probability": _sigmoid(log_odds),
        "drivers": drivers[:3],
    }


def run_forecast() -> list[dict]:
    results = [score(s) for s in SCENARIOS]
    return sorted(results, key=lambda r: r["probability"], reverse=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_report(results: list[dict], today: date) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    lines = [
        f"# Signal Watch forecast, {today.isoformat()}",
        "",
        f"Horizon: {HORIZON}. Each probability is scored independently, so "
        "several scenarios can be likely at once.",
        "",
        "## Scenarios",
        "",
        "| Scenario | Likelihood by horizon | Top drivers |",
        "|---|---|---|",
    ]
    for r in results:
        drivers = ", ".join(
            f"{name} ({'+' if push >= 0 else ''}{push:.2f})" for name, push in r["drivers"]
        )
        lines.append(f"| {r['scenario'].name} | {r['probability']:.0%} | {drivers} |")
    lines += ["", "## Signals", "", "| Signal | Value | Source |", "|---|---|---|"]
    for s in SIGNALS.values():
        lines.append(f"| {s.name} | {s.current:.2f} | {s.source} |")
    lines.append("")

    path = REPORTS_DIR / f"{today.isoformat()}.md"
    path.write_text("\n".join(lines))
    return path


def append_history(results: list[dict], today: date) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    new_file = not HISTORY_FILE.exists()
    with HISTORY_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["date", "scenario", "probability"])
        for r in results:
            writer.writerow([today.isoformat(), r["scenario"].id, f"{r['probability']:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="skip live data fetches")
    parser.add_argument("--signals", type=Path, help="JSON file of signal overrides (0..1)")
    args = parser.parse_args()

    overrides = json.loads(args.signals.read_text()) if args.signals else {}
    collect_signals(args.offline, overrides)

    results = run_forecast()
    today = datetime.now(timezone.utc).date()
    report = write_report(results, today)
    append_history(results, today)

    print(f"Signal Watch forecast for horizon {HORIZON}\n")
    for r in results:
        print(f"  {r['probability']:>5.0%}  {r['scenario'].name}")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
