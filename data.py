"""Real-world data feeds for Signal Watch.

Every indicator is a monthly time series pulled from a public source that needs
no API key. Series are cached as CSV files in data/series/ so the model and the
dashboard work without a network connection, and so every number the model
uses can be inspected.

Usage:
    python data.py            # refresh every series
    python data.py wiki_ai    # refresh one series
"""

from __future__ import annotations

import csv
import io
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

SERIES_DIR = Path(__file__).parent / "data" / "series"
START = date(2015, 7, 1)  # Wikipedia's pageview API begins July 2015
HEADERS = {"User-Agent": "signal-watch/1.0 (https://github.com/RN-Top/Signal-watch)"}


@dataclass(frozen=True)
class Indicator:
    id: str
    name: str
    group: str  # which outcome of the shift this indicator speaks to
    kind: str  # "adoption": an S-curve we can time; "condition": a level whose breakouts matter
    source: str
    url: str
    direction: int = 1  # for conditions: +1 if a rise means more change, -1 if a fall does
    note: str = ""
    rolling12: bool = False  # analyze as a trailing 12-month total (removes seasonal and dating artifacts)


GROUPS = {
    "capability": "Capability and research acceleration",
    "attention": "Mass adoption and public attention",
    "labor": "Labor and income transformation",
    "governance": "New governance order",
    "compute": "Compute build-out and concentration",
    "backlash": "Risk concern and backlash",
}


def _wiki(article: str) -> str:
    return f"https://en.wikipedia.org/wiki/{article}"


def _fred(series: str) -> str:
    return f"https://fred.stlouisfed.org/series/{series}"


INDICATORS: list[Indicator] = [
    Indicator("openalex_ai", "AI research papers per month", "capability", "adoption", "OpenAlex",
              "https://openalex.org/C154945302",
              note="Works tagged Artificial intelligence, as a trailing 12-month total, because many papers carry "
                   "only a publication year and get filed under January. The latest 9 months are left out "
                   "while OpenAlex is still indexing them.",
              rolling12=True),
    Indicator("wiki_llm", "Interest in large language models", "capability", "adoption", "Wikipedia pageviews",
              _wiki("Large_language_model")),
    Indicator("wiki_ai", "Interest in artificial intelligence", "attention", "adoption", "Wikipedia pageviews",
              _wiki("Artificial_intelligence")),
    Indicator("wiki_chatgpt", "Interest in ChatGPT", "attention", "adoption", "Wikipedia pageviews",
              _wiki("ChatGPT")),
    Indicator("wiki_agi", "Interest in artificial general intelligence", "attention", "adoption",
              "Wikipedia pageviews", _wiki("Artificial_general_intelligence")),
    Indicator("fred_unrate", "US unemployment rate", "labor", "condition", "FRED (BLS)", _fred("UNRATE")),
    Indicator("fred_usinfo", "US information-sector jobs", "labor", "condition", "FRED (BLS)", _fred("USINFO"),
              direction=-1, note="A fall means AI-exposed jobs are shrinking."),
    Indicator("fred_productivity", "US labor productivity", "labor", "condition", "FRED (BLS)", _fred("OPHNFB"),
              note="Quarterly series, carried forward to months."),
    Indicator("wiki_ubi", "Interest in universal basic income", "labor", "adoption", "Wikipedia pageviews",
              _wiki("Universal_basic_income")),
    Indicator("wiki_techunemp", "Interest in technological unemployment", "labor", "adoption",
              "Wikipedia pageviews", _wiki("Technological_unemployment")),
    Indicator("wiki_regulation", "Interest in AI regulation", "governance", "adoption", "Wikipedia pageviews",
              _wiki("Regulation_of_artificial_intelligence")),
    Indicator("wiki_euaiact", "Interest in the EU AI Act", "governance", "adoption", "Wikipedia pageviews",
              _wiki("Artificial_Intelligence_Act")),
    Indicator("fred_semis", "US semiconductor production", "compute", "condition", "FRED (Federal Reserve)",
              _fred("IPG3344S")),
    Indicator("wiki_nvidia", "Interest in Nvidia", "compute", "adoption", "Wikipedia pageviews", _wiki("Nvidia")),
    Indicator("wiki_aisafety", "Interest in AI safety", "backlash", "adoption", "Wikipedia pageviews",
              _wiki("AI_safety")),
    Indicator("wiki_xrisk", "Interest in existential risk from AI", "backlash", "adoption", "Wikipedia pageviews",
              _wiki("Existential_risk_from_artificial_intelligence")),
]
BY_ID = {i.id: i for i in INDICATORS}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def load(indicator_id: str) -> list[tuple[date, float]]:
    path = SERIES_DIR / f"{indicator_id}.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return [(date.fromisoformat(r["month"]), float(r["value"])) for r in csv.DictReader(f)]


def save(indicator_id: str, rows: list[tuple[date, float]]) -> None:
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted({d: v for d, v in rows}.items())
    with (SERIES_DIR / f"{indicator_id}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "value"])
        writer.writerows((d.isoformat(), round(v, 4)) for d, v in rows)


def _months(start: date, end: date) -> list[date]:
    out, d = [], start.replace(day=1)
    while d <= end:
        out.append(d)
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return out


def _last_complete_month() -> date:
    today = date.today()
    return date(today.year - (today.month == 1), (today.month - 2) % 12 + 1, 1)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _month_end(month: date) -> date:
    return date(month.year + month.month // 12, month.month % 12 + 1, 1) - timedelta(days=1)


def fetch_wikipedia(article: str) -> list[tuple[date, float]]:
    # The end date must be the month's last day, or the API returns only part of that month.
    end = _month_end(_last_complete_month())
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
        f"{article}/monthly/{START:%Y%m%d}00/{end:%Y%m%d}00"
    )
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return [
        (date(int(i["timestamp"][:4]), int(i["timestamp"][4:6]), 1), float(i["views"]))
        for i in resp.json()["items"]
    ]


def fetch_fred(series: str) -> list[tuple[date, float]]:
    resp = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series},
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = []
    for row in csv.reader(io.StringIO(resp.text)):
        try:
            raw.append((date.fromisoformat(row[0]), float(row[1])))
        except (ValueError, IndexError):
            continue  # header or missing value
    raw = [(d, v) for d, v in raw if d >= date(2010, 1, 1)]
    # Carry quarterly values forward so every series is monthly.
    out, i = [], 0
    for m in _months(raw[0][0], raw[-1][0]):
        while i + 1 < len(raw) and raw[i + 1][0] <= m:
            i += 1
        out.append((m, raw[i][1]))
    return out


def fetch_openalex(concept: str) -> list[tuple[date, float]]:
    """Monthly count of works tagged with an OpenAlex concept."""
    end = _last_complete_month()
    months = _months(START, end)[:-9]  # recent months are still being indexed
    rows = []
    for m in months:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"filter": f"concepts.id:{concept},from_publication_date:{m.isoformat()},"
                              f"to_publication_date:{_month_end(m).isoformat()}",
                    "per-page": 1, "mailto": "signal-watch@users.noreply.github.com"},
            headers=HEADERS, timeout=30,
        )
        resp.raise_for_status()
        rows.append((m, float(resp.json()["meta"]["count"])))
        time.sleep(0.2)
    return rows


def refresh(indicator_id: str) -> int:
    ind = BY_ID[indicator_id]
    if ind.source == "OpenAlex":
        rows = fetch_openalex(ind.url.rsplit("/", 1)[1])
    elif ind.source.startswith("Wikipedia"):
        rows = fetch_wikipedia(ind.url.rsplit("/", 1)[1])
    else:
        rows = fetch_fred(ind.url.rsplit("/", 1)[1])
    if rows:
        save(ind.id, rows)
    return len(rows)


def main() -> None:
    ids = sys.argv[1:] or [i.id for i in INDICATORS]
    failed = []
    for indicator_id in ids:
        try:
            n = refresh(indicator_id)
            print(f"ok    {indicator_id}: {n} months", flush=True)
        except Exception as exc:  # keep the cached copy if a source is down
            failed.append(indicator_id)
            print(f"FAIL  {indicator_id}: {exc}", flush=True)
    if failed and len(failed) == len(ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
