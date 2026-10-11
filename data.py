"""Real-world data feeds for Signal Watch.

Every indicator is a monthly time series pulled from a public source that needs
no API key. Series are cached as CSV files in data/series/ so the model and the
dashboard work without a network connection, and so every number the model
uses can be inspected.

Usage:
    python data.py            # refresh every series
    python data.py arxiv_ai   # refresh one series
"""

from __future__ import annotations

import csv
import io
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
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
    Indicator("arxiv_ai", "AI papers on arXiv per month", "capability", "adoption", "arXiv",
              "https://arxiv.org/list/cs.AI/recent",
              note="New submissions in cs.AI, cs.LG and cs.CL."),
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

def fetch_wikipedia(article: str) -> list[tuple[date, float]]:
    end = _last_complete_month()
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
        f"{article}/monthly/{START:%Y%m%d}00/{end:%Y%m}0100"
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


def fetch_arxiv(existing: list[tuple[date, float]]) -> list[tuple[date, float]]:
    """Monthly AI submissions. Only months not already cached (plus the last two) are fetched."""
    have = {d for d, _ in existing}
    end = _last_complete_month()
    months = _months(START, end)
    todo = [m for m in months if m not in have or m >= months[-2]]
    rows = dict(existing)
    for m in todo:
        nxt = date(m.year + m.month // 12, m.month % 12 + 1, 1)
        query = (f"(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND "
                 f"submittedDate:[{m:%Y%m%d}0000 TO {nxt:%Y%m%d}0000]")
        for attempt in range(3):
            try:
                resp = requests.get("https://export.arxiv.org/api/query",
                                    params={"search_query": query, "max_results": 0},
                                    headers=HEADERS, timeout=60)
                resp.raise_for_status()
                total = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", resp.text)
                if total:
                    rows[m] = float(total.group(1))
                break
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
        time.sleep(3)  # arXiv asks for at most one request every three seconds
    return sorted(rows.items())


def refresh(indicator_id: str) -> int:
    ind = BY_ID[indicator_id]
    if ind.id == "arxiv_ai":
        rows = fetch_arxiv(load(ind.id))
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
            print(f"ok    {indicator_id}: {n} months")
        except Exception as exc:  # keep the cached copy if a source is down
            failed.append(indicator_id)
            print(f"FAIL  {indicator_id}: {exc}")
    if failed and len(failed) == len(ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
