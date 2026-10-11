"""Signal Watch: a truth-seeking test of the hunch that a big AI-driven shift is coming.

The model does not assume when the shift happens or what it becomes. It reads
real monthly data (see data.py) and asks two questions of it:

1. WHAT is changing? For each indicator it measures momentum: how unusual the
   last year's change is compared with the indicator's own history (a z-score).
   Indicators are grouped by the outcome they speak to, so the groups with the
   strongest momentum show what kind of shift the data supports.

2. WHEN does it tip? For indicators that follow an adoption curve, it fits an
   S-curve (logistic) and estimates the inflection point, the month when change
   is fastest and the transition is half done. A block bootstrap (resampling
   the fit's residuals in runs of months) turns that into a distribution of
   dates instead of a single guess. When the data are still in the early
   exponential phase, the inflection can't be pinned down yet, and the model
   says so rather than reporting a number.

Pooling those distributions gives the data's answer to "when". The gut check
compares it with a hypothesis year (2027 by default).

Usage:
    python model.py                    # analyze cached data in data/series/
    python model.py --hypothesis 2028  # test a different year
    python model.py --refresh          # fetch fresh data first (needs internet)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

import data

REPORTS_DIR = Path("reports")
HISTORY_FILE = REPORTS_DIR / "history.csv"
DASHBOARD_DATA = Path("docs") / "forecast.json"

BOOTSTRAPS = 200
BACKTEST_BOOTSTRAPS = 60
BLOCK = 6  # months per resampled block, to respect autocorrelation
MIN_MONTHS = 36
BREAKOUT_Z = 2.0
# An inflection estimate counts as identified only if its 80% interval is this narrow (years).
MAX_INTERVAL_YEARS = 6.0
TIMELINE = (2015, 2040)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decimal_year(d: date) -> float:
    return d.year + (d.month - 0.5) / 12


def _year_label(y: float) -> str:
    year = int(math.floor(y))
    quarter = min(4, int((y - year) * 4) + 1)
    return f"{year}-Q{quarter}"


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Rolling median, to keep one-off viral spikes from dominating a fit."""
    out = values.copy()
    half = window // 2
    for i in range(len(values)):
        out[i] = np.median(values[max(0, i - half): i + half + 1])
    return out


def _logistic(t, ceiling, rate, midpoint, floor):
    return floor + ceiling / (1 + np.exp(-rate * (t - midpoint)))


# ---------------------------------------------------------------------------
# Momentum: is the last year unusual?
# ---------------------------------------------------------------------------

def momentum(values: np.ndarray, direction: int) -> dict:
    logs = np.log(np.maximum(values, 1e-9))
    change12 = logs[12:] - logs[:-12]
    history, current = change12[:-12], change12[-1]
    spread = history.std(ddof=1) or 1e-9
    z = float(np.clip(direction * (current - history.mean()) / spread, -5, 5))
    return {
        "change_12m": float(math.expm1(current)),
        "z": round(z, 2),
        "breakout": z >= BREAKOUT_Z,
    }


# ---------------------------------------------------------------------------
# Timing: when does the adoption curve tip?
# ---------------------------------------------------------------------------

def fit_inflection(t: np.ndarray, values: np.ndarray, rng: np.random.Generator,
                   bootstraps: int = BOOTSTRAPS) -> dict:
    y = _smooth(values) / values.max()
    bounds = ([0.01, 0.05, t[0] - 5, 0.0], [50.0, 10.0, t[-1] + 15, 1.0])
    p0 = [1.0, 1.0, t[-1], max(min(y.min(), 0.99), 0.0)]

    def fit(series):
        params, _ = curve_fit(_logistic, t, series, p0=p0, bounds=bounds, maxfev=5000)
        return params

    try:
        params = fit(y)
    except (RuntimeError, ValueError):
        return {"identified": False, "reason": "No S-curve fits these data."}
    fitted = _logistic(t, *params)
    residuals = y - fitted
    r2 = 1 - residuals.var() / y.var()

    samples = []
    n = len(y)
    for _ in range(bootstraps):
        starts = rng.integers(0, n - BLOCK + 1, size=math.ceil(n / BLOCK))
        resampled = np.concatenate([residuals[s:s + BLOCK] for s in starts])[:n]
        try:
            boot = fit(fitted + resampled)
            samples.append((boot[2], boot[2] - math.log(9) / boot[1]))
        except (RuntimeError, ValueError):
            continue
    if len(samples) < bootstraps // 2:
        return {"identified": False, "reason": "The S-curve fit is unstable."}
    samples, onsets = np.array(samples).T
    # Stability check: refit without the last 12 months. A reliable estimate shouldn't jump.
    try:
        earlier = float(fit(y[:-12])[2]) if len(y) - 12 >= MIN_MONTHS else None
    except (RuntimeError, ValueError):
        earlier = None
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])
    at_edge = p50 > t[-1] + 10
    identified = (p90 - p10) <= MAX_INTERVAL_YEARS and r2 >= 0.6 and not at_edge
    if identified:
        reason = ""
    elif r2 < 0.6:
        reason = "The data don't follow an S-curve closely enough to time it."
    elif p50 > t[-1]:
        reason = (f"Still in the early, accelerating phase. The tipping point is after "
                  f"{_year_label(min(p10, p50))} at the earliest, and the data can't pin it down yet.")
    else:
        reason = "The timing estimate is too uncertain to use."
    return {
        "identified": bool(identified),
        "reason": reason,
        "median": round(float(p50), 2),
        "p10": round(float(p10), 2),
        "p90": round(float(p90), 2),
        "past": bool(p90 < t[-1]),
        "r2": round(float(r2), 2),
        "year_ago_estimate": round(earlier, 2) if earlier is not None else None,
        # The shift "starts" when the curve has covered 10% of its rise.
        "start": {k: round(float(v), 2) for k, v in zip(("p10", "median", "p90"), np.percentile(onsets, [10, 50, 90]))},
        "samples": samples,
        "start_samples": onsets,
    }


# ---------------------------------------------------------------------------
# Whole analysis
# ---------------------------------------------------------------------------

def backtest(series: list[tuple[np.ndarray, np.ndarray, float]], rng: np.random.Generator) -> dict:
    """Would this method have called tipping points that have since happened?

    For every indicator whose tipping point is now confidently in the past, refit
    using only the data available 12, 24 and 36 months ago, and compare that
    earlier call with today's answer. An honest 80% range should contain the
    answer about 80% of the time.
    """
    checks = []
    for t, values, truth in series:
        for lag in (12, 24, 36):
            if len(values) - lag < MIN_MONTHS:
                continue
            past = fit_inflection(t[:-lag], values[:-lag], rng, bootstraps=BACKTEST_BOOTSTRAPS)
            if "median" not in past:
                checks.append({"lag": lag, "called": False})
                continue
            half = max((past["p90"] - past["p10"]) / 2, 0.05)
            checks.append({
                "lag": lag,
                "called": past["identified"],
                "error": abs(past["median"] - truth),
                "scaled_error": abs(past["median"] - truth) / half,
                "hit": past["p10"] <= truth <= past["p90"],
            })
    called = [c for c in checks if c["called"]]
    # How much wider the ranges must be for 80% of past calls to land inside them.
    widen = max(1.0, float(np.percentile([c["scaled_error"] for c in called], 80))) if len(called) >= 5 else 1.0
    return {
        "widen": round(widen, 2),
        "tests": len(checks),
        "indicators": len(series),
        "called": len(called),
        "hit_rate": round(float(np.mean([c["hit"] for c in called])), 2) if called else None,
        "mean_error_years": round(float(np.mean([c["error"] for c in called])), 2) if called else None,
        "by_lag": [
            {"lag": lag,
             "tests": sum(c["lag"] == lag for c in checks),
             "called": sum(c["lag"] == lag for c in called),
             "hit_rate": (round(float(np.mean([c["hit"] for c in called if c["lag"] == lag])), 2)
                          if any(c["lag"] == lag for c in called) else None)}
            for lag in (12, 24, 36)
        ],
    }


def analyze(hypothesis: int = 2027, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    indicators, pooled, pooled_start, completed = [], [], [], []
    for ind in data.INDICATORS:
        rows = data.load(ind.id)
        if len(rows) < MIN_MONTHS:
            continue
        # A final month far below the ones before it is almost always only partly reported.
        if len(rows) > 4 and rows[-1][1] < 0.4 * float(np.median([v for _, v in rows[-4:-1]])):
            rows = rows[:-1]
        if ind.rolling12:
            rows = [(rows[i][0], sum(v for _, v in rows[i - 11:i + 1])) for i in range(11, len(rows))]
        months = [d for d, _ in rows]
        values = np.array([v for _, v in rows], dtype=float)
        t = np.array([_decimal_year(d) for d in months])
        entry = {
            "id": ind.id, "name": ind.name, "group": ind.group, "kind": ind.kind,
            "source": ind.source, "url": ind.url, "note": ind.note,
            "first": months[0].isoformat(), "latest": months[-1].isoformat(),
            "latest_value": float(values[-1]),
            # Last 10 years, every month, for sparklines.
            "spark": [round(float(v), 4) for v in values[-120:]],
            **momentum(values, ind.direction),
        }
        if ind.kind == "adoption":
            timing = fit_inflection(t, values, rng)
            samples = timing.pop("samples", None)
            start_samples = timing.pop("start_samples", None)
            entry["timing"] = timing
            if timing["identified"] and samples is not None:
                pooled.append(samples)
                pooled_start.append(start_samples)
            if timing["identified"] and timing["past"]:
                completed.append((t, values, timing["median"]))
        indicators.append(entry)

    # Calibrate: widen every range by what the backtest says it takes to be honest.
    tested = backtest(completed, rng)
    k = tested["widen"]
    pooled = [np.median(s) + k * (s - np.median(s)) for s in pooled]
    pooled_start = [np.median(s) + k * (s - np.median(s)) for s in pooled_start]
    for entry in indicators:
        timing = entry.get("timing", {})
        if timing.get("identified") and k > 1:
            for est in (timing, timing["start"]):
                m = est["median"]
                est["p10"] = round(m - k * (m - est["p10"]), 2)
                est["p90"] = round(m + k * (est["p90"] - m), 2)

    # Groups: what kind of shift does the evidence point to?
    groups = []
    for gid, gname in data.GROUPS.items():
        members = [i for i in indicators if i["group"] == gid]
        if not members:
            continue
        mean_z = float(np.mean([i["z"] for i in members]))
        timed = [i["timing"] for i in members if i.get("timing", {}).get("identified")]
        groups.append({
            "id": gid, "name": gname,
            "momentum": round(mean_z, 2),
            "breakouts": sum(i["breakout"] for i in members),
            "count": len(members),
            "tipping": ({"median": round(float(np.median([x["median"] for x in timed])), 2),
                         "p10": round(min(x["p10"] for x in timed), 2),
                         "p90": round(max(x["p90"] for x in timed), 2)} if timed else None),
        })
    groups.sort(key=lambda g: g["momentum"], reverse=True)

    # When: pool the identified distributions, one equal share per indicator.
    when = None
    if pooled:
        when = _summarize(pooled, hypothesis, rng)
        when["start"] = _summarize(pooled_start, hypothesis, rng)

    acceleration = float(np.mean([norm.cdf(i["z"]) for i in indicators]) * 100) if indicators else 0.0
    verdict = _verdict(when, hypothesis, indicators, acceleration)
    return {
        "generated": datetime.now(timezone.utc).date().isoformat(),
        "hypothesis": hypothesis,
        "verdict": verdict,
        "acceleration_index": round(acceleration, 1),
        "breakouts": sum(i["breakout"] for i in indicators),
        "indicator_count": len(indicators),
        "when": when,
        "groups": groups,
        "indicators": indicators,
        "backtest": tested,
        "track_record": _track_record(),
        "method": {"bootstraps": BOOTSTRAPS, "block_months": BLOCK, "breakout_z": BREAKOUT_Z},
    }


def _summarize(pools: list[np.ndarray], hypothesis: int, rng: np.random.Generator) -> dict:
    each = min(len(s) for s in pools)
    allsamples = np.concatenate([rng.choice(s, each, replace=False) for s in pools])
    p10, p50, p90 = np.percentile(allsamples, [10, 50, 90])
    lo, hi = TIMELINE
    hist, _ = np.histogram(np.clip(allsamples, lo, hi), bins=np.arange(lo, hi + 0.25, 0.25))
    return {
        "median": round(float(p50), 2), "p10": round(float(p10), 2), "p90": round(float(p90), 2),
        "median_label": _year_label(p50),
        "indicators_timed": len(pools),
        "p_hypothesis_year": round(float(np.mean((allsamples >= hypothesis) & (allsamples < hypothesis + 1))), 3),
        "p_within_one_year": round(float(np.mean(np.abs(allsamples - (hypothesis + 0.5)) <= 1.5)), 3),
        "histogram": {"start": lo, "step": 0.25, "counts": hist.tolist()},
    }


def retarget(result: dict, hypothesis: int) -> dict:
    """Re-run the gut check for another year without refitting anything."""
    result = json.loads(json.dumps(result))
    result["hypothesis"] = hypothesis
    for est in filter(None, [result["when"], (result["when"] or {}).get("start")]):
        h = est["histogram"]
        counts = np.array(h["counts"], dtype=float)
        lefts = h["start"] + h["step"] * np.arange(len(counts))
        total = counts.sum() or 1
        est["p_hypothesis_year"] = round(float(counts[(lefts >= hypothesis) & (lefts < hypothesis + 1)].sum() / total), 3)
        est["p_within_one_year"] = round(float(counts[(lefts >= hypothesis - 1) & (lefts < hypothesis + 2)].sum() / total), 3)
    result["verdict"] = _verdict(result["when"], hypothesis, result["indicators"], result["acceleration_index"])
    return result


def _track_record() -> list[dict]:
    """Earlier runs' calls, so you can see whether the answer drifts or holds."""
    if not HISTORY_FILE.exists():
        return []
    with HISTORY_FILE.open(newline="") as f:
        return [{"date": r["date"], "median": float(r["tipping_median"])}
                for r in csv.DictReader(f) if r.get("tipping_median")]


def _verdict(when: dict | None, hypothesis: int, indicators: list[dict], acceleration: float) -> dict:
    if not indicators:
        return {"label": "No data yet", "detail": "Run python data.py to fetch the indicators."}
    breakouts = sum(i["breakout"] for i in indicators)
    if breakouts:
        now = (f" Right now {breakouts} indicator{'s are' if breakouts > 1 else ' is'} breaking out of "
               f"{'their' if breakouts > 1 else 'its'} normal range, which could be the first sign of a new wave.")
    else:
        toward = f" building toward {hypothesis}" if hypothesis > datetime.now(timezone.utc).year else ""
        now = (" Right now no indicator is moving faster than its own history, so the data show no sign yet "
               f"of a new wave{toward}.")
    if not when:
        if not breakouts:
            return {"label": "No shift detected",
                    "detail": "None of the adoption curves can be timed, and no indicator is moving faster "
                              "than its own history. The data show no sign of an unusual shift yet."}
        return {"label": "Too early to call",
                "detail": "None of the adoption curves can be timed yet, so the data can't confirm or rule "
                          "out your year." + now}
    # The hypothesis is about when the shift starts, so test the start estimate.
    start = when["start"]
    inside = start["p10"] <= hypothesis + 1 and start["p90"] >= hypothesis
    near = abs(start["median"] - (hypothesis + 0.5)) <= 1.0
    if near and inside:
        label, detail = "Supported", (f"The data's best estimate for when the shift starts ({start['median_label']}) "
                                       f"is within a year of {hypothesis}, and {hypothesis} is inside the 80% range.")
    elif inside:
        label, detail = "Plausible", (f"{hypothesis} is inside the data's 80% range for the start, but the best "
                                       f"estimate is {start['median_label']}.")
    elif start["median"] < hypothesis:
        label, detail = "Not supported", (f"The shift these indicators measure already started, around "
                                           f"{start['median_label']}. {hypothesis} is outside the 80% range.")
    else:
        label, detail = "Not supported", (f"The data say the shift starts later, around {start['median_label']}. "
                                           f"{hypothesis} is outside the 80% range.")
    return {"label": label, "detail": detail + now}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(result: dict) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    DASHBOARD_DATA.parent.mkdir(exist_ok=True)
    DASHBOARD_DATA.write_text(json.dumps(result, indent=1))

    w = result["when"]
    lines = [
        f"# Signal Watch, {result['generated']}",
        "",
        f"**Gut check ({result['hypothesis']}): {result['verdict']['label']}.** {result['verdict']['detail']}",
        "",
    ]
    if w:
        lines.append(f"Data-estimated tipping point: **{w['median_label']}** (80% range "
                     f"{_year_label(w['p10'])} to {_year_label(w['p90'])}), from {w['indicators_timed']} indicators.")
    lines += [
        f"Acceleration index: {result['acceleration_index']} / 100. "
        f"{result['breakouts']} of {result['indicator_count']} indicators are breaking out.",
        "", "## What is changing", "", "| Outcome | Momentum (z) | Breakouts | Tipping point |", "|---|---|---|---|",
    ]
    for g in result["groups"]:
        tip = (f"{_year_label(g['tipping']['median'])}" if g["tipping"] else "not identifiable yet")
        lines.append(f"| {g['name']} | {g['momentum']:+.2f} | {g['breakouts']}/{g['count']} | {tip} |")
    lines += ["", "## Indicators", "", "| Indicator | Source | Latest | 12-month change | z | Timing |",
              "|---|---|---|---|---|---|"]
    for i in result["indicators"]:
        timing = i.get("timing")
        tip = ("n/a (condition)" if not timing else
               _year_label(timing["median"]) if timing["identified"] else timing["reason"])
        lines.append(f"| {i['name']} | [{i['source']}]({i['url']}) | {i['latest'][:7]} | "
                     f"{i['change_12m']:+.0%} | {i['z']:+.2f} | {tip} |")
    lines.append("")
    path = REPORTS_DIR / f"{result['generated']}.md"
    path.write_text("\n".join(lines))

    # Prediction log, so the model's calls can be scored against reality later.
    rows = []
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != result["generated"]]
    rows.append({
        "date": result["generated"],
        "tipping_median": w["median"] if w else "",
        "tipping_p10": w["p10"] if w else "",
        "tipping_p90": w["p90"] if w else "",
        "acceleration_index": result["acceleration_index"],
        "leading_group": result["groups"][0]["id"] if result["groups"] else "",
    })
    with HISTORY_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hypothesis", type=int, default=2027, help="the year your gut says")
    parser.add_argument("--refresh", action="store_true", help="fetch fresh data first")
    args = parser.parse_args()
    if args.refresh:
        data.main()

    result = analyze(args.hypothesis)
    report = write_outputs(result)
    w = result["when"]
    print(f"Gut check {args.hypothesis}: {result['verdict']['label']}. {result['verdict']['detail']}")
    if w:
        print(f"Tipping point (data): {w['median_label']}, 80% range {_year_label(w['p10'])} to "
              f"{_year_label(w['p90'])}, P(in {args.hypothesis}) = {w['p_hypothesis_year']:.0%}")
    print(f"Acceleration index {result['acceleration_index']}/100, "
          f"{result['breakouts']}/{result['indicator_count']} indicators breaking out\n")
    for g in result["groups"]:
        print(f"  {g['momentum']:+.2f}  {g['name']}  ({g['breakouts']}/{g['count']} breaking out)")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
