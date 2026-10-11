# Signal Watch

A predictive model for a big AI-driven societal shift, with 2027 as the horizon.

The premise: something large is starting to move, and AI is at the center of it. Signal Watch does not try to say *whether* a shift happens. It tracks several competing **outcome scenarios** for *what the shift turns into*, and updates how likely each one is as the evidence comes in. Run it on a schedule and the forecast history shows whether the signals are building toward a 2027 turning point.

## Scenarios

| Scenario | What it means |
|---|---|
| Labor and income transformation | AI automates much of knowledge work and forces new income systems (national UBI, AI dividends, shorter work weeks). |
| New AI governance order | Binding international AI rules, frontier-model licensing, or a new global AI institution. |
| Concentration of power | A few AI labs and states pull far ahead and become the new center of economic and geopolitical power. |
| Scientific and economic acceleration | AI-driven discovery in science, medicine and energy sharply accelerates growth. |
| Backlash and slowdown | Public backlash, incidents or an investment pullback push the shift past 2027. |

## Signals

Each signal is a leading indicator scaled from 0.0 (quiet) to 1.0 (extreme):

- AI research velocity (fetched live from arXiv: AI papers in the last 30 days vs. a year earlier)
- Frontier model capability jumps
- AI compute and data-center investment
- AI-attributed layoffs and hiring freezes
- AI legislation and regulatory action
- Concentration of AI power in a few labs and states
- Public concern and sentiment about AI

Signals without a live source yet use a baseline value that you can override.

## How the model works

Each scenario starts from a prior probability. Every signal it depends on nudges its log-odds up or down by `weight × (signal − 0.5) × sensitivity`, and the result is turned back into a probability. Scenarios are scored independently, so more than one can be likely at once. The top three drivers of each score are listed in the report, so you can see *why* the model leans the way it does.

## Run it

```bash
pip install -r requirements.txt
python app.py              # fetch live signals, write a report
python app.py --offline    # no network, baseline values only
```

Override any signal with your own reading of the world:

```bash
echo '{"labor_displacement": 0.7, "policy_activity": 0.4}' > signals.json
python app.py --signals signals.json
```

Each run writes `reports/YYYY-MM-DD.md` and appends to `reports/history.csv`.

## Roadmap

1. Live fetchers for the remaining signals (news volume, legislation trackers, job postings, compute spend).
2. A weekly GitHub Action that runs the forecast and commits the report.
3. A "Propose a scenario" issue template; the Action scores new scenarios and replies with a forecast.
4. Calibration: compare past forecasts against what actually happened.
