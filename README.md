# SIGNAL//WATCH

A predictive model for a big AI-driven societal shift, with 2027 as the horizon.

The premise: something large is starting to move, and AI is at the center of it. Signal Watch does not try to say *whether* a shift happens. It tracks several competing **outcome scenarios** for *what the shift turns into*, runs a Monte Carlo ensemble over them every week, and shows the result on a live dashboard.

## Dashboard

`docs/index.html` is the dashboard. It's served by GitHub Pages and reads `docs/forecast.json`. It shows:

- **Shift Proximity Index**: overall signal intensity from 0 to 100, with its uncertainty band
- **Scenario cards**: likelihood by the horizon, P10 to P90 range, tipping-point quarter and top drivers
- **Trajectory to 2028**: quarter-by-quarter cumulative likelihood with uncertainty bands
- **Signal array**: a radar of every leading indicator and its uncertainty
- **Countdown to 2027**

To turn it on: go to **Settings → Pages**, choose "Deploy from a branch", set the branch to `main` and the folder to `/docs`.

To view it locally:

```bash
python -m http.server -d docs 8000   # then open http://localhost:8000
```

## Scenarios

| Scenario | What it means |
|---|---|
| Labor and income transformation | AI automates much of knowledge work and forces new income systems (national UBI, AI dividends, shorter work weeks). |
| New AI governance order | Binding international AI rules, frontier-model licensing, or a new global AI institution. |
| Concentration of power | A few AI labs and states pull far ahead and become the new center of economic and geopolitical power. |
| Scientific and economic acceleration | AI-driven discovery in science, medicine and energy sharply accelerates growth. |
| Backlash and slowdown | Public backlash, incidents or an investment pullback push the shift past 2027. |

## Signals

Each signal is a leading indicator scaled from 0.0 (quiet) to 1.0 (extreme), with an uncertainty spread:

- AI research velocity (fetched live from arXiv: AI papers in the last 30 days vs. a year earlier)
- Frontier capability jumps
- Compute and data-center investment
- AI-driven labor displacement
- AI legislation and regulation
- Concentration of AI power
- Public concern about AI

Signals without a live source yet use a baseline value that you can override.

## How the model works

1. **Monte Carlo ensemble.** Each run draws 20,000 possible worlds. In each one, the signal readings are sampled from Beta distributions around their current values, and every model weight is jittered by its own uncertainty.
2. **Signal scoring.** Each scenario starts from a prior probability. Each signal it depends on moves its log-odds by `weight × (signal − 0.5) × sensitivity`.
3. **Cross-impact analysis.** Scenarios aren't independent. A cross-impact matrix, a standard method from futures studies, lets each scenario reinforce or suppress the others. For example, a backlash makes a new governance order more likely, and a governance order makes power concentration less likely.
4. **Trajectory.** Each scenario's probability by the horizon is converted into a quarterly hazard rate and projected through 2028. The **tipping point** is the quarter where the median trajectory crosses 50%.

The output is a likelihood and P10 to P90 band for each scenario, its trajectory, and the drivers behind it.

## Run it

```bash
pip install -r requirements.txt
python app.py                 # fetch live signals, write report + dashboard data
python app.py --offline       # no network, baseline values only
python app.py --draws 100000  # bigger ensemble
```

Override any signal with your own reading of the world:

```bash
echo '{"labor_displacement": 0.7, "policy_activity": 0.4}' > signals.json
python app.py --signals signals.json
```

Each run writes `reports/YYYY-MM-DD.md`, updates `reports/history.csv`, and refreshes `docs/forecast.json`.

## Automation

- `.github/workflows/forecast.yml` runs the model every Monday and commits the new forecast, which updates the dashboard. You can also start it by hand from the Actions tab.
- `.github/workflows/check.yml` runs the model on every pull request.

## Roadmap

1. Live feeds for the remaining signals (news volume, legislation trackers, job postings, compute spend).
2. Time-series foundation models (such as Chronos or TimesFM) to forecast each signal forward, not just read its current value.
3. A "Propose a scenario" issue template. The Action would score each new scenario and reply with its forecast.
4. Calibration: score past forecasts against what actually happened.
