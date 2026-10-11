# SIGNAL//WATCH

A truth-seeking test of a hunch: that a big shift tied to AI is coming, and that it starts around 2027.

Signal Watch doesn't assume the hunch is right. It reads real public data, works out from the data **what** is changing and **when** each change starts, and reports whether the evidence supports the year you pick. If the data say "already happened", "later" or "nothing unusual right now", that's what it shows. It also tests itself on trends that have already played out, and widens its uncertainty if its past calls were overconfident.

## What it measures

Sixteen monthly series from sources that need no API key, cached in `data/series/` so every number can be checked. A series with less than three years of data is skipped until it has enough history.

| Outcome | Indicators |
|---|---|
| Capability and research acceleration | AI research papers (OpenAlex), interest in large language models |
| Mass adoption and public attention | Interest in AI, ChatGPT, and artificial general intelligence |
| Labor and income transformation | US unemployment, information-sector jobs, labor productivity (FRED); interest in universal basic income and technological unemployment |
| New governance order | Interest in AI regulation and the EU AI Act |
| Compute build-out and concentration | US semiconductor production (FRED), interest in Nvidia |
| Risk concern and backlash | Interest in AI safety and existential risk from AI |

"Interest" means English Wikipedia pageviews. They measure attention, not change in the world. The FRED series measure real economic conditions but move slowly.

## How it works

1. **Momentum.** For each series, the last 12 months of change is compared with the series' own history as a z-score. A z of +2 or more counts as a breakout. Each outcome's momentum is the average of its indicators.
2. **Timing.** For adoption-type series, the model fits an S-curve (logistic) and reads off two dates. The shift *starts* when the curve has covered 10% of its rise, and it is *halfway* at the point of fastest change. A block bootstrap (200 refits on resampled residuals) turns each date into a range. If a curve is still in its early, accelerating phase, the model reports that its timing can't be pinned down yet instead of guessing.
3. **Backtest and calibration.** For every trend whose turning point is now clearly in the past, the model refits using only the data available 1, 2 and 3 years earlier and checks those calls against what happened. If its 80% ranges caught the answer less than 80% of the time, every range is widened to match.
4. **Gut check.** The calibrated start dates are pooled across indicators and compared with your year. The verdict is Supported, Plausible, Not supported, Too early to call, or No shift detected. It also says whether anything is breaking out right now, which is how a new wave would first show up.

Every run is logged in `reports/history.csv`, so you can see over time whether the answer holds or drifts.

## Run it

```bash
pip install -r requirements.txt
python data.py      # fetch fresh data (needs internet)
python model.py     # analyze and write docs/forecast.json and reports/
streamlit run app.py
```

`python model.py --hypothesis 2028` tests a different year from the command line. In the Streamlit app, the sidebar slider does the same instantly.

## Where it runs

- **Streamlit:** deploy this repo with main file `app.py`.
- **GitHub Pages (optional):** `docs/index.html` shows the same dashboard from `docs/forecast.json`. Turn it on under Settings → Pages with branch `main` and folder `/docs`.
- **GitHub Actions:** `forecast.yml` refreshes the data and refits the model every Monday. `refresh-data.yml` refreshes data on demand. `check.yml` runs the model on every push and pull request.

## Limits

- Wikipedia attention can spike for reasons that have nothing to do with structural change, such as a news event or a viral moment.
- An S-curve is one shape of change. A shift that comes in several waves will show the first wave clearly and the next one only once it starts.
- OpenAlex files many papers that only carry a year under January, so research volume is analyzed as a 12-month total, and the latest 9 months are left out while they are still being indexed.
