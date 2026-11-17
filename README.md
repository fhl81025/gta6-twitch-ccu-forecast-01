# GTA VI Twitch CCU Forecaster

Forecasts how many people watch **Grand Theft Auto VI** live on Twitch, hour by hour,
over its release day (19 November 2026, UTC) — for each of hours 2–24, five quantiles
(5%, 25%, 50%, 75%, 95%) of that hour's concurrent viewers in the
"Grand Theft Auto VI" category.

Built for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
`gta6-twitch-ccu-forecast-01`. The entry was submitted on 17 November 2026;
the final result is graded on the recorded release day.

## The idea

A game that unlocks at local midnight rolls around the world, so the UTC
release day has a shape: a modest first hour (only New Zealand and a few
regions live), then a climb as Europe and the Americas unlock, then an
evening peak and a late-night fall. Past launches that look like that have
the same shape once each curve is divided by its own hour-1 print, so the
forecast is built from the comparables' log-ratios `ln(viewers_h / viewers_1)`:

1. **Donor pool.** Keep only past launches whose release day looks like a
   rolling-midnight console launch: hour-1 viewers ≥ 5,000 and the day's
   peak between 1.3× and 8× hour 1. This drops simultaneous global PC
   unlocks (hour 1 already the peak) and games whose UTC day starts hours
   before they unlock.
2. **Per-hour fit.** For each hour, regress the donors' log-ratios on their
   log hour-1 prints — bigger launches grow flatter through the day. The
   fitted slope is shrunk by half (a hedge against over-fitting one noisy
   cross-section) and evaluated at this case's own log hour-1 print.
3. **Spread.** The residual standard deviation around that fitted line,
   smoothed over neighbouring hours (a centred 3-hour mean, floor 0.10),
   so one noisy hour cannot make the band collapse or explode.
4. **Quantiles.** Conditional mean plus the standard normal quantile times
   the spread, exponentiated back and multiplied by hour 1's print.

If anything unexpected happens for a case (no usable donors, degenerate
fit, malformed fields), the forecast falls back to a persistence baseline:
every hour at hour 1's print with a spread that widens like a random walk
in log space. A case is never left without a valid output, and the program
never crashes on a case.

## Validation

Leave-one-out over the comparables (pinball loss on the log scale, lower
is better):

| | LOO loss |
|---|---|
| This forecaster | **0.209** |
| Persistence baseline | 0.281 |

Variant sweeps confirmed the design: slope shrink 0.5 beats 0.0 (0.214)
and 1.0 (0.211); a spread inflate of 1.1 (0.2088) and a 5-hour smooth
(0.2089) were within noise of the chosen values, so they were kept rather
than over-fitting the validation.

## Usage

Python 3.13, standard library only — nothing to install. Deterministic:
no randomness, no clock, no network.

```sh
# Read cases from ./input, write forecasts to ./out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Without those variables it reads `/input/*.in` and writes `/output/*.out`
(one `.out` per case, 23 rows: hours 2–24, five quantiles each). Runs in
about 0.05 s per case.

## Files

- `main.py` — the whole forecaster (320 lines).
- `LICENSE` — MIT.
