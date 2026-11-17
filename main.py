"""GTA VI launch-day Twitch forecast (gta6-twitch-ccu-forecast-01).

Forecasts the Twitch concurrent viewers of the "Grand Theft Auto VI"
category for hours 2..24 of release day (UTC), as five quantiles per hour,
from hour 1's print plus the release-day curves of past big launches
(the case's `comparables`).

Idea
----
A launch that unlocks at local midnight rolls around the world, so the
UTC release day has a shape: a modest first hour (only New Zealand and a
few regions live), then a climb as Europe and the Americas unlock, then an
evening peak and a late-night fall.  Past launches that look like that
have the same shape once each curve is divided by its own hour-1 print,
so the forecast is built from the comparables' log-ratios
ln(viewers_h / viewers_1):

1. Keep only donors whose release day looks like a rolling-midnight
   console launch: hour-1 viewers >= 5,000 and the day's peak between
   1.3x and 8x hour 1.  This drops simultaneous global PC unlocks (hour 1
   already the peak) and games whose UTC day starts hours before they
   unlock (peak ratios in the hundreds).
2. Per hour, regress the donors' log-ratios on their log hour-1 prints:
   bigger launches grow flatter through the day.  The fitted slope is
   shrunk by half (a hedge against over-fitting one noisy cross-section)
   and evaluated at this case's own log hour-1 print, giving the
   conditional mean log-ratio for the hour.
3. The spread is the residual standard deviation around that fitted line,
   smoothed over neighbouring hours (a centred 3-hour mean), so one
   noisy hour cannot make the band collapse or explode.
4. Quantiles are the conditional mean plus the standard normal quantile
   times the spread, exponentiated back and multiplied by hour 1's print.

If anything unexpected happens for a case (no usable donors, degenerate
fit, malformed fields), the forecast falls back to the persistence
baseline: every hour at hour 1's print with a spread that widens like a
random walk in log space.  A case is never left without a valid output
and the program never crashes on a case.

Python 3.13, standard library only.  Deterministic: no randomness, no
clock, no network; file order is sorted and every float operation is a
pure function of the input bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file would still import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- Model constants (validated by leave-one-out on the comparables) --
# FamA targets (17 rolling-midnight launches): LOO pinball 0.209, vs
# 0.281 for the persistence baseline on the same targets (0.930 on all
# usable targets).  Variant sweeps confirmed the design: slope shrink
# 0.5 beats 0.0 (0.214) and 1.0 (0.211); spread inflate 1.1 (0.2088) and
# a 5-hour smooth (0.2089) were within noise of 1.0 / 3-hour, so the
# design values below were kept rather than over-fitting LOO.
POOL_MIN_HOUR1 = 5000.0   # donor hour-1 print must reach this
POOL_PEAK_LO = 1.3        # donor day peak / hour 1, lower bound
POOL_PEAK_HI = 8.0        # donor day peak / hour 1, upper bound
SLOPE_SHRINK = 0.5        # fitted log-ratio slope is halved
SPREAD_SMOOTH_HALF = 1    # centred moving-average half-width for spread
SPREAD_FLOOR = 0.10       # log-spread never narrower than this
SPREAD_INFLATE = 1.0      # multiplicative spread calibration
MIN_DONORS_FOR_FIT = 3    # fewer donors in an hour -> use the plain mean
BASE_SIGMA = 0.15         # fallback baseline: log spread per sqrt hour


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's
    Math.round(x * 100) / 100 rounds it.  A whole number comes back as an
    int, so json.dumps writes 1240 where JavaScript writes 1240."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_time(text):
    """ISO 8601 timestamp (with a trailing Z) as an aware datetime."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def _release_day(comp):
    """A comparable's release-day curve as {hour 1..24: viewers}.

    Hours are counted from the comparable's own `day_start`; only points
    inside the 24 UTC hours of that day are kept.
    """
    start = _parse_time(comp["day_start"])
    curve = {}
    for point in comp.get("hourly") or []:
        try:
            when = _parse_time(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((when - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and viewers > 0:
            curve[hour] = viewers
    return curve


def _hour1_print(case):
    """Hour 1's print; a print below 1 counts as 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                return max(1.0, float(row["viewers"]))
            except (KeyError, TypeError, ValueError):
                break
    return 1.0


def _donor_pool(case):
    """Donors passing the rolling-midnight filter.

    Returns a list of (log hour-1 print, {hour: log-ratio}) with the
    GTA VI category's own pre-release history left out: it has no
    release day yet, so it cannot be a donor.
    """
    pool = []
    for comp in case.get("comparables") or []:
        name = str(comp.get("name") or "")
        if "Grand Theft Auto VI" in name:
            continue
        try:
            curve = _release_day(comp)
        except (KeyError, TypeError, ValueError):
            continue
        first = curve.get(1)
        if first is None or first < POOL_MIN_HOUR1:
            continue
        peak = max(curve.values())
        ratio = peak / first
        if not (POOL_PEAK_LO <= ratio <= POOL_PEAK_HI):
            continue
        log_first = math.log(first)
        ratios = {h: math.log(v / first) for h, v in curve.items()}
        pool.append((log_first, ratios))
    return pool


def _mean(values):
    return sum(values) / len(values)


def _fit_hour(points, log_p1):
    """Conditional mean and spread of the log-ratio for one hour.

    `points` is a list of (donor log hour-1, donor log-ratio).  The mean
    is the OLS line of log-ratio on log hour-1, slope shrunk by
    SLOPE_SHRINK, evaluated at this case's log hour-1.  The spread is
    the residual standard deviation around the shrunk line (the line the
    forecast actually uses).  With too few points, or no variation in
    log hour-1, the mean is the plain average and the spread falls back
    to the residuals around that average.
    """
    xs = [p[0] for p in points]
    rs = [p[1] for p in points]
    n = len(points)
    mean_x = _mean(xs)
    mean_r = _mean(rs)
    if n >= MIN_DONORS_FOR_FIT:
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 0:
            cov = sum((x - mean_x) * (r - mean_r)
                      for x, r in points)
            slope = SLOPE_SHRINK * cov / var_x
            mean = mean_r + slope * (log_p1 - mean_x)
            resid = [r - (mean_r + slope * (x - mean_x))
                     for x, r in points]
            if n > 1:
                spread = math.sqrt(sum(e * e for e in resid) / (n - 1))
            else:
                spread = 0.0
            return mean, spread
    # Too few points or no spread in launch size: plain average.
    resid = [r - mean_r for r in rs]
    if n > 1:
        spread = math.sqrt(sum(e * e for e in resid) / (n - 1))
    else:
        spread = 0.0
    return mean_r, spread


def _smooth(values):
    """Centred moving average over neighbouring hours (list, hours 2..24
    in order).  Edge entries average over the neighbours that exist."""
    half = SPREAD_SMOOTH_HALF
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(_mean(values[lo:hi]))
    return out


def _model_rows(case, p1):
    """The comparables forecast: one row per hour in case["forecast"]."""
    pool = _donor_pool(case)
    if len(pool) < MIN_DONORS_FOR_FIT:
        raise ValueError("donor pool too small")
    log_p1 = math.log(p1)
    hours = [int(row["hour"]) for row in case["forecast"]]
    means = {}
    spreads = {}
    for h in hours:
        points = [(x, ratios[h]) for x, ratios in pool if h in ratios]
        if not points:
            means[h] = 0.0
            spreads[h] = 0.3
            continue
        mean, spread = _fit_hour(points, log_p1)
        means[h] = mean
        spreads[h] = spread
    ordered = [spreads[h] for h in hours]
    smoothed = _smooth(ordered)
    rows = []
    for j, h in enumerate(hours):
        spread = max(SPREAD_FLOOR, smoothed[j] * SPREAD_INFLATE)
        qs = []
        prev = 0.0
        for z in Z:
            value = p1 * math.exp(means[h] + z * spread)
            value = round2(value)
            value = min(max(value, 0.01), 1e9)
            if value < prev:
                value = prev
            prev = value
            qs.append(value)
        rows.append({"hour": h, "q": qs})
    return rows


def _baseline_rows(case, p1):
    """Persistence fallback: every hour at hour 1's print, spread
    widening as a random walk in log space."""
    rows = []
    for row in case["forecast"]:
        h = int(row["hour"])
        qs = []
        prev = 0.0
        for z in Z:
            value = round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)))
            value = min(max(value, 0.01), 1e9)
            if value < prev:
                value = prev
            prev = value
            qs.append(value)
        rows.append({"hour": h, "q": qs})
    return rows


def _fallback_rows(p1):
    """Baseline rows when even the case's forecast list is unusable."""
    rows = []
    for h in range(2, 25):
        qs = []
        prev = 0.0
        for z in Z:
            value = round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)))
            value = min(max(value, 0.01), 1e9)
            if value < prev:
                value = prev
            prev = value
            qs.append(value)
        rows.append({"hour": h, "q": qs})
    return rows


def forecast_case(case):
    """Forecast one case, never raising: any failure degrades to the
    persistence baseline, which is always a valid forecast."""
    p1 = _hour1_print(case)
    try:
        if not case.get("forecast"):
            raise ValueError("no forecast hours")
        rows = _model_rows(case, p1)
    except Exception:
        try:
            rows = _baseline_rows(case, p1)
        except Exception:
            rows = _fallback_rows(p1)
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = {"hours": _fallback_rows(1.0)}
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
