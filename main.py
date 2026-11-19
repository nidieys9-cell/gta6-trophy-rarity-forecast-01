"""GTA VI trophy rarity forecast: log-time trend + earned-counts model,
with a first-refresh anticipation for a wholly stale page.

Model, per trophy, in logit space (logit(p) = ln(p / (100 - p)), p in percent):
  * Trend: ordinary least squares of logit(rate) on x = ln(1 + hours since
    the release day began, 2026-11-19T00:00Z), fitted on the last 24 hours
    of reads when there are at least 3 distinct reads, else on all reads.
    The log-time axis makes growth saturate: a trophy climbs fast while
    players first reach it and then flattens, which is how launch-week
    earn rates behave.
  * Counts: the same log-time fit applied separately to log(players
    tracked) and to log(trophies earned by holders of this trophy),
    projected to the target and divided back into a rate.  This separates
    real progression (earned counts rising) from dilution (new players
    joining with few trophies), which a rate-only trend confounds.
  * The median is the average of the two projections in logit space,
    shrunk toward the last observed rate when the history has few
    distinct reads, and clamped so one noisy slope cannot throw a
    trophy to an extreme.
  * Spread: a base term growing with sqrt(horizon in days), plus the
    fit residual and an extrapolation-uncertainty term, so the 5%-95%
    band is honest about a 4.5-day horizon and narrow on a 90-minute one.

Stale-page refresh: if every read in the history is identical (same
players tracked, same rates) the page has not refreshed since launch
night, so no trend can be read off it at all.  Exophase refreshes a
game's counts on its own schedule, hours apart, and the first refresh
brings in a wave of newly tracked players who hold far fewer trophies
than the launch-night cohort (the kit's own account: the first refresh
took tracked players from 638 to 3,868 while trophies earned went from
943 to 4,276, so almost every newly tracked player held only the
opening trophy).  For a short horizon, when no refresh has been seen
yet, the median is therefore shifted down in logit space for every
trophy except a near-universal one (>= 99.5%, which newcomers hold
too), and the spread is widened to cover how uneven that dilution is
across trophies.  This branch only fires on a wholly flat history
with a short horizon; a final case with two days of 10-minute reads
has refreshes in its history and takes the trend/counts path instead.

Standard library only, offline, deterministic: files are read in name
order, all arithmetic is fixed-order floating point, no randomness.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# First-refresh anticipation (stale-history branch only, logit units).
REFRESH_SHIFT = -1.45
REFRESH_SPREAD = 0.45
UNIVERSAL_SPREAD = 0.15


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 20.0:
        return 99.99
    if lg <= -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, mirroring the starter's rounding."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def ols(xs, ys):
    """Least-squares line y ~ a + b*x; returns (a, b, residual_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid, n


def distinct_points(points):
    """Collapse repeated identical reads to one point each (a stale page
    repeats the same line for hours, and must not count as many reads)."""
    seen = []
    for pt in points:
        if pt not in seen:
            seen.append(pt)
    return seen


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))

    last = history[-1]
    last_players = last["players_tracked"]

    # Is the page wholly stale?  One distinct players count and one
    # distinct rate vector across the whole history.
    players_seen = {h["players_tracked"] for h in history}
    rates_seen = {tuple(h["rates"]) for h in history}
    wholly_flat = len(players_seen) == 1 and len(rates_seen) == 1
    short_horizon = case["horizon_minutes"] <= 720

    # Shared time axis and players series for the counts model.
    xs_all = [math.log1p(hours_since_release(parse_time(h["at"]))) for h in history]
    players_pts = distinct_points(list(zip(xs_all, [math.log(max(1, h["players_tracked"])) for h in history])))
    p_a, p_b, _, p_n = ols([p[0] for p in players_pts], [p[1] for p in players_pts])
    log_players_target = p_a + p_b * x_target if p_n >= 2 else math.log(max(1, last_players))

    rows = []
    for i, trophy in enumerate(trophies):
        series = []
        for h, x in zip(history, xs_all):
            r = h["rates"][i]
            if r is not None:
                series.append((x, logit(r)))
        pts = distinct_points(series)
        if pts:
            last_rate = inv_logit(pts[-1][1])
            last_logit = pts[-1][1]
        else:
            last_rate = 0.5
            last_logit = logit(0.5)
        universal = last_rate >= 99.5

        if wholly_flat and short_horizon:
            # First-refresh branch: no trend is observable, so anticipate
            # the dilution a refresh brings.  A universal trophy is held
            # by the newcomers too and does not fall.
            if universal:
                med_logit = last_logit
                spread = UNIVERSAL_SPREAD
            else:
                med_logit = last_logit + REFRESH_SHIFT
                spread = REFRESH_SPREAD
        else:
            # Trend fit: reads from the last 24 h when there are at
            # least 3 distinct ones, else all distinct reads.
            t_last = parse_time(history[-1]["at"])
            recent = []
            for h, (x, y) in zip(history, series):
                if (t_last - parse_time(h["at"])).total_seconds() <= 24 * 3600:
                    recent.append((x, y))
            recent = distinct_points(recent)
            fit_pts = recent if len(recent) >= 3 else pts
            a, b, resid, n_fit = ols([p[0] for p in fit_pts], [p[1] for p in fit_pts])
            # Shrink the slope when few distinct reads support it.  The
            # shrunk line is re-anchored on the fit's mean point, so the
            # intercept stays consistent with the shrunk slope (shrinking
            # the slope without moving the intercept would pivot the line
            # around x = 0, far outside the data).
            x_mean = sum(p[0] for p in fit_pts) / len(fit_pts)
            y_mean = sum(p[1] for p in fit_pts) / len(fit_pts)
            shrink = n_fit / (n_fit + 2.0) if n_fit else 0.0
            b *= shrink
            trend_logit = y_mean + b * (x_target - x_mean)
            # Earned-counts projection for this trophy.
            earned_pts = []
            for h, x in zip(history, xs_all):
                r = h["rates"][i]
                if r is not None and h["players_tracked"]:
                    earned = max(1.0, r * h["players_tracked"] / 100.0)
                    earned_pts.append((x, math.log(earned)))
            earned_pts = distinct_points(earned_pts)
            e_a, e_b, _, e_n = ols([p[0] for p in earned_pts], [p[1] for p in earned_pts])
            if e_n >= 2:
                e_x_mean = sum(p[0] for p in earned_pts) / len(earned_pts)
                e_y_mean = sum(p[1] for p in earned_pts) / len(earned_pts)
                e_b *= e_n / (e_n + 2.0)
                log_earned_target = e_y_mean + e_b * (x_target - e_x_mean)
                counts_rate = math.exp(log_earned_target - log_players_target) * 100.0
                counts_logit = logit(counts_rate)
            else:
                counts_logit = trend_logit
            med_logit = 0.5 * trend_logit + 0.5 * counts_logit
            # A single noisy slope must not throw a trophy to an extreme.
            delta_cap = 4.0 if horizon_days >= 1.0 else 1.0
            med_logit = min(last_logit + delta_cap, max(last_logit - 2.0, med_logit))
            # Spread: horizon base + fit residual + extrapolation doubt.
            base = 0.30 + 0.18 * math.sqrt(horizon_days)
            x_last = pts[-1][0] if pts else x_target
            extrap = abs(b) * max(0.0, x_target - x_last) * 0.35
            spread = math.sqrt(base * base + resid * resid + extrap * extrap)
            spread = min(1.30, max(0.25, spread))
            if universal:
                # A trophy everyone holds stays held; only the band's
                # lower edge carries the doubt.
                med_logit = last_logit
                spread = min(spread, 0.30)

        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * spread)
            qs.append(round4(clamp_pct(q)))
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
