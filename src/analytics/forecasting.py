"""
Demand forecasting for the pharmaceutical supply chain.

Business purpose
----------------
In pharma, demand planning is the upstream driver of almost every other cost in
the network. A monthly demand plan sets the API procurement volume, the batch
release schedule, the safety stock held in each distribution centre and the
cold-chain capacity booked with carriers. Because manufacturing lead times run
to several months (procurement -> manufacturing -> QC release -> packaging), the
plan has to be committed long before the demand is observed.

The cost of being wrong is asymmetric and both directions hurt:

* **Under-forecast** -> stock-outs, back-orders, therapy interruption for
  patients and SLA penalties with hospital and pharmacy customers.
* **Over-forecast** -> excess finished goods sitting in a warehouse against a
  fixed expiry date, ending in a write-off at near-zero salvage value.

This module therefore does three things, in order of increasing business value:

1. **Describe** history - aggregate the demand fact table into a clean monthly
   series and decompose it into trend, seasonality and residual noise, so a
   planner can see *why* demand moves (respiratory and cardiovascular products
   peak in winter) rather than just *that* it moved.
2. **Forecast** forward over the planning horizon with three candidate models
   of increasing sophistication.
3. **Prove** the chosen model - a walk-forward backtest on a genuine hold-out
   period, reporting MAE, RMSE, MAPE and bias per method. A forecast a planner
   cannot audit is a forecast a planner will not use, so the ranking table is a
   first-class output of this module, not a diagnostic afterthought.

Design notes
------------
* Pure analytics: pandas / numpy / scipy / statsmodels only. No Streamlit and no
  plotting, so the same functions serve the dashboard, the notebooks and tests.
* Every threshold (horizon, seasonal period, hold-out length, confidence level,
  candidate methods) is read from ``config/config.yaml`` under ``forecasting``.
* Every public function accepts an optional pre-loaded frame as its first
  argument, so the presentation layer can push already-filtered data down
  instead of forcing a reload.

Example
-------
>>> from src.analytics import forecasting as fc
>>> fc.demand_kpis()["yoy_growth_pct"]
>>> fc.backtest_forecast().head(1)["method"].iloc[0]
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)

#: Canonical measure of the demand signal. Planning is done on what customers
#: asked for, not on what we managed to ship - fulfilment is a supply
#: constraint, and forecasting it would bake today's stock-outs into
#: tomorrow's plan (the classic "demand censoring" trap).
DEMAND_COLUMN: Final[str] = "units_demanded"

#: Additive measures that are meaningful to sum across drugs and regions.
_SUM_COLUMNS: Final[tuple[str, ...]] = (
    "units_demanded",
    "units_fulfilled",
    "units_backordered",
)

#: User-facing frequency codes mapped to period-start pandas offsets. The demand
#: fact table is stamped on the first of the month, so anchoring on period start
#: keeps history and forecast on the same calendar grid.
_FREQ_ALIASES: Final[dict[str, str]] = {
    "M": "MS", "MS": "MS", "ME": "MS",
    "Q": "QS", "QS": "QS", "QE": "QS",
    "Y": "YS", "YS": "YS", "A": "YS",
    "W": "W", "D": "D",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _forecast_cfg() -> Any:
    """Return the ``forecasting`` configuration block."""
    return get_config().forecasting


def _resolve_freq(freq: str) -> str:
    """Translate a friendly frequency code into a pandas period-start offset."""
    key = str(freq).upper()
    if key not in _FREQ_ALIASES:
        raise ValueError(
            f"Unsupported frequency '{freq}'. Supported: {sorted(_FREQ_ALIASES)}"
        )
    return _FREQ_ALIASES[key]


def _z_score() -> float:
    """Two-sided normal critical value for the configured confidence level."""
    level = float(_forecast_cfg().confidence_level)
    return float(stats.norm.ppf(0.5 + level / 2.0))


def _load(demand: pd.DataFrame | None) -> pd.DataFrame:
    """Return the caller's frame, or load the demand fact table if none given."""
    return loader.load_demand() if demand is None else demand.copy()


def _as_series(data: pd.DataFrame | pd.Series, value_col: str = DEMAND_COLUMN) -> pd.Series:
    """Coerce an aggregated frame (or a raw series) into a date-indexed series.

    Accepting either shape lets callers chain ``aggregate_demand`` straight into
    the modelling functions without knowing the internal representation.
    """
    if isinstance(data, pd.Series):
        series = data.copy()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("A Series input must be indexed by date.")
    else:
        frame = data.copy()
        if "date" not in frame.columns:
            raise KeyError("Expected a 'date' column in the aggregated demand frame.")
        if value_col not in frame.columns:
            raise KeyError(f"Expected a '{value_col}' column in the aggregated demand frame.")
        series = pd.Series(
            frame[value_col].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(frame["date"])),
            name=value_col,
        )
    series = series.astype(float).sort_index()
    # An explicit frequency is what lets statsmodels emit dated forecasts.
    if series.index.freq is None:
        series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
        series = series.asfreq("MS") if len(series) > 1 else series
    return series


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Build the out-of-sample date index immediately following ``series``."""
    freq = series.index.freq or pd.tseries.frequencies.to_offset("MS")
    return pd.date_range(series.index[-1] + freq, periods=horizon, freq=freq)


def _month_dummies(index: pd.DatetimeIndex, drop_first: bool = True) -> pd.DataFrame:
    """Calendar-month indicator matrix used by the linear-trend model.

    Month dummies are the simplest honest way to let a regression carry the
    winter peak of respiratory and cardiovascular products without assuming a
    smooth sinusoid.
    """
    months = pd.Categorical(index.month, categories=list(range(1, 13)))
    dummies = pd.get_dummies(months, prefix="m", drop_first=drop_first, dtype=float)
    dummies.index = index
    return dummies


def _has_enough_history(series: pd.Series) -> bool:
    """True when the series spans at least two full seasonal cycles.

    Holt-Winters must observe a seasonal pattern repeat before it can separate
    season from trend; statsmodels refuses to fit otherwise.
    """
    return len(series.dropna()) >= 2 * int(_forecast_cfg().seasonal_periods)


def _residual_interval(
    point: np.ndarray, residuals: np.ndarray, fallback_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Widen a point forecast into a prediction interval from in-sample errors.

    Holt-Winters in statsmodels exposes no closed-form prediction interval, so
    the standard practical substitute is used: the in-sample one-step residual
    standard deviation, inflated by ``sqrt(h)`` because forecast errors
    accumulate roughly as a random walk the further ahead we look.
    """
    clean = np.asarray(residuals, dtype=float)
    clean = clean[np.isfinite(clean)]
    sigma = float(np.std(clean, ddof=1)) if clean.size > 1 else float(fallback_scale)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(fallback_scale)
    steps = np.arange(1, len(point) + 1, dtype=float)
    half_width = _z_score() * sigma * np.sqrt(steps)
    # Demand cannot be negative; clipping keeps the lower band physically sane.
    return np.maximum(point - half_width, 0.0), point + half_width


# ---------------------------------------------------------------------------
# 1. Aggregation
# ---------------------------------------------------------------------------
def aggregate_demand(
    demand: pd.DataFrame | None = None,
    drug_code: str | None = None,
    region: str | None = None,
    freq: str = "M",
) -> pd.DataFrame:
    """Aggregate the demand fact table into a filtered periodic time series.

    Parameters
    ----------
    demand : pandas.DataFrame, optional
        Pre-loaded demand fact table. When ``None`` the table is loaded through
        :func:`src.data.loader.load_demand`.
    drug_code : str, optional
        Restrict to a single product (e.g. ``'DRG-A'``). ``None`` keeps all.
    region : str, optional
        Restrict to a single sales region. ``None`` keeps all.
    freq : str, default ``'M'``
        Resampling frequency: ``'M'``/``'Q'``/``'Y'``/``'W'``/``'D'``. Monthly is
        the planning cadence used throughout the platform.

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``units_demanded``, ``units_fulfilled``,
        ``units_backordered``, sorted ascending by date with no gaps.
    """
    frame = _load(demand)
    if drug_code is not None:
        frame = frame[frame["drug_code"] == drug_code]
    if region is not None:
        frame = frame[frame["region"] == region]

    if frame.empty:
        log.warning("aggregate_demand: no rows for drug_code=%s region=%s", drug_code, region)
        return pd.DataFrame(columns=["date", *_SUM_COLUMNS])

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rule = _resolve_freq(freq)

    series = (
        frame.set_index("date")[list(_SUM_COLUMNS)]
        .resample(rule)
        .sum()
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    log.info(
        "Aggregated demand to %d %s periods (drug=%s, region=%s, total %.0f units)",
        len(series), rule, drug_code or "ALL", region or "ALL",
        series[DEMAND_COLUMN].sum(),
    )
    return series


# ---------------------------------------------------------------------------
# 2. Decomposition
# ---------------------------------------------------------------------------
def decompose_demand(
    series: pd.DataFrame | pd.Series | None = None,
    value_col: str = DEMAND_COLUMN,
    model: str = "additive",
) -> pd.DataFrame:
    """Split a demand series into trend, seasonal and residual components.

    Separating the components tells a planner which part of a movement is
    structural growth (drives capacity decisions) and which part is the annual
    seasonal swing (drives pre-build and safety-stock timing). What is left in
    the residual is the genuinely unpredictable variation the safety stock has
    to absorb.

    Parameters
    ----------
    series : pandas.DataFrame or pandas.Series, optional
        Output of :func:`aggregate_demand`, or a date-indexed series. When
        ``None`` the full unfiltered monthly series is used.
    value_col : str, default ``'units_demanded'``
        Column to decompose when a DataFrame is supplied.
    model : {'additive', 'multiplicative'}, default ``'additive'``
        Decomposition form passed to
        :func:`statsmodels.tsa.seasonal.seasonal_decompose`.

    Returns
    -------
    pandas.DataFrame
        Tidy frame with columns ``date``, ``observed``, ``trend``, ``seasonal``,
        ``residual``. Trend and residual are ``NaN`` in the half-window at each
        end of the sample, which is inherent to a centred moving-average filter.

    Raises
    ------
    ValueError
        If the series is shorter than two full seasonal cycles, which is the
        minimum statsmodels needs to identify a seasonal component.
    """
    data = aggregate_demand() if series is None else series
    values = _as_series(data, value_col)
    period = int(_forecast_cfg().seasonal_periods)

    if len(values) < 2 * period:
        raise ValueError(
            f"Need at least {2 * period} periods to decompose a series with a "
            f"seasonal period of {period}; received {len(values)}."
        )

    result = seasonal_decompose(values, model=model, period=period, extrapolate_trend=0)
    tidy = pd.DataFrame(
        {
            "date": values.index,
            "observed": result.observed.to_numpy(dtype=float),
            "trend": result.trend.to_numpy(dtype=float),
            "seasonal": result.seasonal.to_numpy(dtype=float),
            "residual": result.resid.to_numpy(dtype=float),
        }
    ).reset_index(drop=True)

    amplitude = float(np.nanmax(tidy["seasonal"]) - np.nanmin(tidy["seasonal"]))
    log.info(
        "Decomposed %d periods (%s, period=%d): peak-to-trough seasonal swing %.0f units",
        len(tidy), model, period, amplitude,
    )
    return tidy


# ---------------------------------------------------------------------------
# 3. Forecasting
# ---------------------------------------------------------------------------
def _fit_holt_winters(series: pd.Series, horizon: int) -> pd.DataFrame:
    """Additive Holt-Winters (level + trend + seasonal) forecast."""
    period = int(_forecast_cfg().seasonal_periods)
    fitted = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=period,
        initialization_method="estimated",
    ).fit(optimized=True)

    point = np.asarray(fitted.forecast(horizon), dtype=float)
    residuals = np.asarray(series.to_numpy(float) - fitted.fittedvalues.to_numpy(float))
    lower, upper = _residual_interval(point, residuals, fallback_scale=float(series.std(ddof=1)))
    return pd.DataFrame(
        {"date": _future_index(series, horizon), "forecast": point,
         "lower_ci": lower, "upper_ci": upper, "method": "holt_winters"}
    )


def _fit_linear_trend(series: pd.Series, horizon: int) -> pd.DataFrame:
    """OLS on a time index with calendar-month dummies.

    Interpretable by construction: the time coefficient is the average units of
    growth per period, which is exactly the number a commercial planner quotes.
    """
    period = int(_forecast_cfg().seasonal_periods)
    n = len(series)
    t = np.arange(n, dtype=float)

    # Dummies cost 11 degrees of freedom; only spend them when the sample can
    # afford it, otherwise the fit is over-parameterised and the CI meaningless.
    use_seasonal = n >= period + 2
    if use_seasonal:
        dummies = _month_dummies(pd.DatetimeIndex(series.index))
        exog = pd.concat([pd.Series(t, index=series.index, name="t"), dummies], axis=1)
    else:
        log.warning(
            "linear_trend: only %d periods available (< %d), fitting trend without "
            "month dummies.", n, period + 2,
        )
        exog = pd.DataFrame({"t": t}, index=series.index)

    exog = sm.add_constant(exog, has_constant="add")
    model = sm.OLS(series.to_numpy(float), exog.astype(float)).fit()

    future_index = _future_index(series, horizon)
    future_t = np.arange(n, n + horizon, dtype=float)
    if use_seasonal:
        future_dummies = _month_dummies(future_index)
        future_exog = pd.concat(
            [pd.Series(future_t, index=future_index, name="t"), future_dummies], axis=1
        )
    else:
        future_exog = pd.DataFrame({"t": future_t}, index=future_index)
    future_exog = sm.add_constant(future_exog, has_constant="add")
    future_exog = future_exog.reindex(columns=exog.columns, fill_value=0.0).astype(float)

    alpha = 1.0 - float(_forecast_cfg().confidence_level)
    summary = model.get_prediction(future_exog).summary_frame(alpha=alpha)
    return pd.DataFrame(
        {
            "date": future_index,
            "forecast": summary["mean"].to_numpy(dtype=float),
            # Observation (prediction) interval, not the mean interval: the
            # planner needs the range a single future month can land in.
            "lower_ci": np.maximum(summary["obs_ci_lower"].to_numpy(dtype=float), 0.0),
            "upper_ci": summary["obs_ci_upper"].to_numpy(dtype=float),
            "method": "linear_trend",
        }
    )


def _fit_moving_average(series: pd.Series, horizon: int, window: int | None = None) -> pd.DataFrame:
    """Trailing-window mean - the naive benchmark every other model must beat.

    The default window is one full seasonal cycle, so the average is taken over
    a complete year and seasonality cancels out instead of leaking into the
    level.
    """
    period = int(_forecast_cfg().seasonal_periods)
    effective = int(window or period)
    effective = max(1, min(effective, len(series)))

    level = float(series.iloc[-effective:].mean())
    point = np.full(horizon, level, dtype=float)

    # One-step-ahead in-sample errors of the same rule give an honest error scale.
    rolled = series.rolling(effective).mean().shift(1)
    residuals = (series - rolled).dropna().to_numpy(dtype=float)
    lower, upper = _residual_interval(point, residuals, fallback_scale=float(series.std(ddof=1)))
    return pd.DataFrame(
        {"date": _future_index(series, horizon), "forecast": point,
         "lower_ci": lower, "upper_ci": upper, "method": "moving_average"}
    )


def _require_history(
    series: pd.Series, minimum: int, drug_code: str | None, region: str | None
) -> None:
    """Fail loudly, and with the offending filter named, on unusable history.

    A silently empty series almost always means a mistyped filter value coming
    from the UI layer, so the message has to point at the filter rather than at
    the modelling code.
    """
    if len(series) >= minimum:
        return
    scope = f"drug_code={drug_code or 'ALL'}, region={region or 'ALL'}"
    raise ValueError(
        f"Only {len(series)} demand period(s) available for {scope}; at least "
        f"{minimum} are required. Check that the filter values exist in the data."
    )


def _forecast_series(series: pd.Series, horizon: int, method: str) -> pd.DataFrame:
    """Dispatch to a fitted model, degrading gracefully on short histories."""
    methods = list(_forecast_cfg().methods)
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Configured methods: {methods}")
    if len(series) < 2:
        raise ValueError("At least two observations are required to forecast.")

    if method == "holt_winters" and not _has_enough_history(series):
        log.warning(
            "holt_winters needs >= %d observations (2 full seasonal cycles) but the "
            "series has %d; falling back to linear_trend.",
            2 * int(_forecast_cfg().seasonal_periods), len(series),
        )
        method = "linear_trend"

    if method == "holt_winters":
        try:
            return _fit_holt_winters(series, horizon)
        except Exception as exc:  # pragma: no cover - optimiser edge cases
            log.warning("holt_winters failed (%s); falling back to linear_trend.", exc)
            return _fit_linear_trend(series, horizon)
    if method == "linear_trend":
        return _fit_linear_trend(series, horizon)
    return _fit_moving_average(series, horizon)


def forecast_demand(
    demand: pd.DataFrame | None = None,
    drug_code: str | None = None,
    region: str | None = None,
    horizon: int | None = None,
    method: str = "holt_winters",
) -> pd.DataFrame:
    """Forecast future monthly demand with prediction intervals.

    Parameters
    ----------
    demand : pandas.DataFrame, optional
        Pre-loaded demand fact table; ``None`` loads it.
    drug_code : str, optional
        Restrict to a single product before aggregating.
    region : str, optional
        Restrict to a single region before aggregating.
    horizon : int, optional
        Number of future months. Defaults to ``forecasting.horizon_months``.
    method : {'holt_winters', 'linear_trend', 'moving_average'}
        Candidate model, validated against ``forecasting.methods``.

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``forecast``, ``lower_ci``, ``upper_ci``, ``method``.
        Dates are strictly in the future relative to the training history, and
        the interval width follows ``forecasting.confidence_level``.

    Notes
    -----
    Holt-Winters needs two complete seasonal cycles. On a heavily filtered or
    short series the call transparently falls back to ``linear_trend`` and logs
    a warning; the returned ``method`` column always names the model that was
    actually used.
    """
    steps = int(horizon if horizon is not None else _forecast_cfg().horizon_months)
    if steps < 1:
        raise ValueError("horizon must be a positive number of periods.")

    history = _as_series(aggregate_demand(demand, drug_code, region))
    _require_history(history, 2, drug_code, region)
    result = _forecast_series(history, steps, method)

    log.info(
        "Forecast %d periods with %s (drug=%s, region=%s): mean %.0f units/period",
        steps, result["method"].iloc[0], drug_code or "ALL", region or "ALL",
        float(result["forecast"].mean()),
    )
    return result


# ---------------------------------------------------------------------------
# 4. Backtesting - the evidence that the chosen model is justified
# ---------------------------------------------------------------------------
def _error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Standard demand-planning error set for one method on one hold-out."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual

    # MAPE is undefined at zero demand; those periods are excluded rather than
    # allowed to poison the mean with an infinity.
    nonzero = np.abs(actual) > 0
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any() else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": mape,
        # Signed bias separates a model that is merely noisy from one that is
        # systematically over- or under-planning - very different remedies.
        "bias_units": float(np.mean(error)),
        "bias_pct": float(np.mean(error) / np.mean(actual) * 100.0) if np.mean(actual) else float("nan"),
    }


def backtest_forecast(
    demand: pd.DataFrame | None = None,
    drug_code: str | None = None,
    region: str | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Compare every configured method on a genuine hold-out period.

    The last ``forecasting.train_test_split_months`` months are withheld, each
    candidate is fitted on the remainder only, and the resulting forecasts are
    scored against the withheld actuals. This is the evidence that the model
    used in production was selected rather than assumed.

    Parameters
    ----------
    demand : pandas.DataFrame, optional
        Pre-loaded demand fact table; ``None`` loads it.
    drug_code : str, optional
        Restrict to a single product.
    region : str, optional
        Restrict to a single region.
    horizon : int, optional
        Hold-out length in months. Defaults to
        ``forecasting.train_test_split_months``.

    Returns
    -------
    pandas.DataFrame
        One row per method, ranked best-first by MAPE, with columns ``rank``,
        ``method``, ``mae``, ``rmse``, ``mape_pct``, ``bias_units``,
        ``bias_pct``, ``train_periods``, ``test_periods``.

    Raises
    ------
    ValueError
        If the series is too short to leave a usable training window.
    """
    cfg = _forecast_cfg()
    holdout = int(horizon if horizon is not None else cfg.train_test_split_months)
    series = _as_series(aggregate_demand(demand, drug_code, region))
    # A hold-out is only meaningful if what remains can still train a model.
    _require_history(series, holdout + 3, drug_code, region)

    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    rows: list[dict[str, Any]] = []

    for method in list(cfg.methods):
        prediction = _forecast_series(train, holdout, method)
        metrics = _error_metrics(test.to_numpy(float), prediction["forecast"].to_numpy(float))
        rows.append(
            {
                "method": method,
                # Record what actually ran: a short series may have degraded a
                # candidate to its fallback, and hiding that would be dishonest.
                "fitted_as": prediction["method"].iloc[0],
                **metrics,
                "train_periods": int(len(train)),
                "test_periods": int(len(test)),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values("mape_pct", kind="stable")
        .reset_index(drop=True)
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))

    log.info(
        "Backtest on %d train / %d test periods: best=%s (MAPE %.2f%%), worst=%s (MAPE %.2f%%)",
        len(train), len(test), results.loc[0, "method"], results.loc[0, "mape_pct"],
        results.iloc[-1]["method"], results.iloc[-1]["mape_pct"],
    )
    return results


def forecast_accuracy_summary(
    demand: pd.DataFrame | None = None,
    drug_code: str | None = None,
    region: str | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Reduce the backtest to a decision: which model, how good, and so what.

    Parameters
    ----------
    demand : pandas.DataFrame, optional
        Pre-loaded demand fact table; ``None`` loads it.
    drug_code : str, optional
        Restrict to a single product.
    region : str, optional
        Restrict to a single region.
    horizon : int, optional
        Hold-out length in months; defaults to the configured split.

    Returns
    -------
    dict
        ``best_method``, ``best_mape_pct``, ``best_mae``, ``best_rmse``,
        ``bias_units``, ``bias_direction``, ``runner_up``, ``improvement_vs_
        baseline_pct``, ``accuracy_grade`` and a plain-English ``verdict``.

    Notes
    -----
    The accuracy bands are derived from ``forecasting.confidence_level`` rather
    than hard-coded: the residual tolerance ``1 - confidence_level`` (5% at the
    configured 95%) is treated as the "excellent" bar, with 2x and 4x that
    tolerance marking "good" and "fair". Tightening the confidence level in the
    config therefore tightens the verdict consistently.
    """
    results = backtest_forecast(demand, drug_code, region, horizon)
    best = results.iloc[0]
    baseline = results[results["method"] == "moving_average"]

    tolerance_pct = (1.0 - float(_forecast_cfg().confidence_level)) * 100.0
    mape = float(best["mape_pct"])
    if mape <= tolerance_pct:
        grade, quality = "A", "excellent"
    elif mape <= 2 * tolerance_pct:
        grade, quality = "B", "good"
    elif mape <= 4 * tolerance_pct:
        grade, quality = "C", "fair"
    else:
        grade, quality = "D", "weak"

    bias_units = float(best["bias_units"])
    if bias_units > 0:
        direction = "over-forecasts"
        bias_note = "risking excess stock that ages against a fixed expiry date"
    elif bias_units < 0:
        direction = "under-forecasts"
        bias_note = "risking back-orders and service-level penalties"
    else:  # pragma: no cover - exact zero is effectively unreachable
        direction = "is unbiased"
        bias_note = "with no systematic planning skew"

    improvement = float("nan")
    if not baseline.empty and np.isfinite(baseline["mape_pct"].iloc[0]) and baseline["mape_pct"].iloc[0] > 0:
        improvement = float(
            (baseline["mape_pct"].iloc[0] - mape) / baseline["mape_pct"].iloc[0] * 100.0
        )

    verdict = (
        f"'{best['method']}' is the most accurate of the {len(results)} candidates on a "
        f"{int(best['test_periods'])}-month hold-out, with a MAPE of {mape:.2f}% "
        f"({quality} accuracy for monthly pharmaceutical demand planning). It "
        f"{direction} by {abs(bias_units):,.0f} units per month on average, {bias_note}. "
    )
    if np.isfinite(improvement):
        verdict += (
            f"That is a {improvement:.1f}% reduction in forecast error versus the "
            "trailing-average baseline."
            if improvement >= 0
            else "It does not beat the trailing-average baseline, so the simpler "
                 "benchmark should be preferred until more history is available."
        )

    summary: dict[str, Any] = {
        "best_method": str(best["method"]),
        "best_mape_pct": round(mape, 3),
        "best_mae": round(float(best["mae"]), 1),
        "best_rmse": round(float(best["rmse"]), 1),
        "bias_units": round(bias_units, 1),
        "bias_pct": round(float(best["bias_pct"]), 3),
        "bias_direction": direction,
        "runner_up": str(results.iloc[1]["method"]) if len(results) > 1 else None,
        "improvement_vs_baseline_pct": (
            round(improvement, 2) if np.isfinite(improvement) else None
        ),
        "accuracy_grade": grade,
        "train_periods": int(best["train_periods"]),
        "test_periods": int(best["test_periods"]),
        "verdict": verdict,
    }
    log.info("Accuracy summary: %s (grade %s, MAPE %.2f%%)", summary["best_method"], grade, mape)
    return summary


# ---------------------------------------------------------------------------
# 5. Headline KPIs
# ---------------------------------------------------------------------------
def demand_kpis(demand: pd.DataFrame | None = None) -> dict[str, Any]:
    """Headline demand metrics for the executive KPI strip.

    Parameters
    ----------
    demand : pandas.DataFrame, optional
        Pre-loaded (and optionally pre-filtered) demand fact table. ``None``
        loads the full table.

    Returns
    -------
    dict
        ``total_demand_units``, ``avg_monthly_demand``, ``yoy_growth_pct``,
        ``avg_fill_rate_pct``, ``total_backorders``, ``peak_month`` plus the
        supporting ``peak_month_units`` and ``months_covered``.

    Notes
    -----
    Year-on-year growth compares the most recent twelve months with the twelve
    before them. A full-year window is used deliberately: any shorter comparison
    would confound genuine growth with the winter seasonality of the portfolio.
    """
    frame = _load(demand)
    if frame.empty:
        log.warning("demand_kpis: empty demand frame.")
        return {
            "total_demand_units": 0, "avg_monthly_demand": 0.0, "yoy_growth_pct": None,
            "avg_fill_rate_pct": None, "total_backorders": 0, "peak_month": None,
            "peak_month_units": 0, "months_covered": 0,
        }

    monthly = aggregate_demand(frame)
    total = float(frame["units_demanded"].sum())
    period = int(_forecast_cfg().seasonal_periods)

    yoy: float | None = None
    if len(monthly) >= 2 * period:
        recent = float(monthly[DEMAND_COLUMN].iloc[-period:].sum())
        prior = float(monthly[DEMAND_COLUMN].iloc[-2 * period:-period].sum())
        yoy = round((recent - prior) / prior * 100.0, 2) if prior else None
    else:
        log.warning(
            "demand_kpis: %d months of history is below the %d needed for a clean "
            "year-on-year comparison.", len(monthly), 2 * period,
        )

    peak_row = monthly.loc[monthly[DEMAND_COLUMN].idxmax()]
    # Fill rate is volume-weighted: a small region missing its plan must not
    # count as heavily as a large one when reporting portfolio service level.
    fill_rate = (
        float(frame["units_fulfilled"].sum() / frame["units_demanded"].sum() * 100.0)
        if frame["units_demanded"].sum() else None
    )

    kpis: dict[str, Any] = {
        "total_demand_units": int(round(total)),
        "avg_monthly_demand": round(float(monthly[DEMAND_COLUMN].mean()), 1),
        "yoy_growth_pct": yoy,
        "avg_fill_rate_pct": round(fill_rate, 2) if fill_rate is not None else None,
        "total_backorders": int(round(float(frame["units_backordered"].sum()))),
        "peak_month": pd.Timestamp(peak_row["date"]).strftime("%Y-%m"),
        "peak_month_units": int(round(float(peak_row[DEMAND_COLUMN]))),
        "months_covered": int(len(monthly)),
    }
    log.info(
        "Demand KPIs: %.0f units over %d months | YoY %s | fill rate %s%%",
        total, kpis["months_covered"], kpis["yoy_growth_pct"], kpis["avg_fill_rate_pct"],
    )
    return kpis


__all__ = [
    "DEMAND_COLUMN",
    "aggregate_demand",
    "decompose_demand",
    "forecast_demand",
    "backtest_forecast",
    "forecast_accuracy_summary",
    "demand_kpis",
]
