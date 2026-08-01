"""
Drug stability and cold-chain integrity analytics.

Business purpose
----------------
A pharmaceutical product is only a product while it is *within specification*.
Regulators (ICH Q1A(R2), "Stability Testing of New Drug Substances and
Products") require the manufacturer to demonstrate, through accelerated and
long-term storage studies, that the active ingredient stays within its potency
limits across the labelled shelf life under the labelled storage condition -
2-8 degC for cold-chain biologics, controlled room temperature for most solid
oral forms. The label claim, the expiry date and the storage statement on the
carton are all consequences of that stability data package.

Real distribution networks do not run at study conditions. Warehouses run warm,
monsoon-season humidity pushes moisture into hygroscopic tablets, batches sit in
a distribution centre for months, and a refrigerated container loses power for
six hours somewhere between the port and the depot. Every one of those events
consumes shelf life that the stability package assumed would still be there when
the product reached the patient.

This module quantifies that consumption from the batch fact table:

* **Univariate effect profiles** (temperature, humidity, storage duration) -
  the same shape of evidence an accelerated stability study produces, but from
  observed distribution data rather than a stability chamber.
* **A fitted degradation model** - an ordinary-least-squares surrogate for the
  Arrhenius-style kinetics that drive potency loss, solved for the duration at
  which a batch stored at reference conditions crosses the out-of-specification
  limit. That crossing point is, in effect, an empirically re-derived shelf life
  to compare with the labelled one.
* **Cold-chain excursion impact** - the headline insight, tested for statistical
  significance rather than asserted, and priced in dollars at risk.

Scientific consistency
----------------------
The underlying potency of each batch was produced by a simplified Arrhenius
relationship: loss scales with exposure time, accelerates with temperature above
the product's labelled reference (``stability.reference_temp_cold`` for
cold-chain items, ``stability.reference_temp_ambient`` otherwise) and is
amplified by moisture above ``stability.humidity_threshold``. Every constant in
this module - the reference conditions, the specification limit
(``stability.potency_spec_min``) and the warning limit
(``stability.potency_warning``) - is read from that same configuration block, so
the analytics and the degradation physics can never drift apart.

Design notes
------------
Pure analytics: pandas / numpy / scipy / statsmodels only, no Streamlit and no
plotting. Every public function takes an optional pre-loaded batch frame first
so the dashboard can push filtered data down rather than reload it.

Example
-------
>>> from src.analytics import stability as st
>>> st.excursion_impact()["p_value"] < 0.05
>>> st.shelf_life_model()["estimated_shelf_life_days"]
"""

from __future__ import annotations

from typing import Any, Callable, Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)

#: Regressors of the degradation surrogate model, in physical order.
_MODEL_FEATURES: Final[tuple[str, ...]] = (
    "storage_temp_c",
    "storage_humidity_pct",
    "storage_duration_days",
    "is_cold_chain",
)

#: Storage-condition columns exposed as binned effect profiles.
_EFFECT_COLUMNS: Final[dict[str, str]] = {
    "storage_temp_c": "Storage temperature (degC)",
    "storage_humidity_pct": "Storage humidity (%RH)",
    "storage_duration_days": "Storage duration (days)",
}

#: Cold-chain cohort labels. Ambient and refrigerated products have different
#: labelled reference conditions, so their profiles are never pooled.
_COLD_LABEL: Final[str] = "Cold chain (2-8C)"
_AMBIENT_LABEL: Final[str] = "Ambient (CRT)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _stability_cfg() -> Any:
    """Return the ``stability`` configuration block."""
    return get_config().stability


def _load(batches: pd.DataFrame | None) -> pd.DataFrame:
    """Return the caller's frame, or load the batch fact table if none given."""
    return loader.load_batches() if batches is None else batches.copy()


def _prepare(batches: pd.DataFrame | None) -> pd.DataFrame:
    """Load and enrich the batch frame with the derived stability flags.

    The flags are computed once here so that every downstream function scores a
    batch against exactly the same specification limits.
    """
    cfg = _stability_cfg()
    frame = _load(batches)
    if frame.empty:
        return frame

    frame["is_cold_chain"] = frame["is_cold_chain"].astype(int)
    # Out of specification: below the lower potency limit from the stability
    # package - a batch here cannot be released to a patient.
    frame["out_of_spec"] = (frame["potency_pct"] < float(cfg.potency_spec_min)).astype(int)
    # Warning band: still releasable, but with too little potency headroom to
    # survive further distribution - the tier QA wants to see early.
    frame["below_warning"] = (frame["potency_pct"] < float(cfg.potency_warning)).astype(int)
    frame["is_high_risk"] = (frame["batch_risk_label"] == "High").astype(int)
    frame["cold_chain_cohort"] = np.where(
        frame["is_cold_chain"] == 1, _COLD_LABEL, _AMBIENT_LABEL
    )
    return frame


def _net_write_off(value_usd: float | np.ndarray) -> float | np.ndarray:
    """Convert gross lost value into a net write-off after salvage recovery.

    Expired or out-of-specification stock is not worth zero: a small fraction is
    recovered through returns handling and reclaim, per
    ``economics.expiry_write_off_recovery``.
    """
    recovery = float(get_config().economics.expiry_write_off_recovery)
    return value_usd * (1.0 - recovery)


def _value_at_risk(frame: pd.DataFrame) -> float:
    """Net USD exposure of the batches in ``frame`` that failed specification.

    Only out-of-specification or High-risk batches are counted: value lost to
    ordinary process yield on a good batch is a manufacturing cost, not a
    stability risk, and conflating the two would inflate the headline number.
    """
    if frame.empty:
        return 0.0
    at_risk = frame[(frame["out_of_spec"] == 1) | (frame["is_high_risk"] == 1)]
    return float(round(_net_write_off(at_risk["value_lost_usd"].sum()), 2))


def _binned_effect(frame: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
    """Profile potency against a binned storage condition, split by cohort.

    Equal-width bins are used rather than quantiles because the business
    question is "what happens above 30 degC", which is a physical threshold,
    not a population percentile.
    """
    cfg = _stability_cfg()
    if frame.empty:
        return pd.DataFrame(
            columns=["driver", "bin", "bin_left", "bin_right", "bin_mid", "cohort",
                     "batches", "avg_potency_pct", "out_of_spec_pct",
                     "below_warning_pct", "avg_value_lost_usd"]
        )

    intervals = pd.cut(frame[column], bins=int(bins))
    working = frame.assign(_bin=intervals)

    grouped = (
        working.groupby(["_bin", "cold_chain_cohort"], observed=False)
        .agg(
            batches=("batch_id", "count"),
            avg_potency_pct=("potency_pct", "mean"),
            out_of_spec_rate=("out_of_spec", "mean"),
            below_warning_rate=("below_warning", "mean"),
            avg_value_lost_usd=("value_lost_usd", "mean"),
        )
        .reset_index()
    )

    grouped["driver"] = _EFFECT_COLUMNS.get(column, column)
    grouped["bin_left"] = grouped["_bin"].map(lambda i: float(i.left)).astype(float)
    grouped["bin_right"] = grouped["_bin"].map(lambda i: float(i.right)).astype(float)
    grouped["bin_mid"] = (grouped["bin_left"] + grouped["bin_right"]) / 2.0
    grouped["bin"] = grouped["_bin"].astype(str)
    grouped = grouped.rename(columns={"cold_chain_cohort": "cohort"})
    grouped["out_of_spec_pct"] = (grouped["out_of_spec_rate"] * 100.0).round(2)
    grouped["below_warning_pct"] = (grouped["below_warning_rate"] * 100.0).round(2)
    grouped["avg_potency_pct"] = grouped["avg_potency_pct"].round(3)
    grouped["avg_value_lost_usd"] = grouped["avg_value_lost_usd"].round(2)

    # Empty cohort/bin cells carry no information and would render as gaps.
    grouped = grouped[grouped["batches"] > 0]

    ordered = grouped[
        ["driver", "bin", "bin_left", "bin_right", "bin_mid", "cohort", "batches",
         "avg_potency_pct", "out_of_spec_pct", "below_warning_pct", "avg_value_lost_usd"]
    ].sort_values(["cohort", "bin_left"]).reset_index(drop=True)

    log.info(
        "%s effect profile: %d populated bins across %d cohorts (spec limit %.1f%%)",
        column, len(ordered), ordered["cohort"].nunique(), float(cfg.potency_spec_min),
    )
    return ordered


# ---------------------------------------------------------------------------
# 1-3. Storage-condition effect profiles
# ---------------------------------------------------------------------------
def temperature_effect(batches: pd.DataFrame | None = None, bins: int = 8) -> pd.DataFrame:
    """Profile potency and out-of-specification rate against storage temperature.

    This is the observational analogue of an accelerated stability study: the
    Arrhenius relationship predicts that degradation rate rises with every
    degree above the labelled storage condition, and the out-of-specification
    rate per temperature band is where that shows up commercially.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded batch fact table; ``None`` loads it.
    bins : int, default 8
        Number of equal-width temperature bands.

    Returns
    -------
    pandas.DataFrame
        One row per populated (band, cohort) pair: ``driver``, ``bin``,
        ``bin_left``, ``bin_right``, ``bin_mid``, ``cohort``, ``batches``,
        ``avg_potency_pct``, ``out_of_spec_pct``, ``below_warning_pct``,
        ``avg_value_lost_usd``.
    """
    return _binned_effect(_prepare(batches), "storage_temp_c", bins)


def humidity_effect(batches: pd.DataFrame | None = None, bins: int = 8) -> pd.DataFrame:
    """Profile potency and out-of-specification rate against storage humidity.

    Moisture uptake above ``stability.humidity_threshold`` %RH accelerates
    hydrolytic degradation in hygroscopic solid forms, which is why the
    threshold - not an arbitrary cut - is the number this profile is read
    against.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded batch fact table; ``None`` loads it.
    bins : int, default 8
        Number of equal-width relative-humidity bands.

    Returns
    -------
    pandas.DataFrame
        Same columns as :func:`temperature_effect`, binned on
        ``storage_humidity_pct``.
    """
    return _binned_effect(_prepare(batches), "storage_humidity_pct", bins)


def duration_effect(batches: pd.DataFrame | None = None, bins: int = 8) -> pd.DataFrame:
    """Profile potency and out-of-specification rate against storage duration.

    Time is the one degradation driver that cannot be engineered away, only
    managed: every extra week a batch spends in a distribution centre is shelf
    life spent before the patient ever sees the product.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded batch fact table; ``None`` loads it.
    bins : int, default 8
        Number of equal-width duration bands, in days.

    Returns
    -------
    pandas.DataFrame
        Same columns as :func:`temperature_effect`, binned on
        ``storage_duration_days``.
    """
    return _binned_effect(_prepare(batches), "storage_duration_days", bins)


# ---------------------------------------------------------------------------
# 4. Fitted degradation model
# ---------------------------------------------------------------------------
def shelf_life_model(batches: pd.DataFrame | None = None) -> dict[str, Any]:
    """Fit an OLS degradation surrogate and solve it for an empirical shelf life.

    Potency is regressed on the three storage stressors plus a cold-chain
    indicator. The true kinetics are multiplicative (time x temperature x
    moisture), so a linear model is deliberately a *local* approximation - but
    it is the form a stability report is written in ("potency falls by X% per
    month of storage, Y% per degree above label"), it is auditable, and over the
    observed operating envelope it captures the dominant behaviour.

    Solving the fitted surface for the duration at which potency crosses
    ``stability.potency_spec_min``, holding temperature and humidity at the
    labelled reference conditions, yields an empirically re-derived shelf life
    to compare against the labelled one.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded batch fact table; ``None`` loads it.

    Returns
    -------
    dict
        ``coefficients`` (per-regressor slope, including ``const``),
        ``r_squared``, ``adj_r_squared``, ``p_values``, ``std_errors``,
        ``conf_int``, ``n_observations``, ``predict`` (a callable taking
        ``temp, humidity, duration, is_cold_chain`` and returning predicted
        potency %), ``estimated_shelf_life_days`` (ambient reference
        conditions), ``estimated_shelf_life_days_by_cohort`` and
        ``reference_conditions``.

    Notes
    -----
    A negative duration coefficient is required for the shelf-life solve to be
    meaningful; if the fit ever produces a non-negative slope the estimate is
    returned as ``None`` rather than as a nonsensical negative or infinite
    number of days.
    """
    cfg = _stability_cfg()
    frame = _prepare(batches)
    if frame.empty:
        raise ValueError("Cannot fit a stability model on an empty batch frame.")

    exog = sm.add_constant(
        frame[list(_MODEL_FEATURES)].astype(float), has_constant="add"
    )
    model = sm.OLS(frame["potency_pct"].astype(float), exog).fit()

    coefficients = {str(k): float(v) for k, v in model.params.items()}
    p_values = {str(k): float(v) for k, v in model.pvalues.items()}
    std_errors = {str(k): float(v) for k, v in model.bse.items()}
    ci = model.conf_int()
    conf_int = {str(idx): (float(row.iloc[0]), float(row.iloc[1])) for idx, row in ci.iterrows()}

    def predict(
        temp: float | np.ndarray,
        humidity: float | np.ndarray,
        duration: float | np.ndarray,
        is_cold_chain: int | bool | np.ndarray = 0,
    ) -> float | np.ndarray:
        """Predict residual potency (%) for given storage conditions.

        Parameters
        ----------
        temp : float or array-like
            Storage temperature in degrees Celsius.
        humidity : float or array-like
            Relative humidity, percent.
        duration : float or array-like
            Storage duration in days.
        is_cold_chain : int, bool or array-like, default 0
            1 for a refrigerated (2-8 degC) product, 0 for ambient.

        Returns
        -------
        float or numpy.ndarray
            Predicted potency as a percentage of label claim.
        """
        temp_a, hum_a, dur_a, cold_a = np.broadcast_arrays(
            np.asarray(temp, dtype=float),
            np.asarray(humidity, dtype=float),
            np.asarray(duration, dtype=float),
            np.asarray(is_cold_chain, dtype=float),
        )
        predicted = (
            coefficients["const"]
            + coefficients["storage_temp_c"] * temp_a
            + coefficients["storage_humidity_pct"] * hum_a
            + coefficients["storage_duration_days"] * dur_a
            + coefficients["is_cold_chain"] * cold_a
        )
        return float(predicted) if predicted.ndim == 0 else predicted

    def _solve_shelf_life(temp: float, humidity: float, cold: int) -> float | None:
        """Days until the fitted surface crosses the specification limit."""
        slope = coefficients["storage_duration_days"]
        if slope >= 0:  # pragma: no cover - degenerate fit guard
            return None
        potency_at_zero = predict(temp, humidity, 0.0, cold)
        days = (float(potency_at_zero) - float(cfg.potency_spec_min)) / -slope
        return round(float(days), 1) if np.isfinite(days) and days > 0 else None

    # Reference conditions mirror the degradation model the data was built on:
    # the labelled storage temperature, and humidity at the threshold below
    # which moisture contributes no extra loss.
    reference_humidity = float(cfg.humidity_threshold)
    ambient_days = _solve_shelf_life(float(cfg.reference_temp_ambient), reference_humidity, 0)
    cold_days = _solve_shelf_life(float(cfg.reference_temp_cold), reference_humidity, 1)

    result: dict[str, Any] = {
        "coefficients": coefficients,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "p_values": p_values,
        "std_errors": std_errors,
        "conf_int": conf_int,
        "n_observations": int(model.nobs),
        "predict": predict,
        "estimated_shelf_life_days": ambient_days,
        "estimated_shelf_life_days_by_cohort": {
            _AMBIENT_LABEL: ambient_days,
            _COLD_LABEL: cold_days,
        },
        "reference_conditions": {
            "ambient_temp_c": float(cfg.reference_temp_ambient),
            "cold_chain_temp_c": float(cfg.reference_temp_cold),
            "humidity_pct": reference_humidity,
            "potency_spec_min": float(cfg.potency_spec_min),
        },
    }
    log.info(
        "Stability model fitted on %d batches: R2=%.3f | %.3f%% potency lost per day, "
        "%.3f%% per degC | reference shelf life %s days (ambient)",
        result["n_observations"], result["r_squared"],
        -coefficients["storage_duration_days"], -coefficients["storage_temp_c"], ambient_days,
    )
    return result


# ---------------------------------------------------------------------------
# 5. Per-product performance
# ---------------------------------------------------------------------------
def batch_performance(batches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarise stability performance per product.

    This is the view a QA review board works from: which products in the
    portfolio are consuming their shelf life fastest, and where the risk is
    concentrated.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded batch fact table; ``None`` loads it.

    Returns
    -------
    pandas.DataFrame
        One row per ``drug_code`` with ``brand_name``, ``batches``,
        ``avg_potency_pct``, ``out_of_spec_pct``, ``high_risk_pct``,
        ``avg_shelf_life_remaining_days``, ``excursion_rate_pct`` and
        ``value_at_risk_usd``, sorted worst-first by out-of-specification rate.
    """
    frame = _prepare(batches)
    if frame.empty:
        return pd.DataFrame(
            columns=["drug_code", "brand_name", "batches", "avg_potency_pct",
                     "out_of_spec_pct", "high_risk_pct",
                     "avg_shelf_life_remaining_days", "excursion_rate_pct",
                     "value_at_risk_usd"]
        )

    summary = (
        frame.groupby("drug_code", observed=True)
        .agg(
            brand_name=("brand_name", "first"),
            batches=("batch_id", "count"),
            avg_potency_pct=("potency_pct", "mean"),
            out_of_spec_rate=("out_of_spec", "mean"),
            high_risk_rate=("is_high_risk", "mean"),
            # Remaining shelf life measured at delivery: the headroom the
            # dispensing pharmacy actually inherits.
            avg_shelf_life_remaining_days=("days_to_expiry_at_delivery", "mean"),
            excursion_rate=("cold_chain_excursion", "mean"),
        )
        .reset_index()
    )

    exposure = (
        frame.groupby("drug_code", observed=True)
        .apply(_value_at_risk, include_groups=False)
        .rename("value_at_risk_usd")
        .reset_index()
    )
    summary = summary.merge(exposure, on="drug_code", how="left")

    summary["avg_potency_pct"] = summary["avg_potency_pct"].round(3)
    summary["out_of_spec_pct"] = (summary.pop("out_of_spec_rate") * 100.0).round(2)
    summary["high_risk_pct"] = (summary.pop("high_risk_rate") * 100.0).round(2)
    summary["excursion_rate_pct"] = (summary.pop("excursion_rate") * 100.0).round(2)
    summary["avg_shelf_life_remaining_days"] = summary["avg_shelf_life_remaining_days"].round(1)

    ordered = summary[
        ["drug_code", "brand_name", "batches", "avg_potency_pct", "out_of_spec_pct",
         "high_risk_pct", "avg_shelf_life_remaining_days", "excursion_rate_pct",
         "value_at_risk_usd"]
    ].sort_values("out_of_spec_pct", ascending=False).reset_index(drop=True)

    log.info(
        "Batch performance computed for %d products (%d batches); worst out-of-spec "
        "rate %.2f%% (%s)",
        len(ordered), int(ordered["batches"].sum()),
        float(ordered.loc[0, "out_of_spec_pct"]), ordered.loc[0, "drug_code"],
    )
    return ordered


# ---------------------------------------------------------------------------
# 6. Expiry risk
# ---------------------------------------------------------------------------
def expiry_risk_profile(batches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bucket delivered batches by remaining shelf life and price the exposure.

    Wholesalers and hospital pharmacies enforce minimum-remaining-shelf-life
    acceptance rules, so a batch delivered with little headroom is commercially
    at risk long before it chemically expires. The buckets below mirror the
    thresholds those acceptance rules are typically written against, with the
    ``inventory.expiry_risk_days`` horizon from the config marking the boundary
    the platform treats as actionable.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded batch fact table; ``None`` loads it.

    Returns
    -------
    pandas.DataFrame
        One row per bucket - ``Expired``, ``<30d``, ``30-90d``, ``90-180d``,
        ``>180d`` - with ``batches``, ``batch_pct``, ``avg_days_to_expiry``,
        ``value_lost_usd``, ``inventory_value_usd`` (``units_dispensed`` x
        ``unit_cost_usd``), ``net_value_at_risk_usd`` and ``within_risk_horizon``.
    """
    frame = _prepare(batches)
    columns = ["bucket", "batches", "batch_pct", "avg_days_to_expiry", "value_lost_usd",
               "inventory_value_usd", "net_value_at_risk_usd", "within_risk_horizon"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    horizon_days = int(get_config().inventory.expiry_risk_days)
    edges = [-np.inf, 0, 30, 90, 180, np.inf]
    labels = ["Expired", "<30d", "30-90d", "90-180d", ">180d"]

    working = frame.assign(
        bucket=pd.cut(
            frame["days_to_expiry_at_delivery"], bins=edges, labels=labels, right=False
        ),
        inventory_value_usd=frame["units_dispensed"] * frame["unit_cost_usd"],
    )

    # observed=False keeps empty buckets in the table: "zero expired batches"
    # is a result a reviewer needs to see stated, not inferred from an absence.
    profile = (
        working.groupby("bucket", observed=False)
        .agg(
            batches=("batch_id", "count"),
            avg_days_to_expiry=("days_to_expiry_at_delivery", "mean"),
            value_lost_usd=("value_lost_usd", "sum"),
            inventory_value_usd=("inventory_value_usd", "sum"),
        )
        .reset_index()
    )

    total = int(profile["batches"].sum())
    profile["batch_pct"] = (profile["batches"] / total * 100.0).round(2) if total else 0.0
    profile["avg_days_to_expiry"] = profile["avg_days_to_expiry"].round(1)
    profile["value_lost_usd"] = profile["value_lost_usd"].round(2)
    profile["inventory_value_usd"] = profile["inventory_value_usd"].round(2)
    # Exposure nets off the salvage recovered on written-off stock.
    profile["net_value_at_risk_usd"] = _net_write_off(
        profile["value_lost_usd"] + profile["inventory_value_usd"]
    ).round(2)
    # Flag the buckets falling inside the configured actionable horizon.
    upper_edge = dict(zip(labels, [0, 30, 90, 180, np.inf]))
    profile["within_risk_horizon"] = profile["bucket"].map(
        lambda b: bool(upper_edge[str(b)] <= horizon_days)
    )
    profile["bucket"] = profile["bucket"].astype(str)

    at_risk = int(profile.loc[profile["within_risk_horizon"], "batches"].sum())
    log.info(
        "Expiry risk profile: %d batches, %d (%.1f%%) inside the %d-day risk horizon",
        total, at_risk, (at_risk / total * 100.0) if total else 0.0, horizon_days,
    )
    return profile[columns]


# ---------------------------------------------------------------------------
# 7. Cold-chain excursion impact - headline insight
# ---------------------------------------------------------------------------
def excursion_impact(batches: pd.DataFrame | None = None) -> dict[str, Any]:
    """Quantify and significance-test the potency cost of a cold-chain excursion.

    A temperature excursion is the single most consequential avoidable event in
    pharmaceutical distribution: the product spends time outside its labelled
    storage window, degradation accelerates, and the batch may fall out of
    specification without any visible sign. This function answers the three
    questions an investment case needs: how much potency is lost, is the
    difference real (an independent two-sample Welch t-test, which does not
    assume the two groups share a variance), and what is it worth in dollars.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded batch fact table; ``None`` loads it.

    Returns
    -------
    dict
        Group sizes and mean potency with and without an excursion, the
        ``potency_difference_pct`` and its confidence interval, ``t_statistic``,
        ``p_value``, ``significant``, ``cohens_d``, the out-of-specification
        rate in each group and the resulting ``oos_rate_lift_pp``,
        ``value_at_risk_usd`` (net exposure of the affected batches),
        ``attributable_value_at_risk_usd`` (the share attributable to the
        excursion itself) and a plain-English ``verdict``.
    """
    cfg = _stability_cfg()
    alpha = float(get_config().ab_testing.alpha)
    confidence = 1.0 - alpha
    frame = _prepare(batches)
    if frame.empty:
        raise ValueError("Cannot assess excursion impact on an empty batch frame.")

    with_exc = frame[frame["cold_chain_excursion"] == 1]
    without_exc = frame[frame["cold_chain_excursion"] == 0]
    n_with, n_without = int(len(with_exc)), int(len(without_exc))
    if n_with < 2 or n_without < 2:
        raise ValueError(
            f"Need at least two batches in each group to test the excursion effect "
            f"(with={n_with}, without={n_without})."
        )

    potency_with = with_exc["potency_pct"].astype(float)
    potency_without = without_exc["potency_pct"].astype(float)
    mean_with, mean_without = float(potency_with.mean()), float(potency_without.mean())
    difference = mean_with - mean_without

    # Welch's t-test: the excursion group is small and its variance is inflated
    # by the excursion itself, so equal variances must not be assumed.
    t_stat, p_value = stats.ttest_ind(potency_with, potency_without, equal_var=False)

    var_with, var_without = float(potency_with.var(ddof=1)), float(potency_without.var(ddof=1))
    se = float(np.sqrt(var_with / n_with + var_without / n_without))
    # Welch-Satterthwaite degrees of freedom for the interval around the effect.
    dof = (var_with / n_with + var_without / n_without) ** 2 / (
        (var_with / n_with) ** 2 / (n_with - 1) + (var_without / n_without) ** 2 / (n_without - 1)
    )
    margin = float(stats.t.ppf(0.5 + confidence / 2.0, dof)) * se
    pooled_sd = float(np.sqrt(((n_with - 1) * var_with + (n_without - 1) * var_without)
                              / (n_with + n_without - 2)))
    cohens_d = difference / pooled_sd if pooled_sd else float("nan")

    oos_with = float(with_exc["out_of_spec"].mean() * 100.0)
    oos_without = float(without_exc["out_of_spec"].mean() * 100.0)
    oos_lift_pp = oos_with - oos_without

    value_at_risk = _value_at_risk(with_exc)
    # Attributable exposure: the excess out-of-specification rate applied to the
    # excursion population, valued at the average batch value. This isolates the
    # loss caused by the excursion from the loss that batch would have carried
    # anyway - the number a cold-chain monitoring business case can defend.
    avg_batch_value = float(frame["batch_value_usd"].mean()) if "batch_value_usd" in frame else float(
        (frame["units_dispensed"] * frame["unit_cost_usd"]).mean()
    )
    attributable = float(_net_write_off(max(oos_lift_pp, 0.0) / 100.0 * n_with * avg_batch_value))

    significant = bool(p_value < alpha)
    verdict = (
        f"Batches that experienced a cold-chain excursion average "
        f"{mean_with:.2f}% potency versus {mean_without:.2f}% for batches that did not, "
        f"a difference of {difference:+.2f} percentage points "
        f"({int(confidence * 100)}% CI {difference - margin:+.2f} to {difference + margin:+.2f}). "
        f"The difference is {'statistically significant' if significant else 'not statistically significant'} "
        f"(Welch t = {float(t_stat):.2f}, p = {float(p_value):.2e}, alpha = {alpha}). "
        f"Out-of-specification incidence rises from {oos_without:.1f}% to {oos_with:.1f}% "
        f"({oos_lift_pp:+.1f} pp) against a {cfg.potency_spec_min}% specification limit, "
        f"placing an estimated ${attributable:,.0f} of product value at risk that would "
        f"not otherwise be exposed."
    )

    result: dict[str, Any] = {
        "n_with_excursion": n_with,
        "n_without_excursion": n_without,
        "excursion_rate_pct": round(float(frame["cold_chain_excursion"].mean() * 100.0), 2),
        "mean_potency_with_excursion": round(mean_with, 3),
        "mean_potency_without_excursion": round(mean_without, 3),
        "potency_difference_pct": round(difference, 3),
        "difference_ci_low": round(difference - margin, 3),
        "difference_ci_high": round(difference + margin, 3),
        "confidence_level": confidence,
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": significant,
        "cohens_d": round(float(cohens_d), 4),
        "out_of_spec_pct_with_excursion": round(oos_with, 2),
        "out_of_spec_pct_without_excursion": round(oos_without, 2),
        "oos_rate_lift_pp": round(oos_lift_pp, 2),
        "value_at_risk_usd": value_at_risk,
        "attributable_value_at_risk_usd": round(attributable, 2),
        "verdict": verdict,
    }
    log.info(
        "Excursion impact: %.2f%% vs %.2f%% potency (diff %+.2f pp, p=%.3g, significant=%s), "
        "$%.0f attributable value at risk",
        mean_with, mean_without, difference, float(p_value), significant, attributable,
    )
    return result


# ---------------------------------------------------------------------------
# 8. Headline KPIs
# ---------------------------------------------------------------------------
def stability_kpis(batches: pd.DataFrame | None = None) -> dict[str, Any]:
    """Headline stability metrics for the executive KPI strip.

    Parameters
    ----------
    batches : pandas.DataFrame, optional
        Pre-loaded (and optionally pre-filtered) batch fact table; ``None``
        loads the full table.

    Returns
    -------
    dict
        ``avg_potency_pct``, ``out_of_spec_pct``, ``high_risk_batch_pct``,
        ``excursion_rate_pct``, ``value_at_risk_usd`` plus the supporting
        ``batches``, ``below_warning_pct``, ``qa_pass_rate_pct``,
        ``avg_days_to_expiry_at_delivery`` and ``potency_spec_min``.
    """
    cfg = _stability_cfg()
    frame = _prepare(batches)
    if frame.empty:
        log.warning("stability_kpis: empty batch frame.")
        return {
            "batches": 0, "avg_potency_pct": None, "out_of_spec_pct": None,
            "high_risk_batch_pct": None, "excursion_rate_pct": None,
            "value_at_risk_usd": 0.0, "below_warning_pct": None,
            "qa_pass_rate_pct": None, "avg_days_to_expiry_at_delivery": None,
            "potency_spec_min": float(cfg.potency_spec_min),
        }

    kpis: dict[str, Any] = {
        "batches": int(len(frame)),
        "avg_potency_pct": round(float(frame["potency_pct"].mean()), 2),
        "out_of_spec_pct": round(float(frame["out_of_spec"].mean() * 100.0), 2),
        "high_risk_batch_pct": round(float(frame["is_high_risk"].mean() * 100.0), 2),
        "excursion_rate_pct": round(float(frame["cold_chain_excursion"].mean() * 100.0), 2),
        "value_at_risk_usd": _value_at_risk(frame),
        "below_warning_pct": round(float(frame["below_warning"].mean() * 100.0), 2),
        "qa_pass_rate_pct": round(float(frame["qa_pass"].mean() * 100.0), 2),
        "avg_days_to_expiry_at_delivery": round(
            float(frame["days_to_expiry_at_delivery"].mean()), 1
        ),
        "potency_spec_min": float(cfg.potency_spec_min),
    }
    log.info(
        "Stability KPIs on %d batches: potency %.2f%% | out of spec %.2f%% | "
        "high risk %.2f%% | $%.0f at risk",
        kpis["batches"], kpis["avg_potency_pct"], kpis["out_of_spec_pct"],
        kpis["high_risk_batch_pct"], kpis["value_at_risk_usd"],
    )
    return kpis


__all__ = [
    "temperature_effect",
    "humidity_effect",
    "duration_effect",
    "shelf_life_model",
    "batch_performance",
    "expiry_risk_profile",
    "excursion_impact",
    "stability_kpis",
]
