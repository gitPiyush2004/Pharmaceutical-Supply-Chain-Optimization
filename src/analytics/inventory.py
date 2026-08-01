"""
Inventory analytics for the pharmaceutical distribution network.

Business purpose
----------------
Pharmaceutical inventory is expensive, perishable and regulated. Every unit
sitting in a distribution centre ties up working capital, accrues holding cost
and marches towards an expiry date after which it must be destroyed rather than
sold. Holding too little is equally costly: a stock-out of a chronic-therapy
product means lost revenue, contractual service-level penalties and, in the
worst case, patients who cannot fill a prescription.

This module turns the monthly warehouse-by-SKU snapshots produced by
:func:`src.data.loader.load_inventory` into the seven views a supply chain
analyst actually works from:

===========================  ==================================================
Function                     Question it answers
===========================  ==================================================
:func:`inventory_turnover`   How fast is each SKU-warehouse position selling
                             through, versus the corporate turns benchmark?
:func:`abc_analysis`         Which handful of SKUs carry most of the value and
                             therefore deserve tight planning attention?
:func:`stockout_risk`        Which positions will run dry before the next
                             replenishment lands, and what does that cost?
:func:`overstock_risk`       Where is working capital trapped in excess cover?
:func:`expiry_risk`          Which stock is about to be written off?
:func:`warehouse_utilisation` Which sites are running out of physical space?
:func:`inventory_kpis`       The headline scalars for the executive summary.
===========================  ==================================================

How to read the output
----------------------
Every risk table is sorted worst-first, so the top rows are the ones to work
today. Risk tables scan **all** snapshot months supplied to them: pass the
latest month only (``inv=inv[inv.snapshot_month == inv.snapshot_month.max()]``)
for a purely operational "what is wrong right now" view, or the full history to
see how often a position has been in trouble. :func:`warehouse_utilisation` is
the exception - physical capacity is a point-in-time question, so it always
resolves to the latest snapshot month present in the data.

All thresholds (turnover benchmark, ABC cut-offs, stock-out and overstock
triggers, expiry window) and all financial rates (holding cost, stock-out
penalty, salvage recovery) come from ``config/config.yaml``. Re-tuning the
business rules never requires touching this file.

Example
-------
>>> from src.analytics.inventory import inventory_kpis, stockout_risk
>>> kpis = inventory_kpis()
>>> worst = stockout_risk().head(10)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)

# --- Calendar constants ----------------------------------------------------
# These are units of time, not tunable business rules, so they stay in code.
_MONTHS_PER_YEAR: int = 12
_DAYS_PER_YEAR: float = 365.0
_PCT: float = 100.0

# Ordered worst-to-best. The stock-out band edges are derived from the single
# configured trigger (``inventory.stockout_risk_threshold``) by splitting it
# evenly across these levels, so there is exactly one number to tune.
_RISK_LEVELS: tuple[str, ...] = ("High", "Medium", "Low")

# Physical warehouse occupancy bands. Below the healthy floor a site is paying
# rent on empty racking; above the constrained ceiling there is no room to
# receive an inbound container without cross-docking.
_UTILISATION_HEALTHY_FLOOR_PCT: float = 50.0
_UTILISATION_CONSTRAINED_PCT: float = 85.0
_UTILISATION_STATUS: tuple[str, str, str] = ("Under-utilised", "Healthy", "Constrained")

# Grain of an inventory "position": one SKU held at one site.
_POSITION_KEYS: list[str] = ["drug_code", "brand_name", "warehouse_id",
                            "warehouse_name", "region"]

__all__ = [
    "inventory_turnover",
    "abc_analysis",
    "stockout_risk",
    "overstock_risk",
    "expiry_risk",
    "warehouse_utilisation",
    "inventory_kpis",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve(inv: pd.DataFrame | None) -> pd.DataFrame:
    """Return the frame to analyse, loading from disk when none was supplied.

    Parameters
    ----------
    inv
        Pre-loaded (and possibly filtered) inventory snapshot frame, or ``None``
        to read the full table through :mod:`src.data.loader`.

    Returns
    -------
    pandas.DataFrame
        A defensive copy, so callers can never mutate the loader's cache.
    """
    frame = loader.load_inventory() if inv is None else inv
    return frame.copy()


def _safe_divide(numerator: Any, denominator: Any, fill: float = np.nan) -> np.ndarray:
    """Element-wise division that never raises or returns an infinity.

    Zero denominators (a warehouse with no demand, a SKU never issued) are the
    normal case in a sparse network rather than an error, so they resolve to
    ``fill`` instead of blowing up the whole report.

    Parameters
    ----------
    numerator, denominator
        Array-likes of equal length.
    fill
        Value substituted wherever the quotient is undefined.

    Returns
    -------
    numpy.ndarray
        Float array of the same length as the inputs.
    """
    num = np.asarray(pd.to_numeric(pd.Series(np.asarray(numerator)).astype("float64")),
                     dtype="float64")
    den = np.asarray(pd.to_numeric(pd.Series(np.asarray(denominator)).astype("float64")),
                     dtype="float64")
    den = np.where(den == 0, np.nan, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, fill)


def _latest_month(frame: pd.DataFrame) -> str:
    """Return the most recent ``snapshot_month`` label present in the frame."""
    return str(frame["snapshot_month"].max())


def _months_covered(frame: pd.DataFrame) -> int:
    """Count distinct snapshot months, used to annualise monthly flows."""
    return max(int(frame["snapshot_month"].nunique()), 1)


# ---------------------------------------------------------------------------
# 1. Turnover
# ---------------------------------------------------------------------------
def inventory_turnover(inv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annualised inventory turns for every SKU-warehouse position.

    Turnover is annualised issues divided by average stock on hand. Average
    on-hand (rather than closing on-hand) is used because a single month-end
    snapshot can be flattered or punished by the timing of one inbound
    container. Days of inventory is the same metric expressed the way planners
    talk about it: how many days of cover the average position carries.

    Positions below ``inventory.target_turnover`` are slow movers - capital is
    turning over more slowly than the corporate benchmark and the stock is
    ageing towards its expiry date.

    Parameters
    ----------
    inv
        Optional pre-loaded inventory frame. Defaults to ``None``, which loads
        the full snapshot table via :mod:`src.data.loader`.

    Returns
    -------
    pandas.DataFrame
        One row per SKU-warehouse position, sorted slowest-turning first, with
        columns ``drug_code``, ``brand_name``, ``warehouse_id``,
        ``warehouse_name``, ``region``, ``snapshots``, ``avg_units_on_hand``,
        ``avg_monthly_units_issued``, ``annual_units_issued``,
        ``avg_inventory_value_usd``, ``turnover``, ``days_of_inventory``,
        ``target_turnover``, ``turnover_gap``, ``meets_target`` and ``status``.
    """
    cfg = get_config()
    frame = _resolve(inv)
    target = float(cfg.inventory.target_turnover)

    out = (
        frame.groupby(_POSITION_KEYS, as_index=False, dropna=False)
        .agg(
            snapshots=("snapshot_month", "nunique"),
            avg_units_on_hand=("units_on_hand", "mean"),
            avg_monthly_units_issued=("units_issued", "mean"),
            avg_inventory_value_usd=("inventory_value_usd", "mean"),
        )
    )

    # Monthly issues scaled to a year, then divided by the capital held to
    # support them: the classic "how many times did we sell through the shelf".
    out["annual_units_issued"] = out["avg_monthly_units_issued"] * _MONTHS_PER_YEAR
    out["turnover"] = _safe_divide(out["annual_units_issued"], out["avg_units_on_hand"])
    out["days_of_inventory"] = _safe_divide(_DAYS_PER_YEAR, out["turnover"])

    out["target_turnover"] = target
    out["turnover_gap"] = out["turnover"] - target
    out["meets_target"] = out["turnover"] >= target
    out["status"] = np.where(out["meets_target"], "At or Above Target", "Slow Moving")

    out = out.sort_values("turnover", ascending=True, na_position="last").reset_index(drop=True)
    log.info(
        "Turnover computed for %d positions | median %.2f turns vs target %.1f | %d slow movers",
        len(out), float(out["turnover"].median(skipna=True)), target,
        int((~out["meets_target"]).sum()),
    )
    return out


# ---------------------------------------------------------------------------
# 2. ABC classification
# ---------------------------------------------------------------------------
def abc_analysis(inv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Classify SKUs into A/B/C bands by share of annual inventory value.

    Value is measured as annualised consumption value (units issued per year
    multiplied by unit cost), the standard Pareto basis: it ranks products by
    the money that flows through them, not by the accident of how much happened
    to be on the shelf at month end. Class A SKUs make up the cumulative share
    given by ``inventory.abc_thresholds.A`` and warrant tight cycle counting,
    weekly review and safety-stock optimisation; class C SKUs can be managed
    with simple reorder rules.

    Parameters
    ----------
    inv
        Optional pre-loaded inventory frame. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per SKU ordered by descending value, with columns
        ``drug_code``, ``brand_name``, ``annual_value_usd``, ``value_share_pct``,
        ``cumulative_share_pct``, ``abc_class`` and ``units_on_hand`` (the
        latest-month network position), plus ``unit_cost_usd``,
        ``annual_units_issued`` and ``on_hand_value_usd`` for context.
    """
    cfg = get_config()
    thresholds = dict(cfg.inventory.abc_thresholds)
    frame = _resolve(inv)
    months = _months_covered(frame)

    out = (
        frame.groupby(["drug_code", "brand_name"], as_index=False, dropna=False)
        .agg(
            unit_cost_usd=("unit_cost_usd", "mean"),
            total_units_issued=("units_issued", "sum"),
        )
    )
    # Snapshots may span any number of months, so scale observed issues to a
    # full year before comparing SKUs.
    out["annual_units_issued"] = out["total_units_issued"] / months * _MONTHS_PER_YEAR
    out["annual_value_usd"] = out["annual_units_issued"] * out["unit_cost_usd"]

    # Current network position, so a planner can see what is physically held
    # behind each value band.
    latest = frame[frame["snapshot_month"] == _latest_month(frame)]
    position = (
        latest.groupby("drug_code", as_index=False, dropna=False)
        .agg(units_on_hand=("units_on_hand", "sum"),
             on_hand_value_usd=("inventory_value_usd", "sum"))
    )
    out = out.merge(position, on="drug_code", how="left")
    out[["units_on_hand", "on_hand_value_usd"]] = (
        out[["units_on_hand", "on_hand_value_usd"]].fillna(0.0)
    )

    out = out.sort_values("annual_value_usd", ascending=False).reset_index(drop=True)
    total_value = float(out["annual_value_usd"].sum())
    out["value_share_pct"] = _safe_divide(out["annual_value_usd"], total_value, fill=0.0) * _PCT
    out["cumulative_share_pct"] = out["value_share_pct"].cumsum()

    # A SKU belongs to the first band whose cumulative cut-off it has not yet
    # breached; anything past the last configured cut-off is the tail class.
    cumulative_fraction = out["cumulative_share_pct"] / _PCT
    conditions = [cumulative_fraction <= float(v) for v in thresholds.values()]
    out["abc_class"] = np.select(conditions, list(thresholds.keys()), default="C")

    out["units_on_hand"] = out["units_on_hand"].astype("int64")
    ordered = ["drug_code", "brand_name", "annual_value_usd", "value_share_pct",
               "cumulative_share_pct", "abc_class", "units_on_hand",
               "unit_cost_usd", "annual_units_issued", "on_hand_value_usd"]
    out = out[ordered]

    counts = out["abc_class"].value_counts().to_dict()
    log.info("ABC analysis: %d SKUs, $%.1fM annual value, class mix %s",
             len(out), total_value / 1e6, counts)
    return out


# ---------------------------------------------------------------------------
# 3. Stock-out risk
# ---------------------------------------------------------------------------
def stockout_risk(inv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flag positions at risk of running dry before replenishment arrives.

    A position is flagged when on-hand stock has fallen below
    ``inventory.stockout_risk_threshold`` of its reorder point. The reorder
    point already encodes lead-time demand plus safety stock, so the gap
    between it and current stock is the planner's best estimate of the units
    that will be short over the replenishment window - that shortfall is what
    the ``economics.stockout_penalty_per_unit`` service-level penalty is
    applied to.

    Severity is graded by splitting the configured trigger evenly across the
    three risk levels, so a position at a third of its reorder point is High
    while one just under the trigger is Low. ``below_safety_stock`` marks the
    positions that have already eaten through their buffer and need expediting
    regardless of band.

    Parameters
    ----------
    inv
        Optional pre-loaded inventory frame. Defaults to ``None``. Pass a
        single snapshot month for a purely operational view.

    Returns
    -------
    pandas.DataFrame
        One row per at-risk position, sorted by descending estimated cost, with
        columns ``snapshot_month``, the position keys, ``units_on_hand``,
        ``reorder_point``, ``safety_stock``, ``monthly_demand_units``,
        ``months_of_supply``, ``risk_ratio``, ``risk_level``,
        ``below_safety_stock``, ``shortfall_units`` and ``stockout_cost_usd``.
        Empty (with the full column set) when nothing is at risk.
    """
    cfg = get_config()
    frame = _resolve(inv)
    trigger = float(cfg.inventory.stockout_risk_threshold)
    penalty = float(cfg.economics.stockout_penalty_per_unit)

    frame["risk_ratio"] = _safe_divide(frame["units_on_hand"], frame["reorder_point"])
    at_risk = frame[frame["risk_ratio"] < trigger].copy()

    # Even bands across the configured trigger: one number to tune, three
    # actionable severities for the planner's work queue.
    edges = [trigger * (i + 1) / len(_RISK_LEVELS) for i in range(len(_RISK_LEVELS))]
    conditions = [at_risk["risk_ratio"] < edge for edge in edges]
    at_risk["risk_level"] = np.select(conditions, list(_RISK_LEVELS), default=_RISK_LEVELS[-1])
    at_risk["below_safety_stock"] = at_risk["units_on_hand"] < at_risk["safety_stock"]

    # Units needed to get back to the reorder point == expected exposure over
    # the lead time, since the reorder point is lead-time demand + safety stock.
    at_risk["shortfall_units"] = (at_risk["reorder_point"] - at_risk["units_on_hand"]).clip(lower=0)
    at_risk["stockout_cost_usd"] = at_risk["shortfall_units"] * penalty

    columns = (["snapshot_month"] + _POSITION_KEYS
               + ["units_on_hand", "reorder_point", "safety_stock", "monthly_demand_units",
                  "months_of_supply", "risk_ratio", "risk_level", "below_safety_stock",
                  "shortfall_units", "stockout_cost_usd"])
    out = (at_risk[columns]
           .sort_values("stockout_cost_usd", ascending=False)
           .reset_index(drop=True))

    log.info(
        "Stock-out risk: %d of %d positions below %.0f%% of reorder point "
        "| exposure $%.2fM | %d already under safety stock",
        len(out), len(frame), trigger * _PCT,
        float(out["stockout_cost_usd"].sum()) / 1e6,
        int(out["below_safety_stock"].sum()),
    )
    return out


# ---------------------------------------------------------------------------
# 4. Overstock risk
# ---------------------------------------------------------------------------
def overstock_risk(inv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Identify positions carrying more cover than the policy allows.

    Anything above ``inventory.overstock_months_of_supply`` months of cover is
    excess: it is working capital that could have been deployed elsewhere, it
    accrues warehousing, insurance and financing cost at
    ``economics.holding_cost_rate_annual``, and in a shelf-life-limited
    portfolio it is the leading indicator of a future write-off.

    Excess is measured against the demand that supports it rather than against
    the reorder point, because the policy is expressed in months of cover.

    Parameters
    ----------
    inv
        Optional pre-loaded inventory frame. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per overstocked position, sorted by descending excess value,
        with columns ``snapshot_month``, the position keys, ``units_on_hand``,
        ``monthly_demand_units``, ``months_of_supply``, ``target_months``,
        ``excess_months``, ``excess_units``, ``excess_value_usd``,
        ``holding_cost_usd`` and ``days_to_expiry``.
    """
    cfg = get_config()
    frame = _resolve(inv)
    target_months = float(cfg.inventory.overstock_months_of_supply)
    holding_rate = float(cfg.economics.holding_cost_rate_annual)

    over = frame[frame["months_of_supply"] > target_months].copy()
    over["target_months"] = target_months
    over["excess_months"] = over["months_of_supply"] - target_months

    # Units held beyond the policy cover. Clipped at zero so a rounding
    # artefact in months_of_supply can never produce negative "excess".
    over["excess_units"] = (
        over["units_on_hand"] - over["monthly_demand_units"] * target_months
    ).clip(lower=0)
    over["excess_value_usd"] = over["excess_units"] * over["unit_cost_usd"]
    over["holding_cost_usd"] = over["excess_value_usd"] * holding_rate

    columns = (["snapshot_month"] + _POSITION_KEYS
               + ["units_on_hand", "monthly_demand_units", "months_of_supply",
                  "target_months", "excess_months", "excess_units",
                  "excess_value_usd", "holding_cost_usd", "days_to_expiry"])
    out = (over[columns]
           .sort_values("excess_value_usd", ascending=False)
           .reset_index(drop=True))

    log.info(
        "Overstock: %d of %d positions above %.1f months cover | $%.2fM excess value, "
        "$%.2fM annual holding cost",
        len(out), len(frame), target_months,
        float(out["excess_value_usd"].sum()) / 1e6,
        float(out["holding_cost_usd"].sum()) / 1e6,
    )
    return out


# ---------------------------------------------------------------------------
# 5. Expiry risk
# ---------------------------------------------------------------------------
def expiry_risk(inv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quantify stock approaching its expiry date.

    Positions with ``days_to_expiry`` inside the ``inventory.expiry_risk_days``
    window are shown with the units already tagged as expiring soon. Expired
    pharmaceutical stock cannot be resold; only the fraction in
    ``economics.expiry_write_off_recovery`` is recoverable (returns credits,
    material salvage), so the write-off is the balance of the value at cost.

    This is the table that drives short-dated stock actions: reallocate to a
    higher-demand region, run a promotion, or accept the loss.

    Parameters
    ----------
    inv
        Optional pre-loaded inventory frame. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per at-risk position, sorted by descending write-off value,
        with columns ``snapshot_month``, the position keys, ``days_to_expiry``,
        ``units_on_hand``, ``units_expiring_soon``, ``expiring_share_pct``,
        ``monthly_demand_units``, ``months_of_supply``, ``at_risk_value_usd``,
        ``salvage_value_usd``, ``write_off_value_usd`` and ``risk_level``.
    """
    cfg = get_config()
    frame = _resolve(inv)
    window_days = float(cfg.inventory.expiry_risk_days)
    recovery = float(cfg.economics.expiry_write_off_recovery)

    near = frame[frame["days_to_expiry"] <= window_days].copy()
    near["at_risk_value_usd"] = near["units_expiring_soon"] * near["unit_cost_usd"]
    near["salvage_value_usd"] = near["at_risk_value_usd"] * recovery
    near["write_off_value_usd"] = near["at_risk_value_usd"] - near["salvage_value_usd"]
    near["expiring_share_pct"] = _safe_divide(
        near["units_expiring_soon"], near["units_on_hand"], fill=0.0) * _PCT

    # Urgency scales with how far into the risk window the stock already is,
    # using the same even-band rule as the stock-out grading.
    edges = [window_days * (i + 1) / len(_RISK_LEVELS) for i in range(len(_RISK_LEVELS))]
    conditions = [near["days_to_expiry"] <= edge for edge in edges]
    near["risk_level"] = np.select(conditions, list(_RISK_LEVELS), default=_RISK_LEVELS[-1])

    columns = (["snapshot_month"] + _POSITION_KEYS
               + ["days_to_expiry", "units_on_hand", "units_expiring_soon",
                  "expiring_share_pct", "monthly_demand_units", "months_of_supply",
                  "at_risk_value_usd", "salvage_value_usd", "write_off_value_usd",
                  "risk_level"])
    out = (near[columns]
           .sort_values("write_off_value_usd", ascending=False)
           .reset_index(drop=True))

    log.info(
        "Expiry risk: %d positions within %.0f days of expiry | %s units at risk | "
        "$%.2fM net write-off exposure",
        len(out), window_days, f"{int(out['units_expiring_soon'].sum()):,}",
        float(out["write_off_value_usd"].sum()) / 1e6,
    )
    return out


# ---------------------------------------------------------------------------
# 6. Warehouse utilisation
# ---------------------------------------------------------------------------
def warehouse_utilisation(inv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Physical space utilisation of each distribution centre.

    Capacity is a point-in-time constraint, so this always resolves to the
    latest snapshot month in the supplied data regardless of how much history
    is passed in. Constrained sites cannot absorb an inbound container without
    cross-docking or overflow storage; under-utilised sites are paying for
    racking they are not using and are candidates for network consolidation.

    Parameters
    ----------
    inv
        Optional pre-loaded inventory frame. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per warehouse, sorted by descending utilisation, with columns
        ``snapshot_month``, ``warehouse_id``, ``warehouse_name``, ``region``,
        ``skus_held``, ``units_on_hand``, ``warehouse_capacity``,
        ``utilisation_pct``, ``free_capacity_units``, ``inventory_value_usd``
        and ``status``.
    """
    frame = _resolve(inv)
    month = _latest_month(frame)
    latest = frame[frame["snapshot_month"] == month]

    out = (
        latest.groupby(["warehouse_id", "warehouse_name", "region"],
                       as_index=False, dropna=False)
        .agg(
            skus_held=("drug_code", "nunique"),
            units_on_hand=("units_on_hand", "sum"),
            # Capacity is a property of the site, repeated on every SKU row.
            warehouse_capacity=("warehouse_capacity", "max"),
            inventory_value_usd=("inventory_value_usd", "sum"),
        )
    )
    out.insert(0, "snapshot_month", month)
    out["utilisation_pct"] = _safe_divide(
        out["units_on_hand"], out["warehouse_capacity"], fill=0.0) * _PCT
    out["free_capacity_units"] = (out["warehouse_capacity"] - out["units_on_hand"]).clip(lower=0)

    under, healthy, constrained = _UTILISATION_STATUS
    out["status"] = np.select(
        [out["utilisation_pct"] < _UTILISATION_HEALTHY_FLOOR_PCT,
         out["utilisation_pct"] <= _UTILISATION_CONSTRAINED_PCT],
        [under, healthy],
        default=constrained,
    )

    out = out.sort_values("utilisation_pct", ascending=False).reset_index(drop=True)
    log.info("Warehouse utilisation for %s: %d sites, mean %.1f%% of capacity used",
             month, len(out), float(out["utilisation_pct"].mean()))
    return out


# ---------------------------------------------------------------------------
# 7. Headline KPIs
# ---------------------------------------------------------------------------
def inventory_kpis(inv: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute the headline inventory scalars for the executive summary.

    Stock value is a balance-sheet quantity, so it is taken from the latest
    snapshot month rather than summed across history (which would multiply the
    same stock by the number of months observed). Risk and cost figures are
    aggregated over everything supplied, matching the detail tables.

    Parameters
    ----------
    inv
        Optional pre-loaded inventory frame. Defaults to ``None``.

    Returns
    -------
    dict
        ``total_inventory_value_usd``, ``avg_turnover``,
        ``pct_skus_at_stockout_risk``, ``expiry_exposure_usd``,
        ``overstock_value_usd``, ``avg_months_of_supply``, plus
        ``stockout_exposure_usd``, ``excess_holding_cost_usd``,
        ``target_turnover``, ``positions_analysed`` and ``latest_snapshot_month``.
    """
    cfg = get_config()
    frame = _resolve(inv)
    month = _latest_month(frame)
    latest = frame[frame["snapshot_month"] == month]

    turns = inventory_turnover(frame)
    risk = stockout_risk(frame)
    over = overstock_risk(frame)
    expiry = expiry_risk(frame)

    kpis: dict[str, Any] = {
        "total_inventory_value_usd": float(latest["inventory_value_usd"].sum()),
        "avg_turnover": float(turns["turnover"].mean(skipna=True)),
        # Share of observed SKU-warehouse-month positions sitting below the
        # reorder trigger - the planner's "how much of my book is on fire".
        "pct_skus_at_stockout_risk": (
            float(len(risk)) / len(frame) * _PCT if len(frame) else 0.0),
        "expiry_exposure_usd": float(expiry["write_off_value_usd"].sum()),
        "overstock_value_usd": float(over["excess_value_usd"].sum()),
        "avg_months_of_supply": float(frame["months_of_supply"].mean(skipna=True)),
        "stockout_exposure_usd": float(risk["stockout_cost_usd"].sum()),
        "excess_holding_cost_usd": float(over["holding_cost_usd"].sum()),
        "target_turnover": float(cfg.inventory.target_turnover),
        "positions_analysed": int(len(frame)),
        "latest_snapshot_month": month,
    }

    log.info(
        "Inventory KPIs @ %s: value $%.2fM | turns %.2f | %.1f%% at stock-out risk | "
        "expiry $%.2fM | overstock $%.2fM",
        month, kpis["total_inventory_value_usd"] / 1e6, kpis["avg_turnover"],
        kpis["pct_skus_at_stockout_risk"], kpis["expiry_exposure_usd"] / 1e6,
        kpis["overstock_value_usd"] / 1e6,
    )
    return kpis
