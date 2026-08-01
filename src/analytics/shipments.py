"""
Shipment and supplier performance analytics for the pharmaceutical network.

Business purpose
----------------
Product only earns revenue once it reaches the hospital or pharmacy shelf. Every
transport leg between the plant and the patient is an opportunity to arrive
late, to arrive damaged, or - for a cold-chain product - to arrive outside its
validated 2-8C temperature envelope and therefore be unsaleable. Late deliveries
trigger contractual service-level credits; damaged and temperature-excursed
units are written off; and freight is one of the largest controllable lines in
the distribution budget.

This module turns the leg-level shipment fact table produced by
:func:`src.data.loader.load_shipments`, enriched with the supplier dimension
from :func:`src.data.loader.load_suppliers`, into the six views used to run a
supplier and carrier review:

==============================  ===============================================
Function                        Question it answers
==============================  ===============================================
:func:`supplier_performance`    Which suppliers are failing their service
                                commitments, and does the contracted tier still
                                reflect how they actually perform?
:func:`regional_performance`    Which lanes are missing the OTIF target?
:func:`carrier_scorecard`       Which carrier and mode combinations deliver
                                reliability worth what they cost?
:func:`late_shipment_analysis`  Where does lateness concentrate, is it getting
                                better, and what is it costing?
:func:`transit_time_distribution` Are planned transit times realistic, and how
                                fat is the tail?
:func:`shipment_kpis`           The headline scalars for the executive summary.
==============================  ===============================================

How to read the output
----------------------
Every scorecard is ranked worst-first so the top rows are the ones to raise in
the next business review. Rates are volume-weighted wherever the underlying
quantity has a natural denominator - damage rate is damaged units over units
shipped, not the mean of per-shipment percentages - so a single large ruined
consignment cannot hide behind many small clean ones. Delay is measured as
lateness (early arrivals count as zero, not as negative delay), because
arriving three days early does not offset arriving three days late on a
different order.

Lateness itself is re-derived from ``delay_days`` against
``shipments.on_time_grace_days`` rather than trusted from the stored flag, so
changing the grace period in configuration immediately changes every number
this module produces. The OTIF target, the supplier tier ladder and the
per-late-shipment penalty all come from ``config/config.yaml``.

Example
-------
>>> from src.analytics.shipments import shipment_kpis, supplier_performance
>>> kpis = shipment_kpis()
>>> worst_suppliers = supplier_performance().head(3)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)

_PCT: float = 100.0

# Tail percentile used for the "bad day" transit time. Not a business rule -
# it is the definition of the reported statistic.
_P90_QUANTILE: float = 0.90

# Derived columns added by :func:`_prepare`; kept private so the public frames
# only expose business-meaningful names.
_LATE_FLAG: str = "_is_late"
_LATENESS: str = "_lateness_days"

__all__ = [
    "supplier_performance",
    "regional_performance",
    "carrier_scorecard",
    "late_shipment_analysis",
    "transit_time_distribution",
    "shipment_kpis",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve(shipments: pd.DataFrame | None) -> pd.DataFrame:
    """Load, copy and enrich the shipment frame.

    Parameters
    ----------
    shipments
        Pre-loaded (and possibly filtered) shipment frame, or ``None`` to read
        the full table through :mod:`src.data.loader`.

    Returns
    -------
    pandas.DataFrame
        A copy carrying the private lateness columns used by every aggregation.
    """
    frame = (loader.load_shipments() if shipments is None else shipments).copy()
    return _prepare(frame)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the configuration-driven lateness columns to a shipment frame.

    A delivery is only late once it exceeds the contractual grace period, so
    lateness is recomputed here instead of relying on the stored ``is_late``
    flag - that keeps the whole module responsive to a change in
    ``shipments.on_time_grace_days``.
    """
    cfg = get_config()
    grace = float(cfg.shipments.on_time_grace_days)
    delay = pd.to_numeric(frame["delay_days"], errors="coerce").fillna(0.0)
    frame[_LATE_FLAG] = (delay > grace).astype("int64")
    # Arriving early is not negative lateness: it cannot compensate for a
    # different shipment arriving late, so the downside is measured on its own.
    frame[_LATENESS] = delay.clip(lower=0.0)
    return frame


def _safe_divide(numerator: Any, denominator: Any, fill: float = np.nan) -> np.ndarray:
    """Element-wise division that never raises or returns an infinity.

    Empty groups (a carrier with no volume on a mode) are expected in a sparse
    network, so an undefined quotient resolves to ``fill``.

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
    num = np.asarray(pd.Series(np.asarray(numerator)).astype("float64"), dtype="float64")
    den = np.asarray(pd.Series(np.asarray(denominator)).astype("float64"), dtype="float64")
    den = np.where(den == 0, np.nan, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, fill)


def _service_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aggregate the standard service-quality metric set over any grouping.

    Every scorecard in this module answers the same four questions - did it
    arrive on time, did it arrive intact, did it stay in temperature, and what
    did it cost - so the arithmetic lives in one place.

    Parameters
    ----------
    frame
        Prepared shipment frame (see :func:`_prepare`).
    keys
        Column names to group by.

    Returns
    -------
    pandas.DataFrame
        One row per group with ``shipments``, ``units_shipped``,
        ``units_received``, ``late_shipments``, ``on_time_pct``,
        ``avg_delay_days``, ``max_delay_days``, ``damage_rate_pct``,
        ``excursion_rate_pct``, ``freight_spend_usd`` and ``cost_per_unit_usd``.
    """
    out = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            shipments=("shipment_id", "count"),
            units_shipped=("units_shipped", "sum"),
            units_received=("units_received", "sum"),
            units_damaged=("units_damaged", "sum"),
            late_shipments=(_LATE_FLAG, "sum"),
            avg_delay_days=(_LATENESS, "mean"),
            max_delay_days=(_LATENESS, "max"),
            excursions=("temperature_excursion", "sum"),
            freight_spend_usd=("freight_cost_usd", "sum"),
        )
    )
    out["on_time_pct"] = (1.0 - _safe_divide(out["late_shipments"], out["shipments"],
                                             fill=0.0)) * _PCT
    # Volume-weighted so one large ruined consignment is not diluted by many
    # small clean ones.
    out["damage_rate_pct"] = _safe_divide(out["units_damaged"], out["units_shipped"],
                                          fill=0.0) * _PCT
    out["excursion_rate_pct"] = _safe_divide(out["excursions"], out["shipments"],
                                             fill=0.0) * _PCT
    out["cost_per_unit_usd"] = _safe_divide(out["freight_spend_usd"], out["units_shipped"],
                                            fill=0.0)
    return out.drop(columns=["units_damaged", "excursions"])


def _late_breakdown(frame: pd.DataFrame, key: str, penalty: float) -> pd.DataFrame:
    """Late-shipment counts, rates and penalty cost for a single dimension."""
    out = (
        frame.groupby(key, as_index=False, dropna=False)
        .agg(
            shipments=("shipment_id", "count"),
            late_shipments=(_LATE_FLAG, "sum"),
            avg_delay_days=(_LATENESS, "mean"),
            units_shipped=("units_shipped", "sum"),
            freight_spend_usd=("freight_cost_usd", "sum"),
        )
    )
    out["late_pct"] = _safe_divide(out["late_shipments"], out["shipments"], fill=0.0) * _PCT
    out["on_time_pct"] = _PCT - out["late_pct"]
    out["late_shipment_cost_usd"] = out["late_shipments"] * penalty
    return out.sort_values("late_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Supplier performance
# ---------------------------------------------------------------------------
def supplier_performance(shipments: pd.DataFrame | None = None,
                         suppliers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank suppliers on delivered service quality, worst performer first.

    The contracted tier held in the supplier dimension is what procurement
    agreed; ``performance_tier`` is what the shipment record actually earns
    against the ``shipments.supplier_tiers`` ladder. Where the two disagree the
    supplier is either over-rated (a Preferred partner delivering Watchlist
    service) or under-rated, and ``tier_matches_contract`` is the flag that
    turns that into an agenda item for the next business review.

    Parameters
    ----------
    shipments
        Optional pre-loaded shipment frame. Defaults to ``None``, which loads
        the full table via :mod:`src.data.loader`.
    suppliers
        Optional pre-loaded supplier dimension. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per supplier ordered worst-first, with ``rank``,
        ``supplier_id``, ``supplier_name``, ``shipments``, ``on_time_pct``,
        ``avg_delay_days``, ``damage_rate_pct``, ``excursion_rate_pct``,
        ``freight_spend_usd``, ``performance_tier``, ``contracted_tier``,
        ``tier_matches_contract``, ``meets_otif_target`` and the dimension
        attributes ``country``, ``reliability``, ``lead_time_days``,
        ``quality_rating`` and ``contracted_otif_pct``.
    """
    cfg = get_config()
    frame = _resolve(shipments)
    dim = (loader.load_suppliers() if suppliers is None else suppliers).copy()
    tier_floors = dict(cfg.shipments.supplier_tiers)
    otif_target = float(cfg.shipments.otif_target_pct)

    out = _service_metrics(frame, ["supplier_id", "supplier_name"])

    # Earned tier: the highest ladder rung whose on-time floor the supplier
    # actually clears. Evaluated from the top down so the best match wins.
    ladder = sorted(tier_floors.items(), key=lambda item: float(item[1]), reverse=True)
    conditions = [out["on_time_pct"] >= float(floor) for _, floor in ladder]
    out["performance_tier"] = np.select(conditions, [name for name, _ in ladder],
                                        default=ladder[-1][0])

    dim_cols = ["supplier_id", "country", "reliability", "lead_time_days",
                "quality_rating", "otif_pct", "tier"]
    out = out.merge(
        dim[[c for c in dim_cols if c in dim.columns]]
        .rename(columns={"tier": "contracted_tier", "otif_pct": "contracted_otif_pct"}),
        on="supplier_id", how="left",
    )
    out["tier_matches_contract"] = out["performance_tier"] == out["contracted_tier"]
    out["otif_target_pct"] = otif_target
    out["meets_otif_target"] = out["on_time_pct"] >= otif_target

    out = out.sort_values(["on_time_pct", "avg_delay_days"],
                          ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    log.info(
        "Supplier performance: %d suppliers over %d shipments | worst %s at %.1f%% on-time "
        "| %d below the %.1f%% OTIF target | %d mis-tiered versus contract",
        len(out), int(out["shipments"].sum()),
        out["supplier_name"].iloc[0] if len(out) else "n/a",
        float(out["on_time_pct"].iloc[0]) if len(out) else float("nan"),
        int((~out["meets_otif_target"]).sum()), otif_target,
        int((~out["tier_matches_contract"]).sum()),
    )
    return out


# ---------------------------------------------------------------------------
# 2. Regional performance
# ---------------------------------------------------------------------------
def regional_performance(shipments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score each region against the corporate OTIF target.

    Regional lanes differ in infrastructure, customs exposure and distance, so
    a network-wide on-time number can hide one lane that is consistently
    failing patients. ``otif_gap_pct`` is the distance from
    ``shipments.otif_target_pct``: negative means the region is short of target
    and is where remediation spend belongs.

    Parameters
    ----------
    shipments
        Optional pre-loaded shipment frame. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per region ordered worst-first, with ``region``, ``shipments``,
        ``on_time_pct``, ``avg_delay_days``, ``damage_rate_pct``,
        ``excursion_rate_pct``, ``freight_spend_usd``, ``cost_per_unit_usd``,
        ``otif_target_pct``, ``otif_gap_pct`` and ``meets_otif_target``.
    """
    cfg = get_config()
    frame = _resolve(shipments)
    otif_target = float(cfg.shipments.otif_target_pct)

    out = _service_metrics(frame, ["region"])
    out["otif_target_pct"] = otif_target
    out["otif_gap_pct"] = out["on_time_pct"] - otif_target
    out["meets_otif_target"] = out["on_time_pct"] >= otif_target

    out = out.sort_values("on_time_pct", ascending=True).reset_index(drop=True)
    log.info("Regional performance: %d regions | %d below the %.1f%% OTIF target",
             len(out), int((~out["meets_otif_target"]).sum()), otif_target)
    return out


# ---------------------------------------------------------------------------
# 3. Carrier scorecard
# ---------------------------------------------------------------------------
def carrier_scorecard(shipments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score every carrier and transport mode combination.

    Carriers are only comparable within a mode - air is faster and dearer than
    sea by definition - so the grain is carrier x mode. Reading
    ``on_time_pct`` against ``cost_per_unit_usd`` is the tender conversation:
    a premium carrier is only worth its rate if the reliability and the
    cold-chain excursion rate justify it.

    Parameters
    ----------
    shipments
        Optional pre-loaded shipment frame. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per carrier-mode pair ordered worst-first on reliability, with
        ``carrier``, ``transport_mode``, ``shipments``, ``units_shipped``,
        ``on_time_pct``, ``avg_delay_days``, ``cost_per_unit_usd``,
        ``excursion_rate_pct``, ``damage_rate_pct`` and ``freight_spend_usd``.
    """
    frame = _resolve(shipments)
    metrics = _service_metrics(frame, ["carrier", "transport_mode"])

    columns = ["carrier", "transport_mode", "shipments", "units_shipped",
               "on_time_pct", "avg_delay_days", "cost_per_unit_usd",
               "excursion_rate_pct", "damage_rate_pct", "freight_spend_usd"]
    out = (metrics[columns]
           .sort_values(["on_time_pct", "cost_per_unit_usd"], ascending=[True, False])
           .reset_index(drop=True))

    log.info("Carrier scorecard: %d carrier x mode combinations across %d carriers",
             len(out), int(out["carrier"].nunique()))
    return out


# ---------------------------------------------------------------------------
# 4. Late shipment analysis
# ---------------------------------------------------------------------------
def late_shipment_analysis(shipments: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Decompose lateness by mode, region, leg and month, and price it.

    Lateness is rarely uniform: it usually concentrates in one mode, one lane
    or one leg of the journey. The four breakdowns locate the concentration,
    ``monthly_trend`` shows whether it is improving or drifting, and
    ``cost_impact`` converts the counts into the service-level credits owed at
    ``economics.late_shipment_penalty`` per late shipment so the remediation
    business case writes itself.

    Parameters
    ----------
    shipments
        Optional pre-loaded shipment frame. Defaults to ``None``.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``by_mode``, ``by_region`` and ``by_leg`` - counts, late rate, average
        lateness and penalty cost per segment, worst-first;
        ``monthly_trend`` - ``year_month`` against ``on_time_pct``, late count
        and cumulative penalty, in calendar order;
        ``cost_impact`` - a tidy ``dimension``/``segment`` stack of the penalty
        cost including an ``Overall`` row, with each segment's share of the
        total bill.
    """
    cfg = get_config()
    frame = _resolve(shipments)
    penalty = float(cfg.economics.late_shipment_penalty)

    by_mode = _late_breakdown(frame, "transport_mode", penalty)
    by_region = _late_breakdown(frame, "region", penalty)
    by_leg = _late_breakdown(frame, "leg", penalty)

    monthly_trend = (
        frame.groupby("year_month", as_index=False, dropna=False)
        .agg(
            shipments=("shipment_id", "count"),
            late_shipments=(_LATE_FLAG, "sum"),
            avg_delay_days=(_LATENESS, "mean"),
            freight_spend_usd=("freight_cost_usd", "sum"),
        )
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    monthly_trend["on_time_pct"] = (1.0 - _safe_divide(
        monthly_trend["late_shipments"], monthly_trend["shipments"], fill=0.0)) * _PCT
    monthly_trend["late_shipment_cost_usd"] = monthly_trend["late_shipments"] * penalty

    total_late = int(frame[_LATE_FLAG].sum())
    total_cost = total_late * penalty
    # Tidy stack: one frame the dashboard can facet by `dimension` rather than
    # three near-identical tables the caller has to concatenate itself.
    pieces = [
        pd.DataFrame({"dimension": ["Overall"], "segment": ["All shipments"],
                      "late_shipments": [total_late],
                      "late_shipment_cost_usd": [total_cost]}),
        *[
            table.assign(dimension=label)
            .rename(columns={key: "segment"})[
                ["dimension", "segment", "late_shipments", "late_shipment_cost_usd"]]
            for label, key, table in (
                ("Transport Mode", "transport_mode", by_mode),
                ("Region", "region", by_region),
                ("Leg", "leg", by_leg),
            )
        ],
    ]
    cost_impact = pd.concat(pieces, ignore_index=True)
    cost_impact["segment"] = cost_impact["segment"].astype("string")
    cost_impact["share_of_cost_pct"] = _safe_divide(
        cost_impact["late_shipment_cost_usd"], total_cost, fill=0.0) * _PCT
    cost_impact["penalty_per_late_shipment_usd"] = penalty

    log.info(
        "Late shipment analysis: %d of %d shipments late (%.1f%%) | $%.2fM in SLA penalties "
        "| worst mode %s, worst region %s",
        total_late, len(frame),
        (total_late / len(frame) * _PCT) if len(frame) else 0.0, total_cost / 1e6,
        by_mode["transport_mode"].iloc[0] if len(by_mode) else "n/a",
        by_region["region"].iloc[0] if len(by_region) else "n/a",
    )
    return {"by_mode": by_mode, "by_region": by_region, "by_leg": by_leg,
            "monthly_trend": monthly_trend, "cost_impact": cost_impact}


# ---------------------------------------------------------------------------
# 5. Transit time distribution
# ---------------------------------------------------------------------------
def transit_time_distribution(shipments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare planned against actual transit times by transport mode.

    Planners promise the planned time; customers experience the actual one. A
    positive ``mean_variance_days`` means the plan is systematically optimistic
    and should be re-baselined rather than policed. The p90 columns expose the
    tail that safety stock has to absorb: a mode whose mean looks fine but
    whose p90 is far above plan is unreliable rather than slow, and unreliable
    lanes are what drive buffer inventory.

    Parameters
    ----------
    shipments
        Optional pre-loaded shipment frame. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per transport mode ordered by descending mean variance, with
        ``transport_mode``, ``shipments``, ``planned_mean_days``,
        ``planned_median_days``, ``planned_p90_days``, ``planned_sd_days``,
        ``actual_mean_days``, ``actual_median_days``, ``actual_p90_days``,
        ``actual_sd_days``, ``mean_variance_days``, ``p90_variance_days`` and
        ``on_time_pct``.
    """
    frame = _resolve(shipments)

    def _p90(series: pd.Series) -> float:
        """Ninetieth-percentile transit time - the realistic worst case."""
        return float(series.quantile(_P90_QUANTILE))

    out = (
        frame.groupby("transport_mode", as_index=False, dropna=False)
        .agg(
            shipments=("shipment_id", "count"),
            planned_mean_days=("planned_transit_days", "mean"),
            planned_median_days=("planned_transit_days", "median"),
            planned_p90_days=("planned_transit_days", _p90),
            planned_sd_days=("planned_transit_days", "std"),
            actual_mean_days=("actual_transit_days", "mean"),
            actual_median_days=("actual_transit_days", "median"),
            actual_p90_days=("actual_transit_days", _p90),
            actual_sd_days=("actual_transit_days", "std"),
            late_shipments=(_LATE_FLAG, "sum"),
        )
    )
    out["mean_variance_days"] = out["actual_mean_days"] - out["planned_mean_days"]
    out["p90_variance_days"] = out["actual_p90_days"] - out["planned_p90_days"]
    out["on_time_pct"] = (1.0 - _safe_divide(out["late_shipments"], out["shipments"],
                                             fill=0.0)) * _PCT
    # A single-shipment mode has no dispersion; report zero rather than NaN so
    # the dashboard does not show a blank cell.
    out[["planned_sd_days", "actual_sd_days"]] = (
        out[["planned_sd_days", "actual_sd_days"]].fillna(0.0)
    )

    out = out.drop(columns=["late_shipments"])
    out = out.sort_values("mean_variance_days", ascending=False).reset_index(drop=True)
    log.info("Transit time distribution across %d modes | worst plan variance %s at %+.2f days",
             len(out), out["transport_mode"].iloc[0] if len(out) else "n/a",
             float(out["mean_variance_days"].iloc[0]) if len(out) else float("nan"))
    return out


# ---------------------------------------------------------------------------
# 6. Headline KPIs
# ---------------------------------------------------------------------------
def shipment_kpis(shipments: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute the headline shipment scalars for the executive summary.

    Parameters
    ----------
    shipments
        Optional pre-loaded shipment frame. Defaults to ``None``.

    Returns
    -------
    dict
        ``on_time_delivery_pct``, ``avg_delay_days``, ``total_freight_spend_usd``,
        ``excursion_rate_pct``, ``late_shipment_cost_usd``, ``total_shipments``,
        plus ``late_shipments``, ``damage_rate_pct``, ``cost_per_unit_usd``,
        ``otif_target_pct`` and ``meets_otif_target`` for context.
    """
    cfg = get_config()
    frame = _resolve(shipments)
    penalty = float(cfg.economics.late_shipment_penalty)
    otif_target = float(cfg.shipments.otif_target_pct)

    total = int(len(frame))
    late = int(frame[_LATE_FLAG].sum())
    units = float(frame["units_shipped"].sum())
    freight = float(frame["freight_cost_usd"].sum())

    on_time_pct = (1.0 - late / total) * _PCT if total else 0.0
    kpis: dict[str, Any] = {
        "on_time_delivery_pct": on_time_pct,
        "avg_delay_days": float(frame[_LATENESS].mean()) if total else 0.0,
        "total_freight_spend_usd": freight,
        "excursion_rate_pct": (
            float(frame["temperature_excursion"].sum()) / total * _PCT if total else 0.0),
        "late_shipment_cost_usd": late * penalty,
        "total_shipments": total,
        "late_shipments": late,
        "damage_rate_pct": (
            float(frame["units_damaged"].sum()) / units * _PCT if units else 0.0),
        "cost_per_unit_usd": freight / units if units else 0.0,
        "otif_target_pct": otif_target,
        "meets_otif_target": bool(on_time_pct >= otif_target),
    }

    log.info(
        "Shipment KPIs: %d shipments | %.2f%% on-time vs %.1f%% target | avg lateness %.2f d "
        "| freight $%.2fM | excursions %.2f%% | SLA penalties $%.2fM",
        total, kpis["on_time_delivery_pct"], otif_target, kpis["avg_delay_days"],
        freight / 1e6, kpis["excursion_rate_pct"], kpis["late_shipment_cost_usd"] / 1e6,
    )
    return kpis
