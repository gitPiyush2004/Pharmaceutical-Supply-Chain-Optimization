"""
Procurement and logistics analytics on the **real** USAID SCMS delivery history.

Everything in this module is measured from 10,324 actual shipments of HIV, malaria
and antiretroviral commodities to 43 countries between 2006 and 2015. Nothing here
is simulated.

How this differs from ``src.analytics.shipments``
-------------------------------------------------
``shipments.py`` analyses the simulated digital twin, where a batch is followed
through three physical transport legs with storage telemetry attached. This module
analyses real procurement line items, where the observable process is the
*paperwork-to-delivery* pipeline: a price quote goes to the client, a purchase
order goes to the vendor, a delivery date is scheduled, the goods arrive, and the
arrival is recorded.

Two measurement decisions worth stating
---------------------------------------
**Small samples are excluded from scorecards.** A vendor with three shipments and
one late delivery would otherwise rank worse than one with five hundred shipments
at 90% on-time. Anything below ``scms.min_shipments_for_scoring`` is reported
separately rather than ranked.

**Structural absence is not missing data.** 5,404 of 10,324 line items were
fulfilled from a regional distribution centre and never had a vendor purchase
order. Their vendor lead time is undefined, not zero, and they are excluded from
lead-time statistics rather than imputed. Every function that touches lead time
reports the denominator it actually used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.scms import load_scms
from src.logger import get_logger

log = get_logger(__name__)


def _cfg():
    return get_config().scms


def _resolve(scms: pd.DataFrame | None) -> pd.DataFrame:
    """Return the caller's frame, or load the real dataset if none was given."""
    return load_scms() if scms is None else scms


def _service_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aggregate the standard service and cost metrics over a grouping."""
    grouped = frame.groupby(keys, dropna=False)
    out = grouped.agg(
        shipments=("shipment_id", "size"),
        late_shipments=("is_late", "sum"),
        avg_delay_days=("delivery_delay_days", "mean"),
        median_delay_days=("delivery_delay_days", "median"),
        p90_delay_days=("delivery_delay_days", lambda s: s.quantile(0.90)),
        line_value_usd=("line_value_usd", "sum"),
        freight_cost_usd=("freight_cost_usd", "sum"),
        # Freight share is wildly right-skewed on real data, so the median is the
        # honest summary; the mean is dominated by a handful of tiny line values.
        median_freight_pct=("freight_pct_of_value", "median"),
        packs=("packs_ordered", "sum"),
        units=("units_ordered", "sum"),
    ).reset_index()

    out["on_time_pct"] = (100 * (1 - out["late_shipments"] / out["shipments"])).round(2)
    out["avg_delay_days"] = out["avg_delay_days"].round(2)
    out["median_delay_days"] = out["median_delay_days"].round(1)
    out["p90_delay_days"] = out["p90_delay_days"].round(1)
    out["median_freight_pct"] = out["median_freight_pct"].round(2)
    # Per *pack*, which is the unit the source data actually prices and ships in.
    # An earlier version called this "per unit" while dividing by pack count.
    out["freight_per_pack_usd"] = np.where(
        out["packs"] > 0, out["freight_cost_usd"] / out["packs"], np.nan).round(4)
    out["late_shipments"] = out["late_shipments"].astype(int)
    return out


# ---------------------------------------------------------------------------
# The real procurement funnel
# ---------------------------------------------------------------------------
def procurement_funnel(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Milestone coverage across the real procurement pipeline.

    Unlike a manufacturing funnel, this is not a volume funnel - no units are
    physically lost between a price quote and a delivery. What *is* lost is
    **traceability**: each milestone is recorded for a different share of line
    items, and the drop-off measures how much of the process is actually
    auditable end to end.

    That is a real and reportable finding: only 44% of line items carry a vendor
    purchase order date, because more than half are fulfilled from regional
    distribution centre stock that bypasses vendor ordering entirely.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage_order``, ``stage``, ``line_items_recorded``,
        ``coverage_pct``, ``structural_gap``, ``missing``, ``interpretation``.
    """
    df = _resolve(scms)
    total = len(df)

    milestones = [
        ("Price Quote Sent", "date_pq_sent"),
        ("PO Sent to Vendor", "date_po_sent"),
        ("Scheduled for Delivery", "date_scheduled"),
        ("Delivered to Client", "date_delivered"),
        ("Delivery Recorded", "date_recorded"),
    ]

    rows: list[dict] = []
    for order, (stage, column) in enumerate(milestones, start=1):
        reason = df.get(f"{column}_reason")
        recorded = int(df[column].notna().sum())
        structural = int((reason == "structural").sum()) if reason is not None else 0
        missing = int((reason == "missing").sum()) if reason is not None else 0

        if structural and stage == "PO Sent to Vendor":
            interpretation = (f"{structural:,} fulfilled from RDC stock - no vendor "
                              "purchase order exists")
        elif structural:
            interpretation = f"{structural:,} predate the current process"
        elif missing:
            interpretation = f"{missing:,} genuinely unrecorded"
        else:
            interpretation = "fully recorded"

        rows.append({
            "stage_order": order, "stage": stage,
            "line_items_recorded": recorded,
            "coverage_pct": round(100 * recorded / total, 2),
            "structural_gap": structural, "missing": missing,
            "interpretation": interpretation,
        })

    funnel = pd.DataFrame(rows)
    log.info("Procurement funnel over %d real line items | PO coverage %.1f%%",
             total, funnel.loc[funnel["stage"] == "PO Sent to Vendor", "coverage_pct"].iloc[0])
    return funnel


def lead_time_breakdown(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Distribution of each measurable interval in the procurement pipeline.

    Reports the denominator alongside each statistic, because the intervals are
    measured over different subsets - vendor lead time only exists where a vendor
    purchase order does.

    Returns
    -------
    pandas.DataFrame
        Columns ``interval``, ``line_items``, ``coverage_pct``, ``mean_days``,
        ``median_days``, ``p90_days``, ``basis``.
    """
    df = _resolve(scms)
    total = len(df)

    intervals = [
        ("Price quote to delivery", "total_lead_time_days",
         "Full pipeline; excludes items predating the quote process"),
        ("Vendor PO to delivery", "vendor_lead_time_days",
         "Direct-drop orders only; RDC fulfilment has no vendor PO"),
        ("Planned quote-to-scheduled", "scheduled_lead_time_days",
         "The lead time originally promised, known at order time"),
        ("Scheduled versus actual delivery", "delivery_delay_days",
         "Negative means early; all line items"),
        ("Delivery to system recording", "recording_lag_days",
         "Administrative lag after physical arrival"),
    ]

    rows: list[dict] = []
    for label, column, basis in intervals:
        series = df[column].dropna()
        rows.append({
            "interval": label,
            "line_items": len(series),
            "coverage_pct": round(100 * len(series) / total, 2),
            "mean_days": round(float(series.mean()), 1) if len(series) else np.nan,
            "median_days": round(float(series.median()), 1) if len(series) else np.nan,
            "p90_days": round(float(series.quantile(0.90)), 1) if len(series) else np.nan,
            "basis": basis,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scorecards
# ---------------------------------------------------------------------------
def vendor_scorecard(scms: pd.DataFrame | None = None,
                     min_shipments: int | None = None) -> pd.DataFrame:
    """Rank real vendors by delivery reliability, worst first.

    Parameters
    ----------
    scms
        Optional pre-filtered frame.
    min_shipments
        Minimum shipment count to be ranked. Defaults to
        ``config.scms.min_shipments_for_scoring``.

    Returns
    -------
    pandas.DataFrame
        Vendor-level service and cost metrics, plus a ``performance_tier``.
    """
    df = _resolve(scms)
    threshold = min_shipments if min_shipments is not None else _cfg().min_shipments_for_scoring

    scorecard = _service_metrics(df, ["vendor"])
    scorecard = scorecard[scorecard["shipments"] >= threshold].copy()

    tiers = get_config().shipments.supplier_tiers
    scorecard["performance_tier"] = np.select(
        [scorecard["on_time_pct"] >= tiers["Preferred"],
         scorecard["on_time_pct"] >= tiers["Approved"]],
        ["Preferred", "Approved"], default="Watchlist")
    scorecard["meets_target"] = scorecard["on_time_pct"] >= _cfg().on_time_target_pct

    scorecard = scorecard.sort_values("on_time_pct").reset_index(drop=True)
    scorecard.insert(0, "rank", range(1, len(scorecard) + 1))
    log.info("Scored %d vendors with >= %d shipments (of %d total vendors)",
             len(scorecard), threshold, df["vendor"].nunique())
    return scorecard


def country_performance(scms: pd.DataFrame | None = None,
                        min_shipments: int | None = None) -> pd.DataFrame:
    """Delivery performance by destination country, worst first."""
    df = _resolve(scms)
    threshold = min_shipments if min_shipments is not None else _cfg().min_shipments_for_scoring

    out = _service_metrics(df, ["country", "region"])
    out = out[out["shipments"] >= threshold]
    return out.sort_values("on_time_pct").reset_index(drop=True)


def region_performance(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Delivery performance by region, worst first."""
    out = _service_metrics(_resolve(scms), ["region"])
    out["target_gap_pct"] = (out["on_time_pct"] - _cfg().on_time_target_pct).round(2)
    return out.sort_values("on_time_pct").reset_index(drop=True)


def mode_performance(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Service and cost by transport mode.

    The central logistics trade-off in this dataset: air freight delivers the best
    on-time performance at the highest cost per kilogram, ocean the reverse.
    """
    out = _service_metrics(_resolve(scms), ["shipment_mode"])
    df = _resolve(scms)
    cost_per_kg = (df.groupby("shipment_mode")["freight_cost_per_kg"]
                   .median().rename("median_freight_per_kg_usd").reset_index())
    out = out.merge(cost_per_kg, on="shipment_mode", how="left")
    out["median_freight_per_kg_usd"] = out["median_freight_per_kg_usd"].round(2)
    return out.sort_values("on_time_pct").reset_index(drop=True)


def product_performance(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Service and spend by commodity group."""
    out = _service_metrics(_resolve(scms), ["product_group", "product_group_name"])
    return out.sort_values("line_value_usd", ascending=False).reset_index(drop=True)


def manufacturing_site_analysis(scms: pd.DataFrame | None = None,
                                min_shipments: int | None = None) -> pd.DataFrame:
    """Delivery reliability by manufacturing site, worst first.

    Distinct from the vendor scorecard: one vendor can ship from several sites,
    and site-level variation is where a quality or capacity problem shows up.
    """
    df = _resolve(scms)
    threshold = min_shipments if min_shipments is not None else _cfg().min_shipments_for_scoring
    out = _service_metrics(df, ["manufacturing_site"])
    out = out[out["shipments"] >= threshold]
    return out.sort_values("on_time_pct").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Economics and trend
# ---------------------------------------------------------------------------
def freight_economics(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Freight cost efficiency by mode and region.

    Uses medians throughout. Freight as a share of commodity value has a mean of
    over 2,500% on this data because some line values are a few dollars while
    their shipping is not - the median of about 11% is the number that describes
    the typical shipment.
    """
    df = _resolve(scms)
    priced = df[df["freight_cost_reason"] == "parsed"]

    out = (priced.groupby(["shipment_mode", "region"], dropna=False)
           .agg(shipments=("shipment_id", "size"),
                freight_spend_usd=("freight_cost_usd", "sum"),
                median_freight_per_kg=("freight_cost_per_kg", "median"),
                median_freight_pct_of_value=("freight_pct_of_value", "median"),
                total_weight_kg=("weight_kg", "sum"))
           .reset_index())
    out = out[out["shipments"] >= 20]
    for column in ("median_freight_per_kg", "median_freight_pct_of_value"):
        out[column] = out[column].round(2)
    out["freight_spend_usd"] = out["freight_spend_usd"].round(0)
    return out.sort_values("median_freight_per_kg", ascending=False).reset_index(drop=True)


def delivery_trend(scms: pd.DataFrame | None = None, period: str = "quarter") -> pd.DataFrame:
    """On-time performance and spend over time.

    Parameters
    ----------
    period
        ``"quarter"`` (default), ``"month"`` or ``"year"``.
    """
    df = _resolve(scms)
    column = {"quarter": "delivery_quarter", "month": "delivery_month",
              "year": "delivery_year"}[period]

    out = (df.groupby(column)
           .agg(shipments=("shipment_id", "size"),
                late_shipments=("is_late", "sum"),
                avg_delay_days=("delivery_delay_days", "mean"),
                line_value_usd=("line_value_usd", "sum"),
                freight_cost_usd=("freight_cost_usd", "sum"))
           .reset_index())
    out["on_time_pct"] = (100 * (1 - out["late_shipments"] / out["shipments"])).round(2)
    out["avg_delay_days"] = out["avg_delay_days"].round(2)
    out["late_shipments"] = out["late_shipments"].astype(int)
    return out.sort_values(column).reset_index(drop=True)


def delay_distribution(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bucket delivery timing from very early to severely late.

    A plain on-time percentage hides that this supply chain delivers *early* far
    more often than late - the median shipment arrives exactly on its scheduled
    date, and the 25th percentile is three days early.
    """
    df = _resolve(scms)
    delays = df["delivery_delay_days"].dropna()

    buckets = [
        ("More than 30 days early", delays < -30),
        ("8 to 30 days early", (delays >= -30) & (delays < -7)),
        ("1 to 7 days early", (delays >= -7) & (delays < 0)),
        ("On the scheduled day", delays == 0),
        ("1 to 7 days late", (delays > 0) & (delays <= 7)),
        ("8 to 30 days late", (delays > 7) & (delays <= 30)),
        ("More than 30 days late", delays > 30),
    ]
    rows = [{"bucket": label, "shipments": int(mask.sum()),
             "share_pct": round(100 * mask.sum() / len(delays), 2)}
            for label, mask in buckets]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------
def scms_kpis(scms: pd.DataFrame | None = None) -> dict:
    """Headline metrics for the real dataset.

    Returns
    -------
    dict
        Scalars covering volume, service, spend and coverage.
    """
    df = _resolve(scms)
    delays = df["delivery_delay_days"].dropna()
    priced = df[df["freight_cost_reason"] == "parsed"]

    kpis = {
        "shipments": int(len(df)),
        "countries": int(df["country"].nunique()),
        "vendors": int(df["vendor"].nunique()),
        "manufacturing_sites": int(df["manufacturing_site"].nunique()),
        "on_time_delivery_pct": round(100 * (1 - df["is_late"].mean()), 2),
        "on_time_target_pct": float(_cfg().on_time_target_pct),
        "late_shipments": int(df["is_late"].sum()),
        "avg_delay_days": round(float(delays.mean()), 2),
        "median_delay_days": round(float(delays.median()), 1),
        "total_commodity_value_usd": round(float(df["line_value_usd"].sum()), 0),
        "total_freight_spend_usd": round(float(priced["freight_cost_usd"].sum()), 0),
        "median_freight_pct_of_value": round(float(df["freight_pct_of_value"].median()), 2),
        # Packs and units are reported separately because the source prices and
        # ships in packs; conflating them under-reported units ~50-fold.
        "total_packs": int(df["packs_ordered"].sum()),
        "total_units": int(df["units_ordered"].sum()),
        "median_vendor_lead_time_days": round(
            float(df["vendor_lead_time_days"].median()), 1),
        "vendor_lead_time_coverage_pct": round(
            100 * df["vendor_lead_time_days"].notna().mean(), 1),
        "date_from": str(df["date_delivered"].min().date()),
        "date_to": str(df["date_delivered"].max().date()),
    }
    kpis["meets_on_time_target"] = (
        kpis["on_time_delivery_pct"] >= kpis["on_time_target_pct"])
    return kpis


__all__ = [
    "procurement_funnel", "lead_time_breakdown", "vendor_scorecard",
    "country_performance", "region_performance", "mode_performance",
    "product_performance", "manufacturing_site_analysis", "freight_economics",
    "delivery_trend", "delay_distribution", "scms_kpis",
]
