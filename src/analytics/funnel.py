"""
Pharmaceutical supply chain funnel analytics.

Business framing
----------------
A marketing funnel tracks users dropping out between page views. A pharma
supply chain funnel tracks *units of medicine* dropping out between physical
stages - and every unit lost is both destroyed capital and a dose that never
reached a patient. This module measures that leakage across the eight stages
from API procurement to patient dispensing.

Two questions drive every output here:

1. **Where do we lose the most units?** - conversion and drop-off per stage,
   converted into dollars so the finding is prioritised the way a business
   would prioritise it.
2. **Where do we lose the most time?** - dwell time per stage, because a slow
   stage consumes shelf life and forces higher safety stock everywhere
   downstream.

A stage is flagged as a **bottleneck** when it breaches either threshold in
``config.funnel.bottleneck``. Reporting both dimensions matters: the stage that
destroys the most units and the stage that consumes the most time are usually
not the same stage, and they need different interventions.

Example
-------
>>> from src.analytics.funnel import funnel_summary, identify_bottlenecks
>>> funnel_summary()[["stage", "units", "dropoff_pct", "avg_delay_days"]]
>>> identify_bottlenecks().query("is_bottleneck")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)


def _delay_column(stage: str) -> str:
    """Map a funnel stage name to its generated dwell-time column.

    ``'Hospital/Pharmacy'`` becomes ``'delay_hospital_pharmacy_days'``, matching
    the naming convention used by the dataset generator.
    """
    slug = stage.lower().replace("/", "_").replace(" ", "_")
    return f"delay_{slug}_days"


def _batches(batches: pd.DataFrame | None) -> pd.DataFrame:
    """Return the caller's frame, or load the full batch fact table."""
    return loader.load_batches() if batches is None else batches


# ---------------------------------------------------------------------------
# Core funnel
# ---------------------------------------------------------------------------
def funnel_summary(batches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the end-to-end funnel: units, conversion, drop-off and dwell time.

    Parameters
    ----------
    batches
        Optional pre-filtered batch table. Loaded from disk when omitted, which
        lets the dashboard pass a region- or product-filtered slice straight in.

    Returns
    -------
    pd.DataFrame
        One row per stage, in process order, with columns:

        ``stage_order``, ``stage``, ``units``, ``batches``,
        ``conversion_from_previous_pct`` - % of the previous stage's units that survived,
        ``dropoff_pct`` - the complement of the above,
        ``units_lost``, ``value_lost_usd``,
        ``cumulative_conversion_pct`` - % of originally procured units still present,
        ``avg_delay_days``, ``median_delay_days``, ``p90_delay_days``.
    """
    cfg = get_config()
    df = _batches(batches)
    stages = list(cfg.funnel.stages)
    unit_cols = cfg.funnel.unit_columns

    # Weighted-average unit cost so the dollar figures reflect the actual
    # product mix in the slice being analysed, not a portfolio-wide average.
    total_units = df["units_procured"].sum()
    avg_unit_cost = (
        float((df["units_procured"] * df["unit_cost_usd"]).sum() / total_units)
        if total_units > 0 else 0.0
    )

    rows: list[dict] = []
    first_stage_units: float | None = None
    previous_units: float | None = None

    for order, stage in enumerate(stages, start=1):
        units = float(df[unit_cols[stage]].sum())
        if first_stage_units is None:
            first_stage_units = units

        conversion = 100.0 if previous_units is None else (
            100.0 * units / previous_units if previous_units > 0 else 0.0)
        units_lost = 0.0 if previous_units is None else max(previous_units - units, 0.0)

        delay_col = _delay_column(stage)
        delays = df[delay_col] if delay_col in df.columns else pd.Series(dtype=float)

        rows.append({
            "stage_order": order,
            "stage": stage,
            "units": units,
            "batches": int(len(df)),
            "conversion_from_previous_pct": round(conversion, 2),
            "dropoff_pct": round(100.0 - conversion, 2),
            "units_lost": units_lost,
            "value_lost_usd": round(units_lost * avg_unit_cost, 2),
            "cumulative_conversion_pct": round(
                100.0 * units / first_stage_units if first_stage_units else 0.0, 2),
            "avg_delay_days": round(float(delays.mean()), 2) if len(delays) else 0.0,
            "median_delay_days": round(float(delays.median()), 2) if len(delays) else 0.0,
            "p90_delay_days": round(float(delays.quantile(0.90)), 2) if len(delays) else 0.0,
        })
        previous_units = units

    summary = pd.DataFrame(rows)
    log.info("Funnel computed over %d batches | end-to-end yield %.2f%%",
             len(df), summary["cumulative_conversion_pct"].iloc[-1])
    return summary


def stage_delays(batches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the dwell-time distribution for every stage, slowest first.

    Returns
    -------
    pd.DataFrame
        ``stage``, ``avg_delay_days``, ``median_delay_days``, ``p90_delay_days``,
        ``sd_days``, ``min_days``, ``max_days``, ``share_of_cycle_pct``.

        ``share_of_cycle_pct`` is the key column - it says what fraction of total
        lead time each stage consumes, which is how you decide where a day saved
        is worth the most.
    """
    cfg = get_config()
    df = _batches(batches)

    rows = []
    for stage in cfg.funnel.stages:
        col = _delay_column(stage)
        if col not in df.columns:
            continue
        series = df[col].dropna()
        rows.append({
            "stage": stage,
            "avg_delay_days": round(float(series.mean()), 2),
            "median_delay_days": round(float(series.median()), 2),
            "p90_delay_days": round(float(series.quantile(0.90)), 2),
            "sd_days": round(float(series.std()), 2),
            "min_days": round(float(series.min()), 2),
            "max_days": round(float(series.max()), 2),
        })

    delays = pd.DataFrame(rows)
    total = delays["avg_delay_days"].sum()
    delays["share_of_cycle_pct"] = (
        (100.0 * delays["avg_delay_days"] / total).round(2) if total > 0 else 0.0)
    return delays.sort_values("avg_delay_days", ascending=False).reset_index(drop=True)


def identify_bottlenecks(batches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank stages by a combined unit-loss and time-loss severity score.

    A stage is flagged when it breaches *either* configured threshold
    (``dropoff_pct_threshold`` or ``delay_days_threshold``). The severity score
    normalises both dimensions to the worst observed stage and averages them, so
    a stage that is mediocre on both ranks above one that is bad on only one.

    Returns
    -------
    pd.DataFrame
        ``stage``, ``dropoff_pct``, ``avg_delay_days``, ``value_lost_usd``,
        ``breaches_dropoff``, ``breaches_delay``, ``is_bottleneck``,
        ``severity_score`` (0-100), ``bottleneck_type``, ``recommended_action``.
    """
    cfg = get_config()
    thresholds = cfg.funnel.bottleneck
    summary = funnel_summary(batches)

    # The first stage has no predecessor, so it cannot have a drop-off.
    analysis = summary.iloc[1:].copy() if len(summary) > 1 else summary.copy()

    max_drop = analysis["dropoff_pct"].max() or 1.0
    max_delay = summary["avg_delay_days"].max() or 1.0
    delay_lookup = summary.set_index("stage")["avg_delay_days"]

    analysis["avg_delay_days"] = analysis["stage"].map(delay_lookup)
    analysis["breaches_dropoff"] = analysis["dropoff_pct"] > thresholds.dropoff_pct_threshold
    analysis["breaches_delay"] = analysis["avg_delay_days"] > thresholds.delay_days_threshold
    analysis["is_bottleneck"] = analysis["breaches_dropoff"] | analysis["breaches_delay"]
    analysis["severity_score"] = (
        50.0 * analysis["dropoff_pct"] / max_drop
        + 50.0 * analysis["avg_delay_days"] / max_delay
    ).round(1)

    analysis["bottleneck_type"] = np.select(
        [analysis["breaches_dropoff"] & analysis["breaches_delay"],
         analysis["breaches_dropoff"], analysis["breaches_delay"]],
        ["Yield + Time", "Yield Loss", "Time Delay"],
        default="Healthy",
    )
    analysis["recommended_action"] = analysis["bottleneck_type"].map({
        "Yield + Time": "Priority 1 - root-cause the stage; it destroys both units and lead time",
        "Yield Loss": "Investigate process capability and upstream input quality",
        "Time Delay": "Target cycle-time reduction - automation or parallel processing",
        "Healthy": "Monitor - performing within tolerance",
    })

    # Rank by how many independent tests a stage fails before ranking by
    # magnitude. A stage that leaks units *and* consumes lead time is a harder
    # constraint than one that merely scores badly on a single dimension, so it
    # must surface as the primary bottleneck.
    analysis["breach_count"] = (
        analysis["breaches_dropoff"].astype(int) + analysis["breaches_delay"].astype(int))

    columns = ["stage", "dropoff_pct", "avg_delay_days", "value_lost_usd",
               "breaches_dropoff", "breaches_delay", "breach_count", "is_bottleneck",
               "severity_score", "bottleneck_type", "recommended_action"]
    result = (analysis[columns]
              .sort_values(["breach_count", "severity_score"], ascending=[False, False])
              .reset_index(drop=True))

    flagged = result.loc[result["is_bottleneck"], "stage"].tolist()
    log.info("Bottleneck stages: %s", flagged or "none")
    return result


# ---------------------------------------------------------------------------
# Segmented views
# ---------------------------------------------------------------------------
def funnel_by_dimension(batches: pd.DataFrame | None = None,
                        dimension: str = "region") -> pd.DataFrame:
    """Compare end-to-end funnel performance across a categorical dimension.

    Parameters
    ----------
    batches
        Optional pre-filtered batch table.
    dimension
        Column to segment by - typically ``region``, ``drug_code``,
        ``supplier_name``, ``warehouse_id`` or ``therapeutic_area``.

    Returns
    -------
    pd.DataFrame
        One row per segment: units in and out, end-to-end yield, QA pass rate,
        cycle time and value lost - sorted by yield ascending so the worst
        performer is the first thing a reader sees.
    """
    cfg = get_config()
    df = _batches(batches)
    if dimension not in df.columns:
        raise KeyError(f"'{dimension}' is not a column of the batch table.")

    unit_cols = cfg.funnel.unit_columns
    first_col, last_col = unit_cols[cfg.funnel.stages[0]], unit_cols[cfg.funnel.stages[-1]]

    grouped = df.groupby(dimension).agg(
        batches=("batch_id", "count"),
        units_in=(first_col, "sum"),
        units_out=(last_col, "sum"),
        value_lost_usd=("value_lost_usd", "sum"),
        avg_cycle_time_days=("total_cycle_time_days", "mean"),
        qa_pass_rate_pct=("qa_pass", "mean"),
    ).reset_index()

    grouped["end_to_end_yield_pct"] = (
        100.0 * grouped["units_out"] / grouped["units_in"].replace(0, np.nan)).round(2)
    grouped["qa_pass_rate_pct"] = (100.0 * grouped["qa_pass_rate_pct"]).round(2)
    grouped["avg_cycle_time_days"] = grouped["avg_cycle_time_days"].round(1)
    grouped["value_lost_usd"] = grouped["value_lost_usd"].round(0)

    return grouped.sort_values("end_to_end_yield_pct").reset_index(drop=True)


def stage_conversion_by_dimension(batches: pd.DataFrame | None = None,
                                  dimension: str = "region") -> pd.DataFrame:
    """Return a segment x stage matrix of cumulative conversion percentages.

    Rendered as a heatmap in the dashboard, this is the fastest way to see
    *which* stage is responsible for a segment's under-performance rather than
    only that the segment under-performs.
    """
    cfg = get_config()
    df = _batches(batches)
    unit_cols = cfg.funnel.unit_columns
    first_col = unit_cols[cfg.funnel.stages[0]]

    rows = []
    for segment, group in df.groupby(dimension):
        base = group[first_col].sum()
        record = {dimension: segment}
        for stage in cfg.funnel.stages:
            record[stage] = round(
                100.0 * group[unit_cols[stage]].sum() / base, 2) if base > 0 else 0.0
        rows.append(record)

    return pd.DataFrame(rows).sort_values(cfg.funnel.stages[-1]).reset_index(drop=True)


def funnel_trend(batches: pd.DataFrame | None = None, period: str = "quarter") -> pd.DataFrame:
    """Track end-to-end yield and cycle time over time.

    Parameters
    ----------
    period
        ``'quarter'``, ``'year_month'`` or ``'year'``.

    Returns
    -------
    pd.DataFrame
        Period-level yield, QA pass rate, cycle time and value lost - used to
        confirm whether an intervention actually moved the trend.
    """
    cfg = get_config()
    df = _batches(batches)
    if period not in df.columns:
        raise KeyError(f"'{period}' is not a column. Use 'quarter', 'year_month' or 'year'.")

    unit_cols = cfg.funnel.unit_columns
    first_col, last_col = unit_cols[cfg.funnel.stages[0]], unit_cols[cfg.funnel.stages[-1]]

    trend = df.groupby(period).agg(
        batches=("batch_id", "count"),
        units_in=(first_col, "sum"),
        units_out=(last_col, "sum"),
        avg_cycle_time_days=("total_cycle_time_days", "mean"),
        qa_pass_rate_pct=("qa_pass", "mean"),
        value_lost_usd=("value_lost_usd", "sum"),
    ).reset_index()

    trend["end_to_end_yield_pct"] = (
        100.0 * trend["units_out"] / trend["units_in"].replace(0, np.nan)).round(2)
    trend["qa_pass_rate_pct"] = (100.0 * trend["qa_pass_rate_pct"]).round(2)
    trend["avg_cycle_time_days"] = trend["avg_cycle_time_days"].round(1)
    return trend.sort_values(period).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loss attribution and headline KPIs
# ---------------------------------------------------------------------------
def loss_attribution(batches: pd.DataFrame | None = None, top_n: int = 15) -> pd.DataFrame:
    """Rank product x region combinations by value destroyed.

    Answers the prioritisation question directly: of everything leaking value,
    which handful of combinations should a team fix first?

    Returns
    -------
    pd.DataFrame
        ``drug_code``, ``brand_name``, ``region``, ``batches``, ``units_lost``,
        ``value_lost_usd``, ``unit_loss_pct``, ``share_of_total_loss_pct``,
        ``cumulative_share_pct`` - the last column supports a Pareto read
        ("the top N combinations account for X% of all losses").
    """
    df = _batches(batches)
    grouped = df.groupby(["drug_code", "brand_name", "region"]).agg(
        batches=("batch_id", "count"),
        units_procured=("units_procured", "sum"),
        units_lost=("units_lost", "sum"),
        value_lost_usd=("value_lost_usd", "sum"),
    ).reset_index()

    grouped["unit_loss_pct"] = (
        100.0 * grouped["units_lost"] / grouped["units_procured"].replace(0, np.nan)).round(2)

    total_loss = grouped["value_lost_usd"].sum()
    grouped = grouped.sort_values("value_lost_usd", ascending=False).reset_index(drop=True)
    grouped["share_of_total_loss_pct"] = (
        100.0 * grouped["value_lost_usd"] / total_loss).round(2) if total_loss > 0 else 0.0
    grouped["cumulative_share_pct"] = grouped["share_of_total_loss_pct"].cumsum().round(2)
    grouped["value_lost_usd"] = grouped["value_lost_usd"].round(0)

    return grouped.head(top_n)


def qa_failure_analysis(batches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Break down quality-testing rejections by documented failure reason.

    Quality Testing is the platform's designed bottleneck, so knowing *why*
    batches fail is what turns the funnel finding into a corrective action.
    """
    df = _batches(batches)
    failed = df[df["qa_pass"] == 0]
    if failed.empty:
        return pd.DataFrame(columns=["qa_fail_reason", "batches", "share_pct", "value_lost_usd"])

    grouped = failed.groupby("qa_fail_reason").agg(
        batches=("batch_id", "count"),
        value_lost_usd=("value_lost_usd", "sum"),
        avg_qa_delay_days=("qa_delay_days", "mean"),
    ).reset_index()

    grouped["share_pct"] = (100.0 * grouped["batches"] / len(failed)).round(2)
    grouped["value_lost_usd"] = grouped["value_lost_usd"].round(0)
    grouped["avg_qa_delay_days"] = grouped["avg_qa_delay_days"].round(1)
    return grouped.sort_values("batches", ascending=False).reset_index(drop=True)


def funnel_kpis(batches: pd.DataFrame | None = None) -> dict[str, float | str]:
    """Headline funnel metrics for the dashboard KPI strip.

    Returns
    -------
    dict
        ``total_batches``, ``units_procured``, ``units_dispensed``,
        ``end_to_end_yield_pct``, ``total_units_lost``, ``total_value_lost_usd``,
        ``avg_cycle_time_days``, ``median_cycle_time_days``, ``qa_pass_rate_pct``,
        ``primary_bottleneck``, ``biggest_dropoff_stage``, ``biggest_dropoff_pct``,
        ``slowest_stage``, ``slowest_stage_days``.
    """
    df = _batches(batches)
    summary = funnel_summary(df)
    bottlenecks = identify_bottlenecks(df)

    # Row 0 of the summary is Procurement, which has no drop-off by definition.
    dropoffs = summary.iloc[1:]
    worst_drop = dropoffs.loc[dropoffs["dropoff_pct"].idxmax()]
    slowest = summary.loc[summary["avg_delay_days"].idxmax()]

    units_in = float(df["units_procured"].sum())
    units_out = float(df["units_dispensed"].sum())

    return {
        "total_batches": int(len(df)),
        "units_procured": units_in,
        "units_dispensed": units_out,
        "end_to_end_yield_pct": round(100.0 * units_out / units_in, 2) if units_in else 0.0,
        "total_units_lost": float(df["units_lost"].sum()),
        "total_value_lost_usd": float(df["value_lost_usd"].sum()),
        "avg_cycle_time_days": round(float(df["total_cycle_time_days"].mean()), 1),
        "median_cycle_time_days": round(float(df["total_cycle_time_days"].median()), 1),
        "qa_pass_rate_pct": round(100.0 * float(df["qa_pass"].mean()), 2),
        "primary_bottleneck": str(bottlenecks.iloc[0]["stage"]),
        "biggest_dropoff_stage": str(worst_drop["stage"]),
        "biggest_dropoff_pct": float(worst_drop["dropoff_pct"]),
        "slowest_stage": str(slowest["stage"]),
        "slowest_stage_days": float(slowest["avg_delay_days"]),
    }


__all__ = [
    "funnel_summary", "stage_delays", "identify_bottlenecks",
    "funnel_by_dimension", "stage_conversion_by_dimension", "funnel_trend",
    "loss_attribution", "qa_failure_analysis", "funnel_kpis",
]
