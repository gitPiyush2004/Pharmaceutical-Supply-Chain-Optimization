"""
Order-to-delivery pipeline analysis on the real USAID SCMS dataset.

Why this is not a unit-attrition funnel
--------------------------------------
The obvious thing to build here is a funnel where units drop stage by stage. **This
dataset cannot support one, and I checked every quantity column before concluding
that.** SCMS records a single ``Line Item Quantity`` per line, stated once at order
time and never restated at delivery. There is no ordered-versus-received pair, no
scrap or short-shipment quantity, and no per-stage weight. A chart showing units
falling between stages would be invented, not measured.

What is genuinely measurable, and used instead:

**1. A value-based service funnel** (:func:`value_funnel`). Every line item is
delivered, so the attrition is not in volume - it is in *timeliness*. Of $1.63bn
ordered, 84.1% arrives on or before its scheduled date and 15.9% arrives late.
That is a real funnel with 100% coverage and a real dollar figure attached.

**2. Lead-time stage decomposition** (in :mod:`src.analytics.procurement`). The
pipeline has four real milestones with real dwell times between them.

**3. Milestone traceability** (:func:`traceability`). How much of the process is
auditable end to end. This is deliberately *not* called a funnel: coverage runs
74% -> 44% -> 100%, which is non-monotone because the dip is a recording artefact
rather than sequential progression.

**4. A cumulative lateness funnel** (:func:`lateness_funnel`), which is monotone by
construction and answers the question a planner actually has: what share of value
arrives within N days of schedule?
"""

from __future__ import annotations

import pandas as pd

from src.analytics.procurement import _resolve
from src.logger import get_logger

log = get_logger(__name__)

#: Day thresholds for the cumulative lateness funnel.
LATENESS_THRESHOLDS = (0, 7, 30, 90)


def value_funnel(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Commodity value flowing through the pipeline, by tightening delivery standard.

    The honest funnel for this dataset: attrition measured in *value delivered to
    schedule* rather than in units lost, because no units are lost.

    Each stage is a strict subset of the one above it, so the funnel is monotone by
    construction. That matters - "delivered early" is *not* a stage here, because
    early arrivals are a subset of on-time ones and drawing them as a fourth band
    would read as further attrition when it is nothing of the kind. Early value is
    reported separately in ``.attrs``, where it belongs: arriving three weeks early
    is a holding-cost problem, not a service failure.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage_order``, ``stage``, ``value_usd``, ``share_pct``,
        ``line_items``, ``basis``. ``.attrs`` carries ``late_value_usd``,
        ``early_value_usd`` and ``exact_day_share_pct``.
    """
    df = _resolve(scms)
    total_value = float(df["line_value_usd"].sum())
    on_time = df[df["is_late"] == 0]
    early = df[df["delivery_delay_days"] < 0]
    exact = df[df["delivery_delay_days"] == 0]

    rows = [
        {"stage_order": 1, "stage": "Value ordered",
         "value_usd": total_value, "line_items": len(df),
         "basis": "All line items, priced at pack price x quantity"},
        {"stage_order": 2, "stage": "Value delivered",
         "value_usd": total_value, "line_items": len(df),
         "basis": "Every line item in this dataset was ultimately delivered"},
        {"stage_order": 3, "stage": "Delivered on or before schedule",
         "value_usd": float(on_time["line_value_usd"].sum()),
         "line_items": len(on_time),
         "basis": "delivery_delay_days <= 0"},
        {"stage_order": 4, "stage": "Delivered exactly on the scheduled day",
         "value_usd": float(exact["line_value_usd"].sum()),
         "line_items": len(exact),
         "basis": "delivery_delay_days == 0 - the tightest reading of on time"},
    ]
    out = pd.DataFrame(rows)
    out["share_pct"] = (100 * out["value_usd"] / total_value).round(2)
    out["value_usd"] = out["value_usd"].round(0)

    late_value = total_value - float(on_time["line_value_usd"].sum())
    log.info("Value funnel: $%.0f ordered, %.1f%% on time, $%.0f late",
             total_value, out.loc[2, "share_pct"], late_value)
    out.attrs["late_value_usd"] = round(late_value, 0)
    out.attrs["early_value_usd"] = round(float(early["line_value_usd"].sum()), 0)
    out.attrs["exact_day_share_pct"] = round(100 * len(exact) / len(df), 2)
    return out


def lateness_funnel(scms: pd.DataFrame | None = None,
                    thresholds: tuple[int, ...] = LATENESS_THRESHOLDS) -> pd.DataFrame:
    """Cumulative share of value arriving within N days of schedule.

    Monotone by construction and 100% covered, so unlike the traceability view this
    reads correctly as a funnel. It answers the planner's actual question: if I can
    tolerate a week of slippage, how much of my value is safe?

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold_days``, ``label``, ``value_usd``, ``share_pct``,
        ``line_items``, ``items_share_pct``.
    """
    df = _resolve(scms)
    total_value = float(df["line_value_usd"].sum())

    rows = []
    for threshold in thresholds:
        within = df[df["delivery_delay_days"] <= threshold]
        label = ("On or before schedule" if threshold == 0
                 else f"Within {threshold} days late")
        rows.append({
            "threshold_days": threshold, "label": label,
            "value_usd": round(float(within["line_value_usd"].sum()), 0),
            "share_pct": round(100 * float(within["line_value_usd"].sum()) / total_value, 2),
            "line_items": len(within),
            "items_share_pct": round(100 * len(within) / len(df), 2),
        })
    # Anything beyond the widest threshold is the irreducible tail.
    rows.append({
        "threshold_days": int(df["delivery_delay_days"].max()),
        "label": "All deliveries (worst case)",
        "value_usd": round(total_value, 0), "share_pct": 100.0,
        "line_items": len(df), "items_share_pct": 100.0,
    })
    return pd.DataFrame(rows)


def traceability(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Milestone recording coverage - how auditable the pipeline is end to end.

    Deliberately not presented as a funnel. Coverage is non-monotone (74% -> 44%
    -> 100%) because the dip reflects a fulfilment route that bypasses vendor
    ordering entirely, not shipments falling out of the process.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage``, ``recorded``, ``coverage_pct``, ``structurally_absent``,
        ``genuinely_missing``, ``interpretation``.
    """
    df = _resolve(scms)
    total = len(df)
    milestones = [
        ("Price quote sent to client", "date_pq_sent"),
        ("Purchase order sent to vendor", "date_po_sent"),
        ("Delivery scheduled", "date_scheduled"),
        ("Delivered to client", "date_delivered"),
        ("Delivery recorded in system", "date_recorded"),
    ]

    rows = []
    for stage, column in milestones:
        reason = df.get(f"{column}_reason")
        structural = int((reason == "structural").sum()) if reason is not None else 0
        missing = int((reason == "missing").sum()) if reason is not None else 0
        recorded = int(df[column].notna().sum())

        if structural and column == "date_po_sent":
            note = (f"{structural:,} fulfilled from regional distribution centre "
                    "stock - no vendor order was ever raised")
        elif structural:
            note = f"{structural:,} predate the current quote process"
        elif missing:
            note = f"{missing:,} genuinely unrecorded"
        else:
            note = "Fully recorded"

        rows.append({"stage": stage, "recorded": recorded,
                     "coverage_pct": round(100 * recorded / total, 2),
                     "structurally_absent": structural,
                     "genuinely_missing": missing, "interpretation": note})
    return pd.DataFrame(rows)


def pipeline_kpis(scms: pd.DataFrame | None = None) -> dict:
    """Headline pipeline figures.

    Returns
    -------
    dict
        Value ordered and delivered late, on-time share, the exact-day
        concentration, and the median quote-to-delivery lead time.
    """
    df = _resolve(scms)
    total_value = float(df["line_value_usd"].sum())
    on_time_value = float(df.loc[df["is_late"] == 0, "line_value_usd"].sum())
    delays = df["delivery_delay_days"]

    return {
        "line_items": int(len(df)),
        "value_ordered_usd": round(total_value, 0),
        "value_on_time_usd": round(on_time_value, 0),
        "value_late_usd": round(total_value - on_time_value, 0),
        "value_on_time_pct": round(100 * on_time_value / total_value, 2),
        "items_on_time_pct": round(100 * float((df["is_late"] == 0).mean()), 2),
        # 61% of shipments land on exactly their scheduled day, which is why the
        # headline on-time rate flatters the underlying planning precision.
        "exact_day_pct": round(100 * float((delays == 0).mean()), 2),
        "early_pct": round(100 * float((delays < 0).mean()), 2),
        "late_pct": round(100 * float((delays > 0).mean()), 2),
        "median_quote_to_delivery_days": round(
            float(df["total_lead_time_days"].median()), 1),
        "p90_quote_to_delivery_days": round(
            float(df["total_lead_time_days"].quantile(0.90)), 1),
        "po_traceability_pct": round(100 * float(df["date_po_sent"].notna().mean()), 2),
    }


__all__ = ["value_funnel", "lateness_funnel", "traceability", "pipeline_kpis",
           "LATENESS_THRESHOLDS"]
