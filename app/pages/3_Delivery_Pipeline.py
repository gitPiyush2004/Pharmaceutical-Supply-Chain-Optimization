"""
Order-to-Delivery Pipeline page - USAID SCMS delivery history.

This page replaces what used to be a unit-attrition funnel. That chart is not
possible on this dataset and the reason is worth stating up front rather than
burying: SCMS records one quantity per line item, set at order time and never
restated at delivery. There is no ordered-versus-received pair, no scrap
quantity, and no per-stage weight. A chart showing units draining away between
stages would have been invented rather than measured.

What replaces it is attrition in *value delivered to schedule*, which is fully
measurable, plus an interval decomposition that shows where the time goes and a
traceability panel that shows how much of the process is auditable at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.analytics import pipeline as pl
from src.analytics import procurement as proc
from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_days, fmt_pct

page_setup(
    title="Order-to-Delivery Pipeline",
    icon="🚚",
    subtitle="Where commodity value arrives late across 10,324 real USAID shipments",
)


@st.cache_data(show_spinner="Loading real SCMS delivery history...")
def _pipeline() -> dict:
    return {
        "kpis": pl.pipeline_kpis(),
        "value": pl.value_funnel(),
        "lateness": pl.lateness_funnel(),
        "trace": pl.traceability(),
        "intervals": proc.lead_time_breakdown(),
        "delays": proc.delay_distribution(),
        "trend": proc.delivery_trend(),
    }


data = _pipeline()
kpis, value, lateness = data["kpis"], data["value"], data["lateness"]

kpi_row([
    {"label": "Commodity Value Ordered",
     "value": fmt_currency(kpis["value_ordered_usd"]),
     "help_text": f"{kpis['line_items']:,} line items at pack price x quantity"},
    {"label": "Value Delivered Late",
     "value": fmt_currency(kpis["value_late_usd"]),
     "delta": f"{100 - kpis['value_on_time_pct']:.1f}% of ordered value",
     "delta_good": False,
     "help_text": "Arrived after the scheduled delivery date"},
    {"label": "Median Quote to Delivery",
     "value": fmt_days(kpis["median_quote_to_delivery_days"]),
     "help_text": f"P90 is {kpis['p90_quote_to_delivery_days']:.0f} days"},
    {"label": "PO Traceability",
     "value": fmt_pct(kpis["po_traceability_pct"]),
     "help_text": "Share of line items with a vendor purchase order on record"},
])

# ---------------------------------------------------------------------------
# Why this is not a volume funnel
# ---------------------------------------------------------------------------
callout(
    "**Why there is no unit-attrition funnel here.** This dataset states "
    "`Line Item Quantity` once, at order time, and never restates it at delivery. "
    "There is no ordered-versus-received pair, no scrap quantity and no per-stage "
    "weight, so units cannot be tracked draining between stages — and every line "
    "item in the file was ultimately delivered. The attrition in this pipeline is "
    "in **timeliness**, not volume, so that is what is measured below.",
    kind="warning", title="A note on what this dataset can support",
)

# ---------------------------------------------------------------------------
# Value funnel
# ---------------------------------------------------------------------------
section(
    "Value Through the Pipeline",
    "Each band is a strictly tighter definition of on time than the one above it, "
    "so the funnel is monotone by construction rather than by coincidence.",
)

left, right = st.columns([1, 1], gap="large")
with left:
    chart(charts.value_funnel_chart(value))
with right:
    chart(charts.lateness_funnel_chart(lateness))

insight(
    f"**{fmt_currency(kpis['value_late_usd'])} "
    f"({100 - kpis['value_on_time_pct']:.1f}% of ordered value) arrives late, but "
    f"the tail is short.** {lateness.loc[1, 'share_pct']:.1f}% of value is within a "
    f"week of schedule and {lateness.loc[2, 'share_pct']:.1f}% within a month — a "
    f"large number of small slips, not a handful of disasters. That matters "
    f"operationally: a week of buffer stock absorbs most of this exposure, whereas "
    f"a few catastrophic delays would need a different fix entirely.\n\n"
    f"Two figures qualify the headline. "
    f"**{kpis['exact_day_pct']:.0f}% of shipments land on exactly their scheduled "
    f"day**, which is implausibly precise for international freight and suggests the "
    f"scheduled date is sometimes back-filled from the actual one. And "
    f"**{kpis['early_pct']:.0f}% arrive early** — counted as a success by the "
    f"on-time metric, but early arrivals carry holding cost and, for cold-chain "
    f"product, extra storage risk. Only "
    f"{value.loc[3, 'share_pct']:.1f}% of value arrives on precisely the day it was "
    f"promised."
)

# ---------------------------------------------------------------------------
# Where the time goes
# ---------------------------------------------------------------------------
section(
    "Where The Time Goes",
    "Each interval is measured on a different number of line items, so the "
    "denominator is drawn on the bar rather than left in a footnote.",
)

chart(charts.dwell_time_chart(data["intervals"]))
show_table(data["intervals"])

quote_to_delivery = data["intervals"].iloc[0]
po_to_delivery = data["intervals"].iloc[1]
insight(
    f"The full pipeline runs a median **{quote_to_delivery['median_days']:.0f} days "
    f"from price quote to delivery**, but that is measured on only "
    f"{quote_to_delivery['coverage_pct']:.0f}% of line items — the rest predate the "
    f"current quote process. The vendor-purchase-order interval looks far shorter at "
    f"{po_to_delivery['median_days']:.0f} days, and it would be a mistake to read "
    f"that as a faster route: it exists **only** for direct-drop orders "
    f"({po_to_delivery['coverage_pct']:.0f}% of items), because stock shipped from a "
    f"regional distribution centre never has a vendor order raised against it. The "
    f"two intervals are measured on different populations and are not comparable."
)

# ---------------------------------------------------------------------------
# Delay distribution
# ---------------------------------------------------------------------------
section(
    "How Late Is Late",
    "The distribution of delivery delay against schedule. Negative is early.",
)

col_a, col_b = st.columns([3, 2], gap="large")
with col_a:
    chart(charts.bar_chart(
        data["delays"], x="bucket", y="shipments",
        title="Shipments by Delay Against Schedule"))
with col_b:
    show_table(data["delays"])

# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
section(
    "On-Time Performance Over Time",
    "Quarterly on-time rate across the full 2006-2015 window.",
)

chart(charts.line_chart(
    data["trend"], x="delivery_quarter", y="on_time_pct",
    title="On-Time Delivery Rate by Quarter"))

insight(
    "Performance is **not stable across the window**, and that single fact governs "
    "how everything else on this dashboard should be read. The programme ran at "
    "roughly 94% on time through 2010 and roughly 85% afterwards. Any comparison "
    "that pools the whole decade is therefore averaging two different operating "
    "regimes — which is exactly the trap the Statistical Testing page unpicks for "
    "the fulfilment-route comparison."
)

# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------
section(
    "Milestone Traceability",
    "How much of the process is auditable end to end — and the difference between "
    "a record that is missing and one that never existed.",
)

chart(charts.traceability_chart(data["trace"]))
show_table(data["trace"])

insight(
    "**Coverage is not monotone (74% → 44% → 100%), which is why this is not drawn "
    "as a funnel.** The dip at the purchase-order milestone is not shipments "
    "dropping out of the process: 5,404 line items are filled from regional "
    "distribution centre stock, so no vendor order was ever raised. That value is "
    "*structurally absent*, not missing.\n\n"
    "The distinction is the whole point. A generic imputer filling those blanks "
    "would have manufactured 5,404 purchase orders that never existed, and every "
    "lead-time figure downstream would have been wrong. Only 533 dates across all "
    "five milestones are genuinely unrecorded."
)

download_button(data["intervals"], "pipeline_intervals.csv",
                label="Download interval breakdown")

methodology("""
**Value basis.** Commodity value is `Pack Price x Line Item Quantity`, which
reconciles with the published `Line Item Value` on 9,685 of 10,324 rows. Freight is
excluded from the value figures because 2,445 rows carry it as a cross-reference to
another shipment rather than as a number.

**On time.** `Delivered to Client Date <= Scheduled Delivery Date`. No grace period
is applied, because the dataset already concentrates 61% of deliveries on exactly
the scheduled day and any grace window would push the rate toward a meaningless
100%.

**Date parsing.** The five milestone columns use two different formats
(`%m/%d/%y` for the quote and purchase-order dates, `%d-%b-%y` for the three
delivery dates). Parsing them all with a single `dayfirst` rule silently produced
478 negative lead times and 1,128 purchase orders that appeared to precede their
own quote. Each column is now parsed with its own explicit format, and
`src/data/scms.py` carries a regression test for it.
""")

sidebar_about()
