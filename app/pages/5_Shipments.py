"""
Shipment Analytics page.

Service performance across the three physical transport legs: who ships late,
which lanes and modes are unreliable, and what the failures cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.dashboard.components import (callout, chart, download_button, insight, kpi_row,
                        methodology, page_setup, section, show_table,
                        sidebar_about, sidebar_filters)
from src.analytics import shipments as shp
from src.data import loader
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_days, fmt_pct, fmt_units

page_setup(
    title="Shipment & Distribution Analytics",
    icon="🚚",
    subtitle="Supplier, carrier, mode and regional service performance across three legs",
)


@st.cache_data(show_spinner="Loading shipment legs...")
def _shipments():
    return loader.load_shipments()


data = sidebar_filters(
    _shipments(),
    ["region", "leg", "transport_mode", "carrier", "supplier_name", "drug_code"],
)
if data.empty:
    st.stop()

kpis = shp.shipment_kpis(data)

section("Service Performance")
kpi_row([
    {"label": "On-Time Delivery", "value": fmt_pct(kpis["on_time_delivery_pct"]),
     "delta": f"{kpis['on_time_delivery_pct'] - kpis['otif_target_pct']:+.1f} pp vs target",
     "delta_good": bool(kpis["meets_otif_target"]),
     "help_text": f"Target {kpis['otif_target_pct']:.0f}%"},
    {"label": "Late Shipments", "value": f"{kpis['late_shipments']:,}",
     "help_text": f"of {kpis['total_shipments']:,} legs"},
    {"label": "Avg Delay", "value": fmt_days(kpis["avg_delay_days"], 2),
     "help_text": "Actual minus planned transit"},
    {"label": "SLA Cost", "value": fmt_currency(kpis["late_shipment_cost_usd"]),
     "help_text": "Late-delivery penalty exposure"},
])

kpi_row([
    {"label": "Freight Spend", "value": fmt_currency(kpis["total_freight_spend_usd"])},
    {"label": "Cost per Unit", "value": f"${kpis['cost_per_unit_usd']:.3f}"},
    {"label": "Temperature Excursions", "value": fmt_pct(kpis["excursion_rate_pct"]),
     "help_text": "Legs with a recorded excursion"},
    {"label": "Damage Rate", "value": fmt_pct(kpis["damage_rate_pct"]),
     "help_text": "Units lost in transit"},
])

gap = kpis["otif_target_pct"] - kpis["on_time_delivery_pct"]
callout(
    f"On-time delivery of {fmt_pct(kpis['on_time_delivery_pct'])} sits "
    f"{gap:+.1f} points against the {kpis['otif_target_pct']:.0f}% contractual "
    f"target, carrying {fmt_currency(kpis['late_shipment_cost_usd'])} of SLA "
    "exposure. The regional and mode breakdowns below show the shortfall is not "
    "uniform - it concentrates in specific lanes, which makes it fixable without "
    "renegotiating the whole carrier base.",
    kind="warning" if gap > 0 else "success", title="Against target",
)

# ---------------------------------------------------------------------------
section(
    "Supplier Scorecard",
    "Ranked worst-first. Contracted OTIF is compared against measured performance, "
    "so a supplier failing its own commitment is visible immediately.",
)

suppliers = shp.supplier_performance(data)
chart(charts.bar_chart(suppliers.sort_values("on_time_pct"),
                       x="supplier_name", y="on_time_pct",
                       title="On-Time Delivery by Supplier (%)",
                       text_format=".1f", height=380))
show_table(
    suppliers[["rank", "supplier_name", "country", "shipments", "on_time_pct",
               "avg_delay_days", "damage_rate_pct", "excursion_rate_pct",
               "performance_tier", "contracted_otif_pct", "tier_matches_contract",
               "freight_spend_usd"]],
    height=340,
)
download_button(suppliers, "supplier_scorecard.csv", "Download supplier scorecard")

breaching = suppliers[~suppliers["meets_otif_target"]]
worst = suppliers.iloc[0]
insight(
    f"**{worst['supplier_name']}** ({worst['country']}) is the weakest performer at "
    f"{fmt_pct(worst['on_time_pct'])} on-time against a contracted "
    f"{fmt_pct(worst['contracted_otif_pct'])}, across {int(worst['shipments']):,} "
    f"legs. {len(breaching)} of {len(suppliers)} suppliers miss the "
    f"{kpis['otif_target_pct']:.0f}% network target. Cross-referencing with the "
    "funnel page, the same suppliers with weak reliability scores are the ones "
    "driving QA rejections - so supplier consolidation improves quality yield and "
    "delivery reliability at the same time."
)

# ---------------------------------------------------------------------------
section("Regional and Carrier Performance")

tab1, tab2, tab3, tab4 = st.tabs(
    ["By Region", "By Carrier", "Transit Times", "Late Shipment Analysis"])

with tab1:
    regional = shp.regional_performance(data)
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        chart(charts.bar_chart(regional.sort_values("on_time_pct"),
                               x="region", y="on_time_pct",
                               title="On-Time Delivery by Region (%)",
                               text_format=".1f", height=360))
    with col2:
        show_table(regional[["region", "shipments", "on_time_pct", "otif_gap_pct",
                             "avg_delay_days", "excursion_rate_pct"]], height=340)
    weakest = regional.iloc[0]
    insight(
        f"**{weakest['region']}** is the weakest region at "
        f"{fmt_pct(weakest['on_time_pct'])} on-time "
        f"({weakest['otif_gap_pct']:+.1f} points against target), with an average "
        f"delay of {weakest['avg_delay_days']:.2f} days. The driver is transport "
        "mix rather than carrier quality: regions leaning on long road and sea legs "
        "absorb more disruption than regions with air capacity."
    )

with tab2:
    carriers = shp.carrier_scorecard(data)
    chart(charts.scatter_chart(carriers, x="cost_per_unit_usd", y="on_time_pct",
                               color="transport_mode", size="shipments",
                               title="Cost versus Reliability by Carrier and Mode",
                               height=440))
    show_table(carriers.sort_values("on_time_pct"), height=340)
    insight(
        "The cost/reliability scatter is the sourcing decision in one view. Points "
        "to the upper-left are the efficient frontier: high on-time performance at "
        "low cost per unit. Air lanes buy reliability at roughly 2-3x the unit cost "
        "of road, which is defensible for short-shelf-life cold-chain product and "
        "hard to justify for long-dated tablets."
    )

with tab3:
    transit = shp.transit_time_distribution(data)
    show_table(transit, height=240)
    chart(charts.bar_chart(transit, x="transport_mode", y="mean_variance_days",
                           title="Mean Transit Variance by Mode (actual minus planned, days)",
                           text_format=".2f", height=340))
    chart(charts.box_chart(data, x="transport_mode", y="actual_transit_days",
                           title="Actual Transit Time Distribution by Mode",
                           height=400))
    insight(
        "Planned versus actual variance, not absolute transit time, is what breaks "
        "a supply plan. A 22-day sea leg that always takes 22 days is easy to plan "
        "around; a 6-day road leg that sometimes takes 14 is not. The p90 variance "
        "column is the number replenishment lead times should actually be set from."
    )

with tab4:
    late = shp.late_shipment_analysis(data)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**By transport mode**")
        show_table(late["by_mode"], height=220)
        st.markdown("**By shipment leg**")
        show_table(late["by_leg"], height=200)
    with c2:
        st.markdown("**By region**")
        show_table(late["by_region"], height=220)
        st.markdown("**Cost impact**")
        show_table(late["cost_impact"], height=200)

    trend = late["monthly_trend"]
    chart(charts.line_chart(trend, x="year_month", y="on_time_pct",
                            title="On-Time Delivery Trend by Month (%)",
                            y_title="On-time %", height=380))
    download_button(trend, "on_time_trend.csv", "Download monthly trend")

# ---------------------------------------------------------------------------
methodology("""
**Three legs per batch.** Every batch generates three shipment records: plant to
warehouse, warehouse to distributor, and distributor to pharmacy. Metrics are
computed per leg, so a network-level on-time figure is the share of *legs*
delivered on time, not of batches.

**On-time definition.** A leg is late when `actual_transit_days > planned_transit_days
+ shipments.on_time_grace_days` (grace = 0, i.e. strict). Planned transit is set by
mode: air 3 days, road 6, rail 9, sea 22.

**Supplier tiering.** Measured on-time performance is banded using
`shipments.supplier_tiers` (Preferred ≥92%, Approved ≥85%, Watchlist below).
`tier_matches_contract` compares that measured tier against the tier the supplier
is contracted at - a mismatch is a commercial conversation.

**Cost of failure.** `economics.late_shipment_penalty` ($1,850 per late leg)
approximates SLA credits and expedite costs. Freight cost is modelled per unit by
mode, with a cold-chain multiplier, so cost per unit is comparable across lanes.

**Temperature excursions** are recorded per leg and are more likely on cold-chain
freight and on legs that run long - the interaction that makes delay a *quality*
risk, not only a service risk. The Drug Stability page quantifies the potency
consequence.
""")

sidebar_about()
