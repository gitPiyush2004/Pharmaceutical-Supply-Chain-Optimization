"""
Vendor & Logistics page - USAID SCMS delivery history.

Who delivers, where to, by what mode, and at what freight cost. 10,324 shipments
of HIV, malaria and antiretroviral commodities to 43 countries between 2006 and
2015, published by USAID as open data.

The pipeline-level view of the same dataset - the value funnel, interval
decomposition and milestone traceability - lives on the Delivery Pipeline page.
This page is about the actors rather than the process: vendor scorecards,
destination performance, and the cost-versus-service trade-off between transport
modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about, sidebar_filters)
from src.analytics import procurement as proc
from src.data import loader
from src.data.scms import parsing_report, scms_provenance
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_days, fmt_pct, fmt_units

page_setup(
    title="Vendor & Logistics Performance",
    icon="🌍",
    subtitle="Who delivers, where to, by what mode, and at what freight cost "
             "- across 10,324 real USAID shipments",
)


@st.cache_data(show_spinner="Loading real SCMS delivery history...")
def _scms():
    return loader.load_scms()


@st.cache_data(show_spinner=False)
def _provenance():
    return scms_provenance(), parsing_report()


provenance, parsing = _provenance()

callout(
    f"**{provenance['name']}** — published by {provenance['publisher']} under the "
    f"{provenance['programme']} programme. {provenance['records']:,} line items, "
    f"{provenance['coverage']}. Licence: {provenance['licence']}. "
    "Every figure on this page is counted from these records.",
    kind="success", title="Real operational data",
)

data = sidebar_filters(
    _scms(),
    ["region", "shipment_mode", "product_group", "fulfil_via", "managed_by"],
)
if data.empty:
    st.stop()

kpis = proc.scms_kpis(data)

# ---------------------------------------------------------------------------
section("Network Performance")
kpi_row([
    {"label": "Shipments", "value": f"{kpis['shipments']:,}",
     "help_text": f"{kpis['date_from']} to {kpis['date_to']}"},
    {"label": "On-Time Delivery", "value": fmt_pct(kpis["on_time_delivery_pct"]),
     "delta": f"{kpis['on_time_delivery_pct'] - kpis['on_time_target_pct']:+.2f} pp "
              f"vs {kpis['on_time_target_pct']:.0f}% target",
     "delta_good": bool(kpis["meets_on_time_target"]),
     "help_text": f"{kpis['late_shipments']:,} late"},
    {"label": "Commodity Value",
     "value": fmt_currency(kpis["total_commodity_value_usd"]),
     "help_text": f"{fmt_units(kpis['total_units'])} units "
                  f"({fmt_units(kpis['total_packs'])} packs)"},
    {"label": "Freight Spend",
     "value": fmt_currency(kpis["total_freight_spend_usd"]),
     "help_text": f"Median {kpis['median_freight_pct_of_value']:.1f}% of commodity value"},
])

kpi_row([
    {"label": "Countries", "value": f"{kpis['countries']}"},
    {"label": "Vendors", "value": f"{kpis['vendors']}",
     "help_text": f"{kpis['manufacturing_sites']} manufacturing sites"},
    {"label": "Median Vendor Lead Time",
     "value": fmt_days(kpis["median_vendor_lead_time_days"], 0),
     "help_text": f"Measurable on {kpis['vendor_lead_time_coverage_pct']:.0f}% of line items"},
    {"label": "Median Delivery Variance",
     "value": fmt_days(kpis["median_delay_days"], 0),
     "help_text": f"Mean {kpis['avg_delay_days']:+.1f} days (negative = early)"},
])

# ---------------------------------------------------------------------------
section(
    "Delivery Timing",
    "A single on-time percentage hides the shape of the distribution. Most "
    "shipments land exactly on their scheduled date, and more arrive early than late.",
)

distribution = proc.delay_distribution(data)
col1, col2 = st.columns([3, 2], gap="large")
with col1:
    chart(charts.bar_chart(distribution, x="bucket", y="shipments",
                           title="Shipments by Delivery Timing", height=380))
with col2:
    show_table(distribution, height=300)

on_day = distribution.loc[distribution["bucket"] == "On the scheduled day",
                          "share_pct"].iloc[0]
early = distribution[distribution["bucket"].str.contains("early")]["share_pct"].sum()
insight(
    f"**{on_day:.0f}% of shipments arrive on exactly their scheduled date** and a "
    f"further **{early:.0f}% arrive early** — only {100 - on_day - early:.0f}% are "
    "late. A concentration this sharp on the exact scheduled day suggests the "
    "schedule is often set conservatively, or reconciled to the actual arrival "
    "after the fact. Either way it means the headline on-time figure overstates "
    "planning precision: the metric to manage is the *late tail*, not the average."
)

# ---------------------------------------------------------------------------
callout(
    "**Milestone traceability and the value funnel are on the Delivery Pipeline "
    "page**, so they are not repeated here. The one figure worth carrying across: "
    "only 44% of line items have a vendor purchase order on record, because the rest "
    "are filled from regional distribution centre stock that never involves a vendor "
    "order. That is why the vendor lead-time column below reports its own "
    "denominator rather than a single programme-wide average.",
    kind="insight", title="Where the process view lives",
)

# ---------------------------------------------------------------------------
section("Where Performance Varies")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Transport Mode", "Vendors", "Regions & Countries", "Commodities", "Trend"])

with tab1:
    modes = proc.mode_performance(data)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        chart(charts.bar_chart(modes, x="shipment_mode", y="on_time_pct",
                               title="On-Time Delivery by Mode (%)",
                               text_format=".1f", height=360))
    with col2:
        chart(charts.bar_chart(modes.dropna(subset=["median_freight_per_kg_usd"]),
                               x="shipment_mode", y="median_freight_per_kg_usd",
                               title="Median Freight Cost per Kilogram (USD)",
                               text_format=".2f", height=360))
    show_table(modes[["shipment_mode", "shipments", "on_time_pct", "avg_delay_days",
                      "median_freight_per_kg_usd", "median_freight_pct",
                      "freight_cost_usd"]], height=240)
    insight(
        "This is the cost-versus-service frontier on real data. **Ocean freight is "
        "the cheapest per kilogram and the least reliable; air is roughly six times "
        "the unit cost and the most reliable.** The decision is therefore not "
        "'which mode is best' but which commodities justify paying for reliability "
        "— short-dated or stock-out-critical products do, bulk resupply of "
        "long-dated product does not. The `Unknown` row is 360 shipments with no "
        "recorded mode; its unusually high on-time rate suggests those records were "
        "reconciled retrospectively, so it should not be read as a real lane."
    )

with tab2:
    vendors = proc.vendor_scorecard(data)
    chart(charts.bar_chart(vendors.head(15), x="vendor", y="on_time_pct",
                           title="Lowest On-Time Delivery by Vendor (%)",
                           orientation="h", text_format=".1f", height=460))
    show_table(vendors[["rank", "vendor", "shipments", "on_time_pct",
                        "avg_delay_days", "p90_delay_days", "performance_tier",
                        "line_value_usd", "median_freight_pct"]], height=340)
    download_button(vendors, "scms_vendor_scorecard.csv", "Download vendor scorecard")

    worst = vendors.iloc[0]
    insight(
        f"The weakest performer is **{worst['vendor']}** at "
        f"{fmt_pct(worst['on_time_pct'])} on-time across "
        f"{int(worst['shipments']):,} shipments carrying "
        f"{fmt_currency(worst['line_value_usd'])} of commodity value. That it is the "
        "regional distribution centre channel rather than a manufacturer is the "
        "important part: the weakest link is an internal fulfilment route, not an "
        "external supplier — which makes it directly fixable without renegotiating "
        "a single contract. Vendors below 30 shipments are excluded from this "
        "ranking, because one late delivery out of three would otherwise outrank a "
        "supplier with hundreds of shipments."
    )

with tab3:
    regions = proc.region_performance(data)
    chart(charts.bar_chart(regions, x="region", y="on_time_pct",
                           title="On-Time Delivery by Region (%)",
                           text_format=".1f", height=360))
    show_table(regions[["region", "shipments", "on_time_pct", "target_gap_pct",
                        "avg_delay_days", "line_value_usd"]], height=240)

    st.markdown("**Country detail** (30+ shipments)")
    countries = proc.country_performance(data)
    show_table(countries[["country", "region", "shipments", "on_time_pct",
                          "avg_delay_days", "line_value_usd"]], height=320)
    download_button(countries, "scms_country_performance.csv", "Download country detail")

with tab4:
    products = proc.product_performance(data)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        chart(charts.bar_chart(products, x="product_group_name", y="line_value_usd",
                               title="Commodity Value by Product Group (USD)",
                               orientation="h", height=340))
    with col2:
        chart(charts.bar_chart(products, x="product_group_name", y="on_time_pct",
                               title="On-Time Delivery by Product Group (%)",
                               orientation="h", text_format=".1f", height=340))
    show_table(products[["product_group_name", "shipments", "on_time_pct",
                         "line_value_usd", "packs", "units", "median_freight_pct"]],
               height=240)

    st.markdown("**Manufacturing site reliability** (30+ shipments)")
    sites = proc.manufacturing_site_analysis(data)
    show_table(sites[["manufacturing_site", "shipments", "on_time_pct",
                      "avg_delay_days", "line_value_usd"]].head(20), height=300)

with tab5:
    trend = proc.delivery_trend(data, period="quarter")
    chart(charts.line_chart(trend, x="delivery_quarter", y="on_time_pct",
                            title="On-Time Delivery by Quarter (%)",
                            y_title="On-time %", height=380))
    chart(charts.line_chart(trend, x="delivery_quarter", y="line_value_usd",
                            title="Commodity Value Delivered by Quarter (USD)",
                            y_title="USD", height=340))
    show_table(trend, height=280)

# ---------------------------------------------------------------------------
section(
    "Freight Economics",
    "Median freight cost per kilogram and as a share of commodity value, by mode "
    "and region.",
)
freight = proc.freight_economics(data)
chart(charts.heatmap(
    freight.pivot_table(index="shipment_mode", columns="region",
                        values="median_freight_per_kg", aggfunc="median"),
    title="Median Freight Cost per Kilogram (USD)", text_format=".2f"))
show_table(freight, height=300)

# ---------------------------------------------------------------------------
section(
    "Data Provenance",
    "How every ambiguous field in the raw export was resolved. This is the audit "
    "trail between the published file and the numbers above.",
)

show_table(parsing, height=280)
callout(
    "The distinction that matters here is **structural absence versus missing "
    "data**. `N/A - From RDC` in the purchase order column is not a gap — it "
    "correctly records that no vendor order existed. `Freight Included in "
    "Commodity Cost` is not a missing price — it says the cost sits in another "
    "line. Collapsing all of these into 'null' and imputing them would fabricate "
    "purchase orders and freight charges. Each is given its own reason code and "
    "excluded from the relevant statistic instead.",
    kind="insight", title="Why the reason codes exist",
)

with st.expander("Source and citation"):
    st.markdown(f"""
- **Dataset:** {provenance['name']}
- **Publisher:** {provenance['publisher']}
- **Programme:** {provenance['programme']}
- **Description:** {provenance['description']}
- **Coverage:** {provenance['coverage']}
- **Licence:** {provenance['licence']}
- **Kaggle mirror:** {provenance['kaggle']}
""")

methodology("""
**What this dataset is.** The USAID Supply Chain Management System delivery
history: every line item shipped under the programme between 2006 and 2015,
covering antiretrovirals, HIV rapid diagnostic tests, artemisinin combination
therapies and anti-malarials. It is published as US Government open data.

**On-time definition.** A line item is late when `Delivered to Client Date` falls
after `Scheduled Delivery Date`. Delivering early counts as on time, which is why
the mean delay is negative while 11.5% of shipments are late.

**Small-sample exclusion.** Vendors, countries and manufacturing sites with fewer
than `scms.min_shipments_for_scoring` (30) shipments are omitted from the
rankings. Ranking a supplier with three shipments against one with five hundred
produces confident nonsense.

**Medians, not means, for freight ratios.** Freight as a share of commodity value
averages over 2,500% on this data, because some line values are a few dollars
while their shipping is not. The median of about 11% is the number that describes
a typical shipment, so every freight ratio here is a median.

**Structural absence.** 5,404 line items were fulfilled from regional distribution
centre stock and never had a vendor purchase order; 2,476 predate the price-quote
process. These are excluded from the affected statistics rather than imputed, and
the reason codes in the provenance table above record every such decision.

**What this dataset does not cover.** SCMS records procurement and logistics, not
manufacturing. It has no batch quality outcomes, no storage temperature or humidity
telemetry and no inventory snapshots — so there is no stability or batch-risk
analysis anywhere in this project. No public dataset carries per-batch storage
telemetry (Kaggle, openFDA, data.gov.in, CDSCO, Mendeley and Zenodo were all
checked; the one promising candidate turned out to be a simulation), so rather than
generate it, those analyses were dropped.
""")

sidebar_about()
