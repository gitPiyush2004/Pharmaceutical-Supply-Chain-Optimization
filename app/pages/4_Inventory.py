"""
Inventory Analytics page.

Working capital versus service level: how much stock is held, how fast it turns,
where it is about to run out, and where it is about to expire.
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
from src.analytics import inventory
from src.data import loader
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct, fmt_units

page_setup(
    title="Inventory Analytics",
    icon="📦",
    subtitle="Turnover, ABC classification, and stock-out, overstock and expiry risk",
)


@st.cache_data(show_spinner="Loading inventory snapshots...")
def _inventory():
    return loader.load_inventory()


inv = sidebar_filters(_inventory(), ["region", "warehouse_name", "drug_code"])
if inv.empty:
    st.stop()

kpis = inventory.inventory_kpis(inv)

section("Inventory Position")
kpi_row([
    {"label": "Inventory Value", "value": fmt_currency(kpis["total_inventory_value_usd"]),
     "help_text": f"{kpis['positions_analysed']:,} warehouse-product positions"},
    {"label": "Turnover", "value": f"{kpis['avg_turnover']:.2f}x",
     "delta": f"{kpis['avg_turnover'] - kpis['target_turnover']:+.2f} vs target",
     "delta_good": kpis["avg_turnover"] >= kpis["target_turnover"],
     "help_text": f"Target {kpis['target_turnover']:.0f}x annually"},
    {"label": "Months of Supply", "value": f"{kpis['avg_months_of_supply']:.2f}",
     "help_text": "Average cover across positions"},
    {"label": "SKUs at Stock-Out Risk", "value": fmt_pct(kpis["pct_skus_at_stockout_risk"]),
     "help_text": fmt_currency(kpis["stockout_exposure_usd"]) + " exposure"},
])

kpi_row([
    {"label": "Expiry Exposure", "value": fmt_currency(kpis["expiry_exposure_usd"]),
     "help_text": "Net write-off value after salvage"},
    {"label": "Overstock Value", "value": fmt_currency(kpis["overstock_value_usd"]),
     "help_text": "Capital tied up above target cover"},
    {"label": "Excess Holding Cost", "value": fmt_currency(kpis["excess_holding_cost_usd"]),
     "help_text": "Annual carrying cost of the excess"},
    {"label": "Latest Snapshot", "value": str(kpis["latest_snapshot_month"]),
     "help_text": "Month used for point-in-time views"},
])

callout(
    f"Turnover of {kpis['avg_turnover']:.2f}x against a {kpis['target_turnover']:.0f}x "
    f"target means stock sits roughly {365 / max(kpis['avg_turnover'], 0.01):.0f} days "
    f"on average. Combined with {fmt_currency(kpis['overstock_value_usd'])} of "
    f"overstock and {fmt_currency(kpis['expiry_exposure_usd'])} of expiry exposure, "
    "the network is carrying too much of the wrong stock rather than too little "
    "overall - a reallocation problem, not a purchasing problem.",
    kind="warning" if kpis["avg_turnover"] < kpis["target_turnover"] else "success",
    title="The headline read",
)

# ---------------------------------------------------------------------------
section(
    "ABC Classification",
    "Products ranked by share of annual inventory value. Class A items justify "
    "tight control and frequent counting; class C items do not.",
)

abc = inventory.abc_analysis(inv)
col1, col2 = st.columns([3, 2], gap="large")
with col1:
    chart(charts.pareto_chart(abc, category="brand_name", value="annual_value_usd",
                              cumulative="cumulative_share_pct",
                              title="Inventory Value Concentration"))
with col2:
    show_table(abc[["drug_code", "brand_name", "annual_value_usd",
                    "value_share_pct", "cumulative_share_pct", "abc_class"]],
               height=340)

a_items = abc[abc["abc_class"] == "A"]
if len(a_items):
    insight(
        f"**{len(a_items)} of {len(abc)} products** are class A, carrying "
        f"**{a_items['value_share_pct'].sum():.0f}%** of inventory value "
        f"({', '.join(a_items['brand_name'])}). These are where cycle-count "
        "frequency, safety-stock tuning and supplier attention should concentrate. "
        "Class C items can move to periodic review without material risk."
    )

# ---------------------------------------------------------------------------
section("Turnover by Product and Warehouse")

turnover = inventory.inventory_turnover(inv)
chart(charts.bar_chart(
    turnover.groupby("brand_name", as_index=False)["turnover"].mean()
            .sort_values("turnover"),
    x="brand_name", y="turnover", title="Average Annual Turnover by Product",
    text_format=".2f", height=360))
show_table(turnover[["brand_name", "warehouse_name", "region", "avg_units_on_hand",
                     "annual_units_issued", "turnover", "days_of_inventory",
                     "turnover_gap", "status"]], height=320)
download_button(turnover, "inventory_turnover.csv", "Download turnover detail")

# ---------------------------------------------------------------------------
section("Risk Registers", "The three ways inventory destroys value.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Stock-Out Risk", "Overstock Risk", "Expiry Risk", "Warehouse Utilisation"])

with tab1:
    risk = inventory.stockout_risk(inv)
    if len(risk):
        high = risk[risk["risk_level"] == "High"]
        kpi_row([
            {"label": "Positions at Risk", "value": f"{len(risk):,}"},
            {"label": "High Risk", "value": f"{len(high):,}"},
            {"label": "Shortfall Units", "value": fmt_units(risk["shortfall_units"].sum())},
            {"label": "Cost Exposure",
             "value": fmt_currency(risk["stockout_cost_usd"].sum())},
        ])
        chart(charts.bar_chart(
            risk.groupby("risk_level", as_index=False)
                .agg(positions=("drug_code", "size")),
            x="risk_level", y="positions", title="Positions by Stock-Out Risk Level",
            height=320))
        show_table(risk.nlargest(30, "stockout_cost_usd")[
            ["snapshot_month", "brand_name", "warehouse_name", "units_on_hand",
             "reorder_point", "risk_ratio", "risk_level", "shortfall_units",
             "stockout_cost_usd"]], height=320)
        download_button(risk, "stockout_risk.csv", "Download stock-out register")
    else:
        st.success("No positions below the stock-out threshold.")

with tab2:
    over = inventory.overstock_risk(inv)
    if len(over):
        kpi_row([
            {"label": "Overstocked Positions", "value": f"{len(over):,}"},
            {"label": "Excess Units", "value": fmt_units(over["excess_units"].sum())},
            {"label": "Excess Value", "value": fmt_currency(over["excess_value_usd"].sum())},
            {"label": "Annual Holding Cost",
             "value": fmt_currency(over["holding_cost_usd"].sum())},
        ])
        chart(charts.bar_chart(
            over.groupby("brand_name", as_index=False)["excess_value_usd"].sum()
                .sort_values("excess_value_usd", ascending=False),
            x="brand_name", y="excess_value_usd",
            title="Excess Inventory Value by Product", height=340))
        show_table(over.nlargest(30, "excess_value_usd"), height=300)
    else:
        st.success("No positions above the overstock threshold.")

with tab3:
    expiry = inventory.expiry_risk(inv)
    if len(expiry):
        kpi_row([
            {"label": "Positions at Risk", "value": f"{len(expiry):,}"},
            {"label": "Units Expiring",
             "value": fmt_units(expiry["units_expiring_soon"].sum())},
            {"label": "Net Write-Off",
             "value": fmt_currency(expiry["write_off_value_usd"].sum())},
            {"label": "Avg Days to Expiry",
             "value": f"{expiry['days_to_expiry'].mean():.0f} d"},
        ])
        chart(charts.bar_chart(
            expiry.groupby("brand_name", as_index=False)["write_off_value_usd"].sum()
                  .sort_values("write_off_value_usd", ascending=False),
            x="brand_name", y="write_off_value_usd",
            title="Expiry Write-Off Exposure by Product", height=340))
        show_table(expiry.nlargest(30, "write_off_value_usd"), height=300)
        insight(
            "Expiry exposure concentrates in the short-shelf-life cold-chain "
            "products. Those are exactly the items being over-held relative to "
            "their remaining life, so the fix is shorter replenishment cycles on "
            "cold chain rather than a blanket reduction in cover."
        )
    else:
        st.success("No positions inside the expiry risk horizon.")

with tab4:
    util = inventory.warehouse_utilisation(inv)
    chart(charts.bar_chart(util.sort_values("utilisation_pct", ascending=False),
                           x="warehouse_name", y="utilisation_pct",
                           title="Warehouse Utilisation (%)",
                           text_format=".1f", height=360))
    show_table(util, height=280)
    constrained = util[util["status"] == "Constrained"]
    under = util[util["status"] == "Under-utilised"]
    if len(constrained) or len(under):
        insight(
            f"{len(constrained)} site(s) are capacity-constrained and "
            f"{len(under)} are under-utilised. That combination is a network "
            "balancing opportunity: moving slow-moving class C stock from "
            "constrained sites to spare capacity elsewhere releases space without "
            "any reduction in total cover."
        )

# ---------------------------------------------------------------------------
methodology("""
**Turnover.** `annual_units_issued / average_units_on_hand`, where annual issues
are the monthly issue rate scaled to twelve months. Days of inventory is
`365 / turnover`. The benchmark is `inventory.target_turnover` (8x).

**ABC.** Positions are ranked by annual inventory value, then cut at the
cumulative-share thresholds in `inventory.abc_thresholds`: class A to 80% of
value, class B to 95%, class C the remainder. This is value concentration, not
volume - a cheap high-volume tablet can easily be class C.

**Stock-out risk.** Flagged when `units_on_hand < reorder_point x
inventory.stockout_risk_threshold` (0.85). The reorder point itself is
`monthly_demand x lead_time_months + safety_stock`, with safety stock set from
the service-level z-score in `inventory.service_level_z` (1.65, i.e. 95%).
Cost exposure prices the projected shortfall at
`economics.stockout_penalty_per_unit`.

**Overstock.** Positions holding more than `inventory.overstock_months_of_supply`
(4 months) of cover. Excess is the stock above that line, and holding cost applies
`economics.holding_cost_rate_annual` (22% of unit cost per year) to it.

**Expiry.** Positions with `days_to_expiry <= inventory.expiry_risk_days` (120).
Write-off value is net of `economics.expiry_write_off_recovery` (5% salvage).

**Why three registers.** Stock-out, overstock and expiry pull in opposite
directions - reducing one usually worsens another. Reporting them together forces
the trade-off to be explicit rather than optimising one metric into a different
problem.
""")

sidebar_about()
