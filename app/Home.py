"""
PharmaChain Analytics - executive landing page.

Answers three questions before a reader clicks anything: how healthy is the
supply chain, where is value leaking, and what should be done about it. The
detail pages then let a reader verify every claim made here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.dashboard.components import (callout, chart, insight, kpi_row, methodology,
                        page_setup, section, show_table, sidebar_about)
from src.analytics import funnel, inventory, shipments, stability
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_days, fmt_pct, fmt_units

page_setup(
    title="Pharmaceutical Supply Chain Optimization",
    icon="💊",
    subtitle="Funnel analytics, statistical experimentation and machine learning "
             "across an eight-stage pharmaceutical supply chain",
)


# ---------------------------------------------------------------------------
# Cached metric loads - one cache entry serves every page in the session.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing supply chain metrics...")
def _headline_metrics() -> dict:
    """Collect the KPI set shown on this page from every analytics module."""
    return {
        "funnel": funnel.funnel_kpis(),
        "shipments": shipments.shipment_kpis(),
        "inventory": inventory.inventory_kpis(),
        "stability": stability.stability_kpis(),
    }


@st.cache_data(show_spinner=False)
def _funnel_views() -> tuple:
    """Funnel summary, bottleneck ranking and loss attribution."""
    return (funnel.funnel_summary(), funnel.identify_bottlenecks(),
            funnel.loss_attribution(top_n=8))


metrics = _headline_metrics()
fk, sk, ik, stk = (metrics["funnel"], metrics["shipments"],
                   metrics["inventory"], metrics["stability"])
summary, bottlenecks, losses = _funnel_views()

# ---------------------------------------------------------------------------
# Executive KPI band
# ---------------------------------------------------------------------------
section(
    "Executive Summary",
    "Network-wide performance across the batch lifecycle, from API procurement "
    "to the unit reaching a patient.",
)

kpi_row([
    {"label": "End-to-End Yield",
     "value": fmt_pct(fk["end_to_end_yield_pct"]),
     "help_text": f"{fmt_units(fk['units_dispensed'])} of "
                  f"{fmt_units(fk['units_procured'])} units reach a patient"},
    {"label": "Value Lost",
     "value": fmt_currency(fk["total_value_lost_usd"]),
     "help_text": "Cost of units that never reach a patient"},
    {"label": "Avg Cycle Time",
     "value": fmt_days(fk["avg_cycle_time_days"], 0),
     "help_text": "Procurement to patient, per batch"},
    {"label": "QA Pass Rate",
     "value": fmt_pct(fk["qa_pass_rate_pct"]),
     "help_text": "Batches released at quality control"},
])

kpi_row([
    {"label": "On-Time Delivery",
     "value": fmt_pct(sk["on_time_delivery_pct"]),
     "delta": f"{sk['on_time_delivery_pct'] - sk['otif_target_pct']:+.1f} pp vs "
              f"{sk['otif_target_pct']:.0f}% target",
     "delta_good": bool(sk["meets_otif_target"]),
     "help_text": f"{sk['late_shipments']:,} of {sk['total_shipments']:,} legs late"},
    {"label": "Inventory Value",
     "value": fmt_currency(ik["total_inventory_value_usd"]),
     "help_text": f"{ik['avg_turnover']:.1f} turns vs "
                  f"{ik['target_turnover']:.0f} target"},
    {"label": "Expiry Exposure",
     "value": fmt_currency(ik["expiry_exposure_usd"]),
     "help_text": "Stock at risk of write-off"},
    {"label": "Batches Out of Spec",
     "value": fmt_pct(stk["out_of_spec_pct"]),
     "help_text": f"Potency below {stk['potency_spec_min']:.0f}% of label claim"},
])

# ---------------------------------------------------------------------------
# The funnel - the analytical centrepiece
# ---------------------------------------------------------------------------
section(
    "The Supply Chain Funnel",
    "Every batch is tracked through eight stages. The width of each band is the "
    "volume still in the chain; the gap between bands is where units are lost.",
)

left, right = st.columns([3, 2], gap="large")
with left:
    chart(charts.funnel_chart(summary, title="Units Surviving Each Stage"))
with right:
    chart(charts.funnel_dropoff_chart(summary))

worst = summary.loc[summary["dropoff_pct"].idxmax()]
slowest = fk["slowest_stage"]
insight(
    f"**{worst['stage']}** is the largest single point of unit loss, shedding "
    f"**{worst['dropoff_pct']:.1f}%** of incoming volume "
    f"({fmt_units(worst['units_lost'])} units, {fmt_currency(worst['value_lost_usd'])}). "
    f"**{slowest}** is the slowest stage at {fk['slowest_stage_days']:.0f} days, "
    f"making it the constraint on cycle time. Both point at the same fix: quality "
    f"testing is where this network loses the most volume *and* the most time, "
    f"which is why it is the first intervention tested on the A/B Testing page."
)

# ---------------------------------------------------------------------------
# Bottlenecks and loss drivers
# ---------------------------------------------------------------------------
section(
    "Where To Intervene",
    "Stages are ranked by a severity score combining unit loss, dwell time and "
    "the value destroyed, so remediation effort follows the money.",
)

flagged = bottlenecks[bottlenecks["is_bottleneck"]]
if len(flagged):
    callout(
        f"{len(flagged)} of {len(bottlenecks)} stages breach the configured "
        f"thresholds: **{', '.join(flagged['stage'].tolist())}**.",
        kind="warning", title="Bottlenecks detected",
    )

show_table(
    bottlenecks[["stage", "dropoff_pct", "avg_delay_days", "value_lost_usd",
                 "bottleneck_type", "severity_score", "recommended_action"]],
    height=300,
)

section("Loss Concentration by Product and Region")
chart(charts.pareto_chart(
    losses, category="brand_name", value="value_lost_usd",
    cumulative="cumulative_share_pct",
    title="Value Lost - Pareto by Product (top 8 product-region pairs)",
))
insight(
    f"The top three product-region combinations account for "
    f"**{losses.head(3)['share_of_total_loss_pct'].sum():.0f}%** of all value lost. "
    "Loss is concentrated, not spread evenly, so a targeted programme on a handful "
    "of lanes recovers most of the available value."
)

# ---------------------------------------------------------------------------
# What this platform contains
# ---------------------------------------------------------------------------
section("How To Navigate This Platform")

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown("""
**Foundation**
- **Data Quality** - profiles the raw extract, then shows what the cleaning
  layer fixed
- **Funnel Analytics** - the eight-stage conversion and delay analysis

**Operations**
- **Inventory** - turnover, ABC, stock-out, overstock and expiry risk
- **Shipments** - supplier, carrier and regional service performance
""")
with col2:
    st.markdown("""
**Forward-looking**
- **Demand Forecasting** - Holt-Winters, trend and moving-average models,
  chosen by backtest rather than assumption
- **Drug Stability** - temperature, humidity and duration effects on potency
- **Simulation** - move seven operational levers and watch KPIs respond

**Decision science**
- **A/B Testing** - four operational interventions tested for significance
""")
with col3:
    st.markdown("""
**Machine learning**
- **Drug Classification** - patient-level prescribing model on the Kaggle
  drug200 dataset
- **Batch Risk** - stability risk classifier on batch telemetry

**Reference**
- **Insights** - the consolidated recommendation set with quantified impact
- **Google Colab** - the full reproducible ML notebook
""")

methodology("""
**Data layers.** The platform separates a *bronze* layer (`data/raw`, what source
systems hand over, complete with sensor dropouts and free-text region spellings)
from a *silver* layer (deduplicated, canonicalised and imputed) that every
analytics module reads. The Data Quality page shows both.

**Two datasets, one business.** `drug200.csv` is the real Kaggle clinical dataset
used for the prescribing model. The supply chain tables are a seeded digital twin
whose product mix is derived from the prescription distribution observed in
`drug200.csv`, so both halves describe the same portfolio. Stage yields, QC
release times and OTIF levels are calibrated to published industry benchmarks and
declared in `config/config.yaml`.

**Reproducibility.** Everything is deterministic. `python scripts/build_dataset.py`
followed by `python scripts/train_models.py` regenerates every number on this
page from a clean checkout.
""")

sidebar_about()
