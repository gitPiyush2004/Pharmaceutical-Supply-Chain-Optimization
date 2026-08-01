"""
Funnel Analytics page.

The analytical centrepiece: every batch tracked through eight supply chain
stages, measuring where volume is lost, where time is spent, and which stage is
the binding constraint.
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
from src.analytics import funnel
from src.data import loader
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_days, fmt_pct, fmt_units

page_setup(
    title="Supply Chain Funnel Analytics",
    icon="🔻",
    subtitle="Procurement → Manufacturing → Quality → Packaging → Warehouse → "
             "Distributor → Pharmacy → Patient",
)


@st.cache_data(show_spinner="Loading batch data...")
def _batches():
    return loader.load_batches()


batches = sidebar_filters(
    _batches(),
    ["region", "drug_code", "supplier_name", "qa_result", "therapeutic_area"],
)
if batches.empty:
    st.stop()

summary = funnel.funnel_summary(batches)
delays = funnel.stage_delays(batches)
bottlenecks = funnel.identify_bottlenecks(batches)
kpis = funnel.funnel_kpis(batches)

# ---------------------------------------------------------------------------
section("Funnel Health")
kpi_row([
    {"label": "End-to-End Yield", "value": fmt_pct(kpis["end_to_end_yield_pct"]),
     "help_text": "Units reaching a patient / units procured"},
    {"label": "Units Lost", "value": fmt_units(kpis["total_units_lost"]),
     "help_text": fmt_currency(kpis["total_value_lost_usd"]) + " of value"},
    {"label": "Avg Cycle Time", "value": fmt_days(kpis["avg_cycle_time_days"], 0),
     "help_text": f"Median {kpis['median_cycle_time_days']:.0f} days"},
    {"label": "Batches", "value": f"{kpis['total_batches']:,}",
     "help_text": f"QA pass rate {fmt_pct(kpis['qa_pass_rate_pct'])}"},
])

# ---------------------------------------------------------------------------
section(
    "Stage Conversion",
    "Each band is the volume still in the chain. The percentage beside a stage is "
    "conversion from the stage before it.",
)

left, right = st.columns([3, 2], gap="large")
with left:
    chart(charts.funnel_chart(summary, title="Units by Stage"))
with right:
    chart(charts.funnel_dropoff_chart(summary))

show_table(
    summary[["stage", "units", "conversion_from_previous_pct", "dropoff_pct",
             "units_lost", "value_lost_usd", "cumulative_conversion_pct",
             "avg_delay_days", "median_delay_days", "p90_delay_days"]],
    height=330,
)
download_button(summary, "funnel_summary.csv", "Download funnel summary")

worst = summary.loc[summary["dropoff_pct"].idxmax()]
insight(
    f"**{worst['stage']}** loses **{worst['dropoff_pct']:.1f}%** of incoming volume - "
    f"{fmt_units(worst['units_lost'])} units worth "
    f"{fmt_currency(worst['value_lost_usd'])}. Cumulative conversion from "
    f"procurement to patient is **{fmt_pct(kpis['end_to_end_yield_pct'])}**, meaning "
    f"roughly {100 - kpis['end_to_end_yield_pct']:.0f}% of everything bought never "
    "reaches a patient. In pharmaceutical manufacturing that is the normal order of "
    "magnitude - the question is which stages are addressable, which the bottleneck "
    "ranking below answers."
)

# ---------------------------------------------------------------------------
section(
    "Where Time Goes",
    "Dwell time per stage. Unit loss and time loss are different problems with "
    "different fixes, so they are measured separately.",
)

col1, col2 = st.columns([3, 2], gap="large")
with col1:
    chart(charts.stage_delay_chart(delays))
with col2:
    show_table(delays[["stage", "avg_delay_days", "median_delay_days",
                       "p90_delay_days", "share_of_cycle_pct"]], height=340)

slowest = delays.loc[delays["avg_delay_days"].idxmax()]
insight(
    f"**{slowest['stage']}** is the slowest stage at "
    f"{slowest['avg_delay_days']:.1f} days on average "
    f"({slowest['share_of_cycle_pct']:.0f}% of total cycle time), with a p90 of "
    f"{slowest['p90_delay_days']:.0f} days. The gap between median and p90 is where "
    "the operational pain sits: a long tail means unpredictable release dates, which "
    "forces downstream teams to hold buffer stock they would not otherwise need."
)

# ---------------------------------------------------------------------------
section(
    "Bottleneck Ranking",
    "A stage is flagged when it breaches the drop-off or delay threshold declared "
    "in `config/config.yaml`. Severity blends both with the value destroyed.",
)

flagged = bottlenecks[bottlenecks["is_bottleneck"]]
if len(flagged):
    callout(
        f"**{len(flagged)} bottleneck stage(s) detected:** "
        f"{', '.join(flagged['stage'])}. "
        f"Primary constraint: **{kpis['primary_bottleneck']}**.",
        kind="warning", title="Action required",
    )
else:
    callout("No stage breaches the configured thresholds.", kind="success")

show_table(
    bottlenecks[["stage", "dropoff_pct", "avg_delay_days", "value_lost_usd",
                 "bottleneck_type", "severity_score", "recommended_action"]],
    height=320,
)

# ---------------------------------------------------------------------------
section("Segment Comparison", "How the funnel differs across the business.")

dimension = st.selectbox(
    "Compare by",
    ["region", "drug_code", "supplier_name", "therapeutic_area", "form"],
    format_func=lambda c: c.replace("_", " ").title(),
)
conversion = funnel.stage_conversion_by_dimension(batches, dimension=dimension)
chart(charts.funnel_comparison_chart(conversion, dimension=dimension))

by_dim = funnel.funnel_by_dimension(batches, dimension=dimension)
show_table(by_dim, height=300)

# ---------------------------------------------------------------------------
section("Quality Failures and Loss Attribution")

tab1, tab2, tab3 = st.tabs(["QA Failure Reasons", "Loss Attribution", "Trend Over Time"])

with tab1:
    qa_fail = funnel.qa_failure_analysis(batches)
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        chart(charts.donut_chart(qa_fail["qa_fail_reason"].tolist(),
                                 qa_fail["batches"].tolist(),
                                 title="Rejected Batches by Root Cause"))
    with c2:
        show_table(qa_fail, height=280)
    top = qa_fail.iloc[0]
    insight(
        f"**{top['qa_fail_reason']}** is the most common rejection reason "
        f"({top['batches']} batches, {top['share_pct']:.0f}% of failures) and "
        f"rejected batches add roughly {top['avg_qa_delay_days']:.0f} days of "
        "investigation and retest. Failure modes cluster around assay and "
        "dissolution, both of which trace back to incoming API quality rather than "
        "to the manufacturing process - which is why the supplier scorecard on the "
        "Shipments page is the right place to act."
    )

with tab2:
    losses = funnel.loss_attribution(batches, top_n=15)
    chart(charts.pareto_chart(losses, category="brand_name",
                              value="value_lost_usd",
                              cumulative="cumulative_share_pct",
                              title="Value Lost by Product and Region (Pareto)"))
    show_table(losses, height=320)
    download_button(losses, "loss_attribution.csv", "Download loss attribution")

with tab3:
    trend = funnel.funnel_trend(batches, period="quarter")
    chart(charts.line_chart(trend, x="quarter",
                            y=["end_to_end_yield_pct", "qa_pass_rate_pct"],
                            title="Yield and QA Pass Rate by Quarter",
                            y_title="Percent"))
    chart(charts.line_chart(trend, x="quarter", y="avg_cycle_time_days",
                            title="Average Cycle Time by Quarter",
                            y_title="Days"))
    show_table(trend, height=280)

# ---------------------------------------------------------------------------
methodology("""
**Funnel construction.** Each batch carries a unit count at all eight stages
(`units_procured` through `units_dispensed`) and the date it cleared each stage.
The funnel aggregates units across the filtered batch population, so it is a true
volume funnel rather than a count of records.

**Conversion versus drop-off.** Conversion is `units_at_stage / units_at_previous_stage`.
Drop-off is `1 - conversion`. Cumulative conversion is measured against
procurement, which is why the last row equals end-to-end yield.

**Delays.** Stage dwell time is the difference between consecutive stage dates.
Mean, median and p90 are all reported: the mean is what plans are built on, the
p90 is what actually causes downstream buffer stock.

**Bottleneck logic.** A stage is flagged when drop-off exceeds
`funnel.bottleneck.dropoff_pct_threshold` (6%) or dwell time exceeds
`funnel.bottleneck.delay_days_threshold` (15 days). Severity normalises drop-off,
delay and value lost onto a common scale and sums them, so a stage that is bad on
two dimensions outranks one that is worse on a single dimension.

**Value.** Units lost are costed at the batch's `unit_cost_usd`. This is
deliberately conservative - it prices lost inventory at cost, not at the revenue
it would have earned, so the numbers understate rather than overstate the case for
intervention.
""")

sidebar_about()
