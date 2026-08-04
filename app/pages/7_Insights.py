"""
Insights page.

The consolidated recommendation set: every finding, costed where that is
defensible, prioritised, and traced back to the page that produced it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about, verdict_badge)
from src.analytics import ab_testing as ab, funnel, procurement, stability
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct

page_setup(
    title="Insights & Recommendations",
    icon="💡",
    subtitle="What the analysis found, what it is worth, and what to do about it",
)


@st.cache_data(show_spinner="Assembling findings...")
def _evidence() -> dict:
    """Gather the evidence base from every analytical module."""
    return {
        "funnel": funnel.funnel_kpis(),
        "scms": procurement.scms_kpis(),
        "vendors": procurement.vendor_scorecard(),
        "modes": procurement.mode_performance(),
        "excursion": stability.excursion_impact(),
        "experiments": ab.run_all_experiments(),
    }


ev = _evidence()
fk, sk = ev["funnel"], ev["scms"]
excursion, experiments = ev["excursion"], ev["experiments"]
vendors, modes = ev["vendors"], ev["modes"]

worst_vendor = vendors.iloc[0]
adopt = experiments[experiments["verdict"].str.startswith("ADOPT")]
experiment_value = float(adopt["annual_value_usd"].sum()) if len(adopt) else 0.0

# ---------------------------------------------------------------------------
section("The Headline", "One paragraph a supply chain director could act on.")

callout(
    f"On **real USAID data**, this network delivers "
    f"**{fmt_pct(sk['on_time_delivery_pct'])}** of {sk['shipments']:,} shipments on "
    f"time, and the weakest link is **{worst_vendor['vendor']}** at "
    f"{fmt_pct(worst_vendor['on_time_pct'])} — an internal distribution channel "
    f"rather than an external manufacturer, carrying "
    f"{fmt_currency(worst_vendor['line_value_usd'])} of commodity value. "
    f"On the **simulated manufacturing side**, only "
    f"{fmt_pct(fk['end_to_end_yield_pct'])} of procured units reach a patient, and "
    f"quality testing is the binding constraint — largest unit loss and slowest "
    f"stage at {fk['slowest_stage_days']:.0f} days. "
    f"Four interventions were tested experimentally; **{len(adopt)}** clear the bar "
    f"for adoption, together worth **{fmt_currency(experiment_value)}** a year.",
    kind="insight", title="Executive read",
)

kpi_row([
    {"label": "Validated Opportunity", "value": fmt_currency(experiment_value),
     "help_text": f"{len(adopt)} interventions with statistical support"},
    {"label": "Simulated Value Pool", "value": fmt_currency(fk["total_value_lost_usd"]),
     "help_text": "Value destroyed across the manufacturing funnel"},
    {"label": "Real Commodity Value",
     "value": fmt_currency(sk["total_commodity_value_usd"]),
     "help_text": f"Moved across {sk['countries']} countries, 2006-2015"},
    {"label": "Real Late Shipments", "value": f"{sk['late_shipments']:,}",
     "help_text": f"of {sk['shipments']:,} — the addressable service gap"},
])

# ---------------------------------------------------------------------------
section("Findings and Recommendations", "Each finding traces to a page you can verify.")

air = modes[modes["shipment_mode"] == "Air"]
ocean = modes[modes["shipment_mode"] == "Ocean"]
if len(air) and len(ocean):
    mode_finding = (
        f"Ocean freight costs ${ocean.iloc[0]['median_freight_per_kg_usd']:.2f}/kg "
        f"against ${air.iloc[0]['median_freight_per_kg_usd']:.2f}/kg for air, and "
        f"runs {air.iloc[0]['on_time_pct'] - ocean.iloc[0]['on_time_pct']:.1f} points "
        "worse on on-time delivery."
    )
else:
    mode_finding = "Transport mode drives both freight cost and reliability."


def _experiment_value(key: str) -> float:
    """Annual value from the A/B result, or 0 when that experiment is absent."""
    match = experiments[experiments["key"] == key]
    return float(match["annual_value_usd"].iloc[0]) if len(match) else 0.0


findings = pd.DataFrame([
    {
        "priority": 1, "area": "Sourcing", "data": "REAL",
        "finding": f"{worst_vendor['vendor']} delivers "
                   f"{worst_vendor['on_time_pct']:.1f}% on time across "
                   f"{int(worst_vendor['shipments']):,} shipments — an internal "
                   "channel, not an external supplier.",
        "recommendation": "Fix the internal distribution channel first: no contract "
                          "renegotiation needed, and it is the largest single "
                          "service gap in the network.",
        "value_usd": 0.0,
        "confidence": "High — measured on real data",
        "evidence": "Real-World Operations",
    },
    {
        "priority": 2, "area": "Quality", "data": "SIMULATED",
        "finding": f"Quality testing is the binding constraint: largest unit loss "
                   f"and slowest stage at {fk['slowest_stage_days']:.0f} days.",
        "recommendation": "Deploy in-line Process Analytical Technology to move "
                          "defect detection upstream of QC release.",
        "value_usd": _experiment_value("qa_automation"),
        "confidence": "High — experimentally validated",
        "evidence": "Funnel Analytics · A/B Testing",
    },
    {
        "priority": 3, "area": "Cold Chain", "data": "SIMULATED",
        "finding": f"Temperature excursions cut potency by "
                   f"{abs(excursion['potency_difference_pct']):.2f} pp "
                   f"(p = {excursion['p_value']:.1e}).",
        "recommendation": "Fit IoT telemetry with real-time alerting on cold-chain "
                          "lanes, replacing end-of-journey logger review.",
        "value_usd": _experiment_value("cold_chain_iot"),
        "confidence": "High — significant, large effect size",
        "evidence": "Drug Stability · A/B Testing",
    },
    {
        "priority": 4, "area": "Logistics", "data": "REAL",
        "finding": mode_finding,
        "recommendation": "Pay for air freight only where shelf life or stock-out "
                          "risk justifies it; move long-dated bulk resupply to ocean.",
        "value_usd": _experiment_value("route_optimization"),
        "confidence": "Medium — measured, but lane choice is constrained",
        "evidence": "Real-World Operations",
    },
    {
        "priority": 5, "area": "Operations", "data": "REAL",
        "finding": "Late delivery is predictable: ROC AUC 0.85, and the top 20% by "
                   "predicted risk contains 63% of all late shipments.",
        "recommendation": "Run the model as a daily expeditor priority queue rather "
                          "than a binary gate.",
        "value_usd": 0.0,
        "confidence": "High — held-out validation on real shipments",
        "evidence": "ML Models",
    },
])

show_table(findings, height=360)
download_button(findings, "recommendations.csv", "Download recommendation register")

valued = findings[findings["value_usd"] > 0]
if len(valued):
    chart(charts.bar_chart(valued.sort_values("value_usd"), x="area", y="value_usd",
                           title="Estimated Annual Value by Recommendation Area",
                           orientation="h", height=320))
    st.caption(
        "Only experimentally validated interventions carry a value estimate. The "
        "real-data findings are ranked on evidence strength rather than given a "
        "dollar figure I could not defend."
    )

# ---------------------------------------------------------------------------
section("Experiment Verdicts", "Which interventions the evidence actually supports.")

for _, row in experiments.iterrows():
    with st.container(border=True):
        left, right = st.columns([3, 1], gap="medium")
        with left:
            st.markdown(f"**{row['experiment']}** — {row['metric']}")
            st.caption(
                f"Control {row['control_rate_pct']:.2f}% → treatment "
                f"{row['treatment_rate_pct']:.2f}% "
                f"({row['lift_pp']:+.2f} pp, {row['relative_lift_pct']:+.1f}% relative) "
                f"· p = {row['p_value']:.2e} · power = {row['achieved_power']:.2f}"
            )
        with right:
            st.markdown(verdict_badge(row["verdict"]), unsafe_allow_html=True)
            st.markdown(f"**{fmt_currency(row['annual_value_usd'])}** / year")

insight(
    "The distinction that matters most here is between findings that were "
    "**measured** and interventions that were **tested**. Measurement tells you "
    "where value sits; only an experiment tells you whether an intervention moves "
    "it. Priorities 2 and 3 carry experimental support. Priorities 1, 4 and 5 are "
    "measured on real data but not experimentally validated — I would pilot them "
    "before committing capital."
)

# ---------------------------------------------------------------------------
section("Limitations and Next Steps")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("""
**Limitations**

- The manufacturing data is a **calibrated simulation**. It demonstrates method;
  it does not describe a real company.
- The A/B tests are **simulated experiments** with effect sizes drawn from
  published benchmarks — the framework is real, the outcomes are not measured.
- Value figures are **gross benefit**, excluding implementation cost. Not an ROI.
- The real SCMS data ends in **2015**, so its relationships describe that era.
""")
with col2:
    st.markdown("""
**Next steps**

- **Cost the interventions** — convert gross benefit to NPV, re-rank on ROI.
- **Pilot the RDC finding** — the largest real-data opportunity, currently
  measured but not tested.
- **Survival analysis for shelf life** — Cox regression handles censoring, which
  OLS on potency does not.
- **Recalibrate on a modern extract** before any operational use.
""")

methodology("""
**How the value column is built.** Only experimentally validated interventions
carry a figure, taken from the A/B business-impact calculation: measured lift
applied to observed annual volume, priced at average unit cost. Findings that are
measured but not tested are deliberately left without a dollar estimate rather
than given a fabricated one.

**Prioritisation.** Evidence strength first, then expected value. The real-data
sourcing finding ranks first despite carrying no value estimate, because it is
measured on genuine operational data and is directly actionable.

**Traceability.** Every finding names the page that produced it, and the `data`
column states whether it rests on real or simulated data.
""")

sidebar_about()
