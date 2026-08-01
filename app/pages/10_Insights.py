"""
Insights page.

The consolidated recommendation set: every finding from the analytical pages,
costed, prioritised, and traced back to the evidence that produced it. This is the
page a client would actually be walked through.
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
from src.analytics import (ab_testing as ab, funnel, inventory, shipments,
                           stability)
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_days, fmt_pct, fmt_units

page_setup(
    title="Consolidated Insights & Recommendations",
    icon="💡",
    subtitle="What the analysis found, what it is worth, and what to do about it",
)


@st.cache_data(show_spinner="Assembling findings...")
def _evidence() -> dict:
    """Gather the evidence base from every analytical module."""
    return {
        "funnel": funnel.funnel_kpis(),
        "bottlenecks": funnel.identify_bottlenecks(),
        "losses": funnel.loss_attribution(top_n=10),
        "shipments": shipments.shipment_kpis(),
        "suppliers": shipments.supplier_performance(),
        "regional": shipments.regional_performance(),
        "inventory": inventory.inventory_kpis(),
        "stability": stability.stability_kpis(),
        "excursion": stability.excursion_impact(),
        "experiments": ab.run_all_experiments(),
    }


ev = _evidence()
fk, sk, ik, stk = ev["funnel"], ev["shipments"], ev["inventory"], ev["stability"]
excursion, experiments = ev["excursion"], ev["experiments"]
suppliers, regional = ev["suppliers"], ev["regional"]

# ---------------------------------------------------------------------------
section(
    "The Headline",
    "One paragraph a supply chain director could act on.",
)

worst_supplier = suppliers.iloc[0]
worst_region = regional.iloc[0]
adopt = experiments[experiments["verdict"].str.startswith("ADOPT")]
experiment_value = float(adopt["annual_value_usd"].sum()) if len(adopt) else 0.0

callout(
    f"This network converts **{fmt_pct(fk['end_to_end_yield_pct'])}** of procured "
    f"units into units dispensed to patients, destroying "
    f"**{fmt_currency(fk['total_value_lost_usd'])}** of value on the way. The loss is "
    f"not evenly spread: **quality testing** is simultaneously the largest "
    f"unit-loss stage and the slowest stage "
    f"({fk['slowest_stage_days']:.0f} days), **{worst_supplier['supplier_name']}** "
    f"delivers only {fmt_pct(worst_supplier['on_time_pct'])} on time while driving a "
    f"disproportionate share of batch rejections, and cold-chain temperature "
    f"excursions cost **{abs(excursion['potency_difference_pct']):.1f} percentage "
    f"points** of potency where they occur. Four interventions were tested "
    f"experimentally; **{len(adopt)}** clear the bar for adoption and together are "
    f"worth **{fmt_currency(experiment_value)}** annually against a gross value pool "
    f"of {fmt_currency(fk['total_value_lost_usd'])}.",
    kind="insight", title="Executive read",
)

kpi_row([
    {"label": "Value Pool", "value": fmt_currency(fk["total_value_lost_usd"]),
     "help_text": "Total value destroyed across the funnel"},
    {"label": "Validated Opportunity", "value": fmt_currency(experiment_value),
     "help_text": f"{len(adopt)} interventions with statistical support"},
    {"label": "Inventory Exposure",
     "value": fmt_currency(ik["expiry_exposure_usd"] + ik["overstock_value_usd"]),
     "help_text": "Expiry risk plus overstock capital"},
    {"label": "SLA Exposure", "value": fmt_currency(sk["late_shipment_cost_usd"]),
     "help_text": "Late-delivery penalty cost"},
])

# ---------------------------------------------------------------------------
section("Findings and Recommendations", "Each finding traces to a page you can verify.")

findings = [
    {
        "priority": 1,
        "area": "Quality",
        "finding": f"Quality testing is the binding constraint - largest unit loss "
                   f"and slowest stage at {fk['slowest_stage_days']:.0f} days "
                   f"({fk['biggest_dropoff_stage']} drops "
                   f"{fk['biggest_dropoff_pct']:.1f}% of volume).",
        "recommendation": "Deploy in-line Process Analytical Technology to shift "
                          "defect detection upstream of QC release.",
        "evidence": "Funnel Analytics - bottleneck ranking; A/B Testing - QA automation",
        "value_usd": float(experiments.loc[
            experiments["key"] == "qa_automation", "annual_value_usd"].iloc[0])
        if (experiments["key"] == "qa_automation").any() else 0.0,
        "confidence": "High - experimentally validated",
    },
    {
        "priority": 2,
        "area": "Sourcing",
        "finding": f"{worst_supplier['supplier_name']} ({worst_supplier['country']}) "
                   f"delivers {fmt_pct(worst_supplier['on_time_pct'])} on time against "
                   f"a contracted {fmt_pct(worst_supplier['contracted_otif_pct'])}, and "
                   f"low-reliability suppliers drive most batch rejections.",
        "recommendation": "Re-qualify or exit the two watchlist API suppliers and "
                          "consolidate volume onto preferred-tier vendors.",
        "evidence": "Shipments - supplier scorecard; Funnel - QA failure analysis",
        "value_usd": float(ev["losses"]["value_lost_usd"].head(3).sum()) * 0.25,
        "confidence": "High - consistent across two independent metrics",
    },
    {
        "priority": 3,
        "area": "Cold Chain",
        "finding": f"Temperature excursions cut potency by "
                   f"{abs(excursion['potency_difference_pct']):.2f} pp "
                   f"(p = {excursion['p_value']:.1e}), lifting the out-of-spec rate "
                   f"by {excursion['oos_rate_lift_pp']:.1f} points.",
        "recommendation": "Fit IoT telemetry with real-time alerting on cold-chain "
                          "lanes, replacing end-of-journey data-logger review.",
        "evidence": "Drug Stability - excursion impact; A/B Testing - cold chain IoT",
        "value_usd": float(experiments.loc[
            experiments["key"] == "cold_chain_iot", "annual_value_usd"].iloc[0])
        if (experiments["key"] == "cold_chain_iot").any() else 0.0,
        "confidence": "High - significant with a large effect size",
    },
    {
        "priority": 4,
        "area": "Working Capital",
        "finding": f"Turnover of {ik['avg_turnover']:.2f}x against a "
                   f"{ik['target_turnover']:.0f}x target, with "
                   f"{fmt_currency(ik['overstock_value_usd'])} overstocked and "
                   f"{fmt_currency(ik['expiry_exposure_usd'])} at expiry risk.",
        "recommendation": "Differentiate replenishment policy by ABC class and shelf "
                          "life: shorter cycles on cold chain, leaner cover on class C.",
        "evidence": "Inventory - ABC analysis, expiry and overstock registers",
        "value_usd": float(ik["excess_holding_cost_usd"] + ik["expiry_exposure_usd"] * 0.4),
        "confidence": "Medium - benefit depends on execution discipline",
    },
    {
        "priority": 5,
        "area": "Logistics",
        "finding": f"{worst_region['region']} runs at "
                   f"{fmt_pct(worst_region['on_time_pct'])} on-time "
                   f"({worst_region['otif_gap_pct']:+.1f} pp against target); network "
                   f"SLA exposure is {fmt_currency(sk['late_shipment_cost_usd'])}.",
        "recommendation": "Introduce dynamic route and carrier selection on the "
                          "weakest lanes before renegotiating the wider carrier base.",
        "evidence": "Shipments - regional performance, transit variance; "
                    "A/B Testing - route optimisation",
        "value_usd": float(experiments.loc[
            experiments["key"] == "route_optimization", "annual_value_usd"].iloc[0])
        if (experiments["key"] == "route_optimization").any() else 0.0,
        "confidence": "Medium - regional effects vary by lane",
    },
]

register = pd.DataFrame(findings)
show_table(
    register[["priority", "area", "finding", "recommendation", "value_usd",
              "confidence", "evidence"]],
    height=420,
)
download_button(register, "insights_register.csv", "Download recommendation register")

chart(charts.bar_chart(
    register.sort_values("value_usd"), x="area", y="value_usd",
    title="Estimated Annual Value by Recommendation Area",
    orientation="h", height=340))

# ---------------------------------------------------------------------------
section("Experiment Verdicts", "Which interventions the evidence actually supports.")

for _, row in experiments.iterrows():
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**{row['experiment']}** - {row['metric']}")
            st.caption(
                f"Control {row['control_rate_pct']:.2f}% → treatment "
                f"{row['treatment_rate_pct']:.2f}% "
                f"({row['lift_pp']:+.2f} pp, {row['relative_lift_pct']:+.1f}% relative) "
                f"· p = {row['p_value']:.2e} · power = {row['achieved_power']:.2f}"
            )
        with right:
            st.markdown(verdict_badge(row["verdict"]), unsafe_allow_html=True)
            st.markdown(f"**{fmt_currency(row['annual_value_usd'])}** / year")

# ---------------------------------------------------------------------------
section("What I Would Do Next", "Honest limitations and the natural extensions.")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("""
**Limitations of this analysis**

- The supply chain data is a **seeded digital twin**, not a production extract.
  Findings demonstrate method rather than describe a real company.
- The A/B tests are **simulated experiments** with effect sizes drawn from
  published benchmarks. They show the decision framework, not measured outcomes.
- Value figures are **gross benefit**: they exclude implementation cost, change
  management and the capital expenditure each intervention requires.
- The simulation is a **first-order elasticity model**, not discrete-event. It has
  no queueing, capacity constraints or stochastic variance.
- Batch risk classification reaches macro F1 ≈ 0.70. That ceiling is genuine
  irreducible noise in the label, and a materially higher score would indicate
  leakage rather than skill.
""")
with col2:
    st.markdown("""
**Natural next steps**

- **Cost the interventions.** Turn gross benefit into NPV with capital cost,
  ramp-up time and a discount rate, then re-rank on ROI rather than value.
- **Sequential testing.** Replace fixed-horizon tests with a sequential design so
  a clearly winning intervention can be adopted early without inflating alpha.
- **Survival analysis on shelf life.** Model time-to-out-of-specification with Cox
  regression instead of OLS on potency, which handles censoring properly.
- **Multi-echelon inventory optimisation.** Optimise safety stock jointly across
  the network rather than position by position, which is where most of the
  working-capital benefit actually sits.
- **Discrete-event simulation.** Model capacity and queueing explicitly to
  validate that compressing QA does not simply move the bottleneck downstream.
""")

insight(
    "The most important number on this page is not the value pool - it is the "
    "distinction between the four interventions that were tested and the ones that "
    "were merely observed. Observation tells you where value sits; only the "
    "experiment tells you whether an intervention moves it. Recommendations 1, 3 "
    "and 5 carry experimental support. Recommendations 2 and 4 are inferences from "
    "consistent observational evidence, and I would want to pilot them before "
    "committing capital."
)

# ---------------------------------------------------------------------------
methodology("""
**How the value column is built.** Each figure has a stated basis rather than being
a single blended estimate:

- **Quality, cold chain and logistics** take the annual value directly from their
  A/B test business-impact calculation: measured lift applied to observed annual
  volume, priced at average unit cost.
- **Sourcing** takes 25% of the value lost in the top three loss-making
  product-region pairs, on the assumption that supplier consolidation addresses
  part but not all of that loss. The 25% is a judgement, and it is stated as one.
- **Working capital** combines the excess holding cost on overstocked positions
  with 40% of expiry exposure, since not all near-expiry stock can be redeployed
  in time.

**Prioritisation.** Ordered by expected value weighted by evidence strength.
Experimentally validated findings outrank observational ones at similar value,
because a validated smaller benefit is worth more than an unvalidated larger one.

**Traceability.** Every finding names the page that produced it. Nothing on this
page is asserted without a chart elsewhere in the platform that a reviewer can
check independently.
""")

sidebar_about()
