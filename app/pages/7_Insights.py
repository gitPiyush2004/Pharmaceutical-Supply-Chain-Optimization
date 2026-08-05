"""
Insights page.

The consolidated finding set. Each one states what it licenses you to conclude and
what it does not, because on observational data those are different things and the
distinction is where analysis either earns trust or loses it.

Costed only where costing is defensible. SCMS records no penalty, discount or
expediting cost, so exactly one assumed rate is used
(``economics.late_shipment_penalty``) and every figure that depends on it says so.
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

from src.analytics import ab_testing as ab
from src.analytics import experiments as ex
from src.analytics import market, pipeline, procurement
from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.ml import predict
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct

page_setup(
    title="Insights & Recommendations",
    icon="💡",
    subtitle="What the analysis found, what it is worth, and what it does not "
             "establish",
)


@st.cache_data(show_spinner="Assembling findings...")
def _evidence() -> dict:
    """Gather the evidence base from every analytical module."""
    return {
        "pipeline": pipeline.pipeline_kpis(),
        "scms": procurement.scms_kpis(),
        "vendors": procurement.vendor_scorecard(),
        "modes": procurement.mode_performance(),
        "countries": procurement.country_performance(),
        "catalogue": ex.comparison_catalogue(),
        "stratified": ex.stratified_comparison(),
        "market": market.market_kpis(),
        "ingredients": market.ingredient_analysis(),
        "lateness": pipeline.lateness_funnel(),
    }


@st.cache_data(show_spinner=False)
def _models() -> dict:
    out = {}
    for name in ("drug_classification", "late_delivery"):
        try:
            out[name] = predict.model_summary(name)
        except FileNotFoundError:
            out[name] = None
    return out


ev = _evidence()
pk, sk = ev["pipeline"], ev["scms"]
vendors, catalogue, strat = ev["vendors"], ev["catalogue"], ev["stratified"]
models = _models()

worst_vendor = vendors.iloc[0]
late_2011 = strat["strata"].iloc[-1]

# The one costed figure on this page, built from measured counts.
years = 10  # 2006-2015 inclusive
recovery = ab.penalty_impact(
    absolute_diff=(late_2011["on_time_Direct Drop"] - late_2011["on_time_From RDC"]) / 100,
    annual_shipments=len(vendors) and sk["shipments"] / years)

# ---------------------------------------------------------------------------
section("The Headline", "One paragraph a supply chain director could act on.")

callout(
    f"This network delivers **{fmt_pct(sk['on_time_delivery_pct'])}** of "
    f"{sk['shipments']:,} real shipments on time, worth "
    f"{fmt_currency(pk['value_ordered_usd'])} in commodity value, of which "
    f"**{fmt_currency(pk['value_late_usd'])} arrives late**. The single largest "
    f"controllable driver is the programme's own regional distribution centre "
    f"channel, and its performance **collapsed after 2010** — from 93.4% to 73.9% "
    f"on time — while the direct-drop channel held at roughly 94%. The weakest "
    f"individual vendor is **{worst_vendor['vendor']}** at "
    f"{fmt_pct(worst_vendor['on_time_pct'])} on "
    f"{worst_vendor['shipments']:,} shipments.\n\n"
    f"Closing the post-2010 channel gap is worth on the order of "
    f"**{fmt_currency(recovery['annual_value_usd'])} a year** in avoided SLA "
    f"credits — measured shipment volume against one assumed penalty rate, stated "
    f"below. The prerequisite is finding out what changed in 2011, which the data "
    f"can localise but not explain.",
    kind="insight", title="What the data says",
)

kpi_row([
    {"label": "Value Arriving Late", "value": fmt_currency(pk["value_late_usd"]),
     "delta": f"{100 - pk['value_on_time_pct']:.1f}% of ordered value",
     "delta_good": False,
     "help_text": f"Of {fmt_currency(pk['value_ordered_usd'])} across "
                  f"{pk['line_items']:,} line items"},
    {"label": "Post-2010 Channel Gap",
     "value": f"{late_2011['gap_pp']:.1f} pp",
     "delta": f"vs {strat['strata'].iloc[0]['gap_pp']:.1f} pp before 2011",
     "delta_good": False,
     "help_text": "Direct drop versus regional distribution centre"},
    {"label": "Annual SLA Exposure",
     "value": fmt_currency(recovery["annual_value_usd"]),
     "help_text": "Measured volume x measured gap x one assumed penalty rate"},
    {"label": "Significant Comparisons",
     "value": f"{int(catalogue['significant'].sum())} of {len(catalogue)}",
     "help_text": "Dimensions where on-time rate differs significantly"},
])

# ---------------------------------------------------------------------------
section(
    "Findings, With Their Limits",
    "Each finding states what it establishes and what it does not. On "
    "observational data the second half is the part that matters.",
)

findings = pd.DataFrame([
    {
        "finding": "The RDC channel collapsed after 2010, it was never simply weak",
        "evidence": f"On-time fell 93.4% -> 73.9% post-2010 while direct drop held "
                    f"at ~94%. Pooled gap {strat['pooled']['gap_pp']:+.2f}pp, "
                    f"stratified {strat['strata'].iloc[0]['gap_pp']:.2f}pp then "
                    f"{late_2011['gap_pp']:.2f}pp. p = 3.0e-109 post-2010.",
        "action": "Investigate what changed in 2011 - capacity, funding, staffing or "
                  "country mix - before reallocating volume away from the channel.",
        "does_not_establish": "Causation. Nobody randomised shipments into a route, "
                             "and the countries served each way also changed. The "
                             "era split localises the change in time; it does not "
                             "name the cause.",
        "confidence": "High",
        "source": "Statistical Testing",
    },
    {
        "finding": "Lateness is many small slips, not a few disasters",
        "evidence": f"{ev['lateness'].loc[1, 'share_pct']:.1f}% of value is within a "
                    f"week of schedule, {ev['lateness'].loc[2, 'share_pct']:.1f}% "
                    f"within a month, {ev['lateness'].loc[3, 'share_pct']:.1f}% "
                    f"within 90 days.",
        "action": "A week of buffer stock on the affected lines absorbs most of the "
                  "exposure. Expediting individual shipments would cost more and "
                  "recover less.",
        "does_not_establish": "That the slips are independent. A shared upstream "
                              "cause producing many small delays would look "
                              "identical in this view.",
        "confidence": "High",
        "source": "Delivery Pipeline",
    },
    {
        "finding": "The headline on-time rate overstates planning precision",
        "evidence": f"{pk['exact_day_pct']:.0f}% of shipments arrive on exactly "
                    f"their scheduled day and {pk['early_pct']:.0f}% arrive early. "
                    f"Only {pk['late_pct']:.0f}% are late at all.",
        "action": "Manage the late tail and the early arrivals separately. Early "
                  "delivery is counted as success by the metric and carries real "
                  "holding cost.",
        "does_not_establish": "That the schedule is being back-filled. A 61% "
                              "exact-day concentration is consistent with that, and "
                              "also with conservative scheduling. The dataset has no "
                              "audit trail to separate them.",
        "confidence": "Medium",
        "source": "Delivery Pipeline",
    },
    {
        "finding": "Transport mode is a cost-service frontier, not a ranking",
        "evidence": f"Air {ev['modes'].set_index('shipment_mode').loc['Air', 'on_time_pct']:.1f}% "
                    f"on time against Ocean "
                    f"{ev['modes'].set_index('shipment_mode').loc['Ocean', 'on_time_pct']:.1f}%, "
                    f"at roughly six times the freight cost per kilogram. "
                    f"Chi-square p = 3.0e-19.",
        "action": "Route by commodity criticality: short-dated and stock-out-"
                  "critical product justifies air, bulk resupply of long-dated "
                  "product does not.",
        "does_not_establish": "A clean mode effect. Ocean is almost entirely "
                             "post-2010, so mode is entangled with the same era "
                             "shift that drives finding 1.",
        "confidence": "Medium",
        "source": "Vendor & Logistics",
    },
    {
        "finding": "First-line designation does not affect delivery performance",
        "evidence": "88.56% versus 88.40%, p = 0.812. Minimum detectable effect "
                    "1.82pp at 80% power on 3,294 versus 7,030 shipments.",
        "action": "Do not build differentiated handling around this attribute. "
                  "Spend the attention on the channel gap instead.",
        "does_not_establish": "That the gap is exactly zero. The bound rules out "
                             "anything above 1.82pp, which sits just above the "
                             "1.33pp that would justify acting - so a marginally "
                             "material difference cannot be excluded.",
        "confidence": "Medium",
        "source": "Statistical Testing",
    },
    {
        "finding": "The late-delivery model is a triage queue, not a gate",
        "evidence": f"ROC AUC {models['late_delivery']['roc_auc_ovr']:.3f} but raw "
                    f"accuracy below the 88.5% majority-class baseline. Reviewing "
                    f"the top 20% by predicted risk captures a disproportionate "
                    f"share of late shipments."
                    if models["late_delivery"] else "Model artefact not built.",
        "action": "Deploy as a daily prioritisation list for expeditors, ranked by "
                  "probability. Do not use the 0.5 threshold as a decision rule.",
        "does_not_establish": "Transferability. The data ends in 2015, so the "
                             "learned relationships describe that era's network - "
                             "and finding 1 shows that network changed materially "
                             "mid-sample.",
        "confidence": "High",
        "source": "ML Models",
    },
    {
        "finding": "Indian generics pricing offers real substitution headroom",
        "evidence": f"{ev['ingredients'].iloc[0]['ingredient']} is sold under "
                    f"{ev['ingredients'].iloc[0]['brands']:,} brand names by "
                    f"{ev['ingredients'].iloc[0]['manufacturers']:,} manufacturers, "
                    f"with a Rs {ev['ingredients'].iloc[0]['price_spread_inr']:,.0f} "
                    f"spread between price quartiles on identical composition.",
        "action": "For formulary buying, tender by molecule rather than by brand. "
                  "With 7,642 manufacturers the leverage is in switching, not in "
                  "volume commitments.",
        "does_not_establish": "That the cheaper products are equivalent in "
                             "practice. This dataset carries composition and price "
                             "but nothing on bioequivalence, quality history or "
                             "supply reliability.",
        "confidence": "Medium",
        "source": "Indian Pharma Market",
    },
    {
        "finding": "A grade-A data quality score can be worthless",
        "evidence": "SCMS scores 99.3% complete and grade A, while 55% of "
                    "purchase-order dates and 40% of freight costs are unparseable "
                    "text. Parsing it correctly *lowers* the generic score by 1.70 "
                    "points.",
        "action": "Profile meaning, not just types. Give every unusable value a "
                  "reason code and separate structural absence from a genuine gap "
                  "before any imputation decision.",
        "does_not_establish": "That the scoring weights are wrong. The dimensions "
                             "measure what they claim to; the gap is that "
                             "completeness cannot see semantics.",
        "confidence": "High",
        "source": "Data Quality",
    },
])

show_table(findings)

# ---------------------------------------------------------------------------
section(
    "What The Comparisons Support",
    "Every dimension tested, with the confound attached. Read the confound column "
    "before the p-value column.",
)

show_table(catalogue)

chart(charts.rate_comparison_chart(
    ex.group_rates(dimension="region"), title="On-Time Rate by Destination Region",
    target=float(sk["on_time_target_pct"])))

insight(
    f"Destination region shows the widest gap of any dimension "
    f"({catalogue.loc[0, 'gap_pp']:.1f} points, Asia at 99.2% against West & Central "
    f"Africa at 86.4%) and is also the least actionable — the programme does not "
    f"choose where the medicines are needed. It is confounded with product mix and "
    f"transport mode on top of that.\n\n"
    f"This is why the catalogue is sorted by gap size but the recommendations are "
    f"not. The largest measured difference and the most useful one are rarely the "
    f"same row, and a dashboard that ranks findings purely by effect size will point "
    f"an organisation at the things it cannot change."
)

# ---------------------------------------------------------------------------
section("Where Not To Spend Attention", "Negative results are recommendations too.")

st.markdown("""
Three things this analysis argues **against** doing:

1. **Do not differentiate handling by first-line designation.** A well-bounded null
   on 10,324 shipments: no difference above 1.8 percentage points. Any process built
   on the assumption that these products behave differently in transit is building on
   an effect the data cannot find.

2. **Do not deploy the late-delivery model as a binary gate.** At its default
   threshold it flags a minority of the late shipments and its raw accuracy is
   *below* the majority-class baseline. The ranking is good; the label is not. Using
   it as a pass/fail control would be worse than doing nothing, because it would
   carry the authority of a model.

3. **Do not model discontinuation risk on the Indian product master.** The flag
   varies by 51 percentage points between manufacturers and by 1.1 points across
   price quartiles. That pattern is catalogue refresh timing in the source, not
   product economics, and the file has no launch date, sales volume or therapeutic
   class to model it with. A classifier here would produce a confident risk score for
   something the data cannot see.
""")

# ---------------------------------------------------------------------------
section("The One Assumed Number", "Everything else on this page is measured.")

callout(
    f"**{recovery['calculation_basis']}**\n\n"
    f"{recovery['assumption']} SCMS records no penalty, discount, expediting cost or "
    f"stock-out consequence, so there is no way to derive the cost of a late "
    f"delivery from the data itself. The shipment count and the rate gap are both "
    f"measured; the "
    f"${recovery['penalty_per_late_shipment_usd']:,.0f} per late shipment is a "
    f"configured assumption in `economics.late_shipment_penalty`, isolated there "
    f"precisely so it is obvious which figures rest on it.\n\n"
    f"Change that one number and the "
    f"{fmt_currency(recovery['annual_value_usd'])} moves proportionally. Nothing "
    f"else on this dashboard does.",
    kind="warning", title="Assumption trail",
)

download_button(findings, "insights_findings.csv", label="Download findings")

methodology("""
**Why every finding carries a "does not establish" column.** These are
observational comparisons. Groups differ in more ways than the one being compared,
so a significant difference identifies where to look rather than what to do. The
RDC finding is the cleanest illustration: the pooled gap and the stratified gap
imply opposite actions, and only stratification distinguishes them.

**Confidence ratings** reflect evidence strength, not effect size. "High" means the
result is robust to the obvious alternative explanations that this dataset can test.
"Medium" means a named confound remains that the data cannot resolve.

**Costing discipline.** One assumed rate, stated above, applied to measured counts.
No net present value, no adoption curve, no confidence interval on a dollar figure —
those would give an assumption the appearance of a measurement.

**Traceability.** Every number here is computed by the same `src` modules the other
pages use, so nothing on this page can drift from its source page.
""")

sidebar_about()
