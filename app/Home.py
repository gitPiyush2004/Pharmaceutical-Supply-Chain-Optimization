"""
PharmaChain Analytics - executive landing page.

Answers three questions before a reader clicks anything: what the project does,
what it found, and where to look next.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.analytics import pipeline, procurement, products
from src.dashboard.components import (callout, chart, insight, kpi_row,
                                      methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.ml import predict
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct

page_setup(
    title="Pharmaceutical Supply Chain Optimization",
    icon="💊",
    subtitle="Order-to-delivery analytics, statistical testing and machine "
             "learning on two real pharmaceutical datasets",
)


@st.cache_data(show_spinner="Loading real data...")
def _metrics() -> dict:
    """Headline numbers from each dataset."""
    return {
        "pipeline": pipeline.pipeline_kpis(),
        "scms": procurement.scms_kpis(),
        "catalogue": products.catalogue_kpis(),
        "pricing": products.pricing_summary(),
        "concentration": products.value_concentration(),
    }


@st.cache_data(show_spinner=False)
def _model_scores() -> dict:
    """Test metrics from the persisted model artefacts."""
    scores = {}
    for name in ("drug_classification", "late_delivery"):
        try:
            scores[name] = predict.model_summary(name)
        except FileNotFoundError:
            scores[name] = None
    return scores


metrics = _metrics()
pk, sk = metrics["pipeline"], metrics["scms"]
ck, pricing, conc = (metrics["catalogue"], metrics["pricing"],
                     metrics["concentration"])
models = _model_scores()

# ---------------------------------------------------------------------------
# What this project is
# ---------------------------------------------------------------------------
section("What This Project Does")

st.markdown("""
Four connected pieces of analytics work on two real datasets:

1. **Order-to-delivery pipeline analysis** on 10,324 genuine USAID shipments —
   where commodity value arrives late, and which part of the pipeline causes it.
2. **Product catalogue and pricing** on the same rows — what was bought, from which
   factory, and whether the price paid was consistent.
3. **Statistical testing** of the differences those two surface — including a
   *bounded* null result, because "we can rule out anything bigger than 1.8 points"
   is a real answer and "not significant" is not.
4. **Two machine learning pipelines** — clinical drug classification and
   late-delivery prediction — with leakage control and honest evaluation.
""")

callout(
    "**Every number on this dashboard is measured from published data.** There is "
    "no simulation anywhere in this project. Two sources: the Kaggle `drug200` "
    "clinical dataset (200 patients) and the USAID SCMS delivery history (10,324 "
    "shipments to 43 countries, 2006-2015). SCMS carries both the logistics *and* "
    "the product catalogue — molecule, brand, dosage, factory and the price actually "
    "paid — so the delivery and pricing questions are answered on the same rows. "
    "Each page names its source and states what that source cannot tell you.",
    kind="insight", title="Data provenance",
)

# ---------------------------------------------------------------------------
# Headline results
# ---------------------------------------------------------------------------
section("Headline Results")

clinical = models.get("drug_classification")
late = models.get("late_delivery")

kpi_row([
    {"label": "Value Delivered Late",
     "value": fmt_currency(pk["value_late_usd"]),
     "delta": f"{100 - pk['value_on_time_pct']:.1f}% of ordered value",
     "delta_good": False,
     "help_text": f"Of {fmt_currency(pk['value_ordered_usd'])} ordered across "
                  f"{pk['line_items']:,} real line items"},
    {"label": "On-Time Delivery",
     "value": fmt_pct(sk["on_time_delivery_pct"]),
     "help_text": f"{sk['shipments']:,} shipments, {sk['countries']} countries, "
                  f"{sk['vendors']} vendors"},
    {"label": "Drug Classification Accuracy",
     "value": fmt_pct(clinical["metrics"]["accuracy"] * 100, 1) if clinical else "-",
     "help_text": "Decision Tree on drug200, tuned by cross-validated macro-F1"},
    {"label": "Late-Delivery Model AUC",
     "value": f"{late['roc_auc_ovr']:.3f}" if late else "-",
     "help_text": "XGBoost on real SCMS shipments, leakage-controlled features"},
])

# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
section(
    "Where Value Arrives Late",
    "Every line item in this dataset was eventually delivered, so the attrition is "
    "not in volume — it is in timeliness. Each band below is a strictly tighter "
    "definition of on time than the one above it.",
)

funnel = pipeline.value_funnel()
lateness = pipeline.lateness_funnel()
left, right = st.columns([1, 1], gap="large")
with left:
    chart(charts.value_funnel_chart(funnel))
with right:
    chart(charts.lateness_funnel_chart(lateness))

insight(
    f"**{fmt_currency(pk['value_late_usd'])} of commodity value "
    f"({100 - pk['value_on_time_pct']:.1f}%) arrives after its scheduled date.** "
    f"But the tail is short: {lateness.loc[1, 'share_pct']:.1f}% of value is within "
    f"a week of schedule and {lateness.loc[2, 'share_pct']:.1f}% within a month, so "
    f"this is a large number of small slips rather than a few disasters. "
    f"The counterweight is that {pk['early_pct']:.0f}% of shipments arrive *early*, "
    f"which is a holding-cost problem rather than a service success — and "
    f"{pk['exact_day_pct']:.0f}% land on exactly their scheduled day, which is what "
    f"makes the headline rate look better than the planning actually is."
)

# ---------------------------------------------------------------------------
# Three findings
# ---------------------------------------------------------------------------
section("Three Findings Worth Knowing")

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown("""
**An internal channel collapsed**

The worst-performing route is the programme's own distribution centre. Stratifying
by era shows it fell from 93.4% to 73.9% on time after 2010 — it degraded rather
than always having been weak. The pooled 11.9-point gap hides that entirely.

*See: Statistical Testing*
""")
with col2:
    st.markdown("""
**Accuracy can be the wrong metric**

The late-delivery model scores below the majority-class baseline on accuracy, yet
reviewing its top 20% by predicted risk catches a disproportionate share of the
late shipments. Ranking is the deployable output, not the label.

*See: ML Models*
""")
with col3:
    st.markdown(f"""
**The same trap, found twice**

Pooled across ten years, identical products show a {pricing['pooled_median_spread_x']:.1f}x
price spread. Within a single year it is {pricing['within_year_median_spread_x']:.1f}x —
the rest was a market-wide price collapse. The same pooling mistake as the delivery
finding, on a completely different question.

*See: Product & Pricing*
""")

# ---------------------------------------------------------------------------
section("How To Navigate")

nav = [
    ("1. Data Quality", "A type-aware audit of both datasets — and why SCMS "
                        "scores 99% on completeness while 40% of its freight data "
                        "is unusable"),
    ("2. ML Models", "Drug classification and late-delivery prediction, with "
                     "cross-validation, calibration and gains curves"),
    ("3. Delivery Pipeline", "Value-based service funnel, interval decomposition "
                             "and milestone traceability"),
    ("4. Vendor & Logistics", "Vendor scorecards, destination performance, "
                              "transport mode and freight economics"),
    ("5. Product & Pricing", "What was bought, from which factory, at what price - "
                             "and why the pooled price spread is twice the real one"),
    ("6. Statistical Testing", "Real group comparisons, stratification against "
                               "misleading averages, and how to report a null result"),
    ("7. Insights", "Consolidated findings with what each one does and does not "
                    "license you to conclude"),
    ("8. Google Colab", "The complete reproducible notebook"),
]
show_table(pd.DataFrame(nav, columns=["Page", "What it covers"]), height=340)

methodology("""
**Reproducibility.** Everything is deterministic under `project.random_seed` in
`config/config.yaml`. From a clean checkout, `python scripts/fetch_data.py`
downloads and caches the SCMS export and `python scripts/train_models.py`
regenerates every model metric on this dashboard.

**One definition per metric.** The dashboard, the notebook and the CLI scripts all
import the same `src` modules, so no figure can drift out of sync between them.

**Observational, not experimental.** The group comparisons here are differences
between groups that already exist in the data — nobody randomised a shipment into a
fulfilment route. Every comparison on the Statistical Testing page therefore carries
the confound that limits how far it can be read. Naming the confound is the
analytical work; the p-value is the easy part.
""")

sidebar_about()
