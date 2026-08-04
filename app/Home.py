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

import streamlit as st

from src.dashboard.components import (callout, chart, insight, kpi_row,
                                      methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.analytics import funnel, procurement
from src.ml import predict
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_days, fmt_pct, fmt_units

page_setup(
    title="Pharmaceutical Supply Chain Optimization",
    icon="💊",
    subtitle="Drug classification, funnel analytics and statistical A/B testing "
             "across a pharmaceutical supply chain",
)


@st.cache_data(show_spinner="Computing metrics...")
def _metrics() -> dict:
    """Headline numbers from each half of the platform."""
    return {"funnel": funnel.funnel_kpis(), "scms": procurement.scms_kpis()}


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
fk, sk = metrics["funnel"], metrics["scms"]
models = _model_scores()

# ---------------------------------------------------------------------------
# What this project is
# ---------------------------------------------------------------------------
section("What This Project Does")

st.markdown("""
Three connected pieces of analytics work on pharmaceutical data:

1. **A machine learning pipeline** that predicts which drug suits a patient from
   their clinical readings — the full workflow, from data quality checks through
   to a deployed prediction interface.
2. **A supply chain funnel analysis** that traces product through eight stages and
   finds where volume and value are lost.
3. **Statistical A/B tests** that establish which operational fixes actually work,
   rather than which ones sound plausible.
""")

callout(
    "**Two datasets are real, one is simulated — and every page says which.** The "
    "clinical model uses the Kaggle `drug200` dataset. The Real-World Operations "
    "page and the late-delivery model use genuine USAID data: 10,324 actual "
    "shipments to 43 countries. The manufacturing funnel and stability pages use a "
    "calibrated simulation, because no public dataset carries per-batch storage "
    "telemetry.",
    kind="insight", title="Data provenance",
)

# ---------------------------------------------------------------------------
# Headline results
# ---------------------------------------------------------------------------
section("Headline Results")

clinical = models.get("drug_classification")
late = models.get("late_delivery")

kpi_row([
    {"label": "Drug Classification Accuracy",
     "value": fmt_pct(clinical["metrics"]["accuracy"] * 100, 1) if clinical else "-",
     "help_text": "Decision Tree on the Kaggle drug200 dataset"},
    {"label": "Late-Delivery Model AUC",
     "value": f"{late['roc_auc_ovr']:.3f}" if late else "-",
     "help_text": "Trained on 10,324 real USAID shipments"},
    {"label": "Supply Chain Yield",
     "value": fmt_pct(fk["end_to_end_yield_pct"]),
     "help_text": "Units procured that reach a patient (simulated)"},
    {"label": "Real On-Time Delivery",
     "value": fmt_pct(sk["on_time_delivery_pct"]),
     "help_text": f"{sk['shipments']:,} real shipments, {sk['countries']} countries"},
])

# ---------------------------------------------------------------------------
# The funnel
# ---------------------------------------------------------------------------
section(
    "Where Value Is Lost",
    "Every batch is tracked through eight stages. The width of each band is the "
    "volume still in the chain; the gap between bands is where units are lost.",
)

summary = funnel.funnel_summary()
left, right = st.columns([3, 2], gap="large")
with left:
    chart(charts.funnel_chart(summary, title="Units Surviving Each Stage"))
with right:
    chart(charts.funnel_dropoff_chart(summary))

worst = summary.loc[summary["dropoff_pct"].idxmax()]
insight(
    f"**{worst['stage']}** is the largest single point of unit loss, shedding "
    f"**{worst['dropoff_pct']:.1f}%** of incoming volume "
    f"({fmt_units(worst['units_lost'])} units, {fmt_currency(worst['value_lost_usd'])}). "
    f"It is also the slowest stage at {fk['slowest_stage_days']:.0f} days, so it "
    "constrains both yield and cycle time — which is why it is the first "
    "intervention tested on the A/B Testing page."
)

# ---------------------------------------------------------------------------
# Three findings
# ---------------------------------------------------------------------------
section("Three Findings Worth Knowing")

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown(f"""
**Quality testing is the bottleneck**

It loses the most volume *and* takes the longest
({fk['slowest_stage_days']:.0f} days). Its failure modes trace back to incoming
raw material, not to the manufacturing process.

*See: Funnel Analytics*
""")
with col2:
    st.markdown("""
**An internal channel collapsed**

The worst performer is the programme's own distribution centre — and stratifying by
era shows it degraded from 93.4% to 73.9% on-time after 2010, rather than always
having been weak.

*See: Statistical Testing*
""")
with col3:
    st.markdown("""
**Accuracy can be the wrong metric**

The late-delivery model scores below the majority-class baseline, yet reviewing
its top 20% by risk catches 63% of late shipments.

*See: ML Models*
""")

# ---------------------------------------------------------------------------
section("How To Navigate")

nav = [
    ("1. Data Quality", "Profiling, cleaning and the audit trail behind every number"),
    ("2. ML Models", "Drug classification, batch risk and late delivery, with full evaluation"),
    ("3. Funnel Analytics", "Eight-stage conversion, drop-off and bottleneck ranking"),
    ("4. Drug Stability", "How storage conditions destroy potency"),
    ("5. Real-World Operations", "Measured performance on 10,324 real USAID shipments"),
    ("6. A/B Testing", "Four interventions tested for statistical and practical significance"),
    ("7. Insights", "Consolidated recommendations, costed and prioritised"),
    ("8. Google Colab", "The complete reproducible notebook"),
]
import pandas as pd  # noqa: E402  (local: only needed for this small table)

show_table(pd.DataFrame(nav, columns=["Page", "What it covers"]), height=320)

methodology("""
**Reproducibility.** Everything is deterministic under `project.random_seed` in
`config/config.yaml`. From a clean checkout, `python scripts/build_dataset.py`
followed by `python scripts/train_models.py` regenerates every number here.

**One definition per metric.** The dashboard, the notebook and the CLI scripts all
import the same `src` modules, so nothing can drift out of sync.

**Honest labelling.** Simulated figures are labelled as simulated wherever they
appear. The manufacturing simulation exists because SCMS records procurement and
logistics but not manufacturing, and no public dataset carries per-batch storage
telemetry.
""")

sidebar_about()
