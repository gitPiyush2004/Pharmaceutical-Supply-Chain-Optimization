"""
Drug Stability page.

How storage conditions destroy product value. Temperature, humidity and exposure
time are each isolated, then combined into a fitted degradation model that yields
a shelf-life estimate and quantifies the cold-chain excursion problem.
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
                                      show_table, sidebar_about, sidebar_filters,
                                      verdict_badge)
from src.analytics import stability
from src.data import loader
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct

page_setup(
    title="Drug Stability Analytics",
    icon="🌡️",
    subtitle="Temperature, humidity and duration effects on potency, shelf life and expiry risk",
)


@st.cache_data(show_spinner="Loading batch telemetry...")
def _batches():
    return loader.load_batches()


batches = sidebar_filters(
    _batches(), ["region", "drug_code", "batch_risk_label", "form"])
if batches.empty:
    st.stop()

kpis = stability.stability_kpis(batches)

# ---------------------------------------------------------------------------
section("Stability Position")
kpi_row([
    {"label": "Average Potency", "value": fmt_pct(kpis["avg_potency_pct"], 2),
     "help_text": f"Specification floor {kpis['potency_spec_min']:.0f}% of label claim"},
    {"label": "Out of Specification", "value": fmt_pct(kpis["out_of_spec_pct"]),
     "help_text": f"Below warning level: {fmt_pct(kpis['below_warning_pct'])}"},
    {"label": "High-Risk Batches", "value": fmt_pct(kpis["high_risk_batch_pct"]),
     "help_text": "Classified High by the batch risk model"},
    {"label": "Value at Risk", "value": fmt_currency(kpis["value_at_risk_usd"]),
     "help_text": "Product carrying stability or expiry risk"},
])

# ---------------------------------------------------------------------------
section(
    "The Cold-Chain Excursion Problem",
    "A documented temperature excursion is the single largest stability event in "
    "this network. This quantifies it with a formal significance test.",
)

impact = stability.excursion_impact(batches)

col1, col2 = st.columns([2, 3], gap="large")
with col1:
    kpi_row([
        {"label": "Potency With Excursion",
         "value": fmt_pct(impact["mean_potency_with_excursion"], 2),
         "help_text": f"{impact['n_with_excursion']} batches"},
    ])
    kpi_row([
        {"label": "Potency Without",
         "value": fmt_pct(impact["mean_potency_without_excursion"], 2),
         "help_text": f"{impact['n_without_excursion']:,} batches"},
    ])
    kpi_row([
        {"label": "Difference",
         "value": f"{impact['potency_difference_pct']:.2f} pp",
         "delta": f"p = {impact['p_value']:.2e}",
         "delta_good": False,
         "help_text": f"95% CI [{impact['difference_ci_low']:.2f}, "
                      f"{impact['difference_ci_high']:.2f}]"},
    ])
with col2:
    comparison = pd.DataFrame({
        "cohort": ["No excursion", "Excursion recorded"],
        "avg_potency_pct": [impact["mean_potency_without_excursion"],
                            impact["mean_potency_with_excursion"]],
        "out_of_spec_pct": [impact["out_of_spec_pct_without_excursion"],
                            impact["out_of_spec_pct_with_excursion"]],
    })
    chart(charts.bar_chart(comparison, x="cohort", y="avg_potency_pct",
                           title="Mean Potency by Excursion Status (%)",
                           text_format=".2f", height=340))

st.markdown(verdict_badge("High" if impact["significant"] else "Medium"),
            unsafe_allow_html=True)
callout(
    f"Batches with a recorded excursion average "
    f"**{impact['mean_potency_with_excursion']:.2f}%** "
    f"potency against **{impact['mean_potency_without_excursion']:.2f}%** for the rest - "
    f"a gap of **{abs(impact['potency_difference_pct']):.2f} percentage points** "
    f"(Welch t = {impact['t_statistic']:.2f}, p = {impact['p_value']:.2e}, "
    f"Cohen's d = {impact['cohens_d']:.2f}). Out-of-specification rate rises from "
    f"{fmt_pct(impact['out_of_spec_pct_without_excursion'])} to "
    f"{fmt_pct(impact['out_of_spec_pct_with_excursion'])}, a lift of "
    f"{impact['oos_rate_lift_pp']:.1f} points. Attributable value at risk is "
    f"{fmt_currency(impact['attributable_value_at_risk_usd'])}. {impact['verdict']}",
    kind="danger" if impact["significant"] else "warning",
    title="Excursion impact - statistically significant" if impact["significant"]
    else "Excursion impact - not significant",
)
insight(
    "This is the business case for the IoT cold-chain monitoring experiment on the "
    "A/B Testing page. Excursions affect a small share of batches but do "
    "disproportionate damage, and because they are detectable in real time they are "
    "among the most tractable losses in the whole network."
)

# ---------------------------------------------------------------------------
section(
    "Isolating the Drivers",
    "Each storage condition is binned and plotted against mean potency and "
    "out-of-specification rate, split by cold-chain and ambient product.",
)

tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "Storage Duration"])
with tab1:
    effect = stability.temperature_effect(batches)
    chart(charts.condition_effect_chart(effect, x_col="bin_mid",
                                        title="Potency versus Storage Temperature",
                                        x_label="Storage temperature (°C)"))
    show_table(effect, height=300)
    insight(
        "Potency falls monotonically with temperature and the out-of-specification "
        "rate rises sharply beyond the labelled storage limit. Cold-chain product "
        "sits in a narrow validated band, so its exposure comes from excursions "
        "rather than from gradual drift - which is why the two cohorts behave "
        "differently and are shown separately."
    )
with tab2:
    effect = stability.humidity_effect(batches)
    chart(charts.condition_effect_chart(effect, x_col="bin_mid",
                                        title="Potency versus Relative Humidity",
                                        x_label="Relative humidity (%RH)"))
    show_table(effect, height=300)
    insight(
        "Humidity has a threshold effect rather than a linear one: below roughly "
        "60% RH it barely matters, above it moisture uptake accelerates degradation. "
        "That makes desiccant packaging and controlled-humidity storage a "
        "targeted fix for hot, humid regions rather than a network-wide spend."
    )
with tab3:
    effect = stability.duration_effect(batches)
    chart(charts.condition_effect_chart(effect, x_col="bin_mid",
                                        title="Potency versus Storage Duration",
                                        x_label="Storage duration (days)"))
    show_table(effect, height=300)
    insight(
        "Exposure time is the driver management controls most directly. Every day "
        "removed from warehouse dwell or quality-testing turnaround is a day of "
        "degradation avoided - which is why the funnel bottleneck and the stability "
        "problem share a single remedy."
    )

# ---------------------------------------------------------------------------
section(
    "Fitted Degradation Model",
    "An OLS regression of potency on the three storage conditions plus cold-chain "
    "status, solved for the duration at which potency crosses the specification limit.",
)

model = stability.shelf_life_model(batches)
col1, col2 = st.columns([2, 3], gap="large")
with col1:
    kpi_row([{"label": "R-squared", "value": f"{model['r_squared']:.4f}",
              "help_text": f"Adjusted {model['adj_r_squared']:.4f}, "
                           f"n = {model['n_observations']:,}"}])
    kpi_row([{"label": "Estimated Shelf Life",
              "value": f"{model['estimated_shelf_life_days']:.0f} d",
              "help_text": "At reference conditions, to the specification floor"}])
with col2:
    coefficients = pd.DataFrame({
        "term": list(model["coefficients"]),
        "coefficient": list(model["coefficients"].values()),
        "p_value": [model["p_values"].get(k) for k in model["coefficients"]],
    })
    show_table(coefficients, height=240)

by_cohort = model.get("estimated_shelf_life_days_by_cohort")
if isinstance(by_cohort, dict) and by_cohort:
    st.markdown("**Estimated shelf life by cohort (days at reference conditions)**")
    show_table(pd.DataFrame({"cohort": list(by_cohort),
                             "estimated_shelf_life_days": list(by_cohort.values())}))

insight(
    f"The model explains **{model['r_squared']:.1%}** of potency variance from "
    "storage conditions alone, and every coefficient carries the physically expected "
    "sign: higher temperature, higher humidity and longer exposure all reduce "
    "potency. Solving for the specification crossing gives an empirical shelf-life "
    f"estimate of **{model['estimated_shelf_life_days']:.0f} days** at reference "
    "conditions - derived from observed batch outcomes rather than assumed from the "
    "label."
)

with st.expander("Predict potency for a hypothetical storage scenario"):
    c1, c2, c3, c4 = st.columns(4)
    temp = c1.number_input("Temperature (°C)", 1.0, 45.0, 25.0, 0.5, key="sl_temp")
    hum = c2.number_input("Humidity (%RH)", 10.0, 95.0, 55.0, 1.0, key="sl_hum")
    dur = c3.number_input("Duration (days)", 1, 400, 60, 1, key="sl_dur")
    cold = c4.selectbox("Cold chain", [0, 1], key="sl_cold",
                        format_func=lambda v: "Yes" if v else "No")
    predicted = model["predict"](temp, hum, dur, cold)
    st.metric("Predicted potency", f"{predicted:.2f}%",
              delta=f"{predicted - kpis['potency_spec_min']:+.2f} pp vs specification")

# ---------------------------------------------------------------------------
section("Portfolio and Expiry Risk")

tab1, tab2, tab3 = st.tabs(["By Product", "Expiry Buckets", "Potency Distribution"])
with tab1:
    performance = stability.batch_performance(batches)
    show_table(performance, height=260)
    chart(charts.bar_chart(performance.sort_values("out_of_spec_pct", ascending=False),
                           x="brand_name", y="out_of_spec_pct",
                           title="Out-of-Specification Rate by Product (%)",
                           text_format=".1f", height=340))
    download_button(performance, "batch_performance.csv", "Download product stability")
with tab2:
    profile = stability.expiry_risk_profile(batches)
    show_table(profile, height=240)
    chart(charts.bar_chart(profile, x="bucket", y="net_value_at_risk_usd",
                           title="Net Value at Risk by Remaining Shelf Life", height=360))
with tab3:
    chart(charts.potency_distribution_chart(batches))
    chart(charts.risk_distribution_chart(
        batches["batch_risk_label"].value_counts()))

# ---------------------------------------------------------------------------
methodology("""
**Degradation model.** Potency is generated by, and analysed with, a simplified
Arrhenius-style relationship: degradation scales with exposure time, accelerates
with temperature above the product's labelled storage condition, and is amplified
by relative humidity above a moisture-uptake threshold. This mirrors ICH Q1A
stability-testing intuition closely enough to be realistic while staying
interpretable - the coefficients mean something physical.

**Reference conditions.** Cold-chain products are referenced to
`stability.reference_temp_cold` (5°C, the middle of the 2-8°C band); ambient
products to `stability.reference_temp_ambient` (25°C, controlled room
temperature). Only excess above the reference contributes to degradation, which is
why the two cohorts are never pooled.

**Specification limits.** `stability.potency_spec_min` (95% of label claim) is the
out-of-specification threshold; `stability.potency_warning` (97%) is the early
warning band.

**Excursion test.** Cold-chain excursions are compared against non-excursion
batches with Welch's two-sample t-test, which does not assume equal variances -
appropriate here because the excursion cohort is both smaller and more dispersed.
Cohen's d reports effect size, because with thousands of batches a trivial
difference can still be statistically significant.

**Shelf-life estimate.** The fitted OLS model is solved for the storage duration at
which predicted potency crosses the specification floor, holding temperature and
humidity at reference. This is an empirical estimate from observed outcomes, not a
regulatory shelf-life determination - a real filing requires formal stability
studies under ICH protocols.
""")

sidebar_about()
