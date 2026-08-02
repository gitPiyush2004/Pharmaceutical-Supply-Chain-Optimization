"""
A/B Testing page.

Four candidate operational interventions, each evaluated as a controlled
experiment: two-proportion z-test on the primary success metric, chi-square on the
contingency table, Welch's t-test on processing time, power analysis, segment
breakdown, and a costed recommendation.
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
from src.analytics import ab_testing as ab
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct

page_setup(
    title="Statistical A/B Testing",
    icon="🧬",
    subtitle="Testing operational interventions for statistical and practical significance",
)


@st.cache_data(show_spinner=False)
def _catalogue():
    return ab.list_experiments()


@st.cache_data(show_spinner="Running experiment...")
def _run(key: str, n: int):
    return ab.run_experiment(key, sample_size=n)


@st.cache_data(show_spinner="Running all experiments...")
def _run_all(n: int):
    return ab.run_all_experiments(sample_size=n)


catalogue = _catalogue()

# ---------------------------------------------------------------------------
section(
    "Experiment Portfolio",
    "Four interventions are on the table. All four are evaluated below with the "
    "same statistical protocol, then ranked by whether the evidence supports "
    "adopting them.",
)

sample_size = st.slider(
    "Sample size per arm", min_value=250, max_value=8000, value=2000, step=250,
    help="Larger samples narrow confidence intervals. Watch how a small effect "
         "becomes statistically detectable as this grows - and why practical "
         "significance has to be judged separately.",
)

portfolio = _run_all(sample_size)
show_table(
    portfolio[["experiment", "metric", "control_rate_pct", "treatment_rate_pct",
               "lift_pp", "relative_lift_pct", "p_value", "significant",
               "achieved_power", "annual_value_usd", "verdict"]],
    height=230,
)

adopt = portfolio[portfolio["verdict"].str.startswith("ADOPT")]
total_value = adopt["annual_value_usd"].sum() if len(adopt) else 0.0
kpi_row([
    {"label": "Experiments Run", "value": f"{len(portfolio)}"},
    {"label": "Statistically Significant",
     "value": f"{int(portfolio['significant'].sum())} of {len(portfolio)}"},
    {"label": "Recommended for Adoption", "value": f"{len(adopt)}"},
    {"label": "Combined Annual Value", "value": fmt_currency(total_value),
     "help_text": "Sum of adopt-recommended interventions"},
])

chart(charts.bar_chart(
    portfolio.sort_values("annual_value_usd"), x="experiment", y="annual_value_usd",
    title="Estimated Annual Value by Intervention", orientation="h", height=380))

# ---------------------------------------------------------------------------
section("Experiment Detail", "Full statistical workup for a single intervention.")

labels = {row["name"]: row["key"] for _, row in catalogue.iterrows()}
choice = st.selectbox("Experiment", list(labels))
key = labels[choice]
result = _run(key, sample_size)

experiment = result["experiment"]
summary = result["summary"]
z = result["z_test"]
chi = result["chi_square"]
t = result["t_test"]
power = result["power_analysis"]
impact = result["business_impact"]
recommendation = result["recommendation"]

st.caption(experiment["description"])
st.markdown(verdict_badge(recommendation["verdict"]), unsafe_allow_html=True)

# --- Primary result ---------------------------------------------------------
kpi_row([
    {"label": "Control Rate", "value": fmt_pct(z["p_control"] * 100, 2),
     "help_text": f"95% CI {z['control_ci']}"},
    {"label": "Treatment Rate", "value": fmt_pct(z["p_treatment"] * 100, 2),
     "help_text": f"95% CI {z['treatment_ci']}"},
    {"label": "Absolute Lift", "value": f"{z['absolute_diff_pp']:+.2f} pp",
     "delta": f"95% CI [{z['ci_lower_pp']:+.2f}, {z['ci_upper_pp']:+.2f}] pp",
     "delta_good": z["absolute_diff_pp"] > 0},
    {"label": "Relative Lift", "value": f"{z['relative_lift_pct']:+.2f}%"},
])

col1, col2 = st.columns(2, gap="large")
with col1:
    chart(charts.ab_comparison_chart(summary, metric_name=experiment["metric"],
                                     title=f"{experiment['metric']}: Control versus Treatment"))
with col2:
    chart(charts.significance_chart(z))

verdict_kind = ("success" if recommendation["verdict"].startswith("ADOPT")
                else "danger" if recommendation["verdict"] == "DO NOT ADOPT"
                else "warning")
callout(
    f"**{recommendation['verdict']}** (confidence: {recommendation['confidence']}). "
    f"{recommendation['rationale']}",
    kind=verdict_kind, title="Recommendation",
)

# --- Statistical tests ------------------------------------------------------
section("Statistical Evidence")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Two-Proportion z-Test", "Chi-Square", "Processing Time (t-Test)",
     "Power Analysis", "Segments"])

with tab1:
    st.markdown(f"**{z['test']}**")
    show_table(pd.DataFrame([{
        "control_rate": f"{z['p_control']:.4f}",
        "treatment_rate": f"{z['p_treatment']:.4f}",
        "absolute_diff": f"{z['absolute_diff']:.4f}",
        "z_statistic": f"{z['z_statistic']:.4f}",
        "p_value": f"{z['p_value']:.3e}",
        "ci_95": f"[{z['ci_lower']:.4f}, {z['ci_upper']:.4f}]",
        "alpha": z["alpha"],
        "significant": z["significant"],
    }]))
    st.markdown(
        f"With p = {z['p_value']:.3e} against alpha = {z['alpha']}, the difference "
        f"**{'is' if z['significant'] else 'is not'}** statistically significant. "
        "The confidence interval is the more useful output: it bounds how large the "
        "true effect plausibly is, which is what a business case needs."
    )

with tab2:
    st.markdown(f"**{chi['test']}**")
    kpi_row([
        {"label": "Chi-Square", "value": f"{chi['chi2_statistic']:.4f}"},
        {"label": "p-value", "value": f"{chi['p_value']:.3e}"},
        {"label": "Degrees of Freedom", "value": f"{chi['degrees_of_freedom']}"},
        {"label": "Cramér's V", "value": f"{chi['cramers_v']:.4f}",
         "help_text": "Effect size for association strength"},
    ])
    st.markdown("**Observed contingency table**")
    show_table(pd.DataFrame(chi["contingency_table"]))
    st.markdown("**Expected frequencies under independence**")
    show_table(pd.DataFrame(chi["expected_frequencies"]).round(1))
    st.caption(
        "The chi-square test asks the same question as the z-test from a different "
        "angle - whether arm and outcome are independent. Agreement between the two "
        "is a consistency check; Cramér's V quantifies how strong the association is "
        "rather than merely whether it exists."
    )

with tab3:
    st.markdown(f"**{t['test']}** on `{t['metric']}`")
    kpi_row([
        {"label": "Control Mean", "value": f"{t['control_mean']:.2f}",
         "help_text": f"SD {t['control_sd']:.2f}"},
        {"label": "Treatment Mean", "value": f"{t['treatment_mean']:.2f}",
         "help_text": f"SD {t['treatment_sd']:.2f}"},
        {"label": "Difference", "value": f"{t['mean_difference']:+.2f}",
         "delta": f"{t['pct_change']:+.2f}%",
         "delta_good": t["mean_difference"] < 0},
        {"label": "p-value", "value": f"{t['p_value']:.3e}",
         "help_text": f"Cohen's d {t['cohens_d']:.3f}"},
    ])
    st.caption(
        "A secondary continuous metric. An intervention that raises the success rate "
        "but also lengthens processing time may not be worth adopting, so both are "
        "tested rather than only the headline proportion."
    )

with tab4:
    kpi_row([
        {"label": "Required n per Arm", "value": f"{power['n_per_arm']:,}",
         "help_text": f"For {int(power['power'] * 100)}% power at alpha "
                      f"{power['alpha']}"},
        {"label": "Required n Total", "value": f"{power['n_total']:,}"},
        {"label": "Actual n per Arm", "value": f"{sample_size:,}"},
        {"label": "Achieved Power", "value": fmt_pct(result["achieved_power"] * 100),
         "delta_good": result["achieved_power"] >= power["power"]},
    ])
    adequate = sample_size >= power["n_per_arm"]
    callout(
        f"The current sample of {sample_size:,} per arm "
        f"**{'meets' if adequate else 'falls short of'}** the "
        f"{power['n_per_arm']:,} needed to detect an effect of this size with "
        f"{int(power['power'] * 100)}% power. "
        + ("The experiment is adequately powered."
           if adequate else
           "An inconclusive result here would be uninformative rather than "
           "evidence of no effect - increase the sample before drawing conclusions."),
        kind="success" if adequate else "warning",
        title="Power adequacy",
    )
    st.caption(f"Effect size (Cohen's h): {power['effect_size_cohens_h']:.4f}")

with tab5:
    segments = result["segments"]
    chart(charts.segment_effect_chart(segments, dimension="region"))
    show_table(segments, height=260)
    significant_segments = segments[segments["significant"]]
    st.caption(
        f"{len(significant_segments)} of {len(segments)} regions show a "
        "individually significant effect. Segment-level tests are exploratory: with "
        "five regions tested at alpha 0.05 the chance of at least one false positive "
        "is roughly 23%, so these guide rollout sequencing rather than justify "
        "separate decisions."
    )

# --- Business impact --------------------------------------------------------
section("Business Impact")

col1, col2 = st.columns([2, 3], gap="large")
with col1:
    kpi_row([{"label": "Annual Value",
              "value": fmt_currency(impact["annual_value_usd"]),
              "help_text": "Value recovered if rolled out network-wide"}])
    kpi_row([{"label": "Units Recovered Annually",
              "value": f"{impact['units_recovered_annually']:,.0f}"}])
    kpi_row([{"label": "Basis", "value": f"{impact['years_of_history']:.1f} yr",
              "help_text": f"Avg unit cost ${impact['avg_unit_cost_usd']:.2f}"}])
with col2:
    st.markdown("**Supporting evidence**")
    for line in recommendation["supporting_evidence"]:
        st.markdown(f"- {line}")
    st.markdown(
        f"**Practical significance threshold:** "
        f"{recommendation['practical_threshold_pct']}% relative lift - "
        f"{'met' if recommendation['practical_significance_met'] else 'not met'}"
    )
    st.caption(impact["calculation_basis"])

insight(
    "A result has to clear three bars before it justifies capital: statistical "
    "significance (the effect is unlikely to be noise), adequate power (the test "
    "could have found the effect if it existed), and practical significance (the "
    "effect is large enough to matter commercially). Reporting all three is what "
    "separates an experiment from a dashboard number - a large sample makes almost "
    "any difference significant, and that alone is never a reason to spend money."
)

download_button(portfolio, "ab_test_portfolio.csv", "Download experiment portfolio")

# ---------------------------------------------------------------------------
methodology("""
**Design.** Each experiment is a randomised two-arm test. Subjects are batches or
shipments assigned to control or treatment with equal probability, using the
platform seed so results are reproducible. Control and treatment rates come from
`config/config.yaml` (`ab_testing.experiments`) and are set from published
benchmarks for each intervention class.

**Primary test.** Two-proportion z-test on the success metric, using the pooled
proportion for the standard error under the null. The confidence interval uses
unpooled standard errors, which is the standard convention for interval estimation.

**Secondary tests.**
- **Chi-square** on the 2x2 contingency table, with Cramér's V as the effect size.
  It answers the same question independently; agreement is a consistency check.
- **Welch's t-test** on processing time, a continuous secondary metric. Welch's is
  used rather than Student's because equal variances cannot be assumed.

**Power.** Required sample size is computed from Cohen's h for the configured
effect at `ab_testing.alpha` (0.05) and `ab_testing.power` (0.80). Achieved power
is computed for the sample actually used, so an inconclusive result can be
correctly read as under-powered rather than as evidence of no effect.

**Practical significance.** Statistical significance alone does not justify
adoption. A minimum relative lift threshold is applied, and the recommendation
requires significance, adequate power *and* practical significance before
returning ADOPT.

**Value.** Impact is the recovered units - the lift applied to observed annual
volume - priced at average unit cost. It excludes implementation cost, so it is a
gross benefit, not a net business case or an ROI.

**Honest framing.** These are simulated experiments on synthetic operational data.
They demonstrate the experimental method and the decision framework; they are not
evidence about real interventions.
""")

sidebar_about()
