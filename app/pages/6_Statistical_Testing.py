"""
Statistical Testing page - real group comparisons on real data.

What changed and why
--------------------
This page used to run four simulated A/B experiments whose control and treatment
rates were written into `config/config.yaml`. Every one came back significant,
because the effect had been put there by hand. That is circular, so it is gone.

What replaces it is harder and more honest: comparisons between groups that
genuinely exist in the USAID SCMS delivery history. The cost of that honesty is
randomisation - nobody assigned a shipment to a fulfilment route, so these are
observational comparisons and every one carries a confound that limits how far it
can be read. Stating the confound is the analytical work. The p-value is the easy
part, and this page is organised to make that point.
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
from src.config import get_config
from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about, verdict_badge)
from src.viz import charts

page_setup(
    title="Statistical Testing",
    icon="🔬",
    subtitle="Two-proportion z-tests, chi-square, Welch and Mann-Whitney on "
             "10,324 real shipments",
)

cfg = get_config()


@st.cache_data(show_spinner="Running comparisons on real data...")
def _catalogue() -> pd.DataFrame:
    return ex.comparison_catalogue()


@st.cache_data(show_spinner=False)
def _comparison(dimension: str) -> dict:
    """One full comparison, with power and a decision attached."""
    result = ex.compare_groups(dimension=dimension)
    rates, z = result["rates"], result["pairwise_z"]
    worst, best = rates.iloc[0], rates.iloc[-1]

    pooled = 1 - rates["late"].sum() / rates["n"].sum()
    power = ab.achieved_power(
        int(worst["n"] - worst["late"]), int(worst["n"]),
        int(best["n"] - best["late"]), int(best["n"]))
    mde = ab.minimum_detectable_effect(pooled, int(worst["n"]), int(best["n"]))

    # Only the on-time metric has an SLA interpretation, so only it gets costed.
    years = 10  # 2006-2015 inclusive, from the dataset's own coverage
    impact = ab.penalty_impact(abs(z["absolute_diff"]), rates["n"].sum() / years)

    result["achieved_power"] = power
    result["mde"] = mde
    result["impact"] = impact
    result["recommendation"] = ab.business_recommendation(
        z, power, impact=impact, mde=mde,
        supporting=[f"Confound: {result['confound']}"] if result["confound"] else None)
    return result


@st.cache_data(show_spinner=False)
def _stratified() -> dict:
    return ex.stratified_comparison()


@st.cache_data(show_spinner=False)
def _continuous(metric: str, dimension: str) -> dict:
    return ex.continuous_comparison(metric=metric, dimension=dimension)


catalogue = _catalogue()

callout(
    "**These are observational comparisons, not randomised experiments.** Nobody "
    "assigned a shipment to a transport mode or a fulfilment route, so a difference "
    "between two groups does not on its own establish that the grouping *caused* it. "
    "Every comparison below therefore names its confound, and the stratification "
    "section shows what happens when you skip that step.",
    kind="warning", title="What these tests can and cannot establish",
)

# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------
section(
    "Every Comparison, Ranked",
    "On-time delivery rate compared across each available dimension, with a "
    "chi-square test of independence and the confound that limits the reading.",
)

show_table(catalogue)

significant = int(catalogue["significant"].sum())
insight(
    f"**{significant} of {len(catalogue)} dimensions show a significant difference "
    f"in on-time delivery.** The largest is destination region at "
    f"{catalogue.loc[0, 'gap_pp']:.1f} points, but the most *actionable* is "
    f"fulfilment route at {catalogue.loc[1, 'gap_pp']:.1f} points, because that is a "
    f"choice the programme controls.\n\n"
    f"The last row is the interesting one. First-line designation shows a "
    f"{catalogue.iloc[-1]['gap_pp']:.2f}-point gap at p = "
    f"{catalogue.iloc[-1]['chi_square_p']:.2f} — a genuine non-result on 10,324 "
    f"observations, and worth more than it looks. See the null-results section below."
)

# ---------------------------------------------------------------------------
# One comparison in full
# ---------------------------------------------------------------------------
section(
    "One Comparison In Full",
    "Pick a dimension to see the rates, the test output, the effect size and the "
    "decision it supports.",
)

dimension = st.selectbox(
    "Dimension to compare", list(ex.COMPARISON_DIMENSIONS),
    format_func=lambda d: d.replace("_", " ").title(),
    help="Every option is a column in the real SCMS dataset.")

result = _comparison(dimension)
rates, z, chi = result["rates"], result["pairwise_z"], result["chi_square"]
rec = result["recommendation"]

kpi_row([
    {"label": "Best Group",
     "value": f"{rates.iloc[-1]['on_time_pct']:.2f}%",
     "help_text": f"{rates.iloc[-1]['level']} (n={rates.iloc[-1]['n']:,})"},
    {"label": "Worst Group",
     "value": f"{rates.iloc[0]['on_time_pct']:.2f}%",
     "help_text": f"{rates.iloc[0]['level']} (n={rates.iloc[0]['n']:,})"},
    {"label": "Gap",
     "value": f"{z['absolute_diff_pp']:+.2f} pp",
     "help_text": f"95% CI [{z['ci_lower_pp']:+.2f}, {z['ci_upper_pp']:+.2f}] pp"},
    {"label": "Chi-Square p",
     "value": f"{chi['p_value']:.2e}",
     "help_text": f"Cramer's V {chi['cramers_v']:.3f}, "
                  f"dof {chi['degrees_of_freedom']}"},
])

col_left, col_right = st.columns([3, 2], gap="large")
with col_left:
    chart(charts.rate_comparison_chart(
        rates, title=f"On-Time Rate by {dimension.replace('_', ' ').title()}",
        target=float(cfg.scms.on_time_target_pct)))
with col_right:
    chart(charts.significance_chart(z))
    st.markdown(f"**Decision:** {verdict_badge(rec['verdict'])}")
    st.markdown(rec["rationale"])

if result["confound"]:
    callout(f"**Confound:** {result['confound']}", kind="warning",
            title="What limits this reading")

with st.expander("Full test output"):
    st.markdown("**Two-proportion z-test** (best versus worst group)")
    show_table(pd.DataFrame([{k: v for k, v in z.items()
                              if not isinstance(v, (tuple, dict))}]).T
               .rename(columns={0: "value"}))
    st.markdown("**Chi-square contingency table** (observed)")
    show_table(chi["contingency_table"].rename(
        columns={0: "On time", 1: "Late"}).reset_index())
    st.markdown(
        f"Minimum expected cell count is **{chi['min_expected_count']:.1f}**. "
        + ("Above 5, so the chi-square approximation is sound."
           if chi["expected_counts_adequate"] else
           "**Below 5**, so the chi-square p-value here is unreliable and the "
           "z-test on the two largest groups should be preferred."))
    if result["levels_excluded"]:
        st.markdown(
            f"Levels excluded for having fewer than {ex.MIN_GROUP_SIZE} shipments, "
            f"or for being recording artefacts: "
            f"`{'`, `'.join(result['levels_excluded'])}`")

# ---------------------------------------------------------------------------
# Simpson's paradox
# ---------------------------------------------------------------------------
section(
    "The Trap: A Pooled Gap That Does Not Exist",
    "The single most important result on this page, and the one that would have "
    "been reported wrongly without stratification.",
)

strat = _stratified()
col_a, col_b = st.columns([3, 2], gap="large")
with col_a:
    chart(charts.stratified_effect_chart(
        strat["strata"], strat["dimension"], strat["stratified_by"],
        title="Fulfilment Route Performance Within Each Era"))
with col_b:
    show_table(strat["strata"])

callout(strat["verdict"], kind="insight", title="Why the pooled number misleads")

st.markdown(f"""
**On naming this correctly.** This is *effect modification* — a strong interaction
between fulfilment route and era — and it is **not** textbook Simpson's paradox,
which requires the difference to reverse sign. Direct drop is ahead in both eras;
what changes is the magnitude, by roughly a factor of ten.

The code distinguishes the two: `is_simpsons_paradox` is
`{strat['is_simpsons_paradox']}` because there is no sign flip, while
`interaction_detected` is `{strat['interaction_detected']}`. Both make a pooled
average misleading in the same way and for the same reason, so the practical
consequence is identical — but claiming a reversal that did not happen would be an
overstatement, and the distinction is cheap to get right.
""")

insight(
    "**The pooled 11.9-point gap between fulfilment routes is not a stable "
    "structural difference — it is a collapse that happened after 2010.** Before "
    "2011 the two routes were within 1.9 points of each other. After 2010 the gap is "
    "20.5 points, because regional-distribution-centre performance fell from 93.4% "
    "to 73.9% while direct-drop held at roughly 94%.\n\n"
    "Those two readings imply opposite actions. A constant 12-point gap says *stop "
    "using the RDC channel*. A post-2010 collapse says *the channel worked, "
    "something changed in 2011, and finding out what is the highest-value question "
    "here* — capacity, staffing, funding, or a shift in which countries were served "
    "that way. Reporting the pooled figure would have sent an investigation in "
    "entirely the wrong direction."
)

# ---------------------------------------------------------------------------
# Null results
# ---------------------------------------------------------------------------
section(
    "Reporting A Null Result Properly",
    "A non-significant result is only informative if the test could have detected "
    "something worth acting on. That takes a minimum-detectable-effect calculation, "
    "not a post-hoc power figure.",
)

null_result = _comparison("first_line_designation")
null_z, null_mde, null_rec = (null_result["pairwise_z"], null_result["mde"],
                              null_result["recommendation"])

kpi_row([
    {"label": "Measured Gap", "value": f"{null_z['absolute_diff_pp']:+.2f} pp",
     "help_text": f"p = {null_z['p_value']:.3f} — not significant"},
    {"label": "Detectable Gap", "value": f"{null_mde['mde_pp']:+.2f} pp",
     "help_text": "Smallest difference this sample could have found at 80% power"},
    {"label": "Post-Hoc Power",
     "value": f"{null_result['achieved_power']:.0%}",
     "help_text": "Deliberately shown to be dismissed — see below"},
    {"label": "Verdict", "value": null_rec["verdict"],
     "help_text": null_rec["confidence"] + " confidence"},
])

st.markdown(f"""
**Post-hoc power is the wrong tool here, and the numbers above show why.** It reads
{null_result['achieved_power']:.0%}, which looks damning — but post-hoc power is
computed *at the observed effect size*, so it is a deterministic function of the
p-value. A null result mechanically returns low observed power every single time.
Using it to judge a null is circular reasoning dressed up as a statistic.

The right question is: **given these sample sizes, how large a gap would have shown
up?** That is the minimum detectable effect, and it is
{abs(null_mde['mde_pp']):.2f} points. So this comparison rules out any difference
larger than {abs(null_mde['mde_pp']):.2f} points — a real, quotable bound on
3,294 versus 7,030 shipments.

What it cannot quite do is close the case. A difference worth acting on at this
baseline is {null_rec['practical_threshold_pp']:.2f} points, which is just *inside*
the detectable limit. So the honest report is
"**{null_rec['verdict']}** — no difference above {abs(null_mde['mde_pp']):.2f}
points, which does not fully exclude one small enough to matter marginally", rather
than a flat "no difference".
""")

callout(
    "This is the result most worth discussing in an interview. Anyone can report a "
    "significant finding. Reporting a bounded null — and knowing why post-hoc power "
    "cannot be used to support it — is the part that distinguishes reading a test "
    "output from understanding one.",
    kind="insight", title="Why this section exists",
)

# ---------------------------------------------------------------------------
# Continuous metrics
# ---------------------------------------------------------------------------
section(
    "Continuous Metrics: When The Tests Disagree",
    "Welch's t-test and Mann-Whitney U on the same data, with the skew that decides "
    "which to quote.",
)

metric_choices = {
    "Freight as % of commodity value — by product group": ("freight_pct_of_value", "product_group"),
    "Delivery delay (days) — by era": ("delivery_delay_days", "era"),
    "Quote-to-delivery lead time (days) — by fulfilment route": ("total_lead_time_days", "fulfil_via"),
    "Freight cost (USD) — by era": ("freight_cost_usd", "era"),
}
choice = st.selectbox("Comparison", list(metric_choices))
metric, dim = metric_choices[choice]
cont = _continuous(metric, dim)

if cont.get("welch") is None:
    callout(cont["verdict"], kind="warning", title="Comparison not available")
else:
    welch, rank = cont["welch"], cont["mann_whitney"]
    kpi_row([
        {"label": "Skew", "value": f"{cont['max_abs_skew']:.1f}",
         "help_text": f"Threshold is {cont['skew_limit']:.1f}; above it, means stop "
                      f"being a fair summary"},
        {"label": "Welch p", "value": f"{welch['p_value']:.2e}",
         "help_text": f"Means {welch['mean_a']:,.1f} vs {welch['mean_b']:,.1f}, "
                      f"Cohen's d {welch['cohens_d']:.2f}"},
        {"label": "Mann-Whitney p", "value": f"{rank['p_value']:.2e}",
         "help_text": f"Medians {rank['median_a']:,.1f} vs {rank['median_b']:,.1f}, "
                      f"rank-biserial {rank['rank_biserial']:.3f}"},
        {"label": "Quote", "value": cont["recommended_test"].replace("_", "-"),
         "help_text": "Which result to report"},
    ])
    callout(cont["verdict"], kind="insight", title="Which test to believe")

    coverage = cont.get("coverage", {})
    if coverage and min(coverage.values()) < 0.95:
        st.markdown(
            "**Coverage note.** This metric is not recorded for every line item: "
            + ", ".join(f"`{k}` {100 * v:.1f}%" for k, v in coverage.items())
            + ". The test runs on the rows that have it, so the comparison describes "
              "the recorded subset rather than the whole programme.")

# ---------------------------------------------------------------------------
# Power planning
# ---------------------------------------------------------------------------
section(
    "Sample Size Planning",
    "Run *before* committing to a measurement. A test without the power to detect "
    "the effect you care about will return 'not significant' regardless of the truth.",
)

col_c, col_d, col_e = st.columns(3)
with col_c:
    baseline = st.number_input("Baseline rate (%)", 50.0, 99.9, 88.5, 0.5)
with col_d:
    target_lift = st.number_input("Improvement to detect (pp)", 0.1, 20.0, 2.0, 0.1)
with col_e:
    st.metric("Significance / power",
              f"{cfg.ab_testing.alpha} / {cfg.ab_testing.power}")

plan = ab.required_sample_size(baseline / 100, (baseline + target_lift) / 100)
st.markdown(
    f"Detecting a **{target_lift:.1f} percentage point** improvement on a "
    f"**{baseline:.1f}%** baseline at {cfg.ab_testing.power:.0%} power needs "
    f"**{plan['n_per_arm']:,} shipments per arm** "
    f"({plan['n_total']:,} total; Cohen's h = "
    f"{plan['effect_size_cohens_h']:.4f}). This dataset has 10,324 line items in "
    f"total, which is what makes the smaller gaps on this page measurable at all — "
    f"and what makes the sub-1-point ones out of reach."
)

download_button(catalogue, "comparison_catalogue.csv",
                label="Download all comparisons")

methodology(f"""
**Why both a z-test and a chi-square test.** For a 2x2 table they are equivalent
(chi2 = z squared), so reporting both is a consistency check: if they ever disagree,
the contingency table was built wrong. The z-test additionally gives a signed
confidence interval on the difference, which is what a decision needs.

**Yates' correction is disabled** on the chi-square. SciPy applies it to 2x2 tables
by default, which would break the equivalence above, and it is over-conservative at
these sample sizes.

**Expected counts** are reported for every chi-square. Below 5 the approximation is
unreliable, and the page says so rather than quoting a p-value the reader would
over-trust.

**Group size floor.** Levels with fewer than {ex.MIN_GROUP_SIZE} shipments are
excluded (`ab_testing.min_group_size`). Below that a single late shipment swings the
rate by several points.

**Excluded levels.** `shipment_mode == "Unknown"` (360 rows, 98.9% on time) is
dropped from mode comparisons. It is entirely pre-2011 and is a recording artefact
of the era, not a transport mode — leaving it in would have made "Unknown" the
best-performing mode on the dashboard.

**Test selection for continuous metrics** follows a written rule, not a judgement
call: absolute skew above `ab_testing.skew_limit` ({cfg.ab_testing.skew_limit})
means the rank-based result is quoted. Both are always computed and shown, because
on this data they disagree in both directions.
""")

sidebar_about()
