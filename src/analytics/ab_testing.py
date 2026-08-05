"""
Statistical testing toolkit - the tests themselves, with no data of their own.

What this module is, and is not
-------------------------------
This is the **statistics layer**: two-proportion z-tests, chi-square tests of
independence, Welch's t-test, Mann-Whitney U, effect sizes and power analysis.
Every function takes counts or Series and returns a result dictionary. It reads no
data and knows nothing about pharmaceuticals.

The **comparisons** - which groups get compared, on what metric, and what
confounds that comparison - live in :mod:`src.analytics.experiments` (supply chain)
and :mod:`src.analytics.products` (procurement pricing). Both read real data.

That split is deliberate. An earlier version of this module carried a catalogue of
four operational interventions with hard-coded control and treatment rates, and
generated Bernoulli draws around them. It produced clean, significant results
because the effect had been written into the configuration file. The whole exercise
was circular, so it is gone. Every p-value the platform now reports comes from a
comparison between groups that exist in a published dataset.

What is lost by that, honestly
------------------------------
Randomisation. These are **observational comparisons**, not randomised
experiments, so a difference between two groups licenses no causal claim on its
own. Direct Drop shipments arrive on time 94.7% of the time against 82.9% for
regional-distribution-centre stock, but nobody randomised a shipment into a
fulfilment route - the two groups differ in destination, era and commodity as well.
This is exactly why :func:`src.analytics.experiments.stratified_comparison` exists
and why every comparison carries a stated confound. Naming the confound is the
analytical work; the p-value is the easy part.

Why both a z-test and a chi-square test
---------------------------------------
For a 2x2 design they are mathematically equivalent - the chi-square statistic is
the square of the z statistic, and the p-values agree. Reporting both is a
deliberate consistency check: if they ever disagree, the contingency table was
built wrong. The z-test additionally yields a signed, directional confidence
interval on the difference in proportions, which is what a decision needs.

Choosing between a mean-based and a rank-based test
--------------------------------------------------
:func:`compare_continuous` runs Welch **and** Mann-Whitney, measures the skew, and
states which result to quote against a configured threshold. This is not a
formality: on SCMS freight ratios Welch returns p = 0.44 where Mann-Whitney returns
p = 6.0e-10 on the same comparison, and on delivery delay by era the disagreement
runs the *other* way. See that function's docstring for both cases and why they need
opposite readings.

Example
-------
>>> from src.analytics.ab_testing import two_proportion_z_test
>>> r = two_proportion_z_test(4479, 5404, 4661, 4920)   # RDC versus Direct Drop
>>> f"{r['absolute_diff_pp']:+.2f}pp, p={r['p_value']:.2e}"
'+11.89pp, p=7.53e-80'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from src.config import get_config
from src.logger import get_logger

log = get_logger(__name__)

#: Minimum relative improvement worth the capital and change-management cost of an
#: operational change. Below this, a statistically significant result is still a
#: "do not act" - this is what stops a large sample turning noise into a project.
PRACTICAL_SIGNIFICANCE_LIFT_PCT = 1.5


def _alpha(override: float | None = None) -> float:
    """Resolve the significance level from config unless explicitly overridden."""
    return float(get_config().ab_testing.alpha if override is None else override)


def _skew_limit() -> float:
    """Absolute skewness above which a mean-based test stops being trustworthy."""
    return float(get_config().ab_testing.skew_limit)


#: Module-level view of the configured skew threshold, for display and tests.
SKEW_LIMIT: float = float(get_config().ab_testing.skew_limit)


# ---------------------------------------------------------------------------
# Binary outcomes
# ---------------------------------------------------------------------------
def two_proportion_z_test(successes_control: int, n_control: int,
                          successes_treatment: int, n_treatment: int,
                          alpha: float | None = None) -> dict:
    """Two-proportion z-test with a confidence interval on the difference.

    "Control" and "treatment" are naming conventions inherited from experimental
    design. On observational data they simply mean *baseline group* and
    *comparison group* - the arithmetic is identical, the causal licence is not.

    Parameters
    ----------
    successes_control, n_control, successes_treatment, n_treatment
        Event counts and sample sizes per group.
    alpha
        Significance level. Defaults to ``config.ab_testing.alpha``.

    Returns
    -------
    dict
        ``p_control``, ``p_treatment``, ``absolute_diff``, ``relative_lift_pct``,
        ``z_statistic``, ``p_value``, ``ci_lower``, ``ci_upper``, ``alpha``,
        ``significant``, plus per-group Wilson confidence intervals.
    """
    alpha = _alpha(alpha)

    p_control = successes_control / n_control if n_control else 0.0
    p_treatment = successes_treatment / n_treatment if n_treatment else 0.0

    z_stat, p_value = proportions_ztest(
        count=np.array([successes_treatment, successes_control]),
        nobs=np.array([n_treatment, n_control]))

    # Unpooled standard error is the correct basis for a CI on the difference
    # (the pooled SE used in the test statistic assumes the null is true).
    se_diff = np.sqrt(p_treatment * (1 - p_treatment) / n_treatment
                      + p_control * (1 - p_control) / n_control)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_treatment - p_control

    ctrl_lo, ctrl_hi = proportion_confint(successes_control, n_control, alpha=alpha, method="wilson")
    trt_lo, trt_hi = proportion_confint(successes_treatment, n_treatment, alpha=alpha, method="wilson")

    return {
        "test": "Two-Proportion Z-Test",
        "p_control": round(p_control, 5),
        "p_treatment": round(p_treatment, 5),
        "absolute_diff": round(diff, 5),
        "absolute_diff_pp": round(100 * diff, 3),
        "relative_lift_pct": round(100 * diff / p_control, 3) if p_control else 0.0,
        "z_statistic": round(float(z_stat), 4),
        "p_value": float(p_value),
        "ci_lower": round(diff - z_crit * se_diff, 5),
        "ci_upper": round(diff + z_crit * se_diff, 5),
        "ci_lower_pp": round(100 * (diff - z_crit * se_diff), 3),
        "ci_upper_pp": round(100 * (diff + z_crit * se_diff), 3),
        "control_ci": (round(ctrl_lo, 5), round(ctrl_hi, 5)),
        "treatment_ci": (round(trt_lo, 5), round(trt_hi, 5)),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def chi_square_test(data: pd.DataFrame, group: str, outcome: str,
                    alpha: float | None = None) -> dict:
    """Chi-square test of independence between a grouping and a binary outcome.

    Works for any number of groups, so it handles the four shipment modes as
    readily as a two-way split. For a 2x2 table it cross-checks the z-test:
    ``chi2 == z**2``, so agreement confirms the contingency table is correct.

    Parameters
    ----------
    data
        Row-level frame.
    group
        Column holding the grouping (e.g. ``shipment_mode``).
    outcome
        Column holding the binary outcome (e.g. ``is_late``).
    alpha
        Significance level. Defaults to config.

    Returns
    -------
    dict
        ``chi2_statistic``, ``p_value``, ``degrees_of_freedom``, ``cramers_v``
        (effect size), ``contingency_table``, ``expected_frequencies``,
        ``min_expected_count``, ``expected_counts_adequate``, ``significant``.
    """
    alpha = _alpha(alpha)
    frame = data[[group, outcome]].dropna()
    table = pd.crosstab(frame[group], frame[outcome])

    # Yates' continuity correction is disabled deliberately. SciPy applies it to
    # 2x2 tables by default, which would make chi2 != z^2 and break the
    # cross-check described in the module docstring. It is also over-conservative
    # at the sample sizes used here (n in the thousands), where the normal
    # approximation is already excellent.
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)

    # Cramer's V normalises chi-square to a 0-1 effect size so results stay
    # comparable across comparisons with very different sample sizes.
    n = int(table.values.sum())
    min_dim = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n and min_dim else 0.0

    return {
        "test": "Chi-Square Test of Independence",
        "group": group,
        "outcome": outcome,
        "chi2_statistic": round(float(chi2), 4),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "contingency_table": table,
        "expected_frequencies": pd.DataFrame(
            expected, index=table.index, columns=table.columns).round(1),
        "min_expected_count": round(float(expected.min()), 1),
        # Below 5 the chi-square approximation is unreliable, so report the fact
        # rather than quoting a p-value the reader will over-trust.
        "expected_counts_adequate": bool(expected.min() >= 5),
        "n": n,
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# Continuous outcomes
# ---------------------------------------------------------------------------
def welch_t_test(a: pd.Series, b: pd.Series, alpha: float | None = None,
                 label_a: str = "Group A", label_b: str = "Group B",
                 metric: str = "value") -> dict:
    """Welch's two-sample t-test on a continuous metric.

    Welch's variant rather than Student's because two operational groups have no
    reason to share a variance - an air lane is both faster *and* more consistent
    than an ocean lane, and assuming equal variance would understate the
    uncertainty on the slower one.

    Returns
    -------
    dict
        Per-group means and sds, ``mean_difference``, ``t_statistic``,
        ``p_value``, ``degrees_of_freedom``, ``cohens_d`` (effect size),
        ``ci_lower``/``ci_upper`` on the difference, ``significant``.
    """
    alpha = _alpha(alpha)
    x, y = pd.Series(a).dropna().astype(float), pd.Series(b).dropna().astype(float)

    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)

    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    # Pooled sd for Cohen's d - the conventional standardised effect size.
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    diff = x.mean() - y.mean()
    cohens_d = diff / pooled_sd if pooled_sd else 0.0

    se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
    # Welch-Satterthwaite degrees of freedom.
    dof = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    return {
        "test": "Welch's Two-Sample T-Test",
        "metric": metric,
        "label_a": label_a,
        "label_b": label_b,
        "n_a": int(n1),
        "n_b": int(n2),
        "mean_a": round(float(x.mean()), 4),
        "mean_b": round(float(y.mean()), 4),
        "sd_a": round(float(s1), 4),
        "sd_b": round(float(s2), 4),
        "mean_difference": round(float(diff), 4),
        "pct_change": round(100 * float(diff) / float(y.mean()), 2) if y.mean() else 0.0,
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "degrees_of_freedom": round(float(dof), 1),
        "cohens_d": round(float(cohens_d), 4),
        "ci_lower": round(float(diff - t_crit * se_diff), 4),
        "ci_upper": round(float(diff + t_crit * se_diff), 4),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def mann_whitney_test(a: pd.Series, b: pd.Series, alpha: float | None = None,
                      label_a: str = "Group A", label_b: str = "Group B") -> dict:
    """Mann-Whitney U test - a rank-based alternative to Welch's t-test.

    Compares *distributions* rather than means, so it survives the extreme right
    skew that is normal in freight and cost data. Reports medians (the statistic it
    actually speaks to) and the rank-biserial correlation as an effect size.

    Returns
    -------
    dict
        Per-group n and median, ``u_statistic``, ``p_value``,
        ``rank_biserial`` (effect size, -1..1), ``alpha``, ``significant``.
    """
    alpha = _alpha(alpha)

    x, y = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(x) < 2 or len(y) < 2:
        return {"test": "Mann-Whitney U", "p_value": float("nan"),
                "significant": False, "note": "insufficient data in one group",
                "n_a": int(len(x)), "n_b": int(len(y))}

    u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
    # Rank-biserial correlation: U rescaled to -1..1, interpretable as the
    # probability that a random draw from one group exceeds one from the other.
    rank_biserial = 2 * u_stat / (len(x) * len(y)) - 1

    return {
        "test": "Mann-Whitney U (rank-based)",
        "label_a": label_a,
        "label_b": label_b,
        "n_a": int(len(x)),
        "n_b": int(len(y)),
        "median_a": round(float(x.median()), 4),
        "median_b": round(float(y.median()), 4),
        "median_difference": round(float(x.median() - y.median()), 4),
        "u_statistic": round(float(u_stat), 1),
        "p_value": float(p_value),
        "rank_biserial": round(float(rank_biserial), 4),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def compare_continuous(a: pd.Series, b: pd.Series, alpha: float | None = None,
                       label_a: str = "Group A", label_b: str = "Group B",
                       metric: str = "value") -> dict:
    """Compare a continuous metric with both a mean-based and a rank-based test.

    Which test to believe is not a matter of taste - it depends on the data. On
    heavily skewed metrics the two disagree wildly, and they disagree in *both*
    directions on this dataset:

    * Freight as a share of commodity value (mean 2,548%, median 10.6%, skew 78)
      compared across product group: Welch returns p = 0.439 and Mann-Whitney
      p = 6.0e-10. Here Welch is wrong - a few extreme ratios have inflated the
      variance until the mean test cannot resolve a difference the medians show
      clearly (11.8% against 9.7%).
    * Delivery delay compared across era: Welch returns p = 8.2e-06 and
      Mann-Whitney p = 0.453. Here *neither* is wrong. The mean delay moved from
      -5.0 to -7.5 days while both medians sit at exactly 0, because 61% of
      deliveries land on their scheduled day. The change is in the tail, not the
      typical shipment.

    This runs both, measures the skew, and states which result to quote - including
    "both" for the second case. The threshold lives in
    ``config.ab_testing.skew_limit`` so the rule is written down once rather than
    decided per chart.

    Returns
    -------
    dict
        ``welch`` and ``mann_whitney`` sub-dicts, ``max_abs_skew``,
        ``recommended_test``, ``tests_agree``, and ``verdict`` (plain English).
    """
    alpha = _alpha(alpha)
    limit = _skew_limit()

    x, y = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(x) < 2 or len(y) < 2:
        return {"recommended_test": "none", "tests_agree": False,
                "verdict": "Insufficient data in one group to compare.",
                "welch": None, "mann_whitney": None, "max_abs_skew": float("nan"),
                "skew_limit": limit}

    welch = welch_t_test(x, y, alpha=alpha, label_a=label_a, label_b=label_b,
                         metric=metric)
    rank = mann_whitney_test(x, y, alpha=alpha, label_a=label_a, label_b=label_b)

    max_abs_skew = float(max(abs(stats.skew(x)), abs(stats.skew(y))))
    skewed = max_abs_skew > limit
    agree = bool(welch["significant"] == rank["significant"])
    if not skewed:
        recommended = "welch"
    elif agree or rank["significant"]:
        recommended = "mann_whitney"
    else:
        # Skewed, and only Welch is significant. Neither test alone is the answer:
        # the mean moved and the median did not, which is a fact about the tail that
        # requires both numbers to state.
        recommended = "both"

    if skewed and not agree and rank["significant"]:
        # Rank test finds an effect the mean test misses: the classic outlier
        # masking case. A handful of extreme values has inflated one group's
        # variance so far that Welch can no longer resolve a difference the bulk
        # of the distribution shows clearly.
        verdict = (
            f"Skew is {max_abs_skew:.1f}, so the means are not a fair summary and "
            f"the two tests disagree. Quote Mann-Whitney (p = {rank['p_value']:.2e}); "
            f"Welch's p = {welch['p_value']:.3f} is an artefact of outliers masking a "
            f"real difference in the bulk of the distribution."
        )
    elif skewed and not agree:
        # The opposite disagreement, and it needs the opposite reading. Welch is
        # significant while the rank test is not, which means the means differ but
        # the central tendency does not - the difference lives in the tail. Calling
        # Welch an artefact here would be wrong: a shift in the mean is real and
        # often the operationally important thing. What it is *not* is a shift in
        # the typical case.
        verdict = (
            f"Skew is {max_abs_skew:.1f} and the tests disagree in the other "
            f"direction: Welch is significant (p = {welch['p_value']:.2e}) while "
            f"Mann-Whitney is not (p = {rank['p_value']:.3f}). Both are correct about "
            f"different things. The means differ by "
            f"{welch['mean_difference']:+.2f} but the medians are "
            f"{rank['median_a']:.1f} against {rank['median_b']:.1f}, so the "
            f"difference is in the tail rather than the typical case. Report it as a "
            f"tail effect, not a shift in the average outcome."
        )
    elif skewed:
        verdict = (
            f"Skew is {max_abs_skew:.1f}, so quote Mann-Whitney "
            f"(p = {rank['p_value']:.2e}). Both tests agree on the conclusion."
        )
    else:
        verdict = (
            f"Skew is {max_abs_skew:.1f}, within the range where means are "
            f"meaningful. Welch's t-test applies (p = {welch['p_value']:.2e})."
        )

    return {
        "welch": welch,
        "mann_whitney": rank,
        "max_abs_skew": round(max_abs_skew, 2),
        "skew_limit": limit,
        "recommended_test": recommended,
        "tests_agree": agree,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------
def required_sample_size(control_rate: float, treatment_rate: float,
                         alpha: float | None = None, power: float | None = None) -> dict:
    """Sample size needed to detect a given effect at the configured power.

    Run *before* committing to a measurement. A test that returns "not
    significant" without the power to have detected the effect has told you
    nothing, and that is the most common way an operational trial wastes a quarter.

    Returns
    -------
    dict
        ``effect_size_cohens_h``, ``n_per_arm``, ``n_total``, ``alpha``,
        ``power``, and the input rates.
    """
    cfg = get_config().ab_testing
    alpha = _alpha(alpha)
    power = float(cfg.power if power is None else power)

    # Cohen's h - the standard effect size for a difference of proportions.
    effect_size = 2 * np.arcsin(np.sqrt(treatment_rate)) - 2 * np.arcsin(np.sqrt(control_rate))
    n_per_arm = NormalIndPower().solve_power(
        effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0,
        alternative="two-sided")

    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "effect_size_cohens_h": round(float(effect_size), 4),
        "n_per_arm": int(np.ceil(n_per_arm)),
        "n_total": int(np.ceil(n_per_arm) * 2),
        "alpha": alpha,
        "power": power,
    }


def achieved_power(successes_control: int, n_control: int,
                   successes_treatment: int, n_treatment: int,
                   alpha: float | None = None) -> float:
    """Post-hoc power actually achieved by the realised sample.

    This is what turns "no significant difference" into a usable statement. A null
    result at 99% power is evidence the effect is absent; the same null at 20%
    power is evidence of nothing at all.
    """
    alpha = _alpha(alpha)
    p1 = successes_control / n_control if n_control else 0.0
    p2 = successes_treatment / n_treatment if n_treatment else 0.0
    effect = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    if effect == 0:
        return 0.0
    return float(NormalIndPower().solve_power(
        effect_size=abs(effect), nobs1=n_control, alpha=alpha,
        ratio=n_treatment / n_control, alternative="two-sided"))


def minimum_detectable_effect(baseline_rate: float, n_control: int,
                              n_treatment: int, alpha: float | None = None,
                              power: float | None = None) -> dict:
    """Smallest difference the realised sample could have detected.

    This is the correct way to interpret a null result, and it exists because the
    obvious alternative is a statistical trap. Post-hoc power computed at the
    *observed* effect size is a deterministic function of the p-value: a
    non-significant result mechanically returns low observed power, so using it to
    judge a null is circular. On the first-line-designation comparison here,
    observed power is 6% purely because the observed gap is 0.16 points - it says
    nothing about whether the test was sensitive.

    What a reader actually needs to know is: given these sample sizes, how large a
    difference *would* have shown up? If the answer is "anything above 1.5 points",
    then a measured 0.16-point gap is meaningful evidence that no material
    difference exists.

    Parameters
    ----------
    baseline_rate
        Pooled event rate, as a proportion.
    n_control, n_treatment
        Realised group sizes.
    alpha, power
        Defaults from ``config.ab_testing``.

    Returns
    -------
    dict
        ``mde_pp`` (the detectable difference in percentage points), the inputs,
        and a plain-English ``interpretation``.
    """
    cfg = get_config().ab_testing
    alpha = _alpha(alpha)
    power = float(cfg.power if power is None else power)

    # Solve for the Cohen's h detectable at this sample size, then invert the
    # arcsine transform around the baseline to express it as a rate difference.
    effect_h = NormalIndPower().solve_power(
        effect_size=None, nobs1=n_control, alpha=alpha, power=power,
        ratio=n_treatment / n_control, alternative="two-sided")
    detectable_rate = np.sin(np.arcsin(np.sqrt(baseline_rate)) + effect_h / 2) ** 2
    mde = float(detectable_rate - baseline_rate)

    return {
        "baseline_rate_pct": round(100 * baseline_rate, 2),
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "effect_size_cohens_h": round(float(effect_h), 4),
        "mde_pp": round(100 * mde, 3),
        "alpha": alpha,
        "power": power,
        "interpretation": (
            f"At n={n_control:,} versus n={n_treatment:,} and a "
            f"{100 * baseline_rate:.1f}% baseline, this comparison could detect a "
            f"difference of {100 * mde:+.2f} percentage points at {power:.0%} power. "
            f"A smaller measured gap than that is evidence of no material "
            f"difference, not an absence of evidence."),
    }


# ---------------------------------------------------------------------------
# Business translation
# ---------------------------------------------------------------------------
def penalty_impact(absolute_diff: float, annual_shipments: float) -> dict:
    """Value a change in on-time rate as avoided SLA penalties.

    This is the only place in the project that multiplies a measured result by an
    assumed rate, and it is deliberately the *whole* of the financial modelling.
    SCMS records no penalty, discount or expediting cost, so the shipment count and
    the rate difference are measured while the per-shipment penalty comes from
    ``config.economics.late_shipment_penalty``. Presenting anything more elaborate
    would be presenting an assumption as a finding.

    Parameters
    ----------
    absolute_diff
        Improvement in on-time rate as a proportion (0.031 = +3.1 points).
    annual_shipments
        Shipments per year exposed to the change - measured, not assumed.

    Returns
    -------
    dict
        ``annual_value_usd``, the inputs, and the calculation written out.
    """
    penalty = float(get_config().economics.late_shipment_penalty)
    annual_value = annual_shipments * absolute_diff * penalty
    return {
        "annual_shipments": round(float(annual_shipments), 0),
        "penalty_per_late_shipment_usd": penalty,
        "annual_value_usd": round(float(annual_value), 0),
        "calculation_basis": (
            f"{annual_shipments:,.0f} shipments/yr x {absolute_diff:.3%} "
            f"x ${penalty:,.0f} penalty avoided"),
        "assumption": (
            "Shipment count and rate difference are measured from SCMS; the "
            "per-shipment penalty is a configured assumption."),
    }


def business_recommendation(z_result: dict, power: float,
                            impact: dict | None = None,
                            supporting: list[str] | None = None,
                            mde: dict | None = None) -> dict:
    """Convert a statistical result into an act / hold / do-not-act decision.

    The decision rule deliberately separates two questions that get conflated:

    * **Is the difference real?** - p-value below alpha, with adequate power.
    * **Is it worth acting on?** - relative difference above the practical
      significance floor.

    A result can be statistically significant and still fail the second test,
    which is exactly when an organisation should decline to spend the money.

    Parameters
    ----------
    z_result
        Output of :func:`two_proportion_z_test`.
    power
        Achieved power, from :func:`achieved_power`.
    impact
        Optional :func:`penalty_impact` output, to attach a dollar figure.
    supporting
        Optional extra evidence lines (e.g. a continuous-metric result, or the
        confound that limits the reading).
    """
    cfg = get_config().ab_testing
    significant = z_result["significant"]
    lift = abs(z_result["relative_lift_pct"])
    practical = lift >= PRACTICAL_SIGNIFICANCE_LIFT_PCT
    well_powered = power >= float(cfg.power)
    money = (f" Estimated annual value ${impact['annual_value_usd']:,.0f}."
             if impact else "")

    # A null result is judged on the minimum detectable effect, never on post-hoc
    # power - see minimum_detectable_effect for why that distinction matters. There
    # are three genuinely different kinds of null and collapsing them loses the
    # most useful part of the answer.
    if not significant and mde is not None:
        detectable_pp = abs(mde["mde_pp"])
        measured_pp = abs(z_result["absolute_diff_pp"])
        # The practical threshold is a *relative* lift, so convert it to points at
        # this baseline before comparing it against a points-based MDE.
        threshold_pp = abs(z_result["p_control"]) * PRACTICAL_SIGNIFICANCE_LIFT_PCT

        if detectable_pp <= threshold_pp:
            verdict, confidence = "NO EFFECT", "High"
            rationale = (
                f"No significant difference (p={z_result['p_value']:.3f} against "
                f"alpha={cfg.alpha}), and the sample was sensitive enough for that "
                f"to mean something: it could have detected a gap of "
                f"{detectable_pp:.2f} points, inside the {threshold_pp:.2f} points "
                f"that would justify acting. The measured gap is {measured_pp:.2f} "
                f"points. This is positive evidence the groups do not differ.")
        else:
            verdict, confidence = "NO EFFECT (BOUNDED)", "Medium"
            rationale = (
                f"No significant difference (p={z_result['p_value']:.3f}); the "
                f"measured gap is {measured_pp:.2f} points. This rules out any "
                f"difference larger than {detectable_pp:.2f} points, which is a real "
                f"and useful bound - but it sits just above the {threshold_pp:.2f} "
                f"points that would justify acting, so a difference small enough to "
                f"matter marginally cannot be excluded. Report the bound, not "
                f"'no difference'.")

        return {
            "verdict": verdict, "confidence": confidence, "rationale": rationale,
            "supporting_evidence": list(supporting or []) + [mde["interpretation"]],
            "practical_significance_met": False,
            "statistically_significant": False,
            "adequately_powered": detectable_pp <= threshold_pp,
            "achieved_power": round(float(power), 4),
            "minimum_detectable_effect_pp": mde["mde_pp"],
            "practical_threshold_pp": round(threshold_pp, 3),
            "practical_threshold_pct": PRACTICAL_SIGNIFICANCE_LIFT_PCT,
        }

    if significant and practical and well_powered:
        verdict, confidence = "ACT", "High"
        rationale = (
            f"The {z_result['absolute_diff_pp']:+.2f}pp gap is statistically "
            f"significant (p={z_result['p_value']:.2e}) and the {lift:.1f}% relative "
            f"difference clears the {PRACTICAL_SIGNIFICANCE_LIFT_PCT}% "
            f"practical-significance floor.{money}"
        )
    elif significant and practical and not well_powered:
        verdict, confidence = "ACT WITH MONITORING", "Medium"
        rationale = (
            f"The gap is significant (p={z_result['p_value']:.2e}) and commercially "
            f"meaningful, but achieved power is {power:.0%}, below the "
            f"{cfg.power:.0%} target. Act on a limited scope and re-measure."
        )
    elif significant and not practical:
        verdict, confidence = "DO NOT ACT", "High"
        rationale = (
            f"The gap is statistically real (p={z_result['p_value']:.2e}) but the "
            f"{lift:.1f}% relative difference is below the "
            f"{PRACTICAL_SIGNIFICANCE_LIFT_PCT}% threshold that justifies the cost "
            f"of change. A large sample has detected a difference too small to matter."
        )
    else:
        verdict, confidence = "INCONCLUSIVE", "Low"
        rationale = (
            f"No significant difference (p={z_result['p_value']:.3f}), and without a "
            f"minimum-detectable-effect calculation there is no basis for calling "
            f"this a null rather than an underpowered test. Compute the MDE before "
            f"reading anything into it."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "supporting_evidence": list(supporting or []),
        "practical_significance_met": practical,
        "statistically_significant": significant,
        "adequately_powered": well_powered,
        "achieved_power": round(float(power), 4),
        "minimum_detectable_effect_pp": mde["mde_pp"] if mde else None,
        "practical_threshold_pct": PRACTICAL_SIGNIFICANCE_LIFT_PCT,
    }


__all__ = [
    "two_proportion_z_test", "chi_square_test", "welch_t_test",
    "mann_whitney_test", "compare_continuous", "required_sample_size",
    "achieved_power", "minimum_detectable_effect", "penalty_impact",
    "business_recommendation",
    "PRACTICAL_SIGNIFICANCE_LIFT_PCT", "SKEW_LIMIT",
]
