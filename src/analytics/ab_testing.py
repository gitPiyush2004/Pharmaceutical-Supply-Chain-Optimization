"""
Statistical A/B testing for supply chain operational interventions.

Business framing
----------------
Operations teams are constantly asked to approve capital projects - automate
the warehouse, put process analytical technology on the QC line, buy an AI
routing engine. Each is pitched with an improvement claim. This module answers
the only question that matters before signing: **is the observed improvement
real, or is it noise?**

Each experiment randomises production batches (or shipments) into a control arm
running the incumbent process and a treatment arm running the intervention, then
evaluates two metrics:

* a **primary binary metric** - did the unit succeed? (batch released, shipment
  on time, batch within potency spec). Tested with a two-proportion z-test and
  cross-checked with a chi-square test of independence.
* a **secondary continuous metric** - how long did it take? Tested with
  Welch's t-test, which does not assume equal variances between arms.

Statistical significance is necessary but not sufficient, so every result is
also scored on **practical significance** (is the lift big enough to be worth
the capital?) and translated into an **annualised dollar impact** before a
recommendation is issued.

Why both a z-test and a chi-square test
---------------------------------------
For a 2x2 design they are mathematically equivalent - the chi-square statistic
is the square of the z statistic, and the p-values agree. Reporting both is a
deliberate consistency check: if they ever disagree, the contingency table was
built wrong. The z-test additionally yields a signed, directional confidence
interval on the difference in proportions, which is what the business needs.

Example
-------
>>> from src.analytics.ab_testing import run_experiment, list_experiments
>>> list_experiments()
>>> result = run_experiment("qa_automation", sample_size=2000)
>>> result["recommendation"]["verdict"]
'ADOPT'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)

#: Minimum relative lift considered worth the capital and change-management
#: cost of an operational intervention. Below this, a statistically significant
#: result is still a "do not adopt" - this is what stops teams chasing noise.
PRACTICAL_SIGNIFICANCE_LIFT_PCT = 1.5

#: Secondary continuous metric: mean and sd of the per-unit processing time in
#: the control arm, and the reduction the intervention is expected to deliver.
_CONTINUOUS_METRIC = {
    "qa_automation":       {"name": "QC Release Time (days)",      "control_mean": 20.0, "sd": 6.0, "treatment_delta": -4.8},
    "warehouse_automation": {"name": "Pick-to-Dispatch Time (hrs)", "control_mean": 14.5, "sd": 4.2, "treatment_delta": -3.6},
    "route_optimization":  {"name": "Transit Time (days)",         "control_mean": 6.4,  "sd": 2.6, "treatment_delta": -0.9},
    "cold_chain_iot":      {"name": "Excursion Response Time (hrs)", "control_mean": 9.1, "sd": 3.4, "treatment_delta": -5.2},
}


# ---------------------------------------------------------------------------
# Experiment catalogue
# ---------------------------------------------------------------------------
def list_experiments() -> pd.DataFrame:
    """Return the catalogue of available operational interventions.

    Returns
    -------
    pd.DataFrame
        ``key``, ``name``, ``metric``, ``control_rate``, ``treatment_rate``,
        ``expected_lift_pct``, ``description``.
    """
    cfg = get_config().ab_testing
    rows = []
    for key, spec in cfg.experiments.items():
        lift = 100.0 * (spec.treatment_rate - spec.control_rate) / spec.control_rate
        rows.append({
            "key": key,
            "name": spec.name,
            "metric": spec.metric,
            "control_rate": spec.control_rate,
            "treatment_rate": spec.treatment_rate,
            "expected_lift_pct": round(lift, 2),
            "description": " ".join(str(spec.description).split()),
        })
    return pd.DataFrame(rows)


def _experiment_spec(key: str):
    """Look up one experiment definition, with a helpful error if unknown."""
    cfg = get_config().ab_testing
    if key not in cfg.experiments:
        raise KeyError(f"Unknown experiment '{key}'. Available: {sorted(cfg.experiments)}")
    return cfg.experiments[key]


# ---------------------------------------------------------------------------
# Experiment data generation
# ---------------------------------------------------------------------------
def simulate_experiment(experiment_key: str, sample_size: int | None = None,
                        seed: int | None = None,
                        treatment_rate_override: float | None = None) -> pd.DataFrame:
    """Generate subject-level randomised experiment data.

    Each subject is one production batch (or shipment) assigned 50/50 to
    control or treatment. Randomisation is what licenses a causal reading of
    the difference between arms.

    Parameters
    ----------
    experiment_key
        Key from :func:`list_experiments`.
    sample_size
        Total subjects across both arms. Defaults to
        ``config.ab_testing.default_sample_size``.
    seed
        Random seed. Defaults to the project seed for reproducibility.
    treatment_rate_override
        Replaces the configured treatment rate. Used by the dashboard slider so
        a user can explore "what if the intervention only delivered X?".

    Returns
    -------
    pd.DataFrame
        One row per subject: ``subject_id``, ``arm``, ``success`` (0/1),
        ``processing_time``, ``region``, ``drug_code``.
    """
    cfg = get_config()
    spec = _experiment_spec(experiment_key)
    n = int(sample_size or cfg.ab_testing.default_sample_size)
    rng = np.random.default_rng(cfg.project.random_seed if seed is None else seed)

    control_rate = float(spec.control_rate)
    treatment_rate = float(spec.treatment_rate if treatment_rate_override is None
                           else treatment_rate_override)

    # Balanced 50/50 assignment - equal arms maximise power for a fixed n.
    n_control = n // 2
    n_treatment = n - n_control
    arm = np.array(["Control"] * n_control + ["Treatment"] * n_treatment)

    success = np.concatenate([
        rng.binomial(1, control_rate, n_control),
        rng.binomial(1, treatment_rate, n_treatment),
    ])

    continuous = _CONTINUOUS_METRIC.get(
        experiment_key, {"control_mean": 10.0, "sd": 3.0, "treatment_delta": -1.0})
    processing_time = np.concatenate([
        rng.normal(continuous["control_mean"], continuous["sd"], n_control),
        rng.normal(continuous["control_mean"] + continuous["treatment_delta"],
                   continuous["sd"], n_treatment),
    ]).clip(min=0.1)

    # Strata are carried through so the dashboard can check that the treatment
    # effect is consistent across segments rather than driven by one region.
    frame = pd.DataFrame({
        "subject_id": [f"SUBJ-{i:05d}" for i in range(1, n + 1)],
        "arm": arm,
        "success": success,
        "processing_time": np.round(processing_time, 2),
        "region": rng.choice(
            ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"],
            size=n, p=[0.34, 0.27, 0.22, 0.10, 0.07]),
        "drug_code": rng.choice(["DrugY", "DrugX", "DrugA", "DrugB", "DrugC"],
                                size=n, p=[0.455, 0.27, 0.115, 0.08, 0.08]),
    })
    # Shuffle so arm order carries no information downstream.
    return frame.sample(frac=1.0, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def two_proportion_z_test(successes_control: int, n_control: int,
                          successes_treatment: int, n_treatment: int,
                          alpha: float | None = None) -> dict:
    """Two-proportion z-test with a confidence interval on the difference.

    Parameters
    ----------
    successes_control, n_control, successes_treatment, n_treatment
        Event counts and sample sizes per arm.
    alpha
        Significance level. Defaults to ``config.ab_testing.alpha``.

    Returns
    -------
    dict
        ``p_control``, ``p_treatment``, ``absolute_diff``, ``relative_lift_pct``,
        ``z_statistic``, ``p_value``, ``ci_lower``, ``ci_upper``, ``alpha``,
        ``significant``, plus per-arm Wilson confidence intervals.
    """
    cfg = get_config().ab_testing
    alpha = float(cfg.alpha if alpha is None else alpha)

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


def chi_square_test(data: pd.DataFrame, alpha: float | None = None) -> dict:
    """Chi-square test of independence between arm and outcome.

    Cross-checks the z-test. For a 2x2 table the two are equivalent
    (chi2 == z^2), so agreement confirms the contingency table is correct.

    Parameters
    ----------
    data
        Subject-level frame with ``arm`` and ``success`` columns.
    alpha
        Significance level. Defaults to config.

    Returns
    -------
    dict
        ``chi2_statistic``, ``p_value``, ``degrees_of_freedom``, ``cramers_v``
        (effect size), ``contingency_table``, ``expected_frequencies``,
        ``significant``.
    """
    cfg = get_config().ab_testing
    alpha = float(cfg.alpha if alpha is None else alpha)

    table = pd.crosstab(data["arm"], data["success"])
    # Yates' continuity correction is disabled deliberately. SciPy applies it to
    # 2x2 tables by default, which would make chi2 != z^2 and break the
    # cross-check described in the module docstring. It is also known to be
    # over-conservative at the sample sizes used here (n in the thousands),
    # where the normal approximation is already excellent.
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)

    # Cramer's V normalises chi-square to a 0-1 effect size so results stay
    # comparable across experiments with different sample sizes.
    n = int(table.values.sum())
    min_dim = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n and min_dim else 0.0

    return {
        "test": "Chi-Square Test of Independence",
        "chi2_statistic": round(float(chi2), 4),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "contingency_table": table,
        "expected_frequencies": pd.DataFrame(
            expected, index=table.index, columns=table.columns).round(1),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def t_test_continuous(data: pd.DataFrame, metric: str = "processing_time",
                      alpha: float | None = None) -> dict:
    """Welch's two-sample t-test on a continuous secondary metric.

    Welch's variant is used rather than Student's because the two arms have no
    reason to share a variance - an automated process is typically both faster
    *and* more consistent than a manual one.

    Returns
    -------
    dict
        Per-arm means and sds, ``mean_difference``, ``t_statistic``,
        ``p_value``, ``degrees_of_freedom``, ``cohens_d`` (effect size),
        ``ci_lower``/``ci_upper`` on the difference, ``significant``.
    """
    cfg = get_config().ab_testing
    alpha = float(cfg.alpha if alpha is None else alpha)

    control = data.loc[data["arm"] == "Control", metric].dropna()
    treatment = data.loc[data["arm"] == "Treatment", metric].dropna()

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    n1, n2 = len(treatment), len(control)
    s1, s2 = treatment.std(ddof=1), control.std(ddof=1)
    # Pooled sd for Cohen's d - the conventional standardised effect size.
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    diff = treatment.mean() - control.mean()
    cohens_d = diff / pooled_sd if pooled_sd else 0.0

    se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
    # Welch-Satterthwaite degrees of freedom.
    dof = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    return {
        "test": "Welch's Two-Sample T-Test",
        "metric": metric,
        "control_mean": round(float(control.mean()), 4),
        "treatment_mean": round(float(treatment.mean()), 4),
        "control_sd": round(float(s2), 4),
        "treatment_sd": round(float(s1), 4),
        "mean_difference": round(float(diff), 4),
        "pct_change": round(100 * float(diff) / float(control.mean()), 2) if control.mean() else 0.0,
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "degrees_of_freedom": round(float(dof), 1),
        "cohens_d": round(float(cohens_d), 4),
        "ci_lower": round(float(diff - t_crit * se_diff), 4),
        "ci_upper": round(float(diff + t_crit * se_diff), 4),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------
def required_sample_size(control_rate: float, treatment_rate: float,
                         alpha: float | None = None, power: float | None = None) -> dict:
    """Sample size needed to detect a given effect at the configured power.

    Run *before* an experiment. An under-powered test that returns "not
    significant" has told you nothing, and that is the most common way
    operational experiments waste a quarter.

    Returns
    -------
    dict
        ``effect_size`` (Cohen's h), ``n_per_arm``, ``n_total``, ``alpha``,
        ``power``, and the input rates.
    """
    cfg = get_config().ab_testing
    alpha = float(cfg.alpha if alpha is None else alpha)
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
    """Post-hoc power actually achieved by the realised sample."""
    cfg = get_config().ab_testing
    alpha = float(cfg.alpha if alpha is None else alpha)
    p1 = successes_control / n_control if n_control else 0.0
    p2 = successes_treatment / n_treatment if n_treatment else 0.0
    effect = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    if effect == 0:
        return 0.0
    return float(NormalIndPower().solve_power(
        effect_size=abs(effect), nobs1=n_control, alpha=alpha,
        ratio=n_treatment / n_control, alternative="two-sided"))


# ---------------------------------------------------------------------------
# Segment analysis and business translation
# ---------------------------------------------------------------------------
def segment_analysis(data: pd.DataFrame, dimension: str = "region") -> pd.DataFrame:
    """Break the treatment effect down by segment.

    A headline lift driven entirely by one region is a very different investment
    case from one that holds everywhere, so this check runs on every experiment.
    """
    rows = []
    for segment, group in data.groupby(dimension):
        control = group[group["arm"] == "Control"]
        treatment = group[group["arm"] == "Treatment"]
        if len(control) < 10 or len(treatment) < 10:
            continue  # too few subjects for a stable segment estimate

        p_c, p_t = control["success"].mean(), treatment["success"].mean()
        _, p_value = proportions_ztest(
            count=np.array([treatment["success"].sum(), control["success"].sum()]),
            nobs=np.array([len(treatment), len(control)]))
        rows.append({
            dimension: segment,
            "n_control": len(control),
            "n_treatment": len(treatment),
            "control_rate_pct": round(100 * p_c, 2),
            "treatment_rate_pct": round(100 * p_t, 2),
            "lift_pp": round(100 * (p_t - p_c), 2),
            "relative_lift_pct": round(100 * (p_t - p_c) / p_c, 2) if p_c else 0.0,
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < get_config().ab_testing.alpha),
        })
    return pd.DataFrame(rows).sort_values("lift_pp", ascending=False).reset_index(drop=True)


def estimate_business_impact(experiment_key: str, absolute_diff: float) -> dict:
    """Translate a lift in the primary metric into an annualised dollar impact.

    Uses the platform's own actuals - annual units processed and the realised
    weighted-average unit cost - so the number is anchored to this business
    rather than to a generic benchmark.

    Parameters
    ----------
    experiment_key
        Which intervention is being valued.
    absolute_diff
        Improvement in the primary rate, as a proportion (0.031 = +3.1pp).

    Returns
    -------
    dict
        ``annual_units``, ``avg_unit_cost_usd``, ``units_recovered_annually``,
        ``annual_value_usd`` and the assumption trail behind them.
    """
    batches = loader.load_batches()
    years = max(batches["year"].nunique(), 1)

    annual_units = float(batches["units_procured"].sum()) / years
    total_units = float(batches["units_procured"].sum())
    avg_unit_cost = float(
        (batches["units_procured"] * batches["unit_cost_usd"]).sum() / total_units)

    units_recovered = annual_units * absolute_diff

    # Late-delivery interventions are valued on avoided SLA penalties rather
    # than on recovered product, because no units are physically saved.
    if experiment_key == "route_optimization":
        shipments = loader.load_shipments()
        annual_shipments = len(shipments) / years
        penalty = float(get_config().economics.late_shipment_penalty)
        annual_value = annual_shipments * absolute_diff * penalty
        basis = f"{annual_shipments:,.0f} shipments/yr x {absolute_diff:.3%} x ${penalty:,.0f} SLA penalty avoided"
    else:
        annual_value = units_recovered * avg_unit_cost
        basis = f"{annual_units:,.0f} units/yr x {absolute_diff:.3%} x ${avg_unit_cost:.2f}/unit"

    return {
        "annual_units": round(annual_units, 0),
        "avg_unit_cost_usd": round(avg_unit_cost, 2),
        "units_recovered_annually": round(units_recovered, 0),
        "annual_value_usd": round(annual_value, 0),
        "calculation_basis": basis,
        "years_of_history": years,
    }


def business_recommendation(z_result: dict, t_result: dict, impact: dict,
                            power: float) -> dict:
    """Convert statistical output into an adopt / hold / reject decision.

    The decision rule deliberately separates two questions that are often
    conflated:

    * **Is the effect real?** - p-value below alpha, with adequate power.
    * **Is the effect worth it?** - relative lift above the practical
      significance floor.

    A result can be statistically significant and still fail the second test,
    which is exactly when a business should decline to spend the capital.
    """
    cfg = get_config().ab_testing
    significant = z_result["significant"]
    lift = z_result["relative_lift_pct"]
    practical = lift >= PRACTICAL_SIGNIFICANCE_LIFT_PCT
    well_powered = power >= float(cfg.power)

    if significant and practical and well_powered:
        verdict, confidence = "ADOPT", "High"
        rationale = (
            f"The {z_result['absolute_diff_pp']:+.2f}pp improvement is statistically "
            f"significant (p={z_result['p_value']:.2e}) and the {lift:.1f}% relative lift "
            f"clears the {PRACTICAL_SIGNIFICANCE_LIFT_PCT}% practical-significance floor. "
            f"Estimated annual value ${impact['annual_value_usd']:,.0f}. Roll out."
        )
    elif significant and practical and not well_powered:
        verdict, confidence = "ADOPT WITH MONITORING", "Medium"
        rationale = (
            f"The effect is significant (p={z_result['p_value']:.2e}) and commercially "
            f"meaningful, but achieved power is {power:.0%}, below the {cfg.power:.0%} "
            f"target. Roll out to a limited set of sites and confirm before full scale-up."
        )
    elif significant and not practical:
        verdict, confidence = "DO NOT ADOPT", "High"
        rationale = (
            f"The effect is statistically real (p={z_result['p_value']:.2e}) but the "
            f"{lift:.1f}% relative lift is below the {PRACTICAL_SIGNIFICANCE_LIFT_PCT}% "
            f"threshold that justifies the capital and change-management cost. "
            f"A large sample has detected a difference too small to be worth acting on."
        )
    else:
        verdict, confidence = "INCONCLUSIVE", "Low"
        rationale = (
            f"No significant difference detected (p={z_result['p_value']:.3f} vs alpha="
            f"{cfg.alpha}). Achieved power is {power:.0%}. "
            f"Either the intervention does not work, or the test is too small to tell - "
            f"extend the run before drawing a conclusion."
        )

    supporting = []
    if t_result["significant"]:
        supporting.append(
            f"Secondary metric ({t_result['metric']}) also improved: "
            f"{t_result['mean_difference']:+.2f} ({t_result['pct_change']:+.1f}%), "
            f"p={t_result['p_value']:.2e}, Cohen's d={t_result['cohens_d']:.2f}."
        )
    else:
        supporting.append(
            f"Secondary metric ({t_result['metric']}) showed no significant change "
            f"(p={t_result['p_value']:.3f})."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "supporting_evidence": supporting,
        "practical_significance_met": practical,
        "statistically_significant": significant,
        "adequately_powered": well_powered,
        "practical_threshold_pct": PRACTICAL_SIGNIFICANCE_LIFT_PCT,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_experiment(experiment_key: str, sample_size: int | None = None,
                   seed: int | None = None,
                   treatment_rate_override: float | None = None) -> dict:
    """Run one intervention end to end: simulate, test, value, recommend.

    Parameters
    ----------
    experiment_key
        Key from :func:`list_experiments`.
    sample_size
        Total subjects. Defaults to config.
    seed
        Random seed for reproducibility.
    treatment_rate_override
        Override the assumed treatment effect (dashboard sensitivity control).

    Returns
    -------
    dict
        ``experiment`` (metadata), ``data`` (subject-level frame),
        ``summary`` (per-arm table), ``z_test``, ``chi_square``, ``t_test``,
        ``power_analysis``, ``achieved_power``, ``segments``,
        ``business_impact``, ``recommendation``.
    """
    spec = _experiment_spec(experiment_key)
    data = simulate_experiment(experiment_key, sample_size, seed, treatment_rate_override)

    control = data[data["arm"] == "Control"]
    treatment = data[data["arm"] == "Treatment"]
    sc, nc = int(control["success"].sum()), len(control)
    st_, nt = int(treatment["success"].sum()), len(treatment)

    z_result = two_proportion_z_test(sc, nc, st_, nt)
    chi_result = chi_square_test(data)
    t_result = t_test_continuous(data)
    power = achieved_power(sc, nc, st_, nt)
    impact = estimate_business_impact(experiment_key, z_result["absolute_diff"])

    summary = pd.DataFrame([
        {"arm": "Control", "subjects": nc, "successes": sc,
         "success_rate_pct": round(100 * sc / nc, 2),
         "ci_lower_pct": round(100 * z_result["control_ci"][0], 2),
         "ci_upper_pct": round(100 * z_result["control_ci"][1], 2),
         "mean_processing_time": t_result["control_mean"]},
        {"arm": "Treatment", "subjects": nt, "successes": st_,
         "success_rate_pct": round(100 * st_ / nt, 2),
         "ci_lower_pct": round(100 * z_result["treatment_ci"][0], 2),
         "ci_upper_pct": round(100 * z_result["treatment_ci"][1], 2),
         "mean_processing_time": t_result["treatment_mean"]},
    ])

    log.info("Experiment '%s': lift %+.2fpp (p=%.2e) -> %s",
             experiment_key, z_result["absolute_diff_pp"], z_result["p_value"],
             business_recommendation(z_result, t_result, impact, power)["verdict"])

    return {
        "experiment": {
            "key": experiment_key,
            "name": spec.name,
            "metric": spec.metric,
            "description": " ".join(str(spec.description).split()),
            "sample_size": len(data),
        },
        "data": data,
        "summary": summary,
        "z_test": z_result,
        "chi_square": chi_result,
        "t_test": t_result,
        "power_analysis": required_sample_size(
            float(spec.control_rate),
            float(spec.treatment_rate if treatment_rate_override is None
                  else treatment_rate_override)),
        "achieved_power": round(power, 4),
        "segments": segment_analysis(data),
        "business_impact": impact,
        "recommendation": business_recommendation(z_result, t_result, impact, power),
    }


def run_all_experiments(sample_size: int | None = None) -> pd.DataFrame:
    """Run every catalogued intervention and rank them by annual value.

    This is the portfolio view an operations director actually wants: not
    "did experiment X work?" but "given a fixed budget, which of these four
    projects should we fund first?"
    """
    rows = []
    for key in get_config().ab_testing.experiments:
        result = run_experiment(key, sample_size=sample_size)
        rows.append({
            "experiment": result["experiment"]["name"],
            "key": key,
            "metric": result["experiment"]["metric"],
            "control_rate_pct": round(100 * result["z_test"]["p_control"], 2),
            "treatment_rate_pct": round(100 * result["z_test"]["p_treatment"], 2),
            "lift_pp": result["z_test"]["absolute_diff_pp"],
            "relative_lift_pct": result["z_test"]["relative_lift_pct"],
            "p_value": result["z_test"]["p_value"],
            "significant": result["z_test"]["significant"],
            "achieved_power": result["achieved_power"],
            "annual_value_usd": result["business_impact"]["annual_value_usd"],
            "verdict": result["recommendation"]["verdict"],
        })
    return pd.DataFrame(rows).sort_values("annual_value_usd", ascending=False).reset_index(drop=True)


__all__ = [
    "list_experiments", "simulate_experiment", "two_proportion_z_test",
    "chi_square_test", "t_test_continuous", "required_sample_size", "achieved_power",
    "segment_analysis", "estimate_business_impact", "business_recommendation",
    "run_experiment", "run_all_experiments", "PRACTICAL_SIGNIFICANCE_LIFT_PCT",
]
