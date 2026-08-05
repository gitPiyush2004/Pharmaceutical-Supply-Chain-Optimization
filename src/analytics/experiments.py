"""
Statistical comparisons on the **real** USAID SCMS dataset.

Why this module exists separately from ``ab_testing.py``
--------------------------------------------------------
``ab_testing.py`` holds the generic statistics - z-tests, chi-square, power
analysis - written to be dataset-agnostic. This module applies them to real
observational groups in the SCMS data and, critically, handles the things that
observational data forces you to handle and a randomised experiment does not.

These are **natural experiments, not randomised trials.** Nobody assigned
shipments to a fulfilment route at random. That means every comparison here is
open to confounding, and the honest job of this module is to surface that rather
than hide it:

* :func:`compare_groups` gives the headline two-proportion result.
* :func:`stratified_comparison` re-runs it **within** levels of a third variable,
  which is the only way to see whether a headline effect is real or an artefact of
  composition. On this data it is not optional - see below.
* :func:`continuous_comparison` runs mean-based and rank-based tests together,
  because on skewed metrics they disagree and only one of them is right.

The confound that matters most
------------------------------
The headline finding - regional distribution centre fulfilment delivers 82.9% on
time against 94.7% for direct drop, a gap of 11.9 points - is **effect
modification**: the gap is not a stable property of the route, it interacts strongly
with era. Stratified:

===============  ============  ============  =========
Era              Direct Drop   From RDC      Gap
===============  ============  ============  =========
2006-2010        95.3%         93.4%         +1.9 pp
2011-2015        94.4%         73.9%         +20.5 pp
===============  ============  ============  =========

The gap barely existed before 2011. What actually happened is that the RDC channel
**collapsed after 2010** while direct drop held steady. Quoting the pooled 11.9
points describes a permanent structural gap that was never there, and it points at
the wrong remediation: this is a degradation to investigate, not a channel that was
always weak.

Known confounds in this dataset, all verified
---------------------------------------------
* ``inco_term`` is **perfectly collinear** with ``fulfil_via`` - all 5,404 RDC rows
  carry ``N/A - From RDC``. They are one split, not two findings.
* ``vendor_lead_time_days`` is **structurally 100% missing** for RDC, so no
  lead-time comparison is possible across that split.
* ``first_line_designation`` **perfectly predicts** whether freight cost was
  recorded (all 2,445 cross-referenced rows are ``No``, all 6,198 parsed are
  ``Yes``), so freight comparisons across it are undefined rather than merely noisy.
* ``shipment_mode == "Unknown"`` (360 rows, 98.9% on time) is entirely pre-2011 and
  is a recording artefact, not a mode. Excluded by default.
* Transport mode is entangled with era (Ocean is 349/371 post-2010).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.ab_testing import (chi_square_test, compare_continuous,
                                      two_proportion_z_test)
from src.config import get_config
from src.data.scms import load_scms
from src.logger import get_logger

log = get_logger(__name__)

#: Minimum group size before a comparison is reported. Below this a single late
#: shipment swings the rate by several points. Read from
#: ``config.ab_testing.min_group_size`` so the threshold is stated in one place.
MIN_GROUP_SIZE: int = int(get_config().ab_testing.min_group_size)

#: Levels excluded by default because they are recording artefacts, not real
#: categories. See the module docstring.
EXCLUDED_LEVELS: dict[str, set[str]] = {"shipment_mode": {"Unknown"}}

#: Dimensions worth comparing on, with the confound a reader must be told about.
COMPARISON_DIMENSIONS: dict[str, str] = {
    "fulfil_via": "Perfectly collinear with inco_term; effect is concentrated "
                  "post-2010 (see stratified_comparison)",
    "shipment_mode": "Entangled with era - Ocean is almost entirely post-2010",
    "region": "Confounded with product mix and transport mode",
    "first_line_designation": "Freight metrics undefined across this split",
    "product_group": "ARV and HRDT dominate; smaller groups fail chi-square "
                     "expected-count assumptions",
    "era": "The time split itself - useful as the stratifying variable",
}


def _resolve(scms: pd.DataFrame | None) -> pd.DataFrame:
    """Return the caller's frame, or load the real dataset, adding an era column."""
    df = load_scms() if scms is None else scms.copy()
    if "era" not in df.columns:
        df["era"] = np.where(df["delivery_year"] <= 2010, "2006-2010", "2011-2015")
    return df


def _usable(df: pd.DataFrame, dimension: str,
            min_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """Drop artefact levels and groups too small to say anything about."""
    out = df[df[dimension].notna() & df["is_late"].notna()].copy()
    excluded = EXCLUDED_LEVELS.get(dimension, set())
    if excluded:
        out = out[~out[dimension].astype(str).isin(excluded)]
    counts = out[dimension].value_counts()
    keep = counts[counts >= min_size].index
    return out[out[dimension].isin(keep)]


# ---------------------------------------------------------------------------
# Proportion comparisons
# ---------------------------------------------------------------------------
def group_rates(scms: pd.DataFrame | None = None, dimension: str = "fulfil_via",
                min_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """On-time rate per level of ``dimension``, worst first.

    Returns
    -------
    pandas.DataFrame
        Columns ``level``, ``n``, ``late``, ``on_time_pct``, ``mean_delay_days``.
    """
    df = _usable(_resolve(scms), dimension, min_size)
    out = (df.groupby(dimension, dropna=False)
           .agg(n=("shipment_id", "size"), late=("is_late", "sum"),
                mean_delay_days=("delivery_delay_days", "mean"))
           .reset_index()
           .rename(columns={dimension: "level"}))
    out["on_time_pct"] = (100 * (1 - out["late"] / out["n"])).round(2)
    out["mean_delay_days"] = out["mean_delay_days"].round(2)
    out["late"] = out["late"].astype(int)
    return out.sort_values("on_time_pct").reset_index(drop=True)


def compare_groups(scms: pd.DataFrame | None = None, dimension: str = "fulfil_via",
                   min_size: int = MIN_GROUP_SIZE) -> dict:
    """Test whether on-time delivery differs across the levels of ``dimension``.

    Runs a chi-square test across all usable levels, and a two-proportion z-test
    between the best and worst level so the effect has a signed size and a
    confidence interval attached.

    Parameters
    ----------
    scms
        Optional pre-filtered frame.
    dimension
        Column to compare on. See :data:`COMPARISON_DIMENSIONS`.
    min_size
        Minimum level size to include.

    Returns
    -------
    dict
        ``dimension``, ``rates`` (DataFrame), ``chi_square``, ``pairwise_z``
        (best vs worst), ``levels_tested``, ``levels_excluded``, ``confound``,
        and ``verdict``.
    """
    df = _resolve(scms)
    rates = group_rates(df, dimension, min_size)
    if len(rates) < 2:
        return {"dimension": dimension, "rates": rates, "chi_square": None,
                "pairwise_z": None, "levels_tested": len(rates),
                "levels_excluded": [], "confound": COMPARISON_DIMENSIONS.get(dimension),
                "verdict": f"Fewer than two usable levels of {dimension}."}

    usable = _usable(df, dimension, min_size)
    chi = chi_square_test(usable, group=dimension, outcome="is_late")
    p_chi = chi["p_value"]

    worst, best = rates.iloc[0], rates.iloc[-1]
    pairwise = two_proportion_z_test(
        successes_control=int(worst["n"] - worst["late"]), n_control=int(worst["n"]),
        successes_treatment=int(best["n"] - best["late"]), n_treatment=int(best["n"]))

    alpha = float(get_config().ab_testing.alpha)
    all_levels = set(df[dimension].dropna().astype(str).unique())
    excluded = sorted(all_levels - set(rates["level"].astype(str)))

    verdict = (
        f"{best['level']} ({best['on_time_pct']:.2f}%) versus {worst['level']} "
        f"({worst['on_time_pct']:.2f}%) differs by "
        f"{best['on_time_pct'] - worst['on_time_pct']:.2f} points, "
        f"p = {pairwise['p_value']:.2e}. "
        + ("Significant across all levels" if p_chi < alpha
           else "No significant difference across levels")
        + f" (chi-square p = {p_chi:.2e})."
    )

    log.info("compare_groups(%s): %d levels, chi2 p=%.2e", dimension, len(rates), p_chi)
    return {
        "dimension": dimension,
        "rates": rates,
        "chi_square": chi,
        "pairwise_z": pairwise,
        "levels_tested": int(len(rates)),
        "levels_excluded": excluded,
        "confound": COMPARISON_DIMENSIONS.get(dimension),
        "verdict": verdict,
    }


def stratified_comparison(scms: pd.DataFrame | None = None,
                          dimension: str = "fulfil_via", by: str = "era",
                          min_size: int = MIN_GROUP_SIZE) -> dict:
    """Re-run a two-group comparison *within* each level of a third variable.

    The guard against a misleading pooled average, and on this dataset it is load
    bearing. The pooled ``fulfil_via`` gap of ~11.9 points is not a stable
    structural difference: it is ~1.9 points before 2011 and ~20.5 points after,
    because the RDC channel degraded rather than having always been weak.

    Two distinct results are reported, and conflating them would be an overstatement:

    ``is_simpsons_paradox``
        The gap **reverses sign** between strata. This is textbook Simpson's paradox
        and is the stronger claim. On the real ``fulfil_via`` case it is ``False``
        - direct drop leads in both eras.
    ``interaction_detected``
        The gap *magnitude* varies by more than half the pooled gap, with or without
        a sign flip. On the real case it is ``True``, by a factor of ten.

    Both make a pooled average misleading in the same way and with the same practical
    consequence, so the verdict text covers both. They are kept separate because
    claiming a reversal that did not happen is exactly the kind of overstatement that
    a reader checking the numbers will catch.

    Parameters
    ----------
    dimension
        The two-level comparison of interest (e.g. ``fulfil_via``).
    by
        The stratifying variable (e.g. ``era`` or ``region``).

    Returns
    -------
    dict
        ``pooled`` (the naive result), ``strata`` (per-level DataFrame),
        ``gap_range``, ``is_simpsons_paradox``, ``interaction_detected``,
        and ``verdict`` stating which number to quote.
    """
    df = _resolve(scms)
    usable = _usable(df, dimension, min_size)
    levels = list(group_rates(usable, dimension, min_size)["level"])
    if len(levels) != 2:
        return {"pooled": None, "strata": pd.DataFrame(),
                "is_simpsons_paradox": False, "interaction_detected": False,
                "verdict": f"stratified_comparison needs exactly two levels of "
                           f"{dimension}; found {len(levels)}."}

    low, high = levels[0], levels[1]   # worst, best

    def _gap(frame: pd.DataFrame) -> tuple[float, float, float, int, int]:
        a = frame[frame[dimension] == high]
        b = frame[frame[dimension] == low]
        if len(a) < min_size or len(b) < min_size:
            return (np.nan,) * 3 + (len(a), len(b))
        rate_high = 100 * (1 - a["is_late"].mean())
        rate_low = 100 * (1 - b["is_late"].mean())
        return rate_high, rate_low, rate_high - rate_low, len(a), len(b)

    pooled_high, pooled_low, pooled_gap, n_high, n_low = _gap(usable)

    rows: list[dict] = []
    for stratum, frame in usable.groupby(by, dropna=False):
        rate_high, rate_low, gap, nh, nl = _gap(frame)
        if np.isnan(gap):
            continue
        z = two_proportion_z_test(
            successes_control=int((frame[dimension] == low).sum()
                                  - frame.loc[frame[dimension] == low, "is_late"].sum()),
            n_control=nl,
            successes_treatment=int((frame[dimension] == high).sum()
                                    - frame.loc[frame[dimension] == high, "is_late"].sum()),
            n_treatment=nh)
        rows.append({
            by: stratum,
            f"n_{high}": nh, f"n_{low}": nl,
            f"on_time_{high}": round(rate_high, 2),
            f"on_time_{low}": round(rate_low, 2),
            "gap_pp": round(gap, 2),
            "p_value": z["p_value"],
            "significant": z["significant"],
        })

    strata = pd.DataFrame(rows)
    if strata.empty:
        return {"pooled": None, "strata": strata, "is_simpsons_paradox": False,
                "interaction_detected": False,
                "verdict": f"No stratum of {by} has enough data in both groups."}

    gaps = strata["gap_pp"]
    gap_range = float(gaps.max() - gaps.min())
    # A sign flip is textbook Simpson's paradox; a large spread without a flip is
    # still an interaction that makes the pooled number misleading.
    sign_flip = bool((gaps > 0).any() and (gaps < 0).any())
    interaction = bool(gap_range > abs(pooled_gap) * 0.5) or sign_flip

    widest = strata.loc[gaps.idxmax()]
    narrowest = strata.loc[gaps.idxmin()]

    if sign_flip:
        verdict = (
            f"Simpson's paradox: the gap reverses sign across {by}. The pooled "
            f"{pooled_gap:+.2f} pp is meaningless - report per-{by} results only."
        )
    elif interaction:
        verdict = (
            f"The pooled gap of {pooled_gap:+.2f} pp hides a strong interaction with "
            f"{by}: it is {narrowest['gap_pp']:+.2f} pp in {narrowest[by]} and "
            f"{widest['gap_pp']:+.2f} pp in {widest[by]}. Report it as a change over "
            f"{by}, not a fixed difference."
        )
    else:
        verdict = (
            f"The gap is stable across {by} "
            f"({narrowest['gap_pp']:+.2f} to {widest['gap_pp']:+.2f} pp), so the "
            f"pooled {pooled_gap:+.2f} pp is a fair summary."
        )

    log.info("stratified_comparison(%s by %s): pooled %+.2fpp, range %.2fpp, "
             "interaction=%s", dimension, by, pooled_gap, gap_range, interaction)
    return {
        "dimension": dimension,
        "stratified_by": by,
        "pooled": {
            "level_high": high, "level_low": low,
            "on_time_high": round(float(pooled_high), 2),
            "on_time_low": round(float(pooled_low), 2),
            "gap_pp": round(float(pooled_gap), 2),
            "n_high": int(n_high), "n_low": int(n_low),
        },
        "strata": strata,
        "gap_range_pp": round(gap_range, 2),
        "is_simpsons_paradox": sign_flip,
        "interaction_detected": interaction,
        "verdict": verdict,
    }


def continuous_comparison(scms: pd.DataFrame | None = None,
                          metric: str = "total_lead_time_days",
                          dimension: str = "fulfil_via",
                          min_size: int = MIN_GROUP_SIZE) -> dict:
    """Compare a continuous metric across two groups, with the right test chosen.

    Delegates to :func:`src.analytics.ab_testing.compare_continuous`, which runs
    Welch's t-test and Mann-Whitney together and states which to quote based on
    measured skew. On freight metrics the two disagree by 90 orders of magnitude in
    p-value, so picking one blindly is not safe.

    Returns
    -------
    dict
        The ``compare_continuous`` result plus ``metric``, ``dimension`` and a
        ``coverage`` note when the metric is structurally absent for one group.
    """
    df = _resolve(scms)
    usable = _usable(df, dimension, min_size)
    levels = list(group_rates(usable, dimension, min_size)["level"])
    if len(levels) != 2:
        return {"metric": metric, "dimension": dimension,
                "verdict": f"Needs exactly two levels of {dimension}; found {len(levels)}."}

    low, high = levels[0], levels[1]
    a = usable.loc[usable[dimension] == high, metric]
    b = usable.loc[usable[dimension] == low, metric]

    result = compare_continuous(a, b, label_a=str(high), label_b=str(low))
    result.update({"metric": metric, "dimension": dimension,
                   "level_high": high, "level_low": low})

    # Structural absence is not noise - say so rather than returning a NaN test.
    coverage_high = float(a.notna().mean()) if len(a) else 0.0
    coverage_low = float(b.notna().mean()) if len(b) else 0.0
    result["coverage"] = {str(high): round(coverage_high, 3),
                          str(low): round(coverage_low, 3)}
    if min(coverage_high, coverage_low) == 0.0:
        absent = high if coverage_high == 0 else low
        result["verdict"] = (
            f"`{metric}` is structurally 100% missing for {absent}, so this "
            f"comparison is undefined rather than merely under-powered."
        )
    return result


def comparison_catalogue(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run every declared dimension and summarise, for a dashboard overview.

    Returns
    -------
    pandas.DataFrame
        Columns ``dimension``, ``levels``, ``best``, ``worst``, ``gap_pp``,
        ``chi_square_p``, ``significant``, ``expected_counts_adequate``,
        ``confound``.
    """
    df = _resolve(scms)
    rows: list[dict] = []
    for dimension in COMPARISON_DIMENSIONS:
        if dimension not in df.columns:
            continue
        result = compare_groups(df, dimension)
        rates, chi = result["rates"], result["chi_square"]
        if chi is None or len(rates) < 2:
            continue
        rows.append({
            "dimension": dimension,
            "levels": result["levels_tested"],
            "best": f"{rates.iloc[-1]['level']} ({rates.iloc[-1]['on_time_pct']:.1f}%)",
            "worst": f"{rates.iloc[0]['level']} ({rates.iloc[0]['on_time_pct']:.1f}%)",
            "gap_pp": round(rates.iloc[-1]["on_time_pct"] - rates.iloc[0]["on_time_pct"], 2),
            "chi_square_p": chi["p_value"],
            "significant": chi["significant"],
            "expected_counts_adequate": chi["expected_counts_adequate"],
            "confound": result["confound"],
        })
    return pd.DataFrame(rows).sort_values("gap_pp", ascending=False).reset_index(drop=True)


__all__ = [
    "group_rates", "compare_groups", "stratified_comparison",
    "continuous_comparison", "comparison_catalogue",
    "COMPARISON_DIMENSIONS", "MIN_GROUP_SIZE", "EXCLUDED_LEVELS",
]
