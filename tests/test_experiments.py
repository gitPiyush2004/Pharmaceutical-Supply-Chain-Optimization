"""
Tests for the statistical-comparison layer on real SCMS data.

Two of these encode findings that were originally reported wrongly, so they exist
to stop the project regressing to the misleading version:

* **B3** - the fulfilment-route gap interacts strongly with era, so the pooled figure
  must not be presented as a stable structural difference. Note the precise claim:
  this is *effect modification*, not Simpson's paradox, because the gap never reverses
  sign - and the tests below assert both halves of that distinction.
* **B4** - on skewed freight metrics a mean-based test and a rank-based test reach
  opposite conclusions, and the skewed one must not be quoted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import experiments as ex
from src.analytics.ab_testing import compare_continuous, mann_whitney_test
from src.data.scms import load_scms


@pytest.fixture(scope="module")
def scms():
    return load_scms()


# ---------------------------------------------------------------------------
class TestGroupComparison:
    def test_artefact_levels_are_excluded(self, scms):
        """`shipment_mode == "Unknown"` is a pre-2011 recording artefact at 98.9%
        on-time. Left in, it looks like the best-performing transport mode."""
        rates = ex.group_rates(scms, "shipment_mode")
        assert "Unknown" not in set(rates["level"].astype(str))

    def test_small_groups_are_excluded(self, scms):
        """A group with 3 shipments and 1 late delivery would otherwise outrank a
        group with 5,000 at 90%."""
        rates = ex.group_rates(scms, "region", min_size=ex.MIN_GROUP_SIZE)
        assert (rates["n"] >= ex.MIN_GROUP_SIZE).all()

    def test_rates_are_worst_first(self, scms):
        rates = ex.group_rates(scms, "region")
        assert rates["on_time_pct"].is_monotonic_increasing

    def test_chi_square_flags_inadequate_expected_counts(self, scms):
        """Below 5 expected per cell the chi-square approximation is unreliable;
        the result must say so rather than quote a p-value the reader trusts."""
        result = ex.compare_groups(scms, "region")
        chi = result["chi_square"]
        assert "expected_counts_adequate" in chi
        assert chi["expected_counts_adequate"] == (chi["min_expected_count"] >= 5)

    def test_every_dimension_carries_a_confound_note(self, scms):
        """An observational comparison without its confound stated is misleading."""
        for dimension in ex.COMPARISON_DIMENSIONS:
            if dimension not in scms.columns and dimension != "era":
                continue
            result = ex.compare_groups(scms, dimension)
            if result["chi_square"] is None:
                continue
            assert result["confound"], f"{dimension} has no confound documented"

    def test_known_null_result_is_reported_as_null(self, scms):
        """`first_line_designation` genuinely has no on-time effect (p ~ 0.84) on
        large balanced samples. Reporting a false positive here would be worse than
        missing a real effect."""
        result = ex.compare_groups(scms, "first_line_designation")
        assert not result["chi_square"]["significant"]
        assert result["chi_square"]["p_value"] > 0.05

    def test_known_strong_result_is_significant(self, scms):
        result = ex.compare_groups(scms, "fulfil_via")
        assert result["chi_square"]["significant"]
        assert result["chi_square"]["p_value"] < 1e-50

    def test_catalogue_covers_the_declared_dimensions(self, scms):
        catalogue = ex.comparison_catalogue(scms)
        assert len(catalogue) >= 4
        assert catalogue["gap_pp"].is_monotonic_decreasing
        # Both a strong effect and a null must appear, or the page tells half a story.
        assert catalogue["significant"].any()
        assert (~catalogue["significant"]).any()


# ---------------------------------------------------------------------------
class TestStratification:
    """B3 - the pooled fulfilment-route gap hides an era interaction."""

    def test_it_is_not_claimed_to_be_a_sign_reversal(self, scms):
        """The precision half of the finding, and it is easy to get wrong.

        Textbook Simpson's paradox requires the difference to reverse sign between
        strata. Here it does not - direct drop leads in both eras - so the honest
        name is effect modification. The README, the dashboard and the notebook all
        say so, and this test fails if the code ever starts claiming otherwise.
        """
        result = ex.stratified_comparison(scms, dimension="fulfil_via", by="era")
        assert not result["is_simpsons_paradox"], (
            "no sign reversal exists in this data; claiming Simpson's paradox "
            "would overstate the finding")
        assert (result["strata"]["gap_pp"] > 0).all(), "the sign must not flip"
        assert "interaction" in result["verdict"].lower()

    def test_interaction_is_detected(self, scms):
        result = ex.stratified_comparison(scms, dimension="fulfil_via", by="era")
        assert result["interaction_detected"], (
            "the era interaction is no longer being detected - the pooled figure "
            "would be reported as a stable gap again")

    def test_gap_widens_sharply_after_2010(self, scms):
        """Verified: ~+1.9pp pre-2011, ~+20.5pp post-2010."""
        result = ex.stratified_comparison(scms, dimension="fulfil_via", by="era")
        strata = result["strata"].set_index("era")
        early = strata.loc["2006-2010", "gap_pp"]
        late = strata.loc["2011-2015", "gap_pp"]
        assert early < 5, f"early-era gap {early} should be small"
        assert late > 15, f"late-era gap {late} should be large"
        assert late - early > 10

    def test_pooled_gap_sits_between_the_strata(self, scms):
        """The signature of the problem: the pooled number describes neither era."""
        result = ex.stratified_comparison(scms, dimension="fulfil_via", by="era")
        pooled = result["pooled"]["gap_pp"]
        gaps = result["strata"]["gap_pp"]
        assert gaps.min() < pooled < gaps.max()

    def test_verdict_tells_the_reader_which_number_to_use(self, scms):
        result = ex.stratified_comparison(scms, dimension="fulfil_via", by="era")
        verdict = result["verdict"].lower()
        assert "interaction" in verdict or "simpson" in verdict
        assert "pooled" in verdict

    def test_stable_effect_is_not_flagged_as_interaction(self, scms):
        """Guard against the detector firing on everything: a synthetic split with
        the same effect in both strata must come back clean."""
        rng = np.random.default_rng(0)
        n = 4000
        synthetic = pd.DataFrame({
            "shipment_id": range(n),
            "fulfil_via": np.tile(["A", "B"], n // 2),
            "era": np.repeat(["early", "late"], n // 2),
            "delivery_delay_days": 0,
        })
        # 10pp gap in both eras, by construction.
        base = np.where(synthetic["fulfil_via"] == "A", 0.20, 0.10)
        synthetic["is_late"] = (rng.random(n) < base).astype(int)
        synthetic["delivery_year"] = np.where(synthetic["era"] == "early", 2008, 2013)

        result = ex.stratified_comparison(synthetic, dimension="fulfil_via", by="era")
        assert not result["is_simpsons_paradox"]
        assert not result["interaction_detected"], (
            "a genuinely stable effect is being flagged as an interaction")


# ---------------------------------------------------------------------------
class TestSkewedMetrics:
    """B4 - Welch's t-test is invalid on freight ratios."""

    def test_freight_is_extremely_skewed(self, scms):
        from scipy import stats

        freight = scms["freight_cost_per_kg"].dropna()
        assert abs(stats.skew(freight)) > ex.MIN_GROUP_SIZE / 10  # comfortably > 2
        # The mean is not a summary of this distribution.
        assert freight.mean() > 4 * freight.median()

    def test_rank_test_is_recommended_for_freight(self, scms):
        result = ex.continuous_comparison(scms, metric="freight_cost_per_kg",
                                          dimension="fulfil_via")
        assert result["recommended_test"] == "mann_whitney"
        assert result["max_abs_skew"] > 2.0

    def test_the_two_tests_disagree_on_freight(self, scms):
        """This is the whole reason the guard exists: Welch says p ~ 0.25,
        Mann-Whitney says p ~ 1e-93, on identical data."""
        result = ex.continuous_comparison(scms, metric="freight_cost_per_kg",
                                          dimension="fulfil_via")
        assert not result["tests_agree"]
        assert result["welch"]["p_value"] > 0.05
        assert result["mann_whitney"]["p_value"] < 1e-20
        assert "artefact of outliers" in result["verdict"]

    def test_welch_is_kept_where_it_is_valid(self, scms):
        """Lead times are only mildly skewed, so the mean-based test applies."""
        result = ex.continuous_comparison(scms, metric="total_lead_time_days",
                                          dimension="fulfil_via")
        assert result["recommended_test"] == "welch"
        assert result["max_abs_skew"] <= 2.0
        assert result["tests_agree"]

    def test_structural_absence_is_named_not_silently_nan(self, scms):
        """`vendor_lead_time_days` is 100% missing for RDC by design. The result
        must say the comparison is undefined, not return a NaN p-value."""
        result = ex.continuous_comparison(scms, metric="vendor_lead_time_days",
                                          dimension="fulfil_via")
        assert min(result["coverage"].values()) == 0.0
        assert "undefined" in result["verdict"]

    def test_mann_whitney_handles_degenerate_input(self):
        result = mann_whitney_test(pd.Series([1.0]), pd.Series([2.0, 3.0]))
        assert not result["significant"]
        assert "insufficient" in result.get("note", "")

    def test_compare_continuous_agrees_on_clean_normal_data(self):
        """Sanity check: with no skew both tests should agree and Welch is chosen."""
        rng = np.random.default_rng(1)
        a = pd.Series(rng.normal(10, 2, 500))
        b = pd.Series(rng.normal(12, 2, 500))
        result = compare_continuous(a, b)
        assert result["recommended_test"] == "welch"
        assert result["tests_agree"]
        assert result["welch"]["significant"]
