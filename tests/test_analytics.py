"""
Tests for the analytics modules: pipeline, market and the statistics toolkit.

The focus is on internal consistency - shares summing to 100, funnels that are
actually monotone, statistics anchored against hand-computed values - because
those are the errors that silently produce a plausible but wrong dashboard rather
than a crash.

Several tests here exist because the bug they guard against was real. The
monotonicity test was written after a "delivered early" band was drawn below
"delivered on time", which reads as further attrition when early deliveries are a
*subset* of on-time ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import ab_testing, market, pipeline


# ---------------------------------------------------------------------------
# Order-to-delivery pipeline
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_value_funnel_is_monotone(self, scms):
        """Each stage must be a strict subset of the one above it.

        A funnel chart implies containment. Plotting a band that is larger than
        its predecessor - or a sibling category rather than a subset - makes the
        chart say something false about the process.
        """
        funnel = pipeline.value_funnel(scms)
        assert funnel["share_pct"].is_monotonic_decreasing
        assert funnel["line_items"].is_monotonic_decreasing
        assert funnel.loc[0, "share_pct"] == pytest.approx(100.0)

    def test_value_funnel_reconciles_with_raw_totals(self, scms):
        funnel = pipeline.value_funnel(scms)
        assert funnel.loc[0, "value_usd"] == pytest.approx(
            scms["line_value_usd"].sum(), rel=1e-6)
        # Late value is the complement of the on-time band, by construction.
        on_time = funnel.loc[2, "value_usd"]
        assert funnel.attrs["late_value_usd"] == pytest.approx(
            funnel.loc[0, "value_usd"] - on_time, abs=1.0)

    def test_lateness_funnel_is_cumulative(self, scms):
        """Wider thresholds must include everything a narrower one caught."""
        lateness = pipeline.lateness_funnel(scms)
        assert lateness["share_pct"].is_monotonic_increasing
        assert lateness["items_share_pct"].is_monotonic_increasing
        assert lateness["share_pct"].iloc[-1] == pytest.approx(100.0)

    def test_traceability_separates_structural_from_missing(self, scms):
        """The distinction the whole data-quality argument rests on."""
        trace = pipeline.traceability(scms)
        total = len(scms)
        # Every line item is accounted for exactly once per milestone.
        accounted = (trace["recorded"] + trace["structurally_absent"]
                     + trace["genuinely_missing"])
        assert (accounted == total).all()
        # The purchase-order milestone is the one with structural absence.
        po = trace[trace["stage"].str.contains("Purchase order")].iloc[0]
        assert po["structurally_absent"] > 5_000

    def test_traceability_is_deliberately_not_monotone(self, scms):
        """Guards the reason this is a stacked bar and not a funnel.

        Coverage runs 74% -> 44% -> 100%. If a future change made it monotone the
        chart choice would need revisiting, so the non-monotonicity is asserted
        rather than assumed.
        """
        trace = pipeline.traceability(scms)
        assert not trace["coverage_pct"].is_monotonic_decreasing

    def test_kpis_are_internally_consistent(self, scms):
        kpis = pipeline.pipeline_kpis(scms)
        assert kpis["value_on_time_usd"] + kpis["value_late_usd"] == pytest.approx(
            kpis["value_ordered_usd"], rel=1e-6)
        # Early, exact-day and late must partition every shipment.
        assert (kpis["early_pct"] + kpis["exact_day_pct"]
                + kpis["late_pct"]) == pytest.approx(100.0, abs=0.05)
        assert 0 <= kpis["po_traceability_pct"] <= 100


# ---------------------------------------------------------------------------
# Indian market structure
# ---------------------------------------------------------------------------
class TestMarket:
    def test_kpis_match_the_frame(self, indian_medicines):
        kpis = market.market_kpis(indian_medicines)
        assert kpis["products"] == len(indian_medicines)
        assert kpis["manufacturers"] == indian_medicines["manufacturer"].nunique()
        assert 0 < kpis["median_price_inr"] < kpis["p90_price_inr"]

    def test_concentration_shares_are_cumulative(self, indian_medicines):
        conc = market.manufacturer_concentration(indian_medicines)
        assert conc["products"].is_monotonic_decreasing
        assert conc["cumulative_share_pct"].is_monotonic_increasing
        assert conc["cumulative_share_pct"].iloc[-1] <= 100.01

    def test_market_is_measured_as_fragmented(self, indian_medicines):
        """The headline market finding, pinned.

        The Herfindahl index here is near zero - 7,642 manufacturers with no
        incumbent. The Home and market pages both assert fragmentation, and a
        regression that grouped manufacturers too aggressively would quietly turn
        that story into a false concentration finding.
        """
        summary = market.concentration_summary(indian_medicines)
        assert summary.attrs["hhi"] < 1500
        assert "Fragmented" in summary.attrs["interpretation"]

    def test_price_buckets_partition_all_priced_products(self, indian_medicines):
        prices = market.price_distribution(indian_medicines)
        # Strictly positive, not merely non-null: four products carry a price of
        # exactly 0, which cannot be placed on a log scale and is treated as
        # unrecorded rather than as free.
        priced = int((indian_medicines["price_inr"] > 0).sum())
        assert prices["products"].sum() == priced
        assert prices["share_pct"].sum() == pytest.approx(100.0, abs=0.1)
        # Log-spaced, so each bucket's floor is the previous bucket's ceiling.
        assert np.allclose(prices["bucket_low"].to_numpy()[1:],
                           prices["bucket_high"].to_numpy()[:-1])

    def test_discontinuation_is_reported_with_its_caveat(self, indian_medicines):
        """The manufacturer effect must dominate the price effect.

        This is why the project reports discontinuation descriptively instead of
        modelling it. If the spreads ever inverted, that decision would need
        revisiting - so the asymmetry is asserted, not just described in prose.
        """
        disc = market.discontinuation_analysis(indian_medicines)
        assert disc["manufacturer_spread_pp"] > 10 * disc["price_band_spread_pp"]
        assert disc["price_is_predictive"] is False
        assert "listing" in disc["caveat"]

    def test_ingredient_analysis_is_ranked_and_plausible(self, indian_medicines):
        ingredients = market.ingredient_analysis(indian_medicines)
        assert ingredients["brands"].is_monotonic_decreasing
        # A molecule cannot be sold by more manufacturers than it has brands.
        assert (ingredients["manufacturers"] <= ingredients["brands"]).all()

    def test_pack_form_shares_are_bounded(self, indian_medicines):
        packs = market.pack_form_analysis(indian_medicines)
        assert packs["share_pct"].between(0, 100).all()
        assert packs["products"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Statistics toolkit
# ---------------------------------------------------------------------------
class TestProportionTests:
    def test_z_test_matches_a_hand_computed_case(self):
        """Anchor the implementation against an independently known result."""
        result = ab_testing.two_proportion_z_test(920, 1000, 951, 1000)
        assert result["p_control"] == pytest.approx(0.920)
        assert result["p_treatment"] == pytest.approx(0.951)
        assert result["absolute_diff"] == pytest.approx(0.031, abs=1e-6)
        # Sign convention is (treatment - control), so an improvement is positive.
        assert result["z_statistic"] > 0
        # Hand-computed: pooled p = 0.9355, SE = 0.01098, z = 0.031 / 0.01098.
        assert result["z_statistic"] == pytest.approx(2.822, abs=0.01)
        assert result["p_value"] < 0.05
        assert result["ci_lower"] < result["absolute_diff"] < result["ci_upper"]

    def test_z_test_finds_no_effect_when_there_is_none(self):
        result = ab_testing.two_proportion_z_test(500, 1000, 500, 1000)
        assert result["absolute_diff"] == pytest.approx(0.0)
        assert result["p_value"] > 0.9
        assert not result["significant"]

    def test_chi_square_equals_z_squared_on_a_2x2(self, scms):
        """Two independent routes to the same question must not disagree.

        This is the cross-check the module docstring promises. It only holds with
        Yates' correction disabled, so it also guards that setting.
        """
        two_level = scms[scms["fulfil_via"].notna()]
        chi = ab_testing.chi_square_test(two_level, "fulfil_via", "is_late")
        groups = sorted(two_level["fulfil_via"].unique())
        counts = [(int((two_level["fulfil_via"] == g).sum()),
                   int(((two_level["fulfil_via"] == g)
                        & (two_level["is_late"] == 0)).sum())) for g in groups]
        z = ab_testing.two_proportion_z_test(
            counts[0][1], counts[0][0], counts[1][1], counts[1][0])
        # Both statistics are rounded to 4 decimals in their result dicts, so the
        # tolerance reflects that rather than the underlying arithmetic.
        assert chi["chi2_statistic"] == pytest.approx(z["z_statistic"] ** 2, rel=1e-3)
        assert chi["significant"] == z["significant"]
        assert chi["degrees_of_freedom"] == 1

    def test_chi_square_flags_inadequate_expected_counts(self):
        """A tiny cell must be reported, not silently given a p-value."""
        frame = pd.DataFrame({
            "group": ["A"] * 100 + ["B"] * 4,
            "late": [0] * 95 + [1] * 5 + [0] * 3 + [1],
        })
        chi = ab_testing.chi_square_test(frame, "group", "late")
        assert chi["min_expected_count"] < 5
        assert chi["expected_counts_adequate"] is False


class TestPowerAnalysis:
    def test_required_sample_size_falls_as_effect_grows(self):
        small_effect = ab_testing.required_sample_size(0.885, 0.895)
        large_effect = ab_testing.required_sample_size(0.885, 0.950)
        assert small_effect["n_per_arm"] > large_effect["n_per_arm"]
        assert large_effect["n_total"] == 2 * large_effect["n_per_arm"]

    def test_minimum_detectable_effect_shrinks_with_sample_size(self):
        """The property that makes the MDE the right tool for a null result."""
        small = ab_testing.minimum_detectable_effect(0.885, 200, 200)
        large = ab_testing.minimum_detectable_effect(0.885, 20_000, 20_000)
        assert abs(large["mde_pp"]) < abs(small["mde_pp"])
        assert 0 < abs(large["mde_pp"]) < 5

    def test_post_hoc_power_is_low_on_a_true_null(self):
        """Documents the trap the MDE exists to avoid.

        Post-hoc power is computed at the *observed* effect size, so a genuine
        null mechanically returns near-zero power however large the sample. This
        test pins that behaviour so nobody later "fixes" it by reading the number
        as evidence of an underpowered test.
        """
        power = ab_testing.achieved_power(8_850, 10_000, 8_851, 10_000)
        assert power < 0.10

    def test_null_verdict_uses_the_mde_not_the_power(self):
        z = ab_testing.two_proportion_z_test(8_850, 10_000, 8_851, 10_000)
        power = ab_testing.achieved_power(8_850, 10_000, 8_851, 10_000)
        mde = ab_testing.minimum_detectable_effect(0.885, 10_000, 10_000)

        without = ab_testing.business_recommendation(z, power)
        with_mde = ab_testing.business_recommendation(z, power, mde=mde)
        assert without["verdict"] == "INCONCLUSIVE"
        assert with_mde["verdict"].startswith("NO EFFECT")
        assert with_mde["minimum_detectable_effect_pp"] == mde["mde_pp"]


class TestContinuousTests:
    def test_welch_matches_scipy_on_a_known_case(self):
        rng = np.random.default_rng(42)
        a = pd.Series(rng.normal(100, 10, 500))
        b = pd.Series(rng.normal(104, 10, 500))
        result = ab_testing.welch_t_test(a, b)
        from scipy import stats

        expected = stats.ttest_ind(a, b, equal_var=False)
        assert result["t_statistic"] == pytest.approx(expected.statistic, abs=1e-4)
        assert result["p_value"] == pytest.approx(expected.pvalue, rel=1e-6)
        assert result["significant"]

    def test_skew_rule_selects_the_rank_test(self, cfg):
        """A heavy tail must route the decision to Mann-Whitney."""
        rng = np.random.default_rng(7)
        a = pd.Series(np.concatenate([rng.normal(10, 2, 500), [50_000, 90_000]]))
        b = pd.Series(rng.normal(12, 2, 500))
        result = ab_testing.compare_continuous(a, b)
        assert result["max_abs_skew"] > cfg.ab_testing.skew_limit
        assert result["recommended_test"] in {"mann_whitney", "both"}

    def test_symmetric_data_keeps_welch(self):
        rng = np.random.default_rng(11)
        a = pd.Series(rng.normal(10, 2, 500))
        b = pd.Series(rng.normal(12, 2, 500))
        result = ab_testing.compare_continuous(a, b)
        assert result["recommended_test"] == "welch"
        assert result["tests_agree"]

    def test_tail_only_difference_is_reported_as_both(self, scms):
        """The other disagreement direction, which needs the opposite reading.

        When Welch is significant and the rank test is not, the means moved and
        the medians did not. Calling Welch an artefact there would be wrong, so
        the module returns "both" and says the effect lives in the tail.

        This uses the real comparison rather than synthetic data, because the real
        one is what motivated the branch: delivery delay by era has means of -5.0
        and -7.5 days but medians of exactly 0 in both eras, since 61% of
        deliveries land on their scheduled day.
        """
        era = np.where(scms["delivery_year"] <= 2010, "early", "late")
        a = scms.loc[era == "late", "delivery_delay_days"]
        b = scms.loc[era == "early", "delivery_delay_days"]
        result = ab_testing.compare_continuous(a, b, label_a="2011-2015",
                                               label_b="2006-2010")
        assert result["welch"]["significant"]
        assert not result["mann_whitney"]["significant"]
        assert result["mann_whitney"]["median_a"] == result["mann_whitney"]["median_b"]
        assert result["recommended_test"] == "both"
        assert "tail" in result["verdict"]


class TestBusinessTranslation:
    def test_penalty_impact_scales_linearly(self, cfg):
        one = ab_testing.penalty_impact(0.10, 1_000)
        two = ab_testing.penalty_impact(0.20, 1_000)
        assert two["annual_value_usd"] == pytest.approx(2 * one["annual_value_usd"])
        assert one["penalty_per_late_shipment_usd"] == pytest.approx(
            float(cfg.economics.late_shipment_penalty))
        assert "assumption" in one

    def test_significant_but_trivial_effect_is_rejected(self):
        """Statistical significance alone must not produce an ACT verdict."""
        # A 0.2pp gap on half a million rows: real, and far too small to act on.
        z = ab_testing.two_proportion_z_test(440_000, 500_000, 441_000, 500_000)
        assert z["significant"]
        verdict = ab_testing.business_recommendation(z, power=0.99)
        assert verdict["verdict"] == "DO NOT ACT"
        assert not verdict["practical_significance_met"]
