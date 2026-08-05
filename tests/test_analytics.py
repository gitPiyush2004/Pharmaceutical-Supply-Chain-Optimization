"""
Tests for the analytics modules: pipeline, products and the statistics toolkit.

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

from src.analytics import ab_testing, pipeline, products


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
# Product catalogue and pricing
# ---------------------------------------------------------------------------
class TestProducts:
    def test_catalogue_kpis_match_the_frame(self, scms):
        kpis = products.catalogue_kpis(scms)
        assert kpis["line_items"] == len(scms)
        assert kpis["molecules"] == scms["molecule"].nunique()
        assert kpis["manufacturing_sites"] == scms["manufacturing_site"].nunique()
        # The pricing analysis runs on a subset, and the page states which.
        assert 0 < kpis["priced_line_items"] <= kpis["line_items"]
        assert kpis["priced_coverage_pct"] == pytest.approx(
            100 * kpis["priced_line_items"] / kpis["line_items"], abs=0.01)

    def test_zero_prices_are_excluded_not_counted(self, scms):
        """A price of zero is an unrecorded price, not a free medicine.

        Counting them would drag every minimum to zero and make every spread ratio
        infinite, so they must be dropped rather than treated as $0 purchases.
        """
        priced = products._priced(scms)
        assert (priced["unit_price_usd"] > 0).all()
        assert len(priced) < len(scms), "some rows do carry an unusable price"

    def test_product_mix_shares_sum_to_100(self, scms):
        mix = products.product_mix(scms)
        assert mix["value_share_pct"].sum() == pytest.approx(100.0, abs=0.1)
        assert mix["value_usd"].is_monotonic_decreasing
        assert mix["on_time_pct"].between(0, 100).all()

    def test_value_concentration_is_cumulative_and_concentrated(self, scms):
        """Spend concentration is the case for doing the pricing work at all."""
        conc = products.value_concentration(scms)
        assert conc["value_usd"].is_monotonic_decreasing
        assert conc["cumulative_share_pct"].is_monotonic_increasing
        assert conc["cumulative_share_pct"].iloc[-1] == pytest.approx(100.0, abs=0.5)
        # Verified: top 5 ~63%, top 15 ~94%. This is what licenses the 80% Pareto
        # line on the page, which is deliberately omitted from flat distributions.
        assert conc.attrs["top_5_share_pct"] > 50
        assert conc.attrs["top_15_share_pct"] > 85

    def test_value_concentration_labels_are_unique(self, scms):
        """Regression: molecule + dosage alone is not a unique product.

        "Abacavir 300mg" exists as a tablet and as a blister-packed tablet,
        "Zidovudine 10mg/ml" as an oral solution and an injection. Those are
        clinically different products, and a shared display label would merge them
        into one bar. The dosage form is appended only where it is needed - 25 of 92
        labels - rather than everywhere, which would push most past 50 characters.
        """
        conc = products.value_concentration(scms)
        assert conc["product"].is_unique
        # Disambiguation is targeted, not blanket: some labels need the form, most
        # do not. A regression that appended it unconditionally would hit 100%.
        with_form = conc["product"].str.contains(r"\(")
        assert 0 < with_form.sum() < len(conc) / 2

        # And it must survive the chart layer's truncation too.
        from src.viz.charts import shorten_labels
        shortened, _ = shorten_labels(conc.copy(), "product")
        assert shortened["product"].is_unique

    def test_pooled_spread_exceeds_within_year_spread(self, scms):
        """The central finding of the module, asserted rather than narrated.

        Pooling ten years of a falling market inflates the apparent price spread.
        If this ever inverted, the page's whole argument would be wrong.
        """
        summary = products.pricing_summary(scms)
        assert (summary["pooled_median_spread_x"]
                > summary["within_year_median_spread_x"])
        assert summary["inflation_factor"] > 1.5
        assert summary["within_year_products"] > summary["pooled_products"], (
            "holding the year fixed should yield more comparable groups, not fewer")

    def test_spread_thresholds_are_enforced(self, scms, cfg):
        """A product bought three times has a range that is just three numbers."""
        floor = cfg.scms.products
        for frame in (products.price_spread(scms),
                      products.price_spread_within_year(scms)):
            assert (frame["line_items"] >= floor.min_shipments_for_spread).all()
            assert (frame["sites"] >= floor.min_sites_for_spread).all()
            # max/min is by definition at least 1.
            assert (frame["spread_x"] >= 1.0).all()

    def test_price_trend_declines_over_the_decade(self, scms):
        """Efavirenz 600mg is the reference case for why pooling misleads."""
        trend = products.price_trend(scms, molecule="Efavirenz", dosage="600mg")
        assert len(trend) >= 8, "expected most of the 2006-2015 window"
        assert trend["delivery_year"].is_monotonic_increasing
        assert trend.attrs["decline_pct"] > 50, "the price collapse is the point"
        assert (trend["min_price"] <= trend["median_price"]).all()
        assert (trend["median_price"] <= trend["max_price"]).all()

    def test_brand_premium_is_like_for_like(self, scms):
        """Both arms must exist in the same product-year, or it is a mix effect."""
        premium = products.brand_premium(scms)
        assert len(premium) > 0
        assert (premium["premium_x"] > 0).all()
        assert premium.attrs["median_premium_x"] > 1.0, (
            "branded product is expected to cost more than the same generic")
        # The headline case: Nevirapine branded as Viramune.
        top = premium.iloc[0]
        assert top["branded_price"] > top["generic_price"]

    def test_brand_premium_label_is_unique_per_row(self, scms):
        """Regression: charting on `molecule` silently stacked four Nevirapine
        product-years into one bar and drew their premiums as a sum - 25x, when the
        true maximum is 7.6x. A price multiple is not additive, so any column used
        as a chart category has to be unique per row.
        """
        premium = products.brand_premium(scms)
        assert premium["label"].is_unique
        assert premium["premium_x"].max() < 20, (
            "a plausible branded premium, not a stacked sum")

    def test_site_prices_are_ranked_cheapest_first(self, scms):
        sites = products.site_prices(scms, molecule="Nevirapine", dosage="200mg")
        assert len(sites) > 1
        assert sites["median_unit_price_usd"].is_monotonic_increasing
        assert sites["on_time_pct"].between(0, 100).all()


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


# ---------------------------------------------------------------------------
# Chart correctness (not cosmetics)
# ---------------------------------------------------------------------------
class TestChartLabels:
    """Label handling that changes what a chart *says*, not how it looks."""

    def test_shortened_labels_never_collide(self):
        """Regression: tail truncation merged two different products.

        "Emtricitabine/Tenofovir Disoproxil Fumarate 200/300mg" and the same
        molecule at "300/200mg" both cut to "Emtricitabine/Tenofovir Dis…". Plotly
        treated them as one x category, stacked their bars and drew a cumulative
        line that ran *backwards*. Duplicate categories are a correctness bug.
        """
        from src.viz.charts import shorten_labels

        frame = pd.DataFrame({"product": [
            "Emtricitabine/Tenofovir Disoproxil Fumarate 200/300mg",
            "Emtricitabine/Tenofovir Disoproxil Fumarate 300/200mg",
            "Emtricitabine/Tenofovir Disoproxil Fumarate 150/300mg",
            "Short name",
        ], "value": [4, 3, 2, 1]})
        out, shortened = shorten_labels(frame, "product")
        assert shortened
        assert out["product"].is_unique, "truncation must not merge categories"
        # The full text stays available for the tooltip.
        assert out["product_full"].tolist() == frame["product"].tolist()

    def test_shortening_keeps_the_distinguishing_tail(self):
        """Dosage lives at the end of a pharmaceutical name, so it must survive."""
        from src.viz.charts import shorten_labels

        frame = pd.DataFrame({"product": [
            "Efavirenz/Lamivudine/Tenofovir Disoproxil Fumarate 600/300/300mg",
        ], "value": [1]})
        out, _ = shorten_labels(frame, "product")
        assert out["product"].iloc[0].endswith("300/300mg")
        assert "…" in out["product"].iloc[0]

    def test_short_labels_are_left_alone(self):
        from src.viz.charts import shorten_labels

        frame = pd.DataFrame({"x": ["Air", "Ocean", "Truck"], "y": [1, 2, 3]})
        out, shortened = shorten_labels(frame, "x")
        assert not shortened
        assert "x_full" not in out.columns

    def test_concentration_chart_preserves_row_order_and_count(self):
        """A merged category would silently drop a bar."""
        from src.analytics import products
        from src.viz import charts

        ranked = products.value_concentration().head(15)
        fig = charts.concentration_chart(
            ranked, "product", "value_usd", "cumulative_share_pct",
            reference_pct=80)
        bars = fig.data[0]
        assert len(bars.x) == len(ranked), "every product must get its own bar"
        assert len(set(bars.x)) == len(ranked), "and its own category"
        # The cumulative line must be monotone, as the underlying data is.
        line = list(fig.data[1].y)
        assert line == sorted(line), "a non-monotone cumulative line means merged rows"
