"""
Tests for the surviving analytics modules: funnel, stability and A/B testing.

The focus is on internal consistency - shares summing to 100, conversions
reconciling with raw units, coefficients carrying the physically correct sign -
because those are the errors that silently produce a plausible but wrong
dashboard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics import ab_testing, funnel, stability


class TestFunnel:
    def test_summary_has_one_row_per_stage(self, batches, cfg):
        summary = funnel.funnel_summary(batches)
        assert len(summary) == len(cfg.funnel.stages)
        assert summary["stage"].tolist() == list(cfg.funnel.stages)

    def test_conversion_reconciles_with_units(self, batches):
        """Each stage's conversion must equal its units over the previous stage's."""
        summary = funnel.funnel_summary(batches)
        for i in range(1, len(summary)):
            expected = 100 * summary.loc[i, "units"] / summary.loc[i - 1, "units"]
            assert summary.loc[i, "conversion_from_previous_pct"] == pytest.approx(
                expected, abs=0.05)

    def test_dropoff_is_complement_of_conversion(self, batches):
        summary = funnel.funnel_summary(batches).iloc[1:]
        total = summary["conversion_from_previous_pct"] + summary["dropoff_pct"]
        assert (total - 100.0).abs().max() < 0.05

    def test_cumulative_conversion_matches_end_to_end_yield(self, batches):
        summary = funnel.funnel_summary(batches)
        kpis = funnel.funnel_kpis(batches)
        assert summary["cumulative_conversion_pct"].iloc[-1] == pytest.approx(
            kpis["end_to_end_yield_pct"], abs=0.05)

    def test_quality_testing_is_flagged_as_bottleneck(self, batches):
        """The designed structural finding - if this stops holding, the
        generator's calibration has drifted."""
        bottlenecks = funnel.identify_bottlenecks(batches)
        flagged = bottlenecks[bottlenecks["is_bottleneck"]]["stage"].tolist()
        assert "Quality Testing" in flagged

    def test_loss_attribution_shares_are_coherent(self, batches):
        losses = funnel.loss_attribution(batches, top_n=10)
        assert len(losses) <= 10
        assert losses["share_of_total_loss_pct"].between(0, 100).all()
        # Cumulative share must be non-decreasing down the ranking.
        assert losses["cumulative_share_pct"].is_monotonic_increasing

    def test_stage_delays_share_of_cycle_sums_to_100(self, batches):
        delays = funnel.stage_delays(batches)
        assert delays["share_of_cycle_pct"].sum() == pytest.approx(100.0, abs=0.5)

    def test_filtered_input_changes_result(self, batches):
        """Passing a filtered frame must actually be honoured, not ignored."""
        europe = batches[batches["region"] == "Europe"]
        assert funnel.funnel_kpis(europe)["total_batches"] == len(europe)


class TestStability:
    def test_degradation_model_has_real_explanatory_power(self, batches):
        """Data is generated from a deterministic model plus noise, so a low
        R-squared means the analysis has a bug."""
        model = stability.shelf_life_model(batches)
        assert model["r_squared"] > 0.5, f"R2 only {model['r_squared']:.3f}"
        assert model["n_observations"] == len(batches)

    def test_degradation_coefficients_have_physical_signs(self, batches):
        """Higher temperature, humidity and duration must all reduce potency."""
        coefficients = stability.shelf_life_model(batches)["coefficients"]
        for term in ("storage_temp_c", "storage_humidity_pct", "storage_duration_days"):
            match = next((k for k in coefficients if term in k), None)
            assert match is not None, f"no coefficient for {term}"
            assert coefficients[match] < 0, f"{match} should reduce potency"

    def test_shelf_life_estimate_is_plausible(self, batches):
        model = stability.shelf_life_model(batches)
        assert 30 < model["estimated_shelf_life_days"] < 3650

    def test_excursion_impact_is_significant_and_negative(self, batches):
        impact = stability.excursion_impact(batches)
        assert impact["p_value"] < 0.05
        assert impact["significant"]
        assert impact["mean_potency_with_excursion"] < impact["mean_potency_without_excursion"]
        # Confidence interval must exclude zero when the result is significant.
        assert impact["difference_ci_low"] * impact["difference_ci_high"] > 0

    def test_condition_effects_are_binned_and_ordered(self, batches):
        for fn in (stability.temperature_effect, stability.humidity_effect,
                   stability.duration_effect):
            effect = fn(batches, bins=6)
            assert len(effect) > 0
            assert effect["batches"].sum() > 0
            assert effect["avg_potency_pct"].between(60, 100).all()

    def test_higher_temperature_lowers_potency(self, batches):
        """Directional check on the binned view, independent of the regression."""
        effect = stability.temperature_effect(batches, bins=6)
        ambient = effect[effect["cohort"].str.contains("Ambient|Room", case=False,
                                                       na=False)]
        frame = ambient if len(ambient) >= 3 else effect
        frame = frame.sort_values("bin_mid")
        assert frame["avg_potency_pct"].iloc[-1] < frame["avg_potency_pct"].iloc[0]

    def test_expiry_buckets_cover_all_batches(self, batches):
        profile = stability.expiry_risk_profile(batches)
        assert profile["batches"].sum() == len(batches)
        assert profile["batch_pct"].sum() == pytest.approx(100.0, abs=0.1)


class TestABTesting:
    def test_catalogue_matches_config(self, cfg):
        catalogue = ab_testing.list_experiments()
        assert len(catalogue) == len(cfg.ab_testing.experiments)
        assert set(catalogue["key"]) == set(cfg.ab_testing.experiments)

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

    def test_chi_square_agrees_with_z_test(self):
        """Two independent routes to the same question must not disagree."""
        experiment = ab_testing.run_experiment("qa_automation", sample_size=4000)
        assert (experiment["chi_square"]["significant"]
                == experiment["z_test"]["significant"])
        # For a 2x2 table, chi-square equals the square of the z-statistic.
        assert experiment["chi_square"]["chi2_statistic"] == pytest.approx(
            experiment["z_test"]["z_statistic"] ** 2, rel=0.05)

    def test_power_rises_with_sample_size(self):
        small = ab_testing.run_experiment("qa_automation", sample_size=300)
        large = ab_testing.run_experiment("qa_automation", sample_size=6000)
        assert large["achieved_power"] > small["achieved_power"]

    def test_experiment_returns_full_workup(self):
        result = ab_testing.run_experiment("cold_chain_iot", sample_size=1500)
        for key in ("experiment", "data", "summary", "z_test", "chi_square",
                    "t_test", "power_analysis", "achieved_power", "segments",
                    "business_impact", "recommendation"):
            assert key in result
        assert result["recommendation"]["verdict"] in {
            "ADOPT", "ADOPT WITH MONITORING", "DO NOT ADOPT", "INCONCLUSIVE"}

    def test_all_experiments_run(self, cfg):
        portfolio = ab_testing.run_all_experiments(sample_size=1000)
        assert len(portfolio) == len(cfg.ab_testing.experiments)
        assert portfolio["p_value"].between(0, 1).all()


