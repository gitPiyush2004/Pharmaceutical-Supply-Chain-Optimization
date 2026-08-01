"""
Tests for the analytics modules.

The focus is on internal consistency - shares summing to 100, conversions
reconciling with raw units, forecasts landing in the future - because those are
the errors that silently produce a plausible but wrong dashboard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics import ab_testing, forecasting, funnel, inventory, simulation, stability
# Aliased: the `shipments` fixture would otherwise shadow the module inside tests.
from src.analytics import shipments as shp


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


class TestInventory:
    def test_abc_shares_sum_to_100(self, inventory_snapshots):
        abc = inventory.abc_analysis(inventory_snapshots)
        assert abc["value_share_pct"].sum() == pytest.approx(100.0, abs=0.1)
        assert abc["cumulative_share_pct"].iloc[-1] == pytest.approx(100.0, abs=0.1)

    def test_abc_classes_follow_thresholds(self, inventory_snapshots, cfg):
        abc = inventory.abc_analysis(inventory_snapshots)
        assert set(abc["abc_class"]).issubset({"A", "B", "C"})
        # Cumulative share must be sorted descending by value, so class A rows
        # all sit at or below the A threshold boundary.
        a_rows = abc[abc["abc_class"] == "A"]
        if len(a_rows):
            boundary = a_rows["cumulative_share_pct"].max()
            assert boundary <= cfg.inventory.abc_thresholds.A * 100 + 25

    def test_turnover_and_days_of_inventory_are_reciprocal(self, inventory_snapshots):
        turnover = inventory.inventory_turnover(inventory_snapshots)
        positive = turnover[turnover["turnover"] > 0]
        expected = 365 / positive["turnover"]
        assert (positive["days_of_inventory"] - expected).abs().max() < 1.0

    def test_stockout_risk_respects_threshold(self, inventory_snapshots, cfg):
        risk = inventory.stockout_risk(inventory_snapshots)
        if len(risk):
            assert (risk["risk_ratio"] < cfg.inventory.stockout_risk_threshold + 1e-9).all()
            assert set(risk["risk_level"]).issubset({"High", "Medium", "Low"})

    def test_overstock_respects_threshold(self, inventory_snapshots, cfg):
        over = inventory.overstock_risk(inventory_snapshots)
        if len(over):
            assert (over["months_of_supply"]
                    > cfg.inventory.overstock_months_of_supply - 1e-9).all()
            assert (over["excess_units"] >= 0).all()

    def test_utilisation_is_a_valid_percentage(self, inventory_snapshots):
        util = inventory.warehouse_utilisation(inventory_snapshots)
        assert (util["utilisation_pct"] >= 0).all()
        assert set(util["status"]).issubset({"Under-utilised", "Healthy", "Constrained"})

    def test_kpis_are_finite(self, inventory_snapshots):
        kpis = inventory.inventory_kpis(inventory_snapshots)
        for key, value in kpis.items():
            if isinstance(value, float):
                assert pd.notna(value), f"{key} is NaN"


class TestShipments:
    def test_on_time_percentages_are_bounded(self, shipments):
        suppliers = shp.supplier_performance(shipments)
        assert suppliers["on_time_pct"].between(0, 100).all()
        assert shp.regional_performance(shipments)["on_time_pct"].between(0, 100).all()
        assert shp.carrier_scorecard(shipments)["on_time_pct"].between(0, 100).all()

    def test_supplier_ranking_is_worst_first(self, shipments):
        """The scorecard exists to surface problems, so the worst must be row one."""
        suppliers = shp.supplier_performance(shipments)
        assert suppliers["on_time_pct"].is_monotonic_increasing

    def test_late_analysis_returns_expected_keys(self, shipments):
        result = shp.late_shipment_analysis(shipments)
        assert set(result) == {"by_mode", "by_region", "by_leg",
                               "monthly_trend", "cost_impact"}
        for frame in result.values():
            assert isinstance(frame, pd.DataFrame)
            assert len(frame) > 0

    def test_network_on_time_matches_raw_data(self, shipments):
        kpis = shp.shipment_kpis(shipments)
        expected = 100 * (1 - shipments["is_late"].mean())
        assert kpis["on_time_delivery_pct"] == pytest.approx(expected, abs=0.01)
        assert kpis["total_shipments"] == len(shipments)
        assert kpis["late_shipments"] == int(shipments["is_late"].sum())

    def test_transit_variance_is_actual_minus_planned(self, shipments):
        transit = shp.transit_time_distribution(shipments)
        computed = transit["actual_mean_days"] - transit["planned_mean_days"]
        assert (computed - transit["mean_variance_days"]).abs().max() < 0.05

    def test_supplier_counts_reconcile_with_source(self, shipments):
        suppliers = shp.supplier_performance(shipments)
        assert suppliers["shipments"].sum() == len(shipments)

    def test_weakest_region_is_middle_east_africa(self, shipments):
        """A designed structural signal: the region with the longest road and sea
        legs should be the worst on-time performer."""
        regional = shp.regional_performance(shipments)
        assert regional.iloc[0]["region"] == "Middle East & Africa"


class TestForecasting:
    def test_forecast_dates_are_in_the_future(self, demand):
        history = forecasting.aggregate_demand(demand)
        forecast = forecasting.forecast_demand(demand, horizon=6)
        assert forecast["date"].min() > history["date"].max()
        assert len(forecast) == 6

    @pytest.mark.parametrize("method", ["holt_winters", "linear_trend", "moving_average"])
    def test_all_methods_produce_usable_output(self, demand, method):
        forecast = forecasting.forecast_demand(demand, horizon=4, method=method)
        assert len(forecast) == 4
        assert forecast["forecast"].notna().all()
        assert (forecast["forecast"] > 0).all()
        # Intervals must bracket the point forecast.
        assert (forecast["lower_ci"] <= forecast["forecast"]).all()
        assert (forecast["upper_ci"] >= forecast["forecast"]).all()

    def test_backtest_ranks_all_methods(self, demand, cfg):
        backtest = forecasting.backtest_forecast(demand)
        assert len(backtest) == len(cfg.forecasting.methods)
        assert backtest["rank"].tolist() == sorted(backtest["rank"])
        assert backtest["mape_pct"].is_monotonic_increasing, "rank must follow MAPE"

    def test_decomposition_reconstructs_the_series(self, demand):
        history = forecasting.aggregate_demand(demand)
        decomposition = forecasting.decompose_demand(history)
        assert len(decomposition) == len(history)
        interior = decomposition.dropna()
        rebuilt = interior["trend"] + interior["seasonal"] + interior["residual"]
        assert (rebuilt - interior["observed"]).abs().max() < 1.0

    def test_demand_kpis_reconcile(self, demand):
        kpis = forecasting.demand_kpis(demand)
        assert kpis["total_demand_units"] == demand["units_demanded"].sum()
        assert 0 <= kpis["avg_fill_rate_pct"] <= 100


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


class TestSimulation:
    def test_baseline_scenario_matches_measured_baseline(self):
        """With levers at their defaults, scenario must equal baseline."""
        result = simulation.simulate(simulation.default_levers())
        for name, entry in result["kpis"].items():
            if isinstance(entry, dict):
                assert entry["scenario"] == pytest.approx(entry["baseline"], rel=0.02), (
                    f"{name} moved with levers at default")

    def test_worse_conditions_increase_cost(self):
        base = simulation.simulate(simulation.default_levers())
        stressed = simulation.simulate({
            "supplier_reliability_pct": 72, "storage_temp_c": 38,
            "storage_humidity_pct": 85, "warehouse_delay_days": 80,
            "demand_change_pct": 40,
        })
        assert (stressed["financials"]["total_cost_usd"]["scenario"]
                > base["financials"]["total_cost_usd"]["scenario"])
        assert len(stressed["alerts"]) > 0

    def test_lower_reliability_reduces_qa_pass_rate(self):
        high = simulation.simulate({"supplier_reliability_pct": 98})
        low = simulation.simulate({"supplier_reliability_pct": 74})
        assert (low["kpis"]["qa_pass_rate_pct"]["scenario"]
                < high["kpis"]["qa_pass_rate_pct"]["scenario"])

    def test_higher_temperature_reduces_potency(self):
        cool = simulation.simulate({"storage_temp_c": 8})
        hot = simulation.simulate({"storage_temp_c": 38})
        assert (hot["kpis"]["avg_potency_pct"]["scenario"]
                < cool["kpis"]["avg_potency_pct"]["scenario"])

    def test_tornado_is_sorted_by_absolute_swing(self):
        tornado = simulation.tornado_analysis(kpi="total_cost_usd")
        swings = tornado["swing"].abs().tolist()
        assert swings == sorted(swings, reverse=True)

    def test_presets_all_simulate(self):
        comparison = simulation.compare_scenarios(simulation.preset_scenarios())
        assert len(comparison) == len(simulation.preset_scenarios())
        assert comparison["total_cost_usd"].notna().all()
