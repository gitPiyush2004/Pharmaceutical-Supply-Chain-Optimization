"""
Tests for configuration, generation determinism and the cleaning layer.

These are the invariants everything else depends on: if generation is not
reproducible or cleaning silently corrupts the data, no downstream metric can be
trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import get_config, resolve_path
from src.data import loader
from src.data.cleaning import clean_table
from src.data.generator import build_dimensions, generate_all


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TestConfiguration:
    def test_config_loads_and_has_required_sections(self, cfg):
        for section in ("project", "paths", "datasets", "generation", "funnel",
                        "data_quality", "ml", "inventory", "shipments",
                        "forecasting", "stability", "ab_testing", "simulation",
                        "economics", "viz"):
            assert section in cfg, f"missing config section: {section}"

    def test_dot_access_works(self, cfg):
        assert isinstance(cfg.project.random_seed, int)
        assert cfg.ml.drug_classification.target == "Drug"

    def test_quality_weights_sum_to_one(self, cfg):
        """A weighted score is only interpretable if the weights are normalised."""
        assert sum(cfg.data_quality.weights.values()) == pytest.approx(1.0)

    def test_funnel_has_eight_ordered_stages(self, cfg):
        assert len(cfg.funnel.stages) == 8
        # Every stage must map to both a unit column and a date column.
        for stage in cfg.funnel.stages:
            assert stage in cfg.funnel.unit_columns
            assert stage in cfg.funnel.date_columns

    def test_paths_resolve_to_absolute(self, cfg):
        assert resolve_path(cfg.paths.data_raw).is_absolute()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class TestGeneration:
    def test_generation_is_deterministic(self):
        """Same seed must yield byte-identical data - the whole platform's
        reproducibility claim rests on this."""
        first = generate_all(save=False)["batches"]
        second = generate_all(save=False)["batches"]
        pd.testing.assert_frame_equal(first, second)

    def test_product_mix_derived_from_clinical_data(self, clinical):
        """The supply chain product mix must track drug200 prescription shares."""
        drugs = build_dimensions()["drugs"]
        assert drugs["volume_share"].sum() == pytest.approx(1.0)

        clinical_share = clinical["Drug"].value_counts(normalize=True)
        top_clinical = clinical_share.idxmax().lower()
        top_supply = drugs.loc[drugs["volume_share"].idxmax(), "drug_code"].lower()
        assert top_clinical == top_supply, (
            "highest-volume product should match the most prescribed drug")

    def test_funnel_volumes_are_monotonically_non_increasing(self, batches, cfg):
        """Units can never increase moving downstream - that would be creating
        product out of nothing."""
        columns = [cfg.funnel.unit_columns[stage] for stage in cfg.funnel.stages]
        totals = [batches[column].sum() for column in columns]
        for earlier, later, stage in zip(totals, totals[1:], cfg.funnel.stages[1:]):
            assert later <= earlier, f"units increased entering {stage}"

    def test_stage_dates_are_ordered(self, batches, cfg):
        """Each stage must complete no earlier than the stage before it."""
        columns = [cfg.funnel.date_columns[stage] for stage in cfg.funnel.stages]
        for earlier, later in zip(columns, columns[1:]):
            violations = (batches[later] < batches[earlier]).sum()
            assert violations == 0, f"{violations} rows where {later} precedes {earlier}"

    def test_calibration_within_realistic_bounds(self, batches, shipments):
        """Generated metrics must stay in ranges a pharma operation could plausibly
        report; a regression in the generator usually shows up here first."""
        procured = batches["units_procured"].sum()
        yield_pct = 100 * batches["units_dispensed"].sum() / procured
        assert 50 <= yield_pct <= 80, f"end-to-end yield {yield_pct:.1f}% implausible"

        qa_rate = 100 * batches["qa_pass"].mean()
        assert 85 <= qa_rate <= 96, f"QA pass rate {qa_rate:.1f}% implausible"

        on_time = 100 * (1 - shipments["is_late"].mean())
        assert 80 <= on_time <= 95, f"on-time rate {on_time:.1f}% implausible"

        potency = batches["potency_pct"].mean()
        assert 94 <= potency <= 99, f"mean potency {potency:.2f}% implausible"

    def test_supplier_reliability_drives_qa_outcomes(self, batches):
        """The intended structural signal: weaker suppliers must fail QA more."""
        by_supplier = batches.groupby("supplier_name").agg(
            reliability=("supplier_reliability", "mean"),
            qa_pass=("qa_pass", "mean"))
        correlation = by_supplier["reliability"].corr(by_supplier["qa_pass"])
        assert correlation > 0.7, (
            f"supplier reliability should predict QA pass rate (r={correlation:.2f})")

    def test_cold_chain_excursions_reduce_potency(self, batches):
        """Excursions must have a material, detectable effect."""
        cold = batches[batches["is_cold_chain"] == 1]
        with_excursion = cold[cold["cold_chain_excursion"] == 1]["potency_pct"]
        without = cold[cold["cold_chain_excursion"] == 0]["potency_pct"]
        assert len(with_excursion) > 10, "too few excursion batches to analyse"
        assert with_excursion.mean() < without.mean() - 2.0, (
            "excursions should cost at least 2 percentage points of potency")

    def test_risk_labels_are_reasonably_balanced(self, batches):
        """All three tiers must be well represented or the classifier is
        learning a degenerate problem."""
        shares = batches["batch_risk_label"].value_counts(normalize=True)
        assert set(shares.index) == {"Low", "Medium", "High"}
        assert shares.min() > 0.10, f"risk tiers too imbalanced: {shares.to_dict()}"

    def test_all_tables_written(self, cfg):
        for name in ("drugs", "suppliers", "warehouses", "batches",
                     "shipments", "demand", "inventory"):
            assert resolve_path(cfg.datasets[name]).exists()


# ---------------------------------------------------------------------------
# Cleaning (bronze -> silver)
# ---------------------------------------------------------------------------
class TestCleaning:
    def test_raw_layer_actually_contains_defects(self, raw_batches):
        """If the bronze layer is clean, the quality module is testing nothing."""
        assert raw_batches["storage_temp_c"].isna().sum() > 0
        assert raw_batches["batch_id"].duplicated().sum() > 0
        assert (raw_batches["potency_pct"] > 100).sum() > 0
        assert (raw_batches["storage_duration_days"] < 0).sum() > 0
        assert raw_batches["region"].nunique() > 5, "expected dirty region spellings"

    def test_cleaning_removes_every_defect_class(self, batches):
        assert batches["storage_temp_c"].isna().sum() == 0
        assert batches["storage_humidity_pct"].isna().sum() == 0
        assert batches["supplier_reliability"].isna().sum() == 0
        assert batches["batch_id"].duplicated().sum() == 0
        assert (batches["potency_pct"] > 100).sum() == 0
        assert (batches["storage_duration_days"] < 0).sum() == 0

    def test_regions_canonicalised_to_five(self, batches):
        assert batches["region"].nunique() == 5
        assert not batches["region"].str.contains(r"^\s|\s$", regex=True).any()

    def test_grouped_imputation_preserves_cold_chain_semantics(self, batches):
        """The important cleaning subtlety: imputing a missing cold-chain
        temperature with the global median would fabricate a 20 degC excursion."""
        cold_mean = batches.loc[batches["is_cold_chain"] == 1, "storage_temp_c"].mean()
        ambient_mean = batches.loc[batches["is_cold_chain"] == 0, "storage_temp_c"].mean()
        assert cold_mean < 12, f"cold chain mean {cold_mean:.1f} degC is too warm"
        assert ambient_mean > 18, f"ambient mean {ambient_mean:.1f} degC is too cool"

    def test_master_data_repair_restores_true_values(self, raw_batches):
        """supplier_reliability is knowable from the dimension, so it should be
        looked up rather than imputed."""
        suppliers = loader.load_table("suppliers", raw=True).set_index("supplier_id")
        cleaned, log = clean_table("batches", raw_batches)
        assert "master_data_repair" in log["step"].values

        sample = cleaned.head(200)
        expected = sample["supplier_id"].map(suppliers["reliability"])
        assert np.allclose(sample["supplier_reliability"], expected)

    def test_cleaning_does_not_drop_legitimate_rows(self, raw_batches, batches):
        """Only duplicates should disappear; extreme-but-real values must stay."""
        duplicates = raw_batches["batch_id"].duplicated().sum()
        assert len(batches) == len(raw_batches) - duplicates

    def test_remediation_log_is_populated(self):
        log = loader.remediation_log()
        assert len(log) > 0
        assert set(["table", "step", "column", "rows_affected"]).issubset(log.columns)
        assert log["rows_affected"].sum() > 0

    def test_cleaning_is_idempotent(self, batches):
        """Cleaning already-clean data must be a no-op."""
        twice, log = clean_table("batches", batches)
        assert len(twice) == len(batches)
        material = log[~log["step"].isin(["whitespace"])]
        assert material["rows_affected"].sum() == 0, (
            f"second pass still changed data: {material.to_dict('records')}")
