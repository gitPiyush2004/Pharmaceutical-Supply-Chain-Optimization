"""
Tests for configuration and the data access layer.

These are the invariants everything else depends on: if configuration is
malformed, or a dataset does not load with the shape the rest of the code expects,
no downstream metric can be trusted.

There is no generation or cleaning to test any more. Both are gone along with the
simulation they existed to serve, so what remains are tests that both real
datasets load correctly and that nothing silently mutates them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import resolve_path
from src.data import loader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TestConfiguration:
    def test_config_loads_and_has_required_sections(self, cfg):
        for section in ("project", "paths", "datasets", "data_quality", "ml",
                        "scms", "ab_testing", "economics", "viz", "logging"):
            assert section in cfg, f"missing config section: {section}"

    def test_no_simulation_sections_remain(self, cfg):
        """The simulated blocks must be gone, not merely unused.

        A dead ``generation:`` block left in the file is an invitation to
        re-import the simulation by accident, and it makes the configuration lie
        about what the project does.
        """
        for section in ("generation", "funnel", "inventory", "stability",
                        "forecasting", "simulation"):
            assert section not in cfg, f"simulated config section survived: {section}"

    def test_dot_access_works(self, cfg):
        assert isinstance(cfg.project.random_seed, int)
        assert cfg.ml.drug_classification.target == "Drug"

    def test_quality_weights_sum_to_one(self, cfg):
        """A weighted score is only interpretable if the weights are normalised."""
        assert sum(cfg.data_quality.weights.values()) == pytest.approx(1.0)

    def test_datasets_are_the_two_real_ones(self, cfg):
        assert set(cfg.datasets) == {"drug200", "scms"}
        assert set(loader.DATASETS) == set(cfg.datasets)

    def test_product_thresholds_are_configured(self, cfg):
        """The pricing analysis reads its noise floors from config, not code."""
        assert cfg.scms.products.min_shipments_for_spread >= 2
        assert cfg.scms.products.min_sites_for_spread >= 2
        assert cfg.scms.products.top_n_products > 0

    def test_statistical_thresholds_are_configured(self, cfg):
        """Test-selection rules must live in config, not in a magic number."""
        assert 0 < cfg.ab_testing.alpha < 0.2
        assert 0.5 < cfg.ab_testing.power < 1.0
        assert cfg.ab_testing.skew_limit > 0
        assert cfg.ab_testing.min_group_size >= 2

    def test_paths_resolve_to_absolute(self, cfg):
        assert resolve_path(cfg.paths.data_raw).is_absolute()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
class TestDatasetLoading:
    def test_every_dataset_loads_non_empty(self):
        for name in loader.DATASETS:
            frame = loader.load_table(name)
            assert isinstance(frame, pd.DataFrame)
            assert len(frame) > 0, f"{name} loaded empty"

    def test_expected_row_counts(self, clinical, scms_raw):
        """Pin the published sizes so a truncated download fails loudly.

        A partial CSV would otherwise sail through every other test while putting
        quietly wrong numbers on every page.
        """
        assert len(clinical) == 200
        assert len(scms_raw) == 10_324

    def test_loaders_return_copies(self):
        """Mutating a returned frame must not corrupt the module-level cache."""
        first = loader.load_scms()
        first["is_late"] = -999
        second = loader.load_scms()
        assert (second["is_late"] != -999).all()

    def test_unknown_dataset_raises_helpfully(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            loader.load_table("batches")

    def test_load_all_returns_both(self):
        tables = loader.load_all()
        assert set(tables) == {"clinical", "scms"}
        assert all(len(frame) > 0 for frame in tables.values())


# ---------------------------------------------------------------------------
# The clinical dataset is used exactly as published
# ---------------------------------------------------------------------------
class TestClinicalDataset:
    def test_is_published_clean(self, clinical):
        """drug200 needs no cleaning, and the project says so. Verify the claim."""
        assert clinical.isna().sum().sum() == 0
        assert clinical.duplicated().sum() == 0
        assert clinical["Age"].between(0, 120).all()
        assert clinical["Na_to_K"].gt(0).all()
        assert set(clinical["Sex"]) == {"M", "F"}
        assert set(clinical["Drug"]) == {"DrugY", "drugA", "drugB", "drugC", "drugX"}

    def test_label_is_a_deterministic_function_of_features(self, clinical):
        """The documented determinism, asserted rather than assumed.

        The ML page and the notebook both claim that 100% accuracy is *attainable*
        on this dataset because the label is a pure function of the features. If
        that ever stopped being true those claims would become wrong, so the
        property is tested rather than trusted.
        """
        high = clinical[clinical["Na_to_K"] >= 15.015]
        low = clinical[clinical["Na_to_K"] < 15.015]
        assert set(high["Drug"]) == {"DrugY"}, "Na/K >= 15.015 must give DrugY"
        assert "DrugY" not in set(low["Drug"]), "DrugY must not appear below it"

        # No two rows share a feature vector while disagreeing on the label.
        features = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
        assert clinical.groupby(features)["Drug"].nunique().max() == 1
