"""
Tests for the ML pipeline and the data quality module.

The ML tests load the persisted artefacts rather than retraining, so the suite
stays fast. Retraining is exercised separately by ``scripts/train_models.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml import predict, preprocess
from src.quality import assessment as dq

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_clean_clinical_normalises_categoricals(self, clinical):
        cleaned = preprocess.clean_clinical(clinical)
        assert set(cleaned["Sex"].unique()).issubset({"F", "M"})
        assert set(cleaned["BP"].unique()).issubset({"HIGH", "NORMAL", "LOW"})
        assert set(cleaned["Cholesterol"].unique()).issubset({"HIGH", "NORMAL"})
        assert cleaned.notna().all().all()

    def test_clinical_feature_engineering_adds_expected_columns(self, clinical):
        engineered = preprocess.engineer_clinical_features(
            preprocess.clean_clinical(clinical))
        for column in ("bp_risk_score", "cholesterol_risk", "combined_risk_score",
                       "high_na_to_k"):
            assert column in engineered.columns
        # Risk scores must be ordinal and bounded, not arbitrary.
        assert engineered["bp_risk_score"].between(0, 3).all()
        assert engineered["high_na_to_k"].isin([0, 1]).all()

    def test_batch_feature_engineering_creates_thermal_load(self, batches):
        engineered = preprocess.engineer_batch_features(batches)
        assert "thermal_load" in engineered.columns
        # Thermal load is excess temperature times exposure, so it cannot be
        # negative and must be zero when there is no excess.
        assert (engineered["thermal_load"] >= 0).all()

    def test_split_is_stratified_and_reproducible(self, clinical, cfg):
        engineered = preprocess.engineer_clinical_features(
            preprocess.clean_clinical(clinical))
        numeric, categorical = preprocess.clinical_feature_columns()
        features = engineered[numeric + categorical]
        target = engineered["Drug"]

        test_size = cfg.ml.drug_classification.test_size
        seed = cfg.project.random_seed
        first = preprocess.split_data(features, target, test_size, seed)
        second = preprocess.split_data(features, target, test_size, seed)
        pd.testing.assert_frame_equal(first[0], second[0])

        train_share = first[2].value_counts(normalize=True)
        full_share = target.value_counts(normalize=True)
        for label in full_share.index:
            assert train_share[label] == pytest.approx(full_share[label], abs=0.06)

    def test_normalise_categorical_rejects_unknown_values(self):
        assert preprocess.normalise_categorical("  high ", ["HIGH", "LOW"], "BP") == "HIGH"
        with pytest.raises(ValueError):
            preprocess.normalise_categorical("MEDIUM", ["HIGH", "LOW"], "BP")


# ---------------------------------------------------------------------------
# Trained models
# ---------------------------------------------------------------------------
class TestTrainedModels:
    def test_artefacts_exist(self, models_available):
        assert models_available, (
            "run `python scripts/train_models.py` before the test suite")

    def test_metadata_is_complete(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        for name in ("drug_classification", "batch_risk"):
            meta = predict.model_summary(name)
            for key in ("model_name", "metrics", "features", "class_labels",
                        "confusion_matrix", "feature_importance", "per_class_roc",
                        "random_seed", "trained_at"):
                assert key in meta, f"{name} metadata missing {key}"

    def test_drug_model_meets_accuracy_floor(self, models_available):
        """drug200 has a near-deterministic decision rule, so anything below 0.9
        indicates a broken pipeline rather than a hard problem."""
        if not models_available:
            pytest.skip("models not trained")
        metrics = predict.model_summary("drug_classification")["metrics"]
        assert metrics["accuracy"] >= 0.90
        assert metrics["f1_macro"] >= 0.85

    def test_batch_risk_model_beats_majority_baseline(self, models_available, batches):
        if not models_available:
            pytest.skip("models not trained")
        metrics = predict.model_summary("batch_risk")["metrics"]
        majority = batches["batch_risk_label"].value_counts(normalize=True).max()
        assert metrics["accuracy"] > majority, (
            f"accuracy {metrics['accuracy']:.3f} does not beat the "
            f"majority-class baseline {majority:.3f}")

    def test_confusion_matrix_totals_match_test_rows(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        for name in ("drug_classification", "batch_risk"):
            meta = predict.model_summary(name)
            total = sum(sum(row) for row in meta["confusion_matrix"]["values"])
            assert total == meta["test_rows"]

    def test_feature_importances_are_normalised(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        for name in ("drug_classification", "batch_risk"):
            importance = pd.DataFrame(predict.model_summary(name)["feature_importance"])
            assert (importance["importance"] >= 0).all()
            assert importance["importance"].sum() == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Prediction interface
# ---------------------------------------------------------------------------
class TestPrediction:
    @pytest.mark.parametrize("age,sex,bp,chol,na_to_k,expected", [
        (23, "F", "HIGH", "HIGH", 25.355, "DrugY"),   # high Na/K dominates
        (47, "M", "LOW", "HIGH", 13.093, "drugC"),
        (28, "F", "NORMAL", "HIGH", 7.798, "drugX"),
        (43, "M", "HIGH", "HIGH", 13.972, "drugA"),
        (74, "M", "HIGH", "HIGH", 9.567, "drugB"),
    ])
    def test_known_clinical_rules_are_reproduced(self, models_available, age, sex,
                                                 bp, chol, na_to_k, expected):
        """Rows taken directly from drug200 - the model must recover the
        documented prescribing logic."""
        if not models_available:
            pytest.skip("models not trained")
        result = predict.predict_drug(age=age, sex=sex, bp=bp,
                                      cholesterol=chol, na_to_k=na_to_k)
        assert result["prediction"] == expected

    def test_prediction_output_contract(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        result = predict.predict_drug(age=45, sex="F", bp="HIGH",
                                      cholesterol="HIGH", na_to_k=15.4)
        assert set(result) >= {"prediction", "confidence", "probabilities",
                              "explanation", "model"}
        assert 0 <= result["confidence"] <= 1
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        assert len(result["explanation"]) > 40

    def test_invalid_input_is_rejected(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        with pytest.raises((ValueError, KeyError)):
            predict.predict_drug(age=45, sex="F", bp="EXTREME",
                                 cholesterol="HIGH", na_to_k=15.4)
        with pytest.raises(ValueError):
            predict.predict_drug(age=-5, sex="F", bp="HIGH",
                                 cholesterol="HIGH", na_to_k=15.4)

    def test_cohort_scoring_preserves_row_order(self, models_available, clinical):
        if not models_available:
            pytest.skip("models not trained")
        sample = clinical.head(20).copy()
        sample["patient_id"] = range(20)
        scored = predict.predict_drug_batch(sample)
        assert len(scored) == 20
        assert scored["patient_id"].tolist() == list(range(20))
        assert scored["prediction"].notna().all()

    def test_batch_risk_discriminates_good_from_bad_conditions(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        bad = predict.predict_batch_risk(
            storage_temp_c=36.0, storage_humidity_pct=88.0, storage_duration_days=200,
            total_cycle_time_days=190, qa_delay_days=40, supplier_reliability=0.75,
            is_cold_chain=1, shelf_life_months=15, drug_code="DrugC",
            region="Middle East & Africa")
        good = predict.predict_batch_risk(
            storage_temp_c=5.0, storage_humidity_pct=42.0, storage_duration_days=20,
            total_cycle_time_days=60, qa_delay_days=14, supplier_reliability=0.97,
            is_cold_chain=1, shelf_life_months=36, drug_code="DrugY",
            region="Europe")
        risk_order = {"Low": 0, "Medium": 1, "High": 2}
        assert risk_order[bad["prediction"]] > risk_order[good["prediction"]]


# ---------------------------------------------------------------------------
# Data quality module
# ---------------------------------------------------------------------------
class TestDataQuality:
    def test_missing_report_matches_pandas(self, raw_batches):
        report = dq.missing_value_report(raw_batches)
        expected = int(raw_batches.isna().sum().sum())
        assert int(report["missing_count"].sum()) == expected

    def test_quality_module_detects_injected_defects(self, raw_batches):
        """The audit must find the defects the generator recorded injecting."""
        report = dq.assess_dataset("batches", raw=True)

        missing = report["missing"]
        flagged = set(missing[missing["missing_count"] > 0]["column"])
        assert {"storage_temp_c", "storage_humidity_pct",
                "supplier_reliability"}.issubset(flagged)

        validity = report["validity"]
        breached = set(validity[validity["violations"] > 0]["column"])
        assert {"potency_pct", "storage_duration_days"}.issubset(breached)

        assert report["duplicates"]["key_duplicate_rows"] > 0

    def test_scores_are_bounded_and_graded(self):
        board = dq.assess_all(raw=True)
        for column in ("completeness", "uniqueness", "validity",
                       "consistency", "accuracy", "overall_score"):
            assert board[column].between(0, 100).all()
        assert board["grade"].isin(["A", "B", "C", "D", "F"]).all()

    def test_scoreboard_is_sorted_worst_first(self):
        board = dq.assess_all(raw=True)
        assert board["overall_score"].is_monotonic_increasing

    def test_cleaning_improves_every_scored_table(self):
        """The uplift must be non-negative everywhere and strictly positive where
        defects were injected."""
        uplift = dq.quality_uplift()
        assert (uplift["uplift"] >= -0.01).all(), "cleaning made a table worse"
        assert uplift.loc[uplift["dataset"] == "batches", "uplift"].iloc[0] > 0

    def test_clinical_dataset_is_never_modified(self):
        """drug200 is real Kaggle data; the pipeline must leave it untouched."""
        bronze = dq.assess_dataset("drug200", raw=True)
        silver = dq.assess_dataset("drug200", raw=False)
        assert bronze["rows"] == silver["rows"] == 200
        assert bronze["score"]["overall_score"] == silver["score"]["overall_score"]

    def test_outliers_are_reported_not_removed(self, raw_batches, batches):
        """Extreme values are signal in stability analytics; only impossible
        values may be touched."""
        outliers = dq.outlier_report(raw_batches)
        assert outliers["outlier_count"].sum() > 0
        # Genuine excursions must survive cleaning.
        assert batches["storage_temp_c"].max() > 30

    def test_recommendations_are_actionable(self, raw_batches):
        recommendations = dq.preprocessing_recommendations(raw_batches)
        assert len(recommendations) > 0
        assert set(["issue", "column", "severity", "recommended_action"]).issubset(
            recommendations.columns)
        assert recommendations["severity"].isin(["High", "Medium", "Low"]).all()
