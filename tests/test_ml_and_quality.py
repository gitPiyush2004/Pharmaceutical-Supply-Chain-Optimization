"""
Tests for the ML pipeline and the data quality module.

The ML tests load the persisted artefacts rather than retraining, so the suite
stays fast. Retraining is exercised separately by ``scripts/train_models.py``.

The data quality tests are unusual in one respect worth flagging: several of them
assert that a score is *low* or that parsing makes a score *worse*. That is
deliberate. The module's value is in describing data honestly, and a test suite
that only ever demanded higher numbers would have blessed the very behaviour this
project argues against.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml import predict, preprocess
from src.quality import assessment as dq

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

MODELS = ("drug_classification", "late_delivery")


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

    def test_no_batch_feature_engineering_survives(self):
        """The simulated feature set must be gone, not merely unused."""
        for gone in ("engineer_batch_features", "batch_feature_columns",
                     "BATCH_ENGINEERED_NUMERIC"):
            assert not hasattr(preprocess, gone), f"{gone} survived the removal"


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
        for name in MODELS:
            meta = predict.model_summary(name)
            for key in ("model_name", "metrics", "features", "class_labels",
                        "confusion_matrix", "feature_importance", "per_class_roc",
                        "random_seed", "trained_at"):
                assert key in meta, f"{name} metadata missing {key}"

    def test_batch_risk_model_is_gone(self):
        """A third model trained on simulated telemetry used to live here."""
        with pytest.raises(ValueError, match="Unknown model"):
            predict.model_summary("batch_risk")

    def test_drug_model_meets_accuracy_floor(self, models_available):
        """The label is a pure function of the features on drug200, so anything
        below 0.9 indicates a broken pipeline rather than a hard problem."""
        if not models_available:
            pytest.skip("models not trained")
        metrics = predict.model_summary("drug_classification")["metrics"]
        assert metrics["accuracy"] >= 0.90
        assert metrics["f1_macro"] >= 0.85

    def test_late_delivery_model_ranks_even_though_accuracy_is_poor(
            self, models_available, scms):
        """The central honest finding about this model, asserted.

        Its accuracy sits at or below the majority-class baseline, because only
        11.5% of shipments are late. That is expected and is exactly why accuracy
        is the wrong metric here - so the test demands discrimination (ROC AUC)
        rather than accuracy, and pins the accuracy result so nobody later
        "improves" the model by optimising the wrong thing.
        """
        if not models_available:
            pytest.skip("models not trained")
        meta = predict.model_summary("late_delivery")
        majority = float((scms["is_late"] == 0).mean())
        assert meta["metrics"]["accuracy"] <= majority + 0.01
        assert meta["roc_auc_ovr"] >= 0.75, "the model must still rank risk well"

    def test_gains_curve_beats_random_targeting(self, models_available):
        """The metric this model is actually deployed on."""
        if not models_available:
            pytest.skip("models not trained")
        gains = predict.late_delivery_targeting_curve()
        assert gains["targeted_pct"].is_monotonic_increasing
        assert gains["capture_rate_pct"].is_monotonic_increasing
        at20 = gains[gains["targeted_pct"] == 20.0].iloc[0]
        # Capturing 20% of late shipments from 20% of the queue would be random.
        assert at20["capture_rate_pct"] > 20.0
        assert at20["lift"] > 1.0

    def test_confusion_matrix_totals_match_test_rows(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        for name in MODELS:
            meta = predict.model_summary(name)
            total = sum(sum(row) for row in meta["confusion_matrix"]["values"])
            assert total == meta["test_rows"]

    def test_feature_importances_are_normalised(self, models_available):
        if not models_available:
            pytest.skip("models not trained")
        for name in MODELS:
            importance = pd.DataFrame(predict.model_summary(name)["feature_importance"])
            assert (importance["importance"] >= 0).all()
            assert importance["importance"].sum() == pytest.approx(1.0, abs=0.02)

    def test_late_delivery_features_exclude_leaky_columns(self, models_available):
        """Leakage control, asserted rather than trusted.

        Every lead-time measure except the *scheduled* one is derived from the
        delivery date, so including one would leak the answer. Vendor identity is
        excluded too: with 73 vendors, several appearing a handful of times, the
        model would memorise suppliers instead of learning transferable structure.
        """
        if not models_available:
            pytest.skip("models not trained")
        features = set(predict.model_summary("late_delivery")["features"])
        for leaky in ("delivery_delay_days", "is_late", "total_lead_time_days",
                      "date_delivered", "vendor_lead_time_days", "vendor",
                      "manufacturing_site"):
            assert leaky not in features, f"leaky feature present: {leaky}"
        assert "scheduled_lead_time_days" in features


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

    def test_late_delivery_separates_a_good_lane_from_a_bad_one(self, models_available):
        """Two real shipment profiles the data says differ, scored in order."""
        if not models_available:
            pytest.skip("models not trained")
        risky = predict.predict_late_delivery(
            quantity=5_000, line_value_usd=40_000.0, unit_price_usd=8.0,
            units_per_pack=60, weight_kg=2_500.0, freight_cost_usd=9_000.0,
            scheduled_lead_time_days=40, shipment_mode="Ocean",
            product_group="ARV", sub_classification="Adult",
            region="West & Central Africa", fulfil_via="From RDC",
            inco_term="N/A - From RDC", managed_by="PMO - US",
            first_line_designation="Yes")
        safe = predict.predict_late_delivery(
            quantity=5_000, line_value_usd=40_000.0, unit_price_usd=8.0,
            units_per_pack=60, weight_kg=400.0, freight_cost_usd=4_000.0,
            scheduled_lead_time_days=200, shipment_mode="Air",
            product_group="HRDT", sub_classification="HIV test",
            region="Asia", fulfil_via="Direct Drop", inco_term="DDP",
            managed_by="PMO - US", first_line_designation="Yes")
        assert risky["late_probability"] > safe["late_probability"]


# ---------------------------------------------------------------------------
# Data quality module
# ---------------------------------------------------------------------------
class TestDataQuality:
    def test_missing_report_matches_pandas(self, scms_raw):
        report = dq.missing_value_report(scms_raw)
        assert int(report["missing_count"].sum()) == int(scms_raw.isna().sum().sum())

    def test_scores_are_bounded_and_graded(self):
        board = dq.assess_all(raw=True)
        for column in ("completeness", "uniqueness", "validity",
                       "consistency", "accuracy", "overall_score"):
            assert board[column].between(0, 100).all()
        assert board["grade"].isin(["A", "B", "C", "D", "F"]).all()

    def test_scoreboard_is_sorted_worst_first(self):
        board = dq.assess_all(raw=True)
        assert board["overall_score"].is_monotonic_increasing

    def test_parsing_scms_lowers_its_generic_score(self):
        """The finding the Data Quality page is built around.

        Parsing replaces unparseable strings with honest nulls, so completeness
        falls. The generic score therefore *rewards* the file for holding non-null
        garbage. This is asserted rather than merely narrated, because a future
        change that made the uplift positive would mean parsing had started
        imputing instead of nulling - a silent and serious regression.
        """
        uplift = dq.quality_uplift().set_index("dataset")
        assert uplift.loc["scms", "uplift"] < 0
        assert uplift.loc["scms", "completeness_delta"] < 0

    def test_parsing_indian_medicines_raises_its_score(self):
        """The opposite case: here parsing genuinely recovers information."""
        uplift = dq.quality_uplift().set_index("dataset")
        assert uplift.loc["indian_medicines", "uplift"] > 0

    def test_clinical_dataset_is_never_modified(self):
        """drug200 is published clean; the pipeline must leave it untouched."""
        published = dq.assess_dataset("drug200", raw=True)
        interpreted = dq.assess_dataset("drug200", raw=False)
        assert published["rows"] == interpreted["rows"] == 200
        assert (published["score"]["overall_score"]
                == interpreted["score"]["overall_score"])

    def test_consistency_checks_only_run_on_comparable_types(self, scms, scms_raw):
        """Guards a bug that produced an 82% violation rate out of nothing.

        The raw SCMS dates are strings. Comparing two date-shaped strings with
        ``<=`` is a lexicographic comparison, which reported almost every row as
        inconsistent. Checks now require both columns to be datetimes or both to
        be numeric.
        """
        raw_checks = dq.consistency_report(scms_raw)
        assert len(raw_checks) == 0, "a string column was compared as if ordered"

        parsed_checks = dq.consistency_report(scms)
        assert len(parsed_checks) >= 3
        # Genuine source errors survive - a handful, not thousands.
        assert parsed_checks["violation_pct"].max() < 1.0

    def test_consistency_returns_empty_when_no_invariant_applies(self,
                                                                indian_medicines):
        """A flat catalogue has nothing to cross-check, and says so."""
        assert len(dq.consistency_report(indian_medicines)) == 0

    def test_container_columns_do_not_break_profiling(self, indian_medicines):
        """The parsed ingredient list is a Python list per row, which is unhashable."""
        assert indian_medicines["ingredients"].map(type).eq(list).any()
        report = dq.assess_dataset("indian_medicines", raw=False)
        assert report["duplicates"]["exact_duplicate_rows"] >= 0
        assert len(report["summary"]) == report["columns"]

    def test_outliers_are_reported_not_removed(self, scms):
        """Extreme values here are real records, and removing them would delete
        the finding that freight ratios need rank-based tests."""
        outliers = dq.outlier_report(scms)
        assert outliers["outlier_count"].sum() > 0
        assert scms["freight_pct_of_value"].max() > 1_000

    def test_recommendations_are_actionable(self, scms_raw):
        recommendations = dq.preprocessing_recommendations(scms_raw)
        assert len(recommendations) > 0
        assert {"issue", "column", "severity", "recommended_action"}.issubset(
            recommendations.columns)
        assert recommendations["severity"].isin(["High", "Medium", "Low"]).all()
