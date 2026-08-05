"""
Tests for the real USAID SCMS dataset layer.

These protect the parts most likely to break silently: the parsers that turn
business strings into numbers and dates, the distinction between structurally
absent and genuinely missing values, and the leakage controls on the model
trained from this data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics import procurement as proc
from src.data import scms as scms_mod


@pytest.fixture(scope="module")
def scms():
    """Cleaned SCMS delivery history."""
    return scms_mod.load_scms()


@pytest.fixture(scope="module")
def scms_raw():
    """SCMS exactly as published."""
    return scms_mod.load_scms_raw()


# ---------------------------------------------------------------------------
class TestLoading:
    def test_raw_shape_matches_published_dataset(self, scms_raw):
        assert len(scms_raw) == 10_324
        assert scms_raw.shape[1] == 33

    def test_clean_preserves_every_row(self, scms, scms_raw):
        """Cleaning must reshape values, never silently drop shipments."""
        assert len(scms) == len(scms_raw)

    def test_expected_scale(self, scms):
        assert scms["country"].nunique() == 43
        assert scms["vendor"].nunique() == 73
        assert scms["date_delivered"].min().year == 2006
        assert scms["date_delivered"].max().year == 2015

    def test_every_country_maps_to_a_region(self, scms):
        """An unmapped country would silently fall into 'Other' and distort
        every regional metric."""
        assert (scms["region"] == "Other").sum() == 0
        assert scms["region"].nunique() >= 5


# ---------------------------------------------------------------------------
class TestParsing:
    def test_delivery_dates_all_parse(self, scms):
        """The three delivery milestones are complete in the source file; if any
        fail to parse, the mixed-format handling has regressed."""
        for column in ("date_scheduled", "date_delivered", "date_recorded"):
            assert scms[column].notna().all(), f"{column} has unparsed values"

    def test_structural_absences_are_labelled_not_imputed(self, scms):
        """`N/A - From RDC` means no vendor PO existed. It must stay null and be
        marked structural - imputing it would invent a purchase order."""
        reason = scms["date_po_sent_reason"]
        structural = reason == "structural"
        assert structural.sum() > 5_000, "expected ~5,404 RDC-fulfilled line items"
        assert scms.loc[structural, "date_po_sent"].isna().all()

        # And those rows must be excluded from vendor lead time, not zero-filled.
        assert scms.loc[structural, "vendor_lead_time_days"].isna().all()

    def test_missing_is_distinguished_from_structural(self, scms):
        """`Date Not Captured` is a real gap; `Pre-PQ Process` is not. They must
        not collapse into one category."""
        reasons = set(scms["date_pq_sent_reason"].unique())
        assert {"parsed", "structural", "missing"}.issubset(reasons)

    def test_numeric_sentinels_are_classified(self, scms):
        """Freight and weight carry business statements and cross-references, and
        each needs its own reason code."""
        for field in ("freight_cost", "weight"):
            reasons = set(scms[f"{field}_reason"].unique())
            assert "parsed" in reasons
            assert "structural" in reasons, f"{field} lost its sentinel category"
            assert "cross_reference" in reasons, f"{field} lost cross-references"

    @pytest.mark.parametrize("value_column,reason_column", [
        ("freight_cost_usd", "freight_cost_reason"),
        ("weight_kg", "weight_reason"),
    ])
    def test_parsed_numerics_are_real_floats(self, scms, value_column, reason_column):
        """A nullable dtype here propagates pd.NA into numpy comparisons and
        breaks the derived economics."""
        assert scms[value_column].dtype == np.float64
        parsed = scms.loc[scms[reason_column] == "parsed", value_column]
        assert parsed.notna().all()
        assert (parsed >= 0).all()

    def test_parsing_report_accounts_for_every_row(self, scms):
        report = scms_mod.parsing_report(scms)
        assert len(report) > 0
        totals = report[["parsed", "structural", "missing",
                         "cross_reference", "unparseable"]].sum(axis=1)
        assert (totals == len(scms)).all(), "reason codes must partition the rows"

    def test_po_coverage_is_the_known_value(self, scms):
        """Documented finding: only ~44% of line items carry a vendor PO date."""
        coverage = 100 * scms["date_po_sent"].notna().mean()
        assert 43 <= coverage <= 46, f"PO coverage drifted to {coverage:.1f}%"


# ---------------------------------------------------------------------------
class TestDateParsingRegression:
    """Regression guards for the ambiguous-date bug (B1).

    The file uses `d-Mon-yy` for the three delivery milestones and US `m/d/yy` for
    the two paperwork milestones. Parsing with `dayfirst=True` silently read
    `5/3/13` as 3 May instead of 5 March, which corrupted every derived lead time.
    """

    def test_formats_are_pinned_per_column(self):
        """The formats must be declared, not inferred - inference is what broke."""
        from src.data.scms import DATE_FORMATS

        assert DATE_FORMATS["Scheduled Delivery Date"] == "%d-%b-%y"
        assert DATE_FORMATS["PQ First Sent to Client Date"] == "%m/%d/%y"
        assert DATE_FORMATS["PO Sent to Vendor Date"] == "%m/%d/%y"

    @pytest.mark.parametrize("column,limit", [
        ("total_lead_time_days", 25),
        ("vendor_lead_time_days", 25),
        ("scheduled_lead_time_days", 25),
    ])
    def test_negative_lead_times_are_rare(self, scms, column, limit):
        """A lead time cannot be negative except as a source-data entry error.

        Before the fix these were 478 / 432 / 457. A handful survive because the
        published file genuinely contains a few bad rows; the limit catches a
        regression in parsing without pretending the source is perfect.
        """
        negatives = int((scms[column] < 0).sum())
        assert negatives < limit, (
            f"{column} has {negatives} negative values - date parsing has regressed")

    def test_purchase_order_follows_its_own_quote(self, scms):
        """A PO cannot precede the quote that generated it. Was 1,128 before the fix."""
        violations = int((scms["date_po_sent"] < scms["date_pq_sent"]).sum())
        assert violations < 25, f"{violations} POs precede their own quote"

    def test_paperwork_dates_stay_inside_the_programme_window(self, scms):
        """A month/day swap pushes dates outside the real 2006-2015 span."""
        for column in ("date_pq_sent", "date_po_sent"):
            parsed = scms[column].dropna()
            assert parsed.min().year >= 2006, f"{column} predates the programme"
            assert parsed.max().year <= 2015, f"{column} postdates the data"


class TestUnitAccountingRegression:
    """Regression guard for the packs-versus-units bug (B2)."""

    def test_quantity_is_packs_not_units(self, scms):
        """`line_value_usd == pack_price_usd * quantity` is what proves it."""
        subset = scms[(scms["pack_price_usd"] > 0) & (scms["line_value_usd"] > 0)]
        implied = subset["pack_price_usd"] * subset["quantity"]
        agreement = (implied - subset["line_value_usd"]).abs() < 0.01
        assert agreement.mean() > 0.90, (
            "quantity no longer behaves like a pack count")

    def test_units_exceed_packs(self, scms):
        """Units are packs times pack size, so they must be the larger figure."""
        assert scms["units_ordered"].sum() > scms["packs_ordered"].sum()
        assert (scms["units_ordered"] >= scms["packs_ordered"]).all()

    def test_total_units_is_the_corrected_magnitude(self):
        """Was reported as 1.89e8 (a pack count mislabelled as units); ~9.98e9."""
        from src.analytics.procurement import scms_kpis

        kpis = scms_kpis()
        assert kpis["total_units"] > 9e9, "total_units has reverted to a pack count"
        assert kpis["total_packs"] < kpis["total_units"]

    def test_raw_duplicate_columns_are_dropped(self, scms):
        """The parsed numerics fully replace the raw text columns (B5)."""
        assert "Freight Cost (USD)" not in scms.columns
        assert "Weight (Kilograms)" not in scms.columns
        assert "freight_cost_usd" in scms.columns
        assert "weight_kg" in scms.columns


class TestDerivedMetrics:
    def test_delay_is_delivered_minus_scheduled(self, scms):
        expected = (scms["date_delivered"] - scms["date_scheduled"]).dt.days
        assert (scms["delivery_delay_days"] - expected).abs().max() == 0

    def test_is_late_follows_delay_sign(self, scms):
        """Early delivery counts as on time - the definition the whole page rests on."""
        late = scms["is_late"] == 1
        assert (scms.loc[late, "delivery_delay_days"] > 0).all()
        assert (scms.loc[~late & scms["is_late"].notna(),
                         "delivery_delay_days"] <= 0).all()

    def test_on_time_rate_matches_published_analysis(self, scms):
        on_time = 100 * (1 - scms["is_late"].mean())
        assert 88 <= on_time <= 89, f"on-time rate drifted to {on_time:.2f}%"

    def test_scheduled_lead_time_uses_no_future_information(self, scms):
        """The only lead-time feature safe for the model: it must be derivable
        without the delivery date."""
        expected = (scms["date_scheduled"] - scms["date_pq_sent"]).dt.days
        assert (scms["scheduled_lead_time_days"].dropna()
                - expected.dropna()).abs().max() == 0


# ---------------------------------------------------------------------------
class TestProcurementAnalytics:
    def test_traceability_covers_all_five_milestones(self, scms):
        """Milestone coverage moved from `procurement_funnel` to
        `pipeline.traceability`, which is the honest name for a non-monotone
        series."""
        from src.analytics import pipeline as pl

        trace = pl.traceability(scms)
        assert len(trace) == 5
        assert trace["coverage_pct"].between(0, 100).all()
        # The three delivery milestones are fully recorded.
        complete = trace[trace["stage"].isin(
            ["Delivery scheduled", "Delivered to client",
             "Delivery recorded in system"])]
        assert len(complete) == 3
        assert (complete["coverage_pct"] == 100.0).all()

    def test_lead_time_reports_its_denominator(self, scms):
        """Every interval is measured over a different subset, so the coverage
        must be stated rather than assumed."""
        breakdown = proc.lead_time_breakdown(scms)
        assert "coverage_pct" in breakdown.columns
        assert breakdown["coverage_pct"].between(0, 100).all()
        assert breakdown["line_items"].min() > 0

    def test_scorecards_exclude_small_samples(self, scms):
        """A vendor with three shipments must not outrank one with five hundred."""
        from src.config import get_config

        threshold = get_config().scms.min_shipments_for_scoring
        for frame in (proc.vendor_scorecard(scms),
                      proc.country_performance(scms),
                      proc.manufacturing_site_analysis(scms)):
            assert (frame["shipments"] >= threshold).all()

    def test_vendor_scorecard_is_worst_first(self, scms):
        scorecard = proc.vendor_scorecard(scms)
        assert scorecard["on_time_pct"].is_monotonic_increasing
        assert scorecard["rank"].tolist() == list(range(1, len(scorecard) + 1))

    def test_rdc_channel_is_the_weakest_vendor(self, scms):
        """A headline finding: the worst performer is an internal fulfilment
        route, not an external manufacturer."""
        worst = proc.vendor_scorecard(scms).iloc[0]
        assert "RDC" in worst["vendor"]
        assert worst["on_time_pct"] < 85

    def test_mode_cost_service_tradeoff_holds(self, scms):
        """Ocean should be cheapest per kg and least reliable; air the reverse."""
        modes = proc.mode_performance(scms).set_index("shipment_mode")
        if {"Ocean", "Air"}.issubset(modes.index):
            assert (modes.loc["Ocean", "median_freight_per_kg_usd"]
                    < modes.loc["Air", "median_freight_per_kg_usd"])
            assert modes.loc["Ocean", "on_time_pct"] < modes.loc["Air", "on_time_pct"]

    def test_delay_buckets_partition_all_shipments(self, scms):
        distribution = proc.delay_distribution(scms)
        assert distribution["shipments"].sum() == scms["delivery_delay_days"].notna().sum()
        assert distribution["share_pct"].sum() == pytest.approx(100.0, abs=0.1)

    def test_kpis_are_internally_consistent(self, scms):
        kpis = proc.scms_kpis(scms)
        assert kpis["shipments"] == len(scms)
        assert kpis["late_shipments"] == int(scms["is_late"].sum())
        assert kpis["on_time_delivery_pct"] == pytest.approx(
            100 * (1 - scms["is_late"].mean()), abs=0.01)

    def test_freight_summaries_use_only_real_prices(self, scms):
        """Rows where freight is bundled into the commodity cost must not be
        counted as zero-cost freight."""
        economics = proc.freight_economics(scms)
        assert (economics["median_freight_per_kg"] > 0).all()


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_meta():
    """Persisted metadata for the late-delivery model, or skip if untrained."""
    from src.ml import predict

    try:
        return predict.model_summary("late_delivery")
    except FileNotFoundError:
        pytest.skip("late_delivery model not trained")


class TestLateDeliveryModel:
    def test_no_leaking_features(self, model_meta):
        """Every excluded feature is computed from the delivery date and would
        hand the model its own answer."""
        leaky = {"date_delivered", "delivery_delay_days", "vendor_lead_time_days",
                 "total_lead_time_days", "recording_lag_days", "is_late"}
        used = set(model_meta["numeric_features"]) | set(model_meta["categorical_features"])
        assert not (used & leaky), f"leaking features present: {used & leaky}"

    def test_high_cardinality_identifiers_excluded(self, model_meta):
        """Encoding 73 vendors invites memorisation and cannot generalise to an
        unseen supplier."""
        used = set(model_meta["numeric_features"]) | set(model_meta["categorical_features"])
        assert not (used & {"vendor", "manufacturing_site", "country", "shipment_id"})

    def test_ranking_power_is_real(self, model_meta):
        """Accuracy is uninformative on this split; AUC is the honest measure."""
        assert model_meta["roc_auc_ovr"] > 0.75, "model lost its discriminative power"

    def test_beats_baseline_on_macro_f1(self, model_meta, scms):
        """Accuracy alone would look worse than the majority baseline, so the
        comparison that matters is macro F1."""
        from sklearn.metrics import f1_score

        n, late = len(scms), int(scms["is_late"].sum())
        truth = np.array(["Late"] * late + ["On Time"] * (n - late))
        baseline = f1_score(truth, np.array(["On Time"] * n),
                            average="macro", zero_division=0)
        assert model_meta["metrics"]["f1_macro"] > baseline + 0.10

    def test_targeting_curve_is_monotonic_and_lifts(self):
        """Capture must rise with coverage, and beat random targeting."""
        from src.ml import predict

        try:
            gains = predict.late_delivery_targeting_curve()
        except FileNotFoundError:
            pytest.skip("late_delivery model not trained")

        assert gains["capture_rate_pct"].is_monotonic_increasing
        assert (gains["lift"] > 1.0).all(), "model is no better than random"
        at20 = gains[gains["targeted_pct"] == 20.0].iloc[0]
        assert at20["capture_rate_pct"] > 40, "top-20% targeting should be useful"

    def test_prediction_interface_contract(self):
        from src.ml import predict

        try:
            result = predict.predict_late_delivery(
                quantity=20000, line_value_usd=80000, unit_price_usd=4.0,
                units_per_pack=60, weight_kg=800, freight_cost_usd=9000,
                scheduled_lead_time_days=160, shipment_mode="Ocean",
                product_group="ARV", sub_classification="Adult",
                region="West & Central Africa", fulfil_via="From RDC",
                inco_term="N/A - From RDC", managed_by="PMO - US",
                first_line_designation="Yes")
        except FileNotFoundError:
            pytest.skip("late_delivery model not trained")

        assert set(result) >= {"prediction", "late_probability", "risk_band",
                               "explanation", "model"}
        assert 0.0 <= result["late_probability"] <= 1.0
        assert result["risk_band"] in {"Low", "Elevated", "High"}

    def test_explanation_names_rdc_as_a_driver(self):
        """The explanation must match what the data shows - RDC fulfilment is the
        risk driver, not direct drop."""
        from src.ml import predict

        try:
            result = predict.predict_late_delivery(
                quantity=1000, line_value_usd=5000, unit_price_usd=5.0,
                units_per_pack=30, weight_kg=50, freight_cost_usd=600,
                scheduled_lead_time_days=120, shipment_mode="Truck",
                product_group="ARV", sub_classification="Adult",
                region="East Africa", fulfil_via="From RDC",
                inco_term="N/A - From RDC", managed_by="PMO - US",
                first_line_designation="Yes")
        except FileNotFoundError:
            pytest.skip("late_delivery model not trained")

        assert "distribution centre" in result["explanation"].lower()
