"""
Pharmaceutical supply chain dataset builder.

Why this module exists
----------------------
The clinical dataset (``drug200.csv``) is real Kaggle data, but no public
pharmaceutical dataset ships batch-level *funnel* telemetry - procurement
through patient dispensing with per-stage timestamps, yields, storage
conditions and shipment legs. Rather than stitch together several incomplete
extracts, this module generates a **deterministic, rule-based digital twin** of
a mid-size pharma supply chain.

Three properties make it defensible for analytics work:

1. **Reproducible.** A fixed seed produces byte-identical output, so every
   metric quoted in the README and dashboard can be re-derived from scratch.
2. **Calibrated.** Stage yields, QC release times, cold-chain excursion rates
   and OTIF levels are set from published industry benchmarks (see
   ``config/config.yaml``), not chosen arbitrarily.
3. **Coupled to the clinical data.** The product mix is derived from the
   observed prescription distribution in ``drug200.csv``, so the two halves of
   the platform describe one coherent business: the drugs the model predicts
   are the drugs the supply chain delivers.

Structural signals deliberately encoded (these are what the analytics layer is
built to *discover*, and what the dashboard narrative explains):

* Quality Testing is the dominant loss stage and the slowest stage - the
  primary bottleneck.
* Two API suppliers with weak reliability scores drive a disproportionate share
  of QA failures.
* Cold-chain products carry materially higher stability and expiry risk.
* One region under-performs on on-time delivery because of its transport mix.

Output
------
Seven CSVs in ``data/raw`` forming a star schema:

===================================  ==========================================
``dim_drugs.csv``                    product master
``dim_suppliers.csv``                API supplier master
``dim_warehouses.csv``               distribution network master
``supply_chain_batches.csv``         funnel fact - one row per batch
``supply_chain_shipments.csv``       logistics fact - one row per shipment leg
``supply_chain_inventory.csv``       monthly inventory snapshots
``supply_chain_demand.csv``          monthly demand history
===================================  ==========================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ensure_directories, get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Reference master data
# ---------------------------------------------------------------------------
# Drug codes intentionally mirror the target classes in drug200.csv so the
# clinical model and the supply chain describe the same product portfolio.
DRUG_CATALOG: list[dict] = [
    {
        "drug_code": "DrugY", "brand_name": "Cardiovex", "therapeutic_area": "Cardiovascular",
        "form": "Tablet", "strength_mg": 50, "unit_cost_usd": 2.40, "shelf_life_months": 36,
        "storage_condition": "Controlled Room Temperature", "is_cold_chain": 0,
    },
    {
        "drug_code": "DrugX", "brand_name": "Nephrolyte", "therapeutic_area": "Nephrology",
        "form": "Tablet", "strength_mg": 25, "unit_cost_usd": 1.85, "shelf_life_months": 30,
        "storage_condition": "Controlled Room Temperature", "is_cold_chain": 0,
    },
    {
        "drug_code": "DrugA", "brand_name": "Angiotrol", "therapeutic_area": "Hypertension",
        "form": "Injection", "strength_mg": 10, "unit_cost_usd": 9.60, "shelf_life_months": 18,
        "storage_condition": "Cold Chain 2-8C", "is_cold_chain": 1,
    },
    {
        "drug_code": "DrugB", "brand_name": "Betacor", "therapeutic_area": "Hypertension",
        "form": "Capsule", "strength_mg": 100, "unit_cost_usd": 3.15, "shelf_life_months": 24,
        "storage_condition": "Controlled Room Temperature", "is_cold_chain": 0,
    },
    {
        "drug_code": "DrugC", "brand_name": "Cholestat", "therapeutic_area": "Lipid Disorders",
        "form": "Injection", "strength_mg": 20, "unit_cost_usd": 12.30, "shelf_life_months": 15,
        "storage_condition": "Cold Chain 2-8C", "is_cold_chain": 1,
    },
]

# `reliability` is the supplier's API quality/consistency score. It feeds QA
# yield directly - the two low-reliability suppliers are the intended finding.
SUPPLIER_CATALOG: list[dict] = [
    {"supplier_id": "SUP-01", "supplier_name": "Aurobindo API Ltd",     "country": "India",       "reliability": 0.94, "lead_time_days": 18},
    {"supplier_id": "SUP-02", "supplier_name": "Divis Laboratories",    "country": "India",       "reliability": 0.96, "lead_time_days": 16},
    {"supplier_id": "SUP-03", "supplier_name": "Zhejiang Kangtai Chem", "country": "China",       "reliability": 0.83, "lead_time_days": 32},
    {"supplier_id": "SUP-04", "supplier_name": "Lonza Group AG",        "country": "Switzerland", "reliability": 0.97, "lead_time_days": 14},
    {"supplier_id": "SUP-05", "supplier_name": "Cambrex Corporation",   "country": "USA",         "reliability": 0.95, "lead_time_days": 12},
    {"supplier_id": "SUP-06", "supplier_name": "Hikal Speciality",      "country": "India",       "reliability": 0.88, "lead_time_days": 24},
    {"supplier_id": "SUP-07", "supplier_name": "Shandong Xinhua Pharm", "country": "China",       "reliability": 0.80, "lead_time_days": 35},
    {"supplier_id": "SUP-08", "supplier_name": "Siegfried Holding",     "country": "Germany",     "reliability": 0.96, "lead_time_days": 15},
]

# `capacity_units` is the storage envelope allocated to THIS five-product
# portfolio, not the whole site. Sizing it to the modelled portfolio is what
# makes the utilisation metric meaningful - measuring a 5-SKU position against
# whole-site capacity would report a meaningless single-digit percentage.
WAREHOUSE_CATALOG: list[dict] = [
    {"warehouse_id": "WH-NA-01", "warehouse_name": "Newark DC",     "region": "North America",       "country": "USA",     "capacity_units": 620_000, "temp_controlled": 1},
    {"warehouse_id": "WH-NA-02", "warehouse_name": "Memphis Hub",   "region": "North America",       "country": "USA",     "capacity_units": 480_000, "temp_controlled": 1},
    {"warehouse_id": "WH-EU-01", "warehouse_name": "Rotterdam DC",  "region": "Europe",              "country": "Netherlands", "capacity_units": 540_000, "temp_controlled": 1},
    {"warehouse_id": "WH-AP-01", "warehouse_name": "Singapore DC",  "region": "Asia-Pacific",        "country": "Singapore", "capacity_units": 400_000, "temp_controlled": 1},
    {"warehouse_id": "WH-LA-01", "warehouse_name": "Sao Paulo DC",  "region": "Latin America",       "country": "Brazil",  "capacity_units": 215_000, "temp_controlled": 0},
    {"warehouse_id": "WH-ME-01", "warehouse_name": "Dubai DC",      "region": "Middle East & Africa", "country": "UAE",    "capacity_units": 150_000, "temp_controlled": 0},
]

REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"]
REGION_DEMAND_SHARE = {
    "North America": 0.34, "Europe": 0.27, "Asia-Pacific": 0.22,
    "Latin America": 0.10, "Middle East & Africa": 0.07,
}
# Regional on-time delivery penalty. Middle East & Africa is the intended
# under-performer, driven by longer road legs and fewer cold-chain lanes.
REGION_DELAY_FACTOR = {
    "North America": 0.92, "Europe": 0.95, "Asia-Pacific": 1.05,
    "Latin America": 1.18, "Middle East & Africa": 1.32,
}

CARRIERS = ["DHL Life Sciences", "Kuehne+Nagel Pharma", "FedEx Custom Critical", "Marken", "Regional 3PL"]
QA_FAILURE_REASONS = [
    "Dissolution Out of Specification", "Microbial Limit Exceeded", "Assay Potency Deviation",
    "Content Uniformity Failure", "Particulate Contamination", "Moisture Content Deviation",
]
SHIPMENT_LEGS = [
    ("Plant to Warehouse", "units_packaged", "units_warehoused"),
    ("Warehouse to Distributor", "units_warehoused", "units_distributed"),
    ("Distributor to Pharmacy", "units_distributed", "units_delivered"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _product_mix_from_clinical_data() -> dict[str, float]:
    """Derive the product volume mix from observed prescriptions in drug200.

    This is the join between the two datasets: whichever drug is prescribed most
    often in the clinical data is the drug the supply chain moves most of.
    Falls back to a documented default mix if the clinical file is unavailable.
    """
    cfg = get_config()
    path = resolve_path(cfg.datasets.drug200)
    fallback = {"DrugY": 0.455, "DrugX": 0.270, "DrugA": 0.115, "DrugB": 0.080, "DrugC": 0.080}

    if not path.exists():
        log.warning("drug200.csv not found - using documented fallback product mix.")
        return fallback

    counts = pd.read_csv(path)["Drug"].value_counts(normalize=True)
    # drug200 labels are inconsistently cased ("DrugY" vs "drugA"); normalise.
    canonical = {d["drug_code"].lower(): d["drug_code"] for d in DRUG_CATALOG}
    mix = {canonical[k.lower()]: float(v) for k, v in counts.items() if k.lower() in canonical}

    total = sum(mix.values())
    if not mix or total <= 0:
        return fallback
    normalised = {k: v / total for k, v in mix.items()}
    log.info("Product mix derived from drug200: %s",
             {k: f"{v:.1%}" for k, v in sorted(normalised.items(), key=lambda x: -x[1])})
    return normalised


def _potency_after_storage(
    temp_c: np.ndarray, humidity_pct: np.ndarray, duration_days: np.ndarray,
    is_cold_chain: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Model residual potency (%) after storage.

    Implements the **Q10 rule**, the Arrhenius approximation that underpins ICH
    Q1A accelerated stability testing: the chemical degradation rate multiplies
    by ``q10_factor`` for every 10 degC above the product's labelled storage
    temperature, and divides by it for every 10 degC below. Moisture uptake
    accelerates hydrolytic degradation roughly linearly with relative humidity.
    Both effects scale with exposure time.

    Choosing a continuous law rather than a one-sided "excursion penalty"
    matters: it means storing product *colder* than the label genuinely slows
    degradation, instead of being scored identically to storage at the limit.

    Returns
    -------
    np.ndarray
        Potency as a percentage of label claim, clipped to [80, 100].
    """
    cfg = get_config().stability
    reference = np.where(is_cold_chain == 1, cfg.reference_temp_cold, cfg.reference_temp_ambient)

    temp_factor = cfg.q10_factor ** ((temp_c - reference) / 10.0)
    humidity_factor = np.clip(
        1.0 + cfg.humidity_sensitivity * (humidity_pct - cfg.reference_humidity_pct), 0.6, None)

    degradation = (
        cfg.base_degradation_rate
        * (duration_days / 30.0)
        * temp_factor
        * humidity_factor
    )
    potency = 100.0 - degradation + rng.normal(0.0, cfg.noise_sd, size=len(temp_c))
    return np.clip(potency, 80.0, 100.0)


def _assign_batch_risk(potency: np.ndarray, days_to_expiry: np.ndarray, qa_pass: np.ndarray) -> np.ndarray:
    """Label each batch Low / Medium / High risk.

    Business rule agreed with (simulated) Quality Assurance:

    * **High**   - out of potency specification, failed QA, or effectively expired.
    * **Medium** - inside specification but with little headroom on either
      potency or remaining shelf life.
    * **Low**    - comfortable on both dimensions.
    """
    cfg = get_config().stability
    high = (potency < cfg.potency_spec_min) | (qa_pass == 0) | (days_to_expiry < 30)
    medium = (potency < cfg.potency_warning) | (days_to_expiry < 120)
    return np.where(high, "High", np.where(medium, "Medium", "Low"))


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------
def build_dimensions() -> dict[str, pd.DataFrame]:
    """Build the three dimension tables (drugs, suppliers, warehouses)."""
    drugs = pd.DataFrame(DRUG_CATALOG)
    mix = _product_mix_from_clinical_data()
    drugs["volume_share"] = drugs["drug_code"].map(mix).fillna(0.0)
    drugs["volume_share"] /= drugs["volume_share"].sum()

    suppliers = pd.DataFrame(SUPPLIER_CATALOG)
    # OTIF and quality rating are monotonic in reliability, with a small offset
    # so the two metrics are correlated but not identical.
    suppliers["quality_rating"] = (suppliers["reliability"] * 100).round(1)
    suppliers["otif_pct"] = (suppliers["reliability"] * 96 + 3).round(1)
    suppliers["tier"] = pd.cut(
        suppliers["quality_rating"], bins=[0, 85, 92, 100],
        labels=["Watchlist", "Approved", "Preferred"],
    ).astype(str)

    warehouses = pd.DataFrame(WAREHOUSE_CATALOG)
    return {"drugs": drugs, "suppliers": suppliers, "warehouses": warehouses}


# ---------------------------------------------------------------------------
# Fact builders
# ---------------------------------------------------------------------------
def build_batches(dims: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    """Build the batch funnel fact table - the analytical core of the platform.

    Each row follows one manufactured batch through all eight supply chain
    stages, recording the units surviving each stage, the date it cleared the
    stage, storage conditions and the resulting quality outcome.
    """
    cfg = get_config()
    gen_cfg = cfg.generation
    n = int(gen_cfg.n_batches)
    drugs, suppliers, warehouses = dims["drugs"], dims["suppliers"], dims["warehouses"]

    # --- Batch identity -----------------------------------------------------
    drug_idx = rng.choice(len(drugs), size=n, p=drugs["volume_share"].to_numpy())
    drug_rows = drugs.iloc[drug_idx].reset_index(drop=True)

    supplier_idx = rng.choice(len(suppliers), size=n)
    supplier_rows = suppliers.iloc[supplier_idx].reset_index(drop=True)

    warehouse_idx = rng.choice(len(warehouses), size=n)
    warehouse_rows = warehouses.iloc[warehouse_idx].reset_index(drop=True)

    start = pd.Timestamp(gen_cfg.start_date)
    end = pd.Timestamp(gen_cfg.end_date)
    # Reserve trailing days so late batches still complete inside the window.
    span_days = (end - start).days - 120
    offsets = rng.integers(0, span_days, size=n)
    date_procurement = start + pd.to_timedelta(offsets, unit="D")

    df = pd.DataFrame({
        "batch_id": [f"BATCH-{i:05d}" for i in range(1, n + 1)],
        "drug_code": drug_rows["drug_code"],
        "brand_name": drug_rows["brand_name"],
        "therapeutic_area": drug_rows["therapeutic_area"],
        "form": drug_rows["form"],
        "is_cold_chain": drug_rows["is_cold_chain"],
        "shelf_life_months": drug_rows["shelf_life_months"],
        "unit_cost_usd": drug_rows["unit_cost_usd"],
        "supplier_id": supplier_rows["supplier_id"],
        "supplier_name": supplier_rows["supplier_name"],
        "supplier_country": supplier_rows["country"],
        "supplier_reliability": supplier_rows["reliability"],
        "warehouse_id": warehouse_rows["warehouse_id"],
        "region": warehouse_rows["region"],
        "date_procurement": date_procurement,
    })

    # --- Stage 1: Procurement ----------------------------------------------
    # Batch sizes are lognormal: many routine batches, a few large campaigns.
    df["units_procured"] = np.round(rng.lognormal(mean=10.4, sigma=0.42, size=n)).astype(int)

    yields = gen_cfg.stage_yield
    reliability = df["supplier_reliability"].to_numpy()

    def _apply_stage(units: np.ndarray, base_yield: float, spread: float) -> np.ndarray:
        """Carry units into the next stage, applying a noisy realised yield."""
        realised = np.clip(rng.normal(base_yield, spread), 0.55, 0.999)
        return np.round(units * realised).astype(int)

    # --- Stage 2: Manufacturing --------------------------------------------
    df["units_manufactured"] = _apply_stage(
        df["units_procured"].to_numpy(), yields.procurement_to_manufacturing, 0.012)

    # --- Stage 3: Quality Testing ------------------------------------------
    # Batch disposition is modelled as an explicit pass/fail decision rather
    # than a yield cut-off, because QC release *is* a regulatory decision. API
    # quality is its dominant driver: a batch from a 97%-reliability supplier
    # clears about 97% of the time, one from an 80% supplier about 78%.
    qa_pass_prob = np.clip(1.0 - 1.10 * (1.0 - reliability), 0.60, 0.99)
    df["qa_pass"] = (rng.random(n) < qa_pass_prob).astype(int)
    df["qa_result"] = np.where(df["qa_pass"] == 1, "Pass", "Fail")
    df["qa_fail_reason"] = np.where(
        # "Not Applicable" rather than "N/A": pandas reads the latter as NaN on
        # round-trip through CSV, which would show up as 90% missing data and
        # send a reviewer chasing a defect that does not exist.
        df["qa_pass"] == 1, "Not Applicable", rng.choice(QA_FAILURE_REASONS, size=n))

    # Released batches lose only sampling and retain quantities. Rejected
    # batches are largely scrapped, with a fraction recovered through rework.
    qa_unit_yield = np.where(
        df["qa_pass"] == 1,
        rng.normal(0.972, 0.012, size=n),
        rng.normal(0.545, 0.090, size=n),
    )
    df["units_qa_passed"] = np.round(
        df["units_manufactured"].to_numpy() * np.clip(qa_unit_yield, 0.30, 0.995)
    ).astype(int)

    # --- Stages 4-8: Packaging through Patient -----------------------------
    df["units_packaged"] = _apply_stage(
        df["units_qa_passed"].to_numpy(), yields.quality_to_packaging, 0.020)
    df["units_warehoused"] = _apply_stage(
        df["units_packaged"].to_numpy(), yields.packaging_to_warehouse, 0.008)
    df["units_distributed"] = _apply_stage(
        df["units_warehoused"].to_numpy(), yields.warehouse_to_distributor, 0.025)
    df["units_delivered"] = _apply_stage(
        df["units_distributed"].to_numpy(), yields.distributor_to_pharmacy, 0.018)
    df["units_dispensed"] = _apply_stage(
        df["units_delivered"].to_numpy(), yields.pharmacy_to_patient, 0.035)

    # --- Stage dwell times --------------------------------------------------
    # Lead time from the API supplier drives the procurement stage directly.
    delays = gen_cfg.stage_delay_days
    region_factor = df["region"].map(REGION_DELAY_FACTOR).to_numpy()

    stage_delay_cols: dict[str, np.ndarray] = {}
    for stage in cfg.funnel.stages:
        spec = delays[stage]
        draw = rng.normal(spec["mean"], spec["sd"], size=n)
        if stage == "Procurement":
            # Anchor on the contracted supplier lead time.
            draw = supplier_rows["lead_time_days"].to_numpy() + rng.normal(0, 4, size=n)
        elif stage == "Quality Testing":
            # Failed batches require investigation and retest - materially slower.
            draw = draw + np.where(df["qa_pass"] == 0, rng.normal(9, 3, size=n), 0)
        elif stage in ("Distributor", "Hospital/Pharmacy"):
            # Downstream legs inherit regional transport performance.
            draw = draw * region_factor
        stage_delay_cols[stage] = np.clip(draw, 1, None)

    # Cumulative dates: each stage completes `delay` days after the previous one.
    running = df["date_procurement"].copy()
    for stage in cfg.funnel.stages:
        column = cfg.funnel.date_columns[stage]
        running = running + pd.to_timedelta(np.round(stage_delay_cols[stage]), unit="D")
        df[column] = running
        df[f"delay_{stage.lower().replace('/', '_').replace(' ', '_')}_days"] = \
            np.round(stage_delay_cols[stage], 1)

    df["total_cycle_time_days"] = (df["date_patient"] - df["date_procurement"]).dt.days
    df["qa_delay_days"] = df["delay_quality_testing_days"]

    # --- Storage conditions and stability ----------------------------------
    cold = df["is_cold_chain"].to_numpy()
    # Ambient products track local climate, so hot regions push both
    # temperature and humidity up.
    hot_region = df["region"].isin(["Middle East & Africa", "Latin America"]).to_numpy()

    # Cold-chain product normally sits in validated 2-8C storage. A minority of
    # batches suffer a documented excursion - failed compressor, broken seal,
    # extended tarmac dwell - which is precisely the failure mode cold-chain
    # monitoring programmes exist to catch. The resulting bimodal profile means
    # cold-chain risk lives in the tail, not the average.
    cold_excursion = (rng.random(n) < 0.13) & (cold == 1)
    cold_temp = np.where(
        cold_excursion, rng.uniform(11.0, 26.0, size=n), rng.normal(5.5, 1.4, size=n))
    ambient_temp = rng.normal(24.0, 3.2, size=n) + np.where(
        hot_region, rng.normal(4.5, 2.0, size=n), 0)
    temp = np.where(cold == 1, cold_temp, ambient_temp)

    humidity = np.clip(
        rng.normal(52, 11, size=n) + np.where(hot_region, 9, 0), 15, 95)

    df["cold_chain_excursion"] = cold_excursion.astype(int)
    df["storage_temp_c"] = np.round(np.clip(temp, 1.5, 45), 1)
    df["storage_humidity_pct"] = np.round(humidity, 1)
    # Storage duration = time held between packaging and pharmacy handover.
    df["storage_duration_days"] = (df["date_pharmacy"] - df["date_packaging"]).dt.days.clip(lower=1)

    df["potency_pct"] = np.round(_potency_after_storage(
        df["storage_temp_c"].to_numpy(), df["storage_humidity_pct"].to_numpy(),
        df["storage_duration_days"].to_numpy(), cold, rng), 2)

    # --- Expiry -------------------------------------------------------------
    df["expiry_date"] = df["date_manufacturing"] + pd.to_timedelta(
        df["shelf_life_months"] * 30.44, unit="D")
    df["days_to_expiry_at_delivery"] = (df["expiry_date"] - df["date_pharmacy"]).dt.days

    df["batch_risk_label"] = _assign_batch_risk(
        df["potency_pct"].to_numpy(), df["days_to_expiry_at_delivery"].to_numpy(),
        df["qa_pass"].to_numpy())

    # --- Financials and calendar keys --------------------------------------
    df["batch_value_usd"] = np.round(df["units_procured"] * df["unit_cost_usd"], 2)
    df["units_lost"] = df["units_procured"] - df["units_dispensed"]
    df["value_lost_usd"] = np.round(df["units_lost"] * df["unit_cost_usd"], 2)
    df["year"] = df["date_procurement"].dt.year
    df["month"] = df["date_procurement"].dt.month
    df["year_month"] = df["date_procurement"].dt.to_period("M").astype(str)
    df["quarter"] = df["date_procurement"].dt.to_period("Q").astype(str)

    log.info("Generated %d batches | end-to-end yield %.1f%% | QA pass rate %.1f%%",
             len(df), 100 * df["units_dispensed"].sum() / df["units_procured"].sum(),
             100 * df["qa_pass"].mean())
    return df


def build_shipments(batches: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Explode each batch into its three physical transport legs."""
    cfg = get_config()
    records: list[pd.DataFrame] = []

    for leg_no, (leg_name, from_col, to_col) in enumerate(SHIPMENT_LEGS, start=1):
        n = len(batches)
        region_factor = batches["region"].map(REGION_DELAY_FACTOR).to_numpy()
        cold = batches["is_cold_chain"].to_numpy()

        # Cold-chain freight moves by air far more often than ambient freight.
        mode = np.where(
            cold == 1,
            rng.choice(["Air", "Road"], size=n, p=[0.72, 0.28]),
            rng.choice(["Road", "Sea", "Air", "Rail"], size=n, p=[0.52, 0.24, 0.14, 0.10]),
        )
        planned = np.select(
            [mode == "Air", mode == "Road", mode == "Rail", mode == "Sea"],
            [3, 6, 9, 22], default=6,
        ).astype(float)

        # Most legs run close to plan. A minority hit a disruption - customs
        # hold, weather, carrier capacity shortfall - which adds a right-skewed
        # delay. Regional logistics maturity drives how often that happens,
        # which is what makes on-time delivery differ by region.
        disrupted = rng.random(n) < np.clip(0.085 * region_factor, 0, 0.5)
        disruption_days = np.where(
            disrupted, rng.gamma(shape=2.0, scale=2.2, size=n) * region_factor, 0.0)
        actual = np.round(planned * rng.normal(0.98, 0.04, size=n) + disruption_days)
        actual = np.maximum(actual, 1)

        leg = pd.DataFrame({
            "shipment_id": [f"SHP-{leg_no}-{i:05d}" for i in range(1, n + 1)],
            "batch_id": batches["batch_id"].to_numpy(),
            "leg": leg_name,
            "leg_sequence": leg_no,
            "drug_code": batches["drug_code"].to_numpy(),
            "region": batches["region"].to_numpy(),
            "warehouse_id": batches["warehouse_id"].to_numpy(),
            "supplier_id": batches["supplier_id"].to_numpy(),
            "supplier_name": batches["supplier_name"].to_numpy(),
            "carrier": rng.choice(CARRIERS, size=n, p=[0.26, 0.24, 0.20, 0.16, 0.14]),
            "transport_mode": mode,
            "is_cold_chain": cold,
            "units_shipped": batches[from_col].to_numpy(),
            "units_received": batches[to_col].to_numpy(),
            "planned_transit_days": planned,
            "actual_transit_days": actual,
        })

        leg["ship_date"] = batches[cfg.funnel.date_columns[
            {"Plant to Warehouse": "Packaging",
             "Warehouse to Distributor": "Warehouse",
             "Distributor to Pharmacy": "Distributor"}[leg_name]]].to_numpy()
        leg["delivery_date"] = leg["ship_date"] + pd.to_timedelta(leg["actual_transit_days"], unit="D")
        leg["delay_days"] = leg["actual_transit_days"] - leg["planned_transit_days"]
        leg["is_late"] = (leg["delay_days"] > cfg.shipments.on_time_grace_days).astype(int)
        leg["units_damaged"] = (leg["units_shipped"] - leg["units_received"]).clip(lower=0)
        leg["damage_rate_pct"] = np.round(
            100 * leg["units_damaged"] / leg["units_shipped"].replace(0, np.nan), 2).fillna(0)

        # Temperature excursions are a cold-chain-specific failure mode and get
        # worse the longer the shipment is in transit.
        excursion_p = np.where(cold == 1, 0.09, 0.03) * (1 + leg["delay_days"].clip(lower=0) / 20)
        leg["temperature_excursion"] = (rng.random(n) < np.clip(excursion_p, 0, 0.6)).astype(int)

        rate = np.select([mode == "Air", mode == "Road", mode == "Rail", mode == "Sea"],
                         [0.42, 0.16, 0.11, 0.06], default=0.16)
        leg["freight_cost_usd"] = np.round(
            leg["units_shipped"] * rate * np.where(cold == 1, 1.55, 1.0), 2)
        leg["year_month"] = pd.to_datetime(leg["ship_date"]).dt.to_period("M").astype(str)
        records.append(leg)

    shipments = pd.concat(records, ignore_index=True)
    log.info("Generated %d shipment legs | on-time rate %.1f%%",
             len(shipments), 100 * (1 - shipments["is_late"].mean()))
    return shipments


def build_demand(dims: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    """Build monthly demand history by drug and region.

    Signal = level x growth trend x annual seasonality x noise. Seasonality is
    strongest for cardiovascular products, which peak in winter months.
    """
    cfg = get_config()
    gen = cfg.generation.demand
    months = pd.date_range(cfg.generation.start_date, cfg.generation.end_date, freq="MS")
    drugs = dims["drugs"]
    rows: list[dict] = []

    for t, month in enumerate(months):
        trend = (1 + gen.annual_growth_rate) ** (t / 12.0)
        # Peak in month 1 (January), trough mid-year - respiratory/cardio pattern.
        seasonal = 1 + gen.seasonal_amplitude * np.cos(2 * np.pi * (month.month - 1) / 12)
        for _, drug in drugs.iterrows():
            for region, share in REGION_DEMAND_SHARE.items():
                base = gen.base_units_per_month * drug["volume_share"] * share
                demanded = base * trend * seasonal * rng.normal(1.0, gen.noise_sd)
                demanded = max(int(round(demanded)), 0)
                # Fill rate degrades in regions with weaker logistics.
                fill = np.clip(rng.normal(0.96 - 0.05 * (REGION_DELAY_FACTOR[region] - 1), 0.03), 0.6, 1.0)
                fulfilled = int(round(demanded * fill))
                rows.append({
                    "date": month, "year_month": month.strftime("%Y-%m"),
                    "drug_code": drug["drug_code"], "brand_name": drug["brand_name"],
                    "region": region, "units_demanded": demanded,
                    "units_fulfilled": fulfilled,
                    "units_backordered": demanded - fulfilled,
                    "fill_rate_pct": round(100 * fill, 2),
                    "revenue_usd": round(fulfilled * drug["unit_cost_usd"] * 2.6, 2),
                })

    demand = pd.DataFrame(rows)
    log.info("Generated %d demand rows across %d months", len(demand), len(months))
    return demand


def build_inventory(dims: dict[str, pd.DataFrame], demand: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Build monthly inventory snapshots per warehouse and drug.

    On-hand cover is drawn around a target months-of-supply that varies by
    product: short-shelf-life cold-chain items are held leaner, long-shelf-life
    tablets are over-stocked - the classic pharma working-capital trade-off.
    """
    cfg = get_config()
    warehouses, drugs = dims["warehouses"], dims["drugs"]
    monthly_regional = (demand.groupby(["year_month", "region", "drug_code"])["units_demanded"]
                        .sum().reset_index())
    rows: list[dict] = []

    for _, wh in warehouses.iterrows():
        region_demand = monthly_regional[monthly_regional["region"] == wh["region"]]
        # Split regional demand across the warehouses serving that region.
        n_wh_in_region = (warehouses["region"] == wh["region"]).sum()

        for _, snap in region_demand.iterrows():
            drug = drugs[drugs["drug_code"] == snap["drug_code"]].iloc[0]
            monthly_demand = max(int(snap["units_demanded"] / n_wh_in_region), 1)

            # Leaner cover for short-shelf-life cold chain, richer for tablets.
            target_cover = 1.6 if drug["is_cold_chain"] else 3.1
            cover = np.clip(rng.normal(target_cover, 0.85), 0.2, 7.0)
            on_hand = int(monthly_demand * cover)

            lead_time_months = 1.2
            safety_stock = int(cfg.inventory.service_level_z * monthly_demand * 0.30)
            reorder_point = int(monthly_demand * lead_time_months + safety_stock)

            # Ageing stock: a small share of each position is close to expiry.
            days_to_expiry = int(np.clip(
                rng.normal(drug["shelf_life_months"] * 30.44 * 0.45, 120), 5, None))
            expiring_share = np.clip(rng.beta(1.6, 12), 0, 0.5)

            rows.append({
                "snapshot_month": snap["year_month"],
                "warehouse_id": wh["warehouse_id"],
                "warehouse_name": wh["warehouse_name"],
                "region": wh["region"],
                "drug_code": drug["drug_code"],
                "brand_name": drug["brand_name"],
                "is_cold_chain": int(drug["is_cold_chain"]),
                "unit_cost_usd": float(drug["unit_cost_usd"]),
                "units_on_hand": on_hand,
                "monthly_demand_units": monthly_demand,
                "units_issued": int(monthly_demand * np.clip(rng.normal(0.95, 0.06), 0.5, 1.0)),
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "months_of_supply": round(cover, 2),
                "days_to_expiry": days_to_expiry,
                "units_expiring_soon": int(on_hand * expiring_share),
                "warehouse_capacity": int(wh["capacity_units"]),
                "inventory_value_usd": round(on_hand * float(drug["unit_cost_usd"]), 2),
            })

    inventory = pd.DataFrame(rows)
    log.info("Generated %d inventory snapshot rows", len(inventory))
    return inventory


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def inject_quality_issues(
    tables: dict[str, pd.DataFrame], rng: np.random.Generator
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Degrade the generated tables into a realistic *raw* (bronze) extract.

    Source systems in a pharmaceutical operation do not hand over clean data.
    Temperature loggers drop readings, ERP double-posts batch confirmations,
    carrier EDI feeds go missing, and region is captured as free text by
    different regional teams. This function reproduces those defect classes so
    that:

    * the data quality module has genuine problems to detect, and
    * ``src.data.cleaning`` has genuine remediation to perform.

    The clinical dataset (``drug200``) and the dimension tables are left
    untouched - ``drug200.csv`` is real Kaggle data and is never modified.

    Parameters
    ----------
    tables
        Clean tables from the builders, keyed by logical name.
    rng
        Seeded generator, so the injected defects are reproducible.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame]
        The degraded tables and a defect log describing what was injected
        (used by the dashboard to show expected-versus-detected defects).
    """
    cfg = get_config().generation.quality_issues
    defects: list[dict] = []
    if not cfg.enabled:
        log.info("Quality issue injection disabled - emitting a pristine extract.")
        return tables, pd.DataFrame(columns=["table", "column", "defect_type", "rows_affected"])

    out = {name: frame.copy() for name, frame in tables.items()}

    def _record(table: str, column: str, defect: str, count: int) -> None:
        if count:
            defects.append({"table": table, "column": column,
                            "defect_type": defect, "rows_affected": int(count)})

    # --- 1. Missing values (sensor and interface dropouts) -----------------
    column_owner = {
        "storage_temp_c": "batches", "storage_humidity_pct": "batches",
        "supplier_reliability": "batches", "actual_transit_days": "shipments",
        "units_on_hand": "inventory",
    }
    for column, rate in cfg.missing.items():
        table = column_owner.get(column)
        if table is None or column not in out[table].columns:
            continue
        frame = out[table]
        mask = rng.random(len(frame)) < rate
        frame.loc[mask, column] = np.nan
        _record(table, column, "missing_value", mask.sum())

    # --- 2. Duplicate rows (ERP double-posting) ---------------------------
    for table in ("batches", "shipments"):
        frame = out[table]
        n_dupes = int(len(frame) * cfg.duplicate_row_rate)
        if n_dupes:
            picked = rng.choice(len(frame), size=n_dupes, replace=False)
            out[table] = pd.concat([frame, frame.iloc[picked]], ignore_index=True)
            _record(table, "<row>", "duplicate_row", n_dupes)

    # --- 3. Invalid categorical values (free-text region entry) -----------
    aliases = list(cfg.region_aliases.keys())
    canonical_to_alias: dict[str, list[str]] = {}
    for alias, canonical in cfg.region_aliases.items():
        canonical_to_alias.setdefault(canonical, []).append(alias)

    for table in ("batches", "shipments", "inventory"):
        frame = out[table]
        if "region" not in frame.columns:
            continue
        mask = rng.random(len(frame)) < cfg.invalid_category_rate
        idx = frame.index[mask]
        # Swap each selected value for one of *its own* dirty spellings, so the
        # defect is a formatting problem rather than a change of meaning.
        replacements = [
            rng.choice(canonical_to_alias[value]) if value in canonical_to_alias
            else rng.choice(aliases)
            for value in frame.loc[idx, "region"]
        ]
        frame.loc[idx, "region"] = replacements
        _record(table, "region", "invalid_category", len(idx))

    # --- 4. Impossible values (unit/sign errors at data entry) ------------
    batches = out["batches"]
    mask = rng.random(len(batches)) < cfg.impossible_value_rate
    batches.loc[mask, "storage_duration_days"] = -rng.integers(1, 15, size=int(mask.sum()))
    _record("batches", "storage_duration_days", "negative_duration", mask.sum())

    mask = rng.random(len(batches)) < cfg.impossible_value_rate
    batches.loc[mask, "potency_pct"] = np.round(rng.uniform(100.5, 108.0, size=int(mask.sum())), 2)
    _record("batches", "potency_pct", "potency_above_100pct", mask.sum())

    # --- 5. Trailing whitespace (CSV export artefact) ---------------------
    for table, column in (("batches", "supplier_name"), ("shipments", "carrier")):
        frame = out[table]
        if column not in frame.columns:
            continue
        mask = rng.random(len(frame)) < cfg.whitespace_rate
        idx = frame.index[mask]
        frame.loc[idx, column] = frame.loc[idx, column].astype(str) + "  "
        _record(table, column, "trailing_whitespace", len(idx))

    defect_log = pd.DataFrame(defects)
    log.info("Injected %d defect groups across the raw extract (%d rows affected)",
             len(defect_log), int(defect_log["rows_affected"].sum()) if len(defect_log) else 0)
    return out, defect_log


def generate_all(save: bool = True) -> dict[str, pd.DataFrame]:
    """Generate the full dataset family and optionally persist it to ``data/raw``.

    Parameters
    ----------
    save
        When True (default) each table is written to the path declared in
        ``config.datasets``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``drugs``, ``suppliers``, ``warehouses``, ``batches``,
        ``shipments``, ``demand``, ``inventory``.
    """
    cfg = get_config()
    ensure_directories()
    rng = np.random.default_rng(cfg.project.random_seed)
    log.info("Building pharmaceutical supply chain dataset (seed=%d)", cfg.project.random_seed)

    dims = build_dimensions()
    batches = build_batches(dims, rng)
    shipments = build_shipments(batches, rng)
    demand = build_demand(dims, rng)
    inventory = build_inventory(dims, demand, rng)

    tables = {
        "drugs": dims["drugs"], "suppliers": dims["suppliers"],
        "warehouses": dims["warehouses"], "batches": batches,
        "shipments": shipments, "demand": demand, "inventory": inventory,
    }

    # Degrade into a realistic raw extract. `data/raw` is the bronze layer:
    # what the source systems actually hand over. `src.data.cleaning` promotes
    # it to the silver layer that the analytics modules consume.
    tables, defect_log = inject_quality_issues(tables, rng)

    if save:
        for name, frame in tables.items():
            path = resolve_path(cfg.datasets[name])
            frame.to_csv(path, index=False)
            log.info("Wrote %-10s -> %-42s (%6d rows, %2d cols)",
                     name, str(path.relative_to(resolve_path("."))), len(frame), frame.shape[1])

        if len(defect_log):
            defect_path = resolve_path(cfg.paths.data_raw) / "injected_defect_log.csv"
            defect_log.to_csv(defect_path, index=False)
            log.info("Wrote defect log -> %s (%d groups)", defect_path.name, len(defect_log))

    return tables


if __name__ == "__main__":  # pragma: no cover
    generate_all()
