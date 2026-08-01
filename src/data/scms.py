"""
SCMS Delivery History - the real-world pharmaceutical supply chain dataset.

Source
------
USAID **Supply Chain Management System (SCMS)** delivery history: 10,324 real
shipments of antiretroviral (ARV), HIV rapid-test, malaria and anti-malarial
commodities to 43 developing countries between 2006 and 2015. Published by USAID
as open data and mirrored on Kaggle as *Supply Chain Shipment Pricing Data*.

This is genuine operational data from a real health supply chain, and it is the
backbone of every procurement, vendor and logistics metric in the platform. The
simulated digital twin (``src/data/generator.py``) is retained only for the two
domains this dataset does not cover - per-batch storage telemetry and inventory
snapshots - and the dashboard labels which is which.

What makes it valuable
----------------------
* **A real procurement funnel.** Four milestone dates per line item: price quote
  sent to client, purchase order sent to vendor, scheduled delivery, delivery to
  client. Real lead times and real slippage.
* **Real vendor and manufacturing performance.** 73 vendors and 88 manufacturing
  sites, so supplier scorecards are measured rather than invented.
* **Real logistics economics.** Freight cost, weight, pack price and unit price
  across four transport modes.
* **Real messiness.** The defects here were not injected by anyone - they are what
  a live ERP export actually looks like, which makes the data quality module a
  genuine test rather than a demonstration.

The real defects this module has to handle
------------------------------------------
==================================  ==========================================
``PQ First Sent to Client Date``    ``Pre-PQ Process`` (2,476), ``Date Not
                                    Captured`` (205) as literal date values
``PO Sent to Vendor Date``          ``N/A - From RDC`` (5,404) - legitimately
                                    absent: stock drawn from a regional
                                    distribution centre never had a vendor PO
``Freight Cost (USD)``              ``Freight Included in Commodity Cost``
                                    (1,442), ``Invoiced Separately`` (239), and
                                    cross-references such as
                                    ``See DN-304 (ID#:10589)``
``Weight (Kilograms)``              ``Weight Captured Separately`` (1,507) plus
                                    the same cross-reference pattern
``Shipment Mode``                   360 nulls
``Dosage``                          1,736 nulls
Mixed date formats                  ``2-Jun-06`` in some columns, ``9/11/14`` in
                                    others, within the same file
==================================  ==========================================

The important judgement is that these are **not all the same kind of problem**.
``N/A - From RDC`` is not missing data - it is a *structurally* absent value that
correctly describes a different fulfilment path. Imputing it would invent a
purchase order that never existed. This module preserves that distinction by
recording a reason code alongside every coerced value, so downstream analysis can
exclude structural absences rather than silently treating them as gaps.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

#: Public mirror used by ``download_scms`` when the file is not present locally.
SCMS_SOURCE_URL = (
    "https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/"
    "master/SCMS_Delivery_History_Dataset.csv"
)

#: Literal strings that appear in date columns instead of dates, and what they mean.
DATE_SENTINELS: dict[str, str] = {
    "Pre-PQ Process": "structural: procured before the price-quote process existed",
    "Date Not Captured": "missing: milestone genuinely not recorded",
    "N/A - From RDC": "structural: fulfilled from a regional distribution centre, no vendor PO",
}

#: Literal strings that appear in numeric cost/weight columns.
NUMERIC_SENTINELS: dict[str, str] = {
    "Freight Included in Commodity Cost": "structural: freight bundled into commodity price",
    "Invoiced Separately": "structural: billed on a separate invoice",
    "Weight Captured Separately": "structural: weight recorded against a different line",
}

#: Cross-references such as "See DN-304 (ID#:10589)" - the value lives on another row.
_CROSS_REFERENCE = re.compile(r"^See\s+\S+", flags=re.IGNORECASE)

#: 43 destination countries grouped into analysis regions.
COUNTRY_REGION: dict[str, str] = {
    # Southern Africa
    "South Africa": "Southern Africa", "Zimbabwe": "Southern Africa",
    "Zambia": "Southern Africa", "Mozambique": "Southern Africa",
    "Namibia": "Southern Africa", "Botswana": "Southern Africa",
    "Lesotho": "Southern Africa", "Swaziland": "Southern Africa",
    "Malawi": "Southern Africa", "Angola": "Southern Africa",
    # East Africa
    "Kenya": "East Africa", "Tanzania": "East Africa", "Uganda": "East Africa",
    "Ethiopia": "East Africa", "Rwanda": "East Africa", "Burundi": "East Africa",
    "South Sudan": "East Africa", "Sudan": "East Africa",
    # West and Central Africa
    "Nigeria": "West & Central Africa", "Ghana": "West & Central Africa",
    "Côte d'Ivoire": "West & Central Africa", "Cameroon": "West & Central Africa",
    "Congo, DRC": "West & Central Africa", "Benin": "West & Central Africa",
    "Burkina Faso": "West & Central Africa", "Mali": "West & Central Africa",
    "Senegal": "West & Central Africa", "Guinea": "West & Central Africa",
    "Liberia": "West & Central Africa", "Sierra Leone": "West & Central Africa",
    "Togo": "West & Central Africa",
    # Latin America and Caribbean
    "Haiti": "Latin America & Caribbean", "Guyana": "Latin America & Caribbean",
    "Guatemala": "Latin America & Caribbean", "Belize": "Latin America & Caribbean",
    "Dominican Republic": "Latin America & Caribbean",
    # Asia, Central Asia and Middle East
    "Vietnam": "Asia", "Pakistan": "Asia", "Afghanistan": "Asia",
    "Kazakhstan": "Central Asia", "Kyrgyzstan": "Central Asia",
    "Lebanon": "Middle East & North Africa", "Libya": "Middle East & North Africa",
}

#: Human-readable product group names (the raw file uses commodity codes).
PRODUCT_GROUP_NAMES: dict[str, str] = {
    "ARV": "Antiretroviral (ARV)",
    "HRDT": "HIV Rapid Diagnostic Test",
    "ACT": "Artemisinin Combination Therapy",
    "MRDT": "Malaria Rapid Diagnostic Test",
    "ANTM": "Anti-malarial",
}

DATE_COLUMNS = [
    "PQ First Sent to Client Date", "PO Sent to Vendor Date",
    "Scheduled Delivery Date", "Delivered to Client Date", "Delivery Recorded Date",
]

#: Tidy snake_case names for the columns the platform actually uses.
RENAME_MAP: dict[str, str] = {
    "ID": "shipment_id", "Project Code": "project_code", "PQ #": "pq_number",
    "PO / SO #": "po_number", "ASN/DN #": "asn_number", "Country": "country",
    "Managed By": "managed_by", "Fulfill Via": "fulfil_via",
    "Vendor INCO Term": "inco_term", "Shipment Mode": "shipment_mode",
    "Product Group": "product_group", "Sub Classification": "sub_classification",
    "Vendor": "vendor", "Item Description": "item_description",
    "Molecule/Test Type": "molecule", "Brand": "brand", "Dosage": "dosage",
    "Dosage Form": "dosage_form", "Unit of Measure (Per Pack)": "units_per_pack",
    "Line Item Quantity": "quantity", "Line Item Value": "line_value_usd",
    "Pack Price": "pack_price_usd", "Unit Price": "unit_price_usd",
    "Manufacturing Site": "manufacturing_site",
    "First Line Designation": "first_line_designation",
    "Line Item Insurance (USD)": "insurance_usd",
    "PQ First Sent to Client Date": "date_pq_sent",
    "PO Sent to Vendor Date": "date_po_sent",
    "Scheduled Delivery Date": "date_scheduled",
    "Delivered to Client Date": "date_delivered",
    "Delivery Recorded Date": "date_recorded",
}


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def scms_path():
    """Absolute path of the raw SCMS CSV inside ``data/external``."""
    return resolve_path(get_config().paths.data_external) / "SCMS_Delivery_History_Dataset.csv"


def download_scms(force: bool = False):
    """Fetch the SCMS CSV from the public mirror if it is not already present.

    Parameters
    ----------
    force
        Re-download even when the file exists.

    Returns
    -------
    pathlib.Path
        Location of the CSV.
    """
    path = scms_path()
    if path.exists() and not force:
        return path

    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading SCMS delivery history from %s", SCMS_SOURCE_URL)
    urllib.request.urlretrieve(SCMS_SOURCE_URL, path)
    log.info("Saved SCMS dataset to %s (%.1f MB)", path, path.stat().st_size / 1_048_576)
    return path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_dates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse a mixed-format date column, classifying every unparseable value.

    The file mixes ``2-Jun-06`` and ``9/11/14`` styles *and* uses literal strings
    where a date is unknown or does not apply. Returns the parsed dates alongside
    a reason code, so a caller can tell a genuinely missing milestone from one
    that structurally never existed.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        ``(parsed_dates, reason)`` where reason is ``"parsed"``, ``"structural"``,
        ``"missing"`` or ``"unparseable"``.
    """
    raw = series.astype("string").str.strip()
    parsed = pd.to_datetime(raw, format="mixed", dayfirst=True, errors="coerce")

    reason = pd.Series("parsed", index=series.index, dtype="object")
    unparsed = parsed.isna()
    for sentinel, description in DATE_SENTINELS.items():
        hit = unparsed & raw.eq(sentinel)
        reason[hit] = "structural" if description.startswith("structural") else "missing"
    reason[unparsed & ~reason.isin(["structural", "missing"])] = "unparseable"
    reason[raw.isna()] = "missing"
    return parsed, reason


def _parse_numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse a numeric column that also carries explanatory text values.

    ``Freight Cost (USD)`` and ``Weight (Kilograms)`` mix real numbers with
    business statements (``Freight Included in Commodity Cost``) and pointers to
    other rows (``See DN-304 (ID#:10589)``). Each gets its own reason code rather
    than collapsing to a single "missing".

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        ``(values, reason)`` where reason is ``"parsed"``, ``"structural"``,
        ``"cross_reference"`` or ``"missing"``.
    """
    raw = series.astype("string").str.strip()
    # Strip thousands separators and stray currency symbols before coercion.
    cleaned = raw.str.replace(r"[,$]", "", regex=True)
    # Cast to plain float64 rather than the nullable Float64 that a "string"
    # input produces: pd.NA propagates into np.where as an ambiguous truth value,
    # whereas np.nan compares cleanly.
    values = pd.to_numeric(cleaned, errors="coerce").astype("float64")

    reason = pd.Series("parsed", index=series.index, dtype="object")
    unparsed = values.isna()
    for sentinel in NUMERIC_SENTINELS:
        reason[unparsed & raw.eq(sentinel)] = "structural"
    is_reference = unparsed & raw.str.match(_CROSS_REFERENCE).fillna(False)
    reason[is_reference] = "cross_reference"
    reason[unparsed & ~reason.isin(["structural", "cross_reference"])] = "missing"
    reason[raw.isna()] = "missing"
    return values, reason


# ---------------------------------------------------------------------------
# Load and clean
# ---------------------------------------------------------------------------
def load_scms_raw() -> pd.DataFrame:
    """Load the SCMS CSV exactly as published - the bronze layer.

    No parsing, no coercion. This is what the Data Quality page profiles.
    """
    path = download_scms()
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    log.debug("Loaded raw SCMS (%d rows x %d cols)", len(frame), frame.shape[1])
    return frame


@lru_cache(maxsize=1)
def load_scms() -> pd.DataFrame:
    """Load and clean SCMS into the analytics-ready silver layer.

    Cleaning steps, in order:

    1. Rename to snake_case.
    2. Parse the five milestone dates, classifying every unparseable value as
       structural, missing or unparseable.
    3. Parse freight cost and weight, distinguishing real numbers from business
       statements and cross-references.
    4. Trim whitespace and normalise the categorical columns.
    5. Map country to region and commodity code to a readable product name.
    6. Derive the delivery, lead-time and cost-efficiency metrics.

    Returns
    -------
    pandas.DataFrame
        One row per shipment line item, with reason-coded provenance columns
        (``*_reason``) preserved alongside the parsed values.
    """
    raw = load_scms_raw()
    df = raw.rename(columns=RENAME_MAP).copy()

    # --- Dates, with provenance -------------------------------------------
    for original in DATE_COLUMNS:
        column = RENAME_MAP[original]
        parsed, reason = _parse_dates(df[column])
        df[column] = parsed
        df[f"{column}_reason"] = reason

    # --- Cost and weight, with provenance ---------------------------------
    freight, freight_reason = _parse_numeric(raw["Freight Cost (USD)"])
    df["freight_cost_usd"] = freight
    df["freight_cost_reason"] = freight_reason

    weight, weight_reason = _parse_numeric(raw["Weight (Kilograms)"])
    df["weight_kg"] = weight
    df["weight_reason"] = weight_reason

    # --- Categorical tidy-up ----------------------------------------------
    for column in ("country", "vendor", "manufacturing_site", "shipment_mode",
                   "product_group", "sub_classification", "dosage_form", "brand",
                   "molecule", "managed_by", "fulfil_via", "inco_term"):
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    # 360 shipment modes are blank. Mode is a recorded operational fact, so it is
    # labelled Unknown rather than imputed - guessing it would corrupt the
    # mode-level service and cost comparisons that depend on it.
    df["shipment_mode"] = df["shipment_mode"].fillna("Unknown")

    df["region"] = df["country"].map(COUNTRY_REGION).fillna("Other")
    df["product_group_name"] = df["product_group"].map(PRODUCT_GROUP_NAMES).fillna(
        df["product_group"])
    df["is_from_rdc"] = (df["fulfil_via"] == "From RDC").astype(int)

    # --- Derived delivery metrics -----------------------------------------
    df["delivery_delay_days"] = (df["date_delivered"] - df["date_scheduled"]).dt.days
    df["is_late"] = (df["delivery_delay_days"] > 0).astype("Int64")
    df["is_late"] = df["is_late"].where(df["delivery_delay_days"].notna())

    # Lead times. `date_po_sent` is absent for RDC fulfilment by design, so the
    # vendor lead time is genuinely undefined for those rows rather than zero.
    df["vendor_lead_time_days"] = (df["date_delivered"] - df["date_po_sent"]).dt.days
    df["total_lead_time_days"] = (df["date_delivered"] - df["date_pq_sent"]).dt.days
    df["recording_lag_days"] = (df["date_recorded"] - df["date_delivered"]).dt.days

    # The *planned* lead time, known at order time. This is the only lead-time
    # measure safe to use as a feature when predicting late delivery - every
    # other one is derived from `date_delivered` and would leak the answer.
    df["scheduled_lead_time_days"] = (df["date_scheduled"] - df["date_pq_sent"]).dt.days

    # --- Derived economics -------------------------------------------------
    df["freight_pct_of_value"] = np.where(
        df["line_value_usd"] > 0,
        100 * df["freight_cost_usd"] / df["line_value_usd"], np.nan)
    df["freight_cost_per_kg"] = np.where(
        df["weight_kg"] > 0, df["freight_cost_usd"] / df["weight_kg"], np.nan)
    df["packs_ordered"] = np.where(
        df["units_per_pack"] > 0, df["quantity"] / df["units_per_pack"], np.nan)

    # --- Calendar keys -----------------------------------------------------
    df["delivery_year"] = df["date_delivered"].dt.year
    df["delivery_month"] = df["date_delivered"].dt.to_period("M").astype(str)
    df["delivery_quarter"] = df["date_delivered"].dt.to_period("Q").astype(str)

    log.info(
        "Loaded SCMS: %d shipments | %d countries | %d vendors | on-time %.1f%% | "
        "%s to %s",
        len(df), df["country"].nunique(), df["vendor"].nunique(),
        100 * (1 - df["is_late"].mean()),
        df["date_delivered"].min().date(), df["date_delivered"].max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Provenance reporting
# ---------------------------------------------------------------------------
def parsing_report(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarise how every ambiguous field was resolved.

    This is the audit trail for the real dataset: how many values parsed cleanly,
    how many were structurally absent (and therefore correctly excluded rather
    than imputed), and how many were genuinely missing.

    Returns
    -------
    pandas.DataFrame
        Columns ``field``, ``parsed``, ``structural``, ``missing``,
        ``cross_reference``, ``unparseable``, ``parsed_pct``.
    """
    data = load_scms() if df is None else df
    rows: list[dict] = []
    for column in [c for c in data.columns if c.endswith("_reason")]:
        counts = data[column].value_counts()
        field = column.removesuffix("_reason")
        total = int(counts.sum())
        rows.append({
            "field": field,
            "parsed": int(counts.get("parsed", 0)),
            "structural": int(counts.get("structural", 0)),
            "missing": int(counts.get("missing", 0)),
            "cross_reference": int(counts.get("cross_reference", 0)),
            "unparseable": int(counts.get("unparseable", 0)),
            "parsed_pct": round(100 * counts.get("parsed", 0) / total, 2) if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values("parsed_pct").reset_index(drop=True)


def scms_provenance() -> dict:
    """Source and licensing metadata, surfaced in the dashboard."""
    return {
        "name": "SCMS Delivery History Dataset",
        "publisher": "United States Agency for International Development (USAID)",
        "programme": "Supply Chain Management System (SCMS)",
        "description": (
            "Delivery history for antiretroviral, HIV rapid-test, malaria and "
            "anti-malarial commodities shipped to USAID-supported countries."
        ),
        "records": 10324,
        "coverage": "2006-2015, 43 destination countries",
        "licence": "Public domain (US Government open data)",
        "kaggle": "https://www.kaggle.com/datasets/sawandikirby/supply-chain-shipment-pricing-data",
        "source_url": SCMS_SOURCE_URL,
    }


__all__ = [
    "load_scms", "load_scms_raw", "download_scms", "scms_path",
    "parsing_report", "scms_provenance",
    "COUNTRY_REGION", "PRODUCT_GROUP_NAMES", "DATE_SENTINELS", "NUMERIC_SENTINELS",
]
