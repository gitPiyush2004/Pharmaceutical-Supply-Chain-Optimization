"""
Cleaning layer: promotes the raw (bronze) extract to the analytics-ready (silver) layer.

The platform separates two layers deliberately, the way a real data pipeline does:

======  ====================  ===================================================
Layer   Location              Contents
======  ====================  ===================================================
Bronze  ``data/raw``          What source systems actually hand over: sensor
                              dropouts, ERP double-postings, free-text region
                              spellings, sign errors, trailing whitespace.
Silver  in-memory / cached    Deduplicated, canonicalised, imputed, range-checked.
                              Everything the analytics modules read.
======  ====================  ===================================================

Every transformation is recorded in a remediation log, so the Data Quality page
can show *what was wrong* next to *what was done about it* - the audit trail a
regulated (GxP) environment expects.

Remediation strategy, in the order applied:

1. **Whitespace and casing** - strip and collapse text fields.
2. **Master-data repair** - restore ``supplier_reliability`` from the supplier
   dimension. Preferred over imputation: the true value is knowable.
3. **Category canonicalisation** - map free-text region spellings to the
   controlled vocabulary.
4. **Deduplication** - drop repeated business keys, keeping the first record.
5. **Range checks** - impossible values (negative durations, potency above
   label-claim tolerance) are treated as unrecoverable and imputed.
6. **Imputation** - group-wise median for numerics, mode for categoricals.
   Grouping matters: imputing a cold-chain storage temperature with the
   portfolio median would fabricate a 20 degC excursion.

Example
-------
>>> from src.data.cleaning import clean_table
>>> clean, log = clean_table("batches", raw_batches)
>>> log[["step", "column", "rows_affected"]].head()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config
from src.logger import get_logger

log = get_logger(__name__)

#: Business key used to detect duplicate records, per table.
BUSINESS_KEYS: dict[str, list[str]] = {
    "batches": ["batch_id"],
    "shipments": ["shipment_id"],
    "inventory": ["snapshot_month", "warehouse_id", "drug_code"],
    "demand": ["year_month", "drug_code", "region"],
}

#: Numeric columns imputed with a *group* median, and the grouping key.
#: The grouping is the important part - see the module docstring.
GROUPED_IMPUTATION: dict[str, dict[str, list[str]]] = {
    "batches": {
        "storage_temp_c": ["drug_code"],
        "storage_humidity_pct": ["region"],
        "storage_duration_days": ["drug_code"],
        "potency_pct": ["drug_code"],
    },
    "shipments": {"actual_transit_days": ["transport_mode"]},
    "inventory": {"units_on_hand": ["warehouse_id", "drug_code"]},
}

#: Physically or contractually impossible ranges: (column, minimum, maximum).
#: Values outside the range are unrecoverable and get imputed.
RANGE_RULES: dict[str, list[tuple[str, float, float]]] = {
    "batches": [
        ("storage_duration_days", 0.0, 3650.0),
        # ICH label-claim tolerance is 95-105%; an assay above 100% can be
        # legitimate, but above 100 in this extract is a transcription error.
        ("potency_pct", 60.0, 100.0),
        ("storage_temp_c", -30.0, 60.0),
        ("storage_humidity_pct", 0.0, 100.0),
    ],
    "shipments": [("actual_transit_days", 0.0, 200.0)],
    "inventory": [("units_on_hand", 0.0, 1e9)],
}

#: Text columns that get whitespace/casing normalisation.
TEXT_COLUMNS: dict[str, list[str]] = {
    "batches": ["supplier_name", "region", "drug_code", "brand_name", "qa_result"],
    "shipments": ["carrier", "region", "transport_mode", "leg", "supplier_name"],
    "inventory": ["region", "warehouse_name", "drug_code", "brand_name"],
    "demand": ["region", "drug_code", "brand_name"],
}


def _canonical_regions() -> dict[str, str]:
    """Alias -> canonical region map, matched case-insensitively after stripping."""
    aliases = get_config().generation.quality_issues.region_aliases
    return {str(k).strip().lower(): v for k, v in aliases.items()}


def clean_table(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean one raw table and return it alongside its remediation log.

    Parameters
    ----------
    name
        Logical table name (``batches``, ``shipments``, ``inventory``, ``demand``,
        or a dimension table, which passes through with text normalisation only).
    frame
        The raw extract.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The cleaned frame, and a log with columns ``step``, ``column``,
        ``action``, ``rows_affected``.
    """
    df = frame.copy()
    actions: list[dict] = []

    def _record(step: str, column: str, action: str, count: int) -> None:
        if count:
            actions.append({"step": step, "column": column,
                            "action": action, "rows_affected": int(count)})

    # --- 1. Whitespace and casing ------------------------------------------
    for column in TEXT_COLUMNS.get(name, []):
        if column not in df.columns:
            continue
        original = df[column].astype("string")
        stripped = original.str.strip().str.replace(r"\s+", " ", regex=True)
        changed = (original != stripped) & original.notna()
        if changed.any():
            df[column] = stripped
            _record("whitespace", column, "stripped and collapsed whitespace", changed.sum())
        else:
            df[column] = stripped

    # --- 2. Master-data repair ---------------------------------------------
    # supplier_reliability is an attribute of the supplier, not of the batch,
    # so a missing value can be *restored* rather than guessed.
    if name == "batches" and "supplier_reliability" in df.columns:
        missing = df["supplier_reliability"].isna()
        if missing.any():
            from src.data import loader  # local import avoids a circular import

            lookup = loader.load_table("suppliers", raw=True).set_index("supplier_id")["reliability"]
            df.loc[missing, "supplier_reliability"] = df.loc[missing, "supplier_id"].map(lookup)
            repaired = missing.sum() - df["supplier_reliability"].isna().sum()
            _record("master_data_repair", "supplier_reliability",
                    "restored from supplier dimension", repaired)

    # --- 3. Category canonicalisation --------------------------------------
    if "region" in df.columns:
        alias_map = _canonical_regions()
        keys = df["region"].astype("string").str.strip().str.lower()
        mapped = keys.map(alias_map)
        changed = mapped.notna() & (mapped != df["region"])
        if changed.any():
            df.loc[changed, "region"] = mapped[changed]
            _record("canonicalisation", "region",
                    "mapped free-text spellings to controlled vocabulary", changed.sum())

    # --- 4. Deduplication ---------------------------------------------------
    keys = BUSINESS_KEYS.get(name)
    if keys and all(k in df.columns for k in keys):
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
        _record("deduplication", "+".join(keys),
                "dropped repeated business keys", before - len(df))

    # --- 5. Range checks ----------------------------------------------------
    # Out-of-range values are set to NaN here and imputed in step 6, so the
    # log distinguishes "was impossible" from "was missing at source".
    for column, low, high in RANGE_RULES.get(name, []):
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        violations = numeric.notna() & ((numeric < low) | (numeric > high))
        if violations.any():
            numeric[violations] = np.nan
            df[column] = numeric
            _record("range_check", column,
                    f"nulled values outside [{low:g}, {high:g}] as unrecoverable",
                    violations.sum())
        else:
            df[column] = numeric

    # --- 6. Imputation ------------------------------------------------------
    for column, group_keys in GROUPED_IMPUTATION.get(name, {}).items():
        if column not in df.columns:
            continue
        missing = df[column].isna()
        if not missing.any():
            continue
        usable_keys = [k for k in group_keys if k in df.columns]
        if usable_keys:
            df[column] = df.groupby(usable_keys)[column].transform(
                lambda s: s.fillna(s.median()))
        # Any group that was entirely missing falls back to the global median.
        df[column] = df[column].fillna(df[column].median())
        _record("imputation", column,
                f"median imputed by {'+'.join(usable_keys) or 'global'}", missing.sum())

    # Remaining categorical gaps take the column mode.
    for column in df.columns:
        if df[column].dtype == object or str(df[column].dtype) == "string":
            missing = df[column].isna()
            if missing.any():
                mode = df[column].mode()
                if len(mode):
                    df[column] = df[column].fillna(mode.iloc[0])
                    _record("imputation", column, "mode imputed", missing.sum())

    remediation = pd.DataFrame(
        actions, columns=["step", "column", "action", "rows_affected"])
    if len(remediation):
        log.info("Cleaned %-10s | %d remediation actions | %d rows touched",
                 name, len(remediation), int(remediation["rows_affected"].sum()))
    return df, remediation


def clean_all(raw_tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Clean every table and return one consolidated remediation log.

    Parameters
    ----------
    raw_tables
        Mapping of logical table name to raw frame.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame]
        Cleaned tables, and a combined log with a leading ``table`` column.
    """
    cleaned: dict[str, pd.DataFrame] = {}
    logs: list[pd.DataFrame] = []
    for name, frame in raw_tables.items():
        result, remediation = clean_table(name, frame)
        cleaned[name] = result
        if len(remediation):
            remediation.insert(0, "table", name)
            logs.append(remediation)

    combined = (pd.concat(logs, ignore_index=True) if logs
                else pd.DataFrame(columns=["table", "step", "column", "action", "rows_affected"]))
    return cleaned, combined


def remediation_summary(combined_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a remediation log by table and step, for dashboard display."""
    if not len(combined_log):
        return pd.DataFrame(columns=["table", "step", "columns_affected", "rows_affected"])
    return (combined_log.groupby(["table", "step"], as_index=False)
            .agg(columns_affected=("column", "nunique"),
                 rows_affected=("rows_affected", "sum"))
            .sort_values("rows_affected", ascending=False)
            .reset_index(drop=True))


__all__ = ["clean_table", "clean_all", "remediation_summary",
           "BUSINESS_KEYS", "RANGE_RULES", "GROUPED_IMPUTATION"]
