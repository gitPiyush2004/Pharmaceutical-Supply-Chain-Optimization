"""
Dataset access layer.

Every consumer - analytics modules, the Streamlit app, notebooks and tests -
reads data through this module rather than touching CSV paths directly. That
keeps parsing rules (date columns, dtypes) in one place and gives the whole
platform a single caching strategy.

If a supply chain table is missing, the loader regenerates the full dataset
automatically, so a fresh clone works with no manual ETL step.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

# Columns parsed as datetimes, per table.
_DATE_COLUMNS: dict[str, list[str]] = {
    "batches": [
        "date_procurement", "date_manufacturing", "date_quality", "date_packaging",
        "date_warehouse", "date_distributor", "date_pharmacy", "date_patient", "expiry_date",
    ],
    "shipments": ["ship_date", "delivery_date"],
    "demand": ["date"],
}

# Supply chain tables that the generator produces (drug200 is external input).
_GENERATED_TABLES = ("drugs", "suppliers", "warehouses", "batches",
                     "shipments", "demand", "inventory")


def _dataset_path(name: str) -> Path:
    """Resolve a logical table name to its absolute CSV path."""
    cfg = get_config()
    if name not in cfg.datasets:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(cfg.datasets)}")
    return resolve_path(cfg.datasets[name])


def datasets_exist() -> bool:
    """True when every generated supply chain table is present on disk."""
    return all(_dataset_path(name).exists() for name in _GENERATED_TABLES)


def ensure_datasets(force: bool = False) -> None:
    """Generate the supply chain dataset if it is missing (or if ``force``)."""
    if force or not datasets_exist():
        from src.data.generator import generate_all  # local import avoids a cycle

        log.info("Supply chain dataset not found - generating it now.")
        generate_all(save=True)
        load_table.cache_clear()


@lru_cache(maxsize=32)
def load_table(name: str, raw: bool = False) -> pd.DataFrame:
    """Load a single table by logical name, with dates already parsed.

    Parameters
    ----------
    name
        Logical table name, as declared in ``config.datasets``.
    raw
        When False (default) the table is passed through
        :func:`src.data.cleaning.clean_table` first, giving the analytics-ready
        *silver* layer. When True the untouched *bronze* extract is returned -
        which is what the Data Quality page profiles.

    Returns
    -------
    pd.DataFrame
        The requested table. Results are cached per ``(name, raw)`` pair, so
        repeated calls inside a Streamlit session or notebook are free. Call
        ``load_table.cache_clear()`` after regenerating the data.
    """
    if name in _GENERATED_TABLES:
        ensure_datasets()

    path = _dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}. "
            "Run `python scripts/build_dataset.py` to build the data layer."
        )

    frame = pd.read_csv(path, parse_dates=_DATE_COLUMNS.get(name))
    if raw:
        log.debug("Loaded raw %s (%d rows x %d cols)", name, len(frame), frame.shape[1])
        return frame

    from src.data.cleaning import clean_table  # local import avoids a cycle

    cleaned, _ = clean_table(name, frame)
    log.debug("Loaded clean %s (%d rows x %d cols)", name, len(cleaned), cleaned.shape[1])
    return cleaned


def load_raw_table(name: str) -> pd.DataFrame:
    """Convenience wrapper for the untouched bronze extract of ``name``."""
    return load_table(name, raw=True).copy()


@lru_cache(maxsize=1)
def remediation_log() -> pd.DataFrame:
    """Consolidated record of every cleaning action applied to the raw extract.

    Surfaced on the Data Quality page as the audit trail between the bronze and
    silver layers.
    """
    from src.data.cleaning import clean_all

    raw = {name: load_table(name, raw=True) for name in _GENERATED_TABLES}
    _, combined = clean_all(raw)
    return combined


def injected_defect_log() -> pd.DataFrame:
    """Defects deliberately injected by the generator, if the log is present.

    Lets the dashboard compare *injected* against *detected* defects - a direct
    check that the data quality module works.
    """
    path = _dataset_path("batches").parent / "injected_defect_log.csv"
    if not path.exists():
        return pd.DataFrame(columns=["table", "column", "defect_type", "rows_affected"])
    return pd.read_csv(path)


# --- Convenience accessors -------------------------------------------------
# Thin, self-documenting wrappers. They return copies so a caller mutating a
# frame cannot corrupt the shared cache.

def load_clinical() -> pd.DataFrame:
    """Kaggle drug200 clinical dataset (200 patients, 5 features, 5 drug classes)."""
    return load_table("drug200").copy()


def load_batches() -> pd.DataFrame:
    """Batch funnel fact table - one row per manufactured batch."""
    return load_table("batches").copy()


def load_shipments() -> pd.DataFrame:
    """Shipment fact table - one row per transport leg."""
    return load_table("shipments").copy()


def load_inventory() -> pd.DataFrame:
    """Monthly inventory snapshots per warehouse and drug."""
    return load_table("inventory").copy()


def load_demand() -> pd.DataFrame:
    """Monthly demand history by drug and region."""
    return load_table("demand").copy()


def load_suppliers() -> pd.DataFrame:
    """API supplier dimension."""
    return load_table("suppliers").copy()


def load_warehouses() -> pd.DataFrame:
    """Distribution network dimension."""
    return load_table("warehouses").copy()


def load_drugs() -> pd.DataFrame:
    """Product dimension."""
    return load_table("drugs").copy()


def load_all() -> dict[str, pd.DataFrame]:
    """Load every table at once - convenient for notebooks and the dashboard."""
    ensure_datasets()
    tables = {name: load_table(name).copy() for name in _GENERATED_TABLES}
    tables["clinical"] = load_clinical()
    return tables


__all__ = [
    "load_table", "load_raw_table", "load_all", "ensure_datasets", "datasets_exist",
    "remediation_log", "injected_defect_log",
    "load_clinical", "load_batches", "load_shipments", "load_inventory",
    "load_demand", "load_suppliers", "load_warehouses", "load_drugs",
]
