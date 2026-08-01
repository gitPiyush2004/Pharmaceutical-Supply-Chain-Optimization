"""
SQL analytics layer.

The CSV star schema is materialised into a local SQLite warehouse so the
platform can answer questions in SQL as well as in pandas. This mirrors how the
work would actually be split in a consulting engagement: heavy set-based
aggregation pushed down to the database, statistical modelling done in Python.

The named queries in :data:`ANALYTICAL_QUERIES` are the ones surfaced in the
dashboard's SQL explorer, so a reviewer can see the exact logic behind every
headline number.

Example
-------
>>> from src.data.database import build_warehouse, run_query, ANALYTICAL_QUERIES
>>> build_warehouse()
>>> run_query(ANALYTICAL_QUERIES["supplier_scorecard"]).head()
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import ensure_directories, get_config, resolve_path
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)

# Logical table name -> physical SQL table name.
TABLE_MAP: dict[str, str] = {
    "drugs": "dim_drug",
    "suppliers": "dim_supplier",
    "warehouses": "dim_warehouse",
    "batches": "fact_batch",
    "shipments": "fact_shipment",
    "inventory": "fact_inventory",
    "demand": "fact_demand",
}

# Indexes on the join and filter keys the analytical queries actually use.
INDEX_DDL: list[str] = [
    "CREATE INDEX IF NOT EXISTS ix_batch_drug     ON fact_batch(drug_code)",
    "CREATE INDEX IF NOT EXISTS ix_batch_supplier ON fact_batch(supplier_id)",
    "CREATE INDEX IF NOT EXISTS ix_batch_region   ON fact_batch(region)",
    "CREATE INDEX IF NOT EXISTS ix_batch_month    ON fact_batch(year_month)",
    "CREATE INDEX IF NOT EXISTS ix_ship_batch     ON fact_shipment(batch_id)",
    "CREATE INDEX IF NOT EXISTS ix_ship_region    ON fact_shipment(region)",
    "CREATE INDEX IF NOT EXISTS ix_inv_wh         ON fact_inventory(warehouse_id)",
    "CREATE INDEX IF NOT EXISTS ix_demand_month   ON fact_demand(year_month)",
]


def database_path() -> Path:
    """Absolute path of the SQLite warehouse file."""
    return resolve_path(get_config().paths.database)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection, closing it on exit."""
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def build_warehouse(force: bool = False) -> Path:
    """Materialise every CSV table into SQLite and create indexes.

    Parameters
    ----------
    force
        Rebuild even when the database file already exists.

    Returns
    -------
    Path
        Location of the SQLite file.
    """
    ensure_directories()
    path = database_path()
    if path.exists() and not force:
        log.debug("SQLite warehouse already present at %s", path)
        return path

    loader.ensure_datasets()
    with connect() as conn:
        for logical, physical in TABLE_MAP.items():
            frame = loader.load_table(logical)
            # SQLite has no native datetime type; ISO strings sort correctly.
            out = frame.copy()
            for column in out.select_dtypes(include=["datetime64[ns]"]).columns:
                out[column] = out[column].dt.strftime("%Y-%m-%d")
            out.to_sql(physical, conn, if_exists="replace", index=False)
            log.info("Materialised %-10s -> %-14s (%d rows)", logical, physical, len(out))

        for ddl in INDEX_DDL:
            conn.execute(ddl)
        conn.commit()

    log.info("SQLite warehouse built at %s", path)
    return path


def run_query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Execute a read-only SQL statement and return the result as a DataFrame."""
    build_warehouse()  # no-op when the file already exists
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Named analytical queries (surfaced in the dashboard's SQL explorer)
# ---------------------------------------------------------------------------
ANALYTICAL_QUERIES: dict[str, str] = {
    "funnel_conversion": """
        -- End-to-end funnel: units surviving each supply chain stage
        SELECT
            SUM(units_procured)     AS procurement,
            SUM(units_manufactured) AS manufacturing,
            SUM(units_qa_passed)    AS quality_testing,
            SUM(units_packaged)     AS packaging,
            SUM(units_warehoused)   AS warehouse,
            SUM(units_distributed)  AS distributor,
            SUM(units_delivered)    AS pharmacy,
            SUM(units_dispensed)    AS patient,
            ROUND(100.0 * SUM(units_dispensed) / SUM(units_procured), 2) AS end_to_end_yield_pct
        FROM fact_batch;
    """,
    "supplier_scorecard": """
        -- Supplier performance: QA pass rate, cycle time and value destroyed
        SELECT
            s.supplier_name,
            s.country,
            s.tier,
            COUNT(b.batch_id)                                  AS batches_supplied,
            ROUND(100.0 * AVG(b.qa_pass), 2)                   AS qa_pass_rate_pct,
            ROUND(AVG(b.total_cycle_time_days), 1)             AS avg_cycle_time_days,
            ROUND(100.0 * SUM(b.units_qa_passed)
                  / NULLIF(SUM(b.units_manufactured), 0), 2)   AS qa_unit_yield_pct,
            ROUND(SUM(b.value_lost_usd), 0)                    AS value_lost_usd
        FROM fact_batch b
        JOIN dim_supplier s ON s.supplier_id = b.supplier_id
        GROUP BY s.supplier_name, s.country, s.tier
        ORDER BY qa_pass_rate_pct ASC;
    """,
    "regional_otd": """
        -- On-time delivery and freight spend by region and transport mode
        SELECT
            region,
            transport_mode,
            COUNT(*)                                    AS shipments,
            ROUND(100.0 * (1 - AVG(is_late)), 2)        AS on_time_delivery_pct,
            ROUND(AVG(delay_days), 2)                   AS avg_delay_days,
            ROUND(100.0 * AVG(temperature_excursion), 2) AS excursion_rate_pct,
            ROUND(SUM(freight_cost_usd), 0)             AS freight_spend_usd
        FROM fact_shipment
        GROUP BY region, transport_mode
        HAVING shipments > 30
        ORDER BY on_time_delivery_pct ASC;
    """,
    "bottleneck_stages": """
        -- Average dwell time in each stage, ranked slowest first
        SELECT 'Procurement'       AS stage, ROUND(AVG(delay_procurement_days), 1)       AS avg_days FROM fact_batch
        UNION ALL SELECT 'Manufacturing',     ROUND(AVG(delay_manufacturing_days), 1)     FROM fact_batch
        UNION ALL SELECT 'Quality Testing',   ROUND(AVG(delay_quality_testing_days), 1)   FROM fact_batch
        UNION ALL SELECT 'Packaging',         ROUND(AVG(delay_packaging_days), 1)         FROM fact_batch
        UNION ALL SELECT 'Warehouse',         ROUND(AVG(delay_warehouse_days), 1)         FROM fact_batch
        UNION ALL SELECT 'Distributor',       ROUND(AVG(delay_distributor_days), 1)       FROM fact_batch
        UNION ALL SELECT 'Hospital/Pharmacy', ROUND(AVG(delay_hospital_pharmacy_days), 1) FROM fact_batch
        UNION ALL SELECT 'Patient',           ROUND(AVG(delay_patient_days), 1)           FROM fact_batch
        ORDER BY avg_days DESC;
    """,
    "cold_chain_risk": """
        -- Stability risk profile: cold chain versus ambient products
        SELECT
            CASE WHEN b.is_cold_chain = 1 THEN 'Cold Chain (2-8C)'
                 ELSE 'Controlled Room Temperature' END       AS storage_class,
            COUNT(*)                                          AS batches,
            ROUND(AVG(b.potency_pct), 2)                      AS avg_potency_pct,
            ROUND(100.0 * SUM(CASE WHEN b.potency_pct < 95 THEN 1 ELSE 0 END)
                  / COUNT(*), 2)                              AS out_of_spec_pct,
            ROUND(AVG(b.days_to_expiry_at_delivery), 0)       AS avg_shelf_life_remaining_days,
            ROUND(100.0 * SUM(CASE WHEN b.batch_risk_label = 'High' THEN 1 ELSE 0 END)
                  / COUNT(*), 2)                              AS high_risk_pct
        FROM fact_batch b
        GROUP BY storage_class;
    """,
    "inventory_abc": """
        -- Inventory value concentration by product (input to ABC classification)
        SELECT
            drug_code,
            brand_name,
            ROUND(SUM(inventory_value_usd), 0)   AS inventory_value_usd,
            SUM(units_on_hand)                   AS units_on_hand,
            SUM(units_expiring_soon)             AS units_expiring_soon,
            ROUND(AVG(months_of_supply), 2)      AS avg_months_of_supply
        FROM fact_inventory
        GROUP BY drug_code, brand_name
        ORDER BY inventory_value_usd DESC;
    """,
    "monthly_demand_trend": """
        -- Monthly demand, fulfilment and backorder trend across the network
        SELECT
            year_month,
            SUM(units_demanded)                   AS units_demanded,
            SUM(units_fulfilled)                  AS units_fulfilled,
            SUM(units_backordered)                AS units_backordered,
            ROUND(100.0 * SUM(units_fulfilled)
                  / NULLIF(SUM(units_demanded), 0), 2) AS fill_rate_pct
        FROM fact_demand
        GROUP BY year_month
        ORDER BY year_month;
    """,
    "top_loss_drivers": """
        -- Where value is destroyed: worst drug x region combinations
        SELECT
            b.drug_code,
            b.brand_name,
            b.region,
            COUNT(*)                             AS batches,
            ROUND(SUM(b.value_lost_usd), 0)      AS value_lost_usd,
            ROUND(100.0 * SUM(b.units_lost)
                  / NULLIF(SUM(b.units_procured), 0), 2) AS unit_loss_pct
        FROM fact_batch b
        GROUP BY b.drug_code, b.brand_name, b.region
        ORDER BY value_lost_usd DESC
        LIMIT 15;
    """,
}


def query_catalog() -> pd.DataFrame:
    """Return the named queries as a browsable catalogue."""
    return pd.DataFrame(
        [{"query_name": name,
          "description": sql.strip().splitlines()[0].lstrip("- ").strip()}
         for name, sql in ANALYTICAL_QUERIES.items()]
    )


__all__ = [
    "build_warehouse", "run_query", "connect", "database_path",
    "ANALYTICAL_QUERIES", "query_catalog", "TABLE_MAP",
]
