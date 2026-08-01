#!/usr/bin/env python
"""
Build the platform data layer.

Generates the pharmaceutical supply chain digital twin, writes the CSV star schema
to ``data/raw``, and materialises it into the SQLite analytics warehouse.

Usage
-----
    python scripts/build_dataset.py                  # build everything
    python scripts/build_dataset.py --force          # rebuild even if present
    python scripts/build_dataset.py --no-database    # CSVs only
    python scripts/build_dataset.py --summary        # print a profile afterwards
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, get_config  # noqa: E402
from src.data import loader  # noqa: E402
from src.data.database import build_warehouse, database_path  # noqa: E402
from src.data.generator import generate_all  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger("scripts.build_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the datasets already exist.")
    parser.add_argument("--no-database", action="store_true",
                        help="Skip building the SQLite warehouse.")
    parser.add_argument("--summary", action="store_true",
                        help="Print a profile of the generated tables.")
    return parser.parse_args()


def print_summary(tables: dict) -> None:
    """Print row/column counts and the headline funnel metrics."""
    print("\n" + "=" * 74)
    print("DATA LAYER SUMMARY")
    print("=" * 74)
    for name, frame in tables.items():
        print(f"  {name:<12} {len(frame):>7,} rows  x {frame.shape[1]:>3} cols")

    batches = tables["batches"]
    procured = batches["units_procured"].sum()
    dispensed = batches["units_dispensed"].sum()
    print("-" * 74)
    print(f"  End-to-end yield      {100 * dispensed / procured:>8.2f}%")
    print(f"  QA pass rate          {100 * batches['qa_pass'].mean():>8.2f}%")
    print(f"  Mean cycle time       {batches['total_cycle_time_days'].mean():>8.1f} days")
    print(f"  Mean potency          {batches['potency_pct'].mean():>8.2f}%")
    print(f"  Value lost            ${batches['value_lost_usd'].sum():>12,.0f}")
    print("  Risk mix             ",
          batches["batch_risk_label"].value_counts(normalize=True).round(3).to_dict())
    print("=" * 74)


def main() -> int:
    args = parse_args()
    cfg = get_config()
    started = time.perf_counter()

    print(f"\n{cfg.project.name} v{cfg.project.version}")
    print(f"Building data layer (seed={cfg.project.random_seed}, deterministic)\n")

    ensure_directories()

    if loader.datasets_exist() and not args.force:
        log.info("Datasets already present - loading them. Use --force to rebuild.")
        tables = {name: loader.load_table(name, raw=True)
                  for name in ("drugs", "suppliers", "warehouses", "batches",
                               "shipments", "demand", "inventory")}
    else:
        tables = generate_all(save=True)
        loader.load_table.cache_clear()

    if not args.no_database:
        path = build_warehouse(force=args.force)
        size_mb = path.stat().st_size / 1_048_576
        log.info("SQLite warehouse ready at %s (%.1f MB)", path, size_mb)

    if args.summary:
        print_summary(tables)

    elapsed = time.perf_counter() - started
    print(f"\nData layer built in {elapsed:.1f}s")
    if not args.no_database:
        print(f"SQLite warehouse: {database_path()}")
    print("Next: python scripts/train_models.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
