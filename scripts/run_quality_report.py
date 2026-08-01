#!/usr/bin/env python
"""
Run the data quality audit and export a multi-sheet Excel report.

Profiles the raw (bronze) extract, scores five quality dimensions per table,
records the uplift delivered by the cleaning layer, and writes everything to
``reports/data_quality_report.xlsx`` for circulation outside the dashboard.

Usage
-----
    python scripts/run_quality_report.py
    python scripts/run_quality_report.py --dataset batches
    python scripts/run_quality_report.py --no-excel      # console output only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import ensure_directories, get_config, resolve_path  # noqa: E402
from src.data import loader  # noqa: E402
from src.data.cleaning import remediation_summary  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.quality import assessment as dq  # noqa: E402

log = get_logger("scripts.run_quality_report")

DATASETS = ["batches", "shipments", "inventory", "demand", "drug200"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS,
                        help="Audit a single dataset instead of all of them.")
    parser.add_argument("--no-excel", action="store_true",
                        help="Print to console without writing the workbook.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_config()
    started = time.perf_counter()
    ensure_directories()

    targets = [args.dataset] if args.dataset else DATASETS

    print(f"\n{cfg.project.name} v{cfg.project.version}")
    print("Data quality audit - profiling the raw (bronze) extract\n")

    scoreboard = dq.assess_all(targets, raw=True)
    print("=" * 96)
    print("SCOREBOARD (worst first)")
    print("=" * 96)
    print(scoreboard.to_string(index=False))

    uplift = dq.quality_uplift(targets)
    print("\n" + "=" * 96)
    print("CLEANING UPLIFT (bronze -> silver)")
    print("=" * 96)
    print(uplift[["dataset", "raw_score", "raw_grade", "clean_score",
                  "clean_grade", "uplift"]].to_string(index=False))

    remediation = loader.remediation_log()
    summary = remediation_summary(remediation)
    print("\n" + "=" * 96)
    print("REMEDIATION ACTIONS APPLIED")
    print("=" * 96)
    print(summary.to_string(index=False) if len(summary) else "  none")

    injected = loader.injected_defect_log()
    if len(injected):
        print("\n" + "=" * 96)
        print("DEFECTS INJECTED BY THE GENERATOR (expected findings)")
        print("=" * 96)
        print(injected.sort_values("rows_affected", ascending=False).to_string(index=False))

    if args.no_excel:
        print(f"\nAudit complete in {time.perf_counter() - started:.1f}s\n")
        return 0

    # --- Excel export -------------------------------------------------------
    output = resolve_path(cfg.paths.reports) / "data_quality_report.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        scoreboard.to_excel(writer, sheet_name="Scoreboard", index=False)
        uplift.to_excel(writer, sheet_name="Cleaning Uplift", index=False)
        summary.to_excel(writer, sheet_name="Remediation", index=False)
        if len(injected):
            injected.to_excel(writer, sheet_name="Injected Defects", index=False)

        # One detail sheet per dataset. Sheet names are capped at 31 characters
        # by the xlsx format, so the dataset name is truncated defensively.
        for name in targets:
            report = dq.assess_dataset(name, raw=True)
            stem = name[:18]
            report["missing"].to_excel(writer, sheet_name=f"{stem}_missing", index=False)
            report["validity"].to_excel(writer, sheet_name=f"{stem}_validity", index=False)
            report["outliers"].to_excel(writer, sheet_name=f"{stem}_outliers", index=False)
            report["consistency"].to_excel(writer, sheet_name=f"{stem}_consist", index=False)
            report["summary"].to_excel(writer, sheet_name=f"{stem}_stats", index=False)
            report["recommendations"].to_excel(writer, sheet_name=f"{stem}_recs", index=False)

    log.info("Wrote Excel report to %s", output)
    print(f"\nAudit complete in {time.perf_counter() - started:.1f}s")
    print(f"Excel report: {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
