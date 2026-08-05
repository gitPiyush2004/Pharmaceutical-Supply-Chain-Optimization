#!/usr/bin/env python
"""
Run the data quality audit and export a multi-sheet Excel report.

Profiles each dataset as published, scores five quality dimensions, measures what
type-aware parsing changes, and writes everything to
``reports/data_quality_report.xlsx`` for circulation outside the dashboard.

The headline result is deliberately unflattering: parsing SCMS *lowers* its generic
score, because replacing unparseable text with honest nulls costs completeness. That
is the point of the report - a generic profiler grades this file an A while 40% of
its freight costs are unusable.

Usage
-----
    python scripts/run_quality_report.py
    python scripts/run_quality_report.py --dataset scms
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
from src.data.scms import parsing_report  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.quality import assessment as dq  # noqa: E402

log = get_logger("scripts.run_quality_report")

DATASETS = ["scms", "drug200"]


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
    print("Data quality audit - both real datasets, as published\n")

    scoreboard = dq.assess_all(targets, raw=True)
    print("=" * 96)
    print("SCOREBOARD (worst first)")
    print("=" * 96)
    print(scoreboard.to_string(index=False))

    uplift = dq.quality_uplift(targets)
    print("\n" + "=" * 96)
    print("EFFECT OF TYPE-AWARE PARSING (as published -> interpreted)")
    print("=" * 96)
    print(uplift[["dataset", "raw_score", "raw_grade", "clean_score",
                  "clean_grade", "uplift"]].to_string(index=False))
    print("\nNote: a NEGATIVE uplift is expected on scms. Parsing replaces")
    print("unparseable text with honest nulls, so generic completeness falls.")
    print("The generic score rewards a file for holding non-null garbage, which")
    print("is precisely the limitation this report exists to expose.")

    parsing = parsing_report()
    print("\n" + "=" * 96)
    print("SCMS FIELD USABILITY (what a completeness check cannot see)")
    print("=" * 96)
    print(parsing.to_string(index=False))

    if args.no_excel:
        print(f"\nAudit complete in {time.perf_counter() - started:.1f}s\n")
        return 0

    # --- Excel export -------------------------------------------------------
    output = resolve_path(cfg.paths.reports) / "data_quality_report.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        scoreboard.to_excel(writer, sheet_name="Scoreboard", index=False)
        uplift.to_excel(writer, sheet_name="Parsing Effect", index=False)
        parsing.to_excel(writer, sheet_name="SCMS Field Usability", index=False)

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
