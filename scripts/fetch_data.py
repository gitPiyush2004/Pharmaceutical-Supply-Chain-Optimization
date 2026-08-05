#!/usr/bin/env python
"""
Download and cache the two external datasets.

Run this once after cloning. ``drug200.csv`` ships with the repository; the other
two are fetched from their public sources and cached under ``data/external``, so
every later run - dashboard, notebook, test suite - works with no network.

Both sources are public and need no authentication. The Indian medicine master is
pulled from raw GitHub rather than its Kaggle mirror for exactly that reason: a
Kaggle download would require an API token, which turns a one-command setup into a
credentials problem.

Usage
-----
    python scripts/fetch_data.py             # fetch anything not already cached
    python scripts/fetch_data.py --force     # re-download even if cached
    python scripts/fetch_data.py --verify    # report what is present, fetch nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, resolve_path  # noqa: E402
from src.data import loader  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger("scripts.fetch_data")

#: Rows each dataset must have once downloaded. A truncated CSV would otherwise
#: pass silently and put quietly wrong numbers on every page.
EXPECTED_ROWS = {
    "drug200": 200,
    "scms": 10_324,
    "indian_medicines": 253_973,
}


def _report() -> int:
    """Print what is cached and whether it looks complete. Returns an exit code."""
    cfg = get_config()
    problems = 0
    print(f"{'dataset':<20} {'rows':>10} {'expected':>10}  status")
    print("-" * 60)
    for name in loader.DATASETS:
        path = resolve_path(cfg.datasets[name])
        expected = EXPECTED_ROWS[name]
        if not path.exists():
            print(f"{name:<20} {'-':>10} {expected:>10,}  MISSING")
            problems += 1
            continue
        rows = len(loader.load_table(name))
        ok = rows == expected
        print(f"{name:<20} {rows:>10,} {expected:>10,}  "
              f"{'ok' if ok else 'ROW COUNT MISMATCH'}")
        problems += 0 if ok else 1
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is already cached")
    parser.add_argument("--verify", action="store_true",
                        help="report what is cached and exit without downloading")
    args = parser.parse_args()

    if args.verify:
        return _report()

    if args.force:
        cfg = get_config()
        for name in ("scms", "indian_medicines"):
            path = resolve_path(cfg.datasets[name])
            if path.exists():
                log.info("Removing cached %s to force a re-download", path.name)
                path.unlink()
        loader.load_table.cache_clear()

    log.info("Fetching external datasets (this needs a network on first run)...")
    loader.ensure_datasets()
    return _report()


if __name__ == "__main__":
    raise SystemExit(main())
