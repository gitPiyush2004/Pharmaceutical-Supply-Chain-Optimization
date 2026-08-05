#!/usr/bin/env python
"""
Export the README figures as static PNGs.

Exists because these images were previously produced ad hoc, which meant they
could silently drift out of step with the code. Now a single command regenerates
all of them from the same chart builders the dashboard uses, so a figure in the
README cannot show a number the dashboard no longer produces.

**Inspect the output.** A Plotly figure that is wrong is usually still a figure,
so it exports without complaint. The horizontal-bar axis-swap bug in
``src/viz/charts.py`` rendered every h-bar chart in the project as unreadable
slivers and was caught only by opening an exported PNG - not by any test.

Requires ``kaleido`` (in requirements.txt).

Usage
-----
    python scripts/export_figures.py
    python scripts/export_figures.py --figure value_funnel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analytics import experiments as ex  # noqa: E402
from src.analytics import pipeline as pl  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.ml import predict  # noqa: E402
from src.viz import charts  # noqa: E402
from src.viz.theme import register_template  # noqa: E402

log = get_logger("scripts.export_figures")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"

#: Export width and scale. 1600px at scale 2 renders crisply on a retina display
#: and still reads at GitHub's rendered width.
WIDTH, SCALE = 1400, 2


def _value_funnel():
    """Commodity value by tightening delivery standard."""
    return charts.value_funnel_chart(
        pl.value_funnel(), title="Commodity Value Through the Delivery Pipeline",
        height=460)


def _simpsons_paradox():
    """The headline statistical finding: a pooled gap that hides a collapse."""
    strat = ex.stratified_comparison()
    return charts.stratified_effect_chart(
        strat["strata"], strat["dimension"], strat["stratified_by"],
        title="Fulfilment Route On-Time Rate Within Each Era", height=440)


def _confusion_matrix():
    """Drug classification performance on the held-out test set."""
    meta = predict.model_summary("drug_classification")
    cm = meta["confusion_matrix"]
    matrix = pd.DataFrame(cm["values"], index=cm["index"], columns=cm["labels"])
    return charts.confusion_matrix_chart(
        matrix, title="Drug Classification - Confusion Matrix (Test Set)", height=460)


def _late_delivery_gains():
    """The metric the late-delivery model is actually deployed on."""
    gains = predict.late_delivery_targeting_curve()
    return charts.line_chart(
        gains, x="targeted_pct", y=["capture_rate_pct", "precision_pct"],
        title="Late-Delivery Model - Gains Curve on Held-Out Data",
        y_title="Percent", height=440)


FIGURES = {
    "value_funnel": _value_funnel,
    "simpsons_paradox": _simpsons_paradox,
    "confusion_matrix": _confusion_matrix,
    "late_delivery_gains": _late_delivery_gains,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=sorted(FIGURES),
                        help="Export a single figure instead of all of them.")
    args = parser.parse_args()

    register_template()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [args.figure] if args.figure else list(FIGURES)

    for name in targets:
        figure = FIGURES[name]()
        path = OUTPUT_DIR / f"{name}.png"
        height = figure.layout.height or 440
        figure.write_image(str(path), width=WIDTH, height=height, scale=SCALE)
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(OUTPUT_DIR.parents[1])}  "
              f"({WIDTH}x{height} @{SCALE}x, {size_kb:.0f} KB)")

    print(f"\n{len(targets)} figure(s) exported. Open each one before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
