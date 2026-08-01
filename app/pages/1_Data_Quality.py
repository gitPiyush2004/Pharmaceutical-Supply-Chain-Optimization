"""
Data Quality page.

Profiles the raw extract exactly as the source systems hand it over, then shows
what the cleaning layer did about it. The before/after pairing is the point: a
quality score in isolation proves nothing, a measured uplift proves the pipeline
earns its place.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.dashboard.components import (callout, chart, insight, kpi_row, methodology,
                        page_setup, section, show_table, sidebar_about,
                        download_button)
from src.data import loader
from src.quality import assessment as dq
from src.data.cleaning import remediation_summary
from src.viz import charts
from src.viz.theme import fmt_pct

page_setup(
    title="Data Quality Assessment",
    icon="🧪",
    subtitle="Profiling the bronze extract, and measuring what cleaning recovered",
)

DATASETS = ["batches", "shipments", "inventory", "demand", "drug200"]


@st.cache_data(show_spinner="Auditing datasets...")
def _scoreboard(raw: bool):
    return dq.assess_all(raw=raw)


@st.cache_data(show_spinner="Comparing layers...")
def _uplift():
    return dq.quality_uplift()


@st.cache_data(show_spinner=False)
def _audit(name: str, raw: bool):
    return dq.assess_dataset(name, raw=raw)


@st.cache_data(show_spinner=False)
def _logs():
    return loader.remediation_log(), loader.injected_defect_log()


raw_board = _scoreboard(raw=True)
uplift = _uplift()
remediation, injected = _logs()

# ---------------------------------------------------------------------------
# Headline scores
# ---------------------------------------------------------------------------
worst = raw_board.iloc[0]
kpi_row([
    {"label": "Datasets Audited", "value": f"{len(raw_board)}",
     "help_text": "Four supply chain tables plus the clinical dataset"},
    {"label": "Lowest Raw Score", "value": f"{worst['overall_score']:.2f}",
     "help_text": f"{worst['dataset']} - grade {worst['grade']}"},
    {"label": "Rows Profiled",
     "value": f"{int(raw_board['rows'].sum()):,}",
     "help_text": "Across every table in the bronze layer"},
    {"label": "Remediation Actions",
     "value": f"{int(remediation['rows_affected'].sum()):,}",
     "help_text": f"{len(remediation)} distinct fixes applied"},
])

# ---------------------------------------------------------------------------
# Bronze vs silver
# ---------------------------------------------------------------------------
section(
    "Bronze versus Silver",
    "The same scoring function applied before and after remediation. The uplift "
    "column is the value the cleaning layer adds.",
)

show_table(
    uplift[["dataset", "raw_score", "raw_grade", "clean_score", "clean_grade",
            "uplift", "completeness_delta", "uniqueness_delta", "validity_delta"]],
    height=250,
)

best = uplift.iloc[0]
insight(
    f"Cleaning lifts **{best['dataset']}** by **{best['uplift']:+.2f} points** "
    f"({best['raw_score']:.2f} to {best['clean_score']:.2f}), driven mostly by "
    f"validity (+{best['validity_delta']:.2f}) and uniqueness "
    f"(+{best['uniqueness_delta']:.2f}). Validity gains come from nulling "
    "physically impossible values - negative storage durations and potency above "
    "label claim - and uniqueness gains from removing ERP double-postings."
)

# ---------------------------------------------------------------------------
# Dimension scores
# ---------------------------------------------------------------------------
section(
    "Quality Dimensions",
    "Each table is scored on five weighted dimensions defined in "
    "`config/config.yaml`: completeness, uniqueness, validity, consistency and accuracy.",
)

dimension_cols = ["completeness", "uniqueness", "validity", "consistency", "accuracy"]
heat = raw_board.set_index("dataset")[dimension_cols]
chart(charts.heatmap(heat, title="Raw Extract Score by Dimension (0-100)",
                     colorscale="Blues", text_format=".1f"))

# ---------------------------------------------------------------------------
# Detected versus injected
# ---------------------------------------------------------------------------
section(
    "Detected versus Injected Defects",
    "The generator records every defect it introduces. Comparing that log against "
    "what the profiler found is a direct test that the quality module works.",
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Injected by the generator**")
    show_table(injected.sort_values("rows_affected", ascending=False), height=320)
with col2:
    st.markdown("**Remediated by the cleaning layer**")
    show_table(remediation_summary(remediation), height=320)

callout(
    "Defect classes reproduced here are the ones that genuinely appear in "
    "pharmaceutical source systems: cold-chain logger dropouts, carrier EDI "
    "messages that never arrive, ERP double-postings, free-text region spellings "
    "(`APAC`, `asia pacific`, `Asia-Pacific`), sign errors on durations, and "
    "trailing whitespace from CSV exports.",
    kind="insight", title="Why these defects",
)

# ---------------------------------------------------------------------------
# Per-dataset drill-down
# ---------------------------------------------------------------------------
section("Dataset Drill-Down", "Full profile for a single table.")

chosen = st.selectbox("Dataset", DATASETS, index=0)
layer = st.radio("Layer", ["Bronze (raw extract)", "Silver (cleaned)"],
                 horizontal=True, index=0)
report = _audit(chosen, raw=layer.startswith("Bronze"))
score = report["score"]

kpi_row([
    {"label": "Overall Score", "value": f"{score['overall_score']:.2f}",
     "help_text": f"Grade {score['grade']}"},
    {"label": "Rows", "value": f"{report['rows']:,}"},
    {"label": "Columns", "value": f"{report['columns']}"},
    {"label": "Weakest Dimension",
     "value": min(dimension_cols, key=lambda d: score[d]).title(),
     "help_text": f"{min(score[d] for d in dimension_cols):.2f}"},
])

st.caption(score["verdict"])

tabs = st.tabs(["Missing Values", "Duplicates", "Validity", "Outliers",
                "Consistency", "Summary Statistics", "Recommendations"])

with tabs[0]:
    missing = report["missing"]
    flagged = missing[missing["missing_count"] > 0]
    if len(flagged):
        show_table(flagged, height=280)
        chart(charts.bar_chart(flagged, x="column", y="missing_pct",
                               title="Missing Data by Column (%)",
                               text_format=".2f", height=340))
    else:
        st.success("No missing values in this table.")

with tabs[1]:
    dup = report["duplicates"]
    kpi_row([
        {"label": "Exact Duplicate Rows", "value": f"{dup['exact_duplicate_rows']:,}",
         "help_text": fmt_pct(dup["duplicate_pct"], 2)},
        {"label": "Business-Key Duplicates", "value": f"{dup['key_duplicate_rows']:,}",
         "help_text": f"Key: {', '.join(dup['key_columns']) or 'n/a'}"},
    ])
    examples = dup.get("key_duplicate_examples")
    if examples is not None and len(examples):
        st.markdown("**Example duplicated records**")
        show_table(examples.head(10), height=240)

with tabs[2]:
    validity = report["validity"]
    breaches = validity[validity["violations"] > 0]
    if len(breaches):
        show_table(breaches, height=280)
        st.caption("Rules are declared per column and evaluated against every row. "
                   "`sample_bad_values` shows the actual offending values.")
    else:
        st.success("Every declared validity rule passes.")
    with st.expander("All rules evaluated"):
        show_table(validity, height=320)

with tabs[3]:
    outliers = report["outliers"]
    flagged = outliers[outliers["outlier_count"] > 0]
    show_table(flagged if len(flagged) else outliers, height=320)
    if len(flagged):
        chart(charts.bar_chart(flagged.nlargest(12, "outlier_pct"),
                               x="column", y="outlier_pct",
                               title="Outlier Rate by Column (%)",
                               text_format=".2f", height=360))
    st.caption("Outliers are flagged, never silently removed: a genuine "
               "temperature excursion is exactly the observation worth keeping.")

with tabs[4]:
    consistency = report["consistency"]
    show_table(consistency, height=340)
    st.caption("Cross-field logic checks - for example, units at each funnel stage "
               "must never exceed the stage before it, and dates must be ordered.")

with tabs[5]:
    show_table(report["summary"], height=420)
    download_button(report["summary"], f"{chosen}_summary_statistics.csv",
                    "Download summary statistics", key="dl_summary")

with tabs[6]:
    recs = report["recommendations"]
    show_table(recs, height=340)
    st.caption("Generated from the profile, ordered by severity. These are the "
               "actions the cleaning layer implements.")

methodology("""
**Scoring.** Each dimension is scored 0-100 and combined using the weights in
`config/config.yaml` (`data_quality.weights`): completeness 30%, validity 25%,
uniqueness 20%, consistency 15%, accuracy 10%. Letter grades follow
`data_quality.grade_bands`.

- **Completeness** - share of non-null cells.
- **Uniqueness** - penalises exact and business-key duplicate rows.
- **Validity** - share of values passing declared per-column rules (ranges,
  allowed categories, non-negativity).
- **Consistency** - cross-field logic: funnel stage volumes must be monotonically
  non-increasing, stage dates must be ordered, derived values must reconcile.
- **Accuracy** - proxied by outlier rate using the IQR method
  (`data_quality.outlier_method`, multiplier 1.5).

**Why profile the bronze layer.** A data quality audit exists to characterise what
arrived, before anyone has intervened. Profiling only the cleaned table would
score the cleaning code, not the data. Both layers are available above.

**Outliers are flagged, not dropped.** In stability analytics an extreme storage
temperature is the signal, not noise. The cleaning layer only nulls values that
are *physically impossible* (a negative duration), never values that are merely
extreme.
""")

sidebar_about()
