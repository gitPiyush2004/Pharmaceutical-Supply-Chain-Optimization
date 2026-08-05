"""
Data Quality page.

Audits all three real datasets on five weighted dimensions, and then makes an
argument against its own headline number. Generic profiling gives the USAID SCMS
file a grade A, and that grade is wrong in a way worth understanding: the file's
defects are semantic rather than structural, so a null check cannot see them.

The section that carries the page is "What Parsing Actually Costs" - parsing SCMS
*lowers* its generic score, because honest nulls score worse than unparseable
text. If a quality module can only ever make a number go up, it is measuring the
module rather than the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.quality import assessment as dq
from src.viz import charts
from src.viz.theme import fmt_pct

page_setup(
    title="Data Quality Assessment",
    icon="🧪",
    subtitle="Auditing three real datasets - and why the headline score on the "
             "messiest one is misleading",
)

DATASETS = ["scms", "indian_medicines", "drug200"]
DIMENSIONS = ["completeness", "uniqueness", "validity", "consistency", "accuracy"]


@st.cache_data(show_spinner="Auditing datasets...")
def _boards() -> tuple:
    return dq.assess_all(DATASETS, raw=True), dq.assess_all(DATASETS, raw=False)


@st.cache_data(show_spinner=False)
def _uplift():
    return dq.quality_uplift(DATASETS)


@st.cache_data(show_spinner=False)
def _audit(name: str, raw: bool):
    return dq.assess_dataset(name, raw=raw)


@st.cache_data(show_spinner=False)
def _scms_parsing():
    from src.data.scms import parsing_report

    return parsing_report()


published, interpreted = _boards()
uplift = _uplift()
parsing = _scms_parsing()

worst_published = published.iloc[0]
kpi_row([
    {"label": "Datasets Audited", "value": f"{len(published)}",
     "help_text": "All three are real and publicly available"},
    {"label": "Rows Profiled",
     "value": f"{int(published['rows'].sum()):,}",
     "help_text": "253,973 Indian products, 10,324 SCMS shipments, 200 patients"},
    {"label": "Lowest Published Score",
     "value": f"{worst_published['overall_score']:.2f}",
     "help_text": f"{worst_published['dataset']} - grade "
                  f"{worst_published['grade']}. Every dataset scores an A, which "
                  f"is the problem."},
    {"label": "SCMS Freight Usable",
     "value": fmt_pct(float(
         parsing.set_index("field").loc["freight_cost", "parsed_pct"])),
     "help_text": "Against a 99.3% completeness score on the same column"},
])

# ---------------------------------------------------------------------------
# The scoreboard
# ---------------------------------------------------------------------------
section(
    "Scoreboard",
    "Each dataset scored on five weighted dimensions defined in "
    "`config/config.yaml` (`data_quality.weights`), as published by the source.",
)

show_table(published)

chart(charts.heatmap(
    published.set_index("dataset")[DIMENSIONS],
    title="Score by Dimension, As Published (0-100)",
    colorscale="Blues", text_format=".1f"))

callout(
    "**All three datasets score grade A, and one of them is a mess.** That is not a "
    "flattering result for the scoring function — it is the finding. Generic "
    "profiling asks whether cells are populated, unique and in range. It cannot ask "
    "whether a populated cell means anything. The next two sections show what it "
    "missed.",
    kind="warning", title="Read this scoreboard sceptically",
)

# ---------------------------------------------------------------------------
# The real story: semantic defects
# ---------------------------------------------------------------------------
section(
    "Where Generic Profiling Falls Short",
    "The most instructive table on this page. The SCMS export scores 99.3% "
    "complete, and that number is close to meaningless.",
)

show_table(parsing)

worst_field = parsing.iloc[0]
callout(
    f"A standard completeness check sees no problem with this file, because its "
    f"defects are **semantic, not structural**. `{worst_field['field']}` is only "
    f"**{worst_field['parsed_pct']:.0f}% usable**, yet a null check passes it "
    f"perfectly: the unusable values are strings like `N/A - From RDC`, "
    f"`Pre-PQ Process`, `Freight Included in Commodity Cost` and "
    f"`See DN-304 (ID#:10589)` sitting in text columns. Every one of them is "
    f"non-null and none of them is a date or a number.\n\n"
    f"This is the difference between profiling **types** and profiling **meaning**. "
    f"Type-aware parsing shows that 55% of purchase-order dates and 40% of freight "
    f"costs cannot be used as dates or numbers at all.",
    kind="warning", title="Why the 99.3% completeness score is misleading",
)

insight(
    "The right response is **not** to impute those values, and this is where a "
    "generic cleaning layer would have done real damage. `N/A - From RDC` is a "
    "**structural absence**: it correctly records that no vendor purchase order "
    "existed, because the goods came from regional distribution centre stock. "
    "Mode-imputing that column — which is exactly what a generic object-column "
    "cleaner does — would have fabricated 5,404 purchase orders and corrupted every "
    "lead-time figure downstream.\n\n"
    "`Freight Included in Commodity Cost` is not a missing price either; it says the "
    "cost is recorded inside another number. And 2,445 `See DN-304 (ID#:10589)` "
    "values are cross-references that embed a resolvable row ID, so they are "
    "*recoverable* rather than lost. Each value gets a reason code — `parsed`, "
    "`structural`, `missing` or `cross_reference` — and is excluded from the "
    "affected statistic, which is why every lead-time figure in this project "
    "reports the denominator it actually used."
)

# ---------------------------------------------------------------------------
# What parsing costs
# ---------------------------------------------------------------------------
section(
    "What Parsing Actually Costs",
    "The same scoring function applied before and after each dataset's own parser. "
    "Note the sign on the SCMS row.",
)

show_table(uplift)

scms_row = uplift[uplift["dataset"] == "scms"].iloc[0]
indian_row = uplift[uplift["dataset"] == "indian_medicines"].iloc[0]
insight(
    f"**Parsing SCMS makes its score go *down*, by "
    f"{abs(scms_row['uplift']):.2f} points** — completeness falls "
    f"{abs(scms_row['completeness_delta']):.2f} points because parsing converts "
    f"unparseable text into honest nulls. A string reading `Pre-PQ Process` counts "
    f"as complete; the `NaT` that replaces it does not.\n\n"
    f"That negative number is the most useful thing on this page. It means the "
    f"generic score **rewards a file for holding non-null garbage**, and any "
    f"pipeline judged on *did the quality score improve* would be incentivised "
    f"to leave the garbage in place. The interpreted layer is unambiguously more "
    f"trustworthy and scores worse for it.\n\n"
    f"The Indian dataset moves the other way ({indian_row['uplift']:+.2f}), because "
    f"there the parsing genuinely recovers information: a price column stored as "
    f"text becomes a number, and pack labels become a form plus a quantity. And "
    f"drug200 is flat at {uplift[uplift['dataset'] == 'drug200'].iloc[0]['uplift']:+.2f} "
    f"— it is published clean, so there is nothing to recover and no attempt is "
    f"made to invent a difference."
)

chart(charts.heatmap(
    interpreted.set_index("dataset")[DIMENSIONS],
    title="Score by Dimension, Interpreted (0-100)",
    colorscale="Blues", text_format=".1f"))

# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------
section(
    "Cross-Field Consistency",
    "The checks that catch what a per-column audit cannot: records where every "
    "individual value is plausible but the combination is impossible.",
)

scms_consistency = _audit("scms", raw=False)["consistency"]
show_table(scms_consistency)

insight(
    "**These three checks caught a real bug in this project.** The five SCMS date "
    "columns use two different formats — `%m/%d/%y` for the quote and "
    "purchase-order dates, `%d-%b-%y` for the three delivery dates. Parsing them "
    "all with one `dayfirst` rule read `5/3/13` as 3 May instead of 5 March, which "
    "produced 478 negative lead times and 1,128 purchase orders that preceded their "
    "own price quote. None of that is visible to a completeness, uniqueness or "
    "range check: every date was populated and every date was a valid date.\n\n"
    "After the fix, 18 violations remain out of 22,559 comparisons — genuine "
    "recording errors in the source, now reported rather than parsed away.\n\n"
    "The Indian master has **no** cross-field invariant to test, and the audit says "
    "so rather than inventing one. An earlier version of this page asserted that "
    "active-ingredient count could not exceed pack quantity, which flagged 7,886 "
    "products as inconsistent. Every one was correct: a single vial of Augmentin "
    "contains two molecules. The rule was removed, not tuned, because the premise "
    "was wrong."
)

# ---------------------------------------------------------------------------
# Drill-down
# ---------------------------------------------------------------------------
section("Dataset Drill-Down", "Full profile for a single dataset.")

chosen = st.selectbox("Dataset", DATASETS, index=0)
layer = st.radio(
    "Layer", ["As published", "Interpreted"], horizontal=True, index=0,
    help="'Interpreted' means the dataset has been through its own parser - "
         "per-column date formats for SCMS, price and pack parsing for the Indian "
         "master. It does not mean imputed.")
report = _audit(chosen, raw=layer == "As published")
score = report["score"]

kpi_row([
    {"label": "Overall Score", "value": f"{score['overall_score']:.2f}",
     "help_text": f"Grade {score['grade']} ({report['layer']})"},
    {"label": "Rows", "value": f"{report['rows']:,}"},
    {"label": "Columns", "value": f"{report['columns']}"},
    {"label": "Weakest Dimension",
     "value": min(DIMENSIONS, key=lambda d: score[d]).title(),
     "help_text": f"{min(score[d] for d in DIMENSIONS):.2f}"},
])

st.caption(score["verdict"])

tabs = st.tabs(["Missing Values", "Duplicates", "Validity", "Outliers",
                "Consistency", "Summary Statistics", "Recommendations"])

with tabs[0]:
    missing = report["missing"]
    flagged = missing[missing["missing_count"] > 0]
    if len(flagged):
        show_table(flagged)
        chart(charts.bar_chart(flagged, x="column", y="missing_pct",
                               title="Missing Data by Column (%)",
                               text_format=".2f", height=360))
        if chosen == "scms" and layer == "Interpreted":
            st.caption(
                "Much of this is *structural* absence rather than a gap — see the "
                "reason-code columns. The Delivery Pipeline page separates the two.")
    else:
        st.success("No missing values in this dataset.")

with tabs[1]:
    dup = report["duplicates"]
    kpi_row([
        {"label": "Exact Duplicate Rows",
         "value": f"{dup['exact_duplicate_rows']:,}",
         "help_text": fmt_pct(dup["duplicate_pct"], 2)},
        {"label": "Business-Key Duplicates",
         "value": f"{dup['key_duplicate_rows']:,}",
         "help_text": f"Key: {', '.join(dup['key_columns']) or 'none declared'}"},
    ])
    examples = dup.get("key_duplicate_examples")
    if examples is not None and len(examples):
        st.markdown("**Example duplicated records**")
        show_table(examples.head(10))
    else:
        st.success("No duplicate rows on the declared business key.")

with tabs[2]:
    validity = report["validity"]
    breaches = validity[validity["violations"] > 0]
    if len(breaches):
        show_table(breaches)
        st.caption("Rules are declared per column and evaluated against every row. "
                   "`sample_bad_values` shows the actual offending values.")
    else:
        st.success("Every declared validity rule passes.")
    with st.expander("All rules evaluated"):
        show_table(validity)

with tabs[3]:
    outliers = report["outliers"]
    flagged = outliers[outliers["outlier_count"] > 0]
    show_table(flagged if len(flagged) else outliers)
    if len(flagged):
        chart(charts.bar_chart(flagged.nlargest(12, "outlier_pct"),
                               x="column", y="outlier_pct",
                               title="Outlier Rate by Column (%)",
                               text_format=".2f", height=380))
    st.caption(
        "Outliers are flagged, never removed. A freight ratio of 13,449,400% is a "
        "genuine record — a tiny line value against a real shipping cost — and it is "
        "exactly why the statistical tests on this project use medians and "
        "rank-based tests on freight data rather than means.")

with tabs[4]:
    consistency = report["consistency"]
    if len(consistency):
        show_table(consistency)
        st.caption("Cross-field logic: a purchase order cannot postdate its own "
                   "delivery, and a delivery cannot be recorded before it arrives.")
    else:
        st.info(
            "No cross-field invariant applies to this dataset. It is a flat table "
            "with no derived quantities and no chronology, so there is nothing to "
            "cross-check — and the audit reports that rather than inventing a rule.")

with tabs[5]:
    show_table(report["summary"])
    download_button(report["summary"], f"{chosen}_summary_statistics.csv",
                    "Download summary statistics", key="dl_summary")

with tabs[6]:
    show_table(report["recommendations"])
    st.caption("Generated from the profile and ordered by severity.")

methodology("""
**Scoring.** Each dimension is scored 0-100 and combined using the weights in
`config/config.yaml` (`data_quality.weights`): completeness 30%, validity 25%,
uniqueness 20%, consistency 15%, accuracy 10%. Letter grades follow
`data_quality.grade_bands`.

- **Completeness** — share of non-null cells. Note the limitation this page is
  built around: a non-null string that cannot be parsed counts as complete.
- **Uniqueness** — penalises exact and business-key duplicate rows.
- **Validity** — share of values passing declared per-column rules (ranges,
  allowed categories, non-negativity).
- **Consistency** — cross-field logic. A check runs only when both its columns are
  present *and* comparably typed; comparing two date-shaped strings
  lexicographically once reported an 82% violation rate that was pure artefact.
- **Accuracy** — proxied by outlier rate using the IQR method
  (`data_quality.outlier_method`, multiplier 1.5).

**There is no generic cleaning layer.** An earlier version routed every dataset
through one, and it mode-imputed every object column — including SCMS, where that
means inventing purchase orders that never existed. Each dataset that needs
interpreting now owns it, because in both cases the interpretation is inseparable
from understanding the source. `drug200` needs none: verified zero nulls, zero
duplicates, no out-of-range values.

**Outliers are flagged, not dropped.** The largest values in this data are usually
real. Removing them would improve the accuracy score and destroy the finding.
""")

sidebar_about()
