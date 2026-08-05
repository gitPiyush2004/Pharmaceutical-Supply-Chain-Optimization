"""
Shared Streamlit UI components.

Every page composes its layout from these helpers so the application reads as
one product: identical KPI cards, section headers, insight callouts, filter
panels and table styling everywhere. Pages are then left containing only what
is genuinely page-specific - the analysis and the narrative.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Path bootstrap -------------------------------------------------------
# Importing this module is enough to put the repository root on sys.path, so a
# page that reaches these components can also reach `src.analytics`, `src.ml`
# and the rest without repeating the bootstrap.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import get_config  # noqa: E402
from src.viz.theme import PALETTE, register_template  # noqa: E402

register_template()

# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
  /* Tighten Streamlit's default vertical rhythm for a denser, more
     enterprise-looking dashboard. */
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
  #MainMenu, footer {{ visibility: hidden; }}

  .pc-hero {{
      background: linear-gradient(120deg, {PALETTE['primary']} 0%, {PALETTE['secondary']} 100%);
      padding: 1.6rem 1.9rem; border-radius: 12px; color: white; margin-bottom: 1.4rem;
  }}
  .pc-hero h1 {{ margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.4px; }}
  .pc-hero p  {{ margin: .45rem 0 0; opacity: .93; font-size: .96rem; line-height: 1.45; }}

  .pc-section {{ margin: 1.8rem 0 .7rem; }}
  .pc-section h3 {{
      margin: 0; font-size: 1.12rem; font-weight: 650; color: #111827;
      border-left: 4px solid {PALETTE['primary']}; padding-left: .6rem;
  }}
  .pc-section p {{ margin: .3rem 0 0 .85rem; color: #4B5563; font-size: .88rem; }}

  .pc-kpi {{
      background: white; border: 1px solid #E5E7EB; border-radius: 10px;
      padding: .95rem 1.05rem; height: 100%;
      box-shadow: 0 1px 2px rgba(16,24,40,.05);
  }}
  .pc-kpi .label {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .55px;
                    color: #6B7280; font-weight: 600; }}
  .pc-kpi .value {{ font-size: 1.5rem; font-weight: 700; color: #111827; margin-top: .25rem;
                    line-height: 1.15; }}
  .pc-kpi .delta {{ font-size: .78rem; margin-top: .2rem; font-weight: 600; }}
  .pc-kpi .help  {{ font-size: .74rem; color: #9CA3AF; margin-top: .28rem; line-height: 1.3; }}

  .pc-callout {{
      border-radius: 9px; padding: .85rem 1.05rem; margin: .55rem 0;
      border-left: 4px solid; font-size: .9rem; line-height: 1.5;
  }}
  .pc-callout b {{ font-weight: 650; }}
  .pc-insight  {{ background: #EFF8FC; border-color: {PALETTE['primary']}; color: #0B3A4D; }}
  .pc-success  {{ background: #ECFAF1; border-color: {PALETTE['success']}; color: #14532D; }}
  .pc-warning  {{ background: #FEF7E7; border-color: {PALETTE['warning']}; color: #713F12; }}
  .pc-danger   {{ background: #FDEEEC; border-color: {PALETTE['danger']}; color: #7F1D1D; }}

  .pc-tag {{
      display: inline-block; padding: .16rem .55rem; border-radius: 999px;
      font-size: .7rem; font-weight: 650; letter-spacing: .3px; margin-right: .35rem;
  }}
  div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
</style>
"""


# ---------------------------------------------------------------------------
# Page scaffolding
# ---------------------------------------------------------------------------
def page_setup(title: str, icon: str, subtitle: str = "", wide: bool = True) -> None:
    """Configure the page and render the gradient hero header.

    Parameters
    ----------
    title
        Page title, also used as the browser tab title.
    icon
        Emoji shown in the tab and sidebar.
    subtitle
        One or two sentences explaining what question this page answers.
    """
    cfg = get_config()
    st.set_page_config(
        page_title=f"{title} | {cfg.project.short_name}",
        page_icon=icon,
        layout="wide" if wide else "centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""<div class="pc-hero">
              <h1>{icon}&nbsp; {title}</h1>
              {f"<p>{subtitle}</p>" if subtitle else ""}
            </div>""",
        unsafe_allow_html=True,
    )


def section(title: str, description: str = "") -> None:
    """Render a section heading with an optional one-line explanation."""
    st.markdown(
        f"""<div class="pc-section"><h3>{title}</h3>
            {f"<p>{description}</p>" if description else ""}</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def kpi_card(label: str, value: str, delta: str | None = None,
             delta_good: bool | None = None, help_text: str | None = None) -> str:
    """Return the HTML for a single KPI card.

    ``delta_good`` controls the delta colour independently of its sign, because
    a falling cost and a falling yield are both "down" but only one is good.
    """
    delta_html = ""
    if delta:
        colour = (PALETTE["success"] if delta_good else PALETTE["danger"]) \
            if delta_good is not None else PALETTE["neutral"]
        delta_html = f'<div class="delta" style="color:{colour}">{delta}</div>'
    help_html = f'<div class="help">{help_text}</div>' if help_text else ""
    return (f'<div class="pc-kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{delta_html}{help_html}</div>')


def kpi_row(cards: list[dict]) -> None:
    """Render a responsive row of KPI cards.

    Parameters
    ----------
    cards
        Each dict accepts ``label``, ``value``, and optionally ``delta``,
        ``delta_good`` and ``help_text``.
    """
    columns = st.columns(len(cards), gap="small")
    for column, card in zip(columns, cards):
        with column:
            st.markdown(kpi_card(**card), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Callouts
# ---------------------------------------------------------------------------
def callout(text: str, kind: str = "insight", title: str | None = None) -> None:
    """Render a coloured callout box.

    Parameters
    ----------
    kind
        ``insight`` (blue), ``success`` (green), ``warning`` (amber) or
        ``danger`` (red).
    """
    heading = f"<b>{title}</b><br>" if title else ""
    st.markdown(f'<div class="pc-callout pc-{kind}">{heading}{text}</div>',
                unsafe_allow_html=True)


def insight(text: str, title: str = "What this means") -> None:
    """Shorthand for the standard blue interpretation callout.

    Used under charts to state the business read explicitly, so a reviewer is
    never left to infer the conclusion from the visual alone.
    """
    callout(text, kind="insight", title=title)


def verdict_badge(verdict: str) -> str:
    """Return a coloured pill for an A/B test verdict or risk tier."""
    colours = {
        "ADOPT": PALETTE["success"], "ADOPT WITH MONITORING": PALETTE["warning"],
        "DO NOT ADOPT": PALETTE["danger"], "INCONCLUSIVE": PALETTE["neutral"],
        "Low": PALETTE["success"], "Medium": PALETTE["warning"], "High": PALETTE["danger"],
        "A": PALETTE["success"], "B": PALETTE["secondary"], "C": PALETTE["neutral"],
    }
    colour = colours.get(verdict, PALETTE["neutral"])
    return (f'<span class="pc-tag" style="background:{colour}1A;color:{colour};'
            f'border:1px solid {colour}55">{verdict}</span>')


# ---------------------------------------------------------------------------
# Data display
# ---------------------------------------------------------------------------
#: Friendly headers for the column names that recur across tables. Raw snake_case
#: is fine in a notebook but looks unfinished in a dashboard.
_COLUMN_LABELS: dict[str, str] = {
    "on_time_pct": "On-Time %", "avg_delay_days": "Avg Delay (d)",
    "median_delay_days": "Median Delay (d)", "p90_delay_days": "P90 Delay (d)",
    "line_value_usd": "Value (USD)", "freight_cost_usd": "Freight (USD)",
    "value_lost_usd": "Value Lost (USD)", "units_lost": "Units Lost",
    "dropoff_pct": "Drop-off %", "conversion_from_previous_pct": "Conversion %",
    "cumulative_conversion_pct": "Cumulative %", "shipment_mode": "Mode",
    "shipments": "Shipments", "vendor": "Vendor", "region": "Region",
    "coverage_pct": "Coverage %", "share_pct": "Share %",
    "missing_count": "Missing", "missing_pct": "Missing %",
    "outlier_count": "Outliers", "outlier_pct": "Outlier %",
    "violation_pct": "Violation %", "rows_affected": "Rows",
    "value_usd": "Value (USD)", "parsed_pct": "Parsed %",
}


def _pretty_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Title-case snake_case headers so tables read as finished UI.

    Known columns get a curated label; other *string* columns fall back to a
    title-cased version of their own name. Non-string headers (a contingency
    table indexed by integers, for instance) are left exactly as they are - they
    already mean something, and renaming them would destroy it.
    """
    renamed = {
        column: _COLUMN_LABELS.get(column, column.replace("_", " ").title())
        for column in data.columns
        if isinstance(column, str)
    }
    return data.rename(columns=renamed) if renamed else data


def show_table(data: pd.DataFrame, height: int | None = None,
               hide_index: bool = True, column_config: dict | None = None,
               prettify: bool = True) -> None:
    """Render a DataFrame with consistent styling and readable headers.

    Notes
    -----
    ``height`` is omitted from the call rather than passed as ``None``: recent
    Streamlit versions reject ``None`` instead of treating it as "auto".

    ``width="stretch"`` replaces the deprecated ``use_container_width=True``,
    which Streamlit removes after 2025-12-31.
    """
    frame = _pretty_columns(data) if prettify else data
    kwargs: dict = {"hide_index": hide_index, "width": "stretch",
                    "column_config": column_config}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(frame, **kwargs)


def download_button(data: pd.DataFrame, filename: str, label: str = "Download CSV",
                    key: str | None = None) -> None:
    """Offer a DataFrame as a CSV download (raw column names, not prettified)."""
    st.download_button(
        label=label, data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv", key=key,
    )


def chart(figure, key: str | None = None) -> None:
    """Render a Plotly figure with the platform's standard options."""
    st.plotly_chart(figure, width="stretch", key=key,
                    config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def sidebar_filters(data: pd.DataFrame, fields: list[str],
                    title: str = "Filters") -> pd.DataFrame:
    """Render sidebar multiselects for the given columns and return filtered data.

    Only fields actually present in the frame are rendered, so a single call
    works across tables with different schemas.

    Parameters
    ----------
    data
        Frame to filter.
    fields
        Candidate column names, e.g. ``["region", "drug_code", "supplier_name"]``.

    Returns
    -------
    pd.DataFrame
        The filtered frame (a copy).
    """
    labels = {
        # SCMS
        "region": "Region", "country": "Destination Country", "vendor": "Vendor",
        "shipment_mode": "Transport Mode", "fulfil_via": "Fulfilment Route",
        "product_group": "Product Group", "sub_classification": "Sub-Classification",
        "managed_by": "Managed By", "inco_term": "INCO Term",
        "manufacturing_site": "Manufacturing Site", "era": "Period",
        "first_line_designation": "First-Line Designation",
        # Indian medicine master
        "manufacturer": "Manufacturer", "pack_form": "Pack Form",
        "price_band": "Price Band",
    }
    st.sidebar.markdown(f"### {title}")
    filtered = data.copy()

    for field in fields:
        if field not in data.columns:
            continue
        options = sorted(data[field].dropna().unique().tolist())
        if len(options) <= 1:
            continue
        selected = st.sidebar.multiselect(
            labels.get(field, field.replace("_", " ").title()),
            options=options, default=options, key=f"filter_{field}",
        )
        if selected:
            filtered = filtered[filtered[field].isin(selected)]

    if len(filtered) < len(data):
        st.sidebar.caption(f"Showing {len(filtered):,} of {len(data):,} rows")
    if filtered.empty:
        st.warning("No records match the current filters. Widen the selection to continue.")
    return filtered


def sidebar_about() -> None:
    """Render the standard sidebar footer: project identity and data provenance."""
    cfg = get_config()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""**{cfg.project.short_name}** v{cfg.project.version}

*{cfg.project.name}*

**Three real datasets**
- `drug200` - Kaggle clinical dataset (200 patients)
- USAID SCMS - 10,324 real shipments, 43 countries
- Indian medicine master - 253,973 products

No simulated data. Deterministic under
`random_seed={cfg.project.random_seed}`.

Built by {cfg.project.author}
"""
    )


def methodology(text: str, title: str = "Methodology") -> None:
    """Collapsible panel documenting how the numbers on a page were produced.

    Present on every analytical page: a reviewer should always be able to check
    the definition behind a metric without leaving the dashboard.
    """
    with st.expander(f"{title}", expanded=False):
        st.markdown(text)


__all__ = [
    "page_setup", "section", "kpi_card", "kpi_row", "callout", "insight",
    "verdict_badge", "show_table", "download_button", "chart",
    "sidebar_filters", "sidebar_about", "methodology",
]
