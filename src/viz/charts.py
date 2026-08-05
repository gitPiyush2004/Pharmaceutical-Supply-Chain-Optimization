"""
Reusable Plotly chart builders.

Every visual in the dashboard is produced here rather than inside a page, for
three reasons: the pages stay readable (layout and narrative only), the charts
stay consistent (one palette, one template, one hover convention), and each
chart can be unit-tested or reused in the notebook without Streamlit.

All builders return a ``plotly.graph_objects.Figure`` with the platform theme
already applied, so a caller only needs ``st.plotly_chart(fig, ...)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import (FUNNEL_SCALE, PALETTE, SEQUENCE, apply_theme,
                           status_colour)


# ---------------------------------------------------------------------------
# Order-to-delivery pipeline
# ---------------------------------------------------------------------------
# There is no unit-attrition funnel here, because SCMS cannot support one: it
# states quantity once at order time and never restates it at delivery. See
# src/analytics/pipeline.py. What these builders draw instead is attrition in
# *value delivered on time*, which is measured rather than inferred.
def value_funnel_chart(funnel: pd.DataFrame, title: str = "Value Through the Pipeline",
                       height: int = 420) -> go.Figure:
    """Commodity value by delivery outcome.

    Parameters
    ----------
    funnel
        Output of :func:`src.analytics.pipeline.value_funnel`.
    """
    # Plotly's own `textinfo="value"` renders 1627584457 as "1.62758B". Building
    # the label explicitly keeps it readable, which is the point of putting the
    # number inside the band.
    labels = [f"${value/1e6:,.0f}M<br>{share:.1f}%"
              for value, share in zip(funnel["value_usd"], funnel["share_pct"])]
    fig = go.Figure(go.Funnel(
        y=funnel["stage"], x=funnel["value_usd"],
        text=labels, textposition="inside", textinfo="text", opacity=0.95,
        marker=dict(color=FUNNEL_SCALE[:len(funnel)], line=dict(width=1, color="white")),
        connector=dict(line=dict(color=PALETTE["neutral"], width=1)),
        customdata=funnel["line_items"],
        hovertemplate=("<b>%{y}</b><br>Value: $%{x:,.0f}<br>"
                       "Line items: %{customdata:,}<extra></extra>"),
    ))
    return apply_theme(fig, title=title, height=height, showlegend=False)


def lateness_funnel_chart(funnel: pd.DataFrame,
                          title: str = "Value Arriving Within N Days of Schedule",
                          height: int = 380) -> go.Figure:
    """Cumulative share of value arriving within each lateness threshold.

    Monotone by construction, so unlike a coverage chart this reads correctly as a
    funnel: each band is a superset of the one above it.
    """
    fig = go.Figure(go.Bar(
        x=funnel["share_pct"], y=funnel["label"], orientation="h",
        marker_color=FUNNEL_SCALE[:len(funnel)],
        text=[f"{v:.1f}%" for v in funnel["share_pct"]], textposition="outside",
        customdata=np.stack([funnel["value_usd"], funnel["line_items"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Share of value: %{x:.2f}%<br>"
                       "Value: $%{customdata[0]:,.0f}<br>"
                       "Line items: %{customdata[1]:,}<extra></extra>"),
    ))
    fig.update_xaxes(title="Cumulative share of commodity value (%)", range=[0, 108])
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, title=title, height=height, showlegend=False)


def dwell_time_chart(breakdown: pd.DataFrame,
                     title: str = "Pipeline Intervals (Median Days)",
                     height: int = 400) -> go.Figure:
    """Median duration of each pipeline interval, annotated with its denominator.

    Every interval is measured on a different number of line items - the vendor-PO
    interval only exists for direct-drop orders - so the coverage is drawn on the
    bar rather than left in a footnote. A reader comparing 92 days against 154 days
    needs to know the second figure rests on 66% more rows.
    """
    data = breakdown.sort_values("median_days")
    fig = go.Figure(go.Bar(
        x=data["median_days"], y=data["interval"], orientation="h",
        marker_color=PALETTE["secondary"],
        text=[f"{v:.0f}d  ({c:.0f}% of items)"
              for v, c in zip(data["median_days"], data["coverage_pct"])],
        textposition="outside",
        customdata=np.stack([data["mean_days"], data["p90_days"],
                             data["line_items"], data["coverage_pct"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Median: %{x:.1f} d<br>"
                       "Mean: %{customdata[0]:.1f} d<br>P90: %{customdata[1]:.1f} d<br>"
                       "Measured on %{customdata[2]:,} items "
                       "(%{customdata[3]:.1f}%)<extra></extra>"),
    ))
    fig.add_vline(x=0, line_color=PALETTE["neutral"])
    fig.update_xaxes(title="Days")
    return apply_theme(fig, title=title, height=height, showlegend=False)


def traceability_chart(trace: pd.DataFrame,
                       title: str = "Milestone Recording Coverage",
                       height: int = 380) -> go.Figure:
    """Recording coverage per milestone, splitting structural absence from gaps.

    Deliberately a stacked bar rather than a funnel. Coverage is non-monotone
    (74% -> 44% -> 100%), and a funnel shape would imply shipments dropping out of
    the process when in fact a fulfilment route simply has no vendor order to
    record. The stack makes the distinction visible: dark blue is recorded, grey is
    structurally absent, red is a genuine gap.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trace["stage"], y=trace["recorded"], name="Recorded",
        marker_color=PALETTE["primary"],
        hovertemplate="<b>%{x}</b><br>Recorded: %{y:,}<extra></extra>"))
    fig.add_trace(go.Bar(
        x=trace["stage"], y=trace["structurally_absent"],
        name="Structurally absent", marker_color="rgba(107,114,128,0.55)",
        customdata=trace["interpretation"],
        hovertemplate="<b>%{x}</b><br>Never existed: %{y:,}<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Bar(
        x=trace["stage"], y=trace["genuinely_missing"], name="Genuinely missing",
        marker_color=PALETTE["danger"],
        hovertemplate="<b>%{x}</b><br>Unrecorded: %{y:,}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Line items")
    fig.update_xaxes(tickangle=-20)
    return apply_theme(fig, title=title, height=height)


# ---------------------------------------------------------------------------
# Generic building blocks
# ---------------------------------------------------------------------------
#: Longest category label rendered on an axis before it is shortened.
MAX_LABEL_CHARS = 28

#: Readable axis titles for the column names that recur across charts. Without
#: this a reader sees the raw column name, and "median_freight_per_kg_usd" on an
#: axis is the kind of detail that makes a dashboard look unfinished.
_AXIS_LABELS: dict[str, str] = {
    # --- delivery performance ---------------------------------------------
    "on_time_pct": "On-time delivery (%)", "avg_delay_days": "Average delay (days)",
    "mean_delay_days": "Mean delay (days)", "median_days": "Median duration (days)",
    "delivery_delay_days": "Delivery delay (days)",
    "scheduled_lead_time_days": "Scheduled lead time (days)",
    "total_lead_time_days": "Quote-to-delivery (days)",
    "coverage_pct": "Milestone coverage (%)", "shipments": "Shipments",
    "line_items": "Line items", "share_pct": "Share of value (%)",
    "items_share_pct": "Share of line items (%)",
    # --- money and volume --------------------------------------------------
    "line_value_usd": "Commodity value (USD)", "freight_cost_usd": "Freight (USD)",
    "value_usd": "Commodity value (USD)",
    "median_freight_per_kg_usd": "Median freight (USD/kg)",
    "freight_pct_of_value": "Freight as share of value (%)",
    "annual_value_usd": "Estimated annual value (USD)",
    "units_ordered": "Units", "packs_ordered": "Packs",
    # --- product catalogue and pricing -------------------------------------
    "unit_price_usd": "Unit price (USD)", "pack_price_usd": "Pack price (USD)",
    "median_price": "Median unit price (USD)", "min_price": "Lowest price paid (USD)",
    "max_price": "Highest price paid (USD)",
    "median_unit_price_usd": "Median unit price (USD)",
    "spread_x": "Price spread (max / min)", "premium_x": "Branded premium (x)",
    "generic_price": "Generic unit price (USD)",
    "branded_price": "Branded unit price (USD)",
    "molecule": "Molecule", "dosage": "Dosage", "dosage_form": "Dosage form",
    "label": "Product and year",
    "brand": "Brand", "product": "Product", "manufacturing_site": "Factory",
    "value_share_pct": "Share of value (%)",
    "cumulative_share_pct": "Cumulative share of value (%)",
    "sites": "Factories", "delivery_year": "Delivery year",
    # --- quality and ML ----------------------------------------------------
    "missing_pct": "Missing (%)", "outlier_pct": "Outliers (%)",
    "importance": "Importance", "probability": "Probability",
    "capture_rate_pct": "Late deliveries captured (%)",
    "targeted_pct": "Shipments reviewed (%)", "precision_pct": "Precision (%)",
    # --- dimensions --------------------------------------------------------
    "vendor": "Vendor", "region": "Region", "shipment_mode": "Transport mode",
    "fulfil_via": "Fulfilment route", "product_group": "Product group",
    "era": "Period", "country": "Destination country",
    "stage": "Stage", "interval": "Pipeline interval", "level": "Group",
    "feature": "Feature", "dataset": "Dataset", "dimension": "Dimension",
}


def shorten_labels(data: pd.DataFrame, column: str,
                   limit: int = MAX_LABEL_CHARS) -> tuple[pd.DataFrame, bool]:
    """Truncate over-long category labels, preserving the original for hover.

    Real vendor names run to 65 characters ("MERCK SHARP & DOHME IDEA GMBH
    (FORMALLY ...)"), which squeezes a horizontal bar chart into a sliver. This
    shortens the displayed label and stashes the full text in a
    ``<column>_full`` column so the tooltip can still show it.

    Truncation happens in the **middle**, not at the end, and collisions are then
    resolved explicitly. Both matter more than they sound:

    * Tail truncation silently merged two different products. "Emtricitabine/Tenofovir
      Disoproxil Fumarate 200/300mg" and the same molecule at "300/200mg" both cut to
      "Emtricitabine/Tenofovir Dis…", Plotly treated them as one x category, stacked
      their bars and drew a cumulative line that went *backwards*. The distinguishing
      part of a pharmaceutical name is usually the dosage at the end, so keeping the
      tail fixes the common case.
    * Middle truncation is not a guarantee, so any labels that still collide get a
      numeric suffix. A duplicated category is a correctness bug, not a cosmetic one -
      it changes what the chart says.

    Parameters
    ----------
    data
        Frame to copy and modify.
    column
        Categorical column to shorten.
    limit
        Maximum displayed characters before text is elided.

    Returns
    -------
    tuple[pandas.DataFrame, bool]
        The frame (a copy) and whether anything was actually shortened.
    """
    if column not in data.columns or not len(data):
        return data, False
    labels = data[column].astype(str)
    if labels.str.len().max() <= limit:
        return data, False

    def _elide(text: str) -> str:
        if len(text) <= limit:
            return text
        # Keep roughly 60% of the budget at the front and the rest at the back, so
        # the distinguishing tail (a dosage, a unit) survives.
        head = max(1, int((limit - 1) * 0.6))
        tail = max(1, limit - 1 - head)
        return f"{text[:head]}…{text[-tail:]}"

    short = labels.map(_elide)

    # Resolve any remaining collisions between labels whose originals differ.
    counts: dict[str, int] = {}
    resolved = []
    seen_full: dict[str, str] = {}
    for full, brief in zip(labels, short):
        if seen_full.get(brief) not in (None, full):
            counts[brief] = counts.get(brief, 1) + 1
            brief = f"{brief} ({counts[brief]})"
        seen_full.setdefault(brief, full)
        resolved.append(brief)

    out = data.copy()
    out[f"{column}_full"] = labels
    out[column] = resolved
    return out, True


def axis_label(column: str) -> str:
    """Turn a snake_case column name into a readable axis title."""
    return _AXIS_LABELS.get(column, column.replace("_", " ").title())


def bar_chart(data: pd.DataFrame, x: str, y: str, title: str = "",
              color: str | None = None, orientation: str = "v",
              text_format: str = ",.0f", height: int = 400,
              colour_override: str | None = None) -> go.Figure:
    """General-purpose bar chart with sensible pharma-dashboard defaults.

    Calling convention: ``x`` is always the **category** and ``y`` always the
    **value**, whichever way the bars point. For ``orientation="h"`` the axes are
    swapped internally - Plotly requires the value on x for horizontal bars, and
    passing them the other way round silently renders an unreadable chart.

    Long category labels are shortened for display and shown in full on hover,
    so a chart of real vendor names stays legible.
    """
    data, shortened = shorten_labels(data, x)
    hover = {f"{x}_full": True, x: False} if shortened else None

    horizontal = orientation == "h"
    plot_x, plot_y = (y, x) if horizontal else (x, y)

    if color and color in data.columns:
        fig = px.bar(data, x=plot_x, y=plot_y, color=color, orientation=orientation,
                     color_discrete_sequence=SEQUENCE, text_auto=text_format,
                     hover_data=hover)
    else:
        fig = px.bar(data, x=plot_x, y=plot_y, orientation=orientation,
                     text_auto=text_format, hover_data=hover)
        fig.update_traces(marker_color=colour_override or PALETTE["primary"])

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title=axis_label(plot_x))
    fig.update_yaxes(title=axis_label(plot_y))

    if horizontal:
        # Leave room for the category labels instead of letting Plotly crop them.
        fig.update_layout(margin=dict(l=min(230, 9 * MAX_LABEL_CHARS)))
    elif shortened:
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(margin=dict(b=140))
    return apply_theme(fig, title=title, height=height)


def line_chart(data: pd.DataFrame, x: str, y: str | list[str], title: str = "",
               color: str | None = None, height: int = 400,
               y_title: str | None = None) -> go.Figure:
    """Time-series line chart with readable axis titles and series names."""
    fig = px.line(data, x=x, y=y, color=color, markers=len(data) <= 40,
                  color_discrete_sequence=SEQUENCE)
    fig.update_traces(line=dict(width=2.5))

    fig.update_xaxes(title=axis_label(x))
    if y_title:
        fig.update_yaxes(title=y_title)
    elif isinstance(y, str):
        fig.update_yaxes(title=axis_label(y))

    # Plotly names the legend "variable" and each series after its raw column
    # when several y columns are plotted; both read as unfinished output.
    if isinstance(y, list):
        for trace in fig.data:
            trace.name = axis_label(trace.name)
        fig.update_layout(legend_title_text="")
    elif color:
        fig.update_layout(legend_title_text=axis_label(color))

    return apply_theme(fig, title=title, height=height)


def heatmap(data: pd.DataFrame, title: str = "", height: int = 420,
            colorscale: str = "Blues", text_format: str = ".1f") -> go.Figure:
    """Annotated heatmap from a wide (already pivoted) frame."""
    fig = go.Figure(go.Heatmap(
        z=data.values, x=list(data.columns), y=list(data.index),
        colorscale=colorscale, text=data.values,
        texttemplate=f"%{{text:{text_format}}}",
        hovertemplate="<b>%{y}</b> / %{x}<br>%{z:.2f}<extra></extra>",
        colorbar=dict(thickness=12),
    ))
    return apply_theme(fig, title=title, height=height)


def donut_chart(labels: list, values: list, title: str = "",
                height: int = 380, colours: list[str] | None = None) -> go.Figure:
    """Donut chart with the total displayed in the centre."""
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=colours or SEQUENCE[:len(labels)],
                    line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{sum(values):,.0f}</b><br>Total",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=15))
    return apply_theme(fig, title=title, height=height, showlegend=True)


def concentration_chart(ranked: pd.DataFrame, category: str, value: str,
                        cumulative: str, title: str = "",
                        height: int = 420, reference_pct: float | None = None,
                        note: str | None = None) -> go.Figure:
    """Ranked bars plus a cumulative-share line, scaled to the data.

    Three layout decisions here exist because the obvious versions collided when the
    exported PNG was actually looked at:

    * **Category labels are shortened.** Untruncated, an ARV product name like
      "Efavirenz/Lamivudine/Tenofovir Disoproxil Fumarate 600/300/300mg" rotated at
      -35 degrees consumed roughly half the figure height and squeezed the bars into
      the top third. Full names remain on hover.
    * **The cumulative axis is scaled to what the curve reaches**, not fixed at 100%,
      and the 80% Pareto reference line is **optional**. A fixed 80% line is the
      conventional default and it is misleading on a flat distribution, where it sits
      far above the curve and makes the chart look broken. Pass ``reference_pct=80``
      only when the curve genuinely gets there - on SCMS product value it reaches 94%
      by the fifteenth product, so it does.
    * **The note sits under the title, not inside the plot.** Anchored to the top
      right of the plot area it overlapped both the secondary axis title and the
      legend. There is no reliably empty region inside a Pareto chart, so the note
      goes where nothing else is drawn.
    """
    ranked, shortened = shorten_labels(ranked, category)
    hover_name = f"{category}_full" if shortened else category

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=ranked[category].astype(str), y=ranked[value], name=axis_label(value),
        marker_color=PALETTE["primary"], customdata=ranked[hover_name],
        hovertemplate="<b>%{customdata}</b><br>%{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=ranked[category].astype(str), y=ranked[cumulative], name="Cumulative share",
        mode="lines+markers", line=dict(color=PALETTE["accent"], width=2.5),
        marker=dict(size=7), customdata=ranked[hover_name],
        hovertemplate="<b>%{customdata}</b><br>Cumulative: %{y:.2f}%<extra></extra>",
    ), secondary_y=True)

    reached = float(ranked[cumulative].max())
    crossed_at = None
    if reference_pct is not None and reached >= reference_pct:
        # No annotation on the line itself. "top left" collided with the primary
        # y-axis zero tick and "right" collided with the secondary axis title, and
        # there is no third position that is safe at every height - so the reference
        # value goes in the subtitle, where it also gets room to say what it means.
        fig.add_hline(y=reference_pct, line_dash="dash",
                      line_color=PALETTE["neutral"], secondary_y=True)
        above = ranked[ranked[cumulative] >= reference_pct]
        if len(above):
            crossed_at = int(ranked.index.get_loc(above.index[0])) + 1

    subtitle = note
    if crossed_at is not None:
        marker = (f"dashed line = {reference_pct:.0f}% of the total, "
                  f"reached at item {crossed_at}")
        subtitle = f"{subtitle}; {marker}" if subtitle else marker
    if subtitle:
        # Plain text, no <br> - this renders on one line beneath the title.
        title = f"{title}<br><sup>{subtitle.replace('<br>', ' · ')}</sup>"

    fig.update_yaxes(title_text=axis_label(value), secondary_y=False)
    fig.update_yaxes(title_text="Cumulative share (%)", secondary_y=True,
                     range=[0, max(reached * 1.15, 1.0)], showgrid=False)
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(margin=dict(b=160, t=90),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    return apply_theme(fig, title=title, height=height)


def confusion_matrix_chart(matrix: pd.DataFrame, title: str = "Confusion Matrix",
                           height: int = 440) -> go.Figure:
    """Annotated confusion matrix (rows = actual, columns = predicted)."""
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=list(matrix.index),
        colorscale="Blues", text=matrix.values, texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="Actual <b>%{y}</b><br>Predicted <b>%{x}</b><br>Count: %{z}<extra></extra>",
        colorbar=dict(thickness=12),
    ))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return apply_theme(fig, title=title, height=height)


def roc_curves_chart(per_class_roc: dict, title: str = "ROC Curves (One-vs-Rest)",
                     height: int = 420) -> go.Figure:
    """Per-class ROC curves with the no-skill diagonal."""
    fig = go.Figure()
    for index, (label, curve) in enumerate(per_class_roc.items()):
        fig.add_trace(go.Scatter(
            x=curve["fpr"], y=curve["tpr"], mode="lines",
            name=f"{label} (AUC={curve['auc']:.3f})",
            line=dict(width=2.5, color=SEQUENCE[index % len(SEQUENCE)]),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="No skill",
        line=dict(dash="dash", color=PALETTE["neutral"], width=1.5),
    ))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return apply_theme(fig, title=title, height=height)


def pr_curves_chart(pr_curves: dict, title: str = "Precision-Recall Curves",
                    height: int = 420) -> go.Figure:
    """Per-class precision-recall curves."""
    fig = go.Figure()
    for index, (label, curve) in enumerate(pr_curves.items()):
        fig.add_trace(go.Scatter(
            x=curve["recall"], y=curve["precision"], mode="lines",
            name=f"{label} (AP={curve['ap']:.3f})",
            line=dict(width=2.5, color=SEQUENCE[index % len(SEQUENCE)]),
        ))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return apply_theme(fig, title=title, height=height)


def feature_importance_chart(importance: pd.DataFrame, top_n: int = 12,
                             title: str = "Feature Importance",
                             height: int = 420) -> go.Figure:
    """Horizontal bar of the most influential model features."""
    data = importance.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=data["importance"], y=data["feature"], orientation="h",
        marker_color=PALETTE["secondary"],
        text=[f"{v:.3f}" for v in data["importance"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Relative importance")
    return apply_theme(fig, title=title, height=height, showlegend=False)


def model_comparison_chart(comparison: pd.DataFrame, metric: str = "test_f1_macro",
                           title: str = "Model Comparison", height: int = 380) -> go.Figure:
    """Grouped bars comparing cross-validation and held-out test performance."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="CV score (mean)", x=comparison["model"], y=comparison["cv_mean"],
        marker_color=PALETTE["neutral"],
        error_y=dict(type="data", array=comparison.get("cv_std"), visible=True),
        text=[f"{v:.3f}" for v in comparison["cv_mean"]], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Test score", x=comparison["model"], y=comparison[metric],
        marker_color=PALETTE["primary"],
        text=[f"{v:.3f}" for v in comparison[metric]], textposition="outside",
    ))
    fig.update_yaxes(title="Score", range=[0, 1.08])
    return apply_theme(fig, title=title, height=height, barmode="group")


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------
def rate_comparison_chart(rates: pd.DataFrame, dimension: str = "level",
                          title: str = "On-Time Rate by Group",
                          target: float | None = None, height: int = 400) -> go.Figure:
    """On-time rate per group, with the group size on the bar.

    Sample size is drawn rather than hidden because on observational data the
    groups are wildly unequal - a 99.2% rate on 253 shipments and an 86.4% rate on
    3,127 are not equally trustworthy, and the chart should not pretend otherwise.

    Parameters
    ----------
    rates
        Output of :func:`src.analytics.experiments.group_rates`.
    target
        Optional target line (e.g. ``config.scms.on_time_target_pct``).
    """
    data = rates.sort_values("on_time_pct")
    fig = go.Figure(go.Bar(
        x=data["on_time_pct"], y=data[dimension].astype(str), orientation="h",
        marker_color=[status_colour(v, target or 90.0) for v in data["on_time_pct"]],
        text=[f"{v:.2f}%  (n={n:,})" for v, n in zip(data["on_time_pct"], data["n"])],
        textposition="outside",
        customdata=np.stack([data["n"], data["late"], data["mean_delay_days"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>On time: %{x:.2f}%<br>"
                       "Shipments: %{customdata[0]:,}<br>"
                       "Late: %{customdata[1]:,}<br>"
                       "Mean delay: %{customdata[2]:.1f} d<extra></extra>"),
    ))
    if target is not None:
        fig.add_vline(x=target, line_dash="dash", line_color=PALETTE["danger"],
                      annotation_text=f"Target {target:.0f}%",
                      annotation_position="top")
    fig.update_xaxes(title="On-time delivery (%)", range=[0, 108])
    return apply_theme(fig, title=title, height=height, showlegend=False)


def stratified_effect_chart(strata: pd.DataFrame, dimension: str, by: str,
                            title: str | None = None, height: int = 400) -> go.Figure:
    """Two group rates plotted within each level of a third variable.

    The stratification chart, and on this dataset it is load bearing: the pooled
    fulfilment-route gap of 11.9 points is 1.9 points before 2011 and 20.5 points
    after. Two converging or diverging lines make an interaction obvious in a way a
    single pooled bar cannot.

    Note that a *diverging* pair like this one is effect modification rather than
    textbook Simpson's paradox - the lines never cross, so the sign never flips. The
    chart handles both; only the wording of the surrounding narrative differs.

    Parameters
    ----------
    strata
        Output of :func:`src.analytics.experiments.stratified_comparison`
        (``strata`` key), with one row per stratum and one column per group.
    """
    # Only the rate columns are series. The frame also carries n_<group> counts,
    # which belong in the hover text rather than on a percentage axis.
    rate_columns = [c for c in strata.columns if c.startswith("on_time_")]
    fig = go.Figure()
    for index, column in enumerate(rate_columns):
        group = column.removeprefix("on_time_")
        counts = strata.get(f"n_{group}")
        fig.add_trace(go.Scatter(
            x=strata[by].astype(str), y=strata[column], mode="lines+markers",
            name=str(group),
            customdata=counts if counts is not None else None,
            line=dict(width=2.8, color=SEQUENCE[index % len(SEQUENCE)]),
            marker=dict(size=11),
            hovertemplate=(f"<b>{group}</b><br>%{{x}}<br>On time: %{{y:.2f}}%"
                           + ("<br>Shipments: %{customdata:,}" if counts is not None else "")
                           + "<extra></extra>"),
        ))
    fig.update_yaxes(title="On-time delivery (%)")
    fig.update_xaxes(title=axis_label(by))
    return apply_theme(
        fig, title=title or f"{axis_label(dimension)} effect within {axis_label(by)}",
        height=height)


def significance_chart(z_result: dict, title: str = "Effect Size with 95% Confidence Interval",
                       height: int = 260) -> go.Figure:
    """Point estimate and confidence interval for the treatment effect.

    The zero line is the decision boundary: an interval that excludes zero is a
    statistically significant result, which makes the chart readable without
    quoting a p-value.
    """
    diff = z_result["absolute_diff_pp"]
    lower, upper = z_result["ci_lower_pp"], z_result["ci_upper_pp"]
    colour = PALETTE["success"] if z_result["significant"] else PALETTE["neutral"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lower, upper], y=[0, 0], mode="lines",
        line=dict(color=colour, width=6), name="95% CI", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[diff], y=[0], mode="markers", name="Observed effect",
        marker=dict(size=17, color=colour, line=dict(color="white", width=2)),
        hovertemplate=f"Effect: {diff:+.2f}pp<br>CI: [{lower:+.2f}, {upper:+.2f}]pp<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text="No effect", annotation_position="top")
    fig.update_xaxes(title="Difference in rate (percentage points)")
    fig.update_yaxes(visible=False, range=[-0.6, 0.6])
    return apply_theme(fig, title=title, height=height, showlegend=False)


def segment_effect_chart(segments: pd.DataFrame, dimension: str = "region",
                         title: str = "Treatment Effect by Segment",
                         height: int = 380) -> go.Figure:
    """Per-segment lift, coloured by whether that segment reached significance."""
    data = segments.sort_values("lift_pp")
    colours = [PALETTE["success"] if sig else PALETTE["neutral"]
               for sig in data["significant"]]
    fig = go.Figure(go.Bar(
        x=data["lift_pp"], y=data[dimension], orientation="h",
        marker_color=colours,
        text=[f"{v:+.2f}pp" for v in data["lift_pp"]], textposition="outside",
        customdata=np.stack([data["p_value"], data["n_control"] + data["n_treatment"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Lift: %{x:+.2f}pp<br>"
                       "p-value: %{customdata[0]:.4f}<br>"
                       "n: %{customdata[1]:,}<extra></extra>"),
    ))
    fig.add_vline(x=0, line_color=PALETTE["neutral"])
    fig.update_xaxes(title="Lift (percentage points)")
    return apply_theme(fig, title=title, height=height, showlegend=False)


__all__ = [
    # pipeline
    "value_funnel_chart", "lateness_funnel_chart", "dwell_time_chart",
    "traceability_chart",
    # generic
    "bar_chart", "line_chart", "heatmap", "donut_chart",
    "concentration_chart",
    "shorten_labels", "axis_label", "MAX_LABEL_CHARS",
    # machine learning
    "confusion_matrix_chart", "roc_curves_chart", "pr_curves_chart",
    "feature_importance_chart", "model_comparison_chart",
    # statistical comparison
    "rate_comparison_chart", "stratified_effect_chart", "significance_chart",
]
