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

from src.config import get_config
from src.viz.theme import (ARM_COLOURS, FUNNEL_SCALE, PALETTE, RISK_COLOURS,
                           SEQUENCE, apply_theme, fmt_units, status_colour)


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------
def funnel_chart(summary: pd.DataFrame, title: str = "Supply Chain Funnel",
                 height: int = 480) -> go.Figure:
    """Eight-stage supply chain funnel with conversion labels.

    Parameters
    ----------
    summary
        Output of :func:`src.analytics.funnel.funnel_summary`.
    """
    # Plotly's own `textinfo="value"` renders 86196541 as "86.19654M". Building the
    # label explicitly keeps it readable at a glance, which is the whole point of
    # putting the number inside the band.
    labels = [f"{fmt_units(units)}<br>{share:.0f}%"
              for units, share in zip(summary["units"],
                                      summary["cumulative_conversion_pct"])]
    fig = go.Figure(go.Funnel(
        y=summary["stage"],
        x=summary["units"],
        text=labels,
        textposition="inside",
        textinfo="text",
        opacity=0.95,
        marker=dict(color=FUNNEL_SCALE[:len(summary)],
                    line=dict(width=1, color="white")),
        connector=dict(line=dict(color=PALETTE["neutral"], width=1)),
        hovertemplate=("<b>%{y}</b><br>Units: %{x:,.0f}<br>"
                       "Share of intake: %{percentInitial:.1%}<extra></extra>"),
    ))
    return apply_theme(fig, title=title, height=height, showlegend=False)


def funnel_dropoff_chart(summary: pd.DataFrame,
                         title: str = "Unit Loss by Stage") -> go.Figure:
    """Bar chart of stage-to-stage drop-off, with the worst stage highlighted."""
    data = summary.iloc[1:].copy()  # Procurement has no predecessor
    worst = data["dropoff_pct"].max()
    colours = [PALETTE["danger"] if v == worst else PALETTE["primary"]
               for v in data["dropoff_pct"]]

    fig = go.Figure(go.Bar(
        x=data["stage"], y=data["dropoff_pct"], marker_color=colours,
        text=[f"{v:.1f}%" for v in data["dropoff_pct"]], textposition="outside",
        customdata=np.stack([data["units_lost"], data["value_lost_usd"]], axis=-1),
        hovertemplate=("<b>%{x}</b><br>Drop-off: %{y:.2f}%<br>"
                       "Units lost: %{customdata[0]:,.0f}<br>"
                       "Value lost: $%{customdata[1]:,.0f}<extra></extra>"),
    ))
    fig.update_yaxes(title="Drop-off (%)")
    return apply_theme(fig, title=title, height=380, showlegend=False)


def stage_delay_chart(delays: pd.DataFrame,
                      title: str = "Average Dwell Time by Stage") -> go.Figure:
    """Horizontal bar of stage dwell time, coloured against the bottleneck threshold."""
    threshold = float(get_config().funnel.bottleneck.delay_days_threshold)
    data = delays.sort_values("avg_delay_days")
    colours = [PALETTE["danger"] if v > threshold else PALETTE["secondary"]
               for v in data["avg_delay_days"]]

    fig = go.Figure(go.Bar(
        x=data["avg_delay_days"], y=data["stage"], orientation="h",
        marker_color=colours,
        text=[f"{v:.1f}d" for v in data["avg_delay_days"]], textposition="outside",
        customdata=np.stack([data["median_delay_days"], data["p90_delay_days"],
                             data["share_of_cycle_pct"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Mean: %{x:.1f} d<br>"
                       "Median: %{customdata[0]:.1f} d<br>P90: %{customdata[1]:.1f} d<br>"
                       "Share of cycle: %{customdata[2]:.1f}%<extra></extra>"),
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"Bottleneck threshold ({threshold:.0f}d)",
                  annotation_position="top")
    fig.update_xaxes(title="Days")
    return apply_theme(fig, title=title, height=400, showlegend=False)


def funnel_comparison_chart(conversion: pd.DataFrame, dimension: str,
                            title: str | None = None) -> go.Figure:
    """Multi-series line showing cumulative conversion by segment across stages."""
    stages = list(get_config().funnel.stages)
    fig = go.Figure()
    for index, (_, row) in enumerate(conversion.iterrows()):
        fig.add_trace(go.Scatter(
            x=stages, y=[row[stage] for stage in stages],
            mode="lines+markers", name=str(row[dimension]),
            line=dict(width=2.5, color=SEQUENCE[index % len(SEQUENCE)]),
            marker=dict(size=7),
            hovertemplate=f"<b>{row[dimension]}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_yaxes(title="Units remaining (% of intake)")
    return apply_theme(fig, title=title or f"Funnel Conversion by {dimension.title()}",
                       height=420)


# ---------------------------------------------------------------------------
# Generic building blocks
# ---------------------------------------------------------------------------
#: Longest category label rendered on an axis before it is shortened.
MAX_LABEL_CHARS = 28

#: Readable axis titles for the column names that recur across charts.
_AXIS_LABELS: dict[str, str] = {
    "on_time_pct": "On-time delivery (%)", "avg_delay_days": "Average delay (days)",
    "dropoff_pct": "Drop-off (%)", "value_lost_usd": "Value lost (USD)",
    "line_value_usd": "Commodity value (USD)", "freight_cost_usd": "Freight (USD)",
    "median_freight_per_kg_usd": "Median freight (USD/kg)",
    "annual_value_usd": "Estimated annual value (USD)",
    "value_usd": "Estimated annual value (USD)",
    "coverage_pct": "Milestone coverage (%)", "shipments": "Shipments",
    "units": "Units", "batches": "Batches", "importance": "Importance",
    "missing_pct": "Missing (%)", "outlier_pct": "Outliers (%)",
    "capture_rate_pct": "Late deliveries captured (%)",
    "targeted_pct": "Shipments reviewed (%)", "precision_pct": "Precision (%)",
    "qa_pass_rate_pct": "QA pass rate (%)", "fill_rate_pct": "Fill rate (%)",
    "end_to_end_yield_pct": "End-to-end yield (%)",
    "out_of_spec_pct": "Out of specification (%)",
    "avg_potency_pct": "Mean potency (%)", "probability": "Probability",
    "vendor": "Vendor", "region": "Region", "shipment_mode": "Transport mode",
    "brand_name": "Product", "stage": "Stage", "bucket": "Bucket",
    "experiment": "Intervention", "area": "Area", "feature": "Feature",
}


def shorten_labels(data: pd.DataFrame, column: str,
                   limit: int = MAX_LABEL_CHARS) -> tuple[pd.DataFrame, bool]:
    """Truncate over-long category labels, preserving the original for hover.

    Real vendor names run to 65 characters ("MERCK SHARP & DOHME IDEA GMBH
    (FORMALLY ...)"), which squeezes a horizontal bar chart into a sliver. This
    shortens the displayed label and stashes the full text in a
    ``<column>_full`` column so the tooltip can still show it.

    Parameters
    ----------
    data
        Frame to copy and modify.
    column
        Categorical column to shorten.
    limit
        Maximum displayed characters before an ellipsis is appended.

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

    out = data.copy()
    out[f"{column}_full"] = labels
    out[column] = labels.where(
        labels.str.len() <= limit, labels.str.slice(0, limit - 1) + "…")
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


def pareto_chart(data: pd.DataFrame, category: str, value: str,
                 cumulative: str, title: str = "", height: int = 420) -> go.Figure:
    """Pareto chart: ranked bars plus a cumulative-share line on a second axis.

    The standard tool for "which few drivers explain most of the problem".
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=data[category], y=data[value], name=value.replace("_", " ").title(),
        marker_color=PALETTE["primary"],
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=data[category], y=data[cumulative], name="Cumulative %",
        mode="lines+markers", line=dict(color=PALETTE["accent"], width=2.5),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color=PALETTE["neutral"],
                  secondary_y=True, annotation_text="80%")
    fig.update_yaxes(title_text="Value", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", range=[0, 105], secondary_y=True)
    fig.update_xaxes(tickangle=-30)
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
# Forecasting
# ---------------------------------------------------------------------------
def ab_comparison_chart(summary: pd.DataFrame, metric_name: str = "Success Rate",
                        title: str = "Control vs Treatment", height: int = 400) -> go.Figure:
    """Arm success rates with Wilson confidence intervals as error bars."""
    fig = go.Figure(go.Bar(
        x=summary["arm"], y=summary["success_rate_pct"],
        marker_color=[ARM_COLOURS.get(a, PALETTE["primary"]) for a in summary["arm"]],
        error_y=dict(
            type="data", symmetric=False,
            array=summary["ci_upper_pct"] - summary["success_rate_pct"],
            arrayminus=summary["success_rate_pct"] - summary["ci_lower_pct"],
            color=PALETTE["neutral"], thickness=1.6, width=8),
        text=[f"{v:.2f}%" for v in summary["success_rate_pct"]], textposition="outside",
        customdata=np.stack([summary["subjects"], summary["successes"]], axis=-1),
        hovertemplate=("<b>%{x}</b><br>Rate: %{y:.2f}%<br>"
                       "Subjects: %{customdata[0]:,}<br>"
                       "Successes: %{customdata[1]:,}<extra></extra>"),
    ))
    fig.update_yaxes(title=f"{metric_name} (%)")
    return apply_theme(fig, title=title, height=height, showlegend=False)


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


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def risk_distribution_chart(counts: pd.Series, title: str = "Batch Risk Distribution",
                            height: int = 340) -> go.Figure:
    """Bar chart of batch counts by risk tier, using the shared risk palette."""
    order = [tier for tier in ("Low", "Medium", "High", "Critical") if tier in counts.index]
    values = [counts[tier] for tier in order]
    fig = go.Figure(go.Bar(
        x=order, y=values, marker_color=[RISK_COLOURS[tier] for tier in order],
        text=[f"{v:,}" for v in values], textposition="outside",
        hovertemplate="<b>%{x} risk</b><br>Batches: %{y:,}<extra></extra>",
    ))
    fig.update_yaxes(title="Batches")
    return apply_theme(fig, title=title, height=height, showlegend=False)


def potency_distribution_chart(batches: pd.DataFrame, title: str = "Potency Distribution",
                               height: int = 400) -> go.Figure:
    """Histogram of batch potency with the specification limit marked."""
    spec = float(get_config().stability.potency_spec_min)
    fig = go.Figure(go.Histogram(
        x=batches["potency_pct"], nbinsx=45, marker_color=PALETTE["primary"],
        opacity=0.85, hovertemplate="Potency: %{x:.1f}%<br>Batches: %{y}<extra></extra>",
    ))
    fig.add_vline(x=spec, line_dash="dash", line_color=PALETTE["danger"], line_width=2,
                  annotation_text=f"Spec limit ({spec:.0f}%)", annotation_position="top left")
    out_of_spec = 100 * (batches["potency_pct"] < spec).mean()
    fig.add_annotation(
        x=0.02, y=0.95, xref="paper", yref="paper", showarrow=False, align="left",
        text=f"<b>{out_of_spec:.1f}%</b> of batches out of specification",
        font=dict(size=12, color=PALETTE["danger"]),
    )
    fig.update_xaxes(title="Potency (% of label claim)")
    fig.update_yaxes(title="Batches")
    return apply_theme(fig, title=title, height=height, showlegend=False)


def condition_effect_chart(effect: pd.DataFrame, x_col: str, title: str,
                           x_label: str, height: int = 400) -> go.Figure:
    """Dual-axis view of mean potency and out-of-spec rate across a condition.

    Accepts either ``avg_potency_pct`` (the name the stability module emits) or
    ``avg_potency``, so the chart is not coupled to one caller's naming.
    """
    potency_col = next(
        (c for c in ("avg_potency_pct", "avg_potency") if c in effect.columns), None)
    if potency_col is None:
        raise KeyError(
            "condition_effect_chart needs an 'avg_potency_pct' or 'avg_potency' "
            f"column; got {list(effect.columns)}"
        )

    # Cohorts (cold chain versus ambient) are plotted as separate potency traces
    # so a reader can see that the two behave differently rather than seeing a
    # misleading pooled average.
    cohorts = ([c for c in effect["cohort"].dropna().unique()]
               if "cohort" in effect.columns else [None])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    totals = (effect.groupby(x_col, as_index=False)["batches"].sum()
              if "cohort" in effect.columns else effect)
    fig.add_trace(go.Bar(
        x=totals[x_col].astype(str), y=totals["batches"], name="Batches",
        marker_color="rgba(107,114,128,0.30)",
        hovertemplate="%{x}<br>Batches: %{y:,}<extra></extra>",
    ), secondary_y=True)

    for index, cohort in enumerate(cohorts):
        subset = effect if cohort is None else effect[effect["cohort"] == cohort]
        subset = subset.sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=subset[x_col].astype(str), y=subset[potency_col],
            name="Mean potency" if cohort is None else f"Potency - {cohort}",
            mode="lines+markers",
            line=dict(color=SEQUENCE[index % len(SEQUENCE)], width=2.5),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Potency: %{y:.2f}%<extra></extra>",
        ), secondary_y=False)

    effect = effect.groupby(x_col, as_index=False).agg(
        out_of_spec_pct=("out_of_spec_pct", "mean")
    ) if "cohort" in effect.columns and "out_of_spec_pct" in effect.columns else effect
    if "out_of_spec_pct" in effect.columns:
        fig.add_trace(go.Scatter(
            x=effect[x_col].astype(str), y=effect["out_of_spec_pct"], name="Out of spec (%)",
            mode="lines+markers", line=dict(color=PALETTE["danger"], width=2.5, dash="dot"),
            marker=dict(size=7),
            hovertemplate="%{x}<br>Out of spec: %{y:.1f}%<extra></extra>",
        ), secondary_y=False)

    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Potency (%) / Out of spec (%)", secondary_y=False)
    fig.update_yaxes(title="Batches", secondary_y=True, showgrid=False)
    return apply_theme(fig, title=title, height=height)


__all__ = [
    "funnel_chart",
    "funnel_dropoff_chart",
    "stage_delay_chart",
    "funnel_comparison_chart",
    "bar_chart",
    "line_chart",
    "heatmap",
    "donut_chart",
    "pareto_chart",
    "confusion_matrix_chart",
    "roc_curves_chart",
    "pr_curves_chart",
    "feature_importance_chart",
    "model_comparison_chart",
    "ab_comparison_chart",
    "significance_chart",
    "segment_effect_chart",
    "risk_distribution_chart",
    "potency_distribution_chart",
    "condition_effect_chart",
]
