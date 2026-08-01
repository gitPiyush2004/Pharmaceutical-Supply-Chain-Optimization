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
def bar_chart(data: pd.DataFrame, x: str, y: str, title: str = "",
              color: str | None = None, orientation: str = "v",
              text_format: str = ",.0f", height: int = 400,
              colour_override: str | None = None) -> go.Figure:
    """General-purpose bar chart with sensible pharma-dashboard defaults."""
    if color and color in data.columns:
        fig = px.bar(data, x=x, y=y, color=color, orientation=orientation,
                     color_discrete_sequence=SEQUENCE, text_auto=text_format)
    else:
        fig = px.bar(data, x=x, y=y, orientation=orientation, text_auto=text_format)
        fig.update_traces(marker_color=colour_override or PALETTE["primary"])
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_theme(fig, title=title, height=height)


def line_chart(data: pd.DataFrame, x: str, y: str | list[str], title: str = "",
               color: str | None = None, height: int = 400,
               y_title: str | None = None) -> go.Figure:
    """Time-series line chart."""
    fig = px.line(data, x=x, y=y, color=color, markers=len(data) <= 40,
                  color_discrete_sequence=SEQUENCE)
    fig.update_traces(line=dict(width=2.5))
    if y_title:
        fig.update_yaxes(title=y_title)
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


def scatter_chart(data: pd.DataFrame, x: str, y: str, title: str = "",
                  color: str | None = None, size: str | None = None,
                  trendline: bool = False, height: int = 420) -> go.Figure:
    """Scatter plot with an optional OLS trendline."""
    fig = px.scatter(
        data, x=x, y=y, color=color, size=size,
        color_discrete_sequence=SEQUENCE,
        color_discrete_map=RISK_COLOURS if color and "risk" in str(color).lower() else None,
        trendline="ols" if trendline else None,
        trendline_color_override=PALETTE["danger"],
        opacity=0.65,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
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


def box_chart(data: pd.DataFrame, x: str, y: str, title: str = "",
              height: int = 400, color: str | None = None) -> go.Figure:
    """Box plot for distribution comparison across categories."""
    fig = px.box(data, x=x, y=y, color=color or x, points="outliers",
                 color_discrete_sequence=SEQUENCE)
    return apply_theme(fig, title=title, height=height, showlegend=color is not None)


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


def gauge_chart(value: float, title: str, target: float, suffix: str = "%",
                height: int = 260, higher_is_better: bool = True,
                max_value: float | None = None) -> go.Figure:
    """Gauge showing one KPI against its target."""
    maximum = max_value if max_value is not None else max(value, target) * 1.35
    colour = status_colour(value, target, higher_is_better)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": suffix, "font": {"size": 30}},
        delta={"reference": target, "suffix": suffix,
               "increasing": {"color": PALETTE["success"] if higher_is_better else PALETTE["danger"]},
               "decreasing": {"color": PALETTE["danger"] if higher_is_better else PALETTE["success"]}},
        gauge={
            "axis": {"range": [0, maximum], "tickwidth": 1},
            "bar": {"color": colour, "thickness": 0.7},
            "bgcolor": "#F1F5F9",
            "borderwidth": 0,
            "threshold": {"line": {"color": PALETTE["danger"], "width": 3},
                          "thickness": 0.85, "value": target},
        },
        title={"text": title, "font": {"size": 13}},
    ))
    return apply_theme(fig, height=height, margin=dict(l=20, r=20, t=50, b=10))


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------
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
def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame,
                   date_col: str = "date", value_col: str = "units_demanded",
                   title: str = "Demand Forecast", height: int = 440) -> go.Figure:
    """Historical demand plus forecast with a shaded confidence band."""
    fig = go.Figure()

    # Confidence band first so the lines draw on top of it.
    # `.to_numpy()` keeps the dates as a datetime64 array. Building the band with
    # `list(series)` instead would yield pandas Timestamps, which Plotly's static
    # image writer cannot serialise - it renders in the browser but fails on export.
    band_dates = forecast[date_col].to_numpy()
    fig.add_trace(go.Scatter(
        x=np.concatenate([band_dates, band_dates[::-1]]),
        y=np.concatenate([forecast["upper_ci"].to_numpy(),
                          forecast["lower_ci"].to_numpy()[::-1]]),
        fill="toself", fillcolor="rgba(11,110,153,0.13)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        name="Confidence interval",
    ))
    fig.add_trace(go.Scatter(
        x=history[date_col], y=history[value_col], mode="lines",
        name="Actual", line=dict(color=PALETTE["primary"], width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=forecast[date_col], y=forecast["forecast"], mode="lines+markers",
        name="Forecast", line=dict(color=PALETTE["accent"], width=2.5, dash="dash"),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
    ))

    if len(history):
        # `add_vline` stores x in layout.shapes, and a pandas Timestamp there is
        # not JSON-serialisable by Plotly's static image writer (it renders fine
        # in the browser, so this only surfaces on export). Pass an ISO string.
        boundary = history[date_col].max()
        if isinstance(boundary, pd.Timestamp):
            boundary = boundary.isoformat()
        fig.add_vline(x=boundary, line_dash="dot",
                      line_color=PALETTE["neutral"],
                      annotation_text="Forecast start", annotation_position="top left")
    fig.update_yaxes(title="Units")
    return apply_theme(fig, title=title, height=height)


def decomposition_chart(decomposition: pd.DataFrame,
                        title: str = "Demand Decomposition", height: int = 560) -> go.Figure:
    """Stacked observed / trend / seasonal / residual panels."""
    components = [("observed", PALETTE["primary"]), ("trend", PALETTE["accent"]),
                  ("seasonal", PALETTE["secondary"]), ("residual", PALETTE["neutral"])]
    available = [(name, colour) for name, colour in components
                 if name in decomposition.columns]

    fig = make_subplots(rows=len(available), cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[name.title() for name, _ in available])
    for index, (name, colour) in enumerate(available, start=1):
        fig.add_trace(go.Scatter(
            x=decomposition["date"], y=decomposition[name], mode="lines",
            name=name.title(), line=dict(color=colour, width=2),
        ), row=index, col=1)
    return apply_theme(fig, title=title, height=height, showlegend=False)


# ---------------------------------------------------------------------------
# A/B testing
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
def tornado_chart(tornado: pd.DataFrame, kpi_label: str = "Total Cost",
                  title: str | None = None, height: int = 400) -> go.Figure:
    """Tornado chart ranking levers by their influence on a KPI."""
    data = tornado.sort_values("swing")
    fig = go.Figure(go.Bar(
        x=data["swing"], y=data["label"], orientation="h",
        marker_color=PALETTE["primary"],
        text=[f"{v:,.0f}" for v in data["swing"]], textposition="outside",
        customdata=np.stack([data["low_value"], data["high_value"],
                             data["swing_pct"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Range: %{customdata[0]} to %{customdata[1]}<br>"
                       "Swing: %{x:,.0f} (%{customdata[2]:.1f}% of baseline)<extra></extra>"),
    ))
    fig.update_xaxes(title=f"Swing in {kpi_label}")
    return apply_theme(fig, title=title or f"Lever Sensitivity - {kpi_label}",
                       height=height, showlegend=False)


def scenario_impact_chart(kpis: dict, title: str = "Baseline vs Scenario",
                          height: int = 420) -> go.Figure:
    """Diverging bars showing the percentage change in each KPI under a scenario."""
    names, deltas, colours = [], [], []
    for key, kpi in kpis.items():
        names.append(key.replace("_pct", "").replace("_usd", "").replace("_", " ").title())
        deltas.append(kpi["delta_pct"])
        colours.append(PALETTE["success"] if kpi.get("improved") else PALETTE["danger"])

    fig = go.Figure(go.Bar(
        x=deltas, y=names, orientation="h", marker_color=colours,
        text=[f"{v:+.1f}%" for v in deltas], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Change: %{x:+.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=PALETTE["neutral"])
    fig.update_xaxes(title="Change vs baseline (%)")
    return apply_theme(fig, title=title, height=height, showlegend=False)


def sensitivity_chart(sensitivity: pd.DataFrame, lever_label: str, kpi_label: str,
                      height: int = 380) -> go.Figure:
    """Response curve of a KPI across one lever's full range."""
    fig = go.Figure(go.Scatter(
        x=sensitivity["lever_value"], y=sensitivity["kpi_value"],
        mode="lines+markers", line=dict(color=PALETTE["primary"], width=2.5),
        marker=dict(size=7), fill="tozeroy", fillcolor="rgba(11,110,153,0.10)",
        hovertemplate=f"{lever_label}: %{{x}}<br>{kpi_label}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_xaxes(title=lever_label)
    fig.update_yaxes(title=kpi_label)
    return apply_theme(fig, title=f"{kpi_label} vs {lever_label}",
                       height=height, showlegend=False)


# ---------------------------------------------------------------------------
# Stability and risk
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
    "funnel_chart", "funnel_dropoff_chart", "stage_delay_chart", "funnel_comparison_chart",
    "bar_chart", "line_chart", "heatmap", "scatter_chart", "donut_chart", "box_chart",
    "pareto_chart", "gauge_chart",
    "confusion_matrix_chart", "roc_curves_chart", "pr_curves_chart",
    "feature_importance_chart", "model_comparison_chart",
    "forecast_chart", "decomposition_chart",
    "ab_comparison_chart", "significance_chart", "segment_effect_chart",
    "tornado_chart", "scenario_impact_chart", "sensitivity_chart",
    "risk_distribution_chart", "potency_distribution_chart", "condition_effect_chart",
]
