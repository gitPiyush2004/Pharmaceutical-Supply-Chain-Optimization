"""
Shared visual identity for every chart in the platform.

Centralising the palette, the Plotly template and the number formatters means
the dashboard, the notebook and any exported figure all look like one product
rather than eight separate analyses.

Example
-------
>>> from src.viz.theme import apply_theme, PALETTE, fmt_units
>>> fig = apply_theme(fig, title="On-Time Delivery by Region")
>>> fmt_units(54628041)
'54.6M'
"""

from __future__ import annotations

import copy

import plotly.graph_objects as go
import plotly.io as pio

from src.config import get_config

_cfg = get_config().viz

# Semantic colours - referenced by name so intent survives a palette change.
PALETTE: dict[str, str] = dict(_cfg.palette)
#: Ordered colours for categorical series.
SEQUENCE: list[str] = list(_cfg.sequence)
#: Dark-to-light ramp used for the value and lateness funnels.
FUNNEL_SCALE: list[str] = list(_cfg.funnel_scale)


def register_template(name: str = "pharmachain") -> str:
    """Register (once) and activate a Plotly template built from the config palette.

    The base template named in ``viz.template`` is deep-copied and then updated,
    rather than reconstructed from its parts, so Plotly's own defaults for
    anything not overridden here survive intact.
    """
    if name not in pio.templates:
        template = copy.deepcopy(pio.templates[_cfg.template])
        template.layout.update(
            colorway=SEQUENCE,
            font=dict(family="Inter, Segoe UI, Helvetica, sans-serif",
                      size=13, color="#1F2937"),
            title=dict(font=dict(size=17, color="#111827"), x=0.0, xanchor="left"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
            margin=dict(l=60, r=30, t=60, b=50),
        )
        # `update` on the axis objects preserves the base template's tick and
        # spike settings, which a wholesale replacement would discard.
        axis_style = dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB",
                          linecolor="#D1D5DB")
        template.layout.xaxis.update(**axis_style)
        template.layout.yaxis.update(**axis_style)
        pio.templates[name] = template
    pio.templates.default = name
    return name


def apply_theme(fig: go.Figure, title: str | None = None, height: int | None = None,
                showlegend: bool | None = None, **layout_kwargs) -> go.Figure:
    """Apply the platform template and common layout defaults to a figure.

    Parameters
    ----------
    fig
        Figure to restyle (modified in place and returned for chaining).
    title
        Optional chart title.
    height
        Optional pixel height.
    showlegend
        Optional legend override.
    **layout_kwargs
        Any additional ``fig.update_layout`` arguments.
    """
    register_template()
    fig.update_layout(template="pharmachain", **layout_kwargs)
    if title is not None:
        fig.update_layout(title=title)
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=12,
                                      bordercolor=PALETTE["neutral"]))
    return fig


# ---------------------------------------------------------------------------
# Number formatting - used in KPI cards, hover text and tables
# ---------------------------------------------------------------------------
def fmt_units(value: float, decimals: int = 1) -> str:
    """Compact unit count: ``54628041 -> '54.6M'``."""
    if value is None:
        return "-"
    value = float(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.{decimals}f}{suffix}"
    return f"{value:,.0f}"


def fmt_currency(value: float, decimals: int = 1) -> str:
    """Compact USD amount: ``12400000 -> '$12.4M'``."""
    if value is None:
        return "-"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            return f"{sign}${value / threshold:.{decimals}f}{suffix}"
    return f"{sign}${value:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Percentage already expressed on a 0-100 scale."""
    return "-" if value is None else f"{float(value):.{decimals}f}%"


def fmt_days(value: float, decimals: int = 1) -> str:
    """Day count with unit suffix."""
    return "-" if value is None else f"{float(value):.{decimals}f} d"


def status_colour(value: float, target: float, higher_is_better: bool = True) -> str:
    """Traffic-light colour for a metric measured against a target.

    Green at or beyond target, amber within 5% of it, red otherwise.
    """
    if higher_is_better:
        if value >= target:
            return PALETTE["success"]
        return PALETTE["warning"] if value >= target * 0.95 else PALETTE["danger"]
    if value <= target:
        return PALETTE["success"]
    return PALETTE["warning"] if value <= target * 1.05 else PALETTE["danger"]


__all__ = [
    "PALETTE", "SEQUENCE", "FUNNEL_SCALE",
    "register_template", "apply_theme",
    "fmt_units", "fmt_currency", "fmt_pct", "fmt_days",
    "status_colour",
]
