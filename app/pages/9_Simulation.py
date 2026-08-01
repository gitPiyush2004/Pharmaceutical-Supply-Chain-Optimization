"""
Interactive Simulation page.

Seven operational levers, one propagation model. Move a lever and every downstream
KPI and cost line updates, so a planner can see the trade-off between service,
quality and working capital before committing to a change.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.analytics import simulation as sim
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct

page_setup(
    title="Interactive Scenario Simulation",
    icon="🎛️",
    subtitle="Move operational levers and watch quality, service and cost respond",
)

levers_meta = sim.lever_definitions()
presets = sim.preset_scenarios()

# ---------------------------------------------------------------------------
section(
    "Scenario Levers",
    "Start from a preset or set each lever manually. Values marked as measured are "
    "calibrated from the observed data rather than invented.",
)

preset_name = st.selectbox("Preset scenario", list(presets), index=0)


def _clamp(value: float, low: float, high: float) -> float:
    """Keep a preset value inside the slider's declared range."""
    return float(min(max(value, low), high))


# Sliders own their state via `key`, so a preset is applied by writing into
# session state and letting Streamlit re-render - never by passing `value`
# alongside an existing key, which Streamlit warns about.
if st.button("Load preset", type="secondary"):
    for _, row in levers_meta.iterrows():
        lever = row["lever"]
        if lever in presets[preset_name]:
            st.session_state[f"lever_{lever}"] = _clamp(
                presets[preset_name][lever], row["min"], row["max"])

st.sidebar.markdown("### Operational levers")
lever_values: dict[str, float] = {}
for _, row in levers_meta.iterrows():
    lever = row["lever"]
    state_key = f"lever_{lever}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _clamp(
            presets[preset_name].get(lever, row["default"]), row["min"], row["max"])
    lever_values[lever] = st.sidebar.slider(
        row["label"] + (f" ({row['unit']})" if row["unit"] else ""),
        min_value=float(row["min"]), max_value=float(row["max"]),
        step=float(row["step"]), key=state_key,
        help=f"{row['description']}"
             + (" Baseline is measured from the data." if row["is_measured"] else ""),
    )

result = sim.simulate(lever_values)
kpis, financials, alerts = result["kpis"], result["financials"], result["alerts"]

# ---------------------------------------------------------------------------
section("Scenario Outcome", "Each card shows the scenario value against baseline.")


_LABELS = {
    "qa_pass_rate_pct": "QA Pass Rate", "end_to_end_yield_pct": "End-to-End Yield",
    "avg_potency_pct": "Average Potency", "out_of_spec_pct": "Out of Specification",
    "service_level_pct": "Service Level", "stockout_risk_pct": "Stock-Out Risk",
    "expiry_risk_pct": "Expiry Risk", "avg_cycle_time_days": "Cycle Time",
    "units_dispensed": "Units Dispensed", "inventory_value_usd": "Inventory Value",
    "total_cost_usd": "Total Cost", "value_lost_usd": "Value Lost",
    "holding_cost_usd": "Holding Cost", "quality_cost_usd": "Quality Cost",
    "stockout_cost_usd": "Stock-Out Cost", "expiry_cost_usd": "Expiry Cost",
}


def _format(value: float, unit: str) -> str:
    """Format a simulated value according to the unit the simulator reports."""
    if unit == "USD":
        return fmt_currency(value)
    if unit == "%":
        return fmt_pct(value)
    if unit == "days":
        return f"{value:,.1f} d"
    return f"{value:,.0f}"


def _cards(keys: list[str], source: dict) -> list[dict]:
    """Build KPI cards from the simulator's baseline/scenario comparison entries.

    Each entry carries ``baseline``, ``scenario``, ``delta``, ``unit`` and an
    ``improved`` flag. ``improved`` drives the delta colour, which matters because
    a falling cost and a falling yield are both negative deltas but only one is
    good news.
    """
    cards: list[dict] = []
    for key in keys:
        entry = source.get(key)
        if not isinstance(entry, dict):
            continue
        unit = entry.get("unit", "")
        delta = entry.get("delta")
        delta_text = None
        if delta is not None:
            sign = "+" if delta > 0 else ""
            delta_text = (f"{sign}{_format(delta, unit)} vs base" if unit == "USD"
                          else f"{delta:+.2f} pp vs base" if unit == "%"
                          else f"{delta:+,.1f} vs base")
        cards.append({
            "label": _LABELS.get(key, key.replace("_", " ").title()),
            "value": _format(entry.get("scenario", 0.0), unit),
            "delta": delta_text,
            "delta_good": entry.get("improved"),
        })
    return cards


quality_cards = _cards(["qa_pass_rate_pct", "end_to_end_yield_pct",
                        "avg_potency_pct", "out_of_spec_pct"], kpis)
if quality_cards:
    st.markdown("**Quality and yield**")
    kpi_row(quality_cards)

service_cards = _cards(["service_level_pct", "stockout_risk_pct",
                        "expiry_risk_pct", "avg_cycle_time_days"], kpis)
if service_cards:
    st.markdown("**Service and risk**")
    kpi_row(service_cards)

cost_cards = _cards(["total_cost_usd", "value_lost_usd", "holding_cost_usd",
                     "quality_cost_usd"], financials)
if cost_cards:
    st.markdown("**Cost impact**")
    kpi_row(cost_cards)

# ---------------------------------------------------------------------------
_SEVERITY_KIND = {"High": "danger", "Medium": "warning", "Low": "insight"}

if alerts:
    section("Alerts", "Thresholds breached by this scenario.")
    for alert in alerts:
        callout(alert.get("message", str(alert)),
                kind=_SEVERITY_KIND.get(alert.get("severity", ""), "warning"),
                title=f"{alert.get('severity', '')} - {alert.get('area', '')}".strip(" -"))
else:
    section("Alerts")
    callout("No thresholds breached by this scenario.", kind="success")

# ---------------------------------------------------------------------------
section("Baseline versus Scenario")
chart(charts.scenario_impact_chart(kpis, title="KPI Movement Against Baseline"))

# ---------------------------------------------------------------------------
section(
    "Which Lever Matters Most",
    "Each lever is swept across its full range while the others hold at baseline. "
    "The longest bar is the biggest source of variation in the chosen outcome.",
)

kpi_choice = st.selectbox(
    "Outcome to analyse",
    ["total_cost_usd", "value_lost_usd", "holding_cost_usd", "stockout_cost_usd",
     "expiry_cost_usd", "quality_cost_usd"],
    format_func=lambda k: k.replace("_", " ").replace("usd", "(USD)").title(),
)
tornado = sim.tornado_analysis(kpi=kpi_choice)
chart(charts.tornado_chart(tornado, kpi_label=kpi_choice.replace("_", " ").title()))
show_table(tornado, height=280)

if len(tornado):
    dominant = tornado.iloc[0]
    insight(
        f"**{dominant['label']}** dominates, swinging "
        f"{kpi_choice.replace('_', ' ')} by "
        f"{fmt_currency(abs(dominant['swing']))} "
        f"({abs(dominant['swing_pct']):.1f}%) across its plausible range. "
        "That makes it the lever to invest management attention in first - the "
        "others move the outcome far less per unit of effort."
    )

# ---------------------------------------------------------------------------
section("Single-Lever Sensitivity")

col1, col2 = st.columns([1, 3], gap="large")
with col1:
    lever_choice = st.selectbox(
        "Lever", levers_meta["lever"].tolist(),
        format_func=lambda l: levers_meta.loc[
            levers_meta["lever"] == l, "label"].iloc[0])
with col2:
    sensitivity = sim.sensitivity_analysis(lever_choice, kpi=kpi_choice)
    label = levers_meta.loc[levers_meta["lever"] == lever_choice, "label"].iloc[0]
    chart(charts.sensitivity_chart(
        sensitivity, lever_label=label,
        kpi_label=kpi_choice.replace("_", " ").title()))

# ---------------------------------------------------------------------------
section("Compare Preset Scenarios")

comparison = sim.compare_scenarios(presets)
show_table(comparison, height=260)
download_button(comparison, "scenario_comparison.csv", "Download scenario comparison")

insight(
    "The preset comparison is the planning conversation in one table. The stress and "
    "crisis scenarios are deliberately pessimistic - a demand surge arriving at the "
    "same time as a supplier failure and a heatwave - because that combination is "
    "what actually breaks pharmaceutical supply chains, and it is far more "
    "informative than testing each shock in isolation."
)

# ---------------------------------------------------------------------------
methodology("""
**How propagation works.** The simulator does not re-run the data generator. It
starts from the measured baseline - actual QA pass rate, yield, potency, cycle time
and inventory position - and applies elasticities declared in
`config/config.yaml` (`simulation.elasticity`) to move each downstream KPI:

- `reliability_to_qa_yield` (0.45) - one percentage point of supplier reliability
  moves QA yield by 0.45 points.
- `delay_to_expiry_risk` (0.32) - each extra day of dwell adds 0.32 points of
  expiry risk.
- `demand_to_stockout` (0.55) - one percentage point of demand adds 0.55 points of
  stock-out risk.

Storage temperature and humidity feed the same Arrhenius-style degradation
function used on the Drug Stability page, so potency responds with consistent
physics rather than a separate approximation.

**Costs.** Financial lines use `config.economics`: holding cost at 22% of unit cost
per year, stock-out penalty per unit, expiry write-off net of 5% salvage, and
quality cost from rejected volume. Total cost is their sum.

**Tornado analysis.** Each lever is swept from its minimum to its maximum with all
others held at baseline, and the resulting swing in the chosen outcome is recorded.
Bars are sorted by absolute swing, so the ranking is a genuine sensitivity ranking
rather than a guess at what matters.

**What this is and is not.** This is a deterministic elasticity model for exploring
directional trade-offs quickly - the kind of tool that supports a planning
conversation. It is not a discrete-event or Monte Carlo simulation and does not
model queueing, capacity constraints or stochastic variance. Elasticities are
first-order and linear, so extreme lever positions should be read as indicative.
""")

sidebar_about()
