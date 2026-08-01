"""
Demand Forecasting page.

Three forecasting methods, compared by backtest rather than asserted. The model
used for the forward view is the one that actually won on held-out months.
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
from src.analytics import forecasting as fc
from src.data import loader
from src.viz import charts
from src.viz.theme import fmt_pct, fmt_units

page_setup(
    title="Demand Forecasting",
    icon="📈",
    subtitle="Holt-Winters, linear trend and moving average - selected by backtest",
)


@st.cache_data(show_spinner="Loading demand history...")
def _demand():
    return loader.load_demand()


@st.cache_data(show_spinner=False)
def _backtest(drug: str | None, region: str | None):
    return fc.backtest_forecast(drug_code=drug, region=region)


@st.cache_data(show_spinner=False)
def _forecast(drug: str | None, region: str | None, horizon: int, method: str):
    return fc.forecast_demand(drug_code=drug, region=region,
                              horizon=horizon, method=method)


demand = _demand()
kpis = fc.demand_kpis(demand)

# ---------------------------------------------------------------------------
section("Demand Overview")
kpi_row([
    {"label": "Total Demand", "value": fmt_units(kpis["total_demand_units"]),
     "help_text": f"Across {kpis['months_covered']} months"},
    {"label": "Avg Monthly", "value": fmt_units(kpis["avg_monthly_demand"])},
    {"label": "YoY Growth", "value": fmt_pct(kpis["yoy_growth_pct"]),
     "delta_good": kpis["yoy_growth_pct"] > 0},
    {"label": "Fill Rate", "value": fmt_pct(kpis["avg_fill_rate_pct"]),
     "help_text": f"{fmt_units(kpis['total_backorders'])} units backordered"},
])

# ---------------------------------------------------------------------------
section("Scope", "Forecast the whole network, or drill into a product and region.")

col1, col2, col3 = st.columns(3)
drug_options = ["All products"] + sorted(demand["drug_code"].unique().tolist())
region_options = ["All regions"] + sorted(demand["region"].unique().tolist())
drug_choice = col1.selectbox("Product", drug_options)
region_choice = col2.selectbox("Region", region_options)
horizon = col3.slider("Forecast horizon (months)", 3, 18, 6, 1)

drug = None if drug_choice == "All products" else drug_choice
region = None if region_choice == "All regions" else region_choice

history = fc.aggregate_demand(demand, drug_code=drug, region=region)

# ---------------------------------------------------------------------------
section(
    "Model Selection by Backtest",
    "The last six months are held out, every method is fitted on the remainder, "
    "and accuracy is measured on months the models never saw.",
)

backtest = _backtest(drug, region)
accuracy = fc.forecast_accuracy_summary(drug_code=drug, region=region)

col1, col2 = st.columns([3, 2], gap="large")
with col1:
    show_table(backtest[["rank", "method", "mae", "rmse", "mape_pct",
                         "bias_units", "bias_pct", "train_periods", "test_periods"]],
               height=200)
with col2:
    chart(charts.bar_chart(backtest.sort_values("mape_pct"), x="method",
                           y="mape_pct", title="Backtest MAPE by Method (%)",
                           text_format=".2f", height=300))

best = backtest.iloc[0]
callout(
    f"**{best['method'].replace('_', ' ').title()}** wins the backtest with a MAPE of "
    f"**{best['mape_pct']:.2f}%** (MAE {best['mae']:,.0f} units, bias "
    f"{best['bias_pct']:+.2f}%). {accuracy.get('verdict', '')}",
    kind="success", title="Selected model",
)

if len(backtest) > 1:
    runner_up = backtest.iloc[1]
    insight(
        f"The margin over {runner_up['method'].replace('_', ' ')} "
        f"({runner_up['mape_pct']:.2f}% MAPE) is "
        f"{runner_up['mape_pct'] - best['mape_pct']:.2f} percentage points. "
        "Bias matters as much as absolute error here: a method with low MAPE but "
        "persistent positive bias systematically over-orders, which shows up two "
        "quarters later as the expiry write-offs on the Inventory page. The bias "
        "column is what a planner should check before trusting a number."
    )

# ---------------------------------------------------------------------------
section("Forecast")

method = st.radio(
    "Method", backtest["method"].tolist(), horizontal=True, index=0,
    format_func=lambda m: m.replace("_", " ").title()
    + (" (backtest winner)" if m == best["method"] else ""),
)
forecast = _forecast(drug, region, horizon, method)

chart(charts.forecast_chart(
    history, forecast, date_col="date", value_col="units_demanded",
    title=f"Demand Forecast - {drug_choice}, {region_choice} "
          f"({method.replace('_', ' ').title()})"))

col1, col2 = st.columns([2, 3], gap="large")
with col1:
    show_table(forecast, height=280)
    download_button(forecast, "demand_forecast.csv", "Download forecast")
with col2:
    total_forecast = forecast["forecast"].sum()
    recent = history.tail(horizon)["units_demanded"].sum()
    kpi_row([
        {"label": f"Next {horizon} Months", "value": fmt_units(total_forecast)},
        {"label": f"Prior {horizon} Months", "value": fmt_units(recent)},
        {"label": "Implied Change",
         "value": fmt_pct(100 * (total_forecast / recent - 1)) if recent else "-",
         "delta_good": total_forecast >= recent},
    ])
    st.caption(
        "Prediction intervals are derived from in-sample residual dispersion, so they "
        "widen for noisier series. They are indicative, not exact coverage guarantees."
    )

# ---------------------------------------------------------------------------
section("Seasonality and Trend Decomposition")

decomposition = fc.decompose_demand(history)
chart(charts.decomposition_chart(decomposition))
insight(
    "The decomposition separates the three things a planner needs to treat "
    "differently: a growth **trend** that should drive capacity decisions, a "
    "repeating **seasonal** pattern that should drive safety stock timing, and a "
    "**residual** that is genuine noise and should not be reacted to at all. "
    "Demand peaks in the winter months, consistent with a cardiovascular and "
    "hypertension portfolio."
)

# ---------------------------------------------------------------------------
section("Demand Mix")

tab1, tab2 = st.tabs(["By Product", "By Region"])
with tab1:
    by_product = (demand.groupby(["year_month", "brand_name"], as_index=False)
                  ["units_demanded"].sum())
    chart(charts.line_chart(by_product, x="year_month", y="units_demanded",
                            color="brand_name", title="Monthly Demand by Product",
                            y_title="Units", height=420))
with tab2:
    by_region = (demand.groupby(["year_month", "region"], as_index=False)
                 ["units_demanded"].sum())
    chart(charts.line_chart(by_region, x="year_month", y="units_demanded",
                            color="region", title="Monthly Demand by Region",
                            y_title="Units", height=420))

# ---------------------------------------------------------------------------
methodology("""
**Methods.** Three are fitted, all configured in `config/config.yaml`
(`forecasting.methods`):

- **Holt-Winters** - `statsmodels` exponential smoothing with additive trend and
  additive seasonality, seasonal period 12. Needs at least two full seasonal
  cycles; the module falls back to linear trend and logs a warning on shorter
  series.
- **Linear trend** - OLS on a time index with month dummies, which captures
  trend plus fixed seasonality without smoothing parameters to tune.
- **Moving average** - trailing window mean. Included as a naive baseline: a
  sophisticated method that cannot beat it is not earning its complexity.

**Backtest.** The final `forecasting.train_test_split_months` (6) months are held
out. Each method is fitted on the remainder and scored on the hold-out with MAE,
RMSE, MAPE and bias. Ranking is by MAPE, so the forward forecast uses a method
chosen on evidence rather than preference.

**Why bias is reported separately.** MAPE treats over- and under-forecasting
symmetrically, but a supply chain does not. Persistent over-forecasting becomes
expiry write-off; persistent under-forecasting becomes stock-out and lost sales.
Both the direction and the magnitude of bias are therefore shown.

**Intervals.** Prediction intervals use residual dispersion at the confidence
level in `forecasting.confidence_level` (95%). They widen with series volatility
but are indicative rather than a formal coverage guarantee.
""")

sidebar_about()
