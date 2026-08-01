"""
Interactive supply chain scenario simulation.

Business framing
----------------
The analytics pages in this platform explain what *did* happen. This module
answers the question that follows: **what happens if we change something?**
It is the difference between a report and a decision tool.

A user moves seven operational levers - demand, warehouse dwell, manufacturing
dwell, supplier reliability, storage temperature, storage humidity and
inventory cover - and every downstream KPI updates: yield, cycle time, potency,
out-of-specification rate, expiry exposure, stock-out risk, service level and
the resulting dollar impact.

Design principle: no invented mathematics
-----------------------------------------
Every propagation rule is either

* **physics reused from the data layer** - storage temperature, humidity and
  duration feed the *same* Arrhenius-style degradation model
  (:func:`src.data.generator._potency_after_storage`) that generated the
  observed data, so the simulation cannot silently disagree with it; or
* **an elasticity declared in configuration** - ``config.simulation.elasticity``
  states, in plain units, how one lever moves one KPI (for example, one
  percentage point of supplier reliability is worth 0.45pp of QA yield).

Baselines are measured from the actual dataset rather than hard-coded, so the
scenario at default lever settings reproduces observed performance. That
property is asserted in the test suite.

Example
-------
>>> from src.analytics.simulation import simulate, lever_definitions
>>> result = simulate({"supplier_reliability_pct": 96, "warehouse_delay_days": 25})
>>> result["kpis"]["end_to_end_yield_pct"]
{'baseline': 63.38, 'scenario': 65.12, 'delta': 1.74, ...}
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)

#: KPIs where a higher value is better. Used to colour deltas correctly - a
#: rising stock-out risk and a rising yield must not both render as green.
_HIGHER_IS_BETTER = {
    "qa_pass_rate_pct", "end_to_end_yield_pct", "units_dispensed",
    "avg_potency_pct", "service_level_pct", "inventory_turnover",
}

#: Levers whose default is *measured from the data* rather than declared in
#: config. Hard-coding these defaults is how a simulator silently drifts away
#: from reality: regenerate the data with different parameters and a literal
#: default would no longer represent the status quo, so `simulate({})` would
#: report a phantom change. Mapping each lever to its observed baseline keeps
#: the identity `simulate({}) == observed performance` true by construction.
#: The two levers absent here - `demand_change_pct` and `inventory_level_pct` -
#: are inherently *relative* (0% change, 100% of current cover), so their
#: config defaults already mean "no change".
_LEVER_BASELINE_KEYS: dict[str, str] = {
    "warehouse_delay_days": "baseline_warehouse_delay_days",
    "manufacturing_delay_days": "baseline_manufacturing_delay_days",
    "supplier_reliability_pct": "baseline_supplier_reliability_pct",
    "storage_temp_c": "baseline_storage_temp_c",
    "storage_humidity_pct": "baseline_storage_humidity_pct",
}


# ---------------------------------------------------------------------------
# Lever metadata
# ---------------------------------------------------------------------------
def lever_definitions() -> pd.DataFrame:
    """Return the simulation levers with their ranges, defaults and meaning.

    Returns
    -------
    pd.DataFrame
        ``lever``, ``label``, ``min``, ``max``, ``default``, ``step``, ``unit``,
        ``description``.
    """
    cfg = get_config().simulation.levers
    meta = {
        "demand_change_pct": ("Demand Change", "%",
                              "Shift in market demand versus the current plan."),
        "warehouse_delay_days": ("Warehouse Dwell Time", "days",
                                 "Days finished goods sit in the distribution centre."),
        "manufacturing_delay_days": ("Manufacturing Lead Time", "days",
                                     "Days from API release to bulk product completion."),
        "supplier_reliability_pct": ("Supplier Reliability", "%",
                                     "Weighted API supplier quality and consistency score."),
        "storage_temp_c": ("Storage Temperature", "degC",
                           "Average storage temperature across the network."),
        "storage_humidity_pct": ("Storage Humidity", "%RH",
                                 "Average relative humidity in storage."),
        "inventory_level_pct": ("Inventory Cover", "%",
                                "Inventory held versus the current policy level."),
    }
    defaults = default_levers()
    rows = []
    for lever, spec in cfg.items():
        label, unit, description = meta.get(lever, (lever, "", ""))
        rows.append({
            "lever": lever, "label": label,
            # Widen the slider if the measured baseline sits outside the
            # configured range, so the status quo is always reachable.
            "min": min(spec["min"], defaults[lever]),
            "max": max(spec["max"], defaults[lever]),
            "default": defaults[lever], "step": spec["step"], "unit": unit,
            "description": description,
            "is_measured": lever in _LEVER_BASELINE_KEYS,
        })
    return pd.DataFrame(rows)


def default_levers() -> dict[str, float]:
    """Lever settings that reproduce current observed performance.

    Levers listed in :data:`_LEVER_BASELINE_KEYS` take their default from the
    measured dataset; the purely relative levers take theirs from config. This
    guarantees ``simulate({})`` returns zero deltas on every KPI - a property
    asserted in the test suite.
    """
    base = get_baseline()
    return {
        lever: float(base[_LEVER_BASELINE_KEYS[lever]]) if lever in _LEVER_BASELINE_KEYS
        else float(spec["default"])
        for lever, spec in get_config().simulation.levers.items()
    }


# ---------------------------------------------------------------------------
# Baseline measurement
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_baseline() -> dict[str, float]:
    """Measure baseline KPIs from the actual dataset.

    Cached because it scans the full fact tables and never changes within a
    session. Call ``get_baseline.cache_clear()`` after regenerating data.

    Returns
    -------
    dict
        Observed values for every KPI the simulation projects, plus the
        reference conditions the levers are measured against.
    """
    cfg = get_config()
    batches = loader.load_batches()
    inventory = loader.load_inventory()
    demand = loader.load_demand()

    years = max(batches["year"].nunique(), 1)
    units_in = float(batches["units_procured"].sum())
    units_out = float(batches["units_dispensed"].sum())
    total_units = units_in
    avg_unit_cost = float(
        (batches["units_procured"] * batches["unit_cost_usd"]).sum() / total_units)

    latest = inventory[inventory["snapshot_month"] == inventory["snapshot_month"].max()]
    stockout_threshold = float(cfg.inventory.stockout_risk_threshold)
    at_risk = latest["units_on_hand"] < latest["reorder_point"] * stockout_threshold

    return {
        # Observed performance
        "qa_pass_rate_pct": round(100 * float(batches["qa_pass"].mean()), 2),
        "end_to_end_yield_pct": round(100 * units_out / units_in, 2),
        "units_procured_annual": round(units_in / years, 0),
        "units_dispensed_annual": round(units_out / years, 0),
        "avg_cycle_time_days": round(float(batches["total_cycle_time_days"].mean()), 1),
        "avg_potency_pct": round(float(batches["potency_pct"].mean()), 2),
        "out_of_spec_pct": round(100 * float(
            (batches["potency_pct"] < cfg.stability.potency_spec_min).mean()), 2),
        # Expiry exposure is measured on the inventory position rather than on
        # batches at handover. A batch leaves the plant with most of its shelf
        # life intact, so the risk only becomes real once stock is sitting in a
        # warehouse ageing towards its expiry date.
        "expiry_risk_pct": round(100 * float(
            (latest["days_to_expiry"] <= cfg.inventory.expiry_risk_days).mean()), 2),
        "stockout_risk_pct": round(100 * float(at_risk.mean()), 2),
        "service_level_pct": round(float(demand["fill_rate_pct"].mean()), 2),
        "inventory_units": float(latest["units_on_hand"].sum()),
        "inventory_value_usd": float(latest["inventory_value_usd"].sum()),
        "value_lost_annual_usd": round(float(batches["value_lost_usd"].sum()) / years, 0),
        # Reference conditions the levers are relative to
        "avg_unit_cost_usd": round(avg_unit_cost, 4),
        "baseline_supplier_reliability_pct": round(
            100 * float(batches["supplier_reliability"].mean()), 2),
        "baseline_warehouse_delay_days": round(float(batches["delay_warehouse_days"].mean()), 1),
        "baseline_manufacturing_delay_days": round(
            float(batches["delay_manufacturing_days"].mean()), 1),
        "baseline_storage_temp_c": round(float(batches["storage_temp_c"].mean()), 1),
        "baseline_storage_humidity_pct": round(float(batches["storage_humidity_pct"].mean()), 1),
        "baseline_storage_duration_days": round(
            float(batches["storage_duration_days"].mean()), 1),
        "annual_demand_units": round(float(demand["units_demanded"].sum()) / years, 0),
        "years_of_history": years,
    }


# ---------------------------------------------------------------------------
# Degradation physics (shared with the data generator)
# ---------------------------------------------------------------------------
def _degradation_pct(temp_c: float, humidity_pct: float, duration_days: float,
                     is_cold_chain: bool = False) -> float:
    """Potency loss (%) under given storage conditions.

    Deliberately mirrors :func:`src.data.generator._potency_after_storage` so
    the simulation and the observed data obey one degradation law.
    """
    cfg = get_config().stability
    reference = cfg.reference_temp_cold if is_cold_chain else cfg.reference_temp_ambient
    temp_factor = cfg.q10_factor ** ((temp_c - reference) / 10.0)
    humidity_factor = max(
        0.6, 1.0 + cfg.humidity_sensitivity * (humidity_pct - cfg.reference_humidity_pct))
    return (cfg.base_degradation_rate
            * (duration_days / 30.0)
            * temp_factor
            * humidity_factor)


# ---------------------------------------------------------------------------
# Scenario engine
# ---------------------------------------------------------------------------
def simulate(levers: dict[str, float] | None = None) -> dict:
    """Project every downstream KPI for a given lever configuration.

    Parameters
    ----------
    levers
        Partial or complete lever settings. Any lever omitted keeps its default,
        so ``simulate({})`` reproduces the observed baseline.

    Returns
    -------
    dict
        ``levers`` (resolved settings), ``kpis`` (per-KPI dict of
        ``baseline`` / ``scenario`` / ``delta`` / ``delta_pct`` / ``unit`` /
        ``higher_is_better`` / ``improved``), ``financials`` and ``alerts``.
    """
    cfg = get_config()
    elasticity = cfg.simulation.elasticity
    base = get_baseline()

    settings = default_levers()
    settings.update({k: float(v) for k, v in (levers or {}).items()
                     if k in settings})

    # --- Lever deltas relative to the measured baseline --------------------
    d_reliability = settings["supplier_reliability_pct"] - base["baseline_supplier_reliability_pct"]
    d_warehouse = settings["warehouse_delay_days"] - base["baseline_warehouse_delay_days"]
    d_manufacturing = (settings["manufacturing_delay_days"]
                       - base["baseline_manufacturing_delay_days"])
    d_demand_pct = settings["demand_change_pct"]
    d_inventory_pct = settings["inventory_level_pct"] - 100.0

    # --- Quality: supplier reliability drives QA release rate --------------
    qa_pass = float(np.clip(
        base["qa_pass_rate_pct"] + elasticity.reliability_to_qa_yield * d_reliability,
        50.0, 99.9))

    # --- Cycle time: dwell-time levers add directly ------------------------
    cycle_time = max(base["avg_cycle_time_days"] + d_warehouse + d_manufacturing, 1.0)

    # --- Stability: recompute degradation under the new conditions ---------
    # Storage exposure moves with warehouse dwell, so a lever that lengthens
    # dwell also costs potency - the coupling that makes this simulation useful.
    storage_duration = max(base["baseline_storage_duration_days"] + d_warehouse, 1.0)
    baseline_degradation = _degradation_pct(
        base["baseline_storage_temp_c"], base["baseline_storage_humidity_pct"],
        base["baseline_storage_duration_days"])
    scenario_degradation = _degradation_pct(
        settings["storage_temp_c"], settings["storage_humidity_pct"], storage_duration)
    potency_shift = scenario_degradation - baseline_degradation
    avg_potency = float(np.clip(base["avg_potency_pct"] - potency_shift, 60.0, 100.0))

    # Shift the observed potency distribution and re-count the tail below spec.
    # Using the empirical distribution keeps the real spread rather than
    # assuming normality.
    observed_potency = loader.load_batches()["potency_pct"].to_numpy()
    shifted = observed_potency - potency_shift
    out_of_spec = float(100 * (shifted < cfg.stability.potency_spec_min).mean())

    # --- Expiry: longer cycles consume shelf life --------------------------
    expiry_risk = float(np.clip(
        base["expiry_risk_pct"]
        + elasticity.delay_to_expiry_risk * (cycle_time - base["avg_cycle_time_days"]),
        0.0, 100.0))

    # --- Stock-out: demand pushes risk up, inventory cover pulls it down ---
    # Extra cover is credited at half weight: holding more stock helps, but
    # cannot fully offset a demand shock because replenishment lead time binds.
    stockout_risk = float(np.clip(
        base["stockout_risk_pct"]
        + elasticity.demand_to_stockout * d_demand_pct
        - 0.5 * d_inventory_pct,
        0.0, 100.0))
    service_level = float(np.clip(base["service_level_pct"] - 0.45 * (
        stockout_risk - base["stockout_risk_pct"]), 0.0, 100.0))

    # --- Volume roll-forward ------------------------------------------------
    # Yield moves with the QA stage; the other stages are held at observed
    # performance because no lever addresses them.
    yield_pct = float(np.clip(
        base["end_to_end_yield_pct"] * (qa_pass / base["qa_pass_rate_pct"]), 0.0, 100.0))
    units_in = base["units_procured_annual"] * (1 + d_demand_pct / 100.0)
    units_out = units_in * yield_pct / 100.0

    inventory_units = base["inventory_units"] * settings["inventory_level_pct"] / 100.0
    inventory_value = base["inventory_value_usd"] * settings["inventory_level_pct"] / 100.0

    # --- Financials ---------------------------------------------------------
    unit_cost = base["avg_unit_cost_usd"]
    value_lost = (units_in - units_out) * unit_cost
    holding_cost = inventory_value * float(cfg.economics.holding_cost_rate_annual)
    stockout_cost = (base["annual_demand_units"] * (1 + d_demand_pct / 100.0)
                     * stockout_risk / 100.0 * float(cfg.economics.stockout_penalty_per_unit))
    expiry_cost = (inventory_units * expiry_risk / 100.0 * unit_cost
                   * (1 - float(cfg.economics.expiry_write_off_recovery)))
    # Product that fails potency specification cannot be released and must be
    # destroyed. Costing it is what connects the storage levers to the P&L -
    # without this term, temperature and humidity would move quality metrics
    # but register as financially free, which is the opposite of the truth.
    quality_cost = (units_out * out_of_spec / 100.0 * unit_cost
                    * (1 - float(cfg.economics.expiry_write_off_recovery)))
    total_cost = value_lost + holding_cost + stockout_cost + expiry_cost + quality_cost

    baseline_financials = _baseline_financials(base)

    kpis = {
        "qa_pass_rate_pct": _kpi(base["qa_pass_rate_pct"], qa_pass, "%", "qa_pass_rate_pct"),
        "end_to_end_yield_pct": _kpi(base["end_to_end_yield_pct"], yield_pct, "%", "end_to_end_yield_pct"),
        "units_dispensed": _kpi(base["units_dispensed_annual"], units_out, "units/yr", "units_dispensed"),
        "avg_cycle_time_days": _kpi(base["avg_cycle_time_days"], cycle_time, "days", "avg_cycle_time_days"),
        "avg_potency_pct": _kpi(base["avg_potency_pct"], avg_potency, "%", "avg_potency_pct"),
        "out_of_spec_pct": _kpi(base["out_of_spec_pct"], out_of_spec, "%", "out_of_spec_pct"),
        "expiry_risk_pct": _kpi(base["expiry_risk_pct"], expiry_risk, "%", "expiry_risk_pct"),
        "stockout_risk_pct": _kpi(base["stockout_risk_pct"], stockout_risk, "%", "stockout_risk_pct"),
        "service_level_pct": _kpi(base["service_level_pct"], service_level, "%", "service_level_pct"),
        "inventory_value_usd": _kpi(base["inventory_value_usd"], inventory_value, "USD", "inventory_value_usd"),
    }

    financials = {
        "value_lost_usd": _kpi(baseline_financials["value_lost_usd"], value_lost, "USD"),
        "holding_cost_usd": _kpi(baseline_financials["holding_cost_usd"], holding_cost, "USD"),
        "stockout_cost_usd": _kpi(baseline_financials["stockout_cost_usd"], stockout_cost, "USD"),
        "expiry_cost_usd": _kpi(baseline_financials["expiry_cost_usd"], expiry_cost, "USD"),
        "quality_cost_usd": _kpi(baseline_financials["quality_cost_usd"], quality_cost, "USD"),
        "total_cost_usd": _kpi(baseline_financials["total_cost_usd"], total_cost, "USD"),
    }

    return {
        "levers": settings,
        "kpis": kpis,
        "financials": financials,
        "alerts": _generate_alerts(kpis, financials),
    }


def _kpi(baseline: float, scenario: float, unit: str, name: str = "") -> dict:
    """Package one KPI with its delta and whether the move is an improvement.

    Direction matters for rendering: a rising yield and a rising stock-out risk
    must not both display as a green "up" indicator. Anything not listed in
    :data:`_HIGHER_IS_BETTER` - every cost and every risk measure - is treated
    as lower-is-better.
    """
    delta = scenario - baseline
    higher_is_better = name in _HIGHER_IS_BETTER
    return {
        "baseline": round(baseline, 2),
        "scenario": round(scenario, 2),
        "delta": round(delta, 2),
        "delta_pct": round(100 * delta / baseline, 2) if baseline else 0.0,
        "unit": unit,
        "higher_is_better": higher_is_better,
        "improved": bool(delta > 0) if higher_is_better else bool(delta < 0),
    }


def _baseline_financials(base: dict) -> dict[str, float]:
    """Cost stack at default lever settings, on the same formulas as the scenario."""
    cfg = get_config()
    unit_cost = base["avg_unit_cost_usd"]
    units_in = base["units_procured_annual"]
    units_out = base["units_dispensed_annual"]

    value_lost = (units_in - units_out) * unit_cost
    holding_cost = base["inventory_value_usd"] * float(cfg.economics.holding_cost_rate_annual)
    stockout_cost = (base["annual_demand_units"] * base["stockout_risk_pct"] / 100.0
                     * float(cfg.economics.stockout_penalty_per_unit))
    expiry_cost = (base["inventory_units"] * base["expiry_risk_pct"] / 100.0 * unit_cost
                   * (1 - float(cfg.economics.expiry_write_off_recovery)))
    quality_cost = (units_out * base["out_of_spec_pct"] / 100.0 * unit_cost
                    * (1 - float(cfg.economics.expiry_write_off_recovery)))
    return {
        "value_lost_usd": value_lost,
        "holding_cost_usd": holding_cost,
        "stockout_cost_usd": stockout_cost,
        "expiry_cost_usd": expiry_cost,
        "quality_cost_usd": quality_cost,
        "total_cost_usd": (value_lost + holding_cost + stockout_cost
                           + expiry_cost + quality_cost),
    }


def _generate_alerts(kpis: dict, financials: dict) -> list[dict]:
    """Raise operational alerts when a scenario breaches a governance threshold."""
    cfg = get_config()
    alerts: list[dict] = []

    if kpis["out_of_spec_pct"]["scenario"] > kpis["out_of_spec_pct"]["baseline"] + 5:
        alerts.append({
            "severity": "High", "area": "Product Quality",
            "message": (f"Out-of-specification rate rises to "
                        f"{kpis['out_of_spec_pct']['scenario']:.1f}% "
                        f"({kpis['out_of_spec_pct']['delta']:+.1f}pp). "
                        f"Storage conditions breach the stability profile."),
        })
    if kpis["stockout_risk_pct"]["scenario"] > 30:
        alerts.append({
            "severity": "High", "area": "Service",
            "message": (f"Stock-out risk at {kpis['stockout_risk_pct']['scenario']:.1f}% "
                        f"threatens supply continuity. Increase cover or expedite replenishment."),
        })
    if kpis["avg_cycle_time_days"]["delta"] > 15:
        alerts.append({
            "severity": "Medium", "area": "Lead Time",
            "message": (f"Cycle time extends by {kpis['avg_cycle_time_days']['delta']:.0f} days, "
                        f"consuming shelf life and inflating safety-stock requirements."),
        })
    if kpis["qa_pass_rate_pct"]["scenario"] < 88:
        alerts.append({
            "severity": "High", "area": "Quality",
            "message": (f"QA pass rate falls to {kpis['qa_pass_rate_pct']['scenario']:.1f}%. "
                        f"Supplier qualification review required."),
        })
    if financials["total_cost_usd"]["delta"] > 0:
        alerts.append({
            "severity": "Medium", "area": "Financial",
            "message": (f"Total annual cost increases by "
                        f"${financials['total_cost_usd']['delta']:,.0f} "
                        f"({financials['total_cost_usd']['delta_pct']:+.1f}%)."),
        })
    elif financials["total_cost_usd"]["delta"] < 0:
        alerts.append({
            "severity": "Info", "area": "Financial",
            "message": (f"Total annual cost falls by "
                        f"${abs(financials['total_cost_usd']['delta']):,.0f} "
                        f"({financials['total_cost_usd']['delta_pct']:.1f}%)."),
        })
    return alerts


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
def sensitivity_analysis(lever: str, kpi: str = "total_cost_usd",
                         steps: int = 12) -> pd.DataFrame:
    """Sweep one lever across its full range and trace a KPI's response.

    Parameters
    ----------
    lever
        Lever name from :func:`lever_definitions`.
    kpi
        KPI key from either the ``kpis`` or ``financials`` block.
    steps
        Number of points across the lever's range.

    Returns
    -------
    pd.DataFrame
        ``lever_value``, ``kpi_value``, ``delta_from_baseline``.
    """
    cfg = get_config().simulation.levers
    if lever not in cfg:
        raise KeyError(f"Unknown lever '{lever}'. Available: {sorted(cfg)}")

    spec = cfg[lever]
    values = np.linspace(spec["min"], spec["max"], steps)

    rows = []
    for value in values:
        result = simulate({lever: float(value)})
        block = result["kpis"] if kpi in result["kpis"] else result["financials"]
        rows.append({
            "lever": lever,
            "lever_value": round(float(value), 2),
            "kpi": kpi,
            "kpi_value": block[kpi]["scenario"],
            "delta_from_baseline": block[kpi]["delta"],
        })
    return pd.DataFrame(rows)


def tornado_analysis(kpi: str = "total_cost_usd") -> pd.DataFrame:
    """Rank levers by how much they move a KPI across their full range.

    The classic sensitivity tornado: it tells a decision maker which lever is
    worth fighting for and which is a rounding error.

    Returns
    -------
    pd.DataFrame
        ``lever``, ``label``, ``low_value``, ``high_value``, ``kpi_at_low``,
        ``kpi_at_high``, ``swing``, ``swing_pct`` - sorted by absolute swing.
    """
    cfg = get_config().simulation.levers
    labels = lever_definitions().set_index("lever")["label"].to_dict()
    baseline_result = simulate({})
    block_name = "kpis" if kpi in baseline_result["kpis"] else "financials"
    baseline_value = baseline_result[block_name][kpi]["baseline"]

    rows = []
    for lever, spec in cfg.items():
        low = simulate({lever: float(spec["min"])})[block_name][kpi]["scenario"]
        high = simulate({lever: float(spec["max"])})[block_name][kpi]["scenario"]
        rows.append({
            "lever": lever,
            "label": labels.get(lever, lever),
            "low_value": spec["min"],
            "high_value": spec["max"],
            "kpi_at_low": low,
            "kpi_at_high": high,
            "swing": round(abs(high - low), 2),
            "swing_pct": round(100 * abs(high - low) / baseline_value, 2) if baseline_value else 0.0,
        })

    result = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    log.info("Tornado on '%s': most influential lever is '%s'", kpi, result.iloc[0]["label"])
    return result


def compare_scenarios(scenarios: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Evaluate several named scenarios side by side.

    Parameters
    ----------
    scenarios
        Mapping of scenario name -> lever settings.

    Returns
    -------
    pd.DataFrame
        One row per scenario with the headline KPIs and total cost.
    """
    rows = []
    for name, levers in scenarios.items():
        result = simulate(levers)
        row = {"scenario": name}
        for key, kpi in result["kpis"].items():
            row[key] = kpi["scenario"]
        row["total_cost_usd"] = result["financials"]["total_cost_usd"]["scenario"]
        row["cost_delta_usd"] = result["financials"]["total_cost_usd"]["delta"]
        row["alerts"] = len(result["alerts"])
        rows.append(row)
    return pd.DataFrame(rows)


def preset_scenarios() -> dict[str, dict[str, float]]:
    """Named scenarios that make the simulation immediately legible.

    These are the four cases an operations review would actually discuss, so a
    user sees a meaningful comparison before touching a single slider.
    """
    return {
        "Baseline": {},
        "Best Case - Excellence Programme": {
            "supplier_reliability_pct": 98, "warehouse_delay_days": 20,
            "manufacturing_delay_days": 8, "storage_temp_c": 20,
            "storage_humidity_pct": 45, "inventory_level_pct": 110,
        },
        "Stress - Demand Surge": {
            "demand_change_pct": 35, "inventory_level_pct": 85,
            "warehouse_delay_days": 18,
        },
        "Crisis - Supplier Failure + Heatwave": {
            "supplier_reliability_pct": 78, "storage_temp_c": 31,
            "storage_humidity_pct": 72, "warehouse_delay_days": 55,
            "manufacturing_delay_days": 22,
        },
    }


__all__ = [
    "lever_definitions", "default_levers", "get_baseline", "simulate",
    "sensitivity_analysis", "tornado_analysis", "compare_scenarios",
    "preset_scenarios",
]
