"""
Data quality assessment for the PharmaChain Analytics platform.

Why this module exists
----------------------
In a regulated pharmaceutical environment a number is not a number until you can
say where it came from and prove it was not quietly corrupted on the way. GxP
data-integrity expectations (FDA 21 CFR Part 11, EU GMP Annex 11, MHRA's
data-integrity guidance) all rest on the **ALCOA+** principles: data must be
Attributable, Legible, Contemporaneous, Original, Accurate - plus Complete,
Consistent, Enduring and Available. An analytics platform that reports batch
yield, cold-chain excursions or fill rate is, in practice, making a claim about
those properties every time it renders a chart.

This module is the gate in front of that claim. It converts the abstract ALCOA+
attributes into measurements a reviewer can audit:

===================  ========================================================
ALCOA+ attribute     What is measured here
===================  ========================================================
Complete             :func:`missing_value_report` - is any field silently
                     absent, and how badly?
Original             :func:`duplicate_report` - has a record been counted
                     twice? A duplicated ``batch_id`` is not a cosmetic
                     problem: it double-counts released product.
Accurate             :func:`validity_report` and :func:`outlier_report` - do
                     values sit inside their declared domain and their
                     plausible physical range?
Consistent           :func:`consistency_report` - do cross-field relationships
                     that must hold by construction actually hold? Units can
                     never *increase* as a batch moves down the funnel, and a
                     batch cannot be packaged before it is manufactured.
Available            :func:`assess_all` - one scoreboard, refreshed on demand,
                     that a reviewer can read in ten seconds.
===================  ========================================================

The output is deliberately shaped like a consulting deliverable rather than a
pass/fail assertion: every check reports the rule it applied, how many records
breached it, and a sample of the offending values, so a finding can be traced
back to individual rows. :func:`preprocessing_recommendations` then turns those
findings into named remediation actions, because an audit that stops at
"3.2% missing" leaves the reader with the hardest part of the job.

A note on honest scoring
------------------------
The synthetic supply chain data this platform generates is clean by
construction. A high score on it is the correct answer, not a failure of the
audit - the module's value is that it *demonstrates* cleanliness with evidence
and would catch a real defect on messier input. Nothing here inflates a problem
to look thorough: binary flag columns are excluded from outlier detection, null
values are excluded from validity checks (absence is a completeness defect, not
an accuracy defect), and the accuracy dimension caps each column's penalty so a
naturally heavy-tailed variable such as batch value cannot drag down a score it
has no business dragging down.

Every threshold, weight and grade band is read from ``config/config.yaml``
(``data_quality`` and ``funnel`` sections). The funnel ordering used by the
consistency checks is the *same* ordering the rest of the platform uses, so
reordering a stage in configuration reorders these checks too.

Example
-------
>>> from src.quality.assessment import assess_all, assess_dataset
>>> assess_all()                      # scoreboard across all five tables
>>> report = assess_dataset("batches")
>>> report["score"]["grade"]
'A'
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import get_config
from src.data import loader
from src.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Everything below is read from ``config.data_quality`` when the key is present,
# so a reviewer can retune the audit without touching code. The literals here
# are the documented fallbacks that apply only when the key is absent from the
# YAML - they are defaults, not magic numbers buried in the logic.
_DQ_DEFAULTS: dict[str, Any] = {
    # Missing-value severity bands, in percent of the column.
    "missing_low_pct": 5.0,
    "missing_high_pct": 20.0,
    # Above this share of nulls, imputation invents more data than it recovers.
    "missing_drop_pct": 50.0,
    # A column with this many or fewer distinct values is a flag/category, not a
    # continuous measurement - Tukey fences on a 0/1 column are pure noise.
    "min_unique_for_outliers": 2,
    # Per-column ceiling on the outlier penalty feeding the accuracy dimension.
    "accuracy_outlier_cap": 0.10,
    # |skew| above this marks a distribution as genuinely heavy-tailed, so its
    # outliers are recommended for investigation rather than winsorisation.
    "heavy_tail_skew": 1.0,
    # A violation rate above this is material enough to escalate to High.
    "material_violation_pct": 1.0,
    # Outlier share above which winsorisation becomes the default advice.
    "material_outlier_pct": 5.0,
    # Rows shown as evidence in duplicate/validity samples.
    "sample_rows": 5,
}

# Severity vocabulary, ordered worst-first for sorting.
_SEVERITY_ORDER: dict[str, int] = {"High": 0, "Medium": 1, "Low": 2, "None": 3}

# The five platform tables and the business key that must be unique in each.
# ``None`` means the table has no single natural key (drug200 is one row per
# patient observation with no identifier column).
_DATASET_KEYS: dict[str, list[str] | None] = {
    "drug200": None,
    "batches": ["batch_id"],
    "shipments": ["shipment_id"],
    "inventory": ["snapshot_month", "warehouse_id", "drug_code"],
    "demand": ["date", "drug_code", "region"],
}


def _dq(key: str) -> Any:
    """Read a data-quality tunable from config, falling back to the default."""
    return get_config().data_quality.get(key, _DQ_DEFAULTS[key])


# ---------------------------------------------------------------------------
# Default validity rule set
# ---------------------------------------------------------------------------
# Declared domains for columns this platform actually produces. A rule is a
# mapping of column -> {'allowed': set} for categorical domains, or
# {'min': x, 'max': y} for ranges (either bound may be omitted).
_NAMED_RULES: dict[str, dict[str, Any]] = {
    # --- clinical (drug200) ------------------------------------------------
    "Sex": {"allowed": {"M", "F"}},
    "BP": {"allowed": {"HIGH", "NORMAL", "LOW"}},
    "Cholesterol": {"allowed": {"HIGH", "NORMAL"}},
    "Drug": {"allowed": {"DrugY", "drugA", "drugB", "drugC", "drugX"}},
    "Age": {"min": 0, "max": 120},
    "Na_to_K": {"min": 0, "max": 100},
    # --- supply chain flags and categoricals -------------------------------
    "qa_result": {"allowed": {"Pass", "Fail"}},
    "qa_pass": {"allowed": {0, 1}},
    "is_cold_chain": {"allowed": {0, 1}},
    "is_late": {"allowed": {0, 1}},
    "cold_chain_excursion": {"allowed": {0, 1}},
    "temperature_excursion": {"allowed": {0, 1}},
    "batch_risk_label": {"allowed": {"Low", "Medium", "High"}},
    # --- supply chain measurements -----------------------------------------
    "potency_pct": {"min": 0, "max": 100},
    # Reliability is stored as a fraction, not a percentage.
    "supplier_reliability": {"min": 0, "max": 1},
    "shelf_life_months": {"min": 0},
    "unit_cost_usd": {"min": 0},
    "planned_transit_days": {"min": 0},
    "actual_transit_days": {"min": 0},
    "leg_sequence": {"min": 1},
    "storage_duration_days": {"min": 0},
    "total_cycle_time_days": {"min": 0},
    "months_of_supply": {"min": 0},
    "safety_stock": {"min": 0},
    "reorder_point": {"min": 0},
    "warehouse_capacity": {"min": 0},
    "days_to_expiry": {"min": 0},
    "days_to_expiry_at_delivery": {"min": 0},
}

# Pattern rules cover families of columns that share a physical constraint, so
# a new ``units_returned`` column inherits the "cannot be negative" rule the day
# it is added rather than the day someone remembers to update this file.
_SUFFIX_RULES: dict[str, dict[str, Any]] = {
    "_usd": {"min": 0},          # money in this platform is never negative
    "_pct": {"min": 0, "max": 100},
}
_PREFIX_RULES: dict[str, dict[str, Any]] = {
    "units_": {"min": 0},        # you cannot ship minus one tablet
}


def default_rules(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build the platform's default validity rule set for one frame.

    Only columns that are actually present get a rule, so the same rule set can
    be pointed at any table in the platform.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame whose columns decide which rules apply.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of column name to a rule of the form ``{'allowed': set}`` or
        ``{'min': x, 'max': y}``.
    """
    rules: dict[str, dict[str, Any]] = {}
    for column in df.columns:
        name = str(column)
        if name in _NAMED_RULES:
            rules[name] = dict(_NAMED_RULES[name])
            continue
        for prefix, rule in _PREFIX_RULES.items():
            if name.startswith(prefix) and pd.api.types.is_numeric_dtype(df[column]):
                rules[name] = dict(rule)
                break
        else:
            for suffix, rule in _SUFFIX_RULES.items():
                if name.endswith(suffix) and pd.api.types.is_numeric_dtype(df[column]):
                    rules[name] = dict(rule)
                    break
    return rules


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _pct(numerator: float, denominator: float) -> float:
    """Percentage that degrades gracefully to 0.0 on an empty denominator."""
    return 0.0 if not denominator else round(100.0 * numerator / denominator, 4)


def _describe_rule(rule: Mapping[str, Any]) -> str:
    """Render a rule as the human-readable string that appears in the report."""
    if "allowed" in rule:
        values = sorted(str(value) for value in rule["allowed"])
        return f"value in {{{', '.join(values)}}}"
    low, high = rule.get("min"), rule.get("max")
    if low is not None and high is not None:
        return f"{low} <= value <= {high}"
    if low is not None:
        return f"value >= {low}"
    if high is not None:
        return f"value <= {high}"
    return "no constraint"


def _sample_values(series: pd.Series, limit: int) -> str:
    """Comma-separated sample of distinct offending values, for traceability."""
    if series.empty:
        return ""
    distinct = pd.unique(series.dropna())[:limit]
    return ", ".join(str(value) for value in distinct)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns only - booleans included, datetimes deliberately not."""
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_datetime64_any_dtype(df[column])
    ]


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """Typed-but-empty result frame, so callers can always trust the schema."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


# ---------------------------------------------------------------------------
# 1. Completeness
# ---------------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value audit, ranked worst-first.

    Completeness is the ALCOA+ attribute most often broken silently: a null in a
    ``potency_pct`` column does not raise an error, it just quietly removes a
    batch from a mean.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.

    Returns
    -------
    pandas.DataFrame
        Columns ``column``, ``dtype``, ``missing_count``, ``missing_pct``,
        ``non_null_count``, ``severity`` (``None`` / ``Low`` / ``Medium`` /
        ``High``), sorted by ``missing_pct`` descending.
    """
    low_band = float(_dq("missing_low_pct"))
    high_band = float(_dq("missing_high_pct"))
    n_rows = len(df)

    rows: list[dict[str, Any]] = []
    for column in df.columns:
        missing = int(df[column].isna().sum())
        pct = _pct(missing, n_rows)
        if missing == 0:
            severity = "None"
        elif pct < low_band:
            severity = "Low"
        elif pct <= high_band:
            severity = "Medium"
        else:
            severity = "High"
        rows.append(
            {
                "column": str(column),
                "dtype": str(df[column].dtype),
                "missing_count": missing,
                "missing_pct": pct,
                "non_null_count": n_rows - missing,
                "severity": severity,
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return _empty_frame(
            ["column", "dtype", "missing_count", "missing_pct",
             "non_null_count", "severity"]
        )
    report = report.sort_values(
        ["missing_pct", "column"], ascending=[False, True]
    ).reset_index(drop=True)
    log.info(
        "Completeness: %d/%d columns carry nulls (worst %.2f%%).",
        int((report["missing_count"] > 0).sum()),
        len(report),
        float(report["missing_pct"].max()) if len(report) else 0.0,
    )
    return report


# ---------------------------------------------------------------------------
# 2. Uniqueness
# ---------------------------------------------------------------------------
def duplicate_report(
    df: pd.DataFrame, subset: Sequence[str] | None = None
) -> dict[str, Any]:
    """Audit record uniqueness, optionally against a declared business key.

    Exact duplicates inflate every count and every sum downstream. Key-level
    duplicates are worse: two rows sharing a ``batch_id`` mean the platform
    cannot say which record is the original, which breaks the ALCOA+
    "Original" attribute outright.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.
    subset : sequence of str, optional
        Business key columns that must be unique (for example ``["batch_id"]``).
        Columns not present in ``df`` are ignored.

    Returns
    -------
    dict
        ``total_rows``, ``exact_duplicate_rows`` (redundant copies, i.e. rows
        beyond the first occurrence), ``duplicate_pct``, ``duplicate_examples``
        (DataFrame of up to five offending rows) and, when ``subset`` resolves
        to at least one present column, ``key_columns``, ``key_duplicate_rows``,
        ``key_duplicate_pct`` and ``key_duplicate_examples``.
    """
    sample_rows = int(_dq("sample_rows"))
    n_rows = len(df)

    exact_mask = df.duplicated(keep="first") if n_rows else pd.Series(dtype=bool)
    exact_count = int(exact_mask.sum())
    result: dict[str, Any] = {
        "total_rows": n_rows,
        "exact_duplicate_rows": exact_count,
        "duplicate_pct": _pct(exact_count, n_rows),
        # keep=False so the reviewer sees the original alongside the copy -
        # you cannot adjudicate a duplicate by looking at one side of it.
        "duplicate_examples": (
            df.loc[df.duplicated(keep=False)].head(sample_rows).copy()
            if exact_count
            else df.head(0).copy()
        ),
    }

    key = [column for column in (subset or []) if column in df.columns]
    if key:
        key_mask = df.duplicated(subset=key, keep="first")
        key_count = int(key_mask.sum())
        result.update(
            {
                "key_columns": key,
                "key_duplicate_rows": key_count,
                "key_duplicate_pct": _pct(key_count, n_rows),
                "key_duplicate_examples": (
                    df.loc[df.duplicated(subset=key, keep=False)]
                    .sort_values(key)
                    .head(sample_rows)
                    .copy()
                    if key_count
                    else df.head(0).copy()
                ),
            }
        )
    elif subset:
        log.info("Duplicate key check skipped - none of %s present.", list(subset))

    log.info(
        "Uniqueness: %d exact duplicate rows of %d (%.2f%%).",
        exact_count, n_rows, result["duplicate_pct"],
    )
    return result


# ---------------------------------------------------------------------------
# 3. Validity
# ---------------------------------------------------------------------------
def _violation_mask(series: pd.Series, rule: Mapping[str, Any]) -> pd.Series:
    """Boolean mask of populated values that BREACH ``rule``.

    Nulls are never flagged here - absence is a completeness defect and is
    already counted by :func:`missing_value_report`.
    """
    present = series.notna()
    if "allowed" in rule:
        return present & ~series.isin(list(rule["allowed"]))
    # Coerce on a copy: a range rule pointed at a text column should report the
    # text as invalid rather than raise.
    numeric = pd.to_numeric(series, errors="coerce")
    bad = present & numeric.isna()
    if rule.get("min") is not None:
        bad = bad | (present & (numeric < rule["min"]))
    if rule.get("max") is not None:
        bad = bad | (present & (numeric > rule["max"]))
    return bad


def validity_report(
    df: pd.DataFrame, rules: Mapping[str, Mapping[str, Any]] | None = None
) -> pd.DataFrame:
    """Check every column against its declared domain.

    Nulls are excluded from the denominator on purpose: a missing value is a
    completeness defect and is already counted there. Charging it twice would
    make the overall score misleading.

    Passing rules are reported alongside failing ones - an audit trail has to
    show what was checked, not only what broke.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.
    rules : mapping, optional
        Column -> ``{'allowed': set}`` or ``{'min': x, 'max': y}``. Defaults to
        :func:`default_rules` for this frame. Rules naming absent columns are
        skipped silently.

    Returns
    -------
    pandas.DataFrame
        Columns ``column``, ``rule``, ``checked``, ``violations``,
        ``violation_pct``, ``sample_bad_values``, sorted worst-first.
    """
    applied = dict(rules) if rules is not None else default_rules(df)
    sample_rows = int(_dq("sample_rows"))

    rows: list[dict[str, Any]] = []
    for column, rule in applied.items():
        if column not in df.columns:
            continue
        series = df[column]
        present = series.notna()
        checked = int(present.sum())
        bad_mask = _violation_mask(series, rule)

        violations = int(bad_mask.sum())
        rows.append(
            {
                "column": str(column),
                "rule": _describe_rule(rule),
                "checked": checked,
                "violations": violations,
                "violation_pct": _pct(violations, checked),
                "sample_bad_values": (
                    _sample_values(series[bad_mask], sample_rows) if violations else ""
                ),
            }
        )

    if not rows:
        log.info("Validity: no default rule matched any column - nothing to check.")
        return _empty_frame(
            ["column", "rule", "checked", "violations",
             "violation_pct", "sample_bad_values"]
        )

    report = pd.DataFrame(rows).sort_values(
        ["violations", "violation_pct", "column"], ascending=[False, False, True]
    ).reset_index(drop=True)
    log.info(
        "Validity: %d rules applied, %d violations across %d columns.",
        len(report),
        int(report["violations"].sum()),
        int((report["violations"] > 0).sum()),
    )
    return report


# ---------------------------------------------------------------------------
# 4. Accuracy / outliers
# ---------------------------------------------------------------------------
def outlier_report(df: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """Flag statistical outliers in every continuous numeric column.

    Binary and near-constant columns are skipped: a ``is_cold_chain`` flag is
    not "an outlier" because 12% of batches are refrigerated, and reporting it
    as one would bury the findings that matter.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.
    method : {'iqr', 'zscore'}, optional
        Detection method. Defaults to ``config.data_quality.outlier_method``.
        ``iqr`` uses Tukey fences at ``iqr_multiplier``; ``zscore`` uses
        ``zscore_threshold`` standard deviations either side of the mean.

    Returns
    -------
    pandas.DataFrame
        Columns ``column``, ``method``, ``lower_bound``, ``upper_bound``,
        ``outlier_count``, ``outlier_pct``, ``min``, ``max``, sorted by
        ``outlier_pct`` descending.

    Raises
    ------
    ValueError
        If ``method`` is neither ``'iqr'`` nor ``'zscore'``.
    """
    cfg = get_config()
    chosen = (method or cfg.data_quality.outlier_method).lower()
    if chosen not in {"iqr", "zscore"}:
        raise ValueError(f"Unknown outlier method '{chosen}'. Use 'iqr' or 'zscore'.")

    multiplier = float(cfg.data_quality.iqr_multiplier)
    threshold = float(cfg.data_quality.zscore_threshold)
    min_unique = int(_dq("min_unique_for_outliers"))

    rows: list[dict[str, Any]] = []
    for column in _numeric_columns(df):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty or values.nunique() <= min_unique:
            continue  # flag/constant column - see docstring

        if chosen == "iqr":
            q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
            spread = q3 - q1
            if spread <= 0:
                continue  # degenerate distribution: fences would flag the tails
            lower, upper = q1 - multiplier * spread, q3 + multiplier * spread
        else:
            mean, std = float(values.mean()), float(values.std(ddof=0))
            if std <= 0:
                continue
            lower, upper = mean - threshold * std, mean + threshold * std

        outliers = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "column": str(column),
                "method": chosen,
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "outlier_count": outliers,
                "outlier_pct": _pct(outliers, len(values)),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
            }
        )

    if not rows:
        log.info("Accuracy: no continuous numeric column to test for outliers.")
        return _empty_frame(
            ["column", "method", "lower_bound", "upper_bound",
             "outlier_count", "outlier_pct", "min", "max"]
        )

    report = pd.DataFrame(rows).sort_values(
        ["outlier_pct", "column"], ascending=[False, True]
    ).reset_index(drop=True)
    log.info(
        "Accuracy: %s method flagged %d values across %d columns.",
        chosen, int(report["outlier_count"].sum()), len(report),
    )
    return report


# ---------------------------------------------------------------------------
# 5. Profile
# ---------------------------------------------------------------------------
def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Profile every column, numeric and categorical, in one tidy frame.

    ``zeros`` and ``negatives`` are carried alongside the usual moments because
    in this domain they are semantic, not statistical: a zero ``units_dispensed``
    is a batch that never reached a patient, and a negative ``units_*`` value is
    impossible and therefore evidence of a defect.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.

    Returns
    -------
    pandas.DataFrame
        One row per column with ``column`` and ``col_type`` (``numeric`` /
        ``categorical`` / ``datetime``) plus, where applicable, ``count``,
        ``mean``, ``std``, ``min``, ``q25``, ``median``, ``q75``, ``max``,
        ``skew``, ``kurtosis``, ``zeros``, ``negatives``, ``unique``, ``top``,
        ``freq``, ``top_pct``.
    """
    numeric_columns = set(_numeric_columns(df))
    rows: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        values = series.dropna()
        count = int(len(values))

        if column in numeric_columns:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            rows.append(
                {
                    "column": str(column),
                    "col_type": "numeric",
                    "count": count,
                    "mean": round(float(numeric.mean()), 4) if count else np.nan,
                    "std": round(float(numeric.std()), 4) if count > 1 else np.nan,
                    "min": round(float(numeric.min()), 4) if count else np.nan,
                    "q25": round(float(numeric.quantile(0.25)), 4) if count else np.nan,
                    "median": round(float(numeric.median()), 4) if count else np.nan,
                    "q75": round(float(numeric.quantile(0.75)), 4) if count else np.nan,
                    "max": round(float(numeric.max()), 4) if count else np.nan,
                    "skew": round(float(numeric.skew()), 4) if count > 2 else np.nan,
                    "kurtosis": round(float(numeric.kurt()), 4) if count > 3 else np.nan,
                    "zeros": int((numeric == 0).sum()),
                    "negatives": int((numeric < 0).sum()),
                    "unique": int(numeric.nunique()),
                }
            )
            continue

        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        counts = values.value_counts()
        top = counts.index[0] if len(counts) else np.nan
        freq = int(counts.iloc[0]) if len(counts) else 0
        rows.append(
            {
                "column": str(column),
                "col_type": "datetime" if is_datetime else "categorical",
                "count": count,
                "unique": int(values.nunique()),
                "top": str(top) if len(counts) else "",
                "freq": freq,
                "top_pct": _pct(freq, count),
                # Range is the useful summary for a date column; the moments are not.
                "min": str(values.min()) if is_datetime and count else np.nan,
                "max": str(values.max()) if is_datetime and count else np.nan,
            }
        )

    if not rows:
        return _empty_frame(["column", "col_type", "count"])

    ordered = [
        "column", "col_type", "count", "unique", "mean", "std", "min", "q25",
        "median", "q75", "max", "skew", "kurtosis", "zeros", "negatives",
        "top", "freq", "top_pct",
    ]
    report = pd.DataFrame(rows)
    report = report[[column for column in ordered if column in report.columns]]
    log.info("Profiled %d columns (%d numeric).", len(report), len(numeric_columns))
    return report.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Consistency
# ---------------------------------------------------------------------------
# Pairwise invariants that are not part of the funnel walk. Each entry is
# (left, right, operator, description) where the rule reads "left <op> right".
_PAIRWISE_CHECKS: tuple[tuple[str, str, str, str], ...] = (
    (
        "expiry_date", "date_manufacturing", ">",
        "Expiry date must fall after the manufacturing date - a batch cannot "
        "expire before it exists.",
    ),
    (
        "units_received", "units_shipped", "<=",
        "Units received cannot exceed units shipped - stock does not multiply "
        "in transit.",
    ),
    (
        "units_fulfilled", "units_demanded", "<=",
        "Units fulfilled cannot exceed units demanded - fill rate would exceed "
        "100%.",
    ),
)
# Note: no ``units_issued`` vs ``units_on_hand`` rule. ``units_on_hand`` is a
# closing snapshot and ``units_issued`` a flow over the month, so neither
# direction is an invariant - asserting one would manufacture violations.


def _ordered_funnel_columns(kind: str) -> list[str]:
    """Funnel column names in configured stage order (``kind`` = units|dates)."""
    cfg = get_config()
    mapping = cfg.funnel.unit_columns if kind == "units" else cfg.funnel.date_columns
    return [mapping[stage] for stage in cfg.funnel.stages if stage in mapping]


def _compare(left: pd.Series, right: pd.Series, operator: str) -> pd.Series:
    """Boolean mask of rows SATISFYING ``left <operator> right``."""
    if operator == "<=":
        return left <= right
    if operator == ">=":
        return left >= right
    if operator == ">":
        return left > right
    return left < right


def _applicable_consistency_checks(
    df: pd.DataFrame,
) -> list[tuple[str, str, str, str]]:
    """Assemble the ``(left, right, operator, description)`` checks this frame supports."""
    checks: list[tuple[str, str, str, str]] = []

    # Funnel monotonicity: units may only ever fall, dates may only ever advance.
    units = [c for c in _ordered_funnel_columns("units") if c in df.columns]
    for left, right in zip(units, units[1:]):
        checks.append(
            (left, right, ">=",
             f"Funnel monotonicity: {left} must be >= {right}; units cannot be "
             "created between stages.")
        )
    dates = [c for c in _ordered_funnel_columns("dates") if c in df.columns]
    for left, right in zip(dates, dates[1:]):
        checks.append(
            (left, right, "<=",
             f"Chronology: {left} must not fall after {right}; a batch cannot "
             "reach a stage before the one preceding it.")
        )
    checks.extend(_PAIRWISE_CHECKS)
    return [
        check for check in checks
        if check[0] in df.columns and check[1] in df.columns
    ]


def _breach_mask(df: pd.DataFrame, check: tuple[str, str, str, str]) -> pd.Series:
    """Boolean mask of rows that BREACH one cross-field invariant."""
    left, right, operator, _ = check
    both_present = df[left].notna() & df[right].notna()
    satisfied = _compare(df[left], df[right], operator)
    return both_present & ~satisfied.fillna(False)


def consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    """Test cross-field invariants that must hold by construction.

    This is the part of the audit that actually proves the data is trustworthy.
    A missing value is visible; a batch whose ``units_packaged`` exceeds its
    ``units_qa_passed`` is not - the frame looks perfectly complete and every
    value is individually plausible, yet the record describes something that
    cannot physically happen. Only a cross-field check catches it.

    The funnel ordering comes from ``config.funnel``, so it stays in lockstep
    with the rest of the platform. Only rows where both sides of a comparison
    are populated are counted, and only checks whose columns are both present
    are run.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.

    Returns
    -------
    pandas.DataFrame
        Columns ``check_name``, ``description``, ``checked``, ``violations``,
        ``violation_pct``, ``passed`` (bool), sorted worst-first.
    """
    rows: list[dict[str, Any]] = []
    for check in _applicable_consistency_checks(df):
        left, right, operator, description = check
        checked = int((df[left].notna() & df[right].notna()).sum())
        if not checked:
            continue
        violations = int(_breach_mask(df, check).sum())
        rows.append(
            {
                "check_name": f"{left} {operator} {right}",
                "description": description,
                "checked": checked,
                "violations": violations,
                "violation_pct": _pct(violations, checked),
                "passed": violations == 0,
            }
        )

    if not rows:
        log.info("Consistency: no cross-field invariant applies to this frame.")
        return _empty_frame(
            ["check_name", "description", "checked", "violations",
             "violation_pct", "passed"]
        )

    report = pd.DataFrame(rows).sort_values(
        ["violations", "check_name"], ascending=[False, True]
    ).reset_index(drop=True)
    log.info(
        "Consistency: %d/%d invariants hold.",
        int(report["passed"].sum()), len(report),
    )
    return report


# ---------------------------------------------------------------------------
# 7. Composite score
# ---------------------------------------------------------------------------
def _grade_for(score: float) -> str:
    """Map a 0-100 score to a letter using ``config.data_quality.grade_bands``."""
    bands = get_config().data_quality.grade_bands
    for letter, lower_bound in sorted(bands.items(), key=lambda item: -item[1]):
        if score >= float(lower_bound):
            return str(letter)
    return "F"


def data_quality_score(
    df: pd.DataFrame, subset: Sequence[str] | None = None
) -> dict[str, Any]:
    """Score a frame on five weighted quality dimensions.

    Each dimension is one minus a defect rate, expressed 0-100:

    ``completeness``
        Share of cells that are populated.
    ``uniqueness``
        Share of rows that are not redundant copies. When ``subset`` names a
        business key, the worse of the row-level and key-level rates is used -
        a duplicated key is at least as serious as a duplicated row.
    ``validity``
        Share of *records* in which every applicable domain rule holds. Measured
        per record rather than per value so that adding more passing rules
        cannot inflate the score by diluting the denominator.
    ``consistency``
        Share of *records* in which every applicable cross-field invariant holds.
    ``accuracy``
        One minus the mean per-column outlier rate, with each column's
        contribution capped at ``accuracy_outlier_cap``. Without the cap a
        single legitimately heavy-tailed variable - batch value, freight cost -
        would drag the score down for being skewed rather than for being wrong.

    Dimensions with nothing to measure score 100: the absence of an applicable
    rule is not evidence of a defect.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.
    subset : sequence of str, optional
        Business key columns that must be unique.

    Returns
    -------
    dict
        ``completeness``, ``uniqueness``, ``validity``, ``consistency``,
        ``accuracy`` (each 0-100), ``overall_score``, ``grade``, ``verdict``,
        plus ``rows`` and ``columns``.
    """
    weights = get_config().data_quality.weights
    cap = float(_dq("accuracy_outlier_cap"))
    n_rows, n_cols = df.shape

    # --- completeness ------------------------------------------------------
    cells = n_rows * n_cols
    missing_rate = (df.isna().to_numpy().sum() / cells) if cells else 0.0

    # --- uniqueness --------------------------------------------------------
    duplicates = duplicate_report(df, subset=subset)
    duplicate_rate = duplicates["duplicate_pct"] / 100.0
    if "key_duplicate_pct" in duplicates:
        duplicate_rate = max(duplicate_rate, duplicates["key_duplicate_pct"] / 100.0)

    # --- validity ----------------------------------------------------------
    # Measured per RECORD, not per checked value. A value-level denominator
    # would mean that adding more passing rules to a table raises its validity
    # score, which is backwards: a record is conformant or it is not.
    rules = default_rules(df)
    invalid_rows = pd.Series(False, index=df.index)
    for column, rule in rules.items():
        invalid_rows = invalid_rows | _violation_mask(df[column], rule)
    invalid_rate = (int(invalid_rows.sum()) / n_rows) if (rules and n_rows) else 0.0

    # --- consistency -------------------------------------------------------
    checks = _applicable_consistency_checks(df)
    inconsistent_rows = pd.Series(False, index=df.index)
    for check in checks:
        inconsistent_rows = inconsistent_rows | _breach_mask(df, check)
    inconsistent_rate = (
        int(inconsistent_rows.sum()) / n_rows if (checks and n_rows) else 0.0
    )

    # --- accuracy ----------------------------------------------------------
    outliers = outlier_report(df)
    if len(outliers):
        capped = (outliers["outlier_pct"] / 100.0).clip(upper=cap)
        outlier_rate = float(capped.mean())
    else:
        outlier_rate = 0.0

    dimensions = {
        "completeness": round(100.0 * (1.0 - missing_rate), 2),
        "uniqueness": round(100.0 * (1.0 - duplicate_rate), 2),
        "validity": round(100.0 * (1.0 - invalid_rate), 2),
        "consistency": round(100.0 * (1.0 - inconsistent_rate), 2),
        "accuracy": round(100.0 * (1.0 - outlier_rate), 2),
    }

    # Normalise defensively: the config comments require the weights to sum to
    # 1.0, but a mis-edit must not silently rescale every score in the platform.
    total_weight = sum(float(weights[name]) for name in dimensions)
    overall = round(
        sum(dimensions[name] * float(weights[name]) for name in dimensions)
        / total_weight,
        2,
    )
    grade = _grade_for(overall)

    weakest = min(dimensions, key=lambda name: dimensions[name])
    fitness = {
        "A": "fit for regulatory-grade reporting as-is.",
        "B": "usable; remediate the weakest dimension before publishing figures.",
        "C": "conditional - remediate before any number leaves the building.",
        "D": "not fit for decision-making; quarantine and re-ingest.",
    }.get(grade, "failing; do not use - trace the defect to the source system.")
    verdict = (
        f"Grade {grade} ({overall:.2f}/100) over {n_rows:,} rows x {n_cols} columns. "
        f"Weakest dimension: {weakest} ({dimensions[weakest]:.2f}). {fitness}"
    )

    result: dict[str, Any] = {**dimensions,
                             "overall_score": overall,
                             "grade": grade,
                             "verdict": verdict,
                             "rows": n_rows,
                             "columns": n_cols}
    log.info("Quality score: %s", verdict)
    return result


# ---------------------------------------------------------------------------
# 8. Remediation plan
# ---------------------------------------------------------------------------
def preprocessing_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the audit findings into named, actionable remediation steps.

    An audit that stops at "3.2% missing" hands the reader the hardest part of
    the job. Each row here names the treatment and, crucially, the reason -
    because under GxP a data correction has to be justified, not just applied.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame in the platform. Never modified.

    Returns
    -------
    pandas.DataFrame
        Columns ``issue``, ``column``, ``severity`` (``High`` / ``Medium`` /
        ``Low``), ``recommended_action``, ``rationale``, sorted by severity.
    """
    drop_band = float(_dq("missing_drop_pct"))
    high_band = float(_dq("missing_high_pct"))
    low_band = float(_dq("missing_low_pct"))
    heavy_tail = float(_dq("heavy_tail_skew"))
    material_violation = float(_dq("material_violation_pct"))
    material_outlier = float(_dq("material_outlier_pct"))

    numeric_columns = set(_numeric_columns(df))
    recommendations: list[dict[str, Any]] = []

    # --- completeness ------------------------------------------------------
    for record in missing_value_report(df).to_dict("records"):
        column, pct = record["column"], record["missing_pct"]
        if record["missing_count"] == 0:
            continue
        count = record["missing_count"]
        if pct >= drop_band:
            action = (
                f"Confirm whether the null is structural (the field only applies "
                f"to a subset of records); if so replace '{column}' with a "
                f"boolean has_{column} indicator, otherwise drop the column."
            )
            rationale = (
                f"{pct:.2f}% of rows are null. Imputing {count:,} values would "
                "invent more data than it recovers, and invented values cannot "
                "be defended in an audit."
            )
            severity = "High"
        elif pct > high_band:
            if column in numeric_columns:
                action = (
                    f"Median-impute '{column}' and add a '{column}_was_missing' "
                    "indicator column."
                )
                rationale = (
                    f"{pct:.2f}% missing is too much to drop rows over, and the "
                    "indicator keeps the imputation visible to every downstream "
                    "model instead of hiding it."
                )
            else:
                action = (
                    f"Impute '{column}' with an explicit 'Unknown' level; do not "
                    "fold it into the modal category."
                )
                rationale = (
                    f"{pct:.2f}% missing. Folding {count:,} unknowns into the "
                    "most common level would manufacture a signal that is not "
                    "in the source data."
                )
            severity = "High"
        elif pct >= low_band:
            skew = float(pd.to_numeric(df[column], errors="coerce").skew()) \
                if column in numeric_columns else 0.0
            if column in numeric_columns:
                central = "Median" if abs(skew) > heavy_tail else "Mean"
                action = f"{central}-impute '{column}' ({count:,} rows)."
                rationale = (
                    f"{pct:.2f}% missing with skew {skew:.2f}; the "
                    f"{central.lower()} is the stable centre for this shape."
                )
            else:
                action = f"Mode-impute '{column}' ({count:,} rows)."
                rationale = (
                    f"{pct:.2f}% missing - small enough that the modal category "
                    "will not materially shift the distribution."
                )
            severity = "Medium"
        else:
            action = (
                f"Median-impute '{column}' or drop the {count:,} affected rows - "
                "either is defensible at this volume."
            )
            rationale = (
                f"Only {pct:.2f}% missing; the choice cannot move any published "
                "aggregate."
            )
            severity = "Low"
        recommendations.append(
            {
                "issue": "Missing values",
                "column": column,
                "severity": severity,
                "recommended_action": action,
                "rationale": rationale,
            }
        )

    # --- uniqueness --------------------------------------------------------
    duplicates = duplicate_report(df)
    if duplicates["exact_duplicate_rows"]:
        id_columns = [c for c in df.columns if str(c).endswith("_id")]
        key_hint = (
            f" then re-check key uniqueness with duplicate_report(df, subset={id_columns})"
            if id_columns else ""
        )
        recommendations.append(
            {
                "issue": "Duplicate rows",
                "column": "<all columns>",
                "severity": "High",
                "recommended_action": (
                    f"Deduplicate with df.drop_duplicates(keep='first'), removing "
                    f"{duplicates['exact_duplicate_rows']:,} redundant rows"
                    f"{key_hint}."
                ),
                "rationale": (
                    f"{duplicates['duplicate_pct']:.2f}% of rows are exact copies. "
                    "Every count, sum and rate computed from this table is "
                    "currently overstated, and ALCOA+ 'Original' cannot be "
                    "asserted while two identical records exist."
                ),
            }
        )

    # --- validity ----------------------------------------------------------
    for record in validity_report(df).to_dict("records"):
        if record["violations"] == 0:
            continue
        is_categorical = record["rule"].startswith("value in")
        severity = "High" if record["violation_pct"] > material_violation else "Medium"
        if is_categorical:
            action = (
                f"Map the out-of-domain values ({record['sample_bad_values']}) to a "
                f"canonical level, or quarantine those {record['violations']:,} rows "
                "for source-system review; never coerce them silently."
            )
            rationale = (
                f"{record['violations']:,} values fall outside the declared domain "
                f"({record['rule']}). An unrecognised level breaks every "
                "encoder, group-by and join that assumes the known set."
            )
        else:
            action = (
                f"Quarantine the {record['violations']:,} rows outside "
                f"{record['rule']}; clip to the bound only once the excursion is "
                "confirmed to be a recording error."
            )
            rationale = (
                f"Values such as {record['sample_bad_values']} are physically "
                "impossible for this field, so the record is either mis-keyed or "
                "mis-mapped upstream."
            )
        recommendations.append(
            {
                "issue": "Invalid values",
                "column": record["column"],
                "severity": severity,
                "recommended_action": action,
                "rationale": rationale,
            }
        )

    # --- consistency -------------------------------------------------------
    for record in consistency_report(df).to_dict("records"):
        if record["passed"]:
            continue
        recommendations.append(
            {
                "issue": "Cross-field inconsistency",
                "column": record["check_name"],
                "severity": "High",
                "recommended_action": (
                    f"Quarantine the {record['violations']:,} violating rows and "
                    "trace the defect to the upstream ETL stage; do not correct "
                    "the values in place."
                ),
                "rationale": (
                    f"{record['description']} {record['violation_pct']:.2f}% of "
                    "evaluated rows breach it, so the table describes something "
                    "that cannot have happened - patching the symptom would hide "
                    "a live pipeline fault."
                ),
            }
        )

    # --- accuracy ----------------------------------------------------------
    for record in outlier_report(df).to_dict("records"):
        column = record["column"]
        if record["outlier_count"] == 0:
            continue
        skew = float(pd.to_numeric(df[column], errors="coerce").skew())
        if abs(skew) > heavy_tail:
            action = (
                f"Investigate rather than remove - consider a log1p transform of "
                f"'{column}' for modelling and keep the raw column for reporting."
            )
            rationale = (
                f"Skew {skew:.2f} indicates a genuine heavy tail (large batches, "
                f"large orders), so the {record['outlier_count']:,} flagged values "
                "are most likely real business events, not errors."
            )
            severity = "Low"
        elif record["outlier_pct"] > material_outlier:
            action = (
                f"Winsorise '{column}' at the Tukey fences "
                f"[{record['lower_bound']}, {record['upper_bound']}] before "
                "distance-based or linear modelling."
            )
            rationale = (
                f"{record['outlier_pct']:.2f}% of values sit outside the fences on "
                "a roughly symmetric distribution, which is enough to distort a "
                "mean or a coefficient."
            )
            severity = "Medium"
        else:
            action = (
                f"Review the {record['outlier_count']:,} flagged rows in '{column}' "
                f"against range [{record['min']}, {record['max']}]; winsorise only "
                "if they prove to be recording errors."
            )
            rationale = (
                f"Only {record['outlier_pct']:.2f}% of values are extreme - too few "
                "to justify a blanket transformation."
            )
            severity = "Low"
        recommendations.append(
            {
                "issue": "Statistical outliers",
                "column": column,
                "severity": severity,
                "recommended_action": action,
                "rationale": rationale,
            }
        )

    # --- zero-variance columns --------------------------------------------
    for column in df.columns:
        if len(df) and df[column].nunique(dropna=True) <= 1:
            recommendations.append(
                {
                    "issue": "Zero-variance column",
                    "column": str(column),
                    "severity": "Low",
                    "recommended_action": f"Drop '{column}' from feature sets.",
                    "rationale": (
                        "A single distinct value carries no information and only "
                        "adds a column to every join and model matrix."
                    ),
                }
            )

    if not recommendations:
        log.info("Remediation: no action required - all checks passed.")
        return _empty_frame(
            ["issue", "column", "severity", "recommended_action", "rationale"]
        )

    report = pd.DataFrame(recommendations)
    report["_rank"] = report["severity"].map(_SEVERITY_ORDER)
    report = (
        report.sort_values(["_rank", "issue", "column"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    log.info(
        "Remediation: %d recommended actions (%d High severity).",
        len(report), int((report["severity"] == "High").sum()),
    )
    return report


# ---------------------------------------------------------------------------
# 9-10. Platform-level orchestration
# ---------------------------------------------------------------------------
def assess_dataset(name: str, raw: bool = True) -> dict[str, Any]:
    """Run the full audit against one named platform table.

    Parameters
    ----------
    name : str
        One of ``'drug200'``, ``'batches'``, ``'shipments'``, ``'inventory'``,
        ``'demand'``.
    raw : bool, default True
        Profile the *bronze* extract, which is the point of a data quality
        audit: you assess what the source systems handed over, before any
        remediation. Pass ``False`` to assess the cleaned *silver* layer -
        useful for proving the cleaning layer actually worked.

    Returns
    -------
    dict
        Stable keys: ``dataset``, ``layer``, ``rows``, ``columns``,
        ``key_columns``, ``missing``, ``duplicates``, ``validity``,
        ``outliers``, ``summary``, ``consistency``, ``recommendations``
        and ``score``.

    Raises
    ------
    KeyError
        If ``name`` is not one of the five platform tables.
    """
    if name not in _DATASET_KEYS:
        raise KeyError(
            f"Unknown dataset '{name}'. Known: {sorted(_DATASET_KEYS)}"
        )

    # load_table caches; .copy() guarantees the audit cannot touch the cache.
    frame = loader.load_table(name, raw=raw).copy()
    key = _DATASET_KEYS[name]
    log.info("Assessing '%s' (%d rows x %d cols).", name, *frame.shape)

    return {
        "dataset": name,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "key_columns": list(key) if key else [],
        "layer": "bronze (raw extract)" if raw else "silver (cleaned)",
        "missing": missing_value_report(frame),
        "duplicates": duplicate_report(frame, subset=key),
        "validity": validity_report(frame),
        "outliers": outlier_report(frame),
        "summary": summary_statistics(frame),
        "consistency": consistency_report(frame),
        "recommendations": preprocessing_recommendations(frame),
        "score": data_quality_score(frame, subset=key),
    }


def assess_all(names: Iterable[str] | None = None, raw: bool = True) -> pd.DataFrame:
    """Score every platform table and return a comparison scoreboard.

    Parameters
    ----------
    names : iterable of str, optional
        Tables to assess. Defaults to all five platform tables.
    raw : bool, default True
        Assess the bronze extract. Pass ``False`` to score the cleaned layer;
        running both is how the dashboard demonstrates the uplift delivered by
        the cleaning pipeline.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset``, ``rows``, ``columns``, ``completeness``,
        ``uniqueness``, ``validity``, ``consistency``, ``accuracy``,
        ``overall_score``, ``grade``, sorted worst-first so the table that needs
        attention is the first row a reviewer reads.
    """
    targets = list(names) if names is not None else list(_DATASET_KEYS)
    rows: list[dict[str, Any]] = []
    for name in targets:
        report = assess_dataset(name, raw=raw)
        score = report["score"]
        rows.append(
            {
                "dataset": name,
                "rows": report["rows"],
                "columns": report["columns"],
                "completeness": score["completeness"],
                "uniqueness": score["uniqueness"],
                "validity": score["validity"],
                "consistency": score["consistency"],
                "accuracy": score["accuracy"],
                "overall_score": score["overall_score"],
                "grade": score["grade"],
            }
        )

    scoreboard = pd.DataFrame(rows).sort_values(
        ["overall_score", "dataset"], ascending=[True, True]
    ).reset_index(drop=True)
    log.info(
        "Scoreboard: %d tables assessed, lowest score %.2f (%s).",
        len(scoreboard),
        float(scoreboard["overall_score"].min()) if len(scoreboard) else 0.0,
        scoreboard["dataset"].iloc[0] if len(scoreboard) else "n/a",
    )
    return scoreboard


def quality_uplift(names: Iterable[str] | None = None) -> pd.DataFrame:
    """Score the bronze and silver layers side by side to quantify the cleaning uplift.

    This is the headline evidence that the pipeline earns its keep: the same
    scoring function applied before and after remediation, with the delta per
    quality dimension.

    Parameters
    ----------
    names : iterable of str, optional
        Tables to compare. Defaults to all platform tables.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset``, ``raw_score``, ``clean_score``, ``uplift``,
        ``raw_grade``, ``clean_grade``, plus per-dimension deltas, sorted by
        the largest uplift first.
    """
    dimensions = ["completeness", "uniqueness", "validity", "consistency", "accuracy"]
    before = assess_all(names, raw=True).set_index("dataset")
    after = assess_all(names, raw=False).set_index("dataset")

    rows: list[dict[str, Any]] = []
    for dataset in before.index:
        if dataset not in after.index:
            continue
        row: dict[str, Any] = {
            "dataset": dataset,
            "raw_score": before.loc[dataset, "overall_score"],
            "clean_score": after.loc[dataset, "overall_score"],
            "uplift": round(
                float(after.loc[dataset, "overall_score"])
                - float(before.loc[dataset, "overall_score"]), 2),
            "raw_grade": before.loc[dataset, "grade"],
            "clean_grade": after.loc[dataset, "grade"],
        }
        for dimension in dimensions:
            row[f"{dimension}_delta"] = round(
                float(after.loc[dataset, dimension]) - float(before.loc[dataset, dimension]), 2)
        rows.append(row)

    uplift = (pd.DataFrame(rows).sort_values("uplift", ascending=False)
              .reset_index(drop=True))
    log.info("Cleaning uplift computed for %d tables (max +%.2f points)",
             len(uplift), float(uplift["uplift"].max()) if len(uplift) else 0.0)
    return uplift


__all__ = [
    "default_rules",
    "missing_value_report",
    "duplicate_report",
    "validity_report",
    "outlier_report",
    "summary_statistics",
    "consistency_report",
    "data_quality_score",
    "preprocessing_recommendations",
    "assess_dataset",
    "assess_all",
    "quality_uplift",
]
