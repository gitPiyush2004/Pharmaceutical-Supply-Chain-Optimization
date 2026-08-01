"""
Cleaning, feature engineering and preprocessing for the PharmaChain ML models.

Business purpose
----------------
Two supervised models sit on top of the platform's data layer:

* **Drug classification** - given a patient's age, sex, blood pressure,
  cholesterol level and serum sodium/potassium ratio, recommend which of the
  five formulary drugs they should receive. This is the clinical decision
  support half of the platform.
* **Batch risk** - given the storage and supply chain conditions a
  manufactured batch actually experienced, predict the quality risk tier
  (Low / Medium / High) that Quality Assurance would assign to it. This is the
  operational half: it turns a retrospective QA label into a forward-looking
  screen.

This module owns everything that happens *before* an estimator sees the data:
normalising messy categorical text, deriving interpretable domain features, and
assembling the sklearn ``ColumnTransformer`` that imputes, scales and one-hot
encodes.

Design notes
------------
* Every threshold is read from ``config.yaml`` (``ml.feature_engineering`` and
  ``stability``), never hard-coded, so a clinical or storage cut-off can be
  revised without touching Python.
* Feature engineering is **deterministic and row-local** for the clinical model,
  which means the exact same transformation can be replayed at inference time
  for a single patient with no fitted state. The one data-dependent statistic in
  the batch model (the median cycle time) is passed in explicitly and persisted
  in the model metadata so training and serving agree.
* No Streamlit, no plotting. Pure pandas / sklearn.

Example
-------
>>> from src.data.loader import load_clinical
>>> from src.ml import preprocess as pp
>>> frame = pp.engineer_clinical_features(pp.clean_clinical(load_clinical()))
>>> sorted(frame["na_to_k_band"].unique())
['High', 'Low', 'Normal']
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import get_config
from src.logger import get_logger

log = get_logger(__name__)

# Raw clinical columns whose values are free-text categories. The Kaggle source
# mixes casing and occasionally carries stray whitespace, so they are all
# normalised to upper case before anything downstream compares them.
_CLINICAL_CATEGORICAL_RAW: tuple[str, ...] = ("Sex", "BP", "Cholesterol")
_CLINICAL_NUMERIC_RAW: tuple[str, ...] = ("Age", "Na_to_K")

# Feature names produced by :func:`engineer_clinical_features`.
CLINICAL_ENGINEERED_NUMERIC: tuple[str, ...] = (
    "bp_risk_score",
    "cholesterol_risk",
    "combined_risk_score",
    "high_na_to_k",
)
CLINICAL_ENGINEERED_CATEGORICAL: tuple[str, ...] = ("age_group", "na_to_k_band")

# Feature names produced by :func:`engineer_batch_features`.
BATCH_ENGINEERED_NUMERIC: tuple[str, ...] = (
    "temp_excursion_c",
    "humidity_excess",
    "thermal_load",
    "cycle_time_ratio",
)

# Batch columns that are categorical in the config feature list; everything
# else in that list is numeric (``is_cold_chain`` is already a 0/1 indicator and
# is left numeric so tree splits can use it directly).
_BATCH_CATEGORICAL: tuple[str, ...] = ("drug_code", "region")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw drug200 clinical frame.

    Strips and upper-cases the categorical columns, coerces the numeric columns,
    trims whitespace from the target label and removes exact duplicate patient
    rows. The target's original casing (``DrugY`` vs ``drugX``) is deliberately
    preserved: those strings are the formulary codes the dashboard displays.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw frame from :func:`src.data.loader.load_clinical`. Must contain
        ``Age``, ``Sex``, ``BP``, ``Cholesterol`` and ``Na_to_K``; ``Drug`` is
        optional so the same function can clean inference payloads.

    Returns
    -------
    pandas.DataFrame
        A cleaned copy. The input is never mutated.

    Raises
    ------
    ValueError
        If any required feature column is missing.
    """
    required = set(_CLINICAL_NUMERIC_RAW) | set(_CLINICAL_CATEGORICAL_RAW)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Clinical frame is missing required column(s): {sorted(missing)}. "
            f"Present: {sorted(df.columns)}"
        )

    out = df.copy()
    changed: dict[str, int] = {}

    for column in _CLINICAL_CATEGORICAL_RAW:
        before = out[column].astype("string")
        after = before.str.strip().str.upper()
        # Count only genuine edits so the log reports real data quality issues
        # rather than the no-op of re-writing an already clean column.
        changed[column] = int((before.fillna("") != after.fillna("")).sum())
        out[column] = after

    for column in _CLINICAL_NUMERIC_RAW:
        coerced = pd.to_numeric(out[column], errors="coerce")
        changed[f"{column}:non_numeric"] = int(coerced.isna().sum() - out[column].isna().sum())
        out[column] = coerced

    cfg_target = get_config().ml.drug_classification.target
    if cfg_target in out.columns:
        out[cfg_target] = out[cfg_target].astype("string").str.strip()

    before_rows = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    duplicates = before_rows - len(out)

    log.info(
        "clean_clinical: %d rows in, %d out | %d duplicate row(s) dropped | "
        "normalised %s | non-numeric coercions %s",
        before_rows,
        len(out),
        duplicates,
        {k: v for k, v in changed.items() if ":" not in k},
        {k.split(":")[0]: v for k, v in changed.items() if ":" in k and v},
    )
    return out


# ---------------------------------------------------------------------------
# Feature engineering - clinical
# ---------------------------------------------------------------------------
def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interpretable clinical features for the drug recommender.

    The point of these features is explainability, not raw accuracy: a
    prescribing decision has to be defensible to a clinician, so the model is
    given the same coarse strata a clinician reasons with (age band, Na/K band,
    BP and cholesterol severity) alongside the continuous measurements.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`clean_clinical`.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with the engineered columns appended.

    Raises
    ------
    ValueError
        If a required source column is absent.
    """
    fe = get_config().ml.feature_engineering.clinical
    required = set(_CLINICAL_NUMERIC_RAW) | set(_CLINICAL_CATEGORICAL_RAW)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot engineer clinical features, missing column(s): {sorted(missing)}"
        )

    out = df.copy()

    # Age band. Drug response is not linear in age - metabolism, renal clearance
    # and comorbidity load shift in steps - so a banded view lets a tree split on
    # a clinically meaningful stratum instead of an arbitrary numeric cut.
    edges = [-np.inf, *[float(b) for b in fe.age_bins], np.inf]
    out["age_group"] = pd.cut(
        out["Age"], bins=edges, labels=list(fe.age_labels), right=False
    ).astype("string")

    # Na/K band. The sodium-to-potassium ratio is the dominant discriminator in
    # this formulary: above the high cut-off the electrolyte imbalance itself
    # drives the prescription regardless of BP or cholesterol. Banding makes
    # that clinical rule explicit rather than something the tree must rediscover.
    na_low = float(fe.na_to_k_low)
    na_high = float(fe.na_to_k_high)
    out["na_to_k_band"] = pd.cut(
        out["Na_to_K"],
        bins=[-np.inf, na_low, na_high, np.inf],
        labels=["Low", "Normal", "High"],
        right=False,
    ).astype("string")

    # Binary flag at the high threshold. Kept alongside the band because the
    # single most important decision boundary in the dataset is "is the ratio
    # above the high cut-off"; a dedicated 0/1 feature makes that split available
    # at depth 1 even in a heavily depth-limited tree.
    out["high_na_to_k"] = (out["Na_to_K"] >= na_high).astype(int)

    # Ordinal severity encodings. HIGH/NORMAL/LOW are ordered clinical states,
    # not unordered categories; encoding the order lets the model express
    # "severity at least NORMAL" in one split instead of two one-hot splits.
    # float64 (not nullable Int64) because an unmapped level must survive as NaN
    # for SimpleImputer to fill it; sklearn cannot consume pandas' masked ints.
    levels = {str(k).upper(): float(v) for k, v in fe.ordinal_levels.items()}
    out["bp_risk_score"] = out["BP"].map(levels).astype("float64")
    out["cholesterol_risk"] = out["Cholesterol"].map(levels).astype("float64")

    # Combined cardiovascular burden. Two moderate risk factors and one severe
    # factor are clinically different situations that the individual scores
    # cannot separate on their own; the sum is the crude burden index used to
    # triage between the non-DrugY alternatives.
    out["combined_risk_score"] = (
        out["bp_risk_score"] + out["cholesterol_risk"]
    ).astype("float64")

    log.info(
        "engineer_clinical_features: added %s | Na/K band mix %s",
        list(CLINICAL_ENGINEERED_CATEGORICAL) + list(CLINICAL_ENGINEERED_NUMERIC),
        out["na_to_k_band"].value_counts().to_dict(),
    )
    return out


# ---------------------------------------------------------------------------
# Feature engineering - batch risk
# ---------------------------------------------------------------------------
def engineer_batch_features(
    df: pd.DataFrame, median_cycle_time: float | None = None
) -> pd.DataFrame:
    """Derive storage-stress features for the batch risk model.

    Raw storage temperature is not comparable across products: 8 degC is a
    breach for an ambient tablet's neighbour on the shelf but the normal
    condition for a cold-chain biologic. These features re-express the raw
    conditions relative to each product's *labelled* storage condition, which is
    what actually determines degradation.

    Parameters
    ----------
    df : pandas.DataFrame
        Batch frame containing at least ``storage_temp_c``,
        ``storage_humidity_pct``, ``storage_duration_days``,
        ``total_cycle_time_days`` and ``is_cold_chain``.
    median_cycle_time : float, optional
        Median end-to-end cycle time used as the denominator of
        ``cycle_time_ratio``. Computed from ``df`` when omitted. Training passes
        ``None`` and persists the resulting value; serving passes the persisted
        value back so a single-batch prediction is scaled identically.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with the engineered columns appended.

    Raises
    ------
    ValueError
        If a required source column is absent, or if the supplied/derived
        median cycle time is not strictly positive.
    """
    cfg = get_config()
    batch_fe = cfg.ml.feature_engineering.batch
    stability = cfg.stability

    required = {
        "storage_temp_c",
        "storage_humidity_pct",
        "storage_duration_days",
        "total_cycle_time_days",
        "is_cold_chain",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot engineer batch features, missing column(s): {sorted(missing)}"
        )

    out = df.copy()

    # Degrees above the *labelled* storage limit for this product. Cold-chain
    # items are licensed for 2-8 degC, everything else for controlled room
    # temperature, so the reference is per-row rather than global. Clipped at
    # zero: storing colder than the limit is not an excursion for these forms.
    cold_limit = float(batch_fe.cold_chain_max_temp_c)
    ambient_limit = float(stability.reference_temp_ambient)
    limit = np.where(out["is_cold_chain"].astype(float) > 0, cold_limit, ambient_limit)
    out["temp_excursion_c"] = (out["storage_temp_c"].astype(float) - limit).clip(lower=0.0)

    # Relative humidity above the moisture-uptake threshold. Below that point
    # hygroscopic uptake is negligible, so only the excess carries signal.
    out["humidity_excess"] = (
        out["storage_humidity_pct"].astype(float) - float(stability.humidity_threshold)
    ).clip(lower=0.0)

    # Cumulative heat exposure. Degradation kinetics integrate temperature over
    # time: a 10-degree excursion for two days and a 1-degree excursion for
    # twenty days are not the same batch. The product is the first-order
    # (Arrhenius-flavoured) proxy for that integral.
    out["thermal_load"] = out["temp_excursion_c"] * out["storage_duration_days"].astype(float)

    # Cycle time relative to the network median. An absolute day count conflates
    # "this lane is naturally long" with "this batch was unusually slow"; the
    # ratio isolates the second, which is what consumes remaining shelf life.
    if median_cycle_time is None:
        median_cycle_time = float(out["total_cycle_time_days"].astype(float).median())
    median_cycle_time = float(median_cycle_time)
    if not np.isfinite(median_cycle_time) or median_cycle_time <= 0:
        raise ValueError(
            f"median_cycle_time must be a positive finite number, got {median_cycle_time!r}"
        )
    out["cycle_time_ratio"] = out["total_cycle_time_days"].astype(float) / median_cycle_time

    log.info(
        "engineer_batch_features: added %s | median cycle time %.1f d | "
        "%.1f%% of rows show a temperature excursion",
        list(BATCH_ENGINEERED_NUMERIC),
        median_cycle_time,
        100.0 * float((out["temp_excursion_c"] > 0).mean()),
    )
    return out


# ---------------------------------------------------------------------------
# Feature specifications
# ---------------------------------------------------------------------------
def clinical_feature_columns() -> tuple[list[str], list[str]]:
    """Return the (numeric, categorical) model input columns for MODEL 1.

    The base columns come from ``ml.drug_classification`` in the config; the
    engineered columns are appended so config remains the single source of truth
    for the raw feature contract.

    Returns
    -------
    tuple of (list of str, list of str)
        Numeric feature names, then categorical feature names.
    """
    cfg = get_config().ml.drug_classification
    numeric = list(cfg.numeric_features) + list(CLINICAL_ENGINEERED_NUMERIC)
    categorical = list(cfg.categorical_features) + list(CLINICAL_ENGINEERED_CATEGORICAL)
    return numeric, categorical


def batch_feature_columns() -> tuple[list[str], list[str]]:
    """Return the (numeric, categorical) model input columns for MODEL 2.

    Returns
    -------
    tuple of (list of str, list of str)
        Numeric feature names, then categorical feature names. The split is
        driven by ``ml.batch_risk.features`` in the config; ``drug_code`` and
        ``region`` are the categorical members of that list.
    """
    configured = list(get_config().ml.batch_risk.features)
    categorical = [c for c in configured if c in _BATCH_CATEGORICAL]
    numeric = [c for c in configured if c not in _BATCH_CATEGORICAL]
    numeric += list(BATCH_ENGINEERED_NUMERIC)
    return numeric, categorical


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scale: bool = True,
) -> ColumnTransformer:
    """Assemble the sklearn preprocessing transformer.

    Numeric branch: median imputation (robust to the skewed storage-condition
    distributions) followed by optional standardisation. Categorical branch:
    mode imputation followed by dense one-hot encoding that ignores unseen
    levels, so a new warehouse region appearing in production degrades to
    "all zeros" instead of raising at inference time.

    Parameters
    ----------
    numeric_features : sequence of str
        Column names routed through the numeric branch.
    categorical_features : sequence of str
        Column names routed through the categorical branch.
    scale : bool, default True
        Append :class:`~sklearn.preprocessing.StandardScaler` to the numeric
        branch. Trees are scale-invariant, so callers fitting only tree models
        may pass ``False``; it is kept on by default so the same transformer can
        feed a distance- or gradient-based model unchanged.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted transformer. Remaining columns are dropped.

    Raises
    ------
    ValueError
        If both feature lists are empty.
    """
    if not numeric_features and not categorical_features:
        raise ValueError("build_preprocessor requires at least one feature column.")

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False keeps the matrix dense so get_feature_names_out()
        # lines up positionally with tree feature_importances_.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric_features)))
    if categorical_features:
        transformers.append(
            ("categorical", Pipeline(categorical_steps), list(categorical_features))
        )

    log.info(
        "build_preprocessor: %d numeric (scale=%s), %d categorical feature(s)",
        len(numeric_features),
        scale,
        len(categorical_features),
    )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    test_size : float
        Held-out fraction, read from config by the caller.
    seed : int
        Random seed, read from ``project.random_seed`` by the caller.
    stratify : bool, default True
        Preserve the class mix in both folds. Essential here: the rarest drug
        class has only a handful of patients, so an unstratified split can drop
        it from the training set entirely.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` have different lengths.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if stratify else None,
    )
    log.info(
        "split_data: %d train / %d test rows (test_size=%.2f, stratify=%s)",
        len(X_train),
        len(X_test),
        test_size,
        stratify,
    )
    return X_train, X_test, y_train, y_test


def normalise_categorical(value: str, allowed: Iterable[str], field: str) -> str:
    """Upper-case and validate a categorical input value.

    Lets the dashboard and API accept ``'high'``, ``'High'`` or ``'HIGH'``
    interchangeably while still failing loudly on genuine typos.

    Parameters
    ----------
    value : str
        Raw user-supplied value.
    allowed : iterable of str
        Permitted values (compared case-insensitively).
    field : str
        Field name, used in the error message.

    Returns
    -------
    str
        The matching entry from ``allowed``, in its canonical casing.

    Raises
    ------
    ValueError
        If ``value`` is not a string or does not match any allowed value.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}: {value!r}")
    lookup = {str(a).strip().upper(): a for a in allowed}
    key = value.strip().upper()
    if key not in lookup:
        raise ValueError(
            f"Invalid {field}={value!r}. Expected one of {sorted(lookup.values())}."
        )
    return lookup[key]


__all__ = [
    "clean_clinical",
    "engineer_clinical_features",
    "engineer_batch_features",
    "clinical_feature_columns",
    "batch_feature_columns",
    "build_preprocessor",
    "split_data",
    "normalise_categorical",
    "CLINICAL_ENGINEERED_NUMERIC",
    "CLINICAL_ENGINEERED_CATEGORICAL",
    "BATCH_ENGINEERED_NUMERIC",
]
