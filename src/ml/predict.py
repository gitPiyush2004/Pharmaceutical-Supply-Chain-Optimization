"""
Inference layer for the PharmaChain ML models.

Business purpose
----------------
Training happens once, offline. Everything that consumes a model afterwards -
the Streamlit dashboard, a notebook, a batch scoring job - goes through this
module. It owns three responsibilities that must not be duplicated at the call
site:

1. **Artefact loading**, cached so a dashboard rerun does not re-read a joblib
   file on every widget interaction.
2. **Input contract enforcement** - accepting the loose, human-typed values a
   UI produces (``'high'``, ``'High'``, ``'HIGH'``) while rejecting genuine
   nonsense with an actionable ``ValueError`` instead of a cryptic sklearn
   traceback deep inside a transformer.
3. **Replaying the exact training-time feature engineering**, including the
   persisted median cycle time, so a single-row prediction is transformed
   identically to the rows the model was fitted on.

Design notes
------------
The serialised object is a complete sklearn ``Pipeline`` (preprocessing plus a
label-decoding estimator), so callers never see one-hot columns or encoded class
integers - predictions come back as the original label strings. No Streamlit
imports and no plotting: this module returns dictionaries and DataFrames.

Example
-------
>>> from src.ml.predict import predict_drug
>>> out = predict_drug(age=45, sex='F', bp='HIGH', cholesterol='HIGH', na_to_k=18.5)
>>> out['prediction']
'DrugY'
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger
from src.ml import preprocess as pp

log = get_logger(__name__)

# Artefact stems written by :func:`src.ml.train.save_artifacts`.
DRUG_MODEL = "drug_classification"
BATCH_MODEL = "batch_risk"
LATE_DELIVERY_MODEL = "late_delivery"
KNOWN_MODELS: tuple[str, ...] = (DRUG_MODEL, BATCH_MODEL, LATE_DELIVERY_MODEL)

# The one instruction a user can act on when an artefact is missing.
_TRAIN_HINT = "run `python scripts/train_models.py` (or `python -c \"from src.ml.train import train_all; train_all()\"`)"

# Pipeline step names, mirrored from src.ml.train so the fitted OneHotEncoder
# can be interrogated for the categories the model actually knows about.
_PREP_STEP = "preprocess"
_CATEGORICAL_BRANCH = "categorical"
_ONEHOT_STEP = "onehot"


class LoadedModel(NamedTuple):
    """A fitted pipeline paired with the metadata written beside it.

    Attributes
    ----------
    name : str
        Artefact stem, e.g. ``'batch_risk'``.
    pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing + estimator pipeline.
    metadata : dict
        Contents of ``{name}_metadata.json``. Treat as read-only: the instance
        is shared by every caller through the load cache.
    """

    name: str
    pipeline: Any
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=len(KNOWN_MODELS))
def load_model(name: str) -> LoadedModel:
    """Load a serialised pipeline and its metadata, cached per artefact name.

    Parameters
    ----------
    name : str
        Artefact stem: ``'drug_classification'`` or ``'batch_risk'``.

    Returns
    -------
    LoadedModel
        Named tuple of ``(name, pipeline, metadata)``.

    Raises
    ------
    ValueError
        If ``name`` is not a known artefact.
    FileNotFoundError
        If the model or metadata file is missing, with the command to create it.
    """
    if name not in KNOWN_MODELS:
        raise ValueError(f"Unknown model '{name}'. Known models: {list(KNOWN_MODELS)}.")

    models_dir: Path = resolve_path(get_config().paths.models)
    model_path = models_dir / f"{name}_model.joblib"
    metadata_path = models_dir / f"{name}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model '{name}' not found at {model_path}. To create it, {_TRAIN_HINT}."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata for model '{name}' not found at {metadata_path}. "
            f"The artefact pair is incomplete - retrain: {_TRAIN_HINT}."
        )

    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log.info(
        "Loaded model '%s' (%s, trained %s, %d features)",
        name,
        metadata.get("model_name", "unknown"),
        metadata.get("trained_at", "unknown"),
        len(metadata.get("features", [])),
    )
    return LoadedModel(name=name, pipeline=pipeline, metadata=metadata)


def model_summary(name: str) -> dict[str, Any]:
    """Return the stored metadata for a trained model, for dashboard display.

    Parameters
    ----------
    name : str
        Artefact stem.

    Returns
    -------
    dict
        A copy of the metadata (model family, tuned parameters, held-out
        metrics, feature list, class labels, row counts, seed and timestamp).
        Copying protects the cached original from an accidental mutation in the
        UI layer.

    Raises
    ------
    ValueError
        If ``name`` is not a known artefact.
    FileNotFoundError
        If the artefact has not been trained yet.
    """
    return dict(load_model(name).metadata)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _known_categories(model: LoadedModel) -> dict[str, list[str]]:
    """Recover the categorical levels the fitted encoder was trained on.

    Reading them off the fitted ``OneHotEncoder`` rather than hard-coding a list
    means the accepted vocabulary can never drift away from what the model
    actually understands.
    """
    categorical = list(model.metadata.get("categorical_features", []))
    try:
        encoder = (
            model.pipeline.named_steps[_PREP_STEP]
            .named_transformers_[_CATEGORICAL_BRANCH]
            .named_steps[_ONEHOT_STEP]
        )
        levels = [[str(v) for v in cats] for cats in encoder.categories_]
    except (KeyError, AttributeError):  # pragma: no cover - defensive
        return {}
    return dict(zip(categorical, levels))


def _validate_number(value: Any, field: str, minimum: float | None = None) -> float:
    """Coerce a user-supplied value to a finite float or explain why it cannot be.

    Parameters
    ----------
    value : Any
        Raw input.
    field : str
        Field name for the error message.
    minimum : float, optional
        Inclusive lower bound to enforce.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If the value is not numeric, not finite, or below ``minimum``.
    """
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number, got {value!r}.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {number}.")
    return number


def _normalise_frame_categoricals(
    frame: pd.DataFrame, allowed: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """Case-normalise and validate every categorical column of an input frame."""
    out = frame.copy()
    for column, levels in allowed.items():
        if column not in out.columns:
            continue
        out[column] = [
            pp.normalise_categorical(str(value), levels, column) for value in out[column]
        ]
    return out


def _probability_dict(labels: Sequence[str], row: np.ndarray) -> dict[str, float]:
    """Pair class labels with probabilities, highest first."""
    pairs = sorted(zip((str(x) for x in labels), (float(p) for p in row)), key=lambda kv: -kv[1])
    return {label: round(prob, 4) for label, prob in pairs}


# ---------------------------------------------------------------------------
# MODEL 1 - drug classification
# ---------------------------------------------------------------------------
def _prepare_clinical_frame(
    raw: pd.DataFrame, model: LoadedModel
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean, validate and engineer a raw patient frame into model input.

    Returns both the feature matrix and the full engineered frame: the second is
    what the explanation is written from, so the narrative always quotes exactly
    the values the model was given.
    """
    allowed = _known_categories(model)
    # Only the three raw clinical categoricals are user-supplied; age_group and
    # na_to_k_band are derived downstream and must not be validated as input.
    user_supplied = {
        key: values
        for key, values in allowed.items()
        if key in {"Sex", "BP", "Cholesterol"}
    }
    cleaned = _normalise_frame_categoricals(pp.clean_clinical(raw), user_supplied)
    engineered = pp.engineer_clinical_features(cleaned)
    return engineered[list(model.metadata["features"])], engineered


def _clinical_explanation(row: pd.Series, prediction: str, confidence: float) -> str:
    """Build a one-line clinical rationale from the engineered features.

    The Na/K band is cited first because it is the dominant discriminator in the
    fitted model; BP and cholesterol severity follow, since they separate the
    non-DrugY alternatives.
    """
    return (
        f"Predicted {prediction} with {confidence:.1%} confidence. "
        f"Na/K ratio {float(row['Na_to_K']):.2f} falls in the {row['na_to_k_band']} band"
        f"{' (above the high-ratio threshold, the dominant prescribing driver)' if int(row['high_na_to_k']) else ''}; "
        f"blood pressure {row['BP']}, cholesterol {row['Cholesterol']}, "
        f"combined cardiovascular risk score {int(row['combined_risk_score'])}/4, "
        f"age group {row['age_group']}."
    )


def predict_drug(
    age: float,
    sex: str,
    bp: str,
    cholesterol: str,
    na_to_k: float,
) -> dict[str, Any]:
    """Recommend a drug for a single patient (MODEL 1).

    Parameters
    ----------
    age : float
        Patient age in years. Must be non-negative.
    sex : str
        ``'F'`` or ``'M'`` (case-insensitive).
    bp : str
        Blood pressure level: ``'HIGH'``, ``'NORMAL'`` or ``'LOW'``
        (case-insensitive).
    cholesterol : str
        Cholesterol level: ``'HIGH'`` or ``'NORMAL'`` (case-insensitive).
    na_to_k : float
        Serum sodium-to-potassium ratio. Must be positive.

    Returns
    -------
    dict
        ``prediction`` (str), ``confidence`` (float), ``probabilities``
        (dict of class -> probability, sorted descending), ``explanation``
        (str) and ``model`` (the family that produced it).

    Raises
    ------
    ValueError
        If a categorical value is outside the trained vocabulary or a numeric
        value is missing, non-numeric or out of range.
    FileNotFoundError
        If the model artefact has not been trained yet.
    """
    model = load_model(DRUG_MODEL)
    raw = pd.DataFrame(
        [
            {
                "Age": _validate_number(age, "age", minimum=0.0),
                "Sex": sex,
                "BP": bp,
                "Cholesterol": cholesterol,
                "Na_to_K": _validate_number(na_to_k, "na_to_k", minimum=0.0),
            }
        ]
    )
    # Engineer first, then read the bands back out for the explanation, so the
    # narrative and the model always cite the same derived values.
    features, engineered = _prepare_clinical_frame(raw, model)
    enriched = engineered.iloc[0]

    labels = [str(c) for c in model.pipeline.named_steps["model"].classes_]
    proba = np.asarray(model.pipeline.predict_proba(features), dtype=float)[0]
    prediction = str(model.pipeline.predict(features)[0])
    confidence = float(proba[labels.index(prediction)])

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": _probability_dict(labels, proba),
        "explanation": _clinical_explanation(enriched, prediction, confidence),
        "model": model.metadata.get("model_name"),
    }
    log.info("predict_drug: %s (confidence %.3f)", prediction, confidence)
    return result


def predict_drug_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a whole cohort of patients at once (MODEL 1).

    Parameters
    ----------
    df : pandas.DataFrame
        One row per patient with columns ``Age``, ``Sex``, ``BP``,
        ``Cholesterol`` and ``Na_to_K``. Extra columns (a patient id, say) are
        preserved in the output.

    Returns
    -------
    pandas.DataFrame
        The input columns plus ``prediction``, ``confidence`` and one
        ``proba_<class>`` column per drug class. Row order is preserved.

    Raises
    ------
    ValueError
        If ``df`` is empty, is not a DataFrame, or contains an invalid value.
    FileNotFoundError
        If the model artefact has not been trained yet.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"predict_drug_batch expects a DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("predict_drug_batch received an empty DataFrame.")

    model = load_model(DRUG_MODEL)
    features, _ = _prepare_clinical_frame(df, model)

    labels = [str(c) for c in model.pipeline.named_steps["model"].classes_]
    proba = np.asarray(model.pipeline.predict_proba(features), dtype=float)
    predictions = np.asarray(model.pipeline.predict(features)).astype(str)

    out = df.reset_index(drop=True).copy()
    out["prediction"] = predictions
    out["confidence"] = proba.max(axis=1).round(4)
    for index, label in enumerate(labels):
        out[f"proba_{label}"] = proba[:, index].round(4)

    log.info(
        "predict_drug_batch: scored %d patient(s) | mix %s",
        len(out),
        pd.Series(predictions).value_counts().to_dict(),
    )
    return out


# ---------------------------------------------------------------------------
# MODEL 2 - batch risk
# ---------------------------------------------------------------------------
def _batch_explanation(row: pd.Series, prediction: str, confidence: float) -> str:
    """Build a one-line supply chain rationale for a batch risk verdict."""
    excursion = float(row["temp_excursion_c"])
    humidity = float(row["humidity_excess"])
    drivers: list[str] = []
    if excursion > 0:
        drivers.append(
            f"{excursion:.1f} degC above the labelled storage limit for "
            f"{int(row['storage_duration_days'])} days (thermal load {float(row['thermal_load']):.0f})"
        )
    else:
        drivers.append("storage temperature within the labelled limit")
    if humidity > 0:
        drivers.append(f"{humidity:.1f} points of relative humidity above the moisture threshold")
    drivers.append(
        f"cycle time {float(row['cycle_time_ratio']):.2f}x the network median "
        f"with a {float(row['qa_delay_days']):.1f}-day QA delay"
    )
    drivers.append(f"supplier reliability {float(row['supplier_reliability']):.2f}")
    return (
        f"Predicted {prediction} risk with {confidence:.1%} confidence. Drivers: "
        + "; ".join(drivers)
        + "."
    )


def predict_batch_risk(**features: Any) -> dict[str, Any]:
    """Predict the QA risk tier for a single manufacturing batch (MODEL 2).

    Parameters
    ----------
    **features
        One keyword per configured raw feature (``ml.batch_risk.features``):
        ``storage_temp_c``, ``storage_humidity_pct``, ``storage_duration_days``,
        ``total_cycle_time_days``, ``qa_delay_days``, ``supplier_reliability``,
        ``is_cold_chain``, ``shelf_life_months``, ``drug_code`` and ``region``.

    Returns
    -------
    dict
        ``prediction`` (``'Low'`` / ``'Medium'`` / ``'High'``), ``confidence``,
        ``probabilities`` (sorted descending), ``explanation`` and ``model``.

    Raises
    ------
    ValueError
        If a required feature is missing, an unexpected keyword is supplied, a
        categorical value is outside the trained vocabulary, or a numeric value
        is not a finite number.
    FileNotFoundError
        If the model artefact has not been trained yet.
    """
    model = load_model(BATCH_MODEL)
    allowed = _known_categories(model)
    raw_required = list(get_config().ml.batch_risk.features)

    missing = [name for name in raw_required if name not in features]
    if missing:
        raise ValueError(
            f"predict_batch_risk is missing required feature(s): {missing}. "
            f"Required: {raw_required}."
        )
    unexpected = [name for name in features if name not in raw_required]
    if unexpected:
        raise ValueError(
            f"predict_batch_risk received unknown feature(s): {unexpected}. "
            f"Accepted: {raw_required}."
        )

    record: dict[str, Any] = {}
    for name in raw_required:
        value = features[name]
        if name in allowed:
            record[name] = pp.normalise_categorical(str(value), allowed[name], name)
        else:
            # supplier_reliability is a 0-1 fraction and is_cold_chain a 0/1
            # flag; both, like every other numeric here, are non-negative.
            record[name] = _validate_number(value, name, minimum=0.0)

    engineered = pp.engineer_batch_features(
        pd.DataFrame([record]),
        median_cycle_time=float(
            model.metadata["feature_engineering"]["median_cycle_time_days"]
        ),
    )
    features_frame = engineered[list(model.metadata["features"])]

    labels = [str(c) for c in model.pipeline.named_steps["model"].classes_]
    proba = np.asarray(model.pipeline.predict_proba(features_frame), dtype=float)[0]
    prediction = str(model.pipeline.predict(features_frame)[0])
    confidence = float(proba[labels.index(prediction)])

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": _probability_dict(labels, proba),
        "explanation": _batch_explanation(engineered.iloc[0], prediction, confidence),
        "model": model.metadata.get("model_name"),
    }
    log.info("predict_batch_risk: %s (confidence %.3f)", prediction, confidence)
    return result


def predict_batch_risk_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Score a whole batch register at once (MODEL 2).

    Parameters
    ----------
    df : pandas.DataFrame
        One row per batch, carrying every column in ``ml.batch_risk.features``.
        Extra columns (``batch_id``, for instance) are preserved.

    Returns
    -------
    pandas.DataFrame
        Input columns plus ``prediction``, ``confidence`` and one
        ``proba_<tier>`` column per risk tier.

    Raises
    ------
    ValueError
        If ``df`` is empty or missing a required feature column.
    FileNotFoundError
        If the model artefact has not been trained yet.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"predict_batch_risk_frame expects a DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("predict_batch_risk_frame received an empty DataFrame.")

    model = load_model(BATCH_MODEL)
    raw_required = list(get_config().ml.batch_risk.features)
    missing = set(raw_required) - set(df.columns)
    if missing:
        raise ValueError(f"Batch frame is missing required column(s): {sorted(missing)}")

    prepared = _normalise_frame_categoricals(df, _known_categories(model))
    engineered = pp.engineer_batch_features(
        prepared,
        median_cycle_time=float(
            model.metadata["feature_engineering"]["median_cycle_time_days"]
        ),
    )
    features_frame = engineered[list(model.metadata["features"])]

    labels = [str(c) for c in model.pipeline.named_steps["model"].classes_]
    proba = np.asarray(model.pipeline.predict_proba(features_frame), dtype=float)
    predictions = np.asarray(model.pipeline.predict(features_frame)).astype(str)

    out = df.reset_index(drop=True).copy()
    out["prediction"] = predictions
    out["confidence"] = proba.max(axis=1).round(4)
    for index, label in enumerate(labels):
        out[f"proba_{label}"] = proba[:, index].round(4)

    log.info(
        "predict_batch_risk_frame: scored %d batch(es) | mix %s",
        len(out),
        pd.Series(predictions).value_counts().to_dict(),
    )
    return out


# ---------------------------------------------------------------------------
# MODEL 3 - late delivery risk on the real SCMS dataset
# ---------------------------------------------------------------------------
def score_late_delivery_risk(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score every real shipment with its predicted probability of arriving late.

    Parameters
    ----------
    scms : pandas.DataFrame, optional
        Pre-loaded SCMS frame. Loads the full dataset when omitted.

    Returns
    -------
    pandas.DataFrame
        The input with ``late_risk_probability`` and ``late_risk_decile`` added
        (decile 1 = highest risk).
    """
    from src.data.scms import load_scms

    model = load_model(LATE_DELIVERY_MODEL)
    frame = (load_scms() if scms is None else scms).copy()

    features = list(model.metadata["numeric_features"]) + \
        list(model.metadata["categorical_features"])
    design = frame[features].copy()
    for column in model.metadata["categorical_features"]:
        design[column] = design[column].astype("object").fillna("Unknown")

    labels = list(model.pipeline.classes_)
    late_index = labels.index("Late") if "Late" in labels else 1
    frame["late_risk_probability"] = model.pipeline.predict_proba(design)[:, late_index]

    # qcut on ranks so ties cannot collapse the deciles.
    frame["late_risk_decile"] = pd.qcut(
        frame["late_risk_probability"].rank(method="first", ascending=False),
        10, labels=range(1, 11)).astype(int)

    log.info("Scored %d shipments for late-delivery risk (mean p=%.3f)",
             len(frame), frame["late_risk_probability"].mean())
    return frame


def late_delivery_targeting_curve() -> pd.DataFrame:
    """Gains curve: what share of late deliveries you catch by inspecting the top N%.

    This is the metric that actually matters for this model. Raw accuracy is
    misleading on an 88.5/11.5 split - always predicting "on time" scores 88.5%
    and is useless. What an expeditor can act on is a *ranking*: given capacity
    to monitor a fraction of shipments, how many of the genuinely late ones does
    the model surface?

    Evaluated strictly on the **held-out test split**, reproduced with the same
    seed and test size used at training time. Scoring the full dataset would
    include rows the model was fitted on and inflate every number here.

    Returns
    -------
    pandas.DataFrame
        Columns ``targeted_pct``, ``shipments_reviewed``, ``late_captured``,
        ``capture_rate_pct``, ``precision_pct``, ``lift``. ``lift`` is the
        capture rate divided by the targeted share - how many times better than
        reviewing shipments at random.
    """
    from src.config import get_config
    from src.data.scms import load_scms

    cfg = get_config()
    settings = cfg.scms.late_delivery_model
    model = load_model(LATE_DELIVERY_MODEL)

    frame = load_scms().copy()
    frame = frame[frame["is_late"].notna()].copy()
    target = frame["is_late"].astype(int).map({0: "On Time", 1: "Late"})

    numeric = list(settings.numeric_features)
    categorical = list(settings.categorical_features)
    design = frame[numeric + categorical].copy()
    for column in categorical:
        design[column] = design[column].astype("object").fillna("Unknown")

    # Same split policy as training, so these are exactly the unseen rows.
    _, X_test, _, y_test = pp.split_data(
        design, target, float(settings.test_size), int(cfg.project.random_seed))

    labels = list(model.pipeline.classes_)
    late_index = labels.index("Late") if "Late" in labels else 1
    probability = model.pipeline.predict_proba(X_test)[:, late_index]

    scored = pd.DataFrame({
        "late_risk_probability": probability,
        "is_late": (y_test.to_numpy() == "Late").astype(int),
    }).sort_values("late_risk_probability", ascending=False)

    total = len(scored)
    total_late = int(scored["is_late"].sum())
    cumulative_late = scored["is_late"].cumsum()

    rows: list[dict[str, Any]] = []
    for share in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        cutoff = max(int(round(total * share)), 1)
        captured = int(cumulative_late.iloc[cutoff - 1])
        capture_rate = 100 * captured / total_late if total_late else 0.0
        rows.append({
            "targeted_pct": round(100 * share, 1),
            "shipments_reviewed": cutoff,
            "late_captured": captured,
            "capture_rate_pct": round(capture_rate, 2),
            "precision_pct": round(100 * captured / cutoff, 2),
            "lift": round(capture_rate / (100 * share), 2),
        })
    return pd.DataFrame(rows)


def predict_late_delivery(**features: Any) -> dict[str, Any]:
    """Predict late-delivery risk for a single planned shipment.

    Parameters
    ----------
    **features
        One value per feature in the model's declared feature list. Missing
        categoricals default to ``"Unknown"``; missing numerics are imputed by
        the pipeline.

    Returns
    -------
    dict
        ``prediction``, ``late_probability``, ``risk_band``, ``explanation``
        and ``model``.
    """
    model = load_model(LATE_DELIVERY_MODEL)
    numeric = list(model.metadata["numeric_features"])
    categorical = list(model.metadata["categorical_features"])

    row = {column: features.get(column, np.nan) for column in numeric}
    row.update({column: features.get(column, "Unknown") for column in categorical})
    design = pd.DataFrame([row])

    labels = list(model.pipeline.classes_)
    late_index = labels.index("Late") if "Late" in labels else 1
    probability = float(model.pipeline.predict_proba(design)[0][late_index])

    # Bands come from the targeting curve rather than the default 0.5 cut: the
    # model is used to prioritise attention, not to make a binary call.
    band = ("High" if probability >= 0.35
            else "Elevated" if probability >= 0.20 else "Low")

    # Driver text mirrors what the fitted model and the scorecards actually show:
    # regional-distribution-centre fulfilment is the dominant risk factor (the
    # "SCMS from RDC" channel is the worst-performing supplier at 82.9% on-time),
    # not direct-drop ordering.
    drivers = []
    if str(row.get("fulfil_via")) == "From RDC":
        drivers.append("regional distribution centre fulfilment "
                       "(the weakest channel at 82.9% on-time)")
    if str(row.get("shipment_mode")) in {"Ocean", "Truck"}:
        drivers.append(f"{row['shipment_mode']} transport "
                       "(the two slowest modes on record)")
    if str(row.get("region")) in {"West & Central Africa", "Southern Africa",
                                  "East Africa"}:
        drivers.append(f"{row['region']} destination (below the on-time target)")
    lead = row.get("scheduled_lead_time_days")
    if isinstance(lead, (int, float)) and not pd.isna(lead) and lead > 250:
        drivers.append(f"long planned lead time ({lead:.0f} days)")

    explanation = (
        f"Predicted {probability:.1%} probability of late delivery ({band} risk)."
        + (" Drivers: " + "; ".join(drivers) + "." if drivers else
           " No dominant risk driver in the supplied attributes.")
    )

    return {
        "prediction": "Late" if probability >= 0.5 else "On Time",
        "late_probability": round(probability, 4),
        "risk_band": band,
        "explanation": explanation,
        "model": model.metadata.get("model_name", "unknown"),
    }


__all__ = [
    "DRUG_MODEL",
    "BATCH_MODEL",
    "LATE_DELIVERY_MODEL",
    "KNOWN_MODELS",
    "LoadedModel",
    "load_model",
    "model_summary",
    "predict_drug",
    "predict_drug_batch",
    "predict_batch_risk",
    "predict_batch_risk_frame",
    "predict_late_delivery",
    "score_late_delivery_risk",
    "late_delivery_targeting_curve",
]
