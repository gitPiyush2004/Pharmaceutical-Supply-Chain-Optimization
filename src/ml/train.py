"""
Model training, tuning and evaluation for the PharmaChain ML package.

Business purpose
----------------
Two independent supervised problems share one training contract:

* **MODEL 1 - ``drug_classification``** - recommend one of five formulary drugs
  from a patient's age, sex, blood pressure, cholesterol level and serum
  sodium/potassium ratio (real Kaggle drug200 data).
* **MODEL 2 - ``late_delivery``** - predict whether a shipment will arrive after
  its scheduled date, trained on the **real** USAID SCMS delivery history of
  10,324 actual pharmaceutical shipments.

Both are trained the same way: clean, engineer, split (stratified), tune
three candidate families with ``GridSearchCV`` over the grids declared in
``config.yaml``, pick the winner by cross-validated score, then report a full
held-out evaluation. Nothing is tuned on the test fold, and using one metric
(macro F1) across both keeps their numbers directly comparable.

What this module returns
------------------------
Data structures only - DataFrames, dicts and numpy arrays. Every curve
(ROC, precision-recall), the confusion matrix and the feature importance table
are returned as plain objects for the Streamlit layer to render. This module
imports no plotting library and no Streamlit.

Reproducibility
---------------
``project.random_seed`` is threaded into every estimator and every split, and is
recorded in the saved metadata alongside the training timestamp, row count,
resolved hyperparameters and class labels, so any published number can be traced
back to the exact configuration that produced it.

Example
-------
>>> from src.ml.train import train_all
>>> results = train_all(tune=True)
>>> results["drug_classification"]["best_model_name"]
'random_forest'
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import ensure_directories, get_config, resolve_path
from src.data.loader import load_clinical
from src.logger import get_logger
from src.ml import preprocess as pp

log = get_logger(__name__)

# Pipeline step names. Referenced when building GridSearchCV parameter keys, so
# they are defined once rather than spelled out in string literals.
_PREP_STEP = "preprocess"
_MODEL_STEP = "model"

# Model families the factory knows about. The config `models` list selects from
# these; an unknown name is a configuration error, not a silent no-op.
_SUPPORTED_MODELS: tuple[str, ...] = ("decision_tree", "random_forest", "xgboost")

# Only the clinical block declares candidate families in config, so the
# late-delivery model reuses that candidate set - the point of the comparison is
# the same three families in both problems.
_DEFAULT_MODELS: tuple[str, ...] = _SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# Target encoding wrapper
# ---------------------------------------------------------------------------
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier so it trains on integer targets but speaks strings.

    XGBoost refuses non-numeric targets. Rather than pushing that constraint out
    to every caller (and risking a saved pipeline that returns ``3`` instead of
    ``drugX``), the label round-trip is folded into the estimator itself: ``fit``
    encodes, ``predict`` decodes. The same wrapper is applied to every candidate
    family so the pipeline shape - and therefore the tuning parameter keys - is
    identical regardless of which model wins.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator, optional
        The wrapped classifier. Cloned before fitting, leaving the passed
        instance untouched.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Original (string) class labels, in ``predict_proba`` column order.
    estimator_ : sklearn.base.BaseEstimator
        The fitted inner classifier.
    """

    def __init__(self, estimator: BaseEstimator | None = None) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> "LabelEncodedClassifier":
        """Encode ``y``, fit the inner estimator and remember the label mapping.

        Parameters
        ----------
        X : array-like
            Transformed feature matrix.
        y : array-like
            Target labels of any dtype.

        Returns
        -------
        LabelEncodedClassifier
            ``self``, fitted.

        Raises
        ------
        ValueError
            If no inner estimator was supplied.
        """
        if self.estimator is None:
            raise ValueError("LabelEncodedClassifier requires an `estimator`.")
        self.label_encoder_ = LabelEncoder().fit(np.asarray(y).astype(str))
        self.classes_ = self.label_encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.label_encoder_.transform(np.asarray(y).astype(str)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict original-label classes.

        Parameters
        ----------
        X : array-like
            Transformed feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted labels as the original strings.
        """
        return self.label_encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities, column-aligned with :attr:`classes_`.

        Parameters
        ----------
        X : array-like
            Transformed feature matrix.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_classes)``.
        """
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Impurity/gain importances of the wrapped estimator.

        Returns
        -------
        numpy.ndarray
            One importance per transformed feature.

        Raises
        ------
        AttributeError
            If the wrapped estimator does not expose importances.
        """
        return self.estimator_.feature_importances_


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_model(name: str, seed: int) -> BaseEstimator:
    """Instantiate a candidate classifier with the platform seed applied.

    Parameters
    ----------
    name : str
        One of ``'decision_tree'``, ``'random_forest'``, ``'xgboost'``.
    seed : int
        Random seed from ``project.random_seed``.

    Returns
    -------
    sklearn.base.BaseEstimator
        An unfitted estimator. Hyperparameters are left at library defaults -
        they are supplied by the config-declared grid during tuning.

    Raises
    ------
    ValueError
        If ``name`` is not a supported model family.
    """
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "random_forest":
        # n_jobs=1 inside the estimator: GridSearchCV already parallelises across
        # folds, and nesting the two thread pools oversubscribes the CPU.
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "xgboost":
        return XGBClassifier(
            random_state=seed,
            n_jobs=1,
            eval_metric="mlogloss",
            tree_method="hist",
        )
    raise ValueError(
        f"Unknown model '{name}'. Supported: {sorted(_SUPPORTED_MODELS)}"
    )


def _build_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_name: str,
    seed: int,
    scale: bool = True,
) -> Pipeline:
    """Compose preprocessing + label-encoded estimator into one Pipeline."""
    return Pipeline(
        steps=[
            (_PREP_STEP, pp.build_preprocessor(numeric_features, categorical_features, scale=scale)),
            (_MODEL_STEP, LabelEncodedClassifier(estimator=get_model(model_name, seed))),
        ]
    )


def _prefix_grid(grid: dict[str, Any]) -> dict[str, list[Any]]:
    """Rewrite a config param grid into Pipeline-addressable parameter keys."""
    return {
        f"{_MODEL_STEP}__estimator__{key}": list(values) for key, values in grid.items()
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Map post-OneHot importances back to readable feature names."""
    preprocessor = pipeline.named_steps[_PREP_STEP]
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    # Strip the ColumnTransformer's "numeric__" / "categorical__" prefixes: the
    # dashboard shows these to analysts, not to sklearn.
    names = [n.split("__", 1)[1] if "__" in n else n for n in names]

    model = pipeline.named_steps[_MODEL_STEP]
    try:
        importances = np.asarray(model.feature_importances_, dtype=float)
    except AttributeError:  # pragma: no cover - all current families expose them
        importances = np.zeros(len(names), dtype=float)

    frame = pd.DataFrame({"feature": names, "importance": importances})
    return frame.sort_values("importance", ascending=False, ignore_index=True)


def _curve_metrics(
    y_test: np.ndarray, proba: np.ndarray, classes: Sequence[str]
) -> tuple[float, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compute one-vs-rest ROC AUC plus per-class ROC and PR curves."""
    binarised = label_binarize(y_test, classes=list(classes))
    # label_binarize collapses a binary problem to a single column; expand it so
    # the per-class loop below is uniform across 2-class and k-class problems.
    if binarised.shape[1] == 1:
        binarised = np.hstack([1 - binarised, binarised])

    try:
        if len(classes) == 2:
            # sklearn rejects multi_class="ovr" for a 2-class problem. Score the
            # positive class probability directly; without this branch a binary
            # model silently reports ROC AUC as NaN.
            roc_auc_ovr = float(roc_auc_score(binarised[:, 1], proba[:, 1]))
        else:
            roc_auc_ovr = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro",
                              labels=list(classes))
            )
    except ValueError:  # a class absent from the test fold makes OvR undefined
        roc_auc_ovr = float("nan")
        log.warning("ROC AUC (OvR) undefined - a class is missing from the test fold.")

    per_class_roc: dict[str, dict[str, Any]] = {}
    pr_curves: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(classes):
        truth = binarised[:, index]
        score = proba[:, index]
        if truth.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(truth, score)
        per_class_roc[str(label)] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc(fpr, tpr)),
        }
        precision, recall, _ = precision_recall_curve(truth, score)
        pr_curves[str(label)] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "ap": float(average_precision_score(truth, score)),
        }
    return roc_auc_ovr, per_class_roc, pr_curves


def _evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Score a fitted pipeline on the held-out fold."""
    y_true = np.asarray(y_test).astype(str)
    y_pred = np.asarray(pipeline.predict(X_test)).astype(str)
    proba = np.asarray(pipeline.predict_proba(X_test), dtype=float)

    # zero_division=0 keeps the report finite when a rare class is never
    # predicted; the confusion matrix still exposes that failure honestly.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(classes)),
        index=pd.Index([f"actual: {c}" for c in classes], name="actual"),
        columns=pd.Index([f"predicted: {c}" for c in classes], name="predicted"),
    )
    report = pd.DataFrame(
        classification_report(
            y_true, y_pred, labels=list(classes), output_dict=True, zero_division=0
        )
    ).transpose()

    roc_auc_ovr, per_class_roc, pr_curves = _curve_metrics(y_true, proba, classes)
    return {
        "test_metrics": metrics,
        "confusion_matrix": matrix,
        "classification_report": report,
        "roc_auc_ovr": roc_auc_ovr,
        "per_class_roc": per_class_roc,
        "pr_curves": pr_curves,
    }


# ---------------------------------------------------------------------------
# Shared training engine
# ---------------------------------------------------------------------------
def _train_classifier(
    *,
    task_name: str,
    frame: pd.DataFrame,
    target: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    settings: Any,
    param_grids: dict[str, Any],
    model_names: Sequence[str],
    tune: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the shared train / tune / compare / evaluate loop for one task.

    Both public trainers delegate here so the two models are guaranteed to be
    evaluated under identical rules - same split policy, same CV object, same
    metric definitions - which is what makes their numbers comparable in the
    dashboard.
    """
    cfg = get_config()
    seed = int(cfg.project.random_seed)
    cv_folds = int(settings.cv_folds)
    scoring = str(settings.scoring)
    test_size = float(settings.test_size)

    features = list(numeric_features) + list(categorical_features)
    usable = frame.dropna(subset=[target])
    if len(usable) < len(frame):
        log.warning("%s: dropped %d row(s) with a missing target.", task_name, len(frame) - len(usable))

    X = usable[features]
    y = usable[target].astype(str)
    classes = sorted(y.unique())
    log.info(
        "%s: %d rows, %d features, %d classes %s | class mix %s",
        task_name, len(X), len(features), len(classes), classes, y.value_counts().to_dict(),
    )

    X_train, X_test, y_train, y_test = pp.split_data(
        X, y, test_size=test_size, seed=seed, stratify=True
    )

    # Shuffled stratified folds with a fixed seed: the source frames are stored
    # in generation order, which is not random with respect to the target.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    best_params: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        started = time.perf_counter()
        pipeline = _build_pipeline(numeric_features, categorical_features, model_name, seed)
        grid = _prefix_grid(param_grids.get(model_name, {})) if tune else {}

        log.info(
            "%s | %s: %s over %d candidate combination(s), %d-fold CV, scoring=%s",
            task_name, model_name, "tuning" if grid else "fitting defaults",
            int(np.prod([len(v) for v in grid.values()])) if grid else 1, cv_folds, scoring,
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - started

        best = search.best_estimator_
        fitted[model_name] = best
        # Strip the pipeline plumbing from the parameter names before they are
        # written to metadata or shown in the dashboard.
        best_params[model_name] = {
            key.replace(f"{_MODEL_STEP}__estimator__", ""): value
            for key, value in search.best_params_.items()
        }

        index = int(search.best_index_)
        cv_mean = float(search.cv_results_["mean_test_score"][index])
        cv_std = float(search.cv_results_["std_test_score"][index])

        evaluation = _evaluate(best, X_test, y_test, classes)
        rows.append(
            {
                "model": model_name,
                "cv_mean": round(cv_mean, 4),
                "cv_std": round(cv_std, 4),
                "test_accuracy": round(evaluation["test_metrics"]["accuracy"], 4),
                "test_f1_macro": round(evaluation["test_metrics"]["f1_macro"], 4),
                "test_precision": round(evaluation["test_metrics"]["precision_macro"], 4),
                "test_recall": round(evaluation["test_metrics"]["recall_macro"], 4),
                "fit_seconds": round(elapsed, 2),
            }
        )
        log.info(
            "%s | %s: cv=%.4f (+/-%.4f) test_acc=%.4f test_f1=%.4f in %.1fs | best=%s",
            task_name, model_name, cv_mean, cv_std,
            evaluation["test_metrics"]["accuracy"], evaluation["test_metrics"]["f1_macro"],
            elapsed, best_params[model_name],
        )

    comparison = pd.DataFrame(rows).sort_values("cv_mean", ascending=False, ignore_index=True)

    # Selection is on cross-validated score only. The test fold is never
    # consulted for model choice - it exists solely to estimate generalisation.
    best_model_name = str(comparison.loc[0, "model"])
    best_pipeline = fitted[best_model_name]
    evaluation = _evaluate(best_pipeline, X_test, y_test, classes)

    log.info(
        "%s: selected '%s' (cv %s=%.4f, test accuracy=%.4f, macro F1=%.4f)",
        task_name, best_model_name, scoring, float(comparison.loc[0, "cv_mean"]),
        evaluation["test_metrics"]["accuracy"], evaluation["test_metrics"]["f1_macro"],
    )

    result: dict[str, Any] = {
        "task": task_name,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "comparison": comparison,
        "feature_importance": _feature_importance(best_pipeline),
        "label_classes": classes,
        "best_params": best_params[best_model_name],
        "all_best_params": best_params,
        "cv_folds": cv_folds,
        "scoring": scoring,
        "test_size": test_size,
        "seed": seed,
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": features,
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "target": target,
        "extra": dict(extra_metadata or {}),
    }
    result.update(evaluation)
    return result


# ---------------------------------------------------------------------------
# MODEL 1 - drug classification
# ---------------------------------------------------------------------------
def train_drug_classifier(tune: bool = True) -> dict[str, Any]:
    """Train, tune and evaluate the patient-to-drug recommender (MODEL 1).

    Loads drug200, normalises the categorical text, derives the interpretable
    clinical features, then tunes a decision tree, a random forest and an
    XGBoost classifier over the grids in ``ml.drug_classification.param_grid``
    and returns the best by cross-validated ``ml.drug_classification.scoring``.

    Parameters
    ----------
    tune : bool, default True
        Run the full grid search. ``False`` fits library defaults instead -
        useful for a fast smoke test, not for a published result.

    Returns
    -------
    dict
        Keys: ``best_model_name``, ``best_pipeline``, ``comparison``,
        ``test_metrics``, ``confusion_matrix``, ``classification_report``,
        ``roc_auc_ovr``, ``per_class_roc``, ``pr_curves``,
        ``feature_importance``, ``label_classes``, ``best_params``,
        ``cv_folds``, plus provenance fields (``seed``, ``n_rows``, ...).

    Raises
    ------
    ValueError
        If the configured model list contains an unsupported family.
    """
    cfg = get_config()
    settings = cfg.ml.drug_classification
    started = time.perf_counter()
    log.info("=== MODEL 1 'drug_classification': training started ===")

    frame = pp.engineer_clinical_features(pp.clean_clinical(load_clinical()))
    numeric_features, categorical_features = pp.clinical_feature_columns()
    model_names = list(settings.get("models", _DEFAULT_MODELS))

    result = _train_classifier(
        task_name="drug_classification",
        frame=frame,
        target=str(settings.target),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        settings=settings,
        param_grids=dict(settings.get("param_grid", {})),
        model_names=model_names,
        tune=tune,
    )
    log.info(
        "=== MODEL 1 'drug_classification': finished in %.1fs ===",
        time.perf_counter() - started,
    )
    return result


# ---------------------------------------------------------------------------
# MODEL 2 - late delivery on the real SCMS data
# ---------------------------------------------------------------------------
def train_late_delivery_classifier(tune: bool = True) -> dict[str, Any]:
    """Predict late delivery on the USAID SCMS dataset (MODEL 2 of 2).

    10,324 real shipments of HIV and malaria commodities to 43 countries between
    2006 and 2015. The target is whether a line item arrived after its scheduled
    delivery date.

    Leakage control is the whole game here. Every feature must be knowable at the
    moment the order is placed, so the model is a genuine forward-looking screen:

    * **Included** - quantity, line value, unit price, pack size, weight, freight
      cost, the *planned* quote-to-scheduled lead time, transport mode, commodity
      group, destination region, fulfilment route, INCO term and managing office.
    * **Deliberately excluded** - ``date_delivered``, ``delivery_delay_days``,
      ``vendor_lead_time_days``, ``total_lead_time_days`` and
      ``recording_lag_days``. Each is computed from the delivery date and would
      hand the model its own answer.

    Vendor identity is also excluded. With 73 vendors, several appearing only a
    handful of times, one-hot encoding vendor invites the model to memorise
    individual suppliers rather than learn transferable structure - and it would
    not generalise to a vendor the model has never seen.

    Parameters
    ----------
    tune : bool, default True
        Run the grid search shared by every task in the platform.

    Returns
    -------
    dict
        Same contract as the other trainers.

    Raises
    ------
    ValueError
        If the SCMS table is missing a configured feature column.
    """
    from src.data.scms import load_scms  # local import keeps ML free of a hard dep

    cfg = get_config()
    settings = cfg.scms.late_delivery_model
    started = time.perf_counter()
    log.info("=== MODEL 2 'late_delivery' (real SCMS data): training started ===")

    frame = load_scms().copy()

    numeric_features = list(settings.numeric_features)
    categorical_features = list(settings.categorical_features)
    missing = set(numeric_features + categorical_features) - set(frame.columns)
    if missing:
        raise ValueError(
            f"SCMS table is missing configured feature(s): {sorted(missing)}")

    # The target is undefined where a delivery date could not be resolved.
    target = str(settings.target)
    frame = frame[frame[target].notna()].copy()
    frame[target] = frame[target].astype(int).map({0: "On Time", 1: "Late"})

    # Categoricals arrive as pandas "string" dtype; the one-hot encoder expects
    # object, and unresolved values become an explicit category rather than NaN
    # so "we did not record the mode" stays a distinguishable signal.
    for column in categorical_features:
        frame[column] = frame[column].astype("object").fillna("Unknown")

    grid_source = cfg.ml.drug_classification
    model_names = list(settings.get("models", grid_source.get("models", _DEFAULT_MODELS)))
    param_grids = dict(settings.get("param_grid", grid_source.get("param_grid", {})))

    result = _train_classifier(
        task_name="late_delivery",
        frame=frame,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        settings=settings,
        param_grids=param_grids,
        model_names=model_names,
        tune=tune,
        extra_metadata={
            "data_source": "USAID SCMS Delivery History (real operational data)",
            "records": int(len(frame)),
            "excluded_for_leakage": [
                "date_delivered", "delivery_delay_days", "vendor_lead_time_days",
                "total_lead_time_days", "recording_lag_days",
            ],
            "excluded_high_cardinality": ["vendor", "manufacturing_site", "country"],
            "class_balance": {
                str(k): int(v) for k, v in frame[target].value_counts().items()},
        },
    )
    log.info("=== MODEL 2 'late_delivery': finished in %.1fs ===",
             time.perf_counter() - started)
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Coerce numpy/pandas scalars into something ``json.dump`` accepts."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_artifacts(result: dict[str, Any], name: str) -> Path:
    """Serialise a trained pipeline and its metadata to ``paths.models``.

    Two files are written: ``{name}_model.joblib`` (the fitted sklearn Pipeline,
    preprocessing included, so serving needs no separate transformer) and
    ``{name}_metadata.json`` (everything needed to interpret and audit it).

    Parameters
    ----------
    result : dict
        A training result from :func:`train_drug_classifier` or
        :func:`train_late_delivery_classifier`.
    name : str
        Artefact stem, e.g. ``'drug_classification'``.

    Returns
    -------
    pathlib.Path
        Path to the written ``.joblib`` file.

    Raises
    ------
    KeyError
        If ``result`` is missing the fitted pipeline.
    """
    if "best_pipeline" not in result:
        raise KeyError("result has no 'best_pipeline' - was training completed?")

    ensure_directories()
    models_dir = resolve_path(get_config().paths.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}_model.joblib"
    joblib.dump(result["best_pipeline"], model_path)

    metadata = {
        "artifact_name": name,
        "task": result.get("task", name),
        "model_name": result["best_model_name"],
        "params": _json_safe(result["best_params"]),
        "all_best_params": _json_safe(result.get("all_best_params", {})),
        "metrics": _json_safe(result["test_metrics"]),
        "roc_auc_ovr": _json_safe(result.get("roc_auc_ovr")),
        "cv_folds": result["cv_folds"],
        "cv_scoring": result.get("scoring"),
        "cv_best_score": float(result["comparison"].loc[0, "cv_mean"]),
        "comparison": _json_safe(result["comparison"].to_dict(orient="records")),
        "features": list(result["features"]),
        "numeric_features": list(result["numeric_features"]),
        "categorical_features": list(result["categorical_features"]),
        "target": result["target"],
        "class_labels": [str(c) for c in result["label_classes"]],
        "top_features": _json_safe(
            result["feature_importance"].head(10).to_dict(orient="records")
        ),
        # Full importance table, confusion matrix and curve data are persisted so
        # the dashboard can render the complete evaluation without retraining.
        "feature_importance": _json_safe(
            result["feature_importance"].to_dict(orient="records")
        ),
        "confusion_matrix": {
            "labels": [str(c) for c in result["confusion_matrix"].columns],
            "index": [str(i) for i in result["confusion_matrix"].index],
            "values": _json_safe(result["confusion_matrix"].values.tolist()),
        },
        "classification_report": _json_safe(
            result["classification_report"].to_dict(orient="index")
            if hasattr(result.get("classification_report"), "to_dict")
            else result.get("classification_report", {})
        ),
        "per_class_roc": _json_safe(result.get("per_class_roc", {})),
        "pr_curves": _json_safe(result.get("pr_curves", {})),
        "row_count": int(result["n_rows"]),
        "train_rows": int(result["n_train"]),
        "test_rows": int(result["n_test"]),
        "test_size": result["test_size"],
        "random_seed": int(result["seed"]),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform_version": get_config().project.version,
        "feature_engineering": _json_safe(result.get("extra", {})),
    }
    metadata_path = models_dir / f"{name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("Saved %s artefacts: %s and %s", name, model_path.name, metadata_path.name)
    return model_path


def train_all(tune: bool = True) -> dict[str, dict[str, Any]]:
    """Train, evaluate and persist both platform models.

    Parameters
    ----------
    tune : bool, default True
        Passed through to every trainer.

    Returns
    -------
    dict
        ``{'drug_classification': ..., 'late_delivery': ...}``.
    """
    started = time.perf_counter()
    log.info("train_all: starting full training run (tune=%s)", tune)

    trainers: dict[str, Callable[[bool], dict[str, Any]]] = {
        "drug_classification": train_drug_classifier,
        "late_delivery": train_late_delivery_classifier,
    }
    results: dict[str, dict[str, Any]] = {}
    for name, trainer in trainers.items():
        result = trainer(tune)
        save_artifacts(result, name)
        results[name] = result

    log.info(
        "train_all: complete in %.1fs | %s",
        time.perf_counter() - started,
        {k: v["best_model_name"] for k, v in results.items()},
    )
    return results


__all__ = [
    "LabelEncodedClassifier",
    "get_model",
    "train_drug_classifier",
    "train_late_delivery_classifier",
    "save_artifacts",
    "train_all",
]
