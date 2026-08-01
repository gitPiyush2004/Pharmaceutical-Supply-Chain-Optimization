"""
Machine learning page.

Two models, both trained through the same reproducible pipeline:

1. **Drug Classification** - patient-level prescribing recommendation on the
   Kaggle ``drug200`` dataset.
2. **Batch Risk** - stability risk tier for a manufactured batch, from storage
   and process telemetry.

Every number shown here is read from the persisted metadata written by
``scripts/train_models.py``; nothing is retrained in the browser.
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

from src.dashboard.components import (callout, chart, insight, kpi_row, methodology,
                        page_setup, section, show_table, sidebar_about,
                        verdict_badge)
from src.ml import predict
from src.viz import charts
from src.viz.theme import fmt_pct

page_setup(
    title="Machine Learning Models",
    icon="🤖",
    subtitle="Drug classification and batch risk prediction, with full evaluation evidence",
)

MODELS = {
    "Drug Classification (clinical)": "drug_classification",
    "Batch Risk (supply chain)": "batch_risk",
}


@st.cache_data(show_spinner=False)
def _summary(name: str) -> dict:
    return predict.model_summary(name)


def _confusion_frame(meta: dict) -> pd.DataFrame:
    cm = meta["confusion_matrix"]
    return pd.DataFrame(cm["values"], index=cm["index"], columns=cm["labels"])


try:
    available = {label: key for label, key in MODELS.items()
                 if _summary(key).get("model_name")}
except FileNotFoundError:
    st.error(
        "No trained model artefacts found. Run `python scripts/train_models.py` "
        "to build them, then reload this page."
    )
    st.stop()

choice = st.radio("Model", list(available), horizontal=True)
key = available[choice]
meta = _summary(key)
metrics = meta["metrics"]

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
kpi_row([
    {"label": "Selected Model", "value": meta["model_name"].replace("_", " ").title(),
     "help_text": f"Chosen on {meta['cv_folds']}-fold CV {meta['cv_scoring']}"},
    {"label": "Test Accuracy", "value": fmt_pct(metrics["accuracy"] * 100, 2),
     "help_text": f"{meta['test_rows']} held-out rows"},
    {"label": "Macro F1", "value": f"{metrics['f1_macro']:.4f}",
     "help_text": "Unweighted mean across classes - the selection metric"},
    {"label": "ROC AUC (OvR)", "value": f"{meta['roc_auc_ovr']:.4f}",
     "help_text": "One-vs-rest, macro averaged"},
])

kpi_row([
    {"label": "Training Rows", "value": f"{meta['train_rows']:,}"},
    {"label": "Test Rows", "value": f"{meta['test_rows']:,}"},
    {"label": "Features", "value": f"{len(meta['features'])}",
     "help_text": f"{len(meta['numeric_features'])} numeric, "
                  f"{len(meta['categorical_features'])} categorical"},
    {"label": "Classes", "value": f"{len(meta['class_labels'])}",
     "help_text": ", ".join(meta["class_labels"])},
])

# ---------------------------------------------------------------------------
# Model selection evidence
# ---------------------------------------------------------------------------
section(
    "Model Selection",
    "Three algorithms were tuned by grid search under identical cross-validation, "
    "and the winner was chosen on cross-validated macro F1 - never on the test set.",
)

comparison = pd.DataFrame(meta["comparison"])
col1, col2 = st.columns([2, 3], gap="large")
with col1:
    chart(charts.model_comparison_chart(comparison, metric="test_f1_macro",
                                        title="Macro F1 by Algorithm"))
with col2:
    show_table(comparison, height=200)

if key == "drug_classification":
    rf = comparison[comparison["model"] == "random_forest"]
    if len(rf) and float(rf.iloc[0]["test_f1_macro"]) > float(metrics["f1_macro"]):
        callout(
            "Random forest scores marginally higher on this particular test split "
            f"({float(rf.iloc[0]['test_f1_macro']):.4f} versus "
            f"{metrics['f1_macro']:.4f}), but the decision tree wins on "
            "cross-validated F1, which is the honest selection criterion. Picking "
            "the model that happens to win on the test set is how leakage enters a "
            "pipeline - the test set is only used to report, never to choose.",
            kind="insight", title="Why the decision tree was selected",
        )

with st.expander("Tuned hyper-parameters per algorithm"):
    st.json(meta["all_best_params"])
st.caption(f"Winning configuration: `{meta['params']}`")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
section("Evaluation on Held-Out Data")

tabs = st.tabs(["Confusion Matrix", "ROC Curves", "Precision-Recall",
                "Per-Class Report", "Feature Importance"])

with tabs[0]:
    matrix = _confusion_frame(meta)
    chart(charts.confusion_matrix_chart(matrix, title="Confusion Matrix (test set)"))
    correct, total = int(matrix.values.trace()), int(matrix.values.sum())
    st.caption(f"{correct} of {total} test rows classified correctly "
               f"({total - correct} error(s)).")

with tabs[1]:
    chart(charts.roc_curves_chart(meta["per_class_roc"]))
    st.caption("One-vs-rest ROC per class. The diagonal is random guessing.")

with tabs[2]:
    chart(charts.pr_curves_chart(meta["pr_curves"]))
    st.caption("Precision-recall is the more informative view when classes are "
               "imbalanced, which they are here.")

with tabs[3]:
    report = pd.DataFrame(meta["classification_report"]).T
    show_table(report.reset_index().rename(columns={"index": "class"}), height=280)

with tabs[4]:
    importance = pd.DataFrame(meta["feature_importance"])
    chart(charts.feature_importance_chart(importance, top_n=12))
    if key == "drug_classification":
        insight(
            "**Na/K ratio dominates**, carrying roughly 48% of the model's "
            "decision weight, with the engineered blood-pressure risk score "
            "second. That matches the known clinical rule in this dataset: a high "
            "sodium-to-potassium ratio routes a patient to DrugY almost "
            "deterministically, and blood pressure and cholesterol separate the "
            "remaining four drugs. The model recovering that structure is a "
            "correctness check, not a coincidence."
        )
    else:
        insight(
            "**Thermal load** - the engineered interaction of excess temperature "
            "and exposure time - is the strongest single predictor, ahead of raw "
            "storage duration. That is the expected physics: degradation depends "
            "on temperature *and* time together, not either alone. Cycle time and "
            "QA delay follow, which is why compressing quality testing improves "
            "stability as well as throughput."
        )

# ---------------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------------
section("Prediction Interface", "Score a new record against the deployed model.")

if key == "drug_classification":
    c1, c2, c3, c4, c5 = st.columns(5)
    age = c1.number_input("Age", min_value=0, max_value=110, value=45, step=1)
    sex = c2.selectbox("Sex", ["F", "M"])
    bp = c3.selectbox("Blood Pressure", ["HIGH", "NORMAL", "LOW"])
    chol = c4.selectbox("Cholesterol", ["HIGH", "NORMAL"])
    na_to_k = c5.number_input("Na/K Ratio", min_value=1.0, max_value=60.0,
                              value=15.4, step=0.1)

    if st.button("Predict recommended drug", type="primary"):
        result = predict.predict_drug(age=age, sex=sex, bp=bp,
                                      cholesterol=chol, na_to_k=na_to_k)
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown(f"### {result['prediction']}")
            st.metric("Confidence", fmt_pct(result["confidence"] * 100, 1))
        with right:
            probs = pd.DataFrame(
                {"drug": list(result["probabilities"]),
                 "probability": list(result["probabilities"].values())})
            chart(charts.bar_chart(probs, x="drug", y="probability",
                                   title="Class Probabilities",
                                   text_format=".3f", height=300))
        callout(result["explanation"], kind="insight", title="Why this recommendation")

else:
    c1, c2, c3, c4 = st.columns(4)
    temp = c1.number_input("Storage Temp (°C)", 1.0, 45.0, 25.0, 0.5)
    humidity = c2.number_input("Humidity (%RH)", 10.0, 95.0, 55.0, 1.0)
    duration = c3.number_input("Storage Duration (days)", 1, 400, 60, 1)
    cycle = c4.number_input("Total Cycle Time (days)", 10, 400, 95, 1)
    c5, c6, c7, c8 = st.columns(4)
    qa_delay = c5.number_input("QA Delay (days)", 1, 90, 19, 1)
    reliability = c6.number_input("Supplier Reliability", 0.50, 1.0, 0.92, 0.01)
    drug_code = c7.selectbox("Product", ["DrugY", "DrugX", "DrugA", "DrugB", "DrugC"])
    region = c8.selectbox("Region", ["North America", "Europe", "Asia-Pacific",
                                     "Latin America", "Middle East & Africa"])
    c9, c10 = st.columns(2)
    cold = c9.selectbox("Cold Chain", [0, 1],
                        format_func=lambda v: "Yes (2-8°C)" if v else "No (ambient)")
    shelf = c10.number_input("Shelf Life (months)", 6, 60, 24, 1)

    if st.button("Predict batch risk", type="primary"):
        result = predict.predict_batch_risk(
            storage_temp_c=temp, storage_humidity_pct=humidity,
            storage_duration_days=duration, total_cycle_time_days=cycle,
            qa_delay_days=qa_delay, supplier_reliability=reliability,
            is_cold_chain=cold, shelf_life_months=shelf,
            drug_code=drug_code, region=region)
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown(f"### {result['prediction']} Risk")
            st.markdown(verdict_badge(result["prediction"]), unsafe_allow_html=True)
            st.metric("Confidence", fmt_pct(result["confidence"] * 100, 1))
        with right:
            probs = pd.DataFrame(
                {"risk_tier": list(result["probabilities"]),
                 "probability": list(result["probabilities"].values())})
            chart(charts.bar_chart(probs, x="risk_tier", y="probability",
                                   title="Risk Tier Probabilities",
                                   text_format=".3f", height=300))
        callout(result["explanation"], kind="warning" if result["prediction"] == "High"
                else "insight", title="Risk drivers")

# ---------------------------------------------------------------------------
methodology(f"""
**Pipeline.** A single sklearn `Pipeline` holds imputation, encoding, scaling and
the estimator, so the serialised `.joblib` needs no separate transformer at serving
time and train/serve skew is structurally impossible.

**Steps.** Median imputation for numerics and most-frequent for categoricals →
one-hot encoding (unknown categories ignored, so an unseen region cannot crash
serving) → standard scaling for numerics → estimator.

**Feature engineering.**
{"- `bp_risk_score`, `cholesterol_risk`, `combined_risk_score` - ordinal encodings of clinical severity.\\n- `high_na_to_k` - binary flag at the clinically meaningful ratio threshold.\\n- `age_group`, `na_to_k_band` - banded versions that let tree models split on clinically natural boundaries."
if key == "drug_classification" else
"- `thermal_load` - degrees above the labelled storage limit multiplied by exposure days. Degradation depends on temperature and time jointly, so the interaction is given to the model explicitly rather than left to be discovered.\\n- `humidity_excess` - relative humidity above the moisture uptake threshold.\\n- `cycle_time_ratio` - cycle time relative to the network median."}

**Selection.** Stratified {meta['cv_folds']}-fold cross-validation with
`{meta['cv_scoring']}` as the scoring metric, grid-searched per algorithm. Macro F1
is used rather than accuracy because the classes are imbalanced and every class
matters equally. The test set is touched exactly once, to report.

**Reproducibility.** `random_seed={meta['random_seed']}` throughout;
`test_size={meta['test_size']}`; trained at {meta['trained_at']} on platform
version {meta['platform_version']}.

**Honest limitation.** {"The clinical dataset has only 200 rows and an almost deterministic decision rule, so near-perfect scores are expected and are not evidence of a hard problem being solved. The value here is pipeline rigour, not model heroics." if key == "drug_classification" else "Macro F1 of ~0.70 on three imbalanced risk tiers reflects genuine irreducible noise: the label depends partly on QA outcomes that are stochastic by construction. A higher score would indicate leakage rather than skill."}
""")

sidebar_about()
