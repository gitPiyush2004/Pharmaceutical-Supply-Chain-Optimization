"""
Google Colab page.

Links to the reproducible drug recommendation notebook and documents what it
covers, so a reviewer can re-run the whole pipeline in a browser with no local
setup.

The notebook is deliberately scoped to the ``drug200`` dataset alone - it loads,
imports and references nothing else. That is why the run instructions below are
split in two: the notebook needs only a clone, while the dashboard needs the
external datasets and the trained models.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import pandas as pd
import streamlit as st

from src.dashboard.components import (callout, kpi_row, methodology, page_setup,
                                      section, show_table, sidebar_about)
from src.config import PROJECT_ROOT, get_config

cfg = get_config()
page_setup(
    title="Google Colab Notebook",
    icon="📓",
    subtitle="Drug recommendation from patient clinical indicators - the complete "
             "pipeline, reproducible in a browser",
)

GITHUB_USER = "gitPiyush2004"
GITHUB_REPO = "Pharmaceutical-Supply-Chain-Optimization"
NOTEBOOK = "notebooks/drug_classification_pipeline.ipynb"

COLAB_URL = (f"https://colab.research.google.com/github/{GITHUB_USER}/"
             f"{GITHUB_REPO}/blob/main/{NOTEBOOK}")
GITHUB_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/main/{NOTEBOOK}"
RAW_URL = (f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/"
           f"main/{NOTEBOOK}")

# ---------------------------------------------------------------------------
section("Open the Notebook")

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown(
        f'<a href="{COLAB_URL}" target="_blank">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        f'alt="Open In Colab"></a>',
        unsafe_allow_html=True,
    )
    st.caption("Runs in the browser. No local install.")
with col2:
    st.link_button("View on GitHub", GITHUB_URL, width="stretch")
    st.caption("Rendered, with all outputs.")
with col3:
    st.link_button("Download .ipynb", RAW_URL, width="stretch")
    st.caption("Raw notebook file.")

notebook_path = PROJECT_ROOT / NOTEBOOK
if notebook_path.exists():
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    kpi_row([
        {"label": "Cells", "value": f"{len(cells)}"},
        {"label": "Code Cells", "value": f"{len(code_cells)}"},
        {"label": "Markdown Cells", "value": f"{len(cells) - len(code_cells)}"},
        {"label": "Size", "value": f"{notebook_path.stat().st_size / 1024:.0f} KB"},
    ])
else:
    callout(
        f"`{NOTEBOOK}` is not present in this checkout. Open the GitHub copy above.",
        kind="warning", title="Notebook not found locally",
    )

# ---------------------------------------------------------------------------
section(
    "What The Notebook Covers",
    "An end-to-end supervised classification pipeline on the Kaggle drug200 "
    "dataset, and nothing else: recommending which of five drugs suits a patient "
    "from their clinical readings. Most of the notebook is the data quality and "
    "cleaning work that comes before the model.",
)

sections = pd.DataFrame([
    ("1-2. Setup and the data",
     "Load drug200: 200 patients, five clinical features, five drug classes. "
     "Quantify the class imbalance (roughly 6:1, DrugY 45% against drugB 8%) and "
     "establish the majority-class baseline of 45.5% that any model must beat."),
    ("3. Data quality assessment",
     "Profile the published file (grade A, and the notebook says why that is not "
     "impressive), then inject a controlled set of realistic defects into a copy — "
     "missing values, duplicate submissions, inconsistent free-text categories, "
     "impossible ages — so the cleaning pipeline is tested against known ground "
     "truth rather than eyeballed."),
    ("4. Cleaning",
     "Text standardisation, range checks, deduplication, then median imputation "
     "for numerics and mode imputation for categoricals, with a verification step "
     "asserting every injected defect was caught — and a documented decision log "
     "recording what was done to each issue and why, before any model is fitted."),
    ("5. Exploratory analysis",
     "Find the real decision boundary: Na/K above ~15 routes almost every patient "
     "to DrugY; below it, blood pressure and cholesterol separate the rest."),
    ("6. Feature engineering",
     "Ordinal risk scores for blood pressure and cholesterol, a combined "
     "cardiovascular severity score, a binary flag at the Na/K threshold, and "
     "banded age."),
    ("7-8. Encoding, scaling, splitting",
     "One-hot for nominal features, ordinal where order matters, standardisation "
     "for numerics — all inside one sklearn Pipeline. Stratified 75/25 split."),
    ("9. Decision Tree, balanced weights",
     "Chosen for interpretability and because the true boundary is axis-aligned. "
     "Balanced class weights make an error on the smallest class cost as much as "
     "one on the largest."),
    ("10. Evaluation",
     "Macro-averaged precision, recall and F1; confusion matrix; per-class ROC/AUC "
     "and precision-recall curves."),
    ("11. Error analysis",
     "Every misclassified case examined individually — where it sits relative to "
     "the decision boundary, and whether the model was confidently wrong."),
    ("12. The learned rules",
     "The fitted tree plotted and exported as readable text, plus feature "
     "importance, so the logic can be checked against clinical reasoning."),
    ("13. Comparison and tuning",
     "Decision Tree against Random Forest under identical cross-validation, then a "
     "grid search. Selection on cross-validated macro F1, never the test set."),
    ("14. Serialisation",
     "Persist the complete pipeline with joblib and expose a prediction function, "
     "validated against known patients."),
], columns=["section", "contents"])
show_table(sections, height=440, prettify=False)

callout(
    "**The most interesting result is a negative one.** The error analysis shows "
    "the single misclassified patient was predicted with **100% confidence** — "
    "because a Decision Tree grown to pure leaves reports every prediction as "
    "certain. Its probabilities carry no information, so the obvious safety net "
    "(escalate low-confidence cases to a pharmacist) cannot be built until they "
    "are calibrated. That limitation surfaced from doing the error analysis; "
    "reporting 98% accuracy and stopping would have hidden it entirely.",
    kind="insight", title="What the notebook found",
)

# ---------------------------------------------------------------------------
section(
    "Running It Yourself",
    "The notebook and the dashboard have different requirements, so they are listed "
    "separately. The notebook needs neither a download nor a training step.",
)

tab_notebook, tab_dashboard = st.tabs(["Run the notebook", "Run this dashboard"])

with tab_notebook:
    st.code(f"""git clone https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git
cd {GITHUB_REPO}
pip install -r requirements.txt

jupyter notebook {NOTEBOOK}""", language="bash")
    st.caption(
        "No `fetch_data.py` and no `train_models.py`. `drug200.csv` is tracked in "
        "the repository, so the notebook runs immediately — it trains its own model "
        "in-notebook and does not read the persisted artefacts.")

with tab_dashboard:
    st.code(f"""git clone https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git
cd {GITHUB_REPO}
pip install -r requirements.txt

python scripts/fetch_data.py        # cache the SCMS export (needed here)
python scripts/train_models.py      # train and persist both models
streamlit run app/Home.py           # launch this dashboard""", language="bash")
    st.caption(
        "The dashboard's other pages read the SCMS export and the persisted model "
        "artefacts, so both steps are required here even though the notebook does "
        "not need them.")

callout(
    f"Everything is seeded with `random_seed={cfg.project.random_seed}`, so a clean "
    "checkout reproduces every figure here. A number that differs is a bug worth "
    "reporting, not expected variance.",
    kind="success", title="Reproducibility",
)

methodology("""
**Why a notebook alongside a dashboard.** They serve different readers. The
dashboard answers *what is happening* for someone making a decision; the notebook
shows *how it was computed* for someone checking the work.

**Colab specifics.** The first cell detects Colab, and only then clones the
repository and installs `requirements.txt` — run locally it instead walks up to the
repository root, so the same notebook works in both places unchanged.

**Runtime.** The analysis itself takes about **5 seconds** end to end, measured — 200
rows is a very small dataset, and even the Section 13 grid search is quick at that
size. On Colab the wall-clock time is dominated almost entirely by that first cell
cloning and installing, typically one to two minutes.

**Scope.** The notebook covers the drug200 dataset **only** — no other dataset is
loaded, imported or referenced anywhere in it. The supply chain, pricing and
statistical analysis lives on the other pages of this dashboard rather than being
mixed into a clinical notebook.

**Where the effort went.** Sections 3 and 4 — data quality and cleaning — are the
largest part of the notebook. The published file is clean, so realistic ingestion
defects are injected into a copy and the pipeline is required to catch every one.
That makes the cleaning code testable against known ground truth instead of
eyeballed, and Section 4 closes with a decision log recording what was done to each
issue and why, before any model is fitted.

**One decision in that log is a decision not to act.** The IQR rule flags 8 extreme
`Na_to_K` values, and they are kept. Section 5 shows a high sodium-to-potassium ratio
is the single strongest driver of the label, so those "outliers" are the most
informative rows in the file — removing them would raise the quality score and delete
the signal.
""")

sidebar_about()
