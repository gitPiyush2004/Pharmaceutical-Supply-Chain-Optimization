"""
Google Colab page.

Links to the reproducible end-to-end notebook and documents exactly what it does,
so a reviewer can re-derive every model in the platform from a browser with no
local setup.
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
    subtitle="The complete machine learning pipeline, reproducible in a browser",
)

GITHUB_USER = "gitPiyush2004"
GITHUB_REPO = "Pharmaceutical-Supply-Chain-Optimization"
NOTEBOOK = "notebooks/pharmaceutical_ml_pipeline.ipynb"

COLAB_URL = (f"https://colab.research.google.com/github/{GITHUB_USER}/"
             f"{GITHUB_REPO}/blob/main/{NOTEBOOK}")
GITHUB_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/main/{NOTEBOOK}"
RAW_URL = (f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/"
           f"main/{NOTEBOOK}")

# ---------------------------------------------------------------------------
section("Open the Notebook")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<a href="{COLAB_URL}" target="_blank">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        f'alt="Open In Colab"></a>',
        unsafe_allow_html=True,
    )
    st.caption("Runs in the browser. No local install.")
with col2:
    st.link_button("View on GitHub", GITHUB_URL, use_container_width=True)
    st.caption("Rendered notebook with outputs.")
with col3:
    st.link_button("Download .ipynb", RAW_URL, use_container_width=True)
    st.caption("Raw notebook file.")

notebook_path = PROJECT_ROOT / NOTEBOOK
if notebook_path.exists():
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    kpi_row([
        {"label": "Notebook Cells", "value": f"{len(cells)}"},
        {"label": "Code Cells", "value": f"{len(code_cells)}"},
        {"label": "Markdown Cells", "value": f"{len(cells) - len(code_cells)}"},
        {"label": "Size", "value": f"{notebook_path.stat().st_size / 1024:.0f} KB"},
    ])
    with open(notebook_path, "rb") as handle:
        st.download_button("Download notebook from this app", handle.read(),
                           file_name=notebook_path.name,
                           mime="application/x-ipynb+json")
else:
    callout(
        f"`{NOTEBOOK}` is not present in this checkout. Build it by running the "
        "training pipeline, or open the GitHub copy above.",
        kind="warning", title="Notebook not found locally",
    )

callout(
    "The Colab badge resolves against the `main` branch on GitHub. It works as soon "
    "as the repository is public - Colab reads the notebook straight from GitHub, so "
    "no separate upload or Drive copy is needed.",
    kind="insight", title="How the link works",
)

# ---------------------------------------------------------------------------
section(
    "What the Notebook Covers",
    "The same pipeline the platform uses, laid out linearly with explanation at each "
    "step. It is the artefact to read to understand *how*, where the dashboard shows "
    "*what*.",
)

sections = pd.DataFrame([
    {"section": "1. Setup and configuration",
     "contents": "Clone the repository, install dependencies, load `config/config.yaml`."},
    {"section": "2. Data generation",
     "contents": "Build the seeded supply chain digital twin; show the product mix "
                 "derived from the drug200 prescription distribution."},
    {"section": "3. Exploratory data analysis",
     "contents": "Distributions, correlations and class balance across both datasets."},
    {"section": "4. Data quality assessment",
     "contents": "Profile the bronze extract, score five quality dimensions, then "
                 "quantify the uplift delivered by the cleaning layer."},
    {"section": "5. Preprocessing and feature engineering",
     "contents": "Median/mode imputation, one-hot encoding, scaling, and the "
                 "engineered clinical and thermal-load features - with the reasoning "
                 "for each."},
    {"section": "6. Model 1 - drug classification",
     "contents": "Decision tree, random forest and XGBoost tuned by grid search "
                 "under stratified 5-fold CV; confusion matrix, ROC, "
                 "precision-recall and feature importance."},
    {"section": "7. Model 2 - batch risk classification",
     "contents": "The same protocol applied to batch stability telemetry, "
                 "predicting Low/Medium/High risk."},
    {"section": "8. Funnel analytics",
     "contents": "Eight-stage conversion, drop-off, dwell time and bottleneck "
                 "identification."},
    {"section": "9. Statistical A/B testing",
     "contents": "Two-proportion z-test, chi-square, Welch's t-test, power analysis "
                 "and the costed recommendation for each intervention."},
    {"section": "10. Demand forecasting",
     "contents": "Holt-Winters against linear trend and moving average, selected by "
                 "backtest on held-out months."},
    {"section": "11. Drug stability modelling",
     "contents": "OLS degradation model, shelf-life estimation and the cold-chain "
                 "excursion significance test."},
    {"section": "12. Model serialisation",
     "contents": "Persist pipelines with joblib and write the metadata the dashboard "
                 "reads."},
])
show_table(sections, height=470)

# ---------------------------------------------------------------------------
section("Running It Locally Instead")

st.code(f"""# Clone and install
git clone https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git
cd {GITHUB_REPO}
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the data layer and train both models (deterministic, seed={cfg.project.random_seed})
python scripts/build_dataset.py
python scripts/train_models.py

# Launch this dashboard
streamlit run app/Home.py""", language="bash")

callout(
    f"Everything is seeded with `random_seed={cfg.project.random_seed}`, so a clean "
    "checkout reproduces every figure in this dashboard byte for byte. If a number "
    "here differs from a number you generate, that is a bug worth reporting - not "
    "expected variance.",
    kind="success", title="Reproducibility guarantee",
)

# ---------------------------------------------------------------------------
methodology("""
**Why a notebook alongside a dashboard.** They serve different readers. The
dashboard answers *what is happening* for someone making a decision; the notebook
shows *how it was computed* for someone checking the work. Keeping both means the
analysis is auditable without making the dashboard a wall of code.

**No duplicated logic.** The notebook imports the same `src` modules the dashboard
and CLI scripts use. It does not reimplement the pipeline, so there is exactly one
definition of every metric and no risk of the notebook and the application drifting
apart.

**Colab specifics.** The first cell clones the repository and installs
`requirements.txt`, so the notebook is self-contained on a fresh Colab runtime.
Runtime is a few minutes end to end, dominated by the two hyper-parameter grid
searches.
""")

sidebar_about()
