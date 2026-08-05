"""
Dataset access layer.

Every consumer - analytics modules, the Streamlit app, notebooks and tests - reads
data through this module rather than touching CSV paths directly. That keeps parsing
rules in one place and gives the whole platform a single caching strategy.

The two datasets, both real
---------------------------
==============  ==========  =========================================================
``drug200``     200 rows    Kaggle clinical dataset - the drug classification model
``scms``        10,324      USAID SCMS delivery history - delivery, vendors, and the
                            product catalogue and pricing analysis
==============  ==========  =========================================================

An Indian medicine catalogue (253,973 products) used to sit alongside these. It was
dropped because SCMS answers the same question better: it carries molecule, brand,
dosage, form, factory and the price *actually paid*, on the same rows as the delivery
performance. A separate list-price catalogue for products nobody in the dataset bought
added scale but no new evidence. See :mod:`src.analytics.products`.

On the absence of a cleaning layer
----------------------------------
An earlier version routed every table through a generic ``clean_table`` step that
imputed and canonicalised on the way through. That existed to service a simulated
extract with deliberately injected defects, and it has been removed along with it.

The two datasets that need real cleaning now own it, because in both cases the
cleaning is dataset-specific and inseparable from correctly interpreting the source:

* :mod:`src.data.scms` parses per-column date formats and classifies every
  ambiguous value with a reason code, distinguishing *structurally absent* from
  *genuinely missing*. A generic imputer would have filled in purchase orders that
  never existed.
``drug200`` needs none - it is published clean, verified: zero nulls, zero
duplicates, no out-of-range values.

The consequence is that ``load_table`` returns the source as published. Anything
that needs interpreting goes through the dedicated module, and there is no longer a
hidden transformation between the file on disk and the frame you get back.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

#: The two real datasets, keyed as they appear in ``config.datasets``.
DATASETS: tuple[str, ...] = ("drug200", "scms")


def _dataset_path(name: str) -> Path:
    """Resolve a logical table name to its absolute path."""
    cfg = get_config()
    if name not in cfg.datasets:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(cfg.datasets)}")
    return resolve_path(cfg.datasets[name])


def ensure_datasets() -> None:
    """Fetch any cached dataset that is not present yet.

    ``drug200`` ships with the repository. SCMS is downloaded on first use and
    cached under ``data/external``, so a fresh clone works without a separate build
    step and later runs need no network.
    """
    from src.data.scms import download_scms

    download_scms()


@lru_cache(maxsize=8)
def load_table(name: str) -> pd.DataFrame:
    """Load a dataset by logical name, exactly as published.

    Results are cached, so repeated calls inside a Streamlit session or notebook are
    free. Call ``load_table.cache_clear()`` if a cached file is replaced.

    Parameters
    ----------
    name
        One of :data:`DATASETS`.

    Returns
    -------
    pandas.DataFrame
        The source data with no transformation applied. For the interpreted
        version use :func:`load_scms`.
    """
    if name == "scms":
        from src.data.scms import download_scms

        download_scms()

    path = _dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}. "
            "Run `python scripts/fetch_data.py` to download the external datasets."
        )

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    log.debug("Loaded %s (%d rows x %d cols)", name, *frame.shape)
    return frame


def load_raw_table(name: str) -> pd.DataFrame:
    """Alias for :func:`load_table`, kept for call sites that read as "raw".

    Since the cleaning layer was removed there is no longer a raw/clean distinction
    at this level - both return the source as published.
    """
    return load_table(name).copy()


# --- Convenience accessors -------------------------------------------------
def load_clinical() -> pd.DataFrame:
    """Kaggle drug200 clinical dataset (200 patients, 5 features, 5 drug classes)."""
    return load_table("drug200").copy()


def load_scms() -> pd.DataFrame:
    """Real USAID SCMS delivery history, parsed and interpreted.

    See :mod:`src.data.scms` - dates are parsed per column and every ambiguous
    value carries a reason code.
    """
    from src.data.scms import load_scms as _load

    return _load().copy()


def load_scms_raw() -> pd.DataFrame:
    """Real SCMS delivery history exactly as published, for quality profiling."""
    from src.data.scms import load_scms_raw as _load_raw

    return _load_raw().copy()


def load_all() -> dict[str, pd.DataFrame]:
    """Load both datasets in their interpreted form."""
    return {
        "clinical": load_clinical(),
        "scms": load_scms(),
    }


__all__ = [
    "DATASETS", "load_table", "load_raw_table", "load_all", "ensure_datasets",
    "load_clinical", "load_scms", "load_scms_raw",
]
