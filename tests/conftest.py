"""
Shared pytest fixtures.

Every dataset is loaded once per session. Two of the three are downloaded on first
use and cached under ``data/external``, so a session-scoped fixture is the
difference between one download and one per test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.data import loader  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """Platform configuration."""
    return get_config()


@pytest.fixture(scope="session", autouse=True)
def _datasets_available():
    """Download and cache the two external datasets before any test runs.

    A no-op once they are on disk, so only the first run needs a network.
    """
    loader.ensure_datasets()


@pytest.fixture(scope="session")
def clinical():
    """Kaggle drug200 clinical dataset - 200 patients, 5 drug classes."""
    return loader.load_clinical()


@pytest.fixture(scope="session")
def scms():
    """USAID SCMS delivery history, parsed and interpreted."""
    return loader.load_scms()


@pytest.fixture(scope="session")
def scms_raw():
    """USAID SCMS delivery history exactly as published."""
    return loader.load_scms_raw()


@pytest.fixture(scope="session")
def indian_medicines():
    """Indian medicine product master, normalised - 253,973 products."""
    return loader.load_indian_medicines()


@pytest.fixture(scope="session")
def indian_medicines_raw():
    """Indian medicine product master exactly as published."""
    return loader.load_indian_medicines_raw()


@pytest.fixture(scope="session")
def models_available() -> bool:
    """True when both trained model artefacts are present on disk."""
    from src.config import resolve_path

    models = resolve_path(get_config().paths.models)
    return all((models / f"{name}_model.joblib").exists()
               for name in ("drug_classification", "late_delivery"))
