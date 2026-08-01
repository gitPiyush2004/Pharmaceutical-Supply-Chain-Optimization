"""
Shared pytest fixtures.

Data is loaded once per session: the generator is deterministic, so there is no
reason to rebuild it per test, and the suite stays fast enough to run on every
commit.
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
    """Ensure the data layer exists before any test runs."""
    loader.ensure_datasets()


@pytest.fixture(scope="session")
def batches():
    """Cleaned (silver) batch funnel fact table."""
    return loader.load_batches()


@pytest.fixture(scope="session")
def raw_batches():
    """Raw (bronze) batch extract, defects intact."""
    return loader.load_raw_table("batches")


@pytest.fixture(scope="session")
def shipments():
    return loader.load_shipments()


@pytest.fixture(scope="session")
def inventory_snapshots():
    return loader.load_inventory()


@pytest.fixture(scope="session")
def demand():
    return loader.load_demand()


@pytest.fixture(scope="session")
def clinical():
    """Kaggle drug200 clinical dataset."""
    return loader.load_clinical()


@pytest.fixture(scope="session")
def models_available() -> bool:
    """True when both trained model artefacts are present on disk."""
    from src.config import resolve_path

    models = resolve_path(get_config().paths.models)
    return all((models / f"{name}_model.joblib").exists()
               for name in ("drug_classification", "batch_risk"))
