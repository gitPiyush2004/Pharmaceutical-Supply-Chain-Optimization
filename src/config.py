"""
Configuration management for the PharmaChain Analytics platform.

Loads ``config/config.yaml`` once, caches it, and exposes it through a small
dot-accessible wrapper so downstream code can write ``cfg.ml.drug_classification``
instead of ``cfg["ml"]["drug_classification"]``.

All filesystem paths in the YAML are relative to the repository root; this
module resolves them to absolute paths so the platform behaves identically
whether it is launched from the repo root, from ``app/``, or from a notebook.

Example
-------
>>> from src.config import get_config, resolve_path
>>> cfg = get_config()
>>> cfg.project.name
'Pharmaceutical Supply Chain Optimization'
>>> resolve_path(cfg.datasets.drug200).name
'drug200.csv'
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

# ---------------------------------------------------------------------------
# Repository root resolution
# ---------------------------------------------------------------------------
# This file lives at <root>/src/config.py, so the root is two levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


class ConfigNode(dict):
    """A ``dict`` that also supports attribute access, recursively.

    Wrapping the parsed YAML in this class keeps call sites readable while
    preserving normal mapping behaviour (``.get()``, ``in``, iteration, and
    JSON serialisation all still work).
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        """Recursively convert nested dicts (and dicts inside lists) to nodes."""
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"No configuration key '{name}'. Available keys: {sorted(self)}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()


@lru_cache(maxsize=1)
def get_config(config_path: str | os.PathLike[str] | None = None) -> ConfigNode:
    """Load and cache the platform configuration.

    Parameters
    ----------
    config_path
        Optional override for the YAML location. Primarily used by tests.

    Returns
    -------
    ConfigNode
        Dot-accessible configuration tree.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path}. "
            "Run the platform from the repository root or set PHARMACHAIN_CONFIG."
        )
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ConfigNode(raw)


def resolve_path(relative: str | os.PathLike[str]) -> Path:
    """Turn a repo-relative path from the config into an absolute path.

    Absolute inputs are returned unchanged, which lets callers pass either.
    """
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def ensure_directories() -> None:
    """Create every directory referenced in ``config.paths`` if it is missing.

    Called at the start of the ETL and training scripts so a fresh clone works
    without any manual ``mkdir``.
    """
    cfg = get_config()
    for value in cfg.paths.values():
        target = resolve_path(value)
        # `database` points at a file, not a directory - create its parent.
        directory = target.parent if target.suffix else target
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "ConfigNode",
    "get_config",
    "resolve_path",
    "ensure_directories",
]
