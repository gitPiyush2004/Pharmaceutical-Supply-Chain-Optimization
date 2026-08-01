"""
Centralised logging for the PharmaChain Analytics platform.

Every module obtains its logger through :func:`get_logger`, which guarantees a
single consistent handler set: coloured console output for interactive work and
a rotating file handler for reproducible run records.

Example
-------
>>> from src.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("Loaded %d batches", 2400)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import get_config, resolve_path

_CONFIGURED = False

# ANSI colours, applied only when stderr is an interactive terminal.
_COLOURS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class _ConsoleFormatter(logging.Formatter):
    """Formatter that colourises the level name on TTY output only."""

    def __init__(self, fmt: str, datefmt: str, use_colour: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colour:
            colour = _COLOURS.get(record.levelname, "")
            # Copy so the file handler still sees the plain level name.
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)


def _configure_root() -> None:
    """Attach console and file handlers to the ``pharmachain`` root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = get_config()
    root = logging.getLogger("pharmachain")
    root.setLevel(getattr(logging, cfg.logging.level.upper(), logging.INFO))
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(
        _ConsoleFormatter(
            fmt=cfg.logging.format,
            datefmt=cfg.logging.date_format,
            use_colour=sys.stderr.isatty(),
        )
    )
    root.addHandler(console)

    # File logging is best-effort: a read-only filesystem (some hosted
    # Streamlit tiers) must not crash the application.
    try:
        log_file = resolve_path(cfg.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(fmt=cfg.logging.format, datefmt=cfg.logging.date_format)
        )
        root.addHandler(file_handler)
    except OSError:  # pragma: no cover - environment dependent
        root.warning("File logging disabled (filesystem not writable).")

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the platform logger.

    Parameters
    ----------
    name
        Usually ``__name__``. A leading ``src.`` is stripped for readability.
    """
    _configure_root()
    clean = name[4:] if name.startswith("src.") else name
    return logging.getLogger(f"pharmachain.{clean}")


__all__ = ["get_logger"]
