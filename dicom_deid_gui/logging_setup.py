from __future__ import annotations

import logging
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path


class GuiLogHandler(logging.Handler):
    def __init__(self, cb: Callable[[str, str], None]):
        super().__init__()
        self.cb = cb

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            msg = self.format(record)
            level = record.levelname.lower()
            # Do not log PHI values; callers must avoid passing PHI
            self.cb(level, msg)
        except Exception:
            pass


def setup_logging(
    log_dir: Path, gui_callback: Callable[[str, str], None] | None = None
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dicom_deid_gui")
    logger.setLevel(logging.INFO)

    # File logger with rotation
    file_handler = RotatingFileHandler(str(log_dir / "app.log"), maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    if gui_callback is not None:
        gui_handler = GuiLogHandler(gui_callback)
        gui_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(gui_handler)

    return logger
