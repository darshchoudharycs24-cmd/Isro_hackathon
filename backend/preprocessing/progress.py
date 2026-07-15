"""
progress.py - Numbered pipeline step progress logger.

Replaces bare "Loading... Done" messages with structured step counters:

    [1/8] Loading imagery          ...
    [1/8] Loading imagery          ✔  0.34 s   (RAM: 142 MB)
    [2/8] Calibration              ...
    ...
    ✔ Pipeline completed in 9.21 s

Usage::

    progress = PipelineProgress(total_steps=8)
    with progress.step("Loading imagery"):
        data = load(...)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _ram_mb() -> float | None:
    """Return current process RSS in megabytes, or None if psutil unavailable."""
    if not _PSUTIL_AVAILABLE:
        return None
    try:
        import os  # noqa: PLC0415
        proc = _psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return None


class PipelineProgress:
    """
    Numbered step progress tracker with timing and memory reporting.

    Parameters
    ----------
    total_steps : int
        Total number of pipeline steps (used for [N/total] display).
    logger_name : str
        Logger name to emit progress lines to.
    """

    def __init__(self, total_steps: int = 8, logger_name: str = __name__) -> None:
        self.total = total_steps
        self._current = 0
        self._pipeline_start = time.perf_counter()
        self._log = logging.getLogger(logger_name)

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        """
        Context manager for a single numbered pipeline step.

        Logs ``[N/total] <label>`` on entry and
        ``[N/total] <label>  ✔  X.XX s  (RAM: Y MB)`` on exit.

        Parameters
        ----------
        label : str
            Human-readable step description.
        """
        self._current += 1
        prefix = f"[{self._current}/{self.total}]"
        ram_before = _ram_mb()

        self._log.info("%-8s %-30s ...", prefix, label)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            ram_after = _ram_mb()

            if ram_after is not None and ram_before is not None:
                ram_delta = ram_after - ram_before
                ram_info = f"   RAM: {ram_after:.0f} MB (Δ{ram_delta:+.0f} MB)"
            else:
                ram_info = ""

            self._log.info(
                "%-8s %-30s ✔  %.2f s%s",
                prefix, label, elapsed, ram_info,
            )

    def complete(self, total_elapsed: float | None = None) -> None:
        """
        Log the final completion message.

        Parameters
        ----------
        total_elapsed : float, optional
            Total pipeline time in seconds.  Computed from construction
            time if not provided.
        """
        if total_elapsed is None:
            total_elapsed = time.perf_counter() - self._pipeline_start

        ram = _ram_mb()
        ram_str = f"  |  RAM: {ram:.0f} MB" if ram else ""
        self._log.info(
            "✔ Pipeline completed in %.2f s%s",
            total_elapsed, ram_str,
        )
