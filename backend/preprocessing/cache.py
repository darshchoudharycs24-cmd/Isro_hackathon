"""
cache.py - File-hash-based intermediate result cache.

Caches expensive intermediate arrays (aligned image, normalised image,
SAR preprocessing) keyed by SHA-256 hash of the input file(s).
Stale entries are invalidated automatically when source files change.

Cache directory: .cache/  (relative to project root by default)

Usage::

    cache = PipelineCache()

    key = cache.hash_file("current.tif") + cache.hash_file("historical.tif")
    aligned = cache.load(key, "aligned")
    if aligned is None:
        aligned = expensive_alignment(...)
        cache.save(key, "aligned", aligned)
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_VERSION = "1"  # bump to invalidate all entries on breaking changes


class PipelineCache:
    """
    Filesystem NumPy cache with SHA-256 file-hash keys.

    Parameters
    ----------
    cache_dir : str or Path
        Root cache directory.  Created on first use.
    max_age_seconds : float
        Entries older than this are treated as stale and ignored.
        Default 7 days.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        max_age_seconds: float = 7 * 24 * 3600,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max_age_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ─── Public API ────────────────────────────────────────────────────────

    def hash_file(self, path: str | Path) -> str:
        """
        Compute SHA-256 hash of a file's content.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        str
            16-character hex prefix (sufficient for collision resistance here).
        """
        path = Path(path)
        if not path.exists():
            return "missing"

        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]

    def make_key(self, *paths: str | Path) -> str:
        """
        Build a compound cache key from one or more file hashes.

        Parameters
        ----------
        *paths
            File paths to hash. Order matters.

        Returns
        -------
        str
            Combined key string.
        """
        parts = [self.hash_file(p) for p in paths]
        parts.append(_CACHE_VERSION)
        return "_".join(parts)

    def load(self, key: str, stage: str) -> np.ndarray | None:
        """
        Load a cached array if it exists and is fresh.

        Parameters
        ----------
        key : str
            Cache key from ``make_key()``.
        stage : str
            Pipeline stage name (e.g. ``'aligned'``, ``'normalised'``).

        Returns
        -------
        np.ndarray or None
            Cached array or None if cache miss / stale.
        """
        cache_path = self._path(key, stage)
        if not cache_path.exists():
            logger.debug("Cache MISS: %s/%s", key[:8], stage)
            return None

        age = time.time() - cache_path.stat().st_mtime
        if age > self.max_age_seconds:
            logger.debug("Cache STALE (%.0f s): %s/%s", age, key[:8], stage)
            cache_path.unlink(missing_ok=True)
            return None

        try:
            arr = np.load(str(cache_path))
            logger.debug("Cache HIT: %s/%s | shape=%s", key[:8], stage, arr.shape)
            return arr
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache load failed (%s); treating as miss.", exc)
            cache_path.unlink(missing_ok=True)
            return None

    def save(self, key: str, stage: str, data: np.ndarray) -> None:
        """
        Persist an array to the cache.

        Parameters
        ----------
        key : str
            Cache key.
        stage : str
            Stage name.
        data : np.ndarray
            Array to cache.
        """
        cache_path = self._path(key, stage)
        try:
            np.save(str(cache_path), data)
            logger.debug("Cache WRITE: %s/%s | shape=%s", key[:8], stage, data.shape)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache write failed (%s); continuing without cache.", exc)

    def invalidate(self, key: str, stage: str | None = None) -> int:
        """
        Remove one or all cached entries for a key.

        Parameters
        ----------
        key : str
            Cache key to invalidate.
        stage : str, optional
            If given, only that stage entry is removed.  Otherwise all
            entries matching the key prefix are removed.

        Returns
        -------
        int
            Number of files deleted.
        """
        if stage:
            p = self._path(key, stage)
            if p.exists():
                p.unlink()
                return 1
            return 0

        pattern = f"{key}_*.npy"
        deleted = 0
        for p in self.cache_dir.glob(pattern):
            p.unlink()
            deleted += 1
        logger.debug("Invalidated %d cache entries for key %s", deleted, key[:8])
        return deleted

    def clear_all(self) -> int:
        """Delete every .npy file in the cache directory."""
        deleted = 0
        for p in self.cache_dir.glob("*.npy"):
            p.unlink()
            deleted += 1
        logger.info("Cache cleared: %d files deleted.", deleted)
        return deleted

    def stats(self) -> dict[str, Any]:
        """Return cache directory statistics."""
        files = list(self.cache_dir.glob("*.npy"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "cache_dir": str(self.cache_dir),
            "entries": len(files),
            "size_mb": round(total_bytes / 1024 / 1024, 2),
        }

    # ─── Internal ──────────────────────────────────────────────────────────

    def _path(self, key: str, stage: str) -> Path:
        return self.cache_dir / f"{key}_{stage}.npy"
