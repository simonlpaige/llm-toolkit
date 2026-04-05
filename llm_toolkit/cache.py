"""
cache.py — Disk-based response caching for LLM API calls.

Caches results to JSON files keyed by a hash of the request parameters.
Supports TTL-based expiry and manual invalidation.

Usage::

    from llm_toolkit.cache import DiskCache

    cache = DiskCache()

    @cache.cached()
    def call_api(prompt, model="gpt-4o"):
        return client.chat.completions.create(...)

    # Or use inline:
    key = cache.make_key(model="gpt-4o", messages=messages)
    result = cache.get(key)
    if result is None:
        result = call_api()
        cache.set(key, result)
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DiskCache:
    """
    A simple disk-based cache backed by JSON files.

    Each entry is stored as ``<cache_dir>/<hash>.json`` with an optional TTL.

    Args:
        cache_dir: Directory for cache files. Defaults to ``$LLM_CACHE_DIR`` or ``.llm_cache``.
        ttl: Time-to-live in seconds. None = never expire.
        enabled: Toggle caching on/off without changing code.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl: Optional[float] = None,
        enabled: bool = True,
    ):
        self.cache_dir = Path(
            cache_dir or os.environ.get("LLM_CACHE_DIR", ".llm_cache")
        )
        self.ttl = ttl
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Core API ──────────────────────────────────────────────────────────────

    def make_key(self, **kwargs: Any) -> str:
        """
        Derive a stable cache key from keyword arguments.

        The key is a SHA-256 hash of the JSON-serialised kwargs (sorted keys).
        """
        serialised = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value for key, or None if missing/expired."""
        if not self.enabled:
            return None

        path = self._path(key)
        if not path.exists():
            self._misses += 1
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cache read error for %s: %s", key, e)
            self._misses += 1
            return None

        # Check TTL
        ttl = entry.get("ttl")
        stored_at = entry.get("stored_at", 0)
        if ttl is not None and (time.time() - stored_at) > ttl:
            path.unlink(missing_ok=True)
            self._misses += 1
            logger.debug("Cache entry expired: %s", key)
            return None

        self._hits += 1
        logger.debug("Cache hit: %s", key)
        return entry.get("value")

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value in the cache."""
        if not self.enabled:
            return

        entry = {
            "key": key,
            "stored_at": time.time(),
            "ttl": ttl if ttl is not None else self.ttl,
            "value": value,
        }

        path = self._path(key)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, default=str)
        except OSError as e:
            logger.warning("Cache write error for %s: %s", key, e)

    def delete(self, key: str) -> bool:
        """Remove a single cache entry. Returns True if it existed."""
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Delete all cache entries. Returns count of deleted files."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cache entries", count)
        return count

    def clear_expired(self) -> int:
        """Remove only expired entries. Returns count of deleted files."""
        count = 0
        now = time.time()
        for f in self.cache_dir.glob("*.json"):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    entry = json.load(fh)
                ttl = entry.get("ttl")
                stored_at = entry.get("stored_at", 0)
                if ttl is not None and (now - stored_at) > ttl:
                    f.unlink()
                    count += 1
            except (json.JSONDecodeError, OSError):
                pass
        return count

    # ── Decorator ─────────────────────────────────────────────────────────────

    def cached(
        self,
        ttl: Optional[float] = None,
        key_args: Optional[list[str]] = None,
    ) -> Callable:
        """
        Decorator that caches the return value of a function.

        Args:
            ttl: Override the cache TTL for this function.
            key_args: If set, only these argument names are included in the cache key.
                      Useful to exclude non-deterministic or irrelevant args.

        Example::

            @cache.cached(ttl=3600)
            def get_embedding(text, model="text-embedding-3-small"):
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Build key from function name + bound arguments
                import inspect
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                all_args = dict(bound.arguments)

                if key_args is not None:
                    key_data = {k: v for k, v in all_args.items() if k in key_args}
                else:
                    key_data = all_args

                key_data["__fn__"] = func.__qualname__
                key = self.make_key(**key_data)

                cached_val = self.get(key)
                if cached_val is not None:
                    return cached_val

                result = func(*args, **kwargs)
                self.set(key, result, ttl=ttl)
                return result

            return wrapper

        return decorator

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "cache_dir": str(self.cache_dir),
            "ttl": self.ttl,
            "enabled": self.enabled,
        }

    def __repr__(self) -> str:
        return (
            f"DiskCache(cache_dir={str(self.cache_dir)!r}, "
            f"ttl={self.ttl}, hits={self._hits}, misses={self._misses})"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
