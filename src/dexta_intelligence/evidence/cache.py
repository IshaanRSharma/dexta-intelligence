"""Process-wide TTL cache for evidence lookups.

Published literature for a stable query (e.g. "overnight hypoglycemia") does not
change minute to minute, so a confirmed pattern's citations can be reused across
page loads and overlapping findings. :class:`CachingEvidenceBackend` wraps any
:class:`~dexta_intelligence.evidence.base.EvidenceBackend` and shares one
process-global store, so a fresh wrapper built per request still hits the cache.

Only non-empty results are cached: the inner backend returns ``[]`` for both
"no results" and "lookup failed" (the never-raise contract), so caching empties
would pin a transient failure. The wrapper preserves that contract - it only
ever delegates and reads the inner result.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from dexta_intelligence.evidence.base import EvidenceBackend, EvidenceHit

__all__ = ["CachingEvidenceBackend", "reset_cache"]

#: (source, query, limit) -> (inserted_monotonic, hits). Shared across wrappers.
_CACHE: dict[tuple[str, str, int], tuple[float, list[EvidenceHit]]] = {}
_LOCK = threading.Lock()


def reset_cache() -> None:
    """Drop every cached lookup (used by tests for isolation)."""
    with _LOCK:
        _CACHE.clear()


class CachingEvidenceBackend:
    """A TTL cache in front of another backend. Implements ``EvidenceBackend``."""

    def __init__(
        self,
        inner: EvidenceBackend,
        *,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        self._clock = clock
        self._source = str(getattr(inner, "source", inner.__class__.__name__))

    def search(self, query: str, *, limit: int = 5) -> list[EvidenceHit]:
        q = query.strip()
        if not q:
            return []
        key = (self._source, q, limit)
        now = self._clock()
        with _LOCK:
            cached = _CACHE.get(key)
            if cached is not None and now - cached[0] < self._ttl:
                return list(cached[1])
        result = self._inner.search(q, limit=limit)
        if result:  # never cache an empty result (could be a transient failure)
            with _LOCK:
                _CACHE[key] = (now, list(result))
        return result
