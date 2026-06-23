"""OpenEvidence backend - optional, gated, experimental.

OpenEvidence is a clinical-evidence service behind an API key; this backend is
a thin, clearly-marked sketch against a *plausible* REST shape, not a verified
integration. It exists so the evidence layer is pluggable today and an operator
who has access can wire it without touching the agent code.

Gating: the constructor raises a clear :class:`RuntimeError` (pointing at the
docs) when ``OPENEVIDENCE_API_KEY`` is absent, so misconfiguration fails loudly
at build time. Once built, :meth:`search` follows the house rule - any error
degrades to ``[]`` and never raises into a reasoning loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from dexta_intelligence.evidence.base import EvidenceHit, EvidenceSource

logger = logging.getLogger(__name__)

__all__ = ["OpenEvidenceBackend"]

SOURCE: EvidenceSource = "openevidence"

# UNVERIFIED against a live API: OpenEvidence has no public, self-serve REST docs
# (access is partnership/enterprise-gated). The base URL, the /v1/search path, the
# request body, and the results[] field names below are a plausible guess. An
# operator with real access should confirm exactly those four before relying on it.
API_BASE = "https://api.openevidence.com"
DOCS_URL = "https://www.openevidence.com/policies/api"
_API_KEY_ENV = "OPENEVIDENCE_API_KEY"
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class OpenEvidenceBackend:
    """Experimental :class:`~dexta_intelligence.evidence.base.EvidenceBackend`
    against a hypothesized OpenEvidence REST endpoint. Requires an API key.
    """

    source = SOURCE

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV, "")
        if not key:
            msg = (
                f"OpenEvidence backend requires the {_API_KEY_ENV} environment "
                f"variable (or an explicit api_key). See {DOCS_URL} to obtain one."
            )
            raise RuntimeError(msg)
        self._api_key = key
        self._client = client if client is not None else httpx.Client(timeout=_TIMEOUT)

    def search(self, query: str, *, limit: int = 5) -> list[EvidenceHit]:
        """Search OpenEvidence for ``query``; at most ``limit`` hits, ``[]`` on error.

        EXPERIMENTAL: the request/response shape is a plausible guess and may
        not match the live API. Failures degrade silently to ``[]``.
        """
        query = query.strip()
        if not query:
            return []
        try:
            payload = self._post_json(
                "/v1/search",
                {"query": query, "limit": max(1, limit)},
            )
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list):
                return []
            hits = [hit for doc in results if (hit := _to_hit(doc)) is not None]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            logger.debug("openevidence: search failed for %r", query, exc_info=True)
            return []
        return hits[:limit]

    def _post_json(self, path: str, body: dict[str, Any]) -> Any:
        response = self._client.post(
            f"{API_BASE}{path}",
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        return response.json()


def _to_hit(doc: Any) -> EvidenceHit | None:
    """One hypothesized OpenEvidence result to :class:`EvidenceHit`."""
    if not isinstance(doc, dict):
        return None
    title = doc.get("title")
    url = doc.get("url")
    if not isinstance(title, str) or not title.strip() or not isinstance(url, str) or not url:
        return None
    snippet = doc.get("snippet")
    year = doc.get("year")
    return EvidenceHit(
        title=title.strip(),
        source=SOURCE,
        id=url,
        year=year if isinstance(year, int) else None,
        snippet=snippet.strip() if isinstance(snippet, str) and snippet.strip() else title.strip(),
    )
