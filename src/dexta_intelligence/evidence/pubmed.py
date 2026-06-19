"""PubMed backend — zero-auth NCBI E-utilities (esearch + esummary).

Two calls, no key required: ``esearch`` ranks PMIDs for the query by relevance,
then ``esummary`` resolves each PMID to a title and publication year. The
snippet is the title — abstracts (``efetch``) are intentionally skipped to keep
the round-trip light; ground a pattern, don't write a review.

NCBI etiquette: requests carry ``tool`` and ``email`` params when an email is
configured (E-utilities asks identifiers so they can contact heavy users before
throttling). Everything is wrapped so the backend degrades to ``[]`` on any
HTTP, timeout, or parse failure — it must never raise into a reasoning loop.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import httpx

from dexta_intelligence.evidence.base import EvidenceHit, EvidenceSource

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

__all__ = ["PubMedBackend"]

SOURCE: EvidenceSource = "pubmed"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

#: NCBI default tool identifier; paired with a configured email per etiquette.
_TOOL_NAME = "dexta-intelligence"
#: An NCBI key raises the rate limit from 3 to 10 requests/second (optional).
_API_KEY_ENV = "NCBI_API_KEY"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
#: One short backoff on a 429 (unkeyed NCBI throttles at 3 req/s).
_RATE_LIMIT_BACKOFF_S = 0.4
#: Hard ceiling so a runaway ``limit`` can't ask NCBI for a huge page.
_MAX_RETMAX = 20


class PubMedBackend:
    """Implements :class:`~dexta_intelligence.evidence.base.EvidenceBackend`
    against NCBI E-utilities (``db=pubmed``), no authentication required.
    """

    source = SOURCE

    def __init__(
        self,
        *,
        email: str = "",
        client: httpx.Client | None = None,
    ) -> None:
        self._email = email
        self._client = client if client is not None else httpx.Client(timeout=_TIMEOUT)

    def search(self, query: str, *, limit: int = 5) -> list[EvidenceHit]:
        """Search PubMed for ``query``; at most ``limit`` hits, ``[]`` on error."""
        query = query.strip()
        if not query:
            return []
        retmax = max(1, min(limit, _MAX_RETMAX))
        try:
            pmids = self._esearch(query, retmax)
            if not pmids:
                return []
            summaries = self._esummary(pmids)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            logger.debug("pubmed: search failed for %r", query, exc_info=True)
            return []

        hits = [hit for pmid in pmids if (hit := _to_hit(pmid, summaries.get(pmid))) is not None]
        return hits[:limit]

    # -- E-utilities plumbing -------------------------------------------------

    def _params(self, extra: dict[str, str | int]) -> dict[str, str | int]:
        params: dict[str, str | int] = {"tool": _TOOL_NAME, **extra}
        if self._email:
            params["email"] = self._email
        key = os.environ.get(_API_KEY_ENV)
        if key:
            params["api_key"] = key
        return params

    def _get_json(self, path: str, params: dict[str, str | int]) -> Any:
        url = f"{EUTILS_BASE}{path}"
        full = self._params(params)
        response = self._client.get(url, params=full)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:  # NCBI throttle
            time.sleep(_RATE_LIMIT_BACKOFF_S)
            response = self._client.get(url, params=full)
        response.raise_for_status()
        return response.json()

    def _esearch(self, query: str, retmax: int) -> list[str]:
        payload = self._get_json(
            "/esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmax": retmax,
                "sort": "relevance",
                "retmode": "json",
            },
        )
        if not isinstance(payload, dict):
            return []
        result = payload.get("esearchresult")
        idlist = result.get("idlist") if isinstance(result, dict) else None
        if not isinstance(idlist, list):
            return []
        return [str(pmid) for pmid in idlist if str(pmid)]

    def _esummary(self, pmids: Iterable[str]) -> dict[str, dict[str, Any]]:
        payload = self._get_json(
            "/esummary.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        )
        if not isinstance(payload, dict):
            return {}
        result = payload.get("result")
        if not isinstance(result, dict):
            return {}
        return {
            str(uid): doc
            for uid, doc in result.items()
            if uid != "uids" and isinstance(doc, dict)
        }


def _to_hit(pmid: str, doc: dict[str, Any] | None) -> EvidenceHit | None:
    """One esummary doc to :class:`EvidenceHit`; ``None`` when unusable."""
    if not isinstance(doc, dict):
        return None
    title = doc.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    title = title.strip().rstrip(".")
    return EvidenceHit(
        title=title,
        source=SOURCE,
        id=pmid,
        year=_year_of(doc.get("pubdate")),
        snippet=title,
    )


def _year_of(pubdate: Any) -> int | None:
    """First four-digit token of an NCBI ``pubdate`` (e.g. ``2021 Mar 4``)."""
    if not isinstance(pubdate, str):
        return None
    head = pubdate.strip()[:4]
    return int(head) if head.isdigit() else None
