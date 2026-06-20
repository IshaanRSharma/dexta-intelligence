"""Evidence contract - one hit shape, one search surface.

A backend turns a free-text query into a short list of :class:`EvidenceHit`
records. The shape is deliberately minimal: enough to cite (title, source, id),
enough to verify (a numeric ``year`` and an id that carries a PMID), and a
``snippet`` for the model to read. Backends own provider I/O and normalization
and must never raise into a reasoning loop - they return ``[]`` on any error.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

__all__ = ["EvidenceBackend", "EvidenceHit", "EvidenceSource"]

EvidenceSource = Literal["pubmed", "openevidence"]


class EvidenceHit(BaseModel):
    """One published result a confirmed pattern can be grounded against.

    ``id`` is a PMID (PubMed) or a URL (OpenEvidence); ``snippet`` is a short
    human-readable excerpt (the title when no abstract is fetched). ``year`` is
    ``None`` when the source omits it. Frozen so a hit can't be mutated after a
    backend hands it to the reasoning loop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    source: EvidenceSource
    id: str
    year: int | None = None
    snippet: str


@runtime_checkable
class EvidenceBackend(Protocol):
    """The one method a literature provider implements.

    Returns at most ``limit`` hits ranked by the provider's relevance order.
    Must never raise - any network, auth, or parse failure yields ``[]`` so an
    agent loop can treat "no evidence" and "lookup failed" identically.
    """

    def search(self, query: str, *, limit: int = 5) -> list[EvidenceHit]: ...
