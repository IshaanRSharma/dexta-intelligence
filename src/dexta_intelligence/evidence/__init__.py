"""Clinical evidence grounding — search published literature to back a pattern.

A reasoning loop confirms a *personal* pattern from the user's own data, then
reaches here to ground it in (or contradict it against) the published record.
Backends are pluggable behind a tiny :class:`EvidenceBackend` protocol; the
default is a zero-auth PubMed (NCBI E-utilities) backend, and an optional gated
OpenEvidence backend sits behind an API key.

Every backend is built to *never raise into an agent loop*: any network or
parse failure degrades to an empty hit list, so a missing-evidence run is
indistinguishable from a clean no-results run.
"""

from __future__ import annotations

from dexta_intelligence.evidence.base import EvidenceBackend, EvidenceHit

__all__ = ["EvidenceBackend", "EvidenceHit"]
