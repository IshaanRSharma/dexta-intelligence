"""Evidence grounding tests — PubMed backend via MockTransport, hit shape, tool wiring.

No live network: the PubMed backend runs against an ``httpx.MockTransport`` that
emulates the NCBI E-utilities ``esearch`` + ``esummary`` JSON responses from
sanitized fixtures. The OpenEvidence backend is exercised only for its
missing-key gate. The ``search_evidence`` tool spec is checked end to end with
the backend factory monkeypatched to a stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from dexta_intelligence.config import EvidenceConfig, load_config
from dexta_intelligence.evidence.base import EvidenceHit
from dexta_intelligence.evidence.openevidence import OpenEvidenceBackend
from dexta_intelligence.evidence.pubmed import PubMedBackend

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


ESEARCH = _load("pubmed_esearch.json")
ESEARCH_EMPTY = _load("pubmed_esearch_empty.json")
ESUMMARY = _load("pubmed_esummary.json")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceConfig:
    def test_defaults(self) -> None:
        cfg = EvidenceConfig()
        assert cfg.backend == "pubmed"
        assert cfg.email == ""
        assert cfg.enabled is True

    def test_present_on_config(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "missing.toml")
        assert config.evidence == EvidenceConfig()


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceHit shape
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceHit:
    def test_minimal_hit(self) -> None:
        hit = EvidenceHit(title="T", source="pubmed", id="123", snippet="T")
        assert hit.year is None
        assert hit.source == "pubmed"

    def test_frozen(self) -> None:
        hit = EvidenceHit(title="T", source="pubmed", id="123", snippet="T")
        with pytest.raises((TypeError, ValueError)):
            hit.title = "other"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# PubMed backend — mocked NCBI E-utilities transport
# ─────────────────────────────────────────────────────────────────────────────


class _PubMedServer:
    """Mock of NCBI E-utilities: esearch ranks PMIDs, esummary resolves them."""

    def __init__(self, *, esearch: dict[str, Any] = ESEARCH, fail: bool = False) -> None:
        self._esearch = esearch
        self._fail = fail
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._fail:
            return httpx.Response(500, json={"error": "server error"})
        path = request.url.path
        if path.endswith("/esearch.fcgi"):
            return httpx.Response(200, json=self._esearch)
        if path.endswith("/esummary.fcgi"):
            return httpx.Response(200, json=ESUMMARY)
        return httpx.Response(404)


def _backend(server: _PubMedServer, *, email: str = "") -> PubMedBackend:
    client = httpx.Client(transport=httpx.MockTransport(server.handler))
    return PubMedBackend(email=email, client=client)


class TestPubMedBackend:
    def test_search_returns_hits(self) -> None:
        hits = _backend(_PubMedServer()).search("exercise glucose", limit=5)
        assert len(hits) == 2
        first = hits[0]
        assert first.source == "pubmed"
        assert first.id == "30100000"
        assert first.year == 2018
        assert "Postprandial exercise" in first.title
        assert first.snippet == first.title  # snippet is the title (no abstract fetch)

    def test_search_preserves_relevance_order(self) -> None:
        hits = _backend(_PubMedServer()).search("dawn phenomenon", limit=5)
        assert [h.id for h in hits] == ["30100000", "29200001"]

    def test_limit_truncates(self) -> None:
        hits = _backend(_PubMedServer()).search("glucose", limit=1)
        assert len(hits) == 1
        assert hits[0].id == "30100000"

    def test_empty_query_skips_network(self) -> None:
        server = _PubMedServer()
        assert _backend(server).search("   ") == []
        assert server.requests == []

    def test_no_pmids_returns_empty(self) -> None:
        server = _PubMedServer(esearch=ESEARCH_EMPTY)
        hits = _backend(server).search("nonsense terms")
        assert hits == []
        # esearch was called, esummary was not (nothing to resolve)
        assert all("esummary" not in r.url.path for r in server.requests)

    def test_http_error_degrades_to_empty(self) -> None:
        hits = _backend(_PubMedServer(fail=True)).search("exercise glucose")
        assert hits == []

    def test_email_etiquette_param_sent(self) -> None:
        server = _PubMedServer()
        _backend(server, email="ops@example.com").search("glucose")
        esearch_req = next(r for r in server.requests if "esearch" in r.url.path)
        assert esearch_req.url.params.get("email") == "ops@example.com"
        assert esearch_req.url.params.get("tool") == "dexta-intelligence"
        assert esearch_req.url.params.get("sort") == "relevance"

    def test_email_absent_omits_param(self) -> None:
        server = _PubMedServer()
        _backend(server).search("glucose")
        esearch_req = next(r for r in server.requests if "esearch" in r.url.path)
        assert "email" not in esearch_req.url.params


# ─────────────────────────────────────────────────────────────────────────────
# OpenEvidence backend — missing-key gate
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenEvidenceBackend:
    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENEVIDENCE_API_KEY", raising=False)
        with pytest.raises(RuntimeError) as exc:
            OpenEvidenceBackend()
        assert "docs.openevidence.com" in str(exc.value)
        assert "OPENEVIDENCE_API_KEY" in str(exc.value)

    def test_env_key_constructs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENEVIDENCE_API_KEY", "k-123")
        backend = OpenEvidenceBackend()
        assert backend.source == "openevidence"

    def test_search_degrades_to_empty_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENEVIDENCE_API_KEY", raising=False)

        def boom(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "down"})

        client = httpx.Client(transport=httpx.MockTransport(boom))
        backend = OpenEvidenceBackend(api_key="k-123", client=client)
        assert backend.search("glucose") == []


# ─────────────────────────────────────────────────────────────────────────────
# search_evidence tool spec — wiring + guard-facing numbers
# ─────────────────────────────────────────────────────────────────────────────


class _StubBackend:
    def __init__(self, hits: list[EvidenceHit]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int = 5) -> list[EvidenceHit]:
        self.calls.append((query, limit))
        return self._hits


def test_tool_specs_includes_search_evidence() -> None:
    import dexta_intelligence.agents.discovery_tools as dt  # noqa: PLC0415

    spec = next(
        s for s in _all_specs(dt) if s.name == "search_evidence"
    )
    assert "PubMed" in spec.description
    assert "Cite only returned PMIDs" in spec.description
    assert spec.parameters["properties"]["limit"]["maximum"] == 8
    assert spec.parameters["required"] == ["query"]


def _all_specs(dt: Any) -> list[Any]:
    """Build tool_specs with lightweight stubs for ctx/toolkit."""
    from datetime import date  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    class _Store:
        def get_glucose(self, *_a: Any, **_k: Any) -> list[Any]:
            return []

        def get_sleep(self, *_a: Any, **_k: Any) -> list[Any]:
            return []

        def get_activity(self, *_a: Any, **_k: Any) -> list[Any]:
            return []

        def get_meals(self, *_a: Any, **_k: Any) -> list[Any]:
            return []

        def get_insulin(self, *_a: Any, **_k: Any) -> list[Any]:
            return []

    ctx = SimpleNamespace(store=_Store(), window=(date(2026, 1, 1), date(2026, 3, 1)))
    toolkit = dt.DiscoveryToolkit(ctx)  # type: ignore[arg-type]
    return dt.tool_specs(ctx, toolkit)  # type: ignore[arg-type]


def test_search_evidence_fn_returns_hits_and_pmid_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dexta_intelligence.agents.discovery_tools as dt  # noqa: PLC0415

    stub = _StubBackend(
        [
            EvidenceHit(
                title="Exercise and glucose",
                source="pubmed",
                id="30100000",
                year=2018,
                snippet="Exercise and glucose",
            ),
            EvidenceHit(
                title="Overnight basal",
                source="pubmed",
                id="29200001",
                year=2017,
                snippet="Overnight basal",
            ),
        ]
    )
    monkeypatch.setattr(dt, "evidence_backend", lambda: stub)

    public, numbers = dt._search_evidence({"query": "exercise glucose", "limit": 5})

    assert [h["id"] for h in public["hits"]] == ["30100000", "29200001"]
    # PMID digits + year land in the guard-facing numbers so cites are traceable.
    assert numbers["hit_0"]["pmid"] == 30100000
    assert numbers["hit_0"]["year"] == 2018
    assert numbers["hit_1"]["pmid"] == 29200001
    assert stub.calls == [("exercise glucose", 5)]


def test_search_evidence_fn_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    import dexta_intelligence.agents.discovery_tools as dt  # noqa: PLC0415

    monkeypatch.setattr(dt, "evidence_backend", lambda: _StubBackend([]))
    public, numbers = dt._search_evidence({"query": "  "})
    assert public["hits"] == []
    assert numbers == {}


def test_search_evidence_fn_backend_failure_noted(monkeypatch: pytest.MonkeyPatch) -> None:
    import dexta_intelligence.agents.discovery_tools as dt  # noqa: PLC0415

    def boom() -> Any:
        raise RuntimeError("no backend")

    monkeypatch.setattr(dt, "evidence_backend", boom)
    public, numbers = dt._search_evidence({"query": "glucose"})
    assert public == {"hits": [], "note": "evidence search unavailable"}
    assert numbers == {}


def test_search_evidence_fn_clamps_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import dexta_intelligence.agents.discovery_tools as dt  # noqa: PLC0415

    stub = _StubBackend([])
    monkeypatch.setattr(dt, "evidence_backend", lambda: stub)
    dt._search_evidence({"query": "glucose", "limit": 99})
    assert stub.calls == [("glucose", 8)]
