"""FreeStyle Libre connector - LibreLinkUp follower data via pylibrelinkup.

LibreLinkUp is Abbott's follower service: a *follower* account (not the
sensor wearer's own LibreLink login) that has accepted a sharing invitation
sees the wearer's readings with roughly one-minute freshness. The connector
serves two planes from that service:

- ``current()`` wraps pylibrelinkup's ``latest()`` - the single freshest
  reading, the live MCP "what is my glucose now?" surface.
- ``pull(since)`` merges ``graph()`` (~12 hours of curve data) with
  ``logbook()`` (~2 weeks of glucose-event readings), dedupes on the
  factory timestamp, and returns the batch. **History is capped at ~2
  weeks** - for older data, backfill through Nightscout or a LibreView CSV
  export and let this connector own the live edge.

The module follows the house connector split:

- **Pure conversion** (:func:`measurement_to_event`) takes anything shaped
  like a pylibrelinkup ``GlucoseMeasurement`` (``value_in_mg_per_dl``,
  ``factory_timestamp``, optional ``trend``) and returns a
  :class:`GlucoseEvent`. No I/O, no pylibrelinkup import required - tests
  run on tiny stub objects.
- **LibreConnector** owns the session: lazy pylibrelinkup import (optional
  ``[libre]`` extra), credential/region handling via :class:`LibreConfig`,
  patient resolution, and the graph+logbook merge.

Timestamps: every LibreLinkUp record carries two timestamps - ``Timestamp``
(the wearer's *local* time, naive) and ``FactoryTimestamp`` (UTC, which
pylibrelinkup parses timezone-aware). Conversion uses ``factory_timestamp``
exclusively and rejects naive values loudly, per the house UTC rule.

Trend gotcha: Libre's cloud trend enum is a **subset** of the
Dexcom/Nightscout direction vocabulary - five members, single arrows only::

    Trend.DOWN_FAST  (1)  ->  SingleDown
    Trend.DOWN_SLOW  (2)  ->  FortyFiveDown
    Trend.STABLE     (3)  ->  Flat
    Trend.UP_SLOW    (4)  ->  FortyFiveUp
    Trend.UP_FAST    (5)  ->  SingleUp

``DoubleUp``/``DoubleDown`` can never be produced by this connector, and
values outside the subset (or readings without a trend - ``graph()`` and
``logbook()`` measurements carry none) normalize to ``trend=None``.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import GlucoseEvent, RawEvent

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from dexta_intelligence.config import LibreConfig

__all__ = ["LibreConnector", "LibreMeasurementLike", "measurement_to_event"]

SOURCE = "libre"

_DEDUPE_MARGIN = timedelta(minutes=5)
#: LinkUp followers see ~minute-fresh readings; anything older than this is a
#: gap (sensor warm-up, phone offline), so ``current()`` reports ``None``.
_FRESHNESS_WINDOW = timedelta(minutes=10)

#: pylibrelinkup ``Trend`` value -> Nightscout/Dexcom direction keyword.
#: Deliberately exhaustive: the cloud enum stops at single arrows.
_TREND_DIRECTIONS: dict[int, str] = {
    1: "SingleDown",  # Trend.DOWN_FAST
    2: "FortyFiveDown",  # Trend.DOWN_SLOW
    3: "Flat",  # Trend.STABLE
    4: "FortyFiveUp",  # Trend.UP_SLOW
    5: "SingleUp",  # Trend.UP_FAST
}


@runtime_checkable
class LibreMeasurementLike(Protocol):
    """The slice of pylibrelinkup's ``GlucoseMeasurement`` that conversion needs.

    Duck-typed on purpose: real ``GlucoseMeasurement`` /
    ``GlucoseMeasurementWithTrend`` objects satisfy this, and so does any
    tiny stub with the same two attributes - conversion is testable without
    pylibrelinkup installed. The trend arrow is *not* part of the protocol
    because only ``latest()`` readings carry one; conversion reads it with
    ``getattr``.
    """

    @property
    def value_in_mg_per_dl(self) -> float:
        """Glucose in mg/dL regardless of the account's display units."""
        ...

    @property
    def factory_timestamp(self) -> datetime:
        """Sensor UTC timestamp; pylibrelinkup parses it timezone-aware."""
        ...


class _PatientLike(Protocol):
    """The slice of pylibrelinkup's ``Patient`` the connector uses."""

    @property
    def patient_id(self) -> UUID | str: ...


class _LinkUpClient(Protocol):
    """The pylibrelinkup ``PyLibreLinkUp`` surface the connector uses
    (stubbable in tests)."""

    def authenticate(self) -> None: ...

    def get_patients(self) -> Sequence[_PatientLike]: ...

    def latest(self, patient_identifier: str) -> LibreMeasurementLike | None: ...

    def graph(self, patient_identifier: str) -> Sequence[LibreMeasurementLike]: ...

    def logbook(self, patient_identifier: str) -> Sequence[LibreMeasurementLike]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion - measurement-shaped objects in, typed events out
# ─────────────────────────────────────────────────────────────────────────────


def _trend_to_direction(trend: object) -> str | None:
    """Clamp Libre's five-member trend enum onto the direction vocabulary.

    Accepts the pylibrelinkup ``Trend`` IntEnum or a plain int; anything
    outside the documented 1-5 subset (or a missing/None trend) is ``None``.
    """
    if isinstance(trend, int):
        return _TREND_DIRECTIONS.get(int(trend))
    return None


def measurement_to_event(measurement: LibreMeasurementLike | None) -> GlucoseEvent | None:
    """One pylibrelinkup-shaped measurement -> :class:`GlucoseEvent` (``None`` -> ``None``).

    Uses ``factory_timestamp`` (sensor UTC) - never the wearer-local
    ``Timestamp`` field. A naive timestamp is rejected loudly: pylibrelinkup
    always attaches UTC to factory timestamps, so a naive value here means a
    caller bug, and silently guessing a zone is exactly the class of CGM
    time bug the models refuse to inherit.
    """
    if measurement is None:
        return None
    ts = measurement.factory_timestamp
    if ts.tzinfo is None:
        msg = "naive measurement timestamp rejected: pylibrelinkup factory timestamps carry UTC"
        raise ValueError(msg)
    return GlucoseEvent(
        ts=ts.astimezone(UTC),
        mg_dl=round(measurement.value_in_mg_per_dl),
        trend=_trend_to_direction(getattr(measurement, "trend", None)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Connector - thin session layer over the pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class LibreConnector:
    """Implements both :class:`~dexta_intelligence.connectors.base.Connector`
    and :class:`~dexta_intelligence.connectors.base.RealtimeConnector` against
    the LibreLinkUp follower API via pylibrelinkup.

    Live-first by design: ``current()`` is the realtime MCP surface, and
    ``pull()`` is bounded by LinkUp's ~12h graph / ~2-week logbook horizon
    (see module docstring). pylibrelinkup itself is an optional extra and
    imported lazily, so the base install never pays for it.
    """

    source = SOURCE

    def __init__(self, config: LibreConfig, *, client: _LinkUpClient | None = None) -> None:
        self._config = config
        self._client_instance = client
        self._authenticated = False
        self._patient_id: str | None = None

    # -- Connector protocol ------------------------------------------------------

    def check(self) -> HealthReport:
        """Authenticate and report the latest available reading.

        The first client use runs pylibrelinkup's ``authenticate()``, so a
        bad email/password (or an unaccepted sharing invitation - no
        patients) fails here. Read-only: nothing is mutated.
        """
        try:
            latest = self._client().latest(patient_identifier=self._patient())
        except RuntimeError:
            # Missing optional dependency is a setup bug, not a connectivity
            # result - keep the "pip install" message loud.
            raise
        except Exception as exc:  # pylibrelinkup error tree + requests transport errors
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        event = measurement_to_event(latest)
        return HealthReport(
            ok=True,
            source=self.source,
            detail="LibreLinkUp session established",
            latest_data_ts=event.ts if event is not None else None,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch readings newer than ``since`` (minus a small dedupe margin).

        Merges the ~12h ``graph()`` curve with the ~2-week ``logbook()``
        event history. The two overlap, so readings are deduped on the
        factory timestamp (``source_id = "linkup:<iso-ts>"``) with the
        graph copy winning - it carries the full curve payload. A ``since``
        older than the logbook horizon yields only the last ~2 weeks -
        never an error, never more.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN
        client = self._client()
        patient = self._patient()
        readings = [
            *client.graph(patient_identifier=patient),
            *client.logbook(patient_identifier=patient),
        ]

        by_id: dict[str, tuple[RawEvent, GlucoseEvent]] = {}
        for measurement in readings:
            event = measurement_to_event(measurement)
            if event is None or event.ts < window_start:
                continue
            source_id = f"linkup:{event.ts.isoformat()}"
            if source_id in by_id:
                continue
            by_id[source_id] = (self._raw_event(measurement, event.ts, source_id), event)

        ordered = sorted(by_id.values(), key=lambda pair: pair[1].ts)
        return NormalizedBatch(
            raw=[raw for raw, _ in ordered],
            glucose=[event for _, event in ordered],
        )

    # -- RealtimeConnector protocol ------------------------------------------------

    def current(self) -> GlucoseEvent | None:
        """Live reading from the last 10 minutes - the MCP "glucose now" surface.

        LinkUp's ``latest()`` always returns the connection's most recent
        measurement no matter how stale, so freshness is enforced here:
        ``None`` means the source has nothing recent (warm-up, phone
        offline); connectivity and auth failures raise.
        """
        event = measurement_to_event(self._client().latest(patient_identifier=self._patient()))
        if event is None or datetime.now(tz=UTC) - event.ts > _FRESHNESS_WINDOW:
            return None
        return event

    # -- session plumbing ----------------------------------------------------------

    def _client(self) -> _LinkUpClient:
        if self._client_instance is None:
            self._client_instance = self._build_client()
        if not self._authenticated:
            self._client_instance.authenticate()
            self._authenticated = True
        return self._client_instance

    def _build_client(self) -> _LinkUpClient:
        try:
            # Deliberately lazy (via importlib, so type-checking never needs
            # the package): Libre support is an optional extra and the rest
            # of the system must be importable without it.
            linkup = importlib.import_module("pylibrelinkup")
        except ImportError as exc:  # pragma: no cover - import-path guard
            msg = (
                "FreeStyle Libre support is not installed. "
                "Install it with: pip install 'dexta-intelligence[libre]'"
            )
            raise RuntimeError(msg) from exc

        client = linkup.PyLibreLinkUp(
            email=self._config.email,
            password=self._config.password,
            api_url=linkup.APIUrl.from_string(self._config.region.value),
        )
        return cast("_LinkUpClient", client)

    def _patient(self) -> str:
        """Configured patient id, or the account's first shared patient."""
        if self._config.patient_id:
            return self._config.patient_id
        if self._patient_id is None:
            patients = self._client().get_patients()
            if not patients:
                msg = (
                    "LibreLinkUp account has no patient connections; accept a "
                    "sharing invitation in the LibreLinkUp app first"
                )
                raise ValueError(msg)
            self._patient_id = str(patients[0].patient_id)
        return self._patient_id

    def _raw_event(
        self, measurement: LibreMeasurementLike, ts: datetime, source_id: str
    ) -> RawEvent:
        # Real GlucoseMeasurement objects are pydantic models - keep the
        # verbatim record; stubs get a synthesized minimal payload.
        dump = getattr(measurement, "model_dump", None)
        payload = dump(mode="json", by_alias=True) if callable(dump) else None
        if not isinstance(payload, dict):
            trend = getattr(measurement, "trend", None)
            payload = {
                "ValueInMgPerDl": measurement.value_in_mg_per_dl,
                "TrendArrow": int(trend) if isinstance(trend, int) else None,
                "FactoryTimestamp": ts.isoformat(),
            }
        return RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=payload)
