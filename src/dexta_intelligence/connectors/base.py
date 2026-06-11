"""Connector contract — batch ingest from any sensor source.

A connector pulls provider records and returns them as immutable
:class:`~dexta_intelligence.models.RawEvent` rows plus normalized timeline
events. The sync workflow (``workflows.sync``) owns persistence and
watermarks; connectors own *only* provider I/O and normalization.

Idempotency is structural: raw events carry ``(source, source_id)`` and the
store skips duplicates, so re-running a connector over an overlapping window
is always safe. A file upload is just a degenerate connector whose ``pull``
reads the file once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

    from dexta_intelligence.models import (
        ActivityEvent,
        DeviceEvent,
        GlucoseEvent,
        InsulinEvent,
        MealEvent,
        PredictionEvent,
        RawEvent,
        RecoveryEvent,
        SleepEvent,
    )

__all__ = ["Connector", "HealthReport", "NormalizedBatch", "RealtimeConnector"]


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Result of a connectivity/auth check, surfaced by ``dexta doctor``."""

    ok: bool
    source: str
    detail: str = ""
    latest_data_ts: datetime | None = None


@dataclass(frozen=True, slots=True)
class NormalizedBatch:
    """One pull's worth of data: verbatim raws + their normalized projections.

    ``raw`` and the typed lists are produced together so provenance
    (``raw_event_id`` linkage) can be wired by the sync workflow after the
    raw rows are assigned ids.
    """

    raw: list[RawEvent] = field(default_factory=list)
    glucose: list[GlucoseEvent] = field(default_factory=list)
    insulin: list[InsulinEvent] = field(default_factory=list)
    meals: list[MealEvent] = field(default_factory=list)
    activity: list[ActivityEvent] = field(default_factory=list)
    sleep: list[SleepEvent] = field(default_factory=list)
    recovery: list[RecoveryEvent] = field(default_factory=list)
    device: list[DeviceEvent] = field(default_factory=list)
    predictions: list[PredictionEvent] = field(default_factory=list)


@runtime_checkable
class Connector(Protocol):
    """One per source: nightscout, dexcom, libre, whoop, file upload."""

    source: str

    def check(self) -> HealthReport:
        """Cheap reachability + auth probe. Must not mutate anything."""
        ...

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch and normalize everything newer than ``since``.

        Must paginate internally, must tolerate provider hiccups by raising
        (the sync workflow handles retry), and must never return events older
        than ``since`` minus a small dedupe margin.
        """
        ...


@runtime_checkable
class RealtimeConnector(Connector, Protocol):
    """A :class:`Connector` whose source can also serve the freshest reading.

    This is the surface live MCP tools call: a "what is my glucose right
    now?" tool invokes :meth:`current` directly instead of round-tripping
    through the sync workflow and the store. Batch-only sources (file
    uploads, exports) implement plain :class:`Connector`; sources with a
    live API (Dexcom Share, Nightscout) can implement this as well.
    """

    def current(self) -> GlucoseEvent | None:
        """Return the single freshest glucose reading, or ``None`` if the
        source has nothing recent (sensor warm-up, gap, follower delay).

        Must hit the provider live - never a cache - and must not mutate
        anything. Connectivity failures raise; "no recent data" is ``None``.
        """
        ...
