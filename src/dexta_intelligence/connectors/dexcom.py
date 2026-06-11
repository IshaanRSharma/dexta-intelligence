"""Dexcom Share connector - live CGM readings via pydexcom.

.. warning:: **The Share API caps history at ~24 hours (288 readings).**
   ``pull(since)`` can never reach further back than that, no matter what
   ``since`` says. This connector exists for *live and recent* data - the
   ``current()`` MCP surface and keeping the last day fresh. For real
   history, backfill through Nightscout or a Dexcom Clarity CSV export and
   let this connector take over at the live edge.

The module follows the house connector split:

- **Pure conversion** (:func:`reading_to_event`) takes anything shaped like
  a pydexcom ``GlucoseReading`` (``value``, ``trend_direction``,
  ``datetime``) and returns a :class:`GlucoseEvent`. No I/O, no pydexcom
  import required - tests run on tiny stub objects.
- **DexcomConnector** owns the session: lazy pydexcom import (optional
  ``[dexcom]`` extra), credential handling via :class:`DexcomConfig`, and
  the ``since`` -> ``minutes`` window arithmetic against the 24h cap.

Trend mapping: pydexcom's ``trend_direction`` uses the *same* vocabulary as
the Nightscout ``direction`` field already stored in ``GlucoseEvent.trend``,
so informative directions pass straight through::

    DoubleUp        rising quickly      (> +3 mg/dL/min)
    SingleUp        rising              (+2 to +3 mg/dL/min)
    FortyFiveUp     rising slightly     (+1 to +2 mg/dL/min)
    Flat            steady              (-1 to +1 mg/dL/min)
    FortyFiveDown   falling slightly    (-2 to -1 mg/dL/min)
    SingleDown      falling             (-3 to -2 mg/dL/min)
    DoubleDown      falling quickly     (< -3 mg/dL/min)

The non-informative directions (``None``, ``NotComputable``,
``RateOutOfRange``) normalize to ``trend=None``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import GlucoseEvent, RawEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.config import DexcomConfig

__all__ = ["DexcomConnector", "DexcomReadingLike", "reading_to_event"]

SOURCE = "dexcom"

_DEDUPE_MARGIN = timedelta(minutes=5)
#: Hard Share API limits: at most 1440 minutes (~24h) and 288 readings per call.
_MAX_MINUTES = 1440
_MAX_COUNT = 288

#: pydexcom trend_direction values that carry information; identical to the
#: Nightscout ``direction`` vocabulary, so they map 1:1 onto GlucoseEvent.trend.
_INFORMATIVE_TRENDS = frozenset(
    {"DoubleUp", "SingleUp", "FortyFiveUp", "Flat", "FortyFiveDown", "SingleDown", "DoubleDown"}
)


@runtime_checkable
class DexcomReadingLike(Protocol):
    """The slice of pydexcom's ``GlucoseReading`` that conversion needs.

    Duck-typed on purpose: real ``GlucoseReading`` objects satisfy this, and
    so does any tiny stub with the same three attributes - conversion is
    testable without pydexcom installed.
    """

    @property
    def value(self) -> int:
        """Glucose in mg/dL."""
        ...

    @property
    def trend_direction(self) -> str:
        """Dexcom trend keyword, e.g. ``"Flat"`` or ``"FortyFiveDown"``."""
        ...

    @property
    def datetime(self) -> datetime:
        """Reading timestamp; pydexcom always supplies it timezone-aware."""
        ...


class _ShareClient(Protocol):
    """The pydexcom ``Dexcom`` surface the connector uses (stubbable in tests)."""

    def get_glucose_readings(
        self, minutes: int = ..., max_count: int = ...
    ) -> Sequence[DexcomReadingLike]: ...

    def get_latest_glucose_reading(self) -> DexcomReadingLike | None: ...

    def get_current_glucose_reading(self) -> DexcomReadingLike | None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion - reading-shaped objects in, typed events out
# ─────────────────────────────────────────────────────────────────────────────


def reading_to_event(reading: DexcomReadingLike | None) -> GlucoseEvent | None:
    """One pydexcom-shaped reading -> :class:`GlucoseEvent` (``None`` -> ``None``).

    Timestamps are converted to UTC. A naive timestamp is rejected loudly:
    pydexcom always attaches the Share API's UTC offset, so a naive value
    here means a caller bug, and silently guessing a zone is exactly the
    class of CGM time bug the models refuse to inherit.
    """
    if reading is None:
        return None
    ts = reading.datetime
    if ts.tzinfo is None:
        msg = "naive reading timestamp rejected: pydexcom datetimes carry a UTC offset"
        raise ValueError(msg)
    trend = reading.trend_direction
    return GlucoseEvent(
        ts=ts.astimezone(UTC),
        mg_dl=int(reading.value),
        trend=trend if trend in _INFORMATIVE_TRENDS else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Connector - thin session layer over the pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class DexcomConnector:
    """Implements both :class:`~dexta_intelligence.connectors.base.Connector`
    and :class:`~dexta_intelligence.connectors.base.RealtimeConnector` against
    the Dexcom Share API via pydexcom.

    Live-first by design: ``current()`` is the realtime MCP surface, and
    ``pull()`` is bounded by Share's ~24h history cap (see module docstring).
    pydexcom itself is an optional extra and imported lazily, so the base
    install never pays for it.
    """

    source = SOURCE

    def __init__(self, config: DexcomConfig, *, client: _ShareClient | None = None) -> None:
        self._config = config
        self._client_instance = client

    # -- Connector protocol ----------------------------------------------------

    def check(self) -> HealthReport:
        """Establish a Share session and report the latest available reading.

        Session creation *is* the auth probe: pydexcom authenticates inside
        the ``Dexcom`` constructor, so a bad username/password fails here.
        """
        try:
            latest = self._client().get_latest_glucose_reading()
        except RuntimeError:
            # Missing optional dependency is a setup bug, not a connectivity
            # result - keep the "pip install" message loud.
            raise
        except Exception as exc:  # pydexcom error tree + requests transport errors
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        event = reading_to_event(latest)
        return HealthReport(
            ok=True,
            source=self.source,
            detail="Dexcom Share session established",
            latest_data_ts=event.ts if event is not None else None,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch readings newer than ``since`` (minus a small dedupe margin).

        **Share caps history at ~24h**: the request window is clamped to
        1440 minutes / 288 readings, so a ``since`` older than a day yields
        only the last day - never an error, never more. Use Nightscout or a
        CSV import for anything older.

        Share records carry no provider id; the reading timestamp is the
        idempotency key (``source_id = "share:<iso-ts>"``), which is safe
        because Dexcom emits at most one reading per 5-minute slot.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN
        wanted = datetime.now(tz=UTC) - window_start
        minutes = min(max(math.ceil(wanted.total_seconds() / 60), 1), _MAX_MINUTES)
        readings = self._client().get_glucose_readings(minutes=minutes, max_count=_MAX_COUNT)

        raw_events: list[RawEvent] = []
        glucose: list[GlucoseEvent] = []
        for reading in readings:
            event = reading_to_event(reading)
            if event is None or event.ts < window_start:
                continue
            raw_events.append(self._raw_event(reading, event.ts))
            glucose.append(event)
        return NormalizedBatch(raw=raw_events, glucose=glucose)

    # -- RealtimeConnector protocol ----------------------------------------------

    def current(self) -> GlucoseEvent | None:
        """Live reading from the last 10 minutes - the MCP "glucose now" surface.

        ``None`` means the source has nothing recent (warm-up, signal gap);
        connectivity and auth failures raise.
        """
        return reading_to_event(self._client().get_current_glucose_reading())

    # -- session plumbing --------------------------------------------------------

    def _client(self) -> _ShareClient:
        if self._client_instance is None:
            self._client_instance = self._build_client()
        return self._client_instance

    def _build_client(self) -> _ShareClient:
        try:
            # Deliberately lazy: Dexcom support is an optional extra and the
            # rest of the system must be importable without it.
            from pydexcom import Dexcom, Region  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import-path guard
            msg = (
                "Dexcom Share support is not installed. "
                "Install it with: pip install 'dexta-intelligence[dexcom]'"
            )
            raise RuntimeError(msg) from exc

        return Dexcom(
            username=self._config.username or None,
            password=self._config.password,
            region=Region.OUS if self._config.ous else Region.US,
        )

    def _raw_event(self, reading: DexcomReadingLike, ts: datetime) -> RawEvent:
        payload = getattr(reading, "json", None)  # real GlucoseReading keeps the API dict
        if not isinstance(payload, dict):
            payload = {
                "Value": reading.value,
                "Trend": reading.trend_direction,
                "DT": ts.isoformat(),
            }
        return RawEvent(
            source=self.source,
            source_id=f"share:{ts.isoformat()}",
            source_ts=ts,
            payload=payload,
        )
