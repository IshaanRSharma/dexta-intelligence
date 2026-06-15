"""Tandem t:slim X2 connector - direct pump data via the t:connect cloud.

.. warning:: **UNOFFICIAL, reverse-engineered API.** This connector talks to
   Tandem's t:connect cloud through `tconnectsync
   <https://github.com/jwoglom/tconnectsync>`_, which scrapes private,
   undocumented endpoints used by the t:connect web and Android apps. It is
   opt-in (the ``[tandem]`` extra), strictly read-only, and **may break without
   notice** if Tandem changes its backend. No write/command path exists or ever
   will - this only ever *reads* therapy data.

The headline of this connector is what it removes: for a Control-IQ user it
delivers real pump data - boluses, basal/temp-basal, suspends, and bolus-wizard
carbs - **without the Nightscout hop**. You no longer need a Nightscout server
and an uploader bridge to get insulin into dexta; t:connect credentials are
enough.

The module follows the house connector split:

- **Pure conversion** (:func:`bolus_to_events`, :func:`basal_to_event`) takes
  tconnectsync-shaped therapy-timeline records - the ``Bolus`` dataclass (all
  string fields) and the basal/temp-basal dicts the ControlIQ parser emits -
  and returns typed :class:`InsulinEvent` / :class:`MealEvent` objects. No I/O,
  no tconnectsync import required: tests run on tiny stubs and plain dicts.
- **TandemConnector** owns the session: lazy tconnectsync import (optional
  ``[tandem]`` extra), credential/region handling via :class:`TandemConfig`,
  and the ``since`` -> ``therapy_timeline`` window.

Event mapping
-------------
- **Bolus** (``Bolus`` record) -> :class:`InsulinEvent` ``kind=BOLUS`` with
  ``units`` from ``insulin`` (delivered), at ``completion_time`` (falling back
  to ``request_time``). A bolus that also carries a bolus-wizard ``carbs`` entry
  additionally yields a :class:`MealEvent` (``carbs_g``) at the same time, so
  one record can produce two events - exactly like a Nightscout "Meal Bolus".
- **Basal** dict, by ``delivery_type``:
    - ``TempRate``  -> ``kind=TEMP_BASAL`` (``duration_min``; ``units`` =
      ``basal_rate`` x ``duration/60`` when both are present - the *scheduled*
      delivery, best-effort, same caveat as the Nightscout connector).
    - ``Algorithm`` -> ``kind=TEMP_BASAL`` as well: Control-IQ's automatic
      adjustments are closed-loop temp basals, recorded with ``automatic=True``.
    - ``Profile``   -> ``kind=BASAL`` (scheduled-rate change).
    - ``Suspension``-> ``kind=SUSPEND`` (``duration_min``; no units).

Timestamps: t:connect events carry **device-local time**. tconnectsync's
parsers attach the user's configured timezone before formatting, so the strings
it returns are ISO 8601 *with an offset* - which conversion normalizes to UTC.
A naive timestamp (no offset, no ``Z``) is **rejected loudly** with
``ValueError`` per the house UTC rule: silently guessing a zone is exactly the
class of pump/CGM time bug the models refuse to inherit, and a naive value here
means tconnectsync had no timezone to apply.

.. note:: The conversion below is written against tconnectsync's **documented
   record shapes** (the ``Bolus`` dataclass and the ControlIQ basal dicts). It
   has not been exercised against a live t:connect account in this environment;
   field names, ``delivery_type`` values, and the timezone behaviour of the
   parsed timestamps **must be validated against real credentials before
   release**.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import (
    InsulinEvent,
    InsulinKind,
    MealEvent,
    RawEvent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.config import TandemConfig

__all__ = [
    "BolusLike",
    "TandemConnector",
    "basal_to_event",
    "bolus_to_events",
]

SOURCE = "tandem"

_DEDUPE_MARGIN = timedelta(minutes=5)
#: t:connect's therapy_timeline takes whole-day bounds (``YYYY-MM-DD``).
_DATE_FMT = "%Y-%m-%d"

#: tconnectsync basal ``delivery_type`` -> our InsulinKind. ``TempRate`` and
#: ``Algorithm`` are both temp basals (Control-IQ's automatic adjustments are
#: closed-loop temp basals); ``Profile`` is a scheduled-rate change; an empty
#: delivery_type (legacy/ws2 path) also reads as a scheduled basal.
_BASAL_KINDS: dict[str, InsulinKind] = {
    "TempRate": InsulinKind.TEMP_BASAL,
    "Algorithm": InsulinKind.TEMP_BASAL,
    "Profile": InsulinKind.BASAL,
    "Suspension": InsulinKind.SUSPEND,
    "": InsulinKind.BASAL,
}
#: delivery_types that are algorithm-issued (Control-IQ closed loop).
_AUTOMATIC_DELIVERY = frozenset({"Algorithm"})


@runtime_checkable
class BolusLike(Protocol):
    """The slice of tconnectsync's ``Bolus`` dataclass conversion needs.

    Duck-typed on purpose: the real ``Bolus`` (all-string fields) satisfies
    this, and so does any tiny stub with the same attributes - conversion is
    testable without tconnectsync installed. All fields are strings in the
    provider record; conversion does the numeric/timestamp coercion.
    """

    @property
    def insulin(self) -> str:
        """Units of insulin actually delivered, as a string."""
        ...

    @property
    def carbs(self) -> str:
        """Bolus-wizard carb entry in grams, as a string (``""``/``"0"`` if none)."""
        ...

    @property
    def completion_time(self) -> str:
        """ISO 8601 completion timestamp with offset (device-local zone applied)."""
        ...

    @property
    def request_time(self) -> str:
        """ISO 8601 request timestamp; the fallback when completion is absent."""
        ...


class _ControlIQApi(Protocol):
    """The ControlIQ surface the connector uses (stubbable in tests)."""

    def therapy_timeline(self, start_date: str, end_date: str) -> dict[str, Any]: ...


class _TConnectClient(Protocol):
    """The tconnectsync ``TConnectApi`` surface the connector uses.

    Only the ControlIQ timeline is consumed; the ws2/android sub-apis are not
    touched (they are CSV/legacy paths). Stubbable in tests via an injected
    object exposing ``controliq.therapy_timeline``.
    """

    @property
    def controliq(self) -> _ControlIQApi: ...


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion - tconnectsync-shaped records in, typed events out
# ─────────────────────────────────────────────────────────────────────────────


def _parse_ts(value: object) -> datetime | None:
    """ISO 8601 string -> aware UTC. ``None``/empty -> ``None``; naive -> raise.

    tconnectsync applies the user's timezone before formatting, so every real
    timestamp carries an offset. A naive value (no offset, no ``Z``) means no
    zone was available - rejected loudly rather than silently assumed.
    """
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        msg = (
            "naive tandem timestamp rejected: t:connect events are device-local "
            "and tconnectsync attaches an offset; a naive value has no timezone"
        )
        raise ValueError(msg)
    return parsed.astimezone(UTC)


def _as_float(value: object) -> float | None:
    """tconnectsync stores numerics as strings; coerce, tolerating blanks."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def bolus_to_events(bolus: BolusLike) -> list[InsulinEvent | MealEvent]:
    """One tconnectsync ``Bolus`` record -> insulin (+ optional meal) events.

    Emits an :class:`InsulinEvent` (``kind=BOLUS``, ``units`` = delivered
    insulin) whenever a positive amount was delivered, and - because a
    bolus-wizard entry rides on the same record - an additional
    :class:`MealEvent` whenever positive carbs are present. A record with
    neither (a zero/aborted bolus, no carbs) yields ``[]``.

    Timestamp is ``completion_time`` (when the delivery actually finished),
    falling back to ``request_time``. A naive timestamp is rejected (see
    :func:`_parse_ts`); a record with no usable timestamp yields ``[]``.
    """
    ts = _parse_ts(bolus.completion_time) or _parse_ts(bolus.request_time)
    if ts is None:
        return []

    events: list[InsulinEvent | MealEvent] = []
    units = _as_float(bolus.insulin)
    if units is not None and units > 0:
        events.append(InsulinEvent(ts=ts, kind=InsulinKind.BOLUS, units=units))

    carbs = _as_float(bolus.carbs)
    if carbs is not None and carbs > 0:
        events.append(MealEvent(ts=ts, carbs_g=carbs))

    return events


def basal_to_event(basal: dict[str, Any]) -> InsulinEvent | None:
    """One tconnectsync ControlIQ basal dict -> :class:`InsulinEvent`.

    Maps ``delivery_type`` onto an :class:`InsulinKind` (see
    :data:`_BASAL_KINDS`). For temp basals an absolute scheduled delivery is
    derived as ``basal_rate x duration_mins/60`` when both are present - the
    *scheduled* units, best-effort and possibly an overstatement if the temp
    was cancelled early (same documented caveat as the Nightscout connector).
    ``BASAL``/``SUSPEND`` carry no units.

    Unknown ``delivery_type`` values and records without a timestamp yield
    ``None``. A naive timestamp is rejected (see :func:`_parse_ts`).
    """
    ts = _parse_ts(basal.get("time"))
    if ts is None:
        return None

    delivery_type = str(basal.get("delivery_type", ""))
    kind = _BASAL_KINDS.get(delivery_type)
    if kind is None:
        return None

    duration_min = _as_float(basal.get("duration_mins"))
    automatic = True if delivery_type in _AUTOMATIC_DELIVERY else None

    units: float | None = None
    if kind is InsulinKind.TEMP_BASAL:
        rate = _as_float(basal.get("basal_rate"))
        if rate is not None and duration_min is not None:
            units = rate * duration_min / 60.0

    return InsulinEvent(
        ts=ts,
        kind=kind,
        units=units,
        duration_min=duration_min if kind is not InsulinKind.BASAL else None,
        automatic=automatic,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Connector - thin session layer over the pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class TandemConnector:
    """Implements the :class:`~dexta_intelligence.connectors.base.Connector`
    protocol against the Tandem t:connect cloud via tconnectsync.

    Batch-only (not :class:`RealtimeConnector`): t:connect is the pump's upload
    target, not a live stream - readings lag the device, so there is no
    meaningful "right now" surface. tconnectsync is an optional extra and
    imported lazily, so the base install never pays for it.
    """

    source = SOURCE

    def __init__(self, config: TandemConfig, *, client: _TConnectClient | None = None) -> None:
        self._config = config
        self._client_instance = client

    # -- Connector protocol ----------------------------------------------------

    def check(self) -> HealthReport:
        """Establish a t:connect session and probe a small recent window.

        Building the client authenticates against t:connect, so bad
        credentials fail here. Read-only: a one-day therapy_timeline is
        fetched purely to prove reachability and surface the latest event.
        """
        try:
            now = datetime.now(tz=UTC)
            timeline = self._client().controliq.therapy_timeline(
                start_date=(now - timedelta(days=1)).strftime(_DATE_FMT),
                end_date=now.strftime(_DATE_FMT),
            )
        except RuntimeError:
            # Missing optional dependency is a setup bug, not a connectivity
            # result - keep the "pip install" message loud.
            raise
        except Exception as exc:  # tconnectsync error tree + requests transport errors
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        latest = self._latest_ts(timeline)
        return HealthReport(
            ok=True,
            source=self.source,
            detail="t:connect session established",
            latest_data_ts=latest,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch the therapy timeline newer than ``since`` (minus a dedupe margin).

        The ControlIQ timeline takes whole-day bounds, so the window is the
        day of ``window_start`` through today; events before ``window_start``
        are filtered out after normalization. Raises on provider hiccups - the
        sync workflow owns retries.

        source_ids are stable per event for idempotency:
        ``tandem:bolus:<iso-ts>``, ``tandem:carbs:<iso-ts>``,
        ``tandem:<kind>:<iso-ts>`` for basals.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN
        now = datetime.now(tz=UTC)
        timeline = self._client().controliq.therapy_timeline(
            start_date=window_start.strftime(_DATE_FMT),
            end_date=now.strftime(_DATE_FMT),
        )

        raw_events: list[RawEvent] = []
        insulin: list[InsulinEvent] = []
        meals: list[MealEvent] = []

        for bolus in self._boluses(timeline):
            for event in bolus_to_events(bolus):
                if event.ts < window_start:
                    continue
                if isinstance(event, InsulinEvent):
                    source_id = f"tandem:bolus:{event.ts.isoformat()}"
                    insulin.append(event)
                else:
                    source_id = f"tandem:carbs:{event.ts.isoformat()}"
                    meals.append(event)
                raw_events.append(self._raw_event(bolus, event.ts, source_id))

        for basal in self._basals(timeline):
            basal_event = basal_to_event(basal)
            if basal_event is None or basal_event.ts < window_start:
                continue
            source_id = f"tandem:{basal_event.kind.value}:{basal_event.ts.isoformat()}"
            insulin.append(basal_event)
            raw_events.append(self._raw_event(basal, basal_event.ts, source_id))

        return NormalizedBatch(raw=raw_events, insulin=insulin, meals=meals)

    # -- timeline shape helpers ------------------------------------------------

    @staticmethod
    def _boluses(timeline: dict[str, Any]) -> Sequence[BolusLike]:
        events = timeline.get("bolus")
        return events if isinstance(events, list) else []

    @staticmethod
    def _basals(timeline: dict[str, Any]) -> Sequence[dict[str, Any]]:
        events = timeline.get("basal")
        if isinstance(events, dict):  # ControlIQ nests basal under {"events": [...]}
            events = events.get("events")
        if not isinstance(events, list):
            return []
        return [e for e in events if isinstance(e, dict)]

    def _latest_ts(self, timeline: dict[str, Any]) -> datetime | None:
        stamps: list[datetime] = []
        for bolus in self._boluses(timeline):
            stamps.extend(e.ts for e in bolus_to_events(bolus))
        for basal in self._basals(timeline):
            event = basal_to_event(basal)
            if event is not None:
                stamps.append(event.ts)
        return max(stamps) if stamps else None

    # -- session plumbing ------------------------------------------------------

    def _client(self) -> _TConnectClient:
        if self._client_instance is None:
            self._client_instance = self._build_client()
        return self._client_instance

    def _build_client(self) -> _TConnectClient:
        try:
            # Deliberately lazy (via importlib, so type-checking never needs the
            # package): Tandem support is an optional extra and the rest of the
            # system must be importable without it.
            tconnectsync = importlib.import_module("tconnectsync")
        except ImportError as exc:  # pragma: no cover - import-path guard
            msg = (
                "Tandem t:connect support is not installed. "
                "Install it with: pip install 'dexta-intelligence[tandem]'"
            )
            raise RuntimeError(msg) from exc

        client = tconnectsync.TConnectApi(
            email=self._config.email,
            password=self._config.password,
        )
        return cast("_TConnectClient", client)

    def _raw_event(self, record: object, ts: datetime, source_id: str) -> RawEvent:
        # Real Bolus records are dataclasses; basals are plain dicts. Keep the
        # verbatim record where we can, else synthesize a minimal payload.
        payload: dict[str, Any]
        if isinstance(record, dict):
            payload = record
        else:
            as_dict = getattr(record, "__dict__", None)
            payload = dict(as_dict) if isinstance(as_dict, dict) else {"repr": repr(record)}
        return RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=payload)
