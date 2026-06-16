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
    "PROFILE_SOURCE_ID",
    "BolusLike",
    "TandemConnector",
    "basal_to_event",
    "bolus_to_events",
    "format_insulin_profile",
    "pump_events_to_batch",
]

SOURCE = "tandem"
PROFILE_SOURCE_ID = "tandem:profile:active"

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


class _TandemSourceApi(Protocol):
    """The Tandem Source surface used for auth, device selection, and pulls."""

    def pump_event_metadata(self) -> list[dict[str, Any]]: ...

    def pump_events(
        self,
        tconnect_device_id: str,
        min_date: str | None = None,
        max_date: str | None = None,
        *,
        fetch_all_event_types: bool = False,
    ) -> Any: ...


class _TConnectClient(Protocol):
    """The tconnectsync ``TConnectApi`` surface the connector uses.

    ``check`` and ``pull`` both use Tandem Source (``tandemsource``), which is
    what tconnectsync v2+ relies on after the legacy t:connect web login was
    retired.
    """

    @property
    def tandemsource(self) -> _TandemSourceApi: ...


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


#: Tandem Source basal ``commandedRateSourceRaw`` values that map to temp basals.
_TEMP_BASAL_SOURCES = frozenset({2, 3, 4})
#: Subset issued by Control-IQ closed loop.
_ALGO_BASAL_SOURCES = frozenset({3, 4})
#: Default segment length when the next basal delivery event is unknown.
_DEFAULT_BASAL_SEGMENT = timedelta(minutes=5)


def _event_ts(value: object) -> datetime:
    """Normalize a tconnectsync event timestamp (arrow or datetime) to UTC."""
    if hasattr(value, "datetime"):  # arrow.Arrow
        inner = value.datetime
        if isinstance(inner, datetime):
            return inner.astimezone(UTC)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            msg = "naive tandem event timestamp rejected"
            raise ValueError(msg)
        return value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            msg = "naive tandem event timestamp rejected"
            raise ValueError(msg)
        return parsed.astimezone(UTC)
    msg = f"unsupported tandem event timestamp: {value!r}"
    raise ValueError(msg)


def pump_events_to_batch(  # noqa: PLR0912, PLR0915 - one branch per tconnect event type
    events: Sequence[Any],
    *,
    window_start: datetime,
    source: str = SOURCE,
    end: datetime | None = None,
) -> NormalizedBatch:
    """Convert decoded Tandem Source pump events into normalized timeline rows.

    Mirrors tconnectsync's Tandem Source processors (bolus grouping by
    ``bolusid``, basal segments from consecutive ``LidBasalDelivery`` events,
    ``LidPumpingSuspended`` -> suspend). Duck-typed on event class names so
    tests can stub without importing tconnectsync.
    """
    window_end = end or datetime.now(tz=UTC)
    sorted_events = sorted(events, key=lambda e: _event_ts(e.eventTimestamp))

    raw_events: list[RawEvent] = []
    insulin: list[InsulinEvent] = []
    meals: list[MealEvent] = []

    bolus_by_id: dict[int, dict[str, Any]] = {}
    basal_deliveries: list[Any] = []
    suspends: list[Any] = []

    for event in sorted_events:
        name = type(event).__name__
        if name.endswith("BolusCompleted") or name == "LidBolexCompleted":
            slot = bolus_by_id.setdefault(int(event.bolusid), {})
            slot["completed"] = event
        elif name.endswith("BolusRequestedMsg1") or name.endswith("BolusRequested1"):
            slot = bolus_by_id.setdefault(int(event.bolusid), {})
            slot["requested1"] = event
        elif name.endswith("BasalDelivery"):
            basal_deliveries.append(event)
        elif name.endswith("PumpingSuspended"):
            suspends.append(event)

    for bolus_id, parts in bolus_by_id.items():
        completed = parts.get("completed")
        if completed is None:
            continue
        ts = _event_ts(completed.eventTimestamp)
        if ts < window_start or ts > window_end:
            continue
        units = _as_float(getattr(completed, "insulindelivered", None))
        requested1 = parts.get("requested1")
        carbs_g = _as_float(getattr(requested1, "carbamount", None)) if requested1 else None
        seq = getattr(completed, "seqNum", bolus_id)

        if units is not None and units > 0:
            bolus_event = InsulinEvent(ts=ts, kind=InsulinKind.BOLUS, units=units)
            insulin.append(bolus_event)
            raw_events.append(
                _raw_event_from_payload(
                    source,
                    f"tandem:bolus:{seq}",
                    ts,
                    _event_payload(completed, requested1),
                )
            )
        if carbs_g is not None and carbs_g > 0:
            meal = MealEvent(ts=ts, carbs_g=carbs_g)
            meals.append(meal)
            raw_events.append(
                _raw_event_from_payload(
                    source,
                    f"tandem:carbs:{seq}",
                    ts,
                    _event_payload(completed, requested1),
                )
            )

    if basal_deliveries:
        segments: list[tuple[datetime, timedelta, Any]] = []
        for idx, event in enumerate(basal_deliveries):
            start = _event_ts(event.eventTimestamp)
            if idx + 1 < len(basal_deliveries):
                nxt = _event_ts(basal_deliveries[idx + 1].eventTimestamp)
                duration = nxt - start
            else:
                duration = min(_DEFAULT_BASAL_SEGMENT, window_end - start)
            if duration.total_seconds() <= 0:
                duration = _DEFAULT_BASAL_SEGMENT
            segments.append((start, duration, event))

        for start, duration, event in segments:
            if start < window_start or start > window_end:
                continue
            basal_event = _basal_delivery_to_event(start, duration, event)
            if basal_event is None:
                continue
            seq = getattr(event, "seqNum", start.isoformat())
            insulin.append(basal_event)
            raw_events.append(
                _raw_event_from_payload(
                    source,
                    f"tandem:{basal_event.kind.value}:{seq}",
                    basal_event.ts,
                    _event_payload(event),
                )
            )

    for event in suspends:
        ts = _event_ts(event.eventTimestamp)
        if ts < window_start or ts > window_end:
            continue
        timeout = _as_float(getattr(event, "rpatimeout", None))
        suspend = InsulinEvent(
            ts=ts,
            kind=InsulinKind.SUSPEND,
            duration_min=timeout,
        )
        seq = getattr(event, "seqNum", ts.isoformat())
        insulin.append(suspend)
        raw_events.append(
            _raw_event_from_payload(source, f"tandem:suspend:{seq}", ts, _event_payload(event))
        )

    return NormalizedBatch(raw=raw_events, insulin=insulin, meals=meals)


def _basal_delivery_to_event(
    start: datetime, duration: timedelta, event: Any
) -> InsulinEvent | None:
    source_raw = int(getattr(event, "commandedRateSourceRaw", -1))
    if source_raw == 0:
        return None
    rate_milli = _as_float(getattr(event, "commandedRate", None))
    if rate_milli is None or rate_milli <= 0:
        return None
    duration_min = duration.total_seconds() / 60.0
    units = (rate_milli / 1000.0) * duration_min / 60.0
    if units <= 0:
        return None
    if source_raw == 1:
        kind = InsulinKind.BASAL
        automatic = None
    elif source_raw in _TEMP_BASAL_SOURCES:
        kind = InsulinKind.TEMP_BASAL
        automatic = source_raw in _ALGO_BASAL_SOURCES
    else:
        return None
    return InsulinEvent(
        ts=start,
        kind=kind,
        units=units,
        duration_min=duration_min,
        automatic=automatic,
    )


def _event_payload(*parts: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        if part is None:
            continue
        todict = getattr(part, "todict", None)
        if callable(todict):
            out.update(todict())
        elif isinstance(part, dict):
            out.update(part)
        else:
            as_dict = getattr(part, "__dict__", None)
            if isinstance(as_dict, dict):
                out.update({k: v for k, v in as_dict.items() if not k.startswith("_")})
    return out


def _raw_event_from_payload(
    source: str, source_id: str, ts: datetime, payload: dict[str, Any]
) -> RawEvent:
    return RawEvent(source=source, source_id=source_id, source_ts=ts, payload=payload)


def _minutes_to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _segment_is_empty(seg: dict[str, Any]) -> bool:
    keys = ("startTime", "basalRate", "isf", "carbRatio", "targetBg")
    return all(int(seg.get(k, 0) or 0) == 0 for k in keys)


def format_insulin_profile(
    settings: dict[str, Any],
    *,
    pump_serial: str | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Convert Tandem Source pump settings into a readable insulin profile snapshot.

    Pure function — no I/O. Values mirror tconnectsync's Nightscout profile
    mapping (basal/carb-ratio in pump milliunits, ISF/target in mg/dL).
    """
    profiles_block = settings.get("profiles") or {}
    active_idp = int(profiles_block.get("activeIdp", 0) or 0)
    profiles_in = profiles_block.get("profile") or []
    formatted_profiles: list[dict[str, Any]] = []
    active_name: str | None = None

    for profile in profiles_in:
        if not isinstance(profile, dict):
            continue
        idp = int(profile.get("idp", 0) or 0)
        name = str(profile.get("name") or f"Profile {idp}")
        segments: list[dict[str, Any]] = []
        for seg in profile.get("tDependentSegs") or []:
            if not isinstance(seg, dict) or _segment_is_empty(seg):
                continue
            start = int(seg.get("startTime", 0) or 0)
            segments.append(
                {
                    "time": _minutes_to_hhmm(start),
                    "basal_u_hr": round(int(seg.get("basalRate", 0) or 0) / 1000.0, 3),
                    "isf": int(seg.get("isf", 0) or 0),
                    "carb_ratio": round(int(seg.get("carbRatio", 0) or 0) / 1000.0, 2),
                    "target_bg": int(seg.get("targetBg", 0) or 0),
                }
            )
        entry: dict[str, Any] = {
            "name": name,
            "idp": idp,
            "active": idp == active_idp,
            "dia_hr": round(int(profile.get("insulinDuration", 240) or 240) / 60.0, 1),
            "max_bolus_u": round(int(profile.get("maxBolus", 0) or 0) / 1000.0, 2),
            "segments": segments,
        }
        formatted_profiles.append(entry)
        if entry["active"]:
            active_name = name

    cgm = settings.get("cgmSettings") or {}
    high = cgm.get("highGlucoseAlert") or {}
    low = cgm.get("lowGlucoseAlert") or {}

    out: dict[str, Any] = {
        "active_profile": active_name,
        "profiles": formatted_profiles,
        "cgm_alerts": {
            "high_mg_dl": int(high.get("mgPerDl", 0) or 0),
            "high_enabled": bool(int(high.get("enabled", 0) or 0)),
            "low_mg_dl": int(low.get("mgPerDl", 0) or 0),
            "low_enabled": bool(int(low.get("enabled", 0) or 0)),
        },
        "units": "mg/dL",
    }
    if pump_serial:
        out["pump_serial"] = pump_serial
    if as_of is not None:
        out["as_of"] = as_of.isoformat()
    return out


def _profile_raw_from_device(device: dict[str, Any]) -> RawEvent | None:
    settings = (device.get("lastUpload") or {}).get("settings")
    if not isinstance(settings, dict) or not settings:
        return None
    as_of = TandemConnector._parse_pump_date(device.get("maxDateWithEvents")) or datetime.now(UTC)
    serial = str(device.get("serialNumber") or "")
    payload = format_insulin_profile(
        settings,
        pump_serial=serial or None,
        as_of=as_of,
    )
    if not payload.get("profiles"):
        return None
    return _raw_event_from_payload(SOURCE, PROFILE_SOURCE_ID, as_of, payload)


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

    def check(self, *, timeout_s: float = 25) -> HealthReport:
        """Establish a Tandem Source session and list pumps on the account.

        tconnectsync v2+ authenticates via ``sso.tandemdiabetes.com`` /
        ``tdcservices.tandemdiabetes.com``, not the retired t:connect web
        login. Bad credentials fail here; a successful call proves reachability.
        """
        import concurrent.futures  # noqa: PLC0415

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._check_once)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                return HealthReport(
                    ok=False,
                    source=self.source,
                    detail=(
                        f"t:connect did not respond within {int(timeout_s)}s — "
                        "check network/VPN or try again later."
                    ),
                )

    def _check_once(self) -> HealthReport:
        try:
            pumps = self._client().tandemsource.pump_event_metadata()
        except RuntimeError:
            # Missing optional dependency is a setup bug, not a connectivity
            # result - keep the "pip install" message loud.
            raise
        except Exception as exc:  # tconnectsync error tree + requests transport errors
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        if not pumps:
            return HealthReport(
                ok=False,
                source=self.source,
                detail="No pumps found on Tandem Source account",
            )

        latest_pump = self._pick_latest_pump(pumps)
        latest_ts = self._parse_pump_date(latest_pump.get("maxDateWithEvents"))
        serial = latest_pump.get("serialNumber", "?")
        return HealthReport(
            ok=True,
            source=self.source,
            detail=f"Tandem Source connected · pump {serial}",
            latest_data_ts=latest_ts,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch pump events newer than ``since`` (minus a dedupe margin).

        Uses Tandem Source ``pump_events`` (tconnectsync v2+). Events before
        ``window_start`` are filtered out after normalization. Raises on provider
        hiccups — the sync workflow owns retries.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN
        now = datetime.now(tz=UTC)
        client = self._client()
        pumps = client.tandemsource.pump_event_metadata()
        if not pumps:
            return NormalizedBatch(raw=[], insulin=[], meals=[])

        device = self._choose_pump(pumps)
        device_id = str(device["tconnectDeviceId"])
        events = client.tandemsource.pump_events(
            device_id,
            window_start.strftime(_DATE_FMT),
            now.strftime(_DATE_FMT),
            fetch_all_event_types=False,
        )
        batch = pump_events_to_batch(events, window_start=window_start, end=now)
        profile_raw = _profile_raw_from_device(device)
        if profile_raw is not None:
            batch = NormalizedBatch(
                raw=[*batch.raw, profile_raw],
                insulin=batch.insulin,
                meals=batch.meals,
            )
        return batch

    def _choose_pump(self, pumps: Sequence[dict[str, Any]]) -> dict[str, Any]:
        serial = self._config.pump_serial.strip()
        by_serial = {str(p.get("serialNumber")): p for p in pumps}
        if serial and serial not in {"", "11111111"}:
            if serial not in by_serial:
                known = ", ".join(sorted(by_serial))
                msg = f"Pump serial {serial} not on account (have: {known})"
                raise RuntimeError(msg)
            return by_serial[serial]
        return self._pick_latest_pump(pumps)

    # -- legacy ControlIQ conversion helpers (tests + Nightscout-shaped stubs) ─

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

    @staticmethod
    def _parse_pump_date(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _pick_latest_pump(pumps: Sequence[dict[str, Any]]) -> dict[str, Any]:
        def sort_key(pump: dict[str, Any]) -> datetime:
            ts = TandemConnector._parse_pump_date(pump.get("maxDateWithEvents"))
            return ts or datetime.min.replace(tzinfo=UTC)

        return max(pumps, key=sort_key)

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
            region=self._tconnect_region(),
        )
        return cast("_TConnectClient", client)

    def _tconnect_region(self) -> str:
        region = self._config.region.strip().upper()
        return region if region in {"US", "EU"} else "US"
