"""Medtronic pump + CGM connector - direct CareLink access via carelink_client.

.. warning:: **UNOFFICIAL, reverse-engineered CareLink API.** This is opt-in,
   read-only, and built on an undocumented endpoint that Medtronic does not
   support: region-split login flow that is fragile and may break without
   notice. There is no write path and there never will be one - the connector
   cannot touch the pump. Its only job is to remove the Nightscout setup hop
   for Medtronic users, pulling sensor glucose and pump markers straight from
   the CareLink "recent data" payload.

.. warning:: **CareLink only serves a recent window (~24h).** Like the Dexcom
   Share path, ``pull(since)`` can never reach further back than CareLink's
   own retention for the recent-data feed - a ``since`` older than that yields
   only what CareLink still holds, never an error, never more. For real
   history, backfill through a CareLink CSV export or Nightscout and let this
   connector own the live edge.

The module follows the house connector split:

- **Pure conversion** (:func:`parse_recent_data`) takes a CareLink
  recent-data payload (a plain dict, exactly what ``carelink_client``'s
  ``recentData()`` returns) and emits typed timeline events. No I/O, no
  carelink_client import required - tests run on tiny dict fixtures.
- **CareLinkConnector** owns the session: lazy carelink_client import
  (``carelink_client`` is not on PyPI; install it from upstream - see the
  import-error message), credential/region/patient handling via
  :class:`CareLinkConfig`, and the ``recentData()`` fetch.

Payload shapes (against documented CareLink recent-data responses):

- ``sgs``: sensor glucose readings, each ``{sg, datetime|timestamp}`` plus an
  optional ``trend``/``sensorState`` arrow. ``sg == 0`` marks a gap (no
  calibration / sensor warm-up) and is dropped.
- ``markers``: pump events tagged by ``type``: ``INSULIN``/``BOLUS`` ->
  :class:`InsulinEvent` (BOLUS), ``AUTO_BASAL_DELIVERY``/``BASAL`` -> BASAL,
  ``TEMP_BASAL`` / auto-mode microbolus -> TEMP_BASAL, ``INSULIN_SUSPEND`` /
  ``SUSPEND`` / ``LGS`` -> SUSPEND, ``MEAL``/``BG``/carb markers ->
  :class:`MealEvent`.

Units: mg/dL is CareLink's native US unit. If the account is configured in
mmol/L (the payload carries ``units == "MMOL_L"`` / ``"MMOL/L"``), sensor
glucose comes through as mmol/L and is converted to mg/dL with the standard
factor (``x 18``, rounded). Carb markers are always grams.

Timestamps: CareLink stamps records either as epoch-millis (``timestamp``) or
an ISO string with offset (``dateTime``/``datetime``). Both convert to UTC.
A naive ISO string with no offset is rejected loudly (``ValueError``), per the
house UTC rule - CareLink local-time strings without a zone are exactly the
class of CGM time bug the models refuse to inherit.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import (
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    MealEvent,
    RawEvent,
)

if TYPE_CHECKING:
    from dexta_intelligence.config import CareLinkConfig

__all__ = ["CareLinkClientLike", "CareLinkConnector", "parse_recent_data"]

SOURCE = "carelink"

_MMOL_TO_MGDL = 18.0
_MMOL_UNIT_TOKENS = frozenset({"mmol_l", "mmol/l", "mmol"})

#: CareLink marker ``type`` substrings -> InsulinKind. Suspend/temp-basal are
#: checked before the broad bolus/basal tokens because their type strings often
#: also contain "basal"/"insulin".
_SUSPEND_TOKENS = ("suspend", "lgs", "low_glucose")
_TEMP_BASAL_TOKENS = ("temp_basal", "tempbasal", "auto_basal", "autobasal", "microbolus")
_BASAL_TOKENS = ("basal",)
_BOLUS_TOKENS = ("bolus", "insulin")
_MEAL_TOKENS = ("meal", "carb", "bg")


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion - CareLink recent-data dict in, typed events out
# ─────────────────────────────────────────────────────────────────────────────


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _parse_ts(record: dict[str, Any]) -> datetime | None:
    """Record timestamp -> aware UTC.

    Prefers the epoch-millis ``timestamp`` (always UTC) and falls back to the
    ISO ``dateTime``/``datetime`` string. A naive ISO string (no offset) is
    rejected loudly: CareLink's wall-clock strings without a zone are silent
    local-time bugs we refuse to inherit.
    """
    millis = record.get("timestamp")
    if isinstance(millis, int | float) and not isinstance(millis, bool):
        return datetime.fromtimestamp(millis / 1000.0, tz=UTC)
    for key in ("dateTime", "datetime"):
        value = record.get(key)
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                msg = (
                    "naive CareLink timestamp rejected: an ISO record without a UTC "
                    "offset cannot be placed on the timeline"
                )
                raise ValueError(msg)
            return parsed.astimezone(UTC)
    return None


def _to_mg_dl(sg: float, *, mmol: bool) -> int:
    return round(sg * _MMOL_TO_MGDL) if mmol else round(sg)


def _trend(record: dict[str, Any]) -> str | None:
    trend = record.get("trend") or record.get("trendArrow")
    return trend if isinstance(trend, str) and trend else None


def _parse_sg(record: dict[str, Any], *, mmol: bool) -> GlucoseEvent | None:
    """One ``sgs`` sensor-glucose record -> :class:`GlucoseEvent`.

    ``sg`` of 0 / missing marks a gap (warm-up, lost calibration) and yields
    ``None``.
    """
    sg = _as_float(record.get("sg"))
    ts = _parse_ts(record)
    if sg is None or sg <= 0 or ts is None:
        return None
    return GlucoseEvent(ts=ts, mg_dl=_to_mg_dl(sg, mmol=mmol), trend=_trend(record))


def _marker_kind(marker_type: str) -> InsulinKind | None:
    if any(tok in marker_type for tok in _SUSPEND_TOKENS):
        return InsulinKind.SUSPEND
    if any(tok in marker_type for tok in _TEMP_BASAL_TOKENS):
        return InsulinKind.TEMP_BASAL
    if any(tok in marker_type for tok in _BASAL_TOKENS):
        return InsulinKind.BASAL
    if any(tok in marker_type for tok in _BOLUS_TOKENS):
        return InsulinKind.BOLUS
    return None


def _parse_marker(record: dict[str, Any]) -> InsulinEvent | MealEvent | None:
    """One ``markers`` record -> insulin/meal event, or ``None`` if irrelevant.

    Carb/meal markers win when carbs are present; otherwise the marker ``type``
    classifies the insulin kind. ``automatic`` is flagged for the closed-loop
    delivery kinds (auto-basal / temp-basal microbolus).
    """
    ts = _parse_ts(record)
    if ts is None:
        return None
    marker_type = str(record.get("type", "")).strip().lower()

    carbs = _as_float(record.get("carbInput"))
    if carbs is None:
        carbs = _as_float(record.get("carbs"))
    if carbs is not None and carbs > 0 and any(tok in marker_type for tok in _MEAL_TOKENS):
        return MealEvent(ts=ts, carbs_g=carbs)

    kind = _marker_kind(marker_type)
    if kind is None:
        return None
    units = _as_float(record.get("amount"))
    if units is None:
        units = _as_float(record.get("deliveredFastAmount")) or _as_float(record.get("units"))
    duration = _as_float(record.get("duration")) or _as_float(record.get("durationInMinutes"))
    automatic = True if kind in (InsulinKind.BASAL, InsulinKind.TEMP_BASAL) else None
    return InsulinEvent(
        ts=ts,
        kind=kind,
        units=units if units is not None and units >= 0 else None,
        duration_min=duration,
        automatic=automatic,
    )


def _is_mmol(payload: dict[str, Any]) -> bool:
    units = payload.get("units") or payload.get("sgUnits") or payload.get("displayUnits")
    return isinstance(units, str) and units.strip().lower().replace(" ", "") in _MMOL_UNIT_TOKENS


def parse_recent_data(
    payload: dict[str, Any],
) -> tuple[list[GlucoseEvent], list[InsulinEvent], list[MealEvent]]:
    """A CareLink recent-data payload -> (glucose, insulin, meals).

    Pure: no I/O, no clock, no config - a plain dict in, typed events out.
    Sensor glucose comes from ``sgs``; pump deliveries and carbs come from
    ``markers``. Unit conversion (mmol/L -> mg/dL) is driven by the payload's
    own unit field.
    """
    mmol = _is_mmol(payload)

    glucose: list[GlucoseEvent] = []
    sgs = payload.get("sgs")
    if isinstance(sgs, list):
        for record in sgs:
            if not isinstance(record, dict):
                continue
            event = _parse_sg(record, mmol=mmol)
            if event is not None:
                glucose.append(event)

    insulin: list[InsulinEvent] = []
    meals: list[MealEvent] = []
    markers = payload.get("markers")
    if isinstance(markers, list):
        for record in markers:
            if not isinstance(record, dict):
                continue
            marker = _parse_marker(record)
            if isinstance(marker, InsulinEvent):
                insulin.append(marker)
            elif isinstance(marker, MealEvent):
                meals.append(marker)

    return glucose, insulin, meals


# ─────────────────────────────────────────────────────────────────────────────
# Client duck type
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class CareLinkClientLike(Protocol):
    """The slice of ``carelink_client``'s client the connector uses.

    Duck-typed: the real ``CareLinkClient`` satisfies this and so does any stub
    with a ``login`` and ``recentData`` - the connector is testable without
    carelink_client installed. ``recentData`` returns the recent-data payload
    dict consumed by :func:`parse_recent_data`.
    """

    def login(self) -> bool: ...

    def recentData(self) -> dict[str, Any] | None: ...  # noqa: N802 - upstream API name


# ─────────────────────────────────────────────────────────────────────────────
# Connector - thin session layer over the pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class CareLinkConnector:
    """Implements the :class:`~dexta_intelligence.connectors.base.Connector`
    protocol against the unofficial CareLink recent-data API via carelink_client.

    Batch-only (no ``current()``): CareLink's recent-data feed is the whole
    surface, and ``pull()`` is bounded by CareLink's own recent window (see
    module docstring - typically ~24h). carelink_client is an optional extra
    imported lazily, so the base install never pays for it.
    """

    source = SOURCE

    def __init__(self, config: CareLinkConfig, *, client: CareLinkClientLike | None = None) -> None:
        self._config = config
        self._client_instance = client

    # -- Connector protocol ----------------------------------------------------

    def check(self) -> HealthReport:
        """Log in and report the latest available sensor glucose reading.

        ``login()`` *is* the auth probe: a bad username/password (or the
        fragile region-split flow failing) surfaces here. Read-only: nothing
        is mutated.
        """
        try:
            client = self._client()
            client.login()
            payload = client.recentData()
        except RuntimeError:
            # Missing optional dependency is a setup bug, not a connectivity
            # result - keep the "pip install" message loud.
            raise
        except Exception as exc:  # carelink_client error tree + transport errors
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        latest_ts: datetime | None = None
        if isinstance(payload, dict):
            glucose, _, _ = parse_recent_data(payload)
            if glucose:
                latest_ts = max(event.ts for event in glucose)

        return HealthReport(
            ok=True,
            source=self.source,
            detail="CareLink session established",
            latest_data_ts=latest_ts,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch everything in CareLink's recent window newer than ``since``.

        **CareLink only serves a recent window (~24h)**: a ``since`` older
        than that yields only what CareLink still holds - never an error,
        never more. Use a CareLink CSV export or Nightscout for older history.

        CareLink records carry no stable provider id, so the event kind + UTC
        timestamp is the idempotency key (``carelink:sg:<iso>``,
        ``carelink:bolus:<iso>``, …), safe because CareLink emits at most one
        record per kind per timestamp slot.
        """
        window_start = since.astimezone(UTC)
        client = self._client()
        client.login()
        payload = client.recentData()
        if not isinstance(payload, dict):
            return NormalizedBatch()

        glucose_all, insulin_all, meals_all = parse_recent_data(payload)

        raw_events: list[RawEvent] = []
        glucose: list[GlucoseEvent] = []
        insulin: list[InsulinEvent] = []
        meals: list[MealEvent] = []

        for event in glucose_all:
            if event.ts < window_start:
                continue
            glucose.append(event)
            raw_events.append(self._raw_event("sg", event.ts, {"sg": event.mg_dl}))
        for ins in insulin_all:
            if ins.ts < window_start:
                continue
            insulin.append(ins)
            payload_ins = {"kind": ins.kind.value, "units": ins.units}
            raw_events.append(self._raw_event(ins.kind.value, ins.ts, payload_ins))
        for meal in meals_all:
            if meal.ts < window_start:
                continue
            meals.append(meal)
            raw_events.append(self._raw_event("meal", meal.ts, {"carbs": meal.carbs_g}))

        return NormalizedBatch(raw=raw_events, glucose=glucose, insulin=insulin, meals=meals)

    # -- session plumbing --------------------------------------------------------

    def _raw_event(self, kind: str, ts: datetime, payload: dict[str, Any]) -> RawEvent:
        source_id = f"carelink:{kind}:{ts.isoformat()}"
        return RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=payload)

    def _client(self) -> CareLinkClientLike:
        if self._client_instance is None:
            self._client_instance = self._build_client()
        return self._client_instance

    def _build_client(self) -> CareLinkClientLike:
        try:
            # Deliberately lazy (via importlib, so type-checking never needs
            # the package): CareLink support is an optional extra and the rest
            # of the system must be importable without it.
            carelink = importlib.import_module("carelink_client")
        except ImportError as exc:  # pragma: no cover - import-path guard
            msg = (
                "Medtronic CareLink support needs the 'carelink_client' package, "
                "which is not published on PyPI. Install it from upstream: "
                "pip install 'git+https://github.com/ondrej1024/carelink-python-client'"
            )
            raise RuntimeError(msg) from exc

        client = carelink.CareLinkClient(
            carelinkUsername=self._config.username,
            carelinkPassword=self._config.password,
            carelinkCountry=self._config.country,
            carelinkPatient=self._config.patient or None,
        )
        return cast("CareLinkClientLike", client)
