"""Sync workflow - ``pull → raw upsert → normalize-insert → daily rollups``.

The workflow owns persistence and watermarks (the
:class:`~dexta_intelligence.connectors.base.Connector` contract); connectors
own only provider I/O and normalization. One :func:`sync` call is one
attempt: connector exceptions propagate to the caller (or to
:func:`sync_all`, which isolates them per source). There is deliberately no
retry framework here.

Idempotency and the overlap margin
----------------------------------
``since`` is the stored watermark minus :data:`OVERLAP_MARGIN` (or
``now - default_lookback`` on first sync). Re-pulling the overlap is free:
the raw layer dedupes on ``(source, source_id)`` and timeline ``insert_*``
methods are expected to skip duplicates likewise, so counts in the report
reflect genuinely *new* rows.

Provenance
----------
Connectors do not carry the raw-to-typed link on typed events, so it is
reconstructed here by timestamp: a typed event's ``ts`` (``ts_start`` for sleep)
is matched against the raw ``source_ts`` of its originating record. The match is
applied only where a ``source_ts`` maps to exactly one raw row; ambiguous
timestamps and typed events with no matching raw are left with
``raw_event_id=None`` rather than guessed.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, TypeVar

from dexta_intelligence.analytics.rollups import daily_rollup

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import date

    from dexta_intelligence.connectors.base import Connector
    from dexta_intelligence.models import RawEvent, Rollup
    from dexta_intelligence.store.port import StoragePort


class _TimelineEvent(Protocol):
    def model_copy(self: _E, *, update: dict[str, object]) -> _E: ...


_E = TypeVar("_E", bound=_TimelineEvent)

__all__ = ["DEFAULT_LOOKBACK", "OVERLAP_MARGIN", "SyncReport", "sync", "sync_all"]

#: First-sync window when a source has no watermark yet.
DEFAULT_LOOKBACK = timedelta(days=30)

#: Re-pull margin subtracted from the watermark - three CGM slots of safety
#: against late-arriving or clock-skewed provider records. Idempotent
#: storage makes the re-pull free.
OVERLAP_MARGIN = timedelta(minutes=15)

#: Raw ``source_id`` values that are singleton snapshots - upserted with
#: replace-on-conflict so each sync refreshes the payload.
_SNAPSHOT_RAW_IDS = frozenset({"tandem:profile:active"})

#: Snapshot ids whose payload is a therapy profile we also version over time.
_PROFILE_RAW_IDS = frozenset({"tandem:profile:active"})


def _capture_profile_versions(store: StoragePort, snapshot_raw: list[RawEvent]) -> None:
    """Record a therapy-profile version for each profile snapshot in the batch.

    Devices report only the current profile, so this keeps a content-addressed
    history: ``add_profile_version`` is a no-op when the payload is unchanged and
    opens a new version when it changes. Best-effort: never fails a sync, and
    silently no-ops on stores without the method (older schemas)."""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    from dexta_intelligence.models import TherapyProfile  # noqa: PLC0415

    add = getattr(store, "add_profile_version", None)
    if add is None:
        return
    for raw in snapshot_raw:
        if raw.source_id not in _PROFILE_RAW_IDS or not isinstance(raw.payload, dict):
            continue
        payload = dict(raw.payload)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        try:
            add(
                TherapyProfile(
                    source=raw.source,
                    name=str(payload.get("active_profile") or "profile"),
                    content=payload,
                    content_hash=digest,
                    active_from=raw.source_ts,
                    created_at=datetime.now(UTC),
                )
            )
        except Exception:
            logger.warning("sync: failed to capture profile version", exc_info=True)


@dataclass(frozen=True, slots=True)
class SyncReport:
    """Outcome of one sync attempt for one source.

    ``errors`` is empty on success. A failed :func:`sync_all` entry carries
    the error string and zeroed counts (``since`` may be ``None`` when the
    failure happened before the window was established).
    """

    source: str
    since: datetime | None
    until: datetime
    raw_new: int = 0
    inserted: dict[str, int] = field(default_factory=dict)
    rollup_days: int = 0
    duration_s: float = 0.0
    errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the sync completed without errors."""
        return not self.errors


def sync(
    connector: Connector,
    store: StoragePort,
    *,
    default_lookback: timedelta = DEFAULT_LOOKBACK,
    overlap: timedelta = OVERLAP_MARGIN,
    now: datetime | None = None,
) -> SyncReport:
    """Run one full sync for one source. Connector exceptions propagate.

    Steps: resolve the window from the watermark, ``connector.pull``, upsert
    raws, insert every normalized stream through the typed port methods,
    then recompute the daily rollup for every UTC day touched by the
    batch's glucose events (reading each day's *full* stored series, so a
    partial pull never produces a partial rollup).

    Args:
        connector: The source to pull from.
        store: The persistence port; the only storage surface used.
        default_lookback: Window for a source with no watermark yet.
        overlap: Safety margin subtracted from the watermark.
        now: Injectable clock (must be timezone-aware); defaults to UTC now.

    Raises:
        ValueError: if ``now`` is naive - all timestamps are UTC-enforced.
    """
    started = time.monotonic()
    if now is not None and now.tzinfo is None:
        msg = "naive datetime rejected: 'now' must be timezone-aware (UTC)"
        raise ValueError(msg)
    until = (now or datetime.now(UTC)).astimezone(UTC)

    watermark = store.get_watermark(connector.source)
    since = watermark - overlap if watermark is not None else until - default_lookback

    batch = connector.pull(since)
    snapshot_raw = [r for r in batch.raw if r.source_id in _SNAPSHOT_RAW_IDS]
    event_raw = [r for r in batch.raw if r.source_id not in _SNAPSHOT_RAW_IDS]
    existing_before = store.existing_raw_ids(event_raw)
    id_map = store.upsert_raw_events(event_raw)
    if snapshot_raw:
        id_map.update(store.replace_raw_events(snapshot_raw))
        _capture_profile_versions(store, snapshot_raw)
    raw_new = sum(1 for sid in id_map if sid not in existing_before)

    ts_to_raw_id = _unambiguous_ts_index(batch.raw, id_map)

    inserted = {
        "glucose": _insert_linked(store.insert_glucose, batch.glucose, ts_to_raw_id, "ts"),
        "insulin": _insert_linked(store.insert_insulin, batch.insulin, ts_to_raw_id, "ts"),
        "meals": _insert_linked(store.insert_meals, batch.meals, ts_to_raw_id, "ts"),
        "activity": _insert_linked(store.insert_activity, batch.activity, ts_to_raw_id, "ts"),
        "sleep": _insert_linked(store.insert_sleep, batch.sleep, ts_to_raw_id, "ts_start"),
        "recovery": _insert_linked(store.insert_recovery, batch.recovery, ts_to_raw_id, "ts"),
        "device": _insert_linked(store.insert_device, batch.device, ts_to_raw_id, "ts"),
        "predictions": _insert_linked(
            store.insert_predictions, batch.predictions, ts_to_raw_id, "ts"
        ),
    }

    touched_days = sorted({g.ts.date() for g in batch.glucose})
    rollups: list[Rollup] = []
    for day in touched_days:
        rollup = _recompute_day(store, day)
        if rollup is not None:
            rollups.append(rollup)
    if rollups:
        store.upsert_rollups(rollups)

    return SyncReport(
        source=connector.source,
        since=since,
        until=until,
        raw_new=raw_new,
        inserted=inserted,
        rollup_days=len(rollups),
        duration_s=time.monotonic() - started,
    )


def sync_all(
    connectors: Iterable[Connector],
    store: StoragePort,
    *,
    default_lookback: timedelta = DEFAULT_LOOKBACK,
    overlap: timedelta = OVERLAP_MARGIN,
    now: datetime | None = None,
) -> list[SyncReport]:
    """Sync every source, isolating failures per source.

    One failing connector never stops the others: its report carries the
    error string and zeroed counts, and the loop continues.
    """
    reports: list[SyncReport] = []
    for connector in connectors:
        started = time.monotonic()
        try:
            reports.append(
                sync(
                    connector,
                    store,
                    default_lookback=default_lookback,
                    overlap=overlap,
                    now=now,
                )
            )
        except Exception as exc:  # per-source isolation is the contract
            until = now if now is not None and now.tzinfo is not None else datetime.now(UTC)
            reports.append(
                SyncReport(
                    source=connector.source,
                    since=None,
                    until=until,
                    duration_s=time.monotonic() - started,
                    errors=(f"{type(exc).__name__}: {exc}",),
                )
            )
    return reports


def _unambiguous_ts_index(
    raw: list[RawEvent], id_map: dict[str, int]
) -> dict[datetime, int]:
    """``source_ts -> raw id`` for instants owned by exactly one raw row.

    Timestamps shared by multiple raws are dropped: a typed event at such an
    instant cannot be attributed to a single source record, so it is left
    unlinked rather than mislinked.
    """
    counts = Counter(r.source_ts for r in raw)
    index: dict[datetime, int] = {}
    for r in raw:
        if counts[r.source_ts] != 1:
            continue
        raw_id = id_map.get(r.source_id)
        if raw_id is not None:
            index[r.source_ts] = raw_id
    return index


def _insert_linked(
    insert: Callable[[list[_E]], int],
    events: list[_E],
    ts_to_raw_id: dict[datetime, int],
    ts_attr: str,
) -> int:
    """Wire ``raw_event_id`` onto each event by timestamp, then insert.

    Events whose timestamp has no unique raw match keep ``raw_event_id=None``.
    """
    if not events:
        return 0
    linked = [
        e.model_copy(update={"raw_event_id": raw_id})
        if (raw_id := ts_to_raw_id.get(getattr(e, ts_attr))) is not None
        else e
        for e in events
    ]
    return insert(linked)


def _recompute_day(store: StoragePort, day: date) -> Rollup | None:
    """Recompute one UTC day's rollup from the store's full timeline."""
    day_start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    return daily_rollup(
        day,
        store.get_glucose(day_start, day_end),
        insulin=store.get_insulin(day_start, day_end),
        meals=store.get_meals(day_start, day_end),
    )
