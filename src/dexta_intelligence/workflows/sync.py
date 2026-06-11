"""Sync workflow — ``pull → raw upsert → normalize-insert → daily rollups``.

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

Provenance note
---------------
:meth:`StoragePort.upsert_raw_events` returns only a count of new rows — the
port exposes no way to learn the ids assigned to raw rows. ``raw_event_id``
linkage therefore cannot be wired by this workflow through the current port;
normalized events are persisted with whatever ``raw_event_id`` the connector
set (typically ``None``). Backends that can resolve provenance internally
(e.g. within one transaction) are free to do so.

Predictions note
----------------
:class:`~dexta_intelligence.connectors.base.NormalizedBatch` may carry
``predictions``, but ``StoragePort`` has no prediction insert method, so
prediction events are NOT persisted; the report notes how many were skipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from dexta_intelligence.analytics.rollups import daily_rollup

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    from dexta_intelligence.connectors.base import Connector
    from dexta_intelligence.models import Rollup
    from dexta_intelligence.store.port import StoragePort

__all__ = ["DEFAULT_LOOKBACK", "OVERLAP_MARGIN", "SyncReport", "sync", "sync_all"]

#: First-sync window when a source has no watermark yet.
DEFAULT_LOOKBACK = timedelta(days=30)

#: Re-pull margin subtracted from the watermark — three CGM slots of safety
#: against late-arriving or clock-skewed provider records. Idempotent
#: storage makes the re-pull free.
OVERLAP_MARGIN = timedelta(minutes=15)


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
        ValueError: if ``now`` is naive — all timestamps are UTC-enforced.
    """
    started = time.monotonic()
    if now is not None and now.tzinfo is None:
        msg = "naive datetime rejected: 'now' must be timezone-aware (UTC)"
        raise ValueError(msg)
    until = (now or datetime.now(UTC)).astimezone(UTC)

    watermark = store.get_watermark(connector.source)
    since = watermark - overlap if watermark is not None else until - default_lookback

    batch = connector.pull(since)
    raw_new = store.upsert_raw_events(batch.raw)

    inserted = {
        "glucose": store.insert_glucose(batch.glucose) if batch.glucose else 0,
        "insulin": store.insert_insulin(batch.insulin) if batch.insulin else 0,
        "meals": store.insert_meals(batch.meals) if batch.meals else 0,
        "activity": store.insert_activity(batch.activity) if batch.activity else 0,
        "sleep": store.insert_sleep(batch.sleep) if batch.sleep else 0,
        "recovery": store.insert_recovery(batch.recovery) if batch.recovery else 0,
        "device": store.insert_device(batch.device) if batch.device else 0,
    }

    notes: tuple[str, ...] = ()
    if batch.predictions:
        notes = (
            f"{len(batch.predictions)} prediction events not persisted: "
            "StoragePort has no prediction insert method",
        )

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
        notes=notes,
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
