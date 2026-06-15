"""Notification sinks for the monitoring pipeline.

A :class:`Notifier` is the outbound seam between deterministic anomaly
detection (``workflows.monitor``) and the outside world. Sinks are
dependency-light and **never raise into the monitor loop** — a failing
delivery is logged and swallowed so one bad sink can't suppress the rest of
the anomalies or crash the run.

Concrete sinks:

- :class:`LogNotifier` — default; emits a structured log line per anomaly.
- :class:`WebhookNotifier` — POSTs the anomaly as JSON to a configured URL
  (lazy ``httpx`` import; real-time outbound for chat/ops integrations).
- :class:`CollectingNotifier` — captures anomalies in memory for tests/GUI.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dexta_intelligence.workflows.monitor import Anomaly

logger = logging.getLogger(__name__)

__all__ = [
    "CollectingNotifier",
    "LogNotifier",
    "Notifier",
    "WebhookNotifier",
]


@runtime_checkable
class Notifier(Protocol):
    """Outbound sink for a single anomaly. Implementations must not raise."""

    def send(self, anomaly: Anomaly) -> None: ...


def _payload(anomaly: Anomaly) -> dict[str, object]:
    """JSON-serializable view of an anomaly (datetimes → ISO 8601)."""
    data = asdict(anomaly)
    start, end = anomaly.window
    data["window"] = {"start": start.isoformat(), "end": end.isoformat()}
    return data


class LogNotifier:
    """Default sink: one structured ``logging`` line per anomaly."""

    def __init__(self, *, level: int = logging.WARNING) -> None:
        self._level = level

    def send(self, anomaly: Anomaly) -> None:
        try:
            logger.log(
                self._level,
                "anomaly[%s] %s: %s | %s",
                anomaly.severity,
                anomaly.name,
                anomaly.headline,
                anomaly.numbers,
            )
        except Exception:
            logger.exception("LogNotifier failed to emit anomaly %s", anomaly.name)


class WebhookNotifier:
    """POST each anomaly as JSON to ``url`` via a lazily imported ``httpx``.

    Network and import errors are swallowed (logged) so the monitor loop is
    never interrupted by an unreachable endpoint or a missing optional dep.
    """

    def __init__(self, url: str, *, timeout: float = 5.0) -> None:
        self._url = url
        self._timeout = timeout

    def send(self, anomaly: Anomaly) -> None:
        try:
            import httpx  # noqa: PLC0415 - lazy: keep httpx optional for non-webhook use

            httpx.post(self._url, json=_payload(anomaly), timeout=self._timeout)
        except Exception:
            logger.exception(
                "WebhookNotifier failed to POST anomaly %s to %s", anomaly.name, self._url
            )


class CollectingNotifier:
    """In-memory sink: appends every anomaly to :attr:`received` (tests/GUI)."""

    def __init__(self) -> None:
        self.received: list[Anomaly] = []

    def send(self, anomaly: Anomaly) -> None:
        try:
            self.received.append(anomaly)
        except Exception:
            logger.exception("CollectingNotifier failed to capture anomaly %s", anomaly.name)
