"""Lenses — named agent routes over a filtered :class:`AgentRegistry`.

A lens selects which *producer* agents run plus an optional window. The skeptic
post-pass is non-routable: it is always appended and can never be excluded
(routing selects what runs, never whether it is honest — see
``docs/INTELLIGENCE.md`` §3.2). User ``[lens.*]`` entries in ``dexta.toml``
override or extend the built-ins below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import AgentRegistry
from dexta_intelligence.config import Config, LensConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ["BUILTIN_LENSES", "PRODUCERS", "SKEPTIC", "build_registry"]

SKEPTIC = "skeptic"
"""The non-routable core: always registered, never excludable."""

BUILTIN_LENSES: dict[str, LensConfig] = {
    "analyze": LensConfig(
        agents=["observation", "pattern", "reconciliation", "discovery", "insulin"],
    ),
    "watch": LensConfig(agents=["observation", "pattern"], window_days=7),
    "why": LensConfig(agents=["reconciliation", "discovery"]),
    "insulin": LensConfig(agents=["insulin"], window_days=30),
}


def _register_observation(registry: AgentRegistry, config: Config, model: Any) -> None:
    from dexta_intelligence.agents import register_observation  # noqa: PLC0415

    del config, model
    register_observation(registry)


def _register_pattern(registry: AgentRegistry, config: Config, model: Any) -> None:
    from dexta_intelligence.agents import register_pattern  # noqa: PLC0415

    del config, model
    register_pattern(registry)


def _register_reconciliation(registry: AgentRegistry, config: Config, model: Any) -> None:
    from dexta_intelligence.agents import register_reconciliation  # noqa: PLC0415

    del config, model
    register_reconciliation(registry)


def _register_discovery(registry: AgentRegistry, config: Config, model: Any) -> None:
    from dexta_intelligence.agents import register_discovery  # noqa: PLC0415

    register_discovery(
        registry,
        model=model,
        target_low=config.analysis.target_low,
        target_high=config.analysis.target_high,
    )


def _register_insulin(registry: AgentRegistry, config: Config, model: Any) -> None:
    from dexta_intelligence.agents import register_insulin  # noqa: PLC0415

    register_insulin(
        registry,
        model=model,
        target_low=config.analysis.target_low,
        target_high=config.analysis.target_high,
    )


def _register_skeptic(registry: AgentRegistry, config: Config, model: Any) -> None:
    from dexta_intelligence.agents import register_skeptic  # noqa: PLC0415

    del config, model
    register_skeptic(registry)


# Producer name → register fn. The skeptic lives in SKEPTIC, not here: it is
# never user-selectable but always appended.
PRODUCERS: dict[str, Callable[[AgentRegistry, Config, Any], None]] = {
    "observation": _register_observation,
    "pattern": _register_pattern,
    "reconciliation": _register_reconciliation,
    "discovery": _register_discovery,
    "insulin": _register_insulin,
}

_REGISTER = {**PRODUCERS, SKEPTIC: _register_skeptic}


def _resolved_lenses(config: Config) -> dict[str, LensConfig]:
    """Built-ins overlaid with user ``[lens.*]`` entries (user wins)."""
    return {**BUILTIN_LENSES, **config.lens}


def build_registry(
    lens_name: str,
    config: Config,
    *,
    model: Any = None,
) -> tuple[AgentRegistry, int | None]:
    """Build the filtered registry for ``lens_name`` plus its window override.

    The returned registry holds the lens's selected producers and **always** the
    skeptic (non-removable). Raises ``ValueError`` on an unknown lens or an
    unknown agent name, listing the known names to keep the error actionable.
    """
    lenses = _resolved_lenses(config)
    lens = lenses.get(lens_name)
    if lens is None:
        known = ", ".join(sorted(lenses))
        msg = f"unknown lens {lens_name!r}; known lenses: {known}"
        raise ValueError(msg)

    registry = AgentRegistry()
    for agent_name in lens.agents:
        if agent_name == SKEPTIC:
            continue  # skeptic is appended unconditionally below; never duplicate
        register = PRODUCERS.get(agent_name)
        if register is None:
            known = ", ".join(sorted(PRODUCERS))
            msg = f"unknown agent {agent_name!r} in lens {lens_name!r}; known agents: {known}"
            raise ValueError(msg)
        register(registry, config, model)

    _register_skeptic(registry, config, model)
    return registry, lens.window_days
