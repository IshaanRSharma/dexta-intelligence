"""Domain-organized assembly of the reasoning-loop tool belt.

The stateful :class:`~dexta_intelligence.agents.tools.toolkit.DiscoveryToolkit`
and the cross-cutting logic functions (``_recall``, ``_search_evidence``,
``evidence_backend``) live in ``toolkit``. This package owns only the
declarative belt: each module turns toolkit methods into ``ToolSpec`` builders,
grouped by domain. :func:`build_belt` stitches them together and applies the
capability filter, replicating the historical ``tool_specs`` exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexta_intelligence.agents.tools.compare import compare_specs
from dexta_intelligence.agents.tools.glucose import glucose_specs
from dexta_intelligence.agents.tools.literature import literature_specs
from dexta_intelligence.agents.tools.manual import manual_specs
from dexta_intelligence.agents.tools.recall import recall_specs
from dexta_intelligence.agents.tools.similar import similar_specs
from dexta_intelligence.agents.tools.time_tools import time_tool_specs
from dexta_intelligence.agents.tools.treatment import treatment_specs

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.agents.reason import ToolSpec
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit

__all__ = ["build_belt"]


def build_belt(ctx: AgentContext, toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """The read-only instruments a reasoning loop may call.

    Gather the domain belts, hide the ones whose data stream is absent
    (capability filter, applied to everything except the always-on surfaces),
    then append manual context and time tools unfiltered. Manual logs are
    independent of glucose coverage; time tools are pure.
    """
    from dexta_intelligence.agents.tools.toolkit import _TOOL_NEEDS  # noqa: PLC0415

    specs = [
        *recall_specs(ctx),
        *glucose_specs(ctx, toolkit),
        *compare_specs(toolkit),
        *treatment_specs(toolkit),
        *similar_specs(toolkit),
        *literature_specs(),
    ]
    caps = toolkit.capabilities()
    specs = [spec for spec in specs if caps.allows(_TOOL_NEEDS.get(spec.name))]
    # Manual context is independent of glucose coverage - always available.
    specs.extend(manual_specs(toolkit))
    specs.extend(time_tool_specs())
    return specs
