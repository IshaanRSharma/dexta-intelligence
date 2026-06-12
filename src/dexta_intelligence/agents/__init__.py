"""Agent plugins for the dexta-intelligence blackboard."""

from dexta_intelligence.agents.base import (
    AgentContext,
    AgentRegistry,
    DataRequirement,
    DextaAgent,
)
from dexta_intelligence.agents.brief import ClinicalBrief, build_brief, render_markdown
from dexta_intelligence.agents.chat import ChatAgent, ChatAnswer
from dexta_intelligence.agents.discovery import (
    DiscoveryAgent,
    discovery_agent,
    register_discovery,
)
from dexta_intelligence.agents.insulin import (
    InsulinAgent,
    insulin_agent,
    register_insulin,
)
from dexta_intelligence.agents.observation import (
    ObservationAgent,
    observation_agent,
    register_observation,
)
from dexta_intelligence.agents.pattern import (
    PatternAgent,
    pattern_agent,
    register_pattern,
)
from dexta_intelligence.agents.reconciliation import (
    PredictionReconciliationAgent,
    reconciliation_agent,
    register_reconciliation,
)
from dexta_intelligence.agents.router import Route, RouterAgent
from dexta_intelligence.agents.seeker import GoalSeekingAgent, Reflection
from dexta_intelligence.agents.skeptic import (
    SkepticAgent,
    register_skeptic,
    skeptic_agent,
)
from dexta_intelligence.agents.trace import TraceLine, render_trace

__all__ = [
    "AgentContext",
    "AgentRegistry",
    "ChatAgent",
    "ChatAnswer",
    "ClinicalBrief",
    "DataRequirement",
    "DextaAgent",
    "DiscoveryAgent",
    "GoalSeekingAgent",
    "InsulinAgent",
    "ObservationAgent",
    "PatternAgent",
    "PredictionReconciliationAgent",
    "Reflection",
    "Route",
    "RouterAgent",
    "SkepticAgent",
    "TraceLine",
    "build_brief",
    "discovery_agent",
    "insulin_agent",
    "observation_agent",
    "pattern_agent",
    "reconciliation_agent",
    "register_discovery",
    "register_insulin",
    "register_observation",
    "register_pattern",
    "register_reconciliation",
    "register_skeptic",
    "render_markdown",
    "render_trace",
    "skeptic_agent",
]
