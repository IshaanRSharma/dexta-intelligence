"""Every migrated agent prompt must equal its intended markdown file.

The .md files were dumped from the original inline constants, so this asserts
the registry migration was byte-identical and that each module loads the right
file (no cross-wiring).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / (
    "src/dexta_intelligence/agents/prompts"
)

# (module, constant, intended markdown name)
_SCALAR = [
    ("chat", "_SYSTEM", "chat_system"),
    ("discovery", "_PLAN_PROMPT", "discovery_plan"),
    ("insulin", "_PLAN_PROMPT", "insulin_plan"),
    ("coordinator", "_PLAN_PROMPT", "coordinator_plan"),
    ("coordinator", "_REPLAN_PROMPT", "coordinator_replan"),
    ("brief", "_SYSTEM", "brief_system"),
    ("brief", "_USER_TEMPLATE", "brief_user"),
    ("orchestrator", "INVESTIGATION_DOCTRINE", "orchestrator_doctrine"),
    ("orchestrator", "_SYSTEM", "orchestrator_system"),
    ("investigator", "_REFLECT_PROMPT", "investigator_reflect"),
    ("investigator", "_WRITE_PROMPT", "investigator_write"),
    ("router", "_SAFETY", "router_safety"),
    ("router", "_ROUTE_PROMPT", "router_route"),
    ("seeker", "_REFLECT_PROMPT", "seeker_reflect"),
    ("seeker", "_GOAL_SYSTEM", "seeker_goal_system"),
    ("advisory", "_REFINE_PROMPT", "advisory_refine"),
]

_FAMILY_KEYS = ("spike_explanation", "time_traversal", "two_group", "memory", "evidence")


def _file(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


@pytest.mark.parametrize(("module", "const", "fname"), _SCALAR)
def test_constant_matches_its_prompt_file(module: str, const: str, fname: str) -> None:
    mod = importlib.import_module(f"dexta_intelligence.agents.{module}")
    assert getattr(mod, const) == _file(fname)


def test_router_family_system_matches_files() -> None:
    router = importlib.import_module("dexta_intelligence.agents.router")
    assert set(router._FAMILY_SYSTEM) == set(_FAMILY_KEYS)
    for key in _FAMILY_KEYS:
        assert router._FAMILY_SYSTEM[key] == _file(f"router_family_{key}")
