"""Headless tests for the ego-graph layout in timeline.js.

Drive layoutGraphNodes (the pure geometry exported when the file is loaded
under Node) with the dense cases the ad hoc layout used to fail on: a run of
rescue carbs and a stack of correction boluses. Skipped when node is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

_JS = (
    Path(__file__).parent.parent
    / "src"
    / "dexta_intelligence"
    / "server"
    / "static"
    / "timeline.js"
)


def _layout(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    script = (
        f"const tl = require({json.dumps(str(_JS))});"
        f"process.stdout.write(JSON.stringify(tl.layoutGraphNodes({json.dumps(links)})));"
    )
    assert NODE is not None
    res = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, check=True, timeout=30
    )
    out: list[dict[str, Any]] = json.loads(res.stdout)
    return out


def _link(link_kind: str, offset_min: float, **detail: Any) -> dict[str, Any]:
    return {
        "kind": link_kind,
        "offset_min": offset_min,
        "ts": "2026-01-01T00:00:00+00:00",
        "detail": detail,
    }


def _assert_on_canvas(nodes: list[dict[str, Any]]) -> None:
    for p in nodes:
        assert 40 <= p["x"] <= 920, p
        assert 30 <= p["y"] <= 590, p


def _min_pairwise_gap(nodes: list[dict[str, Any]]) -> float:
    gap = float("inf")
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            d = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
            gap = min(gap, d)
    return gap


def test_rescue_carb_run_collapses_to_one_cluster() -> None:
    links = [_link("meal", off, carbs_g=15) for off in (10, 25, 41, 56, 72, 90)]
    nodes = _layout(links)
    assert len(nodes) == 1
    cluster = nodes[0]
    assert cluster["cluster"] is True
    assert cluster["kind"] == "meal"
    assert len(cluster["links"]) == 6
    _assert_on_canvas(nodes)


def test_correction_stack_collapses_to_one_cluster() -> None:
    links = [_link("bolus", -off, units=1.5) for off in (5, 17, 28, 39, 50)]
    nodes = _layout(links)
    assert len(nodes) == 1
    assert nodes[0]["cluster"] is True
    assert nodes[0]["side"] == "before"
    _assert_on_canvas(nodes)


def test_small_groups_stay_individual_and_do_not_overlap() -> None:
    links = [
        _link("meal", -20, carbs_g=58),
        _link("meal", -8, carbs_g=12),
        _link("meal", -5, carbs_g=10),
        _link("bolus", 3, units=6.0),
        _link("bolus", 9, units=1.5),
        _link("activity", -60, kind="walk"),
        _link("sleep", 120, score=61.0),
    ]
    nodes = _layout(links)
    assert len(nodes) == 7
    assert all(not n.get("cluster") for n in nodes)
    assert _min_pairwise_gap(nodes) >= 30
    _assert_on_canvas(nodes)


def test_near_identical_offsets_are_pushed_apart() -> None:
    # Three meals within 4 minutes of each other used to land ~26px apart.
    links = [_link("meal", off, carbs_g=15) for off in (10, 12, 14)]
    nodes = _layout(links)
    assert len(nodes) == 3
    assert _min_pairwise_gap(nodes) >= 30
    _assert_on_canvas(nodes)


def test_layout_is_deterministic() -> None:
    links = [
        _link("meal", -20, carbs_g=58),
        _link("bolus", 3, units=6.0),
        _link("sleep", 120, score=61.0),
    ] + [_link("meal", off, carbs_g=15) for off in (10, 25, 41, 56)]
    assert _layout(links) == _layout(links)
