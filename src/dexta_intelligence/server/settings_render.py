"""Render settings schema → template context (no FastAPI imports)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from dexta_intelligence.config import env_override_for
from dexta_intelligence.server.settings_schema import (
    FieldKind,
    FieldSchema,
    PanelSchema,
    panel_banner,
    panel_display_title,
)

if TYPE_CHECKING:
    from datetime import datetime

    from dexta_intelligence.config import Config


def mask_secret(value: str) -> str:
    return f"••••{value[-4:]}" if len(value) >= 8 else "••••"


def field_to_view(
    section_key: str,
    spec: FieldSchema,
    section: Any,
    *,
    editable: bool,
) -> dict[str, Any]:
    env_var = env_override_for(section_key, spec.name)
    raw = getattr(section, spec.name)
    is_checkbox = spec.kind == FieldKind.CHECKBOX
    value = raw
    if is_checkbox:
        display = ""
    elif spec.secret:
        display = ""
    elif value is None:
        display = ""
    elif hasattr(raw, "value") and not isinstance(raw, (str, int, float, bool)):
        display = str(raw.value)
    else:
        display = str(value)

    secret_mask = mask_secret(str(raw)) if spec.secret and env_var is None and raw else ""
    placeholder = spec.placeholder
    if secret_mask:
        placeholder = f"{secret_mask} — leave blank to keep"
    elif not placeholder and spec.secret:
        placeholder = "••••"

    return {
        "name": spec.name,
        "label": spec.label,
        "kind": spec.kind.value,
        "input_type": spec.input_type,
        "secret": spec.secret,
        "optional": spec.optional,
        "placeholder": placeholder,
        "hint": spec.hint,
        "autocomplete": spec.autocomplete,
        "min": spec.min,
        "max": spec.max,
        "options": [{"value": v, "label": lbl} for v, lbl in spec.options],
        "env": env_var,
        "env_set": env_var is not None,
        "disabled": env_var is not None or not editable,
        "checked": bool(value) if is_checkbox else False,
        "value": display,
        "mask": secret_mask,
    }


def panel_to_view(
    spec: PanelSchema,
    cfg: Config,
    *,
    configured: bool,
    editable: bool,
    freshness: datetime | None = None,
    saved: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    section = getattr(cfg, spec.section)
    fields = [
        field_to_view(spec.section, f, section, editable=editable) for f in spec.fields
    ]
    env_keys = [
        {"name": var, "label": label, "set": bool(os.environ.get(var))}
        for var, label in spec.env_keys
    ]
    setup_flows = [list(flow) for flow in spec.setup_flows]
    setup_links = [{"label": link.label, "url": link.url} for link in spec.setup_links]
    return {
        "key": spec.key,
        "section": spec.section,
        "title": panel_display_title(spec),
        "category": spec.category,
        "tier": spec.tier,
        "unofficial": spec.tier == "unofficial",
        "banner": panel_banner(spec),
        "note": spec.note,
        "connector": spec.connector,
        "configured": configured,
        "freshness": freshness.isoformat(sep=" ", timespec="minutes") if freshness else None,
        "editable": editable,
        "saved": saved,
        "error": error,
        "fields": fields,
        "env_keys": env_keys,
        "setup_flows": setup_flows,
        "setup_links": setup_links,
    }
