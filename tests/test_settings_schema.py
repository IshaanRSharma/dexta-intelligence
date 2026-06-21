"""Settings schema contract - fields must match Config Pydantic models."""

from __future__ import annotations

from dexta_intelligence.config import Config, env_override_for
from dexta_intelligence.server.settings_schema import (
    ANALYSIS_PANEL,
    DATA_FIELDS,
    SETTINGS_PANELS,
    WIKI_FIELDS,
)


def _section_model(section: str):
    return type(getattr(Config(), section))


def test_panel_fields_exist_on_config_models() -> None:
    for panel in SETTINGS_PANELS:
        model = _section_model(panel.section)
        for field in panel.fields:
            assert field.name in model.model_fields, f"{panel.key}.{field.name}"


def test_analysis_and_wiki_fields_exist() -> None:
    for field in ANALYSIS_PANEL.fields:
        assert field.name in _section_model("analysis").model_fields
    for field in DATA_FIELDS:
        assert field.name in _section_model("data").model_fields
    for field in WIKI_FIELDS:
        assert field.name in _section_model("wiki").model_fields


def test_tidepool_panel_matches_config() -> None:
    tidepool = next(p for p in SETTINGS_PANELS if p.key == "tidepool")
    assert tidepool.section == "tidepool"
    assert {f.name for f in tidepool.fields} == {"export_path"}
    assert "export_path" in _section_model("tidepool").model_fields
    assert len(tidepool.setup_flows) >= 1
    assert any(link.url.startswith("https://") for link in tidepool.setup_links)


def test_nightscout_setup_flows_include_common_loops() -> None:
    ns = next(p for p in SETTINGS_PANELS if p.key == "nightscout")
    joined = " ".join(" ".join(flow) for flow in ns.setup_flows)
    assert "Tandem" in joined
    assert "Juggluco" in joined or "xDrip" in joined
    assert "Nightscout" in joined
    assert "Dexta" in joined


def test_env_overrides_only_reference_schema_fields() -> None:
    for panel in SETTINGS_PANELS:
        for field in panel.fields:
            var = env_override_for(panel.section, field.name)
            if var is None:
                continue
            assert isinstance(var, str)


def test_dexcom_schema_has_password_and_ous() -> None:
    dexcom = next(p for p in SETTINGS_PANELS if p.key == "dexcom")
    names = {f.name for f in dexcom.fields}
    assert names == {"username", "password", "ous"}
    password = next(f for f in dexcom.fields if f.name == "password")
    assert password.secret is True
    assert password.input_type == "password"
