"""dexta CLI — init, doctor, sync, analyze, upload, wiki."""

from __future__ import annotations

from dexta_intelligence.cli._common import ConnectorFactory as ConnectorFactory
from dexta_intelligence.cli._common import CsvFormatHint as CsvFormatHint
from dexta_intelligence.cli._common import StoreOpener as StoreOpener
from dexta_intelligence.cli._common import build_connectors as build_connectors
from dexta_intelligence.cli._common import discovery_model as discovery_model
from dexta_intelligence.cli._common import init_config_path as init_config_path
from dexta_intelligence.cli._common import is_dexcom_configured as is_dexcom_configured
from dexta_intelligence.cli._common import is_libre_configured as is_libre_configured
from dexta_intelligence.cli._common import is_nightscout_configured as is_nightscout_configured
from dexta_intelligence.cli._common import is_tidepool_configured as is_tidepool_configured
from dexta_intelligence.cli._common import is_whoop_configured as is_whoop_configured
from dexta_intelligence.cli._common import logger as logger
from dexta_intelligence.cli._common import open_sqlite_store as open_sqlite_store
from dexta_intelligence.cli._common import resolve_config_path as resolve_config_path
from dexta_intelligence.cli.analysis import cmd_analyze as cmd_analyze
from dexta_intelligence.cli.analysis import get_registry as get_registry
from dexta_intelligence.cli.data import cmd_doctor as cmd_doctor
from dexta_intelligence.cli.data import cmd_init as cmd_init
from dexta_intelligence.cli.data import cmd_sync as cmd_sync
from dexta_intelligence.cli.data import cmd_upload as cmd_upload
from dexta_intelligence.cli.intelligence import cmd_ask as cmd_ask
from dexta_intelligence.cli.intelligence import cmd_brief as cmd_brief
from dexta_intelligence.cli.intelligence import cmd_demo as cmd_demo
from dexta_intelligence.cli.intelligence import cmd_goals as cmd_goals
from dexta_intelligence.cli.intelligence import cmd_wiki as cmd_wiki
from dexta_intelligence.cli.main import build_parser as build_parser
from dexta_intelligence.cli.main import main as main
from dexta_intelligence.cli.serve import cmd_serve as cmd_serve

__all__ = ["main"]
