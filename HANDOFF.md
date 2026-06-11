# dexta-intelligence — build handoff

_Last updated: 2026-06-10. Snapshot of the scaffolding session for the
open-source agentic health-intelligence harness._

## TL;DR

The foundation, the credibility core, the physiological model, and three
working sensor connectors are **built, typed, and tested** (254 tests, ruff +
mypy-strict clean across 28 source files). What remains is the part that turns
the pieces into a running product: persistence, the agents themselves, the CLI,
and the MCP server.

> **The session crashed mid-Libre-connector.** The only artifact of that work is
> the `libre = ["pylibrelinkup>=0.10"]` extra in `pyproject.toml`. No
> `connectors/libre.py`, no `LibreConfig`, no tests yet. That is the first thing
> to pick back up (details in §"Pick up exactly here").

## How to verify state in 30 seconds

```bash
cd ~/Desktop/dexta-intelligence
.venv/bin/ruff check src tests        # → All checks passed!
.venv/bin/mypy                        # → Success: no issues found in 28 source files
.venv/bin/pytest                      # → 254 passed
```

If those three are green, the picture below is accurate.

## What is DONE (built + tested)

| Area | File(s) | Tests | Notes |
|---|---|---|---|
| Package scaffold | `pyproject.toml`, `README.md`, `.gitignore`, all `__init__.py` | — | Hatchling, ruff (line 100), mypy strict, py3.11+. Optional extras: `llm`, `nightscout`, `dexcom`, `libre`, `whoop`, `postgres`, `all` |
| Core models | `models.py` | (via connectors) | Frozen pydantic, UTC-enforced timestamps. Events: Glucose/Insulin/Meal/Activity/Sleep/Recovery/Device + **PredictionEvent** (oref0/Loop predBG curves). Memory: Finding/Hypothesis/CoverageStats |
| Config | `config.py` | yes | `dexta.toml` + env overrides. Sections: Data, Nightscout, **Whoop**, **Dexcom**, LLM, Analysis |
| Faithfulness guard | `guard/faithfulness.py` | (self) | Numeric-faithfulness audit — LLM prose can't introduce numbers absent from evidence |
| BYOM LLM factory | `llm/factory.py` | — | `init_chat_model` by role; **OpenRouter special-cased** (one key → every model). No provider client outside this module |
| Cold-start gating | `coldstart.py` | — | Capability gates by data coverage |
| Agent contract | `agents/base.py` | — | `DextaAgent` protocol, `DataRequirement`, `AgentContext`, `AgentRegistry` |
| Storage **contract** | `store/port.py` | — | `StoragePort` protocol only — **no backend implementation yet** |
| **Stats core** | `stats/core.py` | 96 | stdlib-only: Pearson/Spearman, Welch-t, Mann-Whitney, Cohen's d / Hedges' g / Cliff's delta, bootstrap CIs. Hardened vs donor (undefined → `None`, not fake `0.0`) |
| **Rigor layer** | `stats/rigor.py` | 25 (+2 calib) | permutation p, BH-FDR, split-half replication, power gate, `assess()` verdict. Calibration verified (null reject 4.5% @ α=.05; FDR controlled) |
| `stats` package API | `stats/__init__.py` | — | Re-exports high-traffic entry points |
| **oref0 math port** | `analytics/oref.py` | 39 | exp/bilinear insulin curves, IOB/activity, BGI, deviations, COB, all 4 predBG curves + eventualBG. **Verified bit-for-bit against oref0 JS under node.** MIT provenance + "not a dosing algorithm" notice |
| **Synthetic golden data** | `testing/synthetic.py` | 10 | Deterministic CGM generator + 4 composable planted effects + null sets + ground-truth `ScenarioManifest`. The eval substrate |
| Connector contract | `connectors/base.py` | — | `Connector` + **`RealtimeConnector`** (`current()` — the live MCP surface) + `HealthReport` + `NormalizedBatch` |
| **Nightscout connector** | `connectors/nightscout.py` | 27 | entries/treatments/devicestatus incl. oref0 **and** Loop prediction curves; descending-cursor pagination; fixtures |
| **Dexcom Share connector** | `connectors/dexcom.py` | 33 | Ported from old Dexter `pydexcom` service. Implements `RealtimeConnector` (`current()`); ~24h Share history cap clamped + documented |
| **Whoop connector** | `connectors/whoop.py` | 24 | v2 sleep/recovery/workout parsers, token refresh, nextToken pagination, skips unscored records; fixtures |

Empty placeholder packages (just `__init__.py`, awaiting code): `memory/`,
`providers/`, `timeline/`, `workflows/`.

## What is NOT done (the build queue)

In rough dependency order:

1. **Libre connector** — see §"Pick up exactly here". (in progress when crashed)
2. **Storage backend** — `store/port.py` is a protocol with no implementation.
   Need a SQLite backend (quick-start) + the migration/schema, then Postgres.
   This unblocks everything stateful (memory, recurrence counts, sync watermarks).
3. **Sync workflow** — `workflows/sync.py`: `pull → raw upsert → normalize →
   rollups`, watermark per source in `sync_state`. Wires connectors to the store.
4. **Prediction Reconciliation Agent** — the flagship. All inputs now exist
   (oref0 curves for Tier B, predBG parsing for Tier A, rigor gate, synthetic
   eval data). Needs the store for recurrence ("similar pattern, N times").
5. **Other agents** — Discovery (port donor researcher pipeline onto rigor +
   memory), Skeptic, Observation/Pattern/Basal/Meal/Correction, Clinical Brief.
6. **CLI** — `dexta init / doctor / sync / analyze` (the README quick-start
   promises these).
7. **MCP server** — FastMCP server over the harness (glucose-over-MCP v1, the
   10-tool contract from the spec §6.1). The published Dexcom MCP is the template.
8. **More connectors** — Oura (cleanest wearable API), Dexcom official API,
   CSV upload, Apple Watch export bridge.
9. **Eval harness** — `eval/` runner producing the headline cross-model table
   (E1–E8 in spec §14).

## Pick up exactly here: the Libre connector

The plan was settled; only the code is missing. Build it to match the other
connectors exactly (`connectors/nightscout.py` is the house-style reference).

- **Library:** [`pylibrelinkup`](https://github.com/robberwick/pylibrelinkup)
  (PyPI v0.10.0, py3.11+, multi-region, actively maintained). Extra already added.
- **Two planes (the key design point from the session):**
  - `latest()` → implement `RealtimeConnector.current()` (~1-min freshness, live MCP)
  - `graph()` (12h) + `logbook()` (~2wk) → `pull()` (batch → brain)
- **Gotcha to bake in:** Libre's trend enum is a **subset** — only
  `SingleDown / FortyFiveDown / Flat / FortyFiveUp / SingleUp` (no double arrows;
  Abbott's cloud doesn't emit them). Clamp the trend mapping accordingly.
- **Add `LibreConfig`** to `config.py` (additive, NightscoutConfig style):
  email/password + region enum + patient id; env overrides
  (`LIBRE_EMAIL` / `LIBRE_PASSWORD` / `LIBRE_REGION`). Lazy-import `pylibrelinkup`
  with the `pip install 'dexta-intelligence[libre]'` error (see `llm/factory.py`).
- **Tests:** `tests/test_libre_connector.py` — pure conversion tests against
  stub readings (UTC, trend clamping, value mapping) + connector tests with a
  stubbed client (no network), mirroring `test_dexcom_connector.py`.
- **Gate:** ruff + mypy strict + the new test file must pass; don't break the
  existing 254.

## Notable decisions made this session

- **OpenRouter is the recommended BYOM default** — one `OPENROUTER_API_KEY`
  unlocks every model and makes the cross-model eval matrix runnable on one
  credential. Direct providers + Ollama (fully local) remain.
- **One MCP server over the whole harness, not one MCP per device.** Per-device
  MCPs would be raw data with no brain — the GlycemicGPT-shaped mistake.
- **Nightscout is a meta-driver.** Everything that uploads to NS (Loop, AAPS,
  xDrip+, Omnipod, Medtronic) arrives through the one connector — that's how we
  match GlycemicGPT's integration breadth without grinding per-platform.
- **Real-time vs batch are separate by design.** Dexcom Share (batch, laggy)
  and the live pydexcom MCP serve different needs; `RealtimeConnector` is the seam.
- **oref0 math, not our own.** Documented, MIT, battle-tested; ported verbatim
  and JS-verified. Powers Tier B reconciliation for non-looping pumps.

## Gotchas / housekeeping

- **Git is NOT initialized.** No commits exist. First real action might be
  `git init && git add -A && git commit`. (A stray `.DS_Store` is present; it's
  in `.gitignore` patterns — confirm before first commit.)
- The spec lives in the **other** repo: `~/Desktop/dexta/docs/DEXTA_OSS_TECHNICAL_SPEC.md`
  (it was updated this session: §7.1 Prediction Reconciliation Agent, OpenRouter
  in §5, expanded provider matrix in §6, Libre three-path strategy).
- Donor code to port from (read-only): the main `~/Desktop/dexta` repo, and the
  old `~/Desktop/Dexter/dex-engine` repo (source of the Dexcom connector).
- All connector additive edits to `models.py` / `config.py` / `connectors/base.py`
  were kept strictly additive so parallel work merged cleanly — keep that
  discipline for Libre's `config.py` edit.

## Build approach that worked

Scaffolding was parallelized across isolated sub-agents, each owning a disjoint
set of files against frozen interfaces (protocols in `*/base.py`, `port.py`,
`models.py`), each required to pass ruff + mypy-strict + its own tests before
finishing. Repeat this for the build queue: anything sharing only protocols can
go in parallel; storage backend + sync workflow are the serializing dependency
for the stateful agents.
