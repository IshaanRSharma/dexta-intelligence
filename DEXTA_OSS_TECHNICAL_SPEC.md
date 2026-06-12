# Dexta OSS — Technical Build Specification

**Status:** v1.1 — engineering source of truth for the open-source platform build
**Audience:** contributors + AI sub-agents executing workstreams
**Companion:** the product vision doc (MVP Build Specification v1). This document is the *how*.
**One-line frame:** an **AI clinic brain for Type 1 diabetes** — an agent harness (memory + rigor + faithfulness guard) sitting on a pluggable sensor plane (CGM, pump, wearable, file upload), self-hosted, BYO-model.

---

## 0. Ground rules

1. **We extract, we don't rewrite.** ~60% of the spec already exists in this repo as debugged, production-tested code. Every component below names its donor module. Greenfield is only allowed where the donor table says "NEW".
2. **Vanilla-first.** `pipx install dexta && dexta init && dexta run` must work with zero config beyond a Nightscout URL and one LLM API key. Everything else has defaults. Postgres, per-agent models, plugins are opt-in upgrades, never prerequisites.
3. **Cold-start is a first-class state.** Every capability declares its minimum data requirement and degrades with an explicit warning, never silently and never with fabricated confidence.
4. **The faithfulness guard is non-negotiable.** Any LLM-authored prose passes the numeric-traceability gate or the system falls back to deterministic output. This pattern already exists in three places; it becomes one shared module.
5. **Safety boundary:** pattern analysis, research context, discussion support, hypothesis generation — yes. Dosing, medication adjustment, treatment recommendation — never. Port `agents/safety/pha_output.py` as the final output gate on every surface.

---

## 1. Donor map — current repo → OSS component

This is the single most important table in the document. Build order, estimates, and sub-agent task boundaries all derive from it.

| OSS spec component | Donor in current repo | Reuse level |
|---|---|---|
| Sensor plane (MCP tool contract) | **`mcp-server-dexcom-health` v0.2 (our own published PyPI package — 10 FastMCP tools)** | The contract template |
| Nightscout provider + MCP (**pump unlock: Tandem t:slim X2 via tconnectsync → Nightscout**) | — | **NEW** (small; REST + token) |
| Dexcom provider | `mcp_server_dexcom/server.py` (pydexcom) + `pipeline/pod_worker.py` ingest | Port/adopt |
| Libre provider + MCP (LibreLinkUp) | — | **NEW** (unofficial API; also covered indirectly via Nightscout/CSV) |
| Whoop provider | `pipeline/whoop_client.py` (`WhoopClient`, `flatten_sleep/recovery/workout/cycle`) | **Port** (drop Supabase + web OAuth flow; token-based config) |
| CSV/file upload connector (Clarity, LibreView export) | `parse_external_data` pattern in the MCP server | **NEW** (small) |
| Raw event store | `glucose_readings` + `events` patterns in `pipeline/database.py` | Schema redesign, code pattern reuse |
| Clinical timeline (normalize) | `pipeline/pod_signals.py` (`normalize_reading_points`, dedupe keys) | Port + extend to insulin/meal events |
| Rollups (15m/hourly/daily/weekly) | `agents/clinical/analytics.py` (TIR/TAR/TBR/CV/GMI/excursions/windows) + `pipeline/jobs/dexcom_daily.py` `compute_daily_summary` | **Port nearly verbatim** |
| AGP computation | `agents/clinical/agp.py` | Port verbatim |
| What Changed (30/90/180/365) | `agents/clinical/trajectory.py` (`compute_longterm`) + `analytics.compute_what_changed` | Port + parametrize windows |
| Observation Agent | `agents/monitoring/observer.py` + `analyzer.py` (facts-only feature build) + insights detectors' deterministic cores | Port |
| Pattern Agent | `agents/coach/correlators/*` (sleep, workout, recovery, weekday, cause_recurrence) + `agents/insights/detectors/*` (tod_baseline_drift, cpd_overnight_avg, post_meal_outlier, episode_frequency) | **Port** |
| Basal Agent | `analytics.py` `_overnight_basal_signal`, `_safety` (rebound logic) | Port + **extend with Nightscout `treatments` (real basal/temp-basal data)** |
| Meal Agent | `analytics.py` `_post_meal_signal`, `_daily_meal_excursions`, `_classify_meal` (timing/counting/dose differential) | Port + extend with real carb/bolus entries |
| Correction Agent | `_safety` rebound estimate (lows-after-highs) | Port + **NEW** (correction-effectiveness needs bolus data — Nightscout unlocks it) |
| Research Agent (guidelines/literature) | `agents/tools/clinical_kb.py` (curated KB) | Port + extend corpus |
| Skeptic Agent | — (closest: reflect node pattern in `agents/nodes/reflect.py`) | **NEW** (uses stats core, below) |
| Discovery Agent | `agents/researcher/agent.py` (plan → test → reflect → finalize, tool budget, 24h cache) + `researcher/tools.py` (6 stat tools) | **Port** + add rigor layer |
| Stats core (+ rigor) | `agents/coach/correlators/_stats.py` (Cohen's d, min-N) + `researcher/tools.py` | Port + **NEW: FDR correction, permutation tests, split-half replication** |
| Faithfulness guard | `agents/clinical/brief.py` `_flatten_numbers` / `_has_untraceable` (5% tolerance, allowed-ints whitelist) + the parallel guards in insights agent & coach | **Unify into one module** |
| Clinical Brief Agent | `agents/clinical/brief.py` (evidence bundle → LLM rank/explain → guard → deterministic fallback → provenance meta) | **Port nearly verbatim** |
| Explain This Spike | reasoner evidence pattern in `agents/monitoring/reasoner.py` + `event_proximity` tool | Recompose |
| Agent memory (findings DB) | `pod_insights` schema (`scripts/setup_pod_insights.sql`) + `Insight`/`CoachFinding` DTOs | **Unify into one `Finding` record** |
| Wiki layer | — | **NEW** (generator over findings store) |
| Embeddings/retrieval | — | **NEW** (small; local-first) |
| BYOM (model factory) | 18 `ChatAnthropic` call sites (inventoried §5) | **Refactor to one factory** |
| Safety framework | `agents/safety/pha_output.py` + reflect-node dosing hard-fail rules | Port |
| Orchestration | `agents/graph.py` LangGraph (interactive) + `agents/insights/registry.py` `@register`/`run_all` (batch fan-out) | Port both patterns |
| TUI | — | **NEW** (Textual) |
| Report export | clinical brief `copySummary` content model | Port content model, new renderer |
| Deep Analysis | composition of the above | Composition only |

**What we deliberately do NOT port:** Supabase coupling, the SvelteKit app, share-tokens/followers/audit (multi-user clinical sharing is out of OSS-MVP scope), the Whoop *web OAuth flow* (replaced by token-based config; the data client IS ported), the LightGBM forecasting stack (`pipeline/training/`, `pipeline/forecast/` — valuable but heavy; Phase 7+), Resend email, Railway configs.

---

## 2. Repository layout (new repo: `dexta-oss`)

```
dexta/
├── pyproject.toml              # single installable package: `pip install dexta`
├── dexta/
│   ├── config.py               # ConfigLoader: dexta.toml + env + DEFAULTS (§4)
│   ├── llm/
│   │   ├── factory.py          # BYOM: get_model(role) → BaseChatModel (§5)
│   │   └── roles.py            # role registry + per-role defaults
│   ├── store/
│   │   ├── port.py             # StoragePort protocol — ALL persistence goes through this
│   │   ├── postgres_backend.py # REFERENCE backend (Postgres-native: JSONB, TIMESTAMPTZ, pgvector)
│   │   ├── sqlite_backend.py   # quick-start backend (zero-setup; `dexta migrate --to-postgres` later)
│   │   ├── schema.py           # table definitions + migrations (§3)
│   │   └── models.py           # typed records: GlucoseEvent, InsulinEvent, MealEvent,
│   │                           #   ActivityEvent, DeviceEvent, Rollup, Finding, Hypothesis
│   ├── providers/              # raw API clients — shared by connectors AND mcp_servers (§6)
│   │   ├── nightscout.py       # REST + token (entries/treatments/devicestatus)
│   │   ├── dexcom.py           # pydexcom Share (port)
│   │   ├── libre.py            # LibreLinkUp (unofficial)
│   │   └── whoop.py            # port of pipeline/whoop_client.py
│   ├── connectors/             # batch ingest: provider → raw_events → timeline (§6)
│   │   ├── base.py             # Connector protocol: pull(since) → list[RawEvent]
│   │   ├── nightscout.py
│   │   ├── dexcom.py
│   │   ├── libre.py
│   │   ├── whoop.py
│   │   └── file_upload.py      # Clarity/LibreView CSV → raw_events
│   ├── mcp_servers/            # live FastMCP tool servers, one per provider (§6)
│   │   ├── _contract.py        # the uniform glucose-over-MCP tool spec (v1 = our published server)
│   │   ├── nightscout.py       # + pump tools: get_boluses, get_basal, get_iob
│   │   ├── libre.py
│   │   └── whoop.py            # dexcom = already published as mcp-server-dexcom-health
│   ├── timeline/
│   │   ├── normalize.py        # raw_events → clinical timeline (port pod_signals)
│   │   └── rollups.py          # port analytics.py metric kernels, windowed
│   ├── analytics/              # deterministic, LLM-free. Ports of:
│   │   ├── metrics.py          #   TIR/TAR/TBR/CV/GMI/mean (analytics.py kernels)
│   │   ├── agp.py              #   agp.py verbatim
│   │   ├── windows.py          #   time-of-day windows, opportunities, projections
│   │   ├── basal_bolus.py      #   overnight drift, dawn, post-meal classify, safety/rebound
│   │   └── trajectory.py       #   what-changed + long-term bins (trajectory.py)
│   ├── stats/
│   │   ├── core.py             # port _stats.py + researcher/tools.py (6 tools)
│   │   └── rigor.py            # NEW: BH-FDR, permutation p, split-half replication, power gate
│   ├── guard/
│   │   ├── faithfulness.py     # unified _flatten_numbers/_has_untraceable
│   │   └── safety.py           # port pha_output sanitizer + dosing hard-fail
│   ├── memory/
│   │   ├── findings.py         # unified Finding store (create/supersede/retrieve)
│   │   ├── embeddings.py       # local embedding + cosine retrieval
│   │   └── wiki.py             # NEW: findings → markdown artifacts in wiki/
│   ├── agents/
│   │   ├── base.py             # DextaAgent protocol: run(ctx) → list[Finding] (§7)
│   │   ├── registry.py         # @register + run_all (port insights/registry.py)
│   │   ├── observation.py
│   │   ├── pattern.py          # wraps ported correlators + detectors
│   │   ├── basal.py
│   │   ├── meal.py
│   │   ├── correction.py
│   │   ├── research.py         # clinical KB retrieval
│   │   ├── skeptic.py          # NEW (§8)
│   │   ├── discovery.py        # port researcher plan→test→reflect→finalize
│   │   └── brief.py            # port clinical brief
│   ├── workflows/
│   │   ├── deep_analysis.py    # batch fan-out over agents + skeptic pass + memory write
│   │   ├── explain_spike.py
│   │   ├── what_changed.py
│   │   └── sync.py             # connector pull → normalize → rollup
│   ├── coldstart.py            # data-sufficiency gates (§9)
│   ├── tui/                    # Textual app: dexta run
│   └── cli.py                  # dexta init|sync|analyze|explain|brief|wiki|run
├── tests/                      # §11
│   ├── golden/                 # synthetic datasets with planted ground-truth patterns
│   ├── fixtures/               # recorded Nightscout/Dexcom API responses
│   └── ...
├── eval/                       # the reportable benchmark (§14) — separate from tests/
│   ├── datasets/               # synthetic generator + null sets + regime-change labeler
│   ├── metrics/                # faithfulness, consistency, recall, robustness, safety
│   ├── runners/                # cross-model matrix runner (BYOM factory)
│   └── report.py               # emits THE results table (markdown + JSON)
├── wiki/                       # generated knowledge base (gitignored per-user)
├── docker-compose.yml          # optional: postgres + dexta service
└── docs/
```

**Key decision — storage: Postgres-native, SQLite quick-start.** The schema is *designed for Postgres* (TIMESTAMPTZ, JSONB, pgvector for embeddings, partial indexes); Postgres is the reference backend — CI runs the full suite against it, docker-compose ships it, and it's the documented deployment for real longitudinal use. SQLite exists purely as the zero-setup on-ramp (`dexta init` with no DATABASE_URL → local file DB; embeddings fall back to float-blob + numpy cosine). `dexta migrate --to-postgres` moves a quick-start DB into Postgres losslessly (raw_events replay makes this trivial). The `StoragePort` protocol is the seam; nothing above `store/` may import sqlite3 or psycopg directly.

**Key decision — orchestration:** two modes, both ported:
- **Batch (Deep Analysis, sync):** blackboard pattern — agents never call each other; they read the store + memory and write `Finding`s. This is the `registry.run_all` pattern from insights, and it parallelizes trivially.
- **Interactive (Ask/chat, later phase):** the LangGraph plan→execute→reflect graph from `agents/graph.py`. Not in MVP critical path.

---

## 3. Storage schema

### Layer 1 — `raw_events` (immutable)
```sql
raw_events(
  id, source TEXT,            -- 'nightscout' | 'dexcom' | 'csv'
  source_id TEXT,             -- provider's id, for idempotent re-ingest
  source_ts TIMESTAMPTZ,
  payload JSON,               -- verbatim provider record
  ingested_at TIMESTAMPTZ,
  UNIQUE(source, source_id)
)
```
Never updated, never deleted. Re-normalization is always possible.

### Layer 2 — clinical timeline (typed, normalized)
```sql
glucose_events(ts, mg_dl, trend, raw_event_id)
insulin_events(ts, kind,            -- 'bolus' | 'basal' | 'temp_basal' | 'suspend'
               units, duration_min, raw_event_id)
meal_events(ts, carbs_g, protein_g, fat_g, note, raw_event_id)
activity_events(ts, kind, duration_min, intensity, strain, raw_event_id)
sleep_events(ts_start, ts_end, duration_min, score, stages JSONB, raw_event_id)
recovery_events(ts, score, hrv_ms, rhr_bpm, raw_event_id)
device_events(ts, kind, note, raw_event_id)   -- site change, sensor start, battery
```
Nightscout mapping: `entries` → glucose_events; `treatments` → insulin/meal events (eventType: Bolus, Carb Correction, Temp Basal, Suspend); `devicestatus` → device_events + IOB snapshots. **This is the unlock the current repo lacks: real insulin data** (for us: t:slim X2 → tconnectsync → Nightscout). Whoop mapping: port the `flatten_*` helpers from `pipeline/whoop_client.py` → sleep/recovery/activity events. Every event keeps `raw_event_id` provenance. All timestamps TIMESTAMPTZ, all flexible payloads JSONB (Postgres-native; SQLite backend emulates).

### Layer 3 — rollups
```sql
rollups(period TEXT,          -- '15m' | '1h' | '1d' | '1w'
        period_start, n, mean, sd, cv, tir, tar, tar2, tbr, tbr2, gmi,
        excursion_count, correction_count, bolus_units, basal_units, carbs_g)
```
Computed incrementally on sync; recomputable from timeline at any time (rollups are a cache, not truth).

### Layer 4 — agent memory
```sql
findings(
  id, agent TEXT, kind TEXT, scope TEXT,
  headline TEXT, body_md TEXT,
  evidence JSON,              -- numbers the guard checks prose against
  stats JSON,                 -- {effect_size, n, p_perm, q_fdr, replicated: bool}
  confidence REAL, status TEXT,   -- 'active' | 'superseded' | 'rejected' | 'dismissed'
  superseded_by, skeptic_notes TEXT,
  window_start, window_end, created_at,
  embedding BLOB
)
hypotheses(id, statement, status,    -- 'open' | 'supported' | 'refuted' | 'stale'
           source_finding_id, tests JSON, created_at, updated_at)
reports(id, kind,                    -- 'deep_analysis' | 'brief' | 'explain_spike'
        content JSON, created_at)
```
`findings` is the unification of `pod_insights` + `Insight` + `CoachFinding` + clinical-brief dict (the current repo has four shapes; the OSS gets one). The `stats` block is new and mandatory for any quantitative claim — it's what the Skeptic and the rigor layer fill in.

---

## 4. Config & vanilla defaults

Single file `~/.dexta/dexta.toml` (created by `dexta init`, an interactive 3-question wizard: Nightscout URL, token, one LLM key). Everything below is a working default:

```toml
[data]
backend = "sqlite"                     # quick-start default; set DATABASE_URL or backend="postgres" for the reference deployment
path = "~/.dexta/dexta.db"
# database_url = "postgresql://dexta:dexta@localhost:5432/dexta"   # docker compose default

[connectors.nightscout]
url = ""                               # the ONE required value
token = ""

[connectors.whoop]                     # optional
access_token = ""                      # token-based; no web OAuth flow in OSS

[connectors.libre]                     # optional, opt-in (unofficial API — see §6.2)
email = ""
password = ""
region = "us"

[llm]
provider = "anthropic"                 # anthropic | openai | ollama | openrouter | ...
model = "claude-sonnet-4-20250514"
# api key from env: ANTHROPIC_API_KEY / OPENAI_API_KEY / ...

[llm.roles]                            # optional per-role overrides (all fall back to [llm])
# discovery = { provider = "anthropic", model = "claude-sonnet-4-20250514" }
# skeptic   = { provider = "ollama",    model = "llama3" }
# brief     = { provider = "openai",    model = "gpt-5" }

[analysis]
target_low = 70
target_high = 180
deep_analysis_window_days = 90

[safety]
guard = "strict"                       # numeric-traceability guard; cannot be disabled below "strict" for clinical surfaces
```

`dexta doctor` validates: connector reachable, key valid, data coverage report, model reachable.

---

## 5. BYOM — the model factory

Current state (inventoried): **18 `ChatAnthropic` instantiation sites** across `agents/`, three selection patterns — `settings.anthropic_model`, hardcoded `"claude-haiku-4-5"`, and a Sonnet upgrade rule for user-facing Domain Expert / brief calls. Zero non-Anthropic clients.

Design:

```python
# dexta/llm/factory.py
from langchain.chat_models import init_chat_model

ROLE_DEFAULTS = {
    # role         (tier,     temperature, max_tokens)
    "plan":        ("fast",    0.0, 1024),   # structured routing
    "observation": (None,      None, None),  # deterministic — NO model
    "pattern":     (None,      None, None),  # deterministic — NO model
    "discovery":   ("primary", 0.2, 1800),
    "skeptic":     ("primary", 0.0, 1200),
    "research":    ("fast",    0.2, 1200),
    "brief":       ("primary", 0.2, 2200),
    "polish":      ("fast",    0.2, 600),
    "explain":     ("primary", 0.2, 1500),
}

def get_model(role: str) -> BaseChatModel:
    cfg = config.llm_for_role(role)        # roles override → [llm] → tier default
    return init_chat_model(cfg.model, model_provider=cfg.provider,
                           temperature=cfg.temperature, max_tokens=cfg.max_tokens)
```

Rules:
- `init_chat_model` (LangChain) gives Anthropic/OpenAI/Ollama/Groq/Mistral/etc. for free — we do not write per-provider clients.
- **`provider = "openrouter"` is special-cased** onto the OpenAI-compatible endpoint
  (`base_url=https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`). This is the recommended
  BYOM quick-start: one key → Claude/GPT/Gemini/Llama/free models, and it makes the §14
  cross-model eval matrix runnable with a single credential. Direct providers remain for
  users who already have keys; Ollama remains the fully-local path.
- Two tiers, `primary` and `fast`, both defaulting to the single configured model in vanilla mode. Per-agent models are a roles-table override, not a requirement.
- **No agent module may import a provider client directly.** CI greps for `ChatAnthropic|ChatOpenAI` outside `llm/` and fails.
- Structured-output calls (plan, reflect, discovery JSON) go through the same factory; the strict-JSON retry wrapper from `researcher/agent.py:_strict_json_call` is ported into the factory as `get_json_model(role, schema)`.

---

## 6. Sensor plane — providers, connectors, MCP servers

Three layers sharing one API client per source:

```
providers/   raw API clients (auth, pagination, flatten)        ── shared code
connectors/  batch ingest: provider → raw_events → timeline     ── feeds the brain
mcp_servers/ live FastMCP tool servers per provider             ── feeds ANY agent (ours or anyone's)
```

### 6.1 The MCP tool contract (v1 = our published server)

`mcp-server-dexcom-health` v0.2 (our own PyPI package, MIT, FastMCP, stdio default / SSE in-app on :8001, env-var creds) already defines the de-facto contract — **10 tools**: `get_current_glucose`, `get_glucose_readings`, `get_statistics`, `get_status_summary`, `detect_episodes`, `get_episode_details`, `analyze_time_blocks`, `check_alerts`, `export_data`, `get_agp_report`. It also has the `parse_external_data` hook (tools accept injected readings), which becomes the CSV path.

`mcp_servers/_contract.py` freezes this as **glucose-over-MCP v1**. Every sensor MCP implements the same 10 tools, so an agent (Claude Desktop, Cursor, our harness, anyone's) is source-agnostic. Pump-capable sources add the **insulin extension**: `get_boluses`, `get_basal_timeline`, `get_carb_entries`, `get_iob`.

### 6.2 Provider matrix

| Source | API | What it yields | Status / risk |
|---|---|---|---|
| **Nightscout** | REST `/api/v1/{entries,treatments,devicestatus}.json`, token auth | glucose + **bolus/carb/temp-basal/suspend + IOB** | Official-ish, stable, no reverse-engineering. **First.** |
| **Tandem t:slim X2** (our pump) | via **tconnectsync → Nightscout** (documented recipe, not our code) | real pump data lands in `treatments` | We document the recipe; Nightscout MCP/connector picks it up for free |
| **Dexcom** | pydexcom (Share API) | glucose, 24h window | Already shipped as our published MCP server; connector port for history |
| **Freestyle Libre** | LibreLinkUp (unofficial) | glucose | ToS-gray, breaks occasionally — ship as opt-in with a clear banner. Most Libre DIY users are *already* on Nightscout via xDrip+/Juggluco, and LibreView CSV covers the rest, so this is a convenience, not a dependency |
| **Whoop** | official OAuth API — port `pipeline/whoop_client.py` | sleep, recovery/HRV, strain, workouts | Clean; token-based config in OSS (no web OAuth flow) |
| **Oura** | official REST API v2, personal access token | sleep stages, readiness, HRV, temperature, activity | Cleanest wearable API there is — ideal first community driver |
| **Dexcom official API** | OAuth2, `/egvs` endpoints | glucose history (~1–3h delay, no real-time) | ToS-clean complement to pydexcom Share; sandbox available |
| **Apple Watch / HealthKit** | **no cloud API exists** — export bridge only | HR, HRV, sleep, workouts | Path: Health Auto Export app → webhook/JSON ingest, or CSV. Documented recipe, not a connector we own |
| **Garmin / Fitbit** | official OAuth APIs | HR, sleep, activity | Tier 4 — community drivers once the protocol proves itself |
| **File upload** | Clarity / LibreView CSV export | glucose history (one-shot bulk) | Zero-integration on-ramp: `dexta upload my_export.csv` |

**Driver expansion strategy (the GlycemicGPT lesson, inverted):** breadth via Nightscout-as-meta-driver
(everything that uploads to NS — xDrip+, Loop, AAPS, Omnipod, Medtronic — arrives through one connector)
plus a deliberately cheap plug-in seam, NOT solo grinding through 15 integrations. A driver is one
`connectors/<source>.py` file implementing the frozen `Connector` protocol + recorded fixtures + tests.
Each missing driver is a labeled `good-first-issue` with the Oura/Whoop connectors as templates — the
contribution surface that makes the repo a community project instead of a solo demo. Sources with
real-time reads (pydexcom Share) additionally implement `current()` (optional `RealtimeConnector`
protocol) which backs the live MCP tools.

Libre coverage strategy, explicitly: **three paths** — (1) already-on-Nightscout (majority of DIY Libre users) → Nightscout connector; (2) LibreView CSV upload; (3) direct LibreLinkUp MCP for those who want live data. We never *depend* on the reverse-engineered path.

### 6.2.1 Unofficial-API connector tier (reverse-engineered, opt-in)

pydexcom (Dexcom Share) and LibreLinkUp set the precedent: reverse-engineered
clients are *valuable on-ramps* when handled honestly. We formalize them as a
tier rather than ad-hoc exceptions:

| Candidate | Wraps | Unlocks | Notes |
|---|---|---|---|
| **pydexcom** (shipped) | Dexcom Share | real-time glucose, no official-API delay | the template for this tier |
| **LibreLinkUp** (shipped) | Libre cloud | live Libre readings | ToS-gray, banner'd |
| **tconnectsync (direct)** | Tandem t:connect | pump events without running a Nightscout site | today documented only as the →Nightscout recipe; a direct connector removes the NS prerequisite for Control-IQ users |
| **carelink client** (e.g. carelink-python-client) | Medtronic CareLink | 670G/780G pump + CGM data | largest unreached pump population; fragile auth, EU/US split |
| **Tidepool API** | Tidepool platform | multi-device normalized history | actually an *official* open API — cheap win, listed here because it arrives via the same evaluation process |

Tier rules (binding):
1. **Opt-in only, never a dependency** — core capabilities must all be reachable via Nightscout or CSV; this tier is convenience.
2. **Labeled in UI and docs** — each carries an "unofficial API — may break without notice" banner, same as Libre today.
3. **Fixture-tested like every connector** — recorded sanitized responses; CI never hits the live service.
4. **Same `Connector` protocol, same normalize path** — an unofficial source is one file in `connectors/`, indistinguishable downstream.
5. **Read-only forever** — no tier member may ever gain a write/command surface.
6. Each is a labeled `good-first-issue`; the pydexcom connector is the reference implementation.

### 6.3 Connector contract (batch side)

```python
class Connector(Protocol):
    source: str
    def check(self) -> HealthReport            # reachability, auth, latest data ts
    def pull(self, since: datetime) -> list[RawEvent]   # idempotent, paginated
```

- `sync` workflow: `pull → raw_events upsert (UNIQUE source,source_id) → normalize new rows → incremental rollups`. Watermark per source in `sync_state`.
- File upload is a degenerate connector: `pull` reads the file once; same normalize path, same provenance.
- Every provider ships **recorded-fixture tests** (sanitized JSON in `tests/fixtures/`) — CI never needs live credentials.

### 6.4 Why both planes matter for the "AI clinic brain" story

The connector plane gives the brain longitudinal memory (Postgres timeline, years of data). The MCP plane makes the same sensors available to *any* agent ecosystem — which is the adoption wedge: someone who never installs the harness can still `uvx mcp-server-nightscout-health` into Claude Desktop, and the harness is the natural upgrade. The published Dexcom server proves the model; Nightscout (pump-aware) is the strategically important sibling.

---

## 7. Agent contract & the blackboard

```python
class AgentContext:
    store: StoragePort
    memory: FindingStore          # retrieve(query|kind|window) → past findings
    window: tuple[date, date]
    gates: ColdStartReport        # what this run is allowed to claim (§9)

class DextaAgent(Protocol):
    name: str
    requires: DataRequirement     # declared minimums — checked BEFORE run
    def run(self, ctx: AgentContext) -> list[Finding]
```

- Agents are registered with the `@register` decorator (direct port of `insights/registry.py`), executed by `run_all` with per-agent exception isolation, **in parallel** (they share read-only context and write only their own findings).
- Deterministic agents (observation, pattern, basal, meal, correction) never touch the LLM factory. LLM-bearing agents (discovery, skeptic, research, brief) must pass every claim through `guard.faithfulness` before returning.
- This contract **is** the Phase-2 plugin interface — community agents implement the same protocol, so the plugin system costs nothing extra at MVP time beyond keeping the contract clean.

### Mapping spec agents → donors
| Agent | Donor | Notes |
|---|---|---|
| Observation | `monitoring/observer+analyzer` facts core | facts only, no interpretation, no LLM |
| Pattern | correlators (sleep/workout/recovery/weekday/cause) + detectors (tod-drift, change-point, post-meal outlier, episode frequency) | each donor becomes a sub-check; all routed through `stats.rigor` |
| Basal | `_overnight_basal_signal` + **new**: temp-basal/suspend awareness from `insulin_events` | "was basal stable" becomes answerable with real data |
| Meal | `_post_meal_signal` + `_classify_meal` + **new**: real carb entries replace inferred meal windows | timing/counting/dose differential gets grounded |
| Correction | rebound logic from `_safety` + **new**: per-correction outcome tracking (bolus → 3h trajectory) | impossible in current repo; Nightscout makes it real |
| Research | `clinical_kb.py` corpus + retrieval | citations attached to findings as `research_context` |
| Discovery | `researcher/agent.py` whole pipeline | keeps plan→test→reflect→finalize, tool budget (8), 24h cache; tools come from `stats/core` |
| Skeptic | NEW | see §8 |
| **Prediction Reconciliation** | NEW (seed: `pattern_baselines` Welford logic in `monitoring/memory_writer.py` + `event_proximity`) | see §7.1 — expected-vs-actual trajectory, contributor attribution, recurrence memory |
| Brief | `clinical/brief.py` | evidence bundle → rank/explain → guard → fallback → provenance meta. Add a "data sources used" line (glucose-only vs glucose+insulin) |

---

### 7.1 Prediction Reconciliation Agent (flagship "why did this happen")

Compares **expected** glucose trajectory vs **actual**, attributes the error, and counts recurrence.

**The unlock:** for looping users, Nightscout `devicestatus` already logs the dosing algorithm's own
forecast each cycle — oref0/AAPS upload `openaps.suggested` prediction curves (**IOBpredBGs** insulin-only,
**COBpredBGs** carbs-as-announced, **UAMpredBGs** unannounced-meal), Loop uploads `loop.predicted`. Expected
trajectory is *logged ground truth of the algorithm's belief*, not something we model.

**Two tiers (declared in the finding, never blurred):**
- **Tier A (OpenAPS/AAPS/Loop/Trio):** reconcile realized CGM against the logged predBG curves.
  Attribution from curve geometry: UAM fits better than COB → carb estimate off; all curves overshoot →
  sensitivity shift (sensitivity, site aging); IOB fine + COB wrong → absorption timing.
- **Tier B (Control-IQ via tconnectsync, MDI):** no published forecasts — reconcile against our own
  deterministic expectation computed by `analytics/oref.py`, a pure-Python port of oref0's documented
  math (MIT): exponential insulin-activity curves (tau/S formulation, peak 75/55 min presets), net IOB
  incl. temp-basal micro-bolus decomposition, BGI = activity × ISF, deviations, deviation-based COB
  decay (`min_5m_carbimpact` 8 mg/dL/5m, CSF = ISF/CR), and the four predBG curves
  (IOB/ZT/COB/UAM) + eventualBG. Same curve vocabulary as Tier A, so attribution logic is shared.
  Weaker evidence class; the finding says so explicitly. Never used to recommend dosing.

**Pipeline position:** Nightscout connector parses `devicestatus` → predictions land in the timeline →
reconciliation runs as a deterministic agent (no LLM) → per-episode error records
(`+42 mg/dL at 120 min, contributor=carb_underestimate, tier=A`) → recurrence counted against the
findings store ("similar pattern, N occurrences") → rigor layer gates significance → discovery/skeptic/brief
consume it like any finding. NOT routed through MCP internally — MCP stays the external exposure layer.

**Schema addition (layer 2):** `prediction_events(ts, source, horizon_min, curve_kind, pred_mg_dl[], raw_event_id)`.

**Eval hook (E6+):** reconciliation findings against documented regime changes — the agent should localize
"sensitivity shifted" to the window where the user actually changed sites/profile.

## 8. Statistical rigor layer (NEW — the credibility core)

Current state, honestly: `groupby_compare` has Cohen's d and min-N guards; **nothing has p-values, multiple-testing correction, or replication.** A discovery engine mining years of data across dozens of dimensions without this is a p-hacking machine and will be dismissed by exactly the audience we want.

`stats/rigor.py` provides:

1. **Permutation p-values** — `perm_test(metric_fn, groups, n=2000)`. No distributional assumptions, works for every comparison the tools make.
2. **Benjamini-Hochberg FDR** — applied per Deep-Analysis run across *all* hypotheses tested in that run. Findings carry `q_fdr`; default surfacing threshold `q ≤ 0.10`.
3. **Split-half replication** — every candidate pattern is re-tested on a temporally disjoint half of the window. `replicated: bool` on the finding; non-replicated findings are demoted to `hypotheses` (status `open`), not surfaced as discoveries.
4. **Power gate** — given the observed effect size and N, refuse to claim anything when power < 0.5; emit a "collecting more data" hypothesis instead.

**The Skeptic agent** consumes findings and:
- re-runs the rigor checks independently (different random seed, different split),
- searches `findings`/`hypotheses` memory for contradicting priors,
- checks for confound candidates (e.g., "weekend effect" vs "sleep effect" when both correlate),
- writes `skeptic_notes` and may downgrade `confidence` or flip status to `rejected`.

Pipeline ordering in Deep Analysis: deterministic agents → discovery → **rigor batch (FDR across the run)** → skeptic → memory write → brief/report. A finding's prose may not exceed what its `stats` block supports (the guard enforces numbers; the skeptic enforces inference).

---

## 9. Cold-start framework

`coldstart.py` computes one `ColdStartReport` per run from the timeline:

| Capability | Minimum | Degraded mode below minimum |
|---|---|---|
| Metrics snapshot (TIR/GMI/CV) | 7 days, ≥70% coverage | shown with "low confidence — N days" banner |
| AGP | 14 days | rendered with consensus-standard warning |
| Pattern agents (weekday, ToD) | 21 days | skipped, listed as "pending: needs ≥21d" |
| Sleep/workout correlations | 8 nights / 6 sessions per bucket | skipped with progress counter ("3 of 8 nights collected") |
| What Changed | 2× window | skipped |
| Discovery (longitudinal) | 60 days | runs in "early signals" mode: hypotheses only, no discoveries |
| Trajectory / long-term | 90 days | skipped |
| Insulin-grounded agents (basal/meal/correction) | ≥50% of days with treatments data | falls back to glucose-shape inference **with explicit "inferred — no insulin data" labeling** (current repo behavior becomes the labeled fallback) |

Rules:
- Gates are computed once and injected into every agent's context; agents *declare* requirements (`DataRequirement`) and the registry refuses to run them under-data rather than trusting each agent to check.
- The TUI/report always shows a **coverage header**: days, %, sources, and a "what unlocks next" line ("17 more days unlocks weekday patterns"). Cold start becomes a progress mechanic instead of a failure.
- Hard floor: <3 days of data → `dexta analyze` prints the coverage report and exits; nothing pretends to analyze.

---

## 10. Dependency graph & build order

```mermaid
flowchart TD
    subgraph F["Phase 0 — Foundation"]
        F1[F1 repo scaffold + CI]
        F2[F2 StoragePort + SQLite + schema]
        F3[F3 LLM factory BYOM]
        F4[F4 config + coldstart framework]
    end
    subgraph D["Phase 1 — Data plane"]
        D1[D1 Nightscout provider + connector]
        D2[D2 normalize → timeline]
        D3[D3 rollups + metrics port]
        D4[D4 CSV/file upload]
        D5[D5 Dexcom connector]
        D6[D6 Whoop provider + connector]
        D7[D7 MCP servers: nightscout/libre/whoop]
    end
    subgraph I["Phase 2 — Deterministic intelligence"]
        I1[I1 stats core port]
        I2[I2 rigor layer NEW]
        I3[I3 observation + pattern agents]
        I4[I4 basal/meal/correction agents]
        I5[I5 guard unification]
    end
    subgraph M["Phase 3 — Memory"]
        M1[M1 findings store unified]
        M2[M2 embeddings + retrieval]
        M3[M3 wiki generator]
    end
    subgraph A["Phase 4 — Agentic layer"]
        A1[A1 discovery agent port]
        A2[A2 skeptic agent NEW]
        A3[A3 research agent KB]
        A4[A4 clinical brief port]
    end
    subgraph W["Phase 5 — Workflows + surfaces"]
        W1[W1 deep analysis]
        W2[W2 explain spike]
        W3[W3 what changed]
        W4[W4 TUI]
        W5[W5 report export]
    end

    F1 --> F2 & F3 & F4
    F2 --> D1 & D2 & M1
    F4 --> D1
    D1 --> D2 --> D3
    D2 --> D4 & D5 & D6
    D1 --> D7
    D3 --> I3 & I4
    I1 --> I2 --> I3 & I4 & A1 & A2
    F3 --> A1 & A2 & A3 & A4
    M1 --> M2 --> M3
    M1 --> A1 & A2 & A4
    I5 --> A1 & A4
    I3 & I4 --> W1
    A1 & A2 --> W1
    A4 --> W5
    D3 --> W3
    I4 --> W2
    W1 & W2 & W3 --> W4
```

**Critical path:** F2 → D1 → D2 → D3 → I3/I4 → W1 (a working Deep Analysis on real Nightscout data). Everything else hangs off it.

**Sequencing notes:**
- F2/F3/F4 are mutually independent after F1 → build in parallel.
- D4/D5/D6/D7 are off the critical path — pure parallel work once D2's normalize contract (and for D7, the provider clients) exist. The MCP servers (D7) are stateless wrappers over `providers/` and can ship to PyPI independently, like the Dexcom one already did.
- I1+I2 (stats) only depend on F1 — can start day one, in parallel with the data plane.
- M1 only needs the schema (F2) — also early-parallel.
- The TUI (W4) needs only *stub* workflow outputs to start; build it against fixture JSON in parallel with Phase 4.

---

## 11. Sub-agent execution map

Explicit spawn points. Each task lists its contract (input → output) so a sub-agent can run it without ambient context. Rule: **a sub-agent owns a module + its tests; interfaces (protocols in §5/6/7) are frozen by the lead before spawning.**

### Wave 0 (after F1; all parallel)
| Sub-agent | Task | Contract |
|---|---|---|
| `agent-store` | F2: StoragePort + SQLite backend + schema + migrations | `store/port.py` protocol (frozen) → backend passing `tests/test_store_contract.py` |
| `agent-llm` | F3: model factory + strict-JSON wrapper port | `ROLE_DEFAULTS` table → factory passing provider-matrix mock tests |
| `agent-config` | F4: config loader + coldstart gate framework | `dexta.toml` spec (§4) + gate table (§9) → modules + tests |
| `agent-stats` | I1: port `_stats.py` + 6 researcher tools into `stats/core` | donor files → pure functions, no DB/LLM imports, golden-value tests |
| `agent-rigor` | I2: permutation/FDR/split-half/power | §8 spec → `stats/rigor.py` + tests incl. null-data calibration (≤10% false-positive rate on synthetic null at q=0.10) |
| `agent-golden` | **Testing agent**: synthetic data generator | spec of planted patterns (dawn effect, weekend effect, post-dinner spikes, basal drift, confounded pairs, NO-pattern null sets) **at parameterized effect sizes** → `tests/golden/` + `eval/datasets/` + ground-truth manifest (feeds E4/E5) |
| `agent-eval` | **Testing agent**: eval harness (§14) | E1–E5 metric definitions → `eval/metrics/` + cross-model runner + `report.py` emitting the headline table; E8 red-team corpus seeded from existing `pha_output` test cases |

### Wave 1 (after F2 + F4)
| Sub-agent | Task | Contract |
|---|---|---|
| `agent-nightscout` | D1+D2: provider + connector + normalization | Connector protocol + recorded fixtures → timeline rows; idempotent re-ingest test |
| `agent-rollups` | D3: port analytics kernels onto timeline | donor `analytics.py`/`agp.py` → `rollups.py`+`analytics/` with **parity tests: identical numbers to donor on identical input** |
| `agent-memory` | M1+M2: findings store + embeddings | `Finding` schema (§3) → store + retrieval with relevance tests |
| `agent-csv` / `agent-dexcom` / `agent-whoop` | D4/D5/D6 | same connector contract + fixtures (Whoop = port `whoop_client.py`, token auth) |
| `agent-mcp-servers` | D7: nightscout/libre/whoop FastMCP servers | `_contract.py` (frozen, = published Dexcom server's 10 tools + insulin extension) → 3 PyPI-publishable packages with fixture-backed tool tests |

### Wave 2 (after D3 + I1/I2)
| Sub-agent | Task | Contract |
|---|---|---|
| `agent-pattern` | I3: port correlators+detectors as Pattern/Observation agents | donor list (§7 table) → agents passing golden-data tests (**must find every planted pattern, must find nothing in null sets**) |
| `agent-insulin` | I4: basal/meal/correction agents with treatments data | §7 notes → agents + golden tests with planted insulin scenarios (missed bolus, over-correction, basal drift) |
| `agent-guard` | I5: unify 3 guard implementations | donor sites → one `guard/faithfulness.py`; **adversarial test suite: 50+ crafted LLM outputs (fabricated numbers, wrong-context numbers, derived ratios) with expected verdicts** |
| `agent-tui-shell` | W4 scaffold against fixture JSON | view specs → Textual app bound to stub data |

### Wave 3 (after Wave 2 + M1)
| Sub-agent | Task | Contract |
|---|---|---|
| `agent-discovery` | A1: port researcher pipeline onto stats/rigor + memory | donor + §8 ordering → discovery agent; golden test: discovers planted longitudinal pattern, refuses null set |
| `agent-skeptic` | A2 | §8 → skeptic; test: rejects a deliberately confounded finding, passes a clean one |
| `agent-reconcile` | §7.1 Prediction Reconciliation | devicestatus predBG parsing + curve-vs-realized error + contributor attribution + recurrence; golden tests with planted prediction-miss scenarios per tier |
| `agent-brief` | A4: port clinical brief | donor → brief agent with guard + fallback + provenance meta; parity test vs donor output |
| `agent-workflows` | W1–W3 composition | workflow specs → CLI commands producing report JSON |

### Continuous testing agents (run every wave)
- `agent-test-runner` — executes full suite per merged task, bisects regressions.
- `agent-parity-auditor` — re-runs donor-vs-port numeric parity on every analytics change (the OSS must produce *identical* metrics to the battle-tested originals).
- `agent-adversary` — grows the guard/safety adversarial corpus: tries to get dosing advice out of every LLM surface, tries to sneak fabricated numbers past the guard; any success becomes a permanent regression test.
- `agent-docs` — keeps README quickstart + module docs in sync with merged interfaces.

**Coordination rules for sub-agents:** (1) interfaces frozen before spawn — protocol changes go through the lead; (2) one module per agent, no cross-module edits; (3) every agent delivers code + tests + a 10-line summary of deviations from spec; (4) parity-with-donor is the acceptance bar for all ports, golden-data ground truth for all new intelligence.

---

## 12. Testing strategy (summary — CI pass/fail; the reportable benchmark is §14)

1. **Parity tests** — ported analytics produce numerically identical output to donor modules on shared input. Non-negotiable for `metrics/agp/windows/trajectory`.
2. **Golden datasets** — synthetic CGM+insulin with planted, documented effects + null sets. Pattern/discovery agents must achieve: 100% planted-pattern recall, 0 findings on null data at default thresholds.
3. **Guard adversarial corpus** — crafted LLM outputs with fabricated/miscontextualized numbers; corpus only grows.
4. **Connector fixtures** — recorded sanitized API responses; idempotency (double-ingest → no dupes) and watermark tests.
5. **Rigor calibration** — false-positive rate of the full discovery pipeline on null datasets ≤ FDR target.
6. **Cold-start matrix** — every capability exercised at 0/3/10/30/90/365 days of data; assert correct gating + messaging, never a crash, never a silent fabrication.
7. **E2E smoke** — `dexta init → sync (fixture server) → analyze → brief → wiki` in CI.

---

## 13. Out of scope for MVP (explicit)

Web UI (Phase 8), plugin packaging/`pip install dexta-agent-*` (contract is ready; packaging later), interactive chat graph (port after TUI), Whoop/Apple Health/Garmin/Oura connectors, LightGBM forecasting stack, multi-user/clinical sharing/auth, email alerts, hosted anything.

---

## 14. Evaluation framework (the reportable benchmark)

Distinct from `tests/` (CI pass/fail): `eval/` produces **the results table** — the artifact that makes the harness credible to the health-AI audience and reproducible by anyone on their own model. Design rule: **every eval's ground truth is non-LLM** (set membership, arithmetic invariants, planted effects, documented real events, or clinician rubric). LLM-as-judge is banned from reported numbers.

### Tier 1 — hard ground truth, fully automatic (runs in CI, every model in the matrix)

| Eval | Protocol | Metrics | Bar |
|---|---|---|---|
| **E1 Numeric faithfulness** | N windows → briefs/findings with guard **off** vs **on**, per model | fabricated-figure rate; miscontextualized-figure rate; guard catch rate; **false-rejection rate** (legit numbers wrongly blocked) | guarded fabrication = 0; false-rejection < 5% |
| **E2 Internal consistency** | machine-checkable invariants per report | violations/report: TIR+TAR+TBR=100±0.2; GMI↔mean formula; combined projection ≤ best single window; exposure h/wk ↔ TAR%; what-changed deltas recomputable | 0 violations |
| **E3 Analytics parity** | port vs donor on identical input + external reference (consensus formulas, Nightscout reports on same data) | max numeric divergence | exact (donor), ±0.1 (external) |

E2 codifies the two real bugs already caught in production (single-vs-combined projection conflation; coverage-blind exposure hours). The class is the eval.

### Tier 2 — constructed ground truth (the discovery evals)

| Eval | Protocol | Metrics | Bar |
|---|---|---|---|
| **E4 Planted-effect recovery** | generator plants effects at controlled sizes (dawn rise, weekend TIR drop, under-bolused dinners, basal drift, rebound over-corrections) + pure-null sets; include **confounded pairs** the system should *flag, not claim* | recall vs (effect size × data span) → **power curve**; null false-discovery rate at q; confound-flag rate | null FDR ≤ q (0.10); 100% recall at large effect/90d |
| **E5 Perturbation robustness** | same data ± 15% dropout, compression lows, dup timestamps, tz shifts, 3-day gap | finding stability (Jaccard clean-vs-corrupted); new spurious findings | Jaccard ≥ 0.8; 0 corruption-induced findings |
| **E6 Regime-change recovery** | real data with **documented true events as labels** (pump setting changes, site changes, illness, travel — timestamps from Nightscout `treatments` / user log) | localization accuracy (right window) + attribution accuracy (right axis) | report-only (no hard bar v1) |

E4's power curve is itself a publishable result ("a 7-point weekday effect needs ~45 days to detect reliably"). E6 is the closest available proxy for "found something true" with ground truth we didn't invent.

### Tier 3 — human ground truth (per major version, not CI)

| Eval | Protocol | Metrics |
|---|---|---|
| **E7 Clinician rubric** | 1–3 endocrinologists rate ~20 briefs: clinically valid / would change what I examine / unsafe / misleading (Likert) | mean scores + inter-rater agreement |
| **E8 Safety red-team** | adversarial corpus extracting dosing math through every LLM surface, incl. indirect framings; corpus only grows; every hit becomes a permanent regression case | dose-advice leakage rate — **zero tolerance, also run in CI** |

### Explicitly NOT evals
LLM-as-judge "insight quality" (circular; dev-time smoke check only). Outcome claims ("users improved TIR") — that is a clinical study, not an eval; never imply it. Testimonials.

### The headline artifact
One table from `eval/report.py` — rows = models (Claude / GPT / Gemini / local Llama), columns = E1 fabrication (raw vs harness), E2 violations, E4 recall + null-FDR, E5 stability, E8 leakage. Cross-model with harness guarantees holding regardless of model **is** the BYOM pitch, reproducible by anyone.

---

## 15. Web GUI — settings & credentials page

The shipped web GUI (`server/`, `dexta serve`, localhost:8787) currently renders
credentials as a **status-only flat env list** (every var an identical gray dot,
"set these in your shell"). That fails two ways: configuration state is
illegible (per-source grouping is absent — users miss that pydexcom creds are
even listed), and the quickstart promise breaks at the first hurdle (a new user
should never need to learn shell env to connect a source).

### Design

**Grouped by source, not by variable.** One card per source (Nightscout, Dexcom
Share, Libre, Whoop, Oura, + LLM provider, + literature), each showing:
configured/unconfigured state, which fields it needs, data freshness (latest
event ts from `sync_state`), and a **Test connection** button that runs
`connector.check()` inline and renders the `HealthReport`.

**Editable, with the right storage.** Form fields write to
`~/.dexta/dexta.toml` (created `0600`), the same file `dexta init` writes — the
GUI is an alternative front-end to the config, not a second config system.
Environment variables remain **read-only status** (highest-precedence override,
shown as "set via env — managed in your shell"). Precedence stays:
env > dexta.toml > defaults.

**Secret hygiene (binding):**
- Masked inputs; stored secrets are **never echoed back** — a configured field
  renders as `••••` + last 4 characters and a "replace" affordance.
- No secret ever appears in a GET URL, server log line, or HTML source.
- The server binds localhost by default and has no auth; the settings page
  carries a visible note to that effect, and `--host` other than loopback
  disables the credential-editing forms (status-only mode) unless
  `DEXTA_ALLOW_REMOTE_SETTINGS=1`.
- Writes are atomic (temp file + rename) and re-validated by the config loader
  before commit; a failed validation never clobbers the existing file.

**Connector-tier awareness.** Sources from the unofficial tier (§6.2.1) render
their "unofficial API" banner on the card itself, at the moment of decision.

---

## 16. Success criteria (engineering restatement)

A new user with a Nightscout site and one API key can: install with pipx, `dexta init` in under 2 minutes, sync years of history, run Deep Analysis and get rigor-gated discoveries with evidence traces, ask "explain this spike," generate a physician brief, watch the wiki accumulate knowledge across runs, swap `[llm]` to any provider — all self-hosted, with every number traceable, every claim statistically gated, and every cold-start state explicitly labeled.
