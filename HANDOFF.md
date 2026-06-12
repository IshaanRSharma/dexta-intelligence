# dexta-intelligence — build handoff

_Last updated: 2026-06-11 (session handoff)._

## TL;DR

Open-source agentic health-intelligence harness for Type 1 diabetes. **Analytics compute
evidence; agents (including LLM agents) produce intelligence on top of it.** Batch path is
built and tested; LLM agents (discovery, brief, chat) are queued.

**695 passed, 41 skipped (env-gated: Postgres parity suite).** Ruff + mypy-strict clean, 75 files.

**Wave 4 done 2026-06-12 — dexta as a time-series health agent** (`docs/WAVE4.md`,
supervised parallel build): the model now traverses time itself. Time-traversal tools in
`discovery_tools.py` (`list_segments` orient · `set_window` re-scope via bisect, no re-query
· `zoom_event` spike drill · `daily_series` trend) — active sub-window is the agent's
time-memory. `agents/router.py` (intent → focused tool family, guard always runs, keyword
fallback). `agents/seeker.py` (`--seek`: reflect + re-scope across ≤3 rounds, evidence
accumulated so cross-round citations stay guard-faithful). `agents/trace.py` (pure
formatter → visible "thinking" path, can't fabricate). Wired: `dexta ask` routes by default
+ prints trace; `dexta ask --seek` goal-seeks. `ChatAnswer.trace` computed in `chat._finish`
so all three agents inherit it. Live-verified: router→list_segments→set_window(April)→
tod_compare→answer with visible trace, real re-scope to 30-day window.

**QA + product review remediated 2026-06-12** (`docs/review/QA_REPORT.md`,
`docs/review/PRODUCT_REVIEW.md`): wiki path-traversal sealed (`is_relative_to`, verified live
0 leaks), GUI settings write to the launched config + reject invalid targets, markdown link-scheme
sanitized (no `javascript:`), `serve --host 0.0.0.0` warns. Finding dedup via
`persist_findings` (supersede-on-rerun, no recurrence inflation, wired into `dexta analyze`).
Goal `--target` flag wired → `_is_achieved` now reachable. README/docs de-oversold: honest test
count, install-from-source (not on PyPI), `--lens why` ≠ spike-explainer, goals = cron ticks not
daemon; new "Status & honest limitations" section. Remaining product gaps: PyPI publish,
`dexta explain <event>`, goal scheduler, cross-model eval matrix (all in Wave 4 / planned list).

**Wave 3 done 2026-06-12:** evidence grounding (`evidence/` — PubMed E-utilities default,
OpenEvidence opt-in via OPENEVIDENCE_API_KEY; `search_evidence` reasoning tool, PMIDs/years
guard-verifiable); **`dexta serve` GUI** (`server/` — FastAPI+HTMX+Jinja, `[gui]` extra:
dashboard/wiki/goals-arcs/chat/settings, env vars as set/unset dots only, localhost default);
per-role model resolution (`model_for_role` — ask→explain, brief/wiki→brief, goals-add→plan);
README launch overhaul w/ real eval table (E1 100%/0%, E4-null 5%, E5 Jaccard 1.0),
docs/CHANGELOG.md chronicle, CONTRIBUTING, CI workflow, connector issue template,
docker-compose, docs/DEMO.md; TESTING_DEBT burned down to 3 unbuilt-feature items.
`cli.py` split into the `cli/` package 2026-06-12 (`_common` / `data` / `analysis` /
`intelligence` / `main`, ~180-250 lines each; `dexta_intelligence.cli` re-exports keep
every import site and the `dexta` script entry working — pure code motion).

**Wave 2 done 2026-06-12 (parallel agent wave):** clinical brief (`agents/brief.py`,
`dexta brief` — guard + dosing-advice regex refusal, deterministic fallback); lenses
(`workflows/lenses.py`, `dexta analyze --lens watch|why|insulin|analyze`, `[lens.*]`
config, skeptic non-routable, cli.py thinned); embeddings recall
(`memory/embeddings.py`, dependency-free lexical vectors + trigrams; `recall` now ranked
+ reads back persisted synthesis — `synthesis.save/load_latest`, supersede-on-save);
`Investigator` base extracted (`agents/investigator.py` — discovery/insulin now thin
config, 778→249 lines; the community-plugin surface); skeptic confound flags auto-bank
"Disentangle X vs Y" open hypotheses in deep analysis (`banked_hypotheses` on report).
TUI is CUT (decision 2026-06-12): GUI via `dexta serve` instead.
**Wave 4 (deferred):** harness-MCP server (`ask`/`recall`/`goals`/`brief` as MCP
 tools — the
adoption wedge), pgvector + `EmbeddingPort` (BYOM embeddings; lexical seam already swappable),
`dexta migrate --to-postgres`, cross-model eval matrix.
Deferred test cases live in `docs/TESTING_DEBT.md` — check items off as they land.
Config stays TOML (stdlib `tomllib`, zero deps) — YAML evaluated and rejected 2026-06-12.

```bash
cd ~/Desktop/dexta-intelligence
.venv/bin/ruff check src tests eval && .venv/bin/mypy && .venv/bin/pytest -q
```

---

## Questions from this session (and answers)

### 1. LangGraph for multi-agent orchestration, or let users route agents?

**Answer: both modes, different jobs — not LangGraph for batch analyze today.**

| Mode | Mechanism | When |
|---|---|---|
| **Batch** (`dexta analyze`, eval) | Blackboard + `AgentRegistry` + `run_deep_analysis` | MVP — **what we ship now** |
| **Interactive** (chat, explain spike) | LangGraph plan→execute→reflect | Later — port donor `agents/graph.py` |
| **User routing** | Custom `AgentRegistry` + future `[analysis].agents` in `dexta.toml` | Power users / plugins |

**Why not LangGraph for batch:** fan-out analytics needs parallel isolated agents, replayable
runs, and audit trails (`Finding` + `stats` + evidence). Agents never call each other on the
blackboard — they read the store and write findings. One fixed post-pass runs **skeptic**
(producers → skeptic → persist).

**Why LangGraph later:** dynamic routing when the *next* step depends on the user's question
(chat/TUI/MCP ask flows).

**Recommendation recorded:** default pipeline stays opinionated (skeptic non-optional for
quantitative claims); add config-driven agent lists for power users without replacing the
blackboard.

### 2. "LLM narrates does not make sense — LLM is the intelligence layer"

**Agreed. Framing corrected.**

Old (wrong) pitch: "deterministic core, LLM narration" — implies the model is decoration.

**Correct model:**

```
Connectors → timeline → analytics/stats (evidence substrate)
                              ↓
              agents: deterministic detectors + LLM intelligence
                              ↓
              skeptic + faithfulness guard (honesty bounds, not replacement)
                              ↓
              memory → brief / chat / MCP
```

- **Analytics/stats** = numbers, rigor, oref math — the evidence pool.
- **LLM** = discovery, synthesis, ranking, explanation, interactive reasoning — the
  intelligence layer.
- **Faithfulness guard** = the model cannot cite numbers absent from evidence. It constrains
  dishonesty; it does not demote the LLM to a parrot.

README and `guard/faithfulness.py` docstrings updated to reflect this (2026-06-11).

### 3. "I'm confused by our wiki/memory documentation"

**Resolved — see `docs/INTELLIGENCE.md` §1.** The spec made wiki/embeddings/findings look
like three memory systems. It's **one store, two projections**:

- **Findings table = the only memory** (write path: agents → skeptic → `insert_finding`).
- **Embeddings = machine index** over findings (agent recall). Rebuildable, not a store.
- **Wiki = human index** over findings (generated markdown, regenerated per run,
  never read by agents, gitignored). If deleted, `dexta wiki` rebuilds it byte-identical.

Wiki file layout, topic-page anatomy, and the **graveyard page** (retracted beliefs +
skeptic notes — the trust artifact) are spec'd in the doc. Generation is deterministic
templating — **no LLM, no embeddings dependency** → moved up the build queue.

### 4. TUI agent routing (user proposal — adopted)

**Design: "lenses" — named agent routes.** See `docs/INTELLIGENCE.md` §3. Built-ins:
`insights` (default — reads memory, zero LLM calls, instant), `analyze`, `why`, `watch`,
`clinic`; custom via `[lens.*]` in `dexta.toml` (generalizes the queued `[analysis].agents`
item). Invariant: routing selects *producers* — skeptic post-pass and guard are not
routable-out. A **router agent** (fast-tier `plan` role) maps natural-language intent →
`{lens, window, scope}`; keyword fallback without an API key. Mechanically a lens is just
a filtered `AgentRegistry` into the existing `run_deep_analysis`.

### 5. GlycemicGPT competitive read

Surveyed 2026-06-11 (111 stars, v0.9.0, active): their LLM is single-inference briefs +
chat — a summarizer with a disclaimer instead of guardrails; no agents, no memory
semantics, no stats. Full differentiation table, steal-list (BLE pump streaming,
caregiver alerts, community ops), and the "five proofs" demo list in
`docs/INTELLIGENCE.md` §4. One-liner: *they put an LLM in front of your data; we put a
research team behind it.*

### 6. Agent timing on 90-day synthetic data

Profiling run on `scenario_all(seed=42)`:

| Agent | Result |
|---|---|
| `observation` | 3 findings in **0.1s** |
| `pattern` | 3 findings in **2.4s** |
| `reconciliation` | **Hung** — no completion in 17+ min; reproduces in isolation |

**Action item:** profile/fix `agents/reconciliation.py` before shipping long-window analyze
to users. Likely O(n²) or unbounded loop over glucose × prediction cycles on full 90d data.

---

## What was built (2026-06-11 session)

Prior handoff listed Libre as crashed mid-build; repo had already moved far past that. This
session added the **agentic glue** and **eval substrate**:

| Deliverable | Path(s) | Notes |
|---|---|---|
| **Skeptic agent** | `agents/skeptic.py`, `tests/test_skeptic_agent.py` | Re-runs `assess()` seed 137, confound flags, prior contradictions, rejects bad stats |
| **Pattern evidence for skeptic** | `agents/pattern.py` | `skeptic_group_a/b` on findings for independent re-check |
| **Memory helpers** | `memory/findings.py`, `memory/__init__.py` | Recurrence, similarity, contradictions |
| **Deep analysis workflow** | `workflows/deep_analysis.py` | Producers → skeptic → persist; wired in CLI |
| **CLI analyze** | `cli.py` | Uses `run_deep_analysis`; prints skeptic notes + rejected findings |
| **E4 eval scaffold** | `eval/metrics/e4_null_fdr.py`, `eval/runner.py`, `eval/report.py` | Null-set FDR calibration; `python -m eval.runner e4-null` |
| **Tests** | `test_deep_analysis.py`, `test_memory_findings.py`, `test_eval_e4.py` | + skeptic tests |
| **Docs** | `README.md`, this file | Quick start no longer "coming soon" |
| **Intelligence-layer design doc** | `docs/INTELLIGENCE.md` | Memory/wiki untangled (one truth, two indexes), lenses/TUI routing, GlycemicGPT competitive read |
| **Wiki generator** | `memory/wiki.py`, `tests/test_wiki.py`, `cli.py` (`dexta wiki`), `[wiki]` config | Deterministic projection of findings store: index + topic pages + hypotheses board + graveyard + per-run changelog; staleness decay (age vs confidence×recurrence); git-native belief history. Pattern adapted from [nex-crm/wuphf](https://github.com/nex-crm/wuphf) (substrate guarantee, no-deletion-only-status, staleness as read-time visibility) — user pointer, explored 2026-06-11 |

### Already built before this session (do not re-build)

Libre connector, SQLite store, sync workflow, observation/pattern/reconciliation agents,
Oura + CSV connectors, MCP server (10-tool contract), CLI init/doctor/sync/upload, 400+
connector/stats/oref tests.

---

## Architecture (current)

```
BATCH (dexta analyze):
  observation · pattern · reconciliation · discovery(LLM)  →  skeptic.review()  →  findings → wiki
         (blackboard — agents do not call each other)

INTERACTIVE (dexta ask):
  question → reason.run_reasoning_loop(model, tools)  →  guard.audit  →  answer
         tools = tod/groupby/event_proximity + coverage + recall(memory)
```

**Two LLM call sites now live:** discovery (plan→probe→judge→claim/wonder) and chat
(native tool-calling loop, the model decides when to compute). Both gate claims through
`stats.rigor` and/or `guard.faithfulness`; reasoning itself is unguarded over read-only tools.

**The reasoning primitive:** `agents/reason.py` — `run_reasoning_loop` is the reusable
ReAct/function-calling engine. Discovery, chat, and (next) goal-workflows all sit on it.
Dependency-light: dict messages + duck-typed model, no `langchain_core` on the hot path.

**Queued LLM agents:** clinical brief; agentic-wiki synthesis layer (LLM connective
narrative, guard-checked); goal-based background workflows (`docs/INTELLIGENCE.md` §7).

**Orchestration seam:** `AgentRegistry` + `run_deep_analysis` (batch); `run_reasoning_loop`
(interactive). LangGraph still not in the codebase — the native tool-calling loop covers
chat without it; revisit only if multi-turn graph state is needed.

---

## Verify state

```bash
.venv/bin/ruff check src tests eval   # All checks passed
.venv/bin/mypy                        # 49 source files
.venv/bin/pytest -q                   # 517 passed
dexta ask "why were my mornings rough this week?"   # needs [llm] + provider key
python -m eval.runner e4-null --datasets 10 --days 90
```

---

## Build queue (dependency order)

1. ~~**Fix reconciliation perf**~~ — **DONE 2026-06-12** (parallel agent wave): Tier-B COB
   recompute was O(N²·M·D) + unbounded permutation pools. Now deviation-series reuse via
   bisect, dose windowing, bounded rigor groups. 17+ min → ~12 s on 90d, numeric parity
   (all prior tests unchanged), perf regression test added (<15 s).
2. ~~**Wiki generator**~~ — **DONE 2026-06-11** (`memory/wiki.py`, `dexta wiki`); remaining
   nice-to-have: auto-regenerate at end of `dexta analyze` (kept separate so tests with
   default config never write to `~/.dexta/wiki`)
3. ~~**Discovery agent**~~ — **DONE 2026-06-11**: first real LLM-reasoning agent.
   `agents/discovery.py` (plan→probe→judge→claim/wonder loop) + `agents/discovery_tools.py`
   (deterministic tool belt: groupby_compare/tod_compare/event_proximity, all read-only).
   Reasoning unguarded; claims gated by `stats.rigor.assess` + `guard.faithfulness.audit`;
   underpowered questions banked as OPEN hypotheses (the curiosity backlog). Degrades to a
   deterministic sweep with no API key. `tests/test_discovery_agent.py` (fake-model loop +
   guard-rejects-fabrication + wonder-banks-hypothesis). Wired into `dexta analyze` via
   `_registry_with_discovery` (model from BYOM factory, `discovery` role).
   Remaining: auto-seed wonders from skeptic confound notes; richer tool belt (correlate, lag).
4. ~~**Chat agent + reasoning loop**~~ — **DONE 2026-06-12**: `agents/reason.py`
   (`run_reasoning_loop` — native tool-calling ReAct engine) + `agents/chat.py` + `dexta ask`.
   The model decides when to call which read-only tool (stats arsenal + `recall` over memory);
   answer guard-audited, untraceable numbers flagged. `tests/test_chat_agent.py` (fake
   tool-calling model: tool→answer loop, max-steps cap, guard-flags-fabrication, recall reads
   memory). This is the "Claude Code for health" surface. GlycemicGPT confirmed to have NO
   such loop (sidecar = Express proxy, zero AI deps).
5. ~~**Goal-based background workflows**~~ — **DONE 2026-06-12**: `workflows/goals.py` +
   `dexta goals add/list/tick`. compose (LLM or keyword fallback) → deterministic success
   metric (`GoalMetric`: tir/nocturnal_tbr/tbr/mean_glucose/cv) → cadence-gated tick that
   investigates via the reasoning loop (model picks tools) or replays the stored plan,
   banks moderate/large observations as open hypotheses, records checkpoints (the arc),
   auto-marks achieved. Goals + arcs render in the wiki (`goals.md`). Store: `goals` +
   `goal_checkpoints` tables (SCHEMA_VERSION 2, idempotent upgrade). Tests in
   `tests/test_goals.py` + CLI tests. Deferred cases tracked in `docs/TESTING_DEBT.md`.
6. ~~**Agentic-wiki synthesis layer**~~ — **DONE 2026-06-12**: `memory/synthesis.py`
   (`synthesize(findings, model) → SynthesisResult`), every paragraph/connection
   guard-audited against finding evidence, unfaithful lines dropped. Wiki renders
   `## Synthesis` per topic + `## Connections` on index; `synthesis=None` byte-identical.
   Wired into `cmd_wiki` (uses configured model when present).
7. ~~**Basal / Meal / Correction**~~ — **DONE 2026-06-12** as `agents/insulin.py`
   (`InsulinAgent`, needs_insulin=True, rigor seed 37) over three new instruments in the
   toolkit: `basal_overnight` (drift, temp-basal/suspend nights excluded),
   `meal_response` (excursion by carb split), `correction_outcome` (post-bolus delta +
   rebound_low_rate). Registered in `dexta analyze`; tools available to chat/goals via
   `tool_specs`.
8. ~~**Postgres backend**~~ — **DONE 2026-06-12**: `store/postgres.py` (psycopg 3, lazy
   import, TIMESTAMPTZ/JSONB, method-for-method sqlite parity, zero semantic deltas).
   Parity suite gated on `TEST_DATABASE_URL` (skips cleanly). `dexta migrate
   --to-postgres` still TODO.
9. ~~**Eval expansion (E1, E5)**~~ — **DONE 2026-06-12**: `eval/metrics/e1_faithfulness.py`
   (guard catch rate 1.0 fabricated + miscontextualized, false-rejection 0.0) and
   `e5_perturbation.py` (Jaccard 1.0 across dropout/dupes/gap/tz-shift, 0 induced kinds);
   `python -m eval.runner e1|e5`. Cross-model matrix still TODO.
10. **Clinical Brief** — LLM synthesis over evidence bundle + faithfulness guard
11. **Lenses (`[lens.*]` config)** — generalizes `[analysis].agents`; skeptic stays post-pass
12. **TUI** — insights-first home + lens routing + router agent
13. **Embeddings** — semantic recall index (upgrades `recall` from keyword match)
14. **`dexta migrate --to-postgres`** + cross-model eval matrix

---

## Notable decisions (carry forward)

- **LLM = intelligence layer; analytics = evidence substrate.** Guard bounds citations, not role.
- **OpenRouter BYOM default** — one key, cross-model eval
- **One MCP server** over the harness (not per-device)
- **Nightscout meta-driver** — Loop/AAPS/xDrip+ through one connector
- **Batch = blackboard; interactive = LangGraph (later)**
- **oref0 math verbatim** — Tier B reconciliation; not a dosing algorithm

---

## References

- Intelligence layer design (memory/wiki/lenses/GlycemicGPT): `docs/INTELLIGENCE.md`
- Spec: `~/Desktop/dexta/docs/DEXTA_OSS_TECHNICAL_SPEC.md`
- Donor repos: `~/Desktop/dexta`, `~/Desktop/Dexter/dex-engine`

---

## Open note

Last user message was truncated: _"create a handoff with todays date including my questions
and what you built. make sure you add"_ — if something specific was meant after "add", append
it here on the next turn.
