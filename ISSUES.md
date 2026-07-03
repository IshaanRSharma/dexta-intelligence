# Issues

Follow-ups from the independent commit review of `d02b538..0ebba2d` on
`feat/agentic-intelligence-harness`. The #1-#6 review items are resolved
(2026-06-20). Deferred cleanups found while polishing are tracked below
(2026-06-21).

## Open / deferred

- **#15** Deterministic context curator, first slice (2026-07-03, from the
  context-engineering research pass): a `select_context(query, budget)` over
  the episode graph and findings. Collect per type (FACTS: tool outputs and
  episode nodes; BELIEFS: findings/hypotheses; CONVENTIONS: the metric
  ontology; HISTORY: turn results), score deterministically (relevance,
  temporal overlap with the asked window, salience, freshness, type prior
  with facts over beliefs), fill under a token budget with per-type floors,
  and return the selection PLUS a drop list with reasons. Invariants: pruning
  reduces tokens never ground-truth availability (the store is truth, the
  window is a view); severe episodes and treatment-gate inputs are never
  droppable; every drop emits a trace line; same query and state yields the
  same selection. No LLM in the write or drop path (optional re-ranker over
  the deterministic top-N only). No memory-graph libraries: Graphiti, Mem0,
  Cognee, and Letta all LLM-author their graph write path, which would put
  ungated model-authored claims upstream of the faithfulness guard; Kuzu is
  archived (Apple acquisition). NetworkX only if multi-hop queries ever
  demand it. Types kept isolated per MemGuard (arXiv 2605.28009); context-rot
  evidence in arXiv 2606.10209.
- **#13** Faithfulness guard is set-membership, not provenance-aware (2026-07-03,
  from the LLM-CGM paper study): the guard's own docstring states it does
  "set-membership checking, not semantic verification. A number can match the
  pool while being cited in the wrong context." That is exactly the failure
  class LLM-CGM measured in code-execution agents (SD reported where CV was
  asked, 0/10 for both code frameworks in their Table 3). Harden the guard to
  verify number -> computation -> window -> metric -> unit provenance, so it
  catches "right number, wrong metric" and not just fabricated numbers.
  Highest-ROI verification hardening; upgrades the safety claim from "we flag
  fabricated numbers" to "we verify every number is computed and cited
  correctly."
  LARGELY RESOLVED (2026-07-03): `guard/metrics.py` adds a deterministic
  metric ontology (evidence keys + prose aliases per metric), and the guard's
  new opt-in provenance pass (`check_provenance=True`, wired into the chat
  answer surface) fires only when a sentence names a metric we hold, the
  number mismatches it, and matches a different held metric. SD-cited-as-CV
  is now caught and surfaced. Remaining: thread ProvenanceViolation into
  trace.py, extend the ontology beyond the core metric set, and the #14
  episode layer.
- **#14** Structured domain-context layer (2026-07-03, same study): LLM-CGM's
  top future-work ask is injected domain definitions and analysis conventions
  (CV is not SD, in-range is 70-180, "today" is the last data day). dexta
  covers pieces (tool belt, timing_context, coldstart gating) but has no
  single structured layer that both the agent and the faithfulness guard
  consult. Also make temporal episodes (excursions, lows, gaps) first-class
  segmented objects rather than per-tool outputs; the paper's worst code-agent
  scores were all temporal segmentation tasks.
  PARTIALLY RESOLVED (2026-07-03): `analytics/episodes.py` makes hypo/hyper
  excursions and sensor gaps first-class `Episode` nodes with typed
  `ContextLink` edges to nearby meals, boluses, activity, and sleep
  (per-kind reach windows), thresholds aligned to the LLM-CGM ground-truth
  definitions, and `summarize()` emitting ontology-aligned keys. Validated
  against the P1 bench ground truth (65 hypo episodes, 25 clinically
  significant). Episodes are now addressable (stable ids, `EpisodeGraph.node`
  / `.at` traversal) and on the reasoning belt as two tools: `episodes` and
  `explain_episode` (by id or timestamp, returning the node with its typed
  context edges, numbers fed to the faithfulness evidence pool). Remaining:
  thread episodes and ProvenanceViolation into trace.py, and the broader
  definitions layer both agent and guard consult.
- **#11** Tandem connector targets the retired backend (2026-07-02, ecosystem
  survey): `connectors/tandem.py` documents and delegates to tconnectsync
  against the t:connect cloud, which Tandem shut down in the US on 2024-09-30.
  tconnectsync v2+/v3 targets the replacement Tandem Source API.
  PARTIALLY RESOLVED (2026-07-02): docstrings and user-facing strings now
  describe the Tandem Source API, `_build_client` enforces tconnectsync >= 2
  at runtime with an upgrade hint, and the `[tandem]` extra is pinned to
  `>=2.0`. Remaining: verify the full check/pull path against a live Tandem
  Source account (all tests run on stubs).
- **#12** Nightscout connector speaks legacy API v1 only (2026-07-02):
  `connectors/nightscout.py` uses `/api/v1/*` throughout. v1 still works, but
  API v3 is the secured, documented interface.
  RESOLVED (2026-07-02): the connector now prefers v3 (`/api/v3/*` with a JWT
  bearer token minted from the configured access token) and automatically
  falls back to the v1 query API on servers without v3. The dialect is
  detected once per connector instance, and v3 responses are normalized to
  the identical event shape v1 produces, so no downstream code or config
  changes are required.

## Resolved (2026-06-21, third pass)

- **#7** Store-layer dedup: the two provably-identical pure helpers (`_opt_json`,
  `_prediction_horizon_min`) are hoisted to `store/_common.py` and shared by both
  backends. The `_row_to_*` mappers genuinely differ (TEXT-JSON vs JSONB) and
  stay per-backend by design. A CI "Postgres parity" job now runs the parity
  suite against a live `postgres:16`, so the backend is tested (it previously
  skipped without `TEST_DATABASE_URL`).
- **#9** Near-duplicate helpers consolidated: one `_relative_time`
  (`server/_format.py`, the `None`-safe superset) replaces the 4 server copies;
  one `parse_json` (`agents/_json.py`, with an optional logging `context`)
  backs the 3 former `_parse_json` copies.

## Resolved (2026-06-21, second pass)

- **#8** `_text_of` name collision: renamed by behavior to `_content_text`
  (extractor in `reason.py` / `spike.py`) and `_strip_code_fence` (fence stripper
  in `brief`/`seeker`/`router`/`goals`/`synthesis`).
- **#10a** MCP `console_scripts`: added `dexta-mcp` entry point.
- **#10b** `pattern.py:647` "dead branch": verified live (both ternary branches
  reachable, `confidence` is used). False flag, closed.
- **#10c** `monitor._severe_high` now breaks a sustained-high run on a sensor gap
  (`SENSOR_GAP_MIN`), so a gap can no longer be counted as sustained
  (regression test `test_sensor_gap_breaks_sustained_high`).

## Resolved

### #1 `/reports` made synchronous PubMed calls on page load

Fixed in `878591a`. The page GET is deterministic and network-free; literature
citations are deferred to a cached, tighter-timeout `/reports/citations` HTMX
fragment. Export keeps citations inline. (`server/app.py`, `evidence/cache.py`,
`evidence/pubmed.py`, `[evidence].cache_ttl_minutes`.)

### #2 SSE error payloads could disclose internal detail

Fixed. The chat and investigate stream handlers now emit a generic client
message and log the exception server-side via `logger.exception`, so an
exception string (and any DB path inside it) never reaches the browser.
(`server/app.py`; tests assert the detail is not leaked.)

### #3 Advisory dosing gate did not cover `evidence_refs` / monitoring / questions

Fixed. `_item_is_safe` gates every text field of a `DiscussionItem` (including
`evidence_refs`), applied to `discuss_now`, `monitoring`, and
`questions_for_clinician`. (`agents/advisory.py`.)

### #4 No `CODE_OF_CONDUCT.md`

Added a Contributor Covenant 2.1 with a no-patient-data clause.

### #5 No `CHANGELOG.md`

Added a Keep a Changelog file with an `[Unreleased]` section over `0.1.0`.

### #6 `_storage_view` rendered the Postgres DSN with credentials

Fixed. `_mask_dsn` replaces the password with `***` before the DSN reaches the
dashboard storage panel. (`server/app.py`.)
