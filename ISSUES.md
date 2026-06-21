# Issues

Follow-ups from the independent commit review of `d02b538..0ebba2d` on
`feat/agentic-intelligence-harness`. The #1-#6 review items are resolved
(2026-06-20). Deferred cleanups found while polishing are tracked below
(2026-06-21).

## Open / deferred (2026-06-21)

### #7 Store-layer dedup needs a verifiable Postgres path

`store/sqlite.py` (~1410) and `store/postgres.py` (~1436) duplicate several
helpers (`_opt_json`, `_count`, `_raw_ids`, the `_row_to_*` mappers). Hoisting
the pure ones into `store/_common.py` is the biggest remaining dedup, but the
Postgres backend is not verifiable locally (44 of 46 store tests skip without a
live DB) and the helpers carry real JSONB-vs-TEXT differences. Do this only with
a live Postgres (or CI matrix) confirming parity; do not refactor it blind.

### #8 `_text_of` is one name for two different helpers

Across `agents/reason.py` + `investigations/spike.py` (content-parts extractor)
vs `agents/brief.py` / `seeker.py` / `router.py`, `workflows/goals.py`,
`memory/synthesis.py` (markdown code-fence stripper). They cannot be merged
(different behavior); rename the two variants so the name stops being misleading.

### #9 Near-duplicate helpers worth consolidating (with care)

`_relative_time` (4 server copies, slightly divergent signatures/bucketing) and
`_parse_json` (3 copies). Consolidating means reconciling small behavioral
differences and asserted UI text, so it is not a free dedup.

### #10 Minor items

- MCP server has no `console_scripts` entry point (run only via module).
- `pattern.py:647` review-flagged "dead confidence branch" (unverified; inspect).
- `workflows/monitor.py` `_severe_high` does not account for sensor gaps in its
  duration check.

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
