# Issues

Follow-ups from the independent commit review of `d02b538..0ebba2d` on
`feat/agentic-intelligence-harness`. The #1-#6 review items are resolved
(2026-06-20). Deferred cleanups found while polishing are tracked below
(2026-06-21).

## Open / deferred (2026-06-21)

### #7 Store-layer dedup (CI unblocker landed; dedup pending)

`store/sqlite.py` and `store/postgres.py` duplicate several helpers. A CI
"Postgres parity" job now runs `tests/test_postgres_store.py` against a live
`postgres:16` service (the suite skips only when `TEST_DATABASE_URL` is unset),
so the backend is now tested in CI. Remaining: once that job is green, hoist the
provably-identical pure helpers into `store/_common.py`, keeping the JSONB-vs-TEXT
differences per backend, and confirm both suites stay green. Do not refactor
blind.

### #9 Near-duplicate helpers worth consolidating (with care)

`_relative_time` (4 server copies, slightly divergent signatures/bucketing) and
`_parse_json` (3 copies). Consolidating means reconciling small behavioral
differences and asserted UI text, so it is not a free dedup.

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
