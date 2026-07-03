# Issues

Follow-ups from the independent commit review of `d02b538..0ebba2d` on
`feat/agentic-intelligence-harness`. The #1-#6 review items are resolved
(2026-06-20). Deferred cleanups found while polishing are tracked below
(2026-06-21).

## Open / deferred

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
  API v3 is the secured, documented interface. Add v3 support with v1 fallback.

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
