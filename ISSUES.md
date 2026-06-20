# Open issues

Tracked follow-ups from the independent commit review of `d02b538..0ebba2d`
(branch `feat/agentic-intelligence-harness`). None are ship blockers. The review
re-ran the full gate from a clean checkout: 1250 passed, 45 skipped, 0 failed;
mypy clean (101 files); ruff clean except a pre-existing `reason.py:98`
complexity. Mirror these to GitHub Issues with the `gh issue create` commands at
the bottom once `gh` is available.

## #1 `/reports` makes synchronous PubMed calls on page load (RESOLVED 2026-06-20)

- Opened: 2026-06-20
- Resolved: 2026-06-20 in commit `878591a`
- Severity: medium (performance / availability), non-blocking
- Area: `src/dexta_intelligence/server/app.py:711` ->
  `src/dexta_intelligence/agents/advisory.py:163` ->
  `src/dexta_intelligence/evidence/pubmed.py:95`

**What:** The `/reports` GET and the report export built a clinical advisory
with `model=None` but `evidence=PubMedBackend`. `build` called `_cite` once per
active finding, each doing synchronous esearch + esummary HTTP round-trips at a
15s read timeout. The default config is `evidence.enabled=True` and
`evidence.backend=pubmed`, so this fired on a default install, not only when
opted in. The call ran in the threadpool and swallowed failures, so it could not
crash or leak, but a slow or unreachable NCBI stalled the page and could tie up
worker threads.

**Fix (shipped):** the `/reports` GET is now deterministic and never touches the
network; literature citations are deferred to a `GET /reports/citations` HTMX
fragment swapped in after first paint (only when evidence is enabled). Backed by
a process-wide TTL cache (`evidence/cache.py`, `[evidence].cache_ttl_minutes`,
default 1 day) and a tighter interactive timeout (4s/2s) via
`PubMedBackend(interactive=True)`. Export keeps citations inline. Tests in
`tests/test_advisory.py` + `tests/test_evidence.py`.

## #2 SSE error payloads can disclose an absolute DB path

- Opened: 2026-06-20
- Severity: low (information disclosure), non-blocking
- Area: `src/dexta_intelligence/server/app.py:936`, `:1097`, `:1131`

**What:** The chat and investigate stream handlers emit
`f"{type(exc).__name__}: {exc}"` to the client over SSE. The payload is rendered
with `textContent` (XSS-safe) and carries no secret on the audited paths: the
Nightscout token is redacted at source (`connectors/nightscout.py:386`) and LLM
API keys never ride in URLs or error strings. The residual risk is that a
store-open failure (for example `sqlite3.OperationalError` / `OSError`) would
stringify an absolute database path into the payload.

**Fix options:** map worker exceptions to a generic client-facing message and
log the detail server-side only.

## #3 Advisory dosing gate does not cover `evidence_refs` / monitoring / questions

- Opened: 2026-06-20
- Severity: low (defense-in-depth), non-blocking
- Area: `src/dexta_intelligence/agents/advisory.py:126`, `:173-192`

**What:** Only `discuss_now` items pass through `_is_safe`. The `monitoring` and
`questions_for_clinician` lists skip the gate, and no list gates the
`evidence_refs` field. Model-generated `Finding.headline` text flows into
`evidence_refs` (`advisory.py:178`, `:188`) unguarded. This is low risk today
because the upstream faithfulness audit (`investigator.py:263`) already forces
headlines to be observation-only and number-faithful with no dosing or units, so
a dosing-style string reaching these fields is doubly improbable. But the
"treatment gate is the hard backstop" claim is not literally true for these
fields.

**Fix options:** run the full rendered text of every `DiscussionItem`
(including `evidence_refs`) through `_is_safe` across all three lists.

## OSS hygiene (minor, low priority)

- #4 (2026-06-20): no `CODE_OF_CONDUCT.md`. Add a Contributor Covenant to round
  out the community health files alongside the existing `CONTRIBUTING.md` /
  `SECURITY.md` / issue + PR templates.
- #5 (2026-06-20): no `CHANGELOG.md`. Start a Keep a Changelog file once the
  first tagged release lands (`__version__` is `0.1.0`).
- #6 (2026-06-20): pre-existing, out of review range. `_storage_view`
  (`server/app.py:1684`) renders the Postgres `database_url` (a DSN that can
  embed `user:password`) to the dashboard storage panel. Low risk for the
  default single-user local app (sqlite default shows only a path; same-origin,
  user's own credentials), but exposed on screenshare / multi-user. Mask the
  password in the rendered DSN.

## Mirror to GitHub Issues

```sh
gh issue create --title "/reports makes synchronous PubMed calls on page load" \
  --label bug --body-file - <<'EOF'
See ISSUES.md #1. Default config (evidence.enabled=True, backend=pubmed) fires
synchronous NCBI round-trips on the /reports GET via advisory._cite. Cache,
lower the timeout, or defer behind an explicit action.
EOF

gh issue create --title "SSE error payloads can disclose an absolute DB path" \
  --label bug --body-file - <<'EOF'
See ISSUES.md #2. app.py stream handlers emit f"{type(exc).__name__}: {exc}".
No secret leaks on audited paths, but a store-open OSError can stringify a DB
path. Map to a generic client message; log detail server-side.
EOF

gh issue create --title "Advisory gate misses evidence_refs / monitoring / questions" \
  --label bug --body-file - <<'EOF'
See ISSUES.md #3. Only discuss_now passes _is_safe; evidence_refs is never
gated. Low risk via the upstream faithfulness audit, but gate the full rendered
DiscussionItem text across all three lists.
EOF
```
