# Memory v2 — parity audit (Task 1)

Read-only audit of the existing memory subsystem against the Memory v2 PRD.
Principle held: extend existing paths, do not build a parallel memory system.
Verdict: the substrate is largely present; the gaps are provenance fields, a
contradiction status, a retrieval guard, and the inspector UI. No vector DB is
needed (PRD agrees).

## Already built (reuse, do not rebuild)

| Area | Status | Where |
| --- | --- | --- |
| Finding / Hypothesis / InvestigationRun stores | existing | store/sqlite.py, store/postgres.py, store/port.py |
| Tool-trace + evidence-item + coverage persistence on runs | existing | InvestigationRun.tool_calls / evidence_items / coverage_summary (JSON cols) |
| Manual events, therapy-profile versions, open investigations, chat turns | existing | store/* + models.py |
| Finding lifecycle: active / superseded / rejected / dismissed / stale | existing | models.FindingStatus |
| Freshness decay (TTL by confidence x recurrence), last_verified, seen_count | existing | memory/freshness.py, models.Finding |
| Supersession link (superseded_by), persist_findings recurrence | existing | models.Finding, workflows/deep_analysis.py |
| Counter-evidence / skeptic notes; contradiction DETECTION | existing | agents/skeptic.py, memory/findings.find_contradictions |
| Retrieval: lexical embeddings, status-weighting, recency boost, synonyms | existing | memory/embeddings.py |
| recall() returns status + confidence + skeptic notes + open questions + connections | existing | agents/tools/toolkit.py `_recall` |
| Coverage gating (run "limited" at <70%) | existing | agents/coordinator.py (`_coverage_summary`, `_final_status`) |
| Faithfulness guard + treatment gate (applied at generation) | existing | guard/faithfulness.py, guard/treatment_gate.py |
| Hypothesis -> Finding link; ManualEvent -> run link | existing | models.Hypothesis.source_finding_id; ManualEvent.linked_run_id |

## Gaps (Memory v2 work)

### A. Finding statuses
- **CONTRADICTED status missing.** `find_contradictions()` runs and downgrades
  confidence + logs to skeptic_notes, but there is no status. PRD wants it. Add
  `FindingStatus.CONTRADICTED` and wire the transition (rejected takes
  precedence over contradicted).
- `candidate` / `retired`: not needed as new states. Hypotheses already model
  the pre-finding stage; `retired` is `stale`. Document, do not add.

### B. Provenance fields on Finding (the foundation for the guard + graph)
All missing today (coverage/guard/profile are run-level or transient only):
- `source_run_id` (and db id): the run that produced the finding (only an
  immutable RunFinding snapshot exists; no run<->finding link).
- `coverage_limited_at_creation`: copy the run's `coverage_summary.limited`.
- `used_manual_context`: was a ManualEvent in the window / consulted.
- `profile_version_id` (or content hash): the therapy-profile version in effect
  at `window_start` (via `get_active_profile`).
- `has_dosing_language`: run `_ADVICE_RE` over headline/body at store time
  (today the dosing check only runs on generated prose, never on stored memory).
- `pubmed_pmids`: persist cited PMIDs (today they are transient in the brief).
- (optional) `faithfulness_ok` / `treatment_gate_ok`: persist the verdicts.

Schema: additive nullable columns on `findings`, both backends, idempotent
(`_add_column` / `ADD COLUMN IF NOT EXISTS` patterns already exist). Note:
Postgres `SCHEMA_VERSION` (7) lags SQLite (9); reconcile while here.

### C. Evidence-graph links (relational, no Neo4j)
- Run <-> Finding: add `source_run_id` to Finding (covered by B).
- ToolCall <-> EvidenceItem: evidence items carry no `tool_name`; tool_calls are
  producer-name + count only. Add `tool_name` to each evidence item.
- Finding -> ProfileVersion: add `profile_version_id` (covered by B).
- Finding -> PubMed: persist `pubmed_pmids` (covered by B).
- Finding -> DoctorBrief: brief rebuilds from active findings; optionally persist
  `source_finding_ids` (LOW priority).

### D. Retrieval guard (core new piece)
- **Bug/gap:** `_recall` filters only `status != STALE`, so REJECTED / SUPERSEDED
  / DISMISSED findings can still be returned (down-ranked, not excluded). PRD:
  "rejected finding should not be reused." Fix to active-only + explicit
  exclusion.
- No "memory used vs excluded" labeling. Add a guard that, per retrieved memory,
  emits a label: used_supporting_context / used_prior_hypothesis /
  not_used_stale / not_used_rejected / not_used_contradicted /
  not_used_out_of_scope / not_used_safety_blocked.
- No safety re-check on STORED memory at retrieval (dosing language on headline,
  profile changed since creation, coverage-limited, manual-context dependency).

### E. Surfaces + evals (not present)
- **Memory Inspector** (drawer/page): memory used, memory excluded, why each,
  evidence links, last verified, coverage, provenance.
- **Memory evals**: stale/rejected not reused; contradiction downgrades; manual
  labeled user-reported; profile-change downweight; low-coverage limited;
  dosing-like blocked; answers list used + excluded.
- Hot/warm/cold tiering: not present (PRD wants retrieval-layer scoping). Treat as
  a scoping/ranking concern over the guard, not a new store.

### F. Semantic/vector
- Lexical only; pgvector seam noted in `embeddings.py`. PRD: optional, not
  required for launch. Defer.

## Build plan (PRD priority, sequenced around the in-flight UI work)

The current working tree has an unrelated UI overhaul in progress (chat.py,
app.py, reconciliation, templates, charts). Memory v2 backend work lives in
DIFFERENT files (models.py, store/*, toolkit `_recall`, skeptic.py,
coordinator.py, eval/), so phases 1-4 can proceed without touching the UI WIP.
The Inspector UI (phase 5) touches app.py/templates and should wait for the UI
work to land.

1. **(done)** Parity audit.
2. **Statuses** — add `CONTRADICTED`; wire the transition; update embeddings
   status-weight + the graveyard view set.
3. **Provenance + evidence-graph fields** — the Finding columns in B/C, schema
   migration both backends, populate at finding-creation (coordinator /
   investigator), reconcile Postgres schema version.
4. **Retrieval guard** — fix `_recall` to active-only; add the used/excluded
   labeling + stored-memory safety re-check; expose the labels in the recall
   payload.
5. **Memory evals** — the PRD eval cases (each: expected, actual, pass/fail,
   retrieved, excluded, guard, safety).
6. **Memory Inspector UI** — deferred until the UI WIP merges.
7. **Optional pgvector** — later, behind the existing `rank()` seam.

## Do not build (per PRD)
Neo4j, a required vector DB, RAG over raw CGM rows, an autonomous LLM memory
writer, dosing-preference memory, cloud-only memory.
