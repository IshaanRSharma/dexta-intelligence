# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The episode graph refuses to let a window artefact read as a fact. An
  excursion clipped by a window edge, the end of the record, or a sensor gap is
  flagged truncated and its duration reported as a lower bound, instead of the
  same 175-minute high silently becoming "85 min" under a window that started
  halfway through it. Every summary now carries observation coverage, so an
  episode count cannot read the same whether the sensor ran for thirty days or
  three. Locating an episode by timestamp returns a miss when nothing was
  happening then, naming the nearest as a separate fact, rather than handing
  back an excursion hours away for the model to explain as the answer. The
  `episodes` tool reports how many episodes matched alongside how many rows it
  returned, and an episode with no context says so in prose instead of
  returning an empty list for the model to fill in.
- Episode detection implements the whole 2019 consensus event definition. An
  excursion now ends only after 15 minutes back in range, so an afternoon spent
  bobbing around 180 is one episode rather than a dozen counts of sensor noise.
  This diverges deliberately from the LLM-CGM benchmark's count-every-dip
  convention, which dexta already declines to answer under. Context edges reach
  back four hours for food and insulin rather than three, matching how long a
  mixed meal absorbs.
- The demo patient's CGM is generated from its own treatment record. Carbs push
  glucose up, damped by how promptly the bolus landed; isolated boluses and
  workouts pull it down; the baseline between them wanders rather than jittering.
  Previously the trace was built independently of the record beside it, so six
  months of meals and boluses moved nothing, every meal-versus-glucose test was
  null by construction, and the patient sat at 98.9% time in range with a CV of
  11. It now sits at 76% in range, CV 32, GMI 6.9, meeting each consensus target
  without meeting it trivially, with bounds asserted in tests. Dinner exists on
  every day rather than one in six, bolus timing varies so late insulin is
  discoverable rather than asserted, and the Mar-Jun excursions carry the events
  that explain them instead of appearing from nowhere.
- `explain_spike` reports moderate confidence on the demo's hero spike rather
  than high. That is the computed reading on a record where ordinary meals also
  run high, which narrows the bolus-delay separation the score depends on.
- README: the demo setup is now a self-contained section covering what gets
  seeded, the record's actual shape and metrics, a guided tour, resetting, and
  moving to your own data.

## [0.2.0]

### Added
- The graph, surfaced as product: a why-chain episode card on chat answers
  (span, duration, extreme, severity, typed context edges with signed
  offsets), recurrence lines on briefs and recall ("seen 7 times since May
  12"), `what_changed` and `contradicted_beliefs` belt tools over the
  SUPERSEDES and CONTRADICTS edges, and a deterministic endo-visit brief
  (top 3 findings with receipts and a neutral question each, dosing-bait and
  unfaithful candidates structurally dropped).
- Conversational capture, thesis-safe: the model proposes structured events
  from free text, a deterministic validator checks type, window, and
  dosing-text rejection, and the user's one-tap confirmation on /log is the
  only commit gate. Unconfirmed proposals live in process memory and cannot
  reach the store, guard, findings, or graph.
- Curated context in the reasoning prompts: the deterministic curator's
  typed selection now feeds chat and orchestrator system prompts, with one
  trace receipt per pruned item.
- Bitemporal finding edges: findings now link through a `finding_edges` table
  (supersedes, contradicts, plus reserved relations), each edge carrying
  event time and knowledge time and a deterministic reason. Edges are
  authored only where the system already computes the fact (supersession,
  contradiction detection, synthesis retirement); nothing model-authored
  enters the graph.
- Deterministic context curator (`memory/curator.py`): `select_context` picks
  typed context (facts, beliefs, conventions, history) under a token budget
  with per-type floors and returns a drop list where every drop carries a
  trace-ready reason. Severe episodes and treatment-gate inputs are never
  droppable; scoring is clock-free and reproducible (ISSUES #15).
- Temporal episode graph (`analytics/episodes.py`): hypo/hyper excursions and
  sensor gaps as first-class episode nodes (span, duration, extreme, severity,
  clinical significance) with typed context edges to the meals, boluses,
  activity, and sleep around each episode. Deterministic, model-free, aligned
  to the LLM-CGM ground-truth definitions; `summarize()` emits ontology-keyed
  rollups ready for the faithfulness guard (ISSUES #14). Plus a timeline
  renderer in `bench/render_episodes.py`. Episodes carry stable ids and are
  exposed on the agent belt as two tools: `episodes` (all nodes plus rollups)
  and `explain_episode` (traverse one node's context edges by id or
  timestamp), so the model reasons over the graph instead of re-deriving
  segmentation per question.
- Provenance layer on the faithfulness guard: a deterministic metric ontology
  (`guard/metrics.py`) binds evidence numbers to the metric they describe, and
  an opt-in provenance pass catches "right number, wrong metric" citations
  (the standard deviation presented as the coefficient of variation), the
  exact failure LLM-CGM measured in code-execution agents. Wired into the
  chat answer surface; all other audit callers unchanged (ISSUES #13).
- Nightscout API v3 support with v1 fallback: the connector prefers the
  secured `/api/v3/*` interface (JWT bearer token minted from the configured
  access token) and falls back to the legacy v1 query API on older servers.
  The dialect is detected once per connector instance; v3 documents are
  normalized to the exact v1 event shape, so downstream code is unchanged
  (ISSUES #12).
- Tandem connector retargeted at the Tandem Source API: docstrings and
  user-facing strings updated, a runtime guard requires tconnectsync >= 2 (v1
  spoke to the retired t:connect cloud), and the `[tandem]` extra now pins
  `tconnectsync>=2.0`. Live-account verification still open (ISSUES #11).

- External benchmark run in `bench/`: dexta vs the same model with the raw data
  in-context, on LLM-CGM (Healey & Kohane, PSB 2025). Head-to-head scripts, raw
  per-question dumps, hand-verified writeups, and the error figure. Mean absolute
  error ~100x lower through the harness on the exactness-scored questions.
- Two agent tools: `find_lows` (discrete hypoglycemia episodes with nadir,
  duration, and clinical significance) and `glucose_extremes` (timestamp, local
  time, and period of the single highest/lowest reading).
- Question-type-aware treatment gate: for a lows question, insulin-on-board
  evidence is the hard requirement and missing carb data becomes an appended
  caveat instead of muting the answer. The spike path is unchanged.

- Deliberate synthesis pass: a finished investigation now produces a grounded
  synthesis (the leading explanation, the alternatives ruled out, the supporting
  evidence, the cross-modal probes, and the open gaps). Every figure is re-audited
  against the tool evidence pool, so the synthesis cannot surface a number the
  tools never produced. Attached to a clean answer.
- Mid-loop context acquisition: when a gap blocks it, the agent can call a
  `request_context` tool for the moment it cannot explain and surface a precise,
  dosing-gated logging request instead of guessing. It reuses the unexplained-spike
  detector's proximity rule and never fabricates the missing value.
- Adaptive stop conditions: the reasoning loop now nudges the model to conclude
  when it reaches high confidence or when the last probes added no new
  information, instead of letting it probe in circles to the step budget. The
  nudges are advisory (the model still writes the answer); the step ceiling stays
  the only hard stop.
- Next-probe guidance: the belief state suggests the most discriminating evidence
  the investigation has not gathered yet for its open hypotheses (a light
  information-gain heuristic over modality coverage), folded into what the model
  reads each step. Advisory, never a controller.
- Hypotheses now steer the live loop: open hypotheses banked by prior analysis
  re-enter a new investigation as competing hypotheses in the belief state, and
  reach the model in the first-turn prompt (with stable ids) so it probes to
  discriminate or refute them and tracks their status in place.
- Working belief state: an investigation now carries an explicit, structured
  understanding across steps (competing hypotheses and their status, evidence,
  open gaps, running confidence) that the model revises through an `update_belief`
  tool. It scaffolds the reasoning without deciding for it, and stays out of the
  faithfulness evidence pool. Threaded through the orchestrator's loop.
- Reasoning-process eval (E7, `eval reasoning`): grades the investigation path,
  not just the answer. Scores cross-modal evidence coverage, probe efficiency,
  gap-handling, and path soundness against the labeled benchmark, so each
  intelligence-flow phase is measured rather than asserted.
- LLM providers: Google DeepMind Gemini (`google_genai`), local Ollama
  (`ollama`, honoring `OLLAMA_HOST`), and local model files via llama.cpp
  (`llamacpp`, the optional `local` extra).
- Memory v2 (temporal evidence memory): a retrieval guard so recall returns only
  active, non-dosing findings and lists what it excluded and why; a new
  `CONTRADICTED` finding status; and a `/memory` inspector page showing memory in
  use vs withheld.
- Active context acquisition: dexta detects unexplained spikes (a high with no
  logged meal or note nearby) and asks the user to log what happened, on a new
  `/context` page. It asks, it never fabricates the missing value.
- Prompt registry: agent prompts live as overridable markdown in
  `agents/prompts/` (`[prompts] dir`), with the dosing rail locked in code.
- `Dockerfile` (the compose reference deployment now builds), `CODE_OF_CONDUCT.md`,
  and `CITATION.cff`.

### Changed

- One brain answers the chat box: the no-JS `POST /api/ask` fallback now runs
  the same traced, railed orchestrator as the streaming path instead of the
  simpler ChatAgent, and surfaces the same faithfulness note.
- The belief-layer reasoning scaffold now defaults off; a real-model on/off A/B
  showed no answer-quality gain on a capable model. The `use_belief` flag stays
  for ablation.
- The `/reports` page renders without a network call; literature citations are
  deferred to a cached `/reports/citations` fragment loaded after first paint.
- The agent tool belt now lives in the `agents/tools/` package.
- The oref insulin-curve math is memoized and skips fully-decayed doses. The
  exponential IOB/activity/constant functions are pure over a handful of
  integer-minute arguments, so they are `lru_cache`d, and `insulin_totals` now
  skips doses past DIA (which contribute exactly zero). Results are byte-for-byte
  unchanged; curve evaluations on the 90-day reconciliation drop from ~4.7M to
  ~700k.

### Removed

- The superseded keyword `RouterAgent` and its prompt files: the orchestrator
  has been the only ask engine in practice, and the claimed no-model fallback
  role did not exist in code. Also removed dead server and CLI accessors
  (`panel_by_key`, `trace_icon_for_text`, `get_registry`) and an unreferenced
  template.

### Fixed

- The one-command Docker demo served a fully seeded database that rendered
  empty. Daily rollups are computed as a side effect of connector sync, which
  demo mode disables, so the demo store never had them and every rollup-backed
  surface (dashboard time in range, goals, trends) read blank on 185 days of
  data. Rollups are now part of the seed, and a store seeded before this change
  is backfilled rather than left permanently blank.
- Dosing gate hardened after an adversarial red-team pass: the shared
  `_ADVICE_RE` backstop only matched four verbs, so "raise your basal",
  "give 2 units", "2 more units", "up your basal" and similar phrasings
  leaked as observation. The verb set is broadened, quantity-implied and
  "up your" branches added, and the match window is clause-tempered so
  "give 0.5 units" is caught while "take your time. Basal was 0.8 u/hr"
  stays observation. Remaining lexical residuals documented in ISSUES #16.
- Faithfulness guard: comma-grouped numbers (`25,920`) no longer split into two
  claims, removing the main source of false grounding warnings.
- The project resolves with `uv lock` again: the `carelink` extra no longer pins
  a package that is not published on PyPI.
- The reports page no longer makes synchronous PubMed calls on load.
- Stream errors and the storage panel no longer expose internal detail (DB path,
  database credentials) to the client.
- PostgresStore no longer closes its connection after the first write. It used
  `with self._conn:` for per-operation transactions (psycopg2 semantics), but in
  psycopg 3 that context manager closes the connection on exit, so every call
  after `migrate()` failed with "the connection is closed". Switched to
  `self._conn.transaction()`, which commits/rolls back without closing.

## [0.1.0]

- Initial alpha: agentic harness with deterministic analytics, statistical
  rigor, the numeric-faithfulness guard, connectors, and the web GUI.
