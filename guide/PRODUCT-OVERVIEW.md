# dexta-intelligence: Product Overview and Delivery Summary

Prepared 2026-06-21. For product review. Status: the core arc is merged to
`main` (PR #1); the latest increment (prompt registry, connector audit, Docker,
OSS hardening) is on `feat/prompts-registry`, green and demo-ready.

---

## 1. What it is

dexta is a self-hosted, agentic intelligence layer for Type 1 diabetes data. It
ingests a person's CGM, insulin, pump, and wearable history (read-only) and turns
it into traceable findings: why something happened, what changed, and what the
system has learned over months. Bring your own model, bring your own database,
the data never leaves the user's infrastructure.

It is decision support, not a medical device. It never gives dosing advice. Every
finding is a hypothesis for the person and their care team to review.

## 2. Why it is different

1. **Determinism computes the facts; the model reasons on top.** Tested analytics
   and statistics produce every number (time in range, rigor gates, oref0
   reconciliation). The model plans investigations, ranks hypotheses, and
   explains. It never invents a figure.
2. **Statistical rigor before any claim.** Discovery passes permutation tests and
   false-discovery control, then survives an independent adversarial skeptic,
   before a finding is shown.
3. **Two hard safety rails, always on.** A faithfulness guard rejects any prose
   whose numbers do not trace to a tool call. A treatment gate blocks dosing,
   basal, carb-ratio, and correction instructions.

## 3. Feature inventory

### Web application (one clear job per tab)

| Tab | What it does |
| --- | --- |
| Chat | Instant Q&A with a live, tool-by-tool trace. |
| Investigations | The deep traced drill (plan to trace to evidence), deep analysis, and the open-investigations queue. |
| Findings | Durable memory: active, hypotheses, rejected, plus the investigation log, with evidence strength and counter-evidence. Prediction reconciliation lives here. |
| Reports | A clinician discussion brief (review now, monitor, questions to ask), grounded in the user's evidence and PubMed, with Markdown export. |
| Goals | Goals run as recurring investigations, with progress and checkpoints. |
| Connectors | Data sources, per-source health, and continuous sync. |
| System | Observability and the evaluation model card. |
| Settings | Configuration, including model provider and prompt overrides. |

### Data and connectors

Read-only ingestion, stored locally (SQLite zero-setup, or a Postgres the user
controls). Idempotent sync, so re-running is always safe.

| Source | Class | API | Notes |
| --- | --- | --- | --- |
| Tandem t:slim X2 / Control-IQ | Pump | unofficial | Verified syncing live. Boluses, temp basals, suspends, and the versioned basal / carb-ratio / ISF profile. |
| Medtronic | Pump | unofficial | Via CareLink; also reachable through Tidepool / Nightscout. |
| Dexcom | CGM | official OAuth + Share | Two paths; Share verified live. |
| Abbott Libre | CGM | unofficial (LinkUp) | |
| Nightscout | Aggregator | self-hosted | Covers DIY closed loops (Loop, AAPS, Trio) and any device feeding Nightscout, including pod boluses/basals and loop forecast curves. |
| Tidepool | Aggregator | official-ish | Broad device set via upload. |
| Oura, Whoop | Wearable | official | Sleep and activity context. |
| CSV upload | Manual | Clarity / LibreView | History backfill. |

Forecast reconciliation: dexta parses the looping algorithm's own predictions
(OpenAPS / AAPS / Loop `devicestatus`: IOB, COB, UAM, ZT) and reconciles them
against realized glucose, surfacing recurring forecast misses.

### Intelligence and rigor

- Deterministic producers run rigor-gated pattern tests (permutation, FDR,
  effect sizes, error grids).
- A coordinator plans which investigations to run; an LLM orchestrator drills a
  single question tool by tool; an adversarial skeptic re-checks every finding.
- Durable agent memory: findings, hypotheses, runs, with a freshness lifecycle
  (stale findings decay so old patterns do not resurface forever).
- Clinical-literature grounding via PubMed, with PMIDs linked in every answer.

### Safety

- Faithfulness guard (every number traces to a tool result).
- Treatment gate (no dosing / basal / carb-ratio / correction advice), applied
  across chat, investigations, briefs, and the advisory.
- Prompt safety rail is a code constant, re-applied even when a user overrides a
  prompt, so it cannot be edited out.

### Evaluation and model card

A reproducible eval harness on synthetic ground truth, surfaced live in the app:

| Eval | Measures |
| --- | --- |
| E1 faithfulness | the guard catches fabricated or miscontextualized numbers |
| E2 power | true-discovery rate on a planted effect |
| E3 accuracy | oref0 forecast vs realized glucose (Clarke / Parkes grid, MARD) |
| E4 null FDR | empirical false-discovery rate on effect-free data |
| E5 perturbation | finding stability under dropout, dupes, gaps, timezone shift |
| E_consensus | rollup metrics match the 2019 international-consensus formulas |
| E6 agentic | end-to-end attribution, faithfulness, and a dosing red team (target zero) |

These are calibration and robustness checks on synthetic data, not clinical
validation.

### Operations and deployment

- CLI: `init`, `doctor`, `sync`, `analyze`, `investigate`, `ask`, `explain`,
  `goals`, `monitor`, `daemon`, `serve`, `demo`, `research`, `upload`.
- `daemon` runs the cadence: sync, monitor, goal ticks, periodic deep analysis.
- Anomaly monitoring (severe lows/highs, TIR cliffs, sensor gaps,
  correction-not-working, rebound lows) with notification sinks.
- Docker reference deployment (Postgres) and zero-setup local SQLite.

### Extensibility

- Bring your own model: Anthropic, OpenAI, Google DeepMind Gemini, OpenRouter
  (one key, any hosted model), local Ollama, and local model files via llama.cpp.
- Bring your own database: SQLite or Postgres behind one storage port.
- Prompts are version-controlled markdown a user can override per deployment.
- Documented connector / agent / tool seams, each backed by a conformance test.

## 4. Recent delivery (this arc)

- Independent multi-agent review of the full arc: SHIP verdict, no blockers.
- Tool belt reorganized into a clean `agents/tools/` package (behavior-preserving).
- New model providers: Google DeepMind Gemini, local Ollama, local model files.
- `/reports` made instant: literature lookups deferred off page load and cached.
- Prompt registry: all agent prompts moved to overridable markdown with a locked
  safety rail (the MWP "interpretable context" idea, applied without diluting the
  rigor design).
- Connector audit and live verification (Tandem and Dexcom confirmed syncing).
- Security and quality fixes: no credential or internal-path leakage to the GUI;
  the project resolves with `uv lock` again; the last lint debt cleared.
- OSS hardening: LICENSE, SECURITY, CONTRIBUTING, CODE_OF_CONDUCT, CHANGELOG,
  CITATION, issue/PR templates, a working Dockerfile, README with diagrams and
  badges.

## 5. Quality bar (current)

- Test suite: 1288 passed, 46 skipped, 0 failed.
- Static analysis: ruff clean (0 findings), mypy strict clean (111 files).
- CI runs lint, types, and tests on Python 3.11 and 3.12.
- Live verification: `dexta demo` end to end; Tandem and Dexcom syncing; eval
  E2 (power) and E4 (null FDR) passing.

## 6. Demo readiness

Ready. `dexta demo` loads a realistic synthetic patient and runs a full,
explained investigation with no API key. For a live demo, Tandem and Dexcom are
confirmed pulling real data. All differentiators have a UI surface
(reconciliation, evals/model card, traced investigations, reports).

## 7. Roadmap (next, prioritized)

1. Omnipod 5 ingestion via Glooko CSV export (the one real connector gap; no
   vendor API exists for Omnipod 5).
2. Apple HealthKit import (covers no-API pumps and many devices via the phone).
3. Verify and surface the loop-prediction reconciliation end to end (the
   parser exists; confirm it feeds the reconciliation view and labels the
   source algorithm).
4. Store-layer dedup (shared SQLite/Postgres helpers), gated on a live Postgres
   for parity.
5. Skeptic LLM critique layer (optional adversarial refutation on top of the
   deterministic checks).

## 8. Limitations and non-goals

- Not a medical device; no dosing advice, by design and by hard rail.
- Evals are calibration and robustness on synthetic data, not clinical validation.
- Several pump/CGM connectors are unofficial (the only available path for pump
  data); ToS risk is disclosed in code and docs.
- Postgres parity is not verifiable in the local dev environment (needs a live
  DB or CI); SQLite is the default and is fully covered.
